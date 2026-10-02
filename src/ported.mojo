"""Life Art kernels: cellular-automata evolution, palette rendering, seeding.

Everything here is bit-exact and allocation-free. The caller owns every buffer,
including the ping-pong pairs, so nothing in this file can leak and nothing can
surprise the host. `capi.mojo` is the only place the Mojo dialect shows
through; the algorithms below stay plain.

Layout notes that matter:

* A grid is `w * h` `UInt8` cells in row-major order, each 0 or 1, with
  **toroidal** wrap in both axes. Toroidal wrap is what makes the vector body
  possible: row `y` only ever reads rows `y-1`, `y` and `y+1`, so every row is
  an independent work item and `parallelize` needs no synchronisation.
* `ages` carries how many consecutive generations a cell has been alive,
  saturating at 255. It is what turns a binary soup into a poster: the renderer
  maps age through a palette, so lineage is visible as colour.
* A rule is a pair of 9-bit masks, `bmask`/`smask`, where bit `k` of `bmask`
  means "a dead cell with `k` live neighbours is born" and bit `k` of `smask`
  means "a live cell with `k` live neighbours survives". Conway is B3/S23.

`unsafe_load` / `unsafe_store` throughout: the plain `load`/`store` spellings
are deprecated on this toolchain, and the wrapped-column handling below means
several accesses are provably in bounds rather than merely usually in bounds.
"""

from max.algorithm import parallelize
from std.sys import simd_width_of

comptime U8Ptr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime SIMD_WIDTH = simd_width_of[DType.uint8]()
comptime W = SIMD_WIDTH
comptime VW = SIMD[DType.uint8, W]

# `parallelize` on this toolchain costs a fixed amount per launch that does not
# shrink as the work per row grows, so the crossover is not a guess about grid
# size -- `bench/bench.py` measures it. What is left here is the cap: 16
# workers is what a poster-sized grid keeps busy, and past that the row count
# is no longer the limit.
comptime PAR_WORKERS = 16
#: Hard floor below which no amount of workers can pay for the launch.
comptime PAR_MIN_CELLS = 1 << 18
#: Rows per `parallelize` work item. A grid is far taller than it is many
#: bands; see the note on `_advance_all`.
comptime ROWS_PER_TASK = 8

# The one rule the fast vector body special-cases. Everything else runs the
# general 9-way unrolled body, which is slower but correct for any totalistic
# rule.
comptime CONWAY_B = 1 << 3
comptime CONWAY_S = (1 << 2) | (1 << 3)


# ------------------------------------------------------------------ compares

def _gt(n: VW, k: Int) -> VW:
    """1 in every lane where `n > k`, 0 elsewhere.

    `max(n, k) - k` is a saturating subtract that cannot wrap, which `n - k`
    on `UInt8` does: `0 - 2` comes back as 254 and every test would pass.
    `min`/`max` are the free functions -- `SIMD` has no `.min()` / `.max()`
    methods.
    """
    return min(VW(1), max(n, VW(k)) - VW(k))


def _ge(n: VW, k: Int) -> VW:
    """1 where `n >= k`. `_ge(n, 0)` is unconditionally 1, so it does not go
    through `_gt`, whose `k - 1` would be negative in a `UInt8` lane."""
    if k <= 0:
        return VW(1)
    return _gt(n, k - 1)


def _eq(n: VW, k: Int) -> VW:
    """1 where `n == k`, as the difference of two thresholds."""
    return _ge(n, k) - _ge(n, k + 1)


# ------------------------------------------------------------ scalar helpers
def _neighbours(up: U8Ptr, cu: U8Ptr, dn: U8Ptr, xm: Int, x: Int, xp: Int) -> Int:
    """The eight neighbours of cell column `x`, with the caller supplying
    already-wrapped column indices so the toroidal edges need no branch here."""
    return (
        Int(up.unsafe_load(xm))
        + Int(up.unsafe_load(x))
        + Int(up.unsafe_load(xp))
        + Int(cu.unsafe_load(xm))
        + Int(cu.unsafe_load(xp))
        + Int(dn.unsafe_load(xm))
        + Int(dn.unsafe_load(x))
        + Int(dn.unsafe_load(xp))
    )


def _age(old: UInt8, nv: UInt8) -> UInt8:
    """Age of a cell that is alive in the new generation.

    254, not 255: `old + 1` on a `UInt8` would wrap 255 to 0 and make an ancient
    cell look newborn. The clamp is here and in the vector body rather than at
    the palette lookup because the age buffer is a public output of `la_step`.
    """
    if nv == 0:
        return UInt8(0)
    if old >= 254:
        return UInt8(255)
    return old + UInt8(1)


def _apply(self: UInt8, n: Int, bmask: Int, smask: Int) -> UInt8:
    """General totalistic B/S rule, scalar."""
    var bit = bmask
    if self == 1:
        bit = smask
    if (bit >> n) & 1 == 1:
        return UInt8(1)
    return UInt8(0)


def _apply_fast(self: UInt8, n: Int) -> UInt8:
    """Conway B3/S23. `n` is at most 8, so the rule is two comparisons."""
    if n == 3 or (self == 1 and n == 2):
        return UInt8(1)
    return UInt8(0)


# ------------------------------------------------------------- one full row
def _advance[fast: Bool](
    cur: U8Ptr,
    nxt: U8Ptr,
    ag: U8Ptr,
    nag: U8Ptr,
    w: Int,
    h: Int,
    y: Int,
    bmask: Int,
    smask: Int,
):
    """Row `y` of one generation. `cur`/`ag` are read; `nxt`/`nag` are written
    whole, so the two ping-pong buffers never alias and rows need no barrier."""
    var up_y = y - 1
    if up_y < 0:
        up_y = h - 1
    var dn_y = y + 1
    if dn_y >= h:
        dn_y = 0

    var up = cur.unsafe_offset(up_y * w)
    var cu = cur.unsafe_offset(y * w)
    var dn = cur.unsafe_offset(dn_y * w)
    var o = nxt.unsafe_offset(y * w)
    var oa = ag.unsafe_offset(y * w)
    var on = nag.unsafe_offset(y * w)

    # The two wrapped columns go through the scalar path so the vector body can
    # run from x=1 to x=w-2 with unaligned loads at i-1, i and i+1 and never
    # read out of bounds.
    var v0 = _edge_cell[fast](up, cu, dn, w - 1, 0, 1, bmask, smask)
    o.unsafe_store(0, v0)
    on.unsafe_store(0, _age(oa.unsafe_load(0), v0))

    var vl = _edge_cell[fast](up, cu, dn, w - 2, w - 1, 0, bmask, smask)
    o.unsafe_store(w - 1, vl)
    on.unsafe_store(w - 1, _age(oa.unsafe_load(w - 1), vl))

    var i = 1
    while i + W <= w - 1:
        var n = (
            up.unsafe_load[width=W](i - 1)
            + up.unsafe_load[width=W](i)
            + up.unsafe_load[width=W](i + 1)
            + cu.unsafe_load[width=W](i - 1)
            + cu.unsafe_load[width=W](i + 1)
            + dn.unsafe_load[width=W](i - 1)
            + dn.unsafe_load[width=W](i)
            + dn.unsafe_load[width=W](i + 1)
        )
        var sv = cu.unsafe_load[width=W](i)

        var nv: VW
        if fast:
            # B3/S23 without a compare: exactly the two neighbour counts Conway
            # reacts to, each a difference of saturating thresholds.
            nv = _eq(n, 3) + (sv & _eq(n, 2))
        else:
            # General totalistic rule. Nine unrolled steps; `bmask`/`smask` are
            # runtime so the loop stays straight-line.
            var bornv = VW(0)
            var survv = VW(0)
            var k = 0
            while k <= 8:
                var e = _eq(n, k)
                bornv = bornv + e * VW(UInt8((bmask >> k) & 1))
                survv = survv + e * VW(UInt8((smask >> k) & 1))
                k += 1
            # `sv` is 0 or 1, so this XOR-multiply is a branchless select:
            # dead lanes take `bornv`, live lanes take `survv`.
            nv = bornv ^ (sv & (survv ^ bornv))

        var av = oa.unsafe_load[width=W](i)
        var nav = (min(av, VW(254)) + VW(1)) * nv

        o.unsafe_store[width=W](i, nv)
        on.unsafe_store[width=W](i, nav)
        i += W

    while i < w - 1:
        var nm = _neighbours(up, cu, dn, i - 1, i, i + 1)
        var sm = cu.unsafe_load(i)
        var tail: UInt8
        if fast:
            tail = _apply_fast(sm, nm)
        else:
            tail = _apply(sm, nm, bmask, smask)
        o.unsafe_store(i, tail)
        on.unsafe_store(i, _age(oa.unsafe_load(i), tail))
        i += 1


def _edge_cell[fast: Bool](
    up: U8Ptr,
    cu: U8Ptr,
    dn: U8Ptr,
    xm: Int,
    x: Int,
    xp: Int,
    bmask: Int,
    smask: Int,
) -> UInt8:
    """New state for one wrapped edge column, scalar.

    Parameterised on `fast` rather than branched on at the call site: a
    `var v = ...; if fast: v = ...` shape reads to the compiler as a dead store
    on the path it does not take, and it warns. Here both arms are the same
    expression and the parameter picks one at compile time.
    """
    var n = _neighbours(up, cu, dn, xm, x, xp)
    var self = cu.unsafe_load(x)
    if fast:
        return _apply_fast(self, n)
    return _apply(self, n, bmask, smask)


# --------------------------------------------------- all rows, serial or par
def _advance_all[fast: Bool](
    cur: U8Ptr,
    nxt: U8Ptr,
    ag: U8Ptr,
    nag: U8Ptr,
    w: Int,
    h: Int,
    bmask: Int,
    smask: Int,
    workers: Int,
):
    """One generation over every row.

    `workers <= 1` forces the serial loop. That is not a fallback for a grid
    too small to parallelise -- it is the measurement lever: `bench/bench.py`
    sweeps it to find where the launch actually starts paying for itself,
    rather than trusting a constant that was guessed on another machine.

    Rows are handed out in contiguous bands rather than one row per work item.
    The work is embarrassingly parallel either way, but a launch costs a fixed
    amount that does not shrink with the size of a work item, so fewer, fatter
    items is the shape that spends it once.
    """
    def band(t: Int) {imm}:
        var y0 = t * ROWS_PER_TASK
        var y1 = min(y0 + ROWS_PER_TASK, h)
        var y = y0
        while y < y1:
            _advance[fast](cur, nxt, ag, nag, w, h, y, bmask, smask)
            y += 1

    var tasks = (h + ROWS_PER_TASK - 1) // ROWS_PER_TASK
    if workers <= 1 or w * h < PAR_MIN_CELLS or tasks <= 1:
        var y = 0
        while y < h:
            _advance[fast](cur, nxt, ag, nag, w, h, y, bmask, smask)
            y += 1
    else:
        parallelize(band, tasks, min(tasks, workers))


def _step(
    cur: U8Ptr,
    nxt: U8Ptr,
    ag: U8Ptr,
    nag: U8Ptr,
    w: Int,
    h: Int,
    bmask: Int,
    smask: Int,
    workers: Int,
):
    if bmask == CONWAY_B and smask == CONWAY_S:
        _advance_all[True](cur, nxt, ag, nag, w, h, bmask, smask, workers)
    else:
        _advance_all[False](cur, nxt, ag, nag, w, h, bmask, smask, workers)


# ------------------------------------------------------------------ rendering
def _render_row(
    ag: U8Ptr,
    w: Int,
    y: Int,
    pal: U8Ptr,
    pscale: Int,
    orow: U8Ptr,
    irow: U8Ptr,
):
    """Age -> palette index -> RGB, for one row.

    The palette index is written alongside the RGB because a GIF is a
    palette-indexed format: recomputing `idx` from `rgb` in Python would be a
    second pass over the same data to recover a number the kernel already had.
    """
    var arow = ag.unsafe_offset(y * w)
    var x = 0
    while x < w:
        var idx = (Int(arow.unsafe_load(x)) * pscale) >> 8
        if idx > 255:
            idx = 255
        var p = pal.unsafe_offset(idx * 3)
        var o = x * 3
        orow.unsafe_store(o, p.unsafe_load(0))
        orow.unsafe_store(o + 1, p.unsafe_load(1))
        orow.unsafe_store(o + 2, p.unsafe_load(2))
        irow.unsafe_store(x, UInt8(idx))
        x += 1


def _render(
    ag: U8Ptr,
    w: Int,
    h: Int,
    pal: U8Ptr,
    pscale: Int,
    out_rgb: U8Ptr,
    out_idx: U8Ptr,
    workers: Int,
):
    def band(t: Int) {imm}:
        var y0 = t * ROWS_PER_TASK
        var y1 = min(y0 + ROWS_PER_TASK, h)
        _render_row(ag, w, y0, pal, pscale, out_rgb.unsafe_offset(y0 * w * 3), out_idx.unsafe_offset(y0 * w))
        var y = y0 + 1
        while y < y1:
            _render_row(ag, w, y, pal, pscale, out_rgb.unsafe_offset(y * w * 3), out_idx.unsafe_offset(y * w))
            y += 1

    var tasks = (h + ROWS_PER_TASK - 1) // ROWS_PER_TASK
    if workers <= 1 or w * h < PAR_MIN_CELLS or tasks <= 1:
        var y = 0
        while y < h:
            _render_row(
                ag, w, y, pal, pscale, out_rgb.unsafe_offset(y * w * 3), out_idx.unsafe_offset(y * w)
            )
            y += 1
    else:
        parallelize(band, tasks, min(tasks, workers))


# ------------------------------------------------------------ scalar sweeps
def _clear(gs: U8Ptr, ag: U8Ptr, n: Int):
    """Zero a grid/age pair.

    Vectorised rather than a byte loop: `clear` is called on every poster run
    and on the `density=0.0` edge case in the parity tests, and a `UInt8` sweep
    is exactly the shape `SIMD` is good at.
    """
    var i = 0
    while i + W <= n:
        gs.unsafe_store[width=W](i, VW(0))
        ag.unsafe_store[width=W](i, VW(0))
        i += W
    while i < n:
        gs.unsafe_store(i, UInt8(0))
        ag.unsafe_store(i, UInt8(0))
        i += 1


def _seed(g: U8Ptr, ag: U8Ptr, n: Int, seed: Int, density: Float64):
    """xorshift64* fill.

    Counter-based rather than threaded through a shared state, so the grid is
    bit-identical no matter how the cells are later distributed over workers,
    and identical to `reference.Reference.randomize` on the Python side.
    """
    var s = UInt64(seed)
    if s == 0:
        s = 0x9E3779B97F4A7C15
    var thr = Int(density * 1000000.0)
    if thr < 0:
        thr = 0
    elif thr > 1000000:
        thr = 1000000
    var i = 0
    while i < n:
        s = s ^ (s >> 12)
        s = s ^ (s << 25)
        s = s ^ (s >> 27)
        var r = (s * 0x2545F4914F6CDD1D) >> 11
        var alive = UInt8(0)
        if Int(r % 1000000) < thr:
            alive = UInt8(1)
        g.unsafe_store(i, alive)
        ag.unsafe_store(i, UInt8(0))
        i += 1


def _population(g: U8Ptr, n: Int) -> Int:
    # Every cell is 0 or 1 and W is at most 64, so the UInt8 accumulator cannot
    # overflow before `reduce_add` hands it to an Int.
    var total = 0
    var i = 0
    while i + W <= n:
        total += Int(g.unsafe_load[width=W](i).reduce_add())
        i += W
    while i < n:
        total += Int(g.unsafe_load(i))
        i += 1
    return total


def _any_ge(ag: U8Ptr, n: Int, k: UInt8) -> Bool:
    var i = 0
    while i + W <= n:
        if Int(_ge(ag.unsafe_load[width=W](i), Int(k)).reduce_add()) > 0:
            return True
        i += W
    while i < n:
        if ag.unsafe_load(i) >= k:
            return True
        i += 1
    return False


def _max_age(ag: U8Ptr, n: Int) -> Int:
    """Largest age present, as a binary search on "does any lane reach k".

    `SIMD` has no `.max()`, and "does any cell have age >= k" is one threshold
    plus one `reduce_add`. The answer is monotone in k, so eight of those beat a
    per-cell branch with a running maximum.
    """
    if n == 0:
        return 0
    var lo = 0
    var hi = 256
    while lo < hi:
        var mid = (lo + hi) >> 1
        if _any_ge(ag, n, UInt8(mid)):
            lo = mid + 1
        else:
            hi = mid
    if lo == 0:
        return 0
    return lo - 1


def _blit(
    dst: U8Ptr, dw: Int, src: U8Ptr, sw: Int, sh: Int, ox: Int, oy: Int
) -> Int:
    """Stamp a pattern with toroidal wrap. Returns how many cells were set."""
    var placed = 0
    var y = 0
    while y < sh:
        var dy = (oy + y) % dw
        var x = 0
        while x < sw:
            var dx = (ox + x) % dw
            if src.unsafe_load(y * sw + x) == 1:
                dst.unsafe_store(dy * dw + dx, UInt8(1))
                placed += 1
            x += 1
        y += 1
    return placed
