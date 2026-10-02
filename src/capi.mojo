"""The C ABI the Python bindings call through.

Every buffer crosses as an `Int` address: `@export` rejects parametric
functions, and a pointer with an inferred origin is parametric. Pointers are
non-nullable here, so each one is constructed inside the branch that uses it.
Callers own all memory, including the ping-pong pairs, so nothing here
allocates and nothing here can leak. This is the only file where the Mojo
divergence lives; the algorithms stay in `ported.mojo`.
"""

from ported import (
    PAR_MIN_CELLS,
    PAR_WORKERS,
    SIMD_WIDTH,
    _blit,
    _clear,
    _max_age,
    _population,
    _render,
    _seed,
    _step,
)


comptime U8Ptr = Pointer[UInt8, AnyOrigin[mut=True]]


@export("la_workers")
def la_workers() abi("C") -> Int:
    """Default worker count the parallel path uses; the Python side reports it
    so the benchmark table cannot claim more parallelism than it got."""
    return PAR_WORKERS


@export("la_par_min_cells")
def la_par_min_cells() abi("C") -> Int:
    """Cells below which the serial loop runs instead of `parallelize`."""
    return PAR_MIN_CELLS


@export("la_simd_width")
def la_simd_width() abi("C") -> Int:
    """Lanes in the `UInt8` vector body.

    Queried rather than restated in Python so the two cannot drift: the tests
    probe widths either side of this number to land on the vector tail.
    """
    return SIMD_WIDTH


@export("la_seed")
def la_seed_c(
    grid: Int, ages: Int, w: Int, h: Int, seed: Int, density: Float64
) abi("C"):
    if grid == 0 or ages == 0:
        return
    if w < 3 or h < 3:
        return
    _seed(
        U8Ptr(unsafe_from_address=grid),
        U8Ptr(unsafe_from_address=ages),
        w * h,
        seed,
        density,
    )


@export("la_zero")
def la_zero_c(grid: Int, ages: Int, n: Int) abi("C"):
    if grid == 0 or ages == 0 or n <= 0:
        return
    _clear(U8Ptr(unsafe_from_address=grid), U8Ptr(unsafe_from_address=ages), n)


@export("la_step")
def la_step_c(
    cur: Int,
    nxt: Int,
    ages: Int,
    nag: Int,
    w: Int,
    h: Int,
    bmask: Int,
    smask: Int,
    workers: Int,
) abi("C"):
    if cur == 0 or nxt == 0 or ages == 0 or nag == 0:
        return
    if w < 3 or h < 3:
        return
    _step(
        U8Ptr(unsafe_from_address=cur),
        U8Ptr(unsafe_from_address=nxt),
        U8Ptr(unsafe_from_address=ages),
        U8Ptr(unsafe_from_address=nag),
        w,
        h,
        bmask,
        smask,
        workers,
    )


@export("la_run")
def la_run_c(
    grid: Int,
    scratch: Int,
    ages: Int,
    scratch_ages: Int,
    w: Int,
    h: Int,
    bmask: Int,
    smask: Int,
    gens: Int,
    workers: Int,
) abi("C") -> Int:
    """Run `gens` generations ping-ponging between `grid` and `scratch`.

    Returns which buffer holds the final state: 0 for the caller's `grid`
    buffer, 1 for `scratch`. An even generation count ends back in `grid`, an
    odd one in `scratch`, and a caller that assumes otherwise reads a stale
    generation -- so the answer is returned rather than guessed. No copy-back
    happens; the parity tests drive it one generation at a time through
    `la_step`, and `Simulation.run` swaps its own pointer accordingly.
    """
    if grid == 0 or scratch == 0 or ages == 0 or scratch_ages == 0:
        return 0
    if w < 3 or h < 3 or gens <= 0:
        return 0
    var p0 = U8Ptr(unsafe_from_address=grid)
    var p1 = U8Ptr(unsafe_from_address=scratch)
    var q0 = U8Ptr(unsafe_from_address=ages)
    var q1 = U8Ptr(unsafe_from_address=scratch_ages)
    var cur_p = p0
    var dst_p = p1
    var cur_a = q0
    var dst_a = q1
    var g = 0
    while g < gens:
        _step(cur_p, dst_p, cur_a, dst_a, w, h, bmask, smask, workers)
        var tp = cur_p
        cur_p = dst_p
        dst_p = tp
        var tq = cur_a
        cur_a = dst_a
        dst_a = tq
        g += 1
    return (gens & 1)


@export("la_run_frames")
def la_run_frames_c(
    grid: Int,
    scratch: Int,
    ages: Int,
    scratch_ages: Int,
    w: Int,
    h: Int,
    bmask: Int,
    smask: Int,
    gens: Int,
    stride: Int,
    out_rgb: Int,
    out_idx: Int,
    pal: Int,
    pscale: Int,
    max_frames: Int,
    workers: Int,
) abi("C") -> Int:
    """Evolve `gens` generations, rendering a frame every `stride` generations.

    Returns `(frames_written << 1) | final_buffer`, where the buffer bit is 0
    for `grid` and 1 for `scratch` on the same odd/even convention as `la_run`.
    Frame 0 is the starting state, so a GIF has one more frame than
    `gens // stride`; the last frame is always the final generation rather than
    whatever the stride last landed on, unless `max_frames` ran out first --
    in which case the run genuinely stopped there and the frames say so.
    """
    if (
        grid == 0
        or scratch == 0
        or ages == 0
        or scratch_ages == 0
        or out_rgb == 0
        or out_idx == 0
        or pal == 0
    ):
        return 0
    if w < 3 or h < 3 or stride < 1 or max_frames < 1:
        return 0
    var n = w * h
    var p0 = U8Ptr(unsafe_from_address=grid)
    var p1 = U8Ptr(unsafe_from_address=scratch)
    var q0 = U8Ptr(unsafe_from_address=ages)
    var q1 = U8Ptr(unsafe_from_address=scratch_ages)
    var p = U8Ptr(unsafe_from_address=pal)
    var orgb = U8Ptr(unsafe_from_address=out_rgb)
    var oidx = U8Ptr(unsafe_from_address=out_idx)
    var cur_p = p0
    var dst_p = p1
    var cur_a = q0
    var dst_a = q1
    var frames = 0
    var g = 0
    while True:
        var want = g == gens
        if not want and g % stride == 0 and frames < max_frames:
            want = True
        if want:
            _render(
                cur_a,
                w,
                h,
                p,
                pscale,
                orgb.unsafe_offset(frames * n * 3),
                oidx.unsafe_offset(frames * n),
                workers,
            )
            frames += 1
        if g == gens or frames >= max_frames:
            break
        _step(cur_p, dst_p, cur_a, dst_a, w, h, bmask, smask, workers)
        var tp = cur_p
        cur_p = dst_p
        dst_p = tp
        var tq = cur_a
        cur_a = dst_a
        dst_a = tq
        g += 1
    return (frames << 1) | (g & 1)


@export("la_render")
def la_render_c(
    grid: Int,
    ages: Int,
    w: Int,
    h: Int,
    pal: Int,
    pscale: Int,
    out_rgb: Int,
    out_idx: Int,
    workers: Int,
) abi("C"):
    if grid == 0 or ages == 0 or pal == 0 or out_rgb == 0 or out_idx == 0:
        return
    if w < 3 or h < 3:
        return
    _render(
        U8Ptr(unsafe_from_address=ages),
        w,
        h,
        U8Ptr(unsafe_from_address=pal),
        pscale,
        U8Ptr(unsafe_from_address=out_rgb),
        U8Ptr(unsafe_from_address=out_idx),
        workers,
    )


@export("la_population")
def la_population_c(grid: Int, n: Int) abi("C") -> Int:
    if grid == 0 or n <= 0:
        return 0
    return _population(U8Ptr(unsafe_from_address=grid), n)


@export("la_max_age")
def la_max_age_c(ages: Int, n: Int) abi("C") -> Int:
    if ages == 0 or n <= 0:
        return 0
    return _max_age(U8Ptr(unsafe_from_address=ages), n)


@export("la_blit")
def la_blit_c(
    grid: Int,
    w: Int,
    h: Int,
    pat: Int,
    pw: Int,
    ph: Int,
    ox: Int,
    oy: Int,
) abi("C") -> Int:
    if grid == 0 or pat == 0 or pw < 1 or ph < 1 or w < 3 or h < 3:
        return 0
    return _blit(
        U8Ptr(unsafe_from_address=grid),
        w,
        U8Ptr(unsafe_from_address=pat),
        pw,
        ph,
        ox,
        oy,
    )