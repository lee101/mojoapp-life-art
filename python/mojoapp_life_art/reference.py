"""Pure-Python reference: the definition of correct, and the benchmark baseline.

Deliberately written the way a person writes Life — flat lists, explicit
neighbour loops, no clever tricks — because its job is to be obviously right
and obviously slow. `tests/test_parity.py` asserts the Mojo kernels agree with
it cell for cell; `bench/bench.py` reports the ratio without rounding the
unflattering rows out.

The semantics it pins down, and the Mojo kernels mirror:

* toroidal wrap in both axes
* a rule is a pair of 9-bit masks; bit `k` of `birth` births a dead cell with
  `k` live neighbours, bit `k` of `survive` keeps a live one alive
* `ages` counts consecutive generations alive, saturating at 255
"""

from __future__ import annotations

CONWAY = (1 << 3, (1 << 2) | (1 << 3))

# The rules worth having by name. Totalistic B/S notation, encoded as the two
# masks the kernels take.
RULES: dict[str, tuple[str, int, int]] = {
    "conway": ("B3/S23", *CONWAY),
    "highlife": ("B36/S23", (1 << 3) | (1 << 6), (1 << 2) | (1 << 3)),
    "daynight": ("B3678/S34678", 0b11110000, 0b11110100),
    "seeds": ("B2/S", 1 << 2, 0),
    "maze": ("B3/S12345", 1 << 3, 0b111110),
    "coral": ("B3/S45678", 1 << 3, 0b11110000),
    "replicator": ("B1357/S1357", 0b10101010, 0b10101010),
    "2x2": ("B36/S125", (1 << 3) | (1 << 6), 0b100110),
}


class Reference:
    """One toroidal grid plus its age buffer."""

    def __init__(self, width: int, height: int):
        if width < 3 or height < 3:
            raise ValueError("grid must be at least 3x3 for toroidal wrap")
        self.w = width
        self.h = height
        self.grid = [0] * (width * height)
        self.ages = [0] * (width * height)

    # --------------------------------------------------------------- helpers
    def idx(self, x: int, y: int) -> int:
        return (y % self.h) * self.w + (x % self.w)

    def get(self, x: int, y: int) -> int:
        return self.grid[self.idx(x, y)]

    def set(self, x: int, y: int, v: int) -> None:
        self.grid[self.idx(x, y)] = v

    def live_cells(self) -> list[tuple[int, int]]:
        return [
            (i % self.w, i // self.w)
            for i, v in enumerate(self.grid)
            if v
        ]

    def population(self) -> int:
        return sum(self.grid)

    def max_age(self) -> int:
        return max(self.ages) if self.ages else 0

    # ---------------------------------------------------------------- seeding
    def randomize(self, seed: int = 1, density: float = 0.3) -> None:
        """The same xorshift64* fill the Mojo `_seed` does, so a seeded grid is
        bit-identical across the two implementations and the demo is
        reproducible from the seed alone."""
        s = seed & 0xFFFFFFFFFFFFFFFF or 0x9E3779B97F4A7C15
        thr = max(0, min(1_000_000, int(density * 1_000_000)))
        mask = (1 << 64) - 1
        for i in range(self.w * self.h):
            s ^= (s >> 12) & mask
            s = (s ^ (s << 25)) & mask
            s ^= s >> 27
            r = ((s * 0x2545F4914F6CDD1D) & mask) >> 11
            self.grid[i] = 1 if (r % 1_000_000) < thr else 0
            self.ages[i] = 0

    def clear(self) -> None:
        self.grid = [0] * (self.w * self.h)
        self.ages = [0] * (self.w * self.h)

    def blit(self, pattern: list[str], ox: int = 0, oy: int = 0) -> int:
        """Stamp a pattern from text rows like `["010", "101"]`."""
        placed = 0
        for y, row in enumerate(pattern):
            for x, ch in enumerate(row):
                if ch in "1#O*":
                    self.set(ox + x, oy + y, 1)
                    placed += 1
        return placed

    # --------------------------------------------------------------- stepping
    def step(self, rule: tuple[int, int] = CONWAY) -> int:
        """One generation. Returns the new population."""
        birth, survive = rule
        w, h = self.w, self.h
        g = self.grid
        out = [0] * (w * h)
        ages = self.ages
        nages = [0] * (w * h)
        for y in range(h):
            ym = (y - 1) % h
            yp = (y + 1) % h
            row = y * w
            for x in range(w):
                n = 0
                for ny in (ym, y, yp):
                    nrow = ny * w
                    for nx in ((x - 1) % w, x, (x + 1) % w):
                        if ny == y and nx == x:
                            continue
                        n += g[nrow + nx]
                selfv = g[row + x]
                bit = survive if selfv else birth
                if (bit >> n) & 1:
                    out[row + x] = 1
                    old = ages[row + x]
                    nages[row + x] = 255 if old >= 254 else old + 1
        self.grid = out
        self.ages = nages
        return sum(out)

    def run(self, gens: int, rule: tuple[int, int] = CONWAY) -> int:
        pop = 0
        for _ in range(gens):
            pop = self.step(rule)
        return pop

    # -------------------------------------------------------------- rendering
    def render(self, palette, pscale: int = 256):
        """Same mapping the Mojo renderer does: age -> palette index -> RGB."""
        rgb = bytearray(self.w * self.h * 3)
        idx = bytearray(self.w * self.h)
        for i, age in enumerate(self.ages):
            p = (age * pscale) >> 8
            if p > 255:
                p = 255
            r, g, b = palette[p]
            rgb[i * 3] = r
            rgb[i * 3 + 1] = g
            rgb[i * 3 + 2] = b
            idx[i] = p
        return rgb, idx


# Small known patterns, as text rows. Used by the demo, the CLI and the tests.
PATTERNS: dict[str, list[str]] = {
    "glider": [".1.", "..1", "111"],
    "block": ["11", "11"],
    "blinker": ["111"],
    "r_pentomino": [".11", "11.", ".1."],
    "acorn": [".1.....", "...1...", "1..1.11"],
    "diehard": ["......1.", ".1....1.", "1.1..1.."],
    "pulsar": [
        "..11111..",
        ".........",
        "1.1...1.1",
        ".........",
        "..1.1.1..",
        "..1.1.1..",
        "..1.1.1..",
        ".........",
        "1.1...1.1",
        ".........",
        "..11111..",
    ],
}