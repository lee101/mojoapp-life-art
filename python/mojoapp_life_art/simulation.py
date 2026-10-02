"""The simulation object: NumPy buffers in, compiled Mojo out.

This is the whole bridge. It owns two grid buffers and two age buffers, hands
their addresses to the shared library as `Int`, and keeps the parity between
the buffers straight — the FFI cannot check it, so the tests do.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

from . import palettes, reference
from ._lib import addr, lib

U8 = np.uint8

# The buffer contract every kernel in src/ assumes. Stated here as an
# assertion rather than a comment because a silent width error would produce a
# plausible-looking wrong answer instead of a crash.
_CONTIG = np.ascontiguousarray

#: Grid used for the one-time serial/parallel calibration. Large on purpose: at
#: a small size the launch cost swamps the work and the measurement is noise.
#: 2048x2048 is a few milliseconds serial, so the whole calibration costs tens
#: of milliseconds and still sees a clear signal.
_PROBE_SIDE = 2048
_PROBE_GENS = 2
_PROBE_REPEATS = 7

_calibration: dict | None = None


def calibrate(repeats: int = _PROBE_REPEATS) -> dict:
    """Measure where the serial loop stops beating the parallel one.

    Two numbers decide it: how long one generation costs in the serial loop,
    and how much a `parallelize` launch adds on top. A launch costs a fixed
    amount per generation, so the break-even cell count is a ratio of the two --
    which is why it is measured rather than hard-coded. A constant written on
    one machine is wrong on every other one, and on a host this contended the
    difference is not a rounding error.

    The two paths are sampled alternately rather than in blocks, so drift in
    host load lands on both instead of on whichever ran second. Each takes its
    own minimum: the question is how fast the loop goes when it is not
    competing with something else, and on a shared host the minimum sample is
    the only one not measuring the neighbours.

    Runs once per process and caches. The two paths are asserted cell-for-cell
    identical in `tests/test_parity.py`, so choosing one or the other cannot
    change an answer -- only how long it takes.
    """
    global _calibration
    if _calibration is not None:
        return _calibration

    cells = _PROBE_SIDE * _PROBE_SIDE
    workers = default_workers()
    probe = Simulation(_PROBE_SIDE, _PROBE_SIDE, workers=1)

    def sample(path_workers: int) -> float:
        probe.workers = path_workers
        probe.randomize(seed=0, density=0.3)
        t0 = time.perf_counter()
        probe.run(_PROBE_GENS)
        return time.perf_counter() - t0

    sample(1)
    sample(workers)

    serial_s = math.inf
    parallel_s = math.inf
    for _ in range(repeats):
        serial_s = min(serial_s, sample(1))
        parallel_s = min(parallel_s, sample(workers))

    serial_per_cell = serial_s / (_PROBE_GENS * cells)
    parallel_per_cell = parallel_s / (_PROBE_GENS * cells)
    launch_overhead = parallel_per_cell - serial_per_cell

    if launch_overhead <= 0:
        # The parallel path was not slower at the probe size: the launch is
        # cheap here and it wins from the kernel's own floor upward.
        threshold = float(par_min_cells())
    else:
        threshold = cells * serial_per_cell / launch_overhead

    _calibration = {
        "probe_cells": cells,
        "probe_workers": workers,
        "serial_ns_per_cell": serial_per_cell * 1e9,
        "parallel_ns_per_cell": parallel_per_cell * 1e9,
        "parallel_wins_at_probe": parallel_per_cell < serial_per_cell,
        "launch_ns_per_generation": launch_overhead * 1e9 * cells,
        "threshold_cells": threshold,
    }
    return _calibration


def parallel_threshold_cells() -> float:
    """Cells per generation at or above which the parallel loop wins.

    `inf` means it never wins on this host, and `Simulation` runs everything
    serially. That is a real, measured answer rather than a missing feature.
    """
    return calibrate()["threshold_cells"]


def _grid(w: int, h: int) -> np.ndarray:
    return _CONTIG(np.zeros((h, w), dtype=U8))


class Simulation:
    """A toroidal CA grid backed by the compiled kernels.

    `width` and `height` must be at least 3: the vector body reads columns
    `i-1`, `i` and `i+1` starting at 1, and a narrower grid has no interior.
    """

    def __init__(self, width: int, height: int, workers: int | None = None):
        """`workers` is the worker count handed to `parallelize`.

        It defaults to the kernel's own default (16) and is clamped to what the
        machine can use. `workers=1` forces the serial loop, which is not a
        size-based fallback but the lever `bench/bench.py` uses to measure
        where the parallel launch actually starts paying for itself.
        """
        if width < 3 or height < 3:
            raise ValueError(
                f"grid must be at least 3x3 for toroidal wrap, got {width}x{height}"
            )
        self.w = width
        self.h = height
        self.n = width * height
        if workers is None:
            self.workers = default_workers()
            self._auto = True
        else:
            # An explicit count is a request to use exactly that path, and the
            # calibration is not allowed to override it.
            self.workers = max(1, int(workers))
            self._auto = False
        self._effective: int | None = None
        self._a = _grid(width, height)
        self._b = _grid(width, height)
        self._age_a = _grid(width, height)
        self._age_b = _grid(width, height)
        self._a_in = self._a  # which buffer holds the live grid

    # --------------------------------------------------------------- scheduling
    def _workers_for(self) -> int:
        """The worker count this grid will actually be stepped with.

        An explicit `workers` skips the calibration but not the kernel's own
        size floor: asking for 8 workers on a 32x32 grid cannot produce a
        parallel launch, and reporting that it did would be a lie.

        One calibration per process, then a comparison per grid. Cheap enough to
        do inside the call: the alternative is a stale constant that is wrong
        on whatever machine did not write it.
        """
        if self._effective is None:
            parallel = self.n >= par_min_cells()
            if self._auto:
                parallel = parallel and self.n >= parallel_threshold_cells()
            self._effective = self.workers if parallel else 1
        return self._effective

    @property
    def uses_parallel(self) -> bool:
        """Whether this grid is large enough -- on this host, today -- to be
        worth a `parallelize` launch."""
        return self._workers_for() > 1

    # ------------------------------------------------------------------ state
    @property
    def grid(self) -> np.ndarray:
        """The live grid as an `(h, w)` `uint8` view. Read-only in practice:
        mutate a copy, or use `set_grid`."""
        return self._a_in

    @property
    def ages(self) -> np.ndarray:
        return self._age_a if self._a_in is self._a else self._age_b

    def _live_ages(self) -> np.ndarray:
        return self._age_a if self._a_in is self._a else self._age_b

    def _spare_ages(self) -> np.ndarray:
        return self._age_b if self._a_in is self._a else self._age_a

    def _spare_grid(self) -> np.ndarray:
        return self._b if self._a_in is self._a else self._a

    def copy(self) -> "Simulation":
        # Explicit worker count, not the default: the calibration is a property
        # of this process, not of the grid being copied.
        other = Simulation(self.w, self.h, self.workers)
        other._a[...] = self._a_in
        other._b[...] = self._spare_grid()
        other._age_a[...] = self._live_ages()
        other._age_b[...] = self._spare_ages()
        other._a_in = other._a if self._a_in is self._a else other._b
        return other

    # ------------------------------------------------------------------- seed
    def clear(self) -> None:
        lib().la_zero(addr(self._a), addr(self._age_a), self.n)
        lib().la_zero(addr(self._b), addr(self._age_b), self.n)

    def randomize(self, seed: int = 1, density: float = 0.3) -> None:
        lib().la_seed(
            addr(self._a), addr(self._age_a), self.w, self.h, int(seed), float(density)
        )
        self._a_in = self._a

    def set_grid(self, grid) -> None:
        """Load a grid from a nested sequence, an `(h, w)` array, or a flat
        row-major sequence of 0/1."""
        arr = np.asarray(grid, dtype=U8)
        if arr.size != self.n:
            raise ValueError(
                f"grid must hold {self.n} cells ({self.h}x{self.w}), got {arr.size}"
            )
        flat = _CONTIG(arr.reshape(-1))
        if not np.all((flat == 0) | (flat == 1)):
            raise ValueError("cells must be 0 or 1")
        self._a[...] = 0
        self._a_in[...] = flat.reshape(self.h, self.w)
        self._age_a[...] = 0
        self._a_in = self._a

    def blit(self, pattern: list[str], ox: int = 0, oy: int = 0) -> int:
        """Stamp a text pattern, e.g. `["010", "101"]`."""
        rows = len(pattern)
        cols = max(len(r) for r in pattern)
        flat = np.zeros(rows * cols, dtype=U8)
        for y, row in enumerate(pattern):
            for x, ch in enumerate(row):
                if ch in "1#O*":
                    flat[y * cols + x] = 1
        return int(
            lib().la_blit(
                addr(self._a_in), self.w, self.h, addr(flat), cols, rows, ox, oy
            )
        )

    def live_cells(self) -> list[tuple[int, int]]:
        """Live cells as `(x, y)`, row-major. Convenience for tests and for
        inspecting a run by hand; not on any hot path."""
        ys, xs = np.nonzero(self.grid)
        return [(int(x), int(y)) for y, x in zip(ys, xs)]

    # --------------------------------------------------------------- stepping
    def step(self, rule: tuple[int, int] = reference.CONWAY) -> None:
        """One generation, ping-ponging which buffer is live."""
        birth, survive = rule
        live = self._a_in
        spare = self._spare_grid()
        live_ages = self._live_ages()
        spare_ages = self._spare_ages()
        lib().la_step(
            addr(live),
            addr(spare),
            addr(live_ages),
            addr(spare_ages),
            self.w,
            self.h,
            birth,
            survive,
            self._workers_for(),
        )
        self._a_in = spare

    def run(self, gens: int, rule: tuple[int, int] = reference.CONWAY) -> None:
        """`gens` generations in one FFI call.

        The ping-pong and the odd-generation swap both live in Mojo, so the
        per-generation ctypes round trip never happens. `la_run` returns which
        buffer the result ended up in -- an odd count leaves it in the spare --
        and `self._a_in` follows it. There is no copy-back, because a copy of
        a grid the kernel just wrote would be pure overhead on the path this
        method exists to keep short.
        """
        if gens <= 0:
            return
        birth, survive = rule
        landed_in_spare = int(
            lib().la_run(
                addr(self._a_in),
                addr(self._spare_grid()),
                addr(self._live_ages()),
                addr(self._spare_ages()),
                self.w,
                self.h,
                birth,
                survive,
                int(gens),
                self._workers_for(),
            )
        )
        if landed_in_spare:
            self._a_in = self._spare_grid()

    def run_frames(
        self,
        gens: int,
        stride: int,
        rule: tuple[int, int] = reference.CONWAY,
        palette: str | None = None,
        max_frames: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evolve and render along the way. Returns `(rgb, indices)`.

        `rgb` is `(frames, h, w, 3)` `uint8`, `indices` is `(frames, h, w)`
        `uint8` — the same values, quantised to palette slots, which is what
        the GIF encoder wants. Frame 0 is the starting state and the last frame
        is always generation `gens`.

        When `max_frames` is too small to reach `gens` the run genuinely stops
        early, and the arrays come back shorter than the worst case. The final
        grid left on `self` is the one the last frame was rendered from, so
        `population()` agrees with `indices[-1]`.
        """
        birth, survive = rule
        pal = palettes.get(palette or palettes.DEFAULT_PALETTE)
        pal_b = np.frombuffer(palettes.to_bytes(pal), dtype=U8)
        # ceil((gens + 1) / stride) frames is the true count: frame 0 plus one
        # per stride plus the forced final, unless the cap truncates it.
        want = max_frames if max_frames is not None else (gens // stride) + 2
        want = max(1, min(want, (gens // stride) + 2))
        rgb = np.zeros((want, self.h, self.w, 3), dtype=U8)
        idx = np.zeros((want, self.h, self.w), dtype=U8)
        packed = int(
            lib().la_run_frames(
                addr(self._a_in),
                addr(self._spare_grid()),
                addr(self._live_ages()),
                addr(self._spare_ages()),
                self.w,
                self.h,
                birth,
                survive,
                int(gens),
                int(stride),
                addr(rgb),
                addr(idx),
                addr(pal_b),
                256,
                want,
                self._workers_for(),
            )
        )
        frames, landed_in_spare = packed >> 1, packed & 1
        if landed_in_spare:
            self._a_in = self._spare_grid()
        return rgb[:frames], idx[:frames]

    # ------------------------------------------------------------- reductions
    def population(self) -> int:
        return int(lib().la_population(addr(self._a_in), self.n))

    def max_age(self) -> int:
        return int(lib().la_max_age(addr(self._live_ages()), self.n))

    # -------------------------------------------------------------- rendering
    def render(self, palette: str | None = None, pscale: int = 256):
        """Render the current state. Returns `(rgb, indices)`, both shaped
        `(h, w)` / `(h, w, 3)`."""
        pal = palettes.get(palette or palettes.DEFAULT_PALETTE)
        pal_b = np.frombuffer(palettes.to_bytes(pal), dtype=U8)
        rgb = np.zeros((self.h, self.w, 3), dtype=U8)
        idx = np.zeros((self.h, self.w), dtype=U8)
        lib().la_render(
            addr(self._a_in),
            addr(self._live_ages()),
            self.w,
            self.h,
            addr(pal_b),
            int(pscale),
            addr(rgb),
            addr(idx),
            self._workers_for(),
        )
        return rgb, idx


# -------------------------------------------------------------- library facts
def workers() -> int:
    """Worker count the kernel uses by default."""
    return int(lib().la_workers())


def default_workers() -> int:
    """The default, capped at what this machine can actually use.

    The kernel's constant is 16 because that is what a poster-sized grid keeps
    busy; asking for 64 on a 4-core box would be a worse number, not a truer
    one.
    """
    return max(1, min(workers(), os.cpu_count() or 1))


def par_min_cells() -> int:
    return int(lib().la_par_min_cells())


def simd_width() -> int:
    return int(lib().la_simd_width())


def rule_masks(name: str) -> tuple[int, int]:
    """Look up a named rule's `(birth, survive)` masks."""
    return check_rule(name)


def check_rule(name: str) -> tuple[int, int]:
    """Resolve a rule name to its masks, with a message that lists the
    alternatives. Every entry point that takes `--rule` goes through here, so
    a typo is a message rather than a `KeyError` traceback."""
    try:
        entry = reference.RULES[name]
    except KeyError:
        raise ValueError(
            f"unknown rule {name!r}; choose from {', '.join(sorted(reference.RULES))}"
        ) from None
    return entry[1], entry[2]


def rule_names() -> list[str]:
    return sorted(reference.RULES)


def rule_notation(name: str) -> str:
    return reference.RULES[name][0]