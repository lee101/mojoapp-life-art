"""Parity between the compiled Mojo kernels and the pure-Python reference.

The reference in `mojoapp_life_art.reference` is the specification: flat lists,
explicit neighbour loops, no cleverness. If the two disagree on a single cell,
one of them is wrong, and the tests below say which by printing the offending
coordinate rather than just `assert False`.

Coverage is deliberately layered:
  * every named rule, so the general unrolled vector body is exercised as well
    as the Conway special case
  * widths either side of the SIMD width (64) and of the parallel threshold
    (65536 cells), because those are the two boundaries where the vector tail,
    the wrapped edge columns, and the serial/parallel switch live
  * seeds, so the deterministic fill is checked rather than assumed
"""

from __future__ import annotations

import numpy as np
import pytest

from mojoapp_life_art import reference
from mojoapp_life_art.simulation import Simulation, par_min_cells, simd_width

ALL_RULES = sorted(reference.RULES)

# Below the parallel threshold: serial path. Above it: `parallelize`.
SMALL = (37, 29)
BIG = (320, 240)


def ref_array(r: reference.Reference, field: str = "grid") -> np.ndarray:
    return np.array(getattr(r, field), dtype=np.uint8).reshape(r.h, r.w)


def assert_parity(sim: Simulation, ref: reference.Reference, context: str) -> None:
    """Both buffers must agree, and the mismatch message must name a cell."""
    for field, sim_arr in (("grid", sim.grid), ("ages", sim.ages)):
        ref_arr = ref_array(ref, field)
        if sim_arr.shape != ref_arr.shape:
            pytest.fail(
                f"{context}: {field} shape {sim_arr.shape} != {ref_arr.shape}"
            )
        if np.array_equal(sim_arr, ref_arr):
            continue
        diff = sim_arr.astype(np.int16) != ref_arr.astype(np.int16)
        ys, xs = np.nonzero(diff)
        y, x = int(ys[0]), int(xs[0])
        pytest.fail(
            f"{context}: {field} differs in {int(diff.sum())}/{diff.size} cells; "
            f"first at (x={x}, y={y}): mojo={int(sim_arr[y, x])} "
            f"reference={int(ref_arr[y, x])}"
        )


# ------------------------------------------------------------------- seeding
@pytest.mark.parametrize("seed", [0, 1, 7, 12345, 2**63 + 17])
@pytest.mark.parametrize("density", [0.0, 0.05, 0.3, 0.5, 1.0])
def test_seed_matches_reference(seed, density):
    """The xorshift64* fill must be bit-identical, or the demo stops being
    reproducible from its seed."""
    sim = Simulation(64, 48)
    sim.randomize(seed=seed, density=density)
    ref = reference.Reference(64, 48)
    ref.randomize(seed=seed, density=density)
    assert_parity(sim, ref, f"seed={seed} density={density}")


def test_seed_extremes():
    """density=0 must be empty and density=1 must be full — a threshold bug in
    `_seed` would pass every middle-density parity test."""
    for density, expected in ((0.0, 0), (1.0, 64 * 48)):
        sim = Simulation(64, 48)
        sim.randomize(seed=99, density=density)
        assert sim.population() == expected


# ------------------------------------------------------- stepping, all rules
@pytest.mark.parametrize("rule_name", ALL_RULES)
@pytest.mark.parametrize("shape", [SMALL, BIG])
def test_step_parity_all_rules(rule_name, shape):
    """Every named rule, six generations, both grid sizes.

    `coral` is in this list because it happens to die out to an empty grid from
    a 0.28 soup in a handful of generations — which makes it a real test of the
    empty-grid path and would hide a "everything dies" bug if it were alone.
    """
    birth, survive = reference.RULES[rule_name][1:]
    rule = (birth, survive)
    w, h = shape
    sim = Simulation(w, h)
    sim.randomize(seed=3, density=0.28)
    ref = reference.Reference(w, h)
    ref.randomize(seed=3, density=0.28)
    for gen in range(6):
        sim.step(rule)
        ref.step(rule)
        assert_parity(sim, ref, f"rule={rule_name} shape={shape} gen={gen + 1}")


def test_parity_across_simd_boundary():
    """Widths immediately either side of `simd_width_of[uint8]`, which is where
    the unaligned vector body meets the scalar tail."""
    w = simd_width()
    assert w > 1, "the vector width should be more than one lane"
    for width in (w - 2, w - 1, w, w + 1, w + 2, 2 * w + 3):
        sim = Simulation(width, 11)
        sim.randomize(seed=5, density=0.35)
        ref = reference.Reference(width, 11)
        ref.randomize(seed=5, density=0.35)
        for _ in range(3):
            sim.step(reference.CONWAY)
            ref.step(reference.CONWAY)
        assert_parity(sim, ref, f"width={width} (simd_width={w})")


def test_parity_across_parallel_threshold():
    """Just under and just over the cell count at which `parallelize` takes
    over. The serial and parallel paths must be the same answer."""
    threshold = par_min_cells()
    for cells in (threshold - 1, threshold, threshold + 1):
        side = int(cells**0.5) + 1
        sim = Simulation(side, side)
        sim.randomize(seed=11, density=0.3)
        ref = reference.Reference(side, side)
        ref.randomize(seed=11, density=0.3)
        for _ in range(2):
            sim.step(reference.CONWAY)
            ref.step(reference.CONWAY)
        assert_parity(sim, ref, f"cells~{cells} side={side} threshold={threshold}")


# ---------------------------------------------------------------- long runs
@pytest.mark.parametrize("rule_name", ["conway", "daynight"])
def test_parity_over_many_generations(rule_name):
    """50 generations, so a bug that only shows up once a run settles — an age
    that never saturates, say — cannot pass."""
    rule = reference.RULES[rule_name][1:]
    sim = Simulation(96, 96)
    sim.randomize(seed=21, density=0.25)
    ref = reference.Reference(96, 96)
    ref.randomize(seed=21, density=0.25)
    for gen in range(50):
        sim.step(rule)
        ref.step(rule)
    assert_parity(sim, ref, f"rule={rule_name} gen=50")


def test_run_matches_repeated_step():
    """`la_run` ping-pongs inside one FFI call; driving it one generation at a
    time through `la_step` must land on the same state, including the ages."""
    sim_run = Simulation(80, 80)
    sim_run.randomize(seed=31, density=0.3)
    sim_step = sim_run.copy()
    ref = reference.Reference(80, 80)
    ref.randomize(seed=31, density=0.3)

    sim_run.run(17, reference.CONWAY)
    for _ in range(17):
        sim_step.step(reference.CONWAY)
        ref.step(reference.CONWAY)

    assert_parity(sim_run, ref, "la_run(17) vs 17x la_step vs reference")
    assert_parity(sim_step, ref, "17x la_step vs reference")


@pytest.mark.parametrize("gens", [1, 2, 3, 8, 9])
def test_run_odd_and_even_copy_back(gens):
    """An even generation count ends the ping-pong back in the original buffer
    and an odd one does not, so the copy-back in `la_run` is what keeps
    `Simulation.run` honest. Odd counts are the ones that can get it wrong."""
    sim = Simulation(48, 48)
    sim.randomize(seed=41, density=0.3)
    ref = reference.Reference(48, 48)
    ref.randomize(seed=41, density=0.3)
    sim.run(gens, reference.CONWAY)
    ref.run(gens, reference.CONWAY)
    assert_parity(sim, ref, f"run({gens})")


def test_run_zero_and_negative_are_noops():
    sim = Simulation(32, 32)
    sim.randomize(seed=2, density=0.3)
    before = sim.grid.copy()
    ages_before = sim.ages.copy()
    sim.run(0)
    sim.run(-5)
    assert np.array_equal(sim.grid, before)
    assert np.array_equal(sim.ages, ages_before)


# ------------------------------------------------------------------ patterns
@pytest.mark.parametrize("pattern", sorted(reference.PATTERNS))
def test_known_patterns_match_reference(pattern):
    """Guns, gliders, still lifes and oscillators blitted at an offset, so the
    wrap-around blit path is covered as well as the rules."""
    rows = reference.PATTERNS[pattern]
    ph = len(rows)
    pw = max(len(r) for r in rows)
    sim = Simulation(64, 64)
    sim.blit(rows, 5, 7)
    ref = reference.Reference(64, 64)
    ref.blit(rows, 5, 7)
    assert_parity(sim, ref, f"pattern={pattern} after blit")
    for gen in range(4):
        sim.step(reference.CONWAY)
        ref.step(reference.CONWAY)
    assert_parity(sim, ref, f"pattern={pattern} gen={gen + 1}")


def test_glider_translates_one_cell_per_four_generations():
    """The textbook property of a glider, asserted against the compiled kernel
    alone. A still-life or a periodic soup would pass a parity test but fail
    this, which is exactly why it is here."""
    sim = Simulation(64, 64)
    sim.blit(reference.PATTERNS["glider"], 10, 10)
    start = sorted(
        (int(x), int(y)) for y, x in sim.live_cells()
    )
    sim.run(4, reference.CONWAY)
    after = sorted((int(x), int(y)) for y, x in sim.live_cells())
    assert len(after) == 5
    # Four generations displace a glider by exactly one row and one column,
    # preserving its shape. Compare modulo the torus so the test does not care
    # where on the grid it started.
    shifted = sorted(((x + 1) % 64, (y + 1) % 64) for x, y in start)
    assert after == shifted


def test_block_is_still_life_and_blinker_is_period_two():
    """Two structural invariants, independent of any reference implementation."""
    block = Simulation(32, 32)
    block.blit(reference.PATTERNS["block"], 4, 4)
    block.run(10, reference.CONWAY)
    assert block.population() == 4
    assert block.max_age() == 10

    blinker = Simulation(32, 32)
    blinker.blit(reference.PATTERNS["blinker"], 4, 4)
    first = blinker.grid.copy()
    blinker.step(reference.CONWAY)
    second = blinker.grid.copy()
    assert not np.array_equal(first, second), "a blinker must change shape"
    blinker.step(reference.CONWAY)
    assert np.array_equal(first, blinker.grid), "a blinker is period 2"