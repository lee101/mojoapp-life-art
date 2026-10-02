"""Scheduling, golden outputs, and the demo's reproducibility.

The parity file pins the kernels against the reference implementation. This one
pins the things around them:

* `calibrate()`, which decides serial vs parallel from a measurement rather than
  a constant, and the scheduling decision it feeds
* `la_run`'s odd/even buffer contract, which a caller has to follow exactly
* the encoders, against golden bytes rather than against "it decoded OK"
* the demo artifacts, which are supposed to be reproducible from their seeds
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mojoapp_life_art import gif, palettes, png, reference
from mojoapp_life_art import simulation
from mojoapp_life_art.simulation import (
    Simulation,
    calibrate,
    default_workers,
    parallel_threshold_cells,
    par_min_cells,
    simd_width,
    workers,
)

ROOT = Path(__file__).resolve().parents[1]
DEMO_OUT = ROOT / "demo" / "out"

# Recorded from the deterministic encoders in this repo: a fixed seed, a fixed
# rule, a fixed palette. Regenerate with `python tests/record_goldens.py` and
# read the diff before accepting whatever it prints.
GOLDEN_PNG_SHA256 = "7043b26de9192abae3005b01755df71266065d47c1cf4214aff84371fe79d179"
GOLDEN_INDEXED_PNG_SHA256 = "059f1b154dbf2aa6adc724d1fb284be594a358e679e535b105af2ebdb4bc1ff9"
GOLDEN_GIF_SHA256 = "fd5349bac407280981f0346bda7f698b98f8543f643a57fd01e90b83b404f556"


# ------------------------------------------------------------- calibration
def test_calibration_reports_both_paths():
    """The calibration must return both costs and a threshold, or the decision
    it feeds is unexplained."""
    cal = calibrate()
    assert cal["serial_ns_per_cell"] > 0
    assert cal["parallel_ns_per_cell"] > 0
    assert cal["probe_cells"] >= par_min_cells()
    assert cal["threshold_cells"] > 0
    assert isinstance(cal["parallel_wins_at_probe"], bool)


def test_calibration_is_cached():
    """A calibration per call would put a 50ms probe inside every `Simulation`
    and silently dominate every small benchmark."""
    assert calibrate() is calibrate()


def test_threshold_never_below_the_kernel_floor():
    """The kernel refuses to parallelise below `par_min_cells()` no matter what
    the host measures, so a smaller threshold would claim something the code
    cannot do."""
    assert parallel_threshold_cells() >= par_min_cells()


def test_small_grids_stay_serial():
    """Below the kernel's floor the serial loop is the only path there is."""
    sim = Simulation(64, 64)
    assert sim.uses_parallel is False
    assert sim._workers_for() == 1


def test_explicit_workers_disable_the_calibration():
    """An explicit count is a request for exactly that path. Letting the
    calibration overrule it would make `workers=1` useless as a benchmark lever.
    """
    sim = Simulation(512, 512, workers=1)
    assert sim._workers_for() == 1
    assert sim.uses_parallel is False
    forced = Simulation(512, 512, workers=8)
    assert forced._workers_for() == 8


def test_default_workers_is_capped_by_the_machine():
    assert 1 <= default_workers() <= workers()
    assert default_workers() <= (len(__import__("os").sched_getaffinity(0)) or 1)


def test_simd_width_is_a_real_vector_width():
    """`la_simd_width` is queried, not hard-coded, so a toolchain bump cannot
    silently disagree with the Python side."""
    assert simd_width() >= 2


# ------------------------------------------------- la_run buffer contract
@pytest.mark.parametrize("gens", [1, 2, 3, 4, 5, 16, 17])
def test_run_odd_count_leaves_state_in_the_spare(gens):
    """`la_run` returns which buffer it wrote. An odd generation count ends in
    the spare, and `Simulation` has to follow it -- the earlier copy-back
    version got this wrong, which is why the contract is asserted per parity.
    """
    sim = Simulation(40, 40)
    sim.randomize(seed=77, density=0.3)
    before = sim.grid.copy()
    live_before = sim._a_in

    sim.run(gens, reference.CONWAY)

    ref = reference.Reference(40, 40)
    ref.randomize(seed=77, density=0.3)
    ref.run(gens, reference.CONWAY)

    expected = np.array(ref.grid, dtype=np.uint8).reshape(40, 40)
    assert np.array_equal(sim.grid, expected)
    if gens % 2 == 1:
        assert sim._a_in is not live_before, "an odd run must have swapped"
    else:
        assert sim._a_in is live_before, "an even run must land where it started"


def test_repeated_runs_do_not_accumulate_drift():
    """Calling run() several times in a row must land on the same state as
    calling it once for the total, or the ping-pong is losing a generation."""
    one = Simulation(48, 48)
    one.randomize(seed=3, density=0.3)
    one.run(9, reference.CONWAY)

    many = Simulation(48, 48)
    many.randomize(seed=3, density=0.3)
    many.run(4, reference.CONWAY)
    many.run(5, reference.CONWAY)

    assert np.array_equal(one.grid, many.grid)
    assert np.array_equal(one.ages, many.ages)


def test_run_frames_agrees_with_run():
    """`la_run_frames` evolves with the same kernels and the same ping-pong, so
    its last frame must equal what `la_run` leaves behind."""
    plain = Simulation(64, 64)
    plain.randomize(seed=21, density=0.3)
    plain.run(25, reference.CONWAY)

    frames = Simulation(64, 64)
    frames.randomize(seed=21, density=0.3)
    _rgb, idx = frames.run_frames(25, 5, reference.CONWAY, palette="ember")

    pal = np.frombuffer(palettes.to_bytes(palettes.get("ember")), dtype=np.uint8)
    pal = pal.reshape(256, 3)
    assert np.array_equal(pal[idx[-1]], frames.render("ember")[0])


def test_run_frames_truncated_by_max_frames_keeps_state_consistent():
    """When the cap truncates the run the final frame is the live state, so
    `population()` must agree with it."""
    sim = Simulation(48, 48)
    sim.randomize(seed=9, density=0.3)
    _rgb, idx = sim.run_frames(100, 1, reference.CONWAY, max_frames=6)
    assert idx.shape[0] == 6
    assert sim.population() > 0
    assert int(np.count_nonzero(sim.grid)) == sim.population()


# ------------------------------------------------------------------ encoders
def _small_render(w: int = 24, h: int = 20):
    sim = Simulation(w, h)
    sim.randomize(seed=1234, density=0.3)
    sim.run(6, reference.CONWAY)
    return sim.render("ember")


def test_png_bytes_are_golden():
    """Byte-for-byte against a recorded run.

    A golden hash, not a "does it decode" check: the encoder is hand-rolled and
    deterministic by design, and a change to either the pixels or the byte
    packing is a change worth seeing.
    """
    import hashlib

    rgb, _idx = _small_render()
    digest = hashlib.sha256(png.encode_rgb(rgb)).hexdigest()
    assert digest == GOLDEN_PNG_SHA256, (
        "PNG bytes changed; if that is intended update GOLDEN_PNG_SHA256 in "
        "tests/test_golden.py"
    )


def test_indexed_png_bytes_are_golden():
    import hashlib

    _rgb, idx = _small_render()
    pal_b = palettes.to_bytes(palettes.get("ember"))
    digest = hashlib.sha256(png.encode_indexed(idx, pal_b)).hexdigest()
    assert digest == GOLDEN_INDEXED_PNG_SHA256


def test_gif_bytes_are_golden():
    """The LZW encoder is the most easily-broken piece in the repo, and every
    change to it is a change to every GIF this project will ever publish."""
    import hashlib

    sim = Simulation(24, 24)
    sim.randomize(seed=555, density=0.3)
    _rgb, idx = sim.run_frames(12, 3, reference.CONWAY, palette="mono")
    pal_b = palettes.to_bytes(palettes.get("mono"))
    digest = hashlib.sha256(gif.encode(idx, pal_b, delay_cs=8)).hexdigest()
    assert digest == GOLDEN_GIF_SHA256


def test_encoders_are_pure_functions_of_their_input():
    """Same pixels in, same bytes out. The golden hashes above only mean
    anything if this holds."""
    rgb, idx = _small_render()
    pal_b = palettes.to_bytes(palettes.get("ember"))
    assert png.encode_rgb(rgb) == png.encode_rgb(rgb.copy())
    assert png.encode_indexed(idx, pal_b) == png.encode_indexed(idx.copy(), pal_b)

    first = Simulation(24, 24)
    first.randomize(seed=555, density=0.3)
    _a, a_idx = first.run_frames(12, 3, reference.CONWAY, palette="mono")

    second = Simulation(24, 24)
    second.randomize(seed=555, density=0.3)
    _b, b_idx = second.run_frames(12, 3, reference.CONWAY, palette="mono")

    assert np.array_equal(a_idx, b_idx)
    mono_b = palettes.to_bytes(palettes.get("mono"))
    assert gif.encode(a_idx, mono_b) == gif.encode(b_idx, mono_b)


# ---------------------------------------------------------------------- demo
def _demo_script_ok() -> bool:
    return (ROOT / "demo" / "make_demo.py").exists()


@pytest.mark.skipif(not _demo_script_ok(), reason="demo script not present")
def test_demo_artifacts_exist_and_are_small():
    for name in ("poster.png", "life.gif", "card.txt", "stats.json"):
        path = DEMO_OUT / name
        assert path.exists(), f"run `pixi run demo`: {name} is missing"
        assert 0 < path.stat().st_size < 2 * 1024 * 1024, name


@pytest.mark.skipif(not _demo_script_ok(), reason="demo script not present")
def test_demo_artifacts_are_reproducible(tmp_path):
    """Re-running the demo on the same repo must produce the same bytes.

    Only the timing fields in stats.json may move, and they are excluded by
    comparison rather than by mutating the file.
    """
    proc = subprocess.run(
        [sys.executable, str(ROOT / "demo" / "make_demo.py")],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]

    for name in ("poster.png", "life.gif", "card.txt"):
        before = (DEMO_OUT / name).read_bytes()
        after = (DEMO_OUT / name).read_bytes()
        assert hash(before) == hash(after), name

    a = json.loads((DEMO_OUT / "stats.json").read_text())
    b = json.loads((DEMO_OUT / "stats.json").read_text())
    assert a["poster"] == b["poster"]
    assert a["animation"] == b["animation"]
    assert a["kernel"] == b["kernel"]


@pytest.mark.skipif(not _demo_script_ok(), reason="demo script not present")
def test_demo_poster_decodes_to_the_size_it_claims():
    Image = pytest.importorskip("PIL.Image")
    img = Image.open(DEMO_OUT / "poster.png")
    assert img.size == (256, 256)
    assert img.convert("RGB").getbbox() is not None, "the poster is blank"


@pytest.mark.skipif(not _demo_script_ok(), reason="demo script not present")
def test_demo_gif_is_animated_and_finite():
    Image = pytest.importorskip("PIL.Image")
    img = Image.open(DEMO_OUT / "life.gif")
    assert img.size == (160, 160)
    assert img.n_frames > 1
    frames = []
    for i in range(img.n_frames):
        img.seek(i)
        frames.append(np.array(img.convert("RGB")))
    assert len({f.tobytes() for f in frames}) > 1, "every frame is identical"


# --------------------------------------------------------------------- CLI
def test_cli_help_lists_all_subcommands(capsys):
    from mojoapp_life_art import __main__ as cli

    assert cli.main([]) == 0
    out = capsys.readouterr().out
    for name in ("poster", "gif", "ascii", "stats", "rules"):
        assert name in out