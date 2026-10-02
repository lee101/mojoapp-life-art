"""Rendering, encoding, palette and CLI behaviour.

These guard the parts a user actually touches: that a poster has the right
pixels, that a GIF round-trips through a real decoder, that the CLI writes the
file it says it wrote, and that the arguments are validated rather than
silently producing a 3x3 grid.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mojoapp_life_art import __main__ as cli
from mojoapp_life_art import gif, palettes, png, reference
from mojoapp_life_art.simulation import Simulation

ROOT = Path(__file__).resolve().parents[1]


def _gif_file(data: bytes) -> Path:
    p = Path(tempfile.mkdtemp()) / "t.gif"
    p.write_bytes(data)
    return p


# ------------------------------------------------------------------ palettes
def test_palette_ramps_hit_their_endpoints():
    """A ramp whose first and last entries drift from its stops would make the
    young end of the poster a different colour than the palette advertises."""
    for name in palettes.names():
        pal = palettes.get(name)
        assert len(pal) == 256, name
        if name != "spectrum":
            assert pal[0] != pal[-1], name
    # `spectrum` wraps a full hue turn, so it is legitimately cyclic.
    cycle = palettes.get("spectrum")
    assert cycle[0] == cycle[-1]


def test_multi_stop_anchors_its_endpoints():
    """`multi_stop` must land exactly on its first and last stop colours."""
    pal = palettes.multi_stop([(0.0, "#102030"), (1.0, "#405060")])
    assert pal[0] == (0x10, 0x20, 0x30)
    assert pal[-1] == (0x40, 0x50, 0x60)


def test_palette_entries_are_bytes():
    for name in palettes.names():
        for r, g, b in palettes.get(name):
            assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255


def test_unknown_palette_names_the_alternatives():
    with pytest.raises(ValueError) as e:
        palettes.get("chartreuse")
    assert "ember" in str(e.value)


def test_to_bytes_flattens_to_768():
    assert len(palettes.to_bytes(palettes.get("ember"))) == 768


# ------------------------------------------------------------------ rendering
def test_render_shapes_and_dtypes():
    sim = Simulation(40, 30)
    sim.randomize(seed=1, density=0.3)
    sim.run(10)
    rgb, idx = sim.render("ember")
    assert rgb.shape == (30, 40, 3)
    assert idx.shape == (30, 40)
    assert rgb.dtype == np.uint8 and idx.dtype == np.uint8


def test_render_is_indexed_by_age():
    """The whole point of the age buffer: colour must track age, so a cell that
    has been alive 5 generations and one alive 50 cannot share a colour, and
    the index must equal the age scaled into the palette."""
    sim = Simulation(32, 32)
    sim.blit(reference.PATTERNS["block"], 4, 4)
    sim.run(20, reference.CONWAY)
    rgb, idx = sim.render("ember")
    ys, xs = np.nonzero(idx)
    # Only the block is alive; every live cell has age 20.
    assert len(ys) == 4
    assert set(idx[idx > 0].tolist()) == {20}
    pal = palettes.get("ember")
    for y, x in zip(ys, xs):
        assert tuple(int(v) for v in rgb[y, x]) == pal[20]


def test_dead_cells_take_palette_zero():
    sim = Simulation(16, 16)
    sim.clear()
    rgb, idx = sim.render("ink")
    assert set(np.unique(idx).tolist()) == {0}
    pal = palettes.get("ink")
    assert tuple(int(v) for v in rgb[0, 0]) == pal[0]


def test_max_age_saturates():
    """`old + 1` on a UInt8 would wrap 255 to 0 and make an ancient cell look
    newborn; `_age` clamps at 254 for that reason."""
    sim = Simulation(24, 24)
    sim.blit(reference.PATTERNS["block"], 2, 2)
    sim.run(300, reference.CONWAY)
    assert sim.max_age() == 255
    assert sim.population() == 4


def test_render_matches_reference_renderer():
    sim = Simulation(48, 48)
    sim.randomize(seed=13, density=0.3)
    sim.run(15)
    ref = reference.Reference(48, 48)
    ref.randomize(seed=13, density=0.3)
    ref.run(15)
    rgb, idx = sim.render("dusk")
    pal = palettes.get("dusk")
    ref_rgb, ref_idx = ref.render(pal)
    assert np.array_equal(idx, np.frombuffer(ref_idx, dtype=np.uint8).reshape(48, 48))
    assert np.array_equal(
        rgb, np.frombuffer(ref_rgb, dtype=np.uint8).reshape(48, 48, 3)
    )


# ------------------------------------------------------------------------ PNG
def test_png_is_a_valid_png_of_the_right_size():
    sim = Simulation(40, 24)
    sim.randomize(seed=2, density=0.3)
    sim.run(8)
    rgb, _idx = sim.render("ember")
    data = png.encode_rgb(rgb)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data.endswith(b"IEND\xaeB`\x82")
    # IHDR: width, height, bit depth 8, colour type 2 (truecolour RGB).
    assert data[16:24] == struct.pack(">II", 40, 24)
    assert data[24:29] == bytes([8, 2, 0, 0, 0])


def test_png_writes_the_bytes_it_encoded(tmp_path):
    sim = Simulation(20, 20)
    sim.blit(reference.PATTERNS["glider"], 3, 3)
    rgb, _ = sim.render("terminal")
    out = tmp_path / "x.png"
    n = png.write(out, rgb)
    assert out.stat().st_size == n
    assert out.read_bytes() == png.encode_rgb(rgb)


def test_png_is_deterministic():
    """Golden-image tests compare bytes, so the encoder must be a pure function
    of the pixels."""
    sim = Simulation(32, 32)
    sim.randomize(seed=4, density=0.3)
    rgb, _ = sim.render("ember")
    assert png.encode_rgb(rgb) == png.encode_rgb(rgb)


def test_indexed_png_matches_rgb_pixel_for_pixel():
    """Indexed PNG is colour-type 3 and decodes to exactly the same pixels as
    the truecolour one.

    It is deliberately NOT asserted to be smaller. An indexed PNG always
    carries a 768-byte palette chunk, so it loses to truecolour on a flat image,
    and a poster that spreads across the whole age ramp gives zlib more to work
    with in truecolour. Measured on this repo's own output it is sometimes
    smaller and sometimes larger. The animation path is the GIF, where a shared
    global palette is amortised across every frame; that is where the byte
    saving actually is."""
    sim = Simulation(96, 96)
    sim.randomize(seed=6, density=0.3)
    sim.run(20)
    rgb, idx = sim.render("ember")
    pal_b = palettes.to_bytes(palettes.get("ember"))
    indexed = png.encode_indexed(idx, pal_b)
    # Same pixels either way.
    pal = np.frombuffer(pal_b, dtype=np.uint8).reshape(256, 3)
    assert np.array_equal(pal[idx], rgb)
    assert indexed[24] == 8, "bit depth 8"
    assert indexed[25] == 3, "colour type 3 is indexed"
    assert b"PLTE" in indexed
    assert len(pal_b) == 768


def test_png_rejects_bad_input():
    with pytest.raises(ValueError):
        png.encode_rgb(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        png.encode_indexed(np.zeros((4, 4), dtype=np.uint8), b"\x00" * 10)


# ------------------------------------------------------------------------ GIF
def test_gif_round_trips_through_a_real_decoder():
    """The LZW encoder is hand-rolled, so it is checked against a decoder that
    shares none of its code."""
    Image = pytest.importorskip("PIL.Image")
    sim = Simulation(48, 48)
    sim.randomize(seed=8, density=0.3)
    rgb, idx = sim.run_frames(24, 6, reference.CONWAY, palette="ember")
    pal_b = palettes.to_bytes(palettes.get("ember"))
    data = gif.encode(idx, pal_b, delay_cs=8)
    img = Image.open(_gif_file(data))
    assert img.size == (48, 48)
    assert img.n_frames == idx.shape[0]
    frames = []
    for i in range(img.n_frames):
        img.seek(i)
        frames.append(np.array(img.convert("RGB")))
    # Frames must actually differ, or the GIF is a still wearing a costume.
    assert len({f.tobytes() for f in frames}) > 1
    # The first and last decoded frames must equal what the kernel rendered.
    assert np.array_equal(frames[0], rgb[0])
    assert np.array_equal(frames[-1], rgb[-1])


def test_gif_header_and_terminator():
    sim = Simulation(16, 16)
    sim.randomize(seed=9, density=0.3)
    _rgb, idx = sim.run_frames(4, 2, reference.CONWAY, palette="mono")
    pal_b = palettes.to_bytes(palettes.get("mono"))
    data = gif.encode(idx, pal_b)
    assert data[:6] == b"GIF89a"
    assert data[-1:] == b"\x3b"
    assert b"NETSCAPE2.0" in data, "an animated gif needs the looping extension"


def test_gif_write_returns_size(tmp_path):
    sim = Simulation(16, 16)
    sim.randomize(seed=9, density=0.3)
    _rgb, idx = sim.run_frames(4, 2, reference.CONWAY, palette="mono")
    pal_b = palettes.to_bytes(palettes.get("mono"))
    out = tmp_path / "a.gif"
    n = gif.write(out, idx, pal_b)
    assert out.stat().st_size == n


def test_gif_needs_frames_and_a_palette():
    with pytest.raises(ValueError):
        gif.encode(np.zeros((0, 4, 4), dtype=np.uint8), b"\x00" * 768)
    with pytest.raises(ValueError):
        gif.encode(np.zeros((1, 4, 4), dtype=np.uint8), b"\x00" * 10)


def test_run_frames_frame_count_and_endpoints():
    """Frame 0 is the starting state and the last frame is generation `gens`,
    even when `gens` is not a multiple of the stride."""
    sim = Simulation(32, 32)
    sim.randomize(seed=10, density=0.3)
    _rgb, idx = sim.run_frames(10, 4, reference.CONWAY, palette="ember")
    # gens=10, stride=4 -> frames at 0, 4 and 8, plus the forced final at 10.
    assert idx.shape[0] == 4
    assert not np.array_equal(idx[0], idx[-1])


def test_run_frames_respects_max_frames():
    sim = Simulation(32, 32)
    sim.randomize(seed=12, density=0.3)
    _rgb, idx = sim.run_frames(100, 1, reference.CONWAY, max_frames=5)
    assert idx.shape[0] == 5


# ------------------------------------------------------------------------ CLI
def test_cli_help_exits_zero():
    proc = subprocess.run(
        [sys.executable, "-m", "mojoapp_life_art", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "python")},
    )
    assert proc.returncode == 0
    assert "poster" in proc.stdout and "gif" in proc.stdout


def test_cli_no_command_prints_help(capsys):
    assert cli.main([]) == 0
    assert "usage" in capsys.readouterr().out


def test_cli_version(capsys):
    assert cli.main(["--version"]) == 0
    assert "mojoapp-life-art" in capsys.readouterr().out


def test_cli_rules_lists_every_rule(capsys):
    assert cli.main(["rules"]) == 0
    out = capsys.readouterr().out
    for name in reference.RULES:
        assert name in out
    for name in palettes.names():
        assert name in out


def test_cli_poster_writes_a_png(tmp_path, capsys):
    out = tmp_path / "poster.png"
    assert cli.main(
        [
            "poster", "--seed", "7", "--gens", "30",
            "--width", "48", "--height", "48", "-o", str(out),
        ]
    ) == 0
    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert "wrote" in capsys.readouterr().out


def test_cli_gif_writes_a_gif(tmp_path, capsys):
    out = tmp_path / "run.gif"
    assert cli.main(
        [
            "gif", "--seed", "7", "--gens", "20", "--stride", "4",
            "--width", "32", "--height", "32", "-o", str(out),
        ]
    ) == 0
    data = out.read_bytes()
    assert data[:6] == b"GIF89a"
    assert "frames" in capsys.readouterr().out


def test_cli_ascii_prints_a_card(capsys):
    assert cli.main(
        ["ascii", "--seed", "3", "--gens", "20", "--width", "24", "--height", "8"]
    ) == 0
    out = capsys.readouterr().out
    assert out.count("\n") >= 8
    assert "population" in out


def test_cli_ascii_writes_a_file(tmp_path, capsys):
    out = tmp_path / "card.txt"
    assert cli.main(
        ["ascii", "--seed", "3", "--gens", "10", "--width", "20",
         "--height", "6", "-o", str(out)]
    ) == 0
    text = out.read_text()
    assert "life-art" in text and "population" in text
    assert "wrote" in capsys.readouterr().out


def test_cli_stats_emits_json(tmp_path, capsys):
    out = tmp_path / "stats.json"
    assert cli.main(
        ["stats", "--seed", "5", "--gens", "24", "--width", "32",
         "--height", "32", "-o", str(out)]
    ) == 0
    printed = capsys.readouterr().out
    record = json.loads(printed[: printed.rindex("}") + 1])
    assert record["rule"] == "conway"
    assert record["final_population"] > 0
    assert record["generations_per_second"] > 0
    assert record["kernel"]["workers"] > 0
    assert len(record["population_curve"]) > 1
    assert json.loads(out.read_text()) == record


def test_cli_rejects_bad_arguments():
    for argv in (
        ["poster", "--width", "2"],
        ["poster", "--density", "1.5"],
        ["poster", "--gens", "-1"],
    ):
        with pytest.raises(SystemExit):
            cli.main(argv)


def test_cli_unknown_rule_exits_two(capsys):
    assert cli.main(["poster", "--rule", "nonsense"]) == 2
    assert "unknown rule" in capsys.readouterr().err


def test_cli_unknown_palette_exits_two(capsys):
    assert cli.main(["poster", "--palette", "chartreuse"]) == 2
    assert "unknown palette" in capsys.readouterr().err


# ------------------------------------------------------------------ geometry
def test_simulation_rejects_tiny_grids():
    for w, h in ((2, 5), (5, 2), (0, 0)):
        with pytest.raises(ValueError):
            Simulation(w, h)


def test_copy_is_independent():
    sim = Simulation(24, 24)
    sim.randomize(seed=1, density=0.3)
    clone = sim.copy()
    sim.run(5)
    assert not np.array_equal(sim.grid, clone.grid)
    assert np.array_equal(clone.ages, np.zeros((24, 24), dtype=np.uint8))


def test_set_grid_accepts_nested_and_flat():
    sim = Simulation(4, 3)
    sim.set_grid([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]])
    assert sim.population() == 6
    sim2 = Simulation(4, 3)
    sim2.set_grid([1] * 12)
    assert sim2.population() == 12


def test_live_cells_are_xy():
    sim = Simulation(8, 8)
    # The glider pattern is [".1.", "..1", "111"], blitted at (1, 2).
    sim.blit(reference.PATTERNS["glider"], 1, 2)
    assert set(sim.live_cells()) == {(2, 2), (3, 3), (1, 4), (2, 4), (3, 4)}


def test_clear_empties_both_buffers():
    sim = Simulation(16, 16)
    sim.randomize(seed=1, density=0.9)
    sim.clear()
    assert sim.population() == 0
    sim.run(3)
    assert sim.population() == 0

