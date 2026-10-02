"""Generate the shareable demo artifact set into `demo/out/`.

Everything here is deterministic: the seeds are constants in this file, the
seeding is a counter-based xorshift64* in the kernel, and the encoders are
hand-rolled functions of the pixels. Re-running this script on the same repo
produces byte-identical files, which is what makes the outputs worth checking
into git and what lets a test assert against them.

    pixi run demo

Four artifacts, all under 2 MB:

    poster.png     256x256 still, 220 generations, cell age through `ember`
    life.gif       160x160 animated, 90 frames, `ice`
    card.txt       72x28 text card, readable in a terminal
    stats.json     the run behind them: rule, seed, populations, timings

The headline in `stats.json` is measured here and now; it is not a constant
copied from a README.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402

from mojoapp_life_art import gif, palettes, png, reference  # noqa: E402
from mojoapp_life_art.simulation import (  # noqa: E402
    Simulation,
    calibrate,
    par_min_cells,
    simd_width,
    workers,
)

OUT = ROOT / "demo" / "out"

# The demo parameters. Named, not inlined, so `stats.json` can record exactly
# what produced the files next to it.
#
# Density and generation count were chosen by looking at the output, not by
# guessing: below ~8% live pixels a Conway soup decays into isolated clusters
# and the age ramp -- the thing that makes this a poster rather than a diagram
# -- collapses into one dark band.
POSTER = {"width": 256, "height": 256, "gens": 220, "seed": 4242,
          "density": 0.36, "rule": "conway", "palette": "ember"}
ANIMATION = {"width": 160, "height": 160, "gens": 178, "stride": 2, "seed": 20260915,
             "density": 0.32, "rule": "conway", "palette": "ice"}
CARD = {"width": 72, "height": 28, "gens": 90, "seed": 404,
        "density": 0.40, "rule": "conway", "palette": "terminal"}
#: Soft ceiling for a shareable artifact. Anything above it is a bug in the
#: frame budget, not something to explain away.
MAX_BYTES = 2 * 1024 * 1024


def _ascii_card(sim: Simulation, idx: np.ndarray, spec: dict) -> str:
    ramp = " .:-=+*#%@"
    h, w = idx.shape
    lines = [
        "".join(ramp[min(len(ramp) - 1, int(idx[y, x]) * len(ramp) // 256)]
                for x in range(w))
        for y in range(h)
    ]
    rule = reference.RULES[spec["rule"]]
    head = (
        f"life-art  {spec['rule']} ({rule[0]})  {spec['width']}x{spec['height']}  "
        f"gens={spec['gens']}  seed={spec['seed']}  density={spec['density']}"
    )
    foot = (
        f"population {sim.population()}   max age {sim.max_age()}   "
        f"palette {spec['palette']}"
    )
    bar = "+" + "-" * w + "+"
    return "\n".join([head, bar, *(f"|{ln}|" for ln in lines), bar, foot, ""])


def _poster() -> tuple[Path, dict]:
    sim = Simulation(POSTER["width"], POSTER["height"])
    sim.randomize(seed=POSTER["seed"], density=POSTER["density"])
    t0 = time.perf_counter()
    sim.run(POSTER["gens"], reference.RULES[POSTER["rule"]][1:])
    evolve_s = time.perf_counter() - t0
    rgb, _ = sim.render(POSTER["palette"])
    out = OUT / "poster.png"
    nbytes = png.write(out, rgb)
    return out, {
        "artifact": out.name,
        "bytes": nbytes,
        "population": sim.population(),
        "max_age": sim.max_age(),
        "evolve_seconds": round(evolve_s, 6),
        "generations_per_second": round(POSTER["gens"] / evolve_s, 1),
    }


def _animation() -> tuple[Path, dict]:
    sim = Simulation(ANIMATION["width"], ANIMATION["height"])
    sim.randomize(seed=ANIMATION["seed"], density=ANIMATION["density"])
    t0 = time.perf_counter()
    rgb, idx = sim.run_frames(
        ANIMATION["gens"],
        ANIMATION["stride"],
        reference.RULES[ANIMATION["rule"]][1:],
        palette=ANIMATION["palette"],
    )
    evolve_s = time.perf_counter() - t0
    pal_b = palettes.to_bytes(palettes.get(ANIMATION["palette"]))
    out = OUT / "life.gif"
    nbytes = gif.write(out, idx, pal_b, delay_cs=6)
    return out, {
        "artifact": out.name,
        "bytes": nbytes,
        "frames": int(idx.shape[0]),
        "stride": ANIMATION["stride"],
        "population": sim.population(),
        "evolve_seconds": round(evolve_s, 6),
        "generations_per_second": round(ANIMATION["gens"] / evolve_s, 1),
    }


def _card() -> tuple[Path, dict]:
    sim = Simulation(CARD["width"], CARD["height"])
    sim.randomize(seed=CARD["seed"], density=CARD["density"])
    sim.run(CARD["gens"], reference.RULES[CARD["rule"]][1:])
    _rgb, idx = sim.render(CARD["palette"])
    out = OUT / "card.txt"
    text = _ascii_card(sim, idx, CARD)
    out.write_text(text)
    return out, {
        "artifact": out.name,
        "bytes": out.stat().st_size,
        "population": sim.population(),
        "max_age": sim.max_age(),
    }


def _stats(measured: list[dict]) -> Path:
    # The measured threshold, not the kernel's constant: which of these runs
    # actually took the parallel path is decided by `calibrate()`, and a
    # stats file that claimed the other would be wrong.
    cal = calibrate()
    record = {
        "poster": {k: POSTER[k] for k in ("width", "height", "gens", "seed",
                                          "density", "rule", "palette")},
        "animation": {k: ANIMATION[k] for k in ("width", "height", "gens",
                                                "stride", "seed", "density",
                                                "rule", "palette")},
        "measured": measured,
        "kernel": {
            "simd_width": simd_width(),
            "default_workers": workers(),
            "kernel_floor_cells": par_min_cells(),
            "measured_break_even_cells": round(cal["threshold_cells"]),
            "parallel_at_probe": cal["parallel_wins_at_probe"],
        },
        "numpy": np.__version__,
    }
    out = OUT / "stats.json"
    with out.open("w") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    measured = []
    for make in (_poster, _animation, _card):
        out, info = make()
        if info["bytes"] > MAX_BYTES:
            print(
                f"error: {out.name} is {info['bytes'] / 1024:.0f} KiB, over the "
                f"{MAX_BYTES // 1024} KiB shareable limit",
                file=sys.stderr,
            )
            return 1
        measured.append(info)
        print(f"  {out.name:<14} {info['bytes'] / 1024:8.1f} KiB  {info}")

    out = _stats(measured)
    print(f"  {out.name:<14} {out.stat().st_size / 1024:8.1f} KiB")
    print(f"\nwrote {len(measured) + 1} artifacts to {OUT}")
    print(
        "note: the timing fields above are measured on this machine at this "
        "moment and differ between runs; everything else is byte-reproducible."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())