"""Regenerate the golden digests in `tests/test_golden.py`.

    python tests/record_goldens.py

Run this only when a change to the encoders or the kernels is intended, and read
the diff before committing it. The digests pin three things that must stay
byte-stable: the PNG encoder, the indexed PNG encoder, and the hand-rolled GIF
LZW stream.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from mojoapp_life_art import gif, palettes, png, reference  # noqa: E402
from mojoapp_life_art.simulation import Simulation  # noqa: E402


def digests() -> dict[str, str]:
    sim = Simulation(24, 20)
    sim.randomize(seed=1234, density=0.3)
    sim.run(6, reference.CONWAY)
    rgb, idx = sim.render("ember")
    pal_b = palettes.to_bytes(palettes.get("ember"))

    anim = Simulation(24, 24)
    anim.randomize(seed=555, density=0.3)
    _rgb, a_idx = anim.run_frames(12, 3, reference.CONWAY, palette="mono")
    mono_b = palettes.to_bytes(palettes.get("mono"))

    return {
        "GOLDEN_PNG_SHA256": hashlib.sha256(png.encode_rgb(rgb)).hexdigest(),
        "GOLDEN_INDEXED_PNG_SHA256": hashlib.sha256(png.encode_indexed(idx, pal_b)).hexdigest(),
        "GOLDEN_GIF_SHA256": hashlib.sha256(gif.encode(a_idx, mono_b, delay_cs=8)).hexdigest(),
    }


def main() -> int:
    target = ROOT / "tests" / "test_golden.py"
    text = target.read_text()
    for name, value in digests().items():
        import re

        new_text, n = re.subn(
            rf'^{name} = ".*"$', f'{name} = "{value}"', text, flags=re.M
        )
        if n != 1:
            print(f"error: could not find exactly one {name} to replace", file=sys.stderr)
            return 1
        text = new_text
    target.write_text(text)
    print("updated", target)
    for name, value in digests().items():
        print(f"  {name} = {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())