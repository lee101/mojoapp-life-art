"""Life Art — Conway and cellular-automata poster generator.

The compute-heavy core is compiled Mojo (`src/ported.mojo`, built to
`dist/libmojoapp-life-art.so` and called through ctypes). This package is the
front end: seeding, evolution, palette rendering, and PNG/GIF encoding.

Quick start:

    import numpy as np
    from mojoapp_life_art import Simulation

    sim = Simulation(256, 256)
    sim.randomize(seed=7, density=0.3)
    sim.run(200)                    # 200 generations, one FFI call
    rgb, idx = sim.render("ember")  # indexed by cell age
    rgb.tofile("poster.raw")

`reference.Reference` is the same automaton in plain Python. It is the
definition of correct (`tests/test_parity.py` asserts they agree cell for
cell) and the baseline `bench/bench.py` measures against.
"""

from __future__ import annotations

from . import gif, palettes, png, reference
from .reference import PATTERNS, RULES, Reference
from .simulation import (
    Simulation,
    par_min_cells,
    rule_masks,
    rule_names,
    rule_notation,
    simd_width,
    workers,
)

__version__ = "0.1.0"

__all__ = [
    "Simulation",
    "Reference",
    "PATTERNS",
    "RULES",
    "gif",
    "palettes",
    "png",
    "reference",
    "par_min_cells",
    "rule_masks",
    "rule_names",
    "rule_notation",
    "simd_width",
    "workers",
    "__version__",
]