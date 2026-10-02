"""The SAME hot loop, in plain Python, timed against CPython via `mojosub`.

What this shows. The inner loop below is a transcription of
`Simulation.render`'s body -- the loop in `src/ported.mojo::_render_row` that
turns cell ages into palette indices. It is the same arithmetic, written the
way a person writes it: flat lists, an explicit index loop, no cleverness.

`mojosub` (https://github.com/lee101/mojosub) compiles such a function to native
code and calls it from Python. Wrapping a hot loop in `@jit` is therefore a
one-line change to an existing Python program, which is the point of the
showcase: the gap between "this loop is slow" and "this loop is native" is one
decorator, not a rewrite into a systems language.

What it does NOT show is that mojosub beats the compiled Mojo in `src/`. It is
not measured here, because that is `bench/bench.py`'s job and the two are
measured differently (per-function JIT vs a whole shared library). The number
printed is `mojosub vs CPython on this loop`, full stop.

    pixi run showcase

If `mojosub` is not importable this script skips with a message naming the fix.
It never prints a speedup it did not measure.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from mojoapp_life_art import palettes  # noqa: E402

# Sized so the CPython baseline takes long enough to time reliably and the
# whole script still finishes in seconds.
W = H = 256
REPEATS = 7
WARMUPS = 2


def render_indexed(ages, palette, pscale: int = 256):
    """Age -> palette index -> RGB, row-major. The transcription.

    Deliberately the same shape as `_render_row`: a scalar walk with a multiply,
    a shift and a clamp per cell. `jit` compiles the function as written; the
    point of the comparison is the same work, not more of it.
    """
    n = len(ages)
    rgb = bytearray(n * 3)
    idx = bytearray(n)
    for i in range(n):
        p = (ages[i] * pscale) >> 8
        if p > 255:
            p = 255
        r, g, b = palette[p]
        rgb[i * 3] = r
        rgb[i * 3 + 1] = g
        rgb[i * 3 + 2] = b
        idx[i] = p
    return bytes(rgb), bytes(idx)


def make_ages(n: int) -> bytearray:
    """A plausible age buffer: mostly dead, a few ancient lineages."""
    ages = bytearray(n)
    v = 20260915
    for i in range(n):
        v ^= (v << 13) & 0xFFFFFFFF
        v ^= v >> 17
        v ^= (v << 5) & 0xFFFFFFFF
        ages[i] = (v >> 7) & 0x3F if (v & 7) == 0 else 0
    return ages


def _median(fn, *args) -> float:
    for _ in range(WARMUPS):
        fn(*args)
    samples = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn(*args)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _load_mojosub():
    """Import the `mojosub` package, or None with a skip message already printed.

    Two guards, both necessary. `MOJOSUB_PATH` is honoured because the project
    ships as a repository rather than a wheel: point it at the clone and the
    import below finds it. And `spec.loader_from_cwd` is not enough -- this file
    is literally named `mojosub.py`, so running it as `python showcase/mojosub.py`
    puts its own directory on `sys.path` and `import mojosub` imports *this file*.
    A package has no `jit`, the import fails, and the skip message never prints
    because the exception is an ImportError from the wrong place. Deleting this
    script's own entry from the front of `sys.path` is what makes the guard
    mean something.
    """
    here = str(Path(__file__).resolve().parent)
    # This file is named `mojosub.py`, so its own directory on `sys.path`
    # shadows the package. Remove it, or `import mojosub` finds this file.
    sys.path = [p for p in sys.path if p not in (here, os.getcwd())]
    local = os.environ.get("MOJOSUB_PATH")
    if local:
        sys.path.insert(0, local)
    try:
        import mojosub
    except ImportError as e:
        print(
            "skip: the `mojosub` package is not importable.\n"
            "  It is not on PyPI -- it ships from https://github.com/lee101/mojosub.\n"
            "  Either clone it and point this script at the checkout:\n"
            "      export MOJOSUB_PATH=/path/to/mojosub\n"
            "  or install it into this environment:\n"
            "      pip install git+https://github.com/lee101/mojosub.git\n"
            f"  (import failed with: {e})\n"
            "  Nothing was compiled and no speedup is claimed."
        )
        return None
    if not hasattr(mojosub, "jit"):
        print(
            "skip: an importable `mojosub` has no `jit`, so it is not the SDK\n"
            "  this script expects. Nothing was compiled and no speedup is\n"
            "  claimed."
        )
        return None
    return mojosub


def main() -> int:
    mojosub = _load_mojosub()
    if mojosub is None:
        return 0

    from mojosub import jit  # noqa: PLC0415  (deferred: needs the env var above)

    ages = make_ages(W * H)
    pal = palettes.get("ember")

    compiled = jit(render_indexed)
    cpython_s = _median(render_indexed, ages, pal)
    native_s = _median(compiled, ages, pal)

    # The two must agree exactly, or the speedup below is meaningless.
    a = render_indexed(ages, pal)
    b = compiled(ages, pal)
    if a != b:
        print("error: the mojosub build does not match CPython output", file=sys.stderr)
        return 1

    speedup = cpython_s / native_s
    print(f"loop      age -> palette index -> RGB, {W}x{H} = {W * H} cells")
    print(f"source    showcase/mojosub.py::render_indexed")
    print(f"method    {WARMUPS} warmups, {REPEATS} repeats, median wall time")
    print()
    print("| implementation | median ms | vs CPython |")
    print("| --- | ---: | ---: |")
    print(f"| CPython (plain Python) | {cpython_s * 1e3:.2f} | 1.00x |")
    print(f"| mojosub `@jit` (native) | {native_s * 1e3:.2f} | {speedup:.2f}x |")
    print()
    print(f"speedup   {speedup:.2f}x over CPython on this loop")
    print("output    byte-identical to the CPython result (checked above)")
    print()
    print(
        "note      this is mojosub vs CPython, one function. The compiled Mojo\n"
        "          core in src/ is benchmarked separately in bench/bench.py."
    )

    out = ROOT / "showcase" / "last_mojosub.json"
    out.write_text(
        json.dumps(
            {
                "loop": "render_indexed",
                "cells": W * H,
                "cpython_ms": cpython_s * 1e3,
                "mojosub_ms": native_s * 1e3,
                "speedup": speedup,
                "outputs_match": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())