"""Loads (and if necessary builds) the compiled Mojo library.

Mirrors the loader style used across the mojo-* ports: a signature table, a
stale-checked build, and a single cached `ctypes.CDLL`. Buffers cross as
`Int` addresses, so the only thing this module knows about Mojo is the ABI.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJOAPP_LIFE_ART_LIB") or os.path.join(
    ROOT, "dist", "libmojoapp-life-art.so"
)

I = ctypes.c_int64
D = ctypes.c_double

# name -> (argtypes, restype)
_SIGNATURES = {
    "la_workers": ([], I),
    "la_par_min_cells": ([], I),
    "la_simd_width": ([], I),
    "la_seed": ([I, I, I, I, I, D], None),
    "la_zero": ([I, I, I], None),
    "la_step": ([I] * 9, None),
    "la_run": ([I] * 10, I),
    "la_run_frames": ([I] * 15, I),
    "la_render": ([I] * 9, None),
    "la_population": ([I, I], I),
    "la_max_age": ([I, I], I),
    "la_blit": ([I] * 8, I),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJOAPP_LIFE_ART_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJOAPP_LIFE_ART_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/capi.mojo` into `dist/libmojoapp-life-art.so` if stale."""
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        "-I",
        SRC,
        os.path.join(SRC, "capi.mojo"),
        "-o",
        LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def addr(a) -> int:
    return a.ctypes.data


def main() -> int:
    """`python -m mojoapp_life_art._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())