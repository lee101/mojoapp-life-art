"""Run the pure-Python Life reference on https://mojojojo.cc through its SDK.

The point of this script is not speed. It is agreement: the same step loop that
`tests/test_parity.py` pins against the compiled Mojo kernels is shipped
verbatim into a remote run, and the remote population is asserted against the
local one. If the two grids ever diverge, the platform changed something
underneath us.

`mojojojo` is not on PyPI and there is no API key in this repo, so both are
optional: an unset key or a missing SDK is a clean skip (exit 0), never a
fabricated number. Every timing printed below was measured here or came back in
the `Result`.

    export MOJOJOJO_KEY=...     # from https://mojojojo.cc/keys
    python showcase/mojojojo.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from mojoapp_life_art.reference import CONWAY, Reference  # noqa: E402

# A 96x96 grid for 5 generations: small enough that the remote finishes in the
# time it takes to read this comment, big enough that the population is a real
# number and not a rounding artefact. This is a demonstration of the SDK, not a
# benchmark -- `bench/` measures performance and does so locally.
W = 96
H = 96
GENS = 5
SEED = 20260915
DENSITY = 0.3
TIMEOUT_MS = 60_000

REMOTE_SRC = '''\
"""The reference step loop, shipped as a standalone program.

Printed output is exactly one line, `RESULT <population> <seconds>`, so the
caller can parse it without guessing. No other stdout: a stray print here would
make the contract ambiguous.
"""

import time

W = {w}
H = {h}
GENS = {gens}
BIRTH = {birth}
SURVIVE = {survive}
GRID = {grid!r}


def step(g, ages):
    """One generation: 8 neighbours, toroidal wrap, B/S masks, ages cap at 255."""
    birth, survive = BIRTH, SURVIVE
    w, h = W, H
    out = [0] * (w * h)
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
            bit = survive if g[row + x] else birth
            if (bit >> n) & 1:
                out[row + x] = 1
                old = ages[row + x]
                nages[row + x] = 255 if old >= 254 else old + 1
    return out, nages


g = list(GRID)
ages = [0] * (W * H)

# Timed here, remotely: the clock has to live with the loop it measures, and the
# platform's own wall_ms includes queueing, transfer and interpreter startup.
t0 = time.perf_counter()
for _ in range(GENS):
    g, ages = step(g, ages)
elapsed = time.perf_counter() - t0

print("RESULT %d %.6f" % (sum(g), elapsed))
'''


def _load_sdk():
    """Return the `mojojojo` module, or None with a skip message already printed.

    Three guards, all necessary. `MOJOJOJO_PATH` is honoured because the SDK
    ships as a source checkout rather than a wheel. This file is named
    `mojojojo.py`, so running it directly puts its own directory on `sys.path`
    and `import mojojojo` imports *this file* -- which has no `run`, and the
    resulting AttributeError happens inside the try block below and reads like
    an SDK failure rather than a path mistake. And the exception types are
    checked with `getattr` for the same reason: they may not be there at all.
    """
    here = str(Path(__file__).resolve().parent)
    sys.path = [p for p in sys.path if p not in (here, os.getcwd())]
    local = os.environ.get("MOJOJOJO_PATH")
    if local:
        sys.path.insert(0, local)
    try:
        import mojojojo
    except ImportError as e:
        print(
            "skip: the `mojojojo` SDK is not importable.\n"
            "  It is not on PyPI -- it ships as a source release from the mojojojo\n"
            "  GitHub organisation and from the https://mojojojo.cc dashboard. Clone\n"
            "  it and either install it into this environment or point the env var at\n"
            "  the checkout:  export MOJOJOJO_PATH=/path/to/mojojojo-python\n"
            f"  (import failed with: {e})\n"
            "  Nothing was run and nothing was charged."
        )
        return None
    if not hasattr(mojojojo, "run"):
        print(
            "skip: an importable `mojojojo` has no `run`, so it is not the SDK\n"
            "  this script expects. Nothing was run and nothing was charged."
        )
        return None
    return mojojojo


def _build_payload() -> tuple[str, Reference]:
    """Seed the grid here, then bake it into the remote source as literals.

    Seeding locally is what makes the comparison meaningful: both sides start
    from the identical flat list, byte for byte.
    """
    ref = Reference(W, H)
    ref.randomize(seed=SEED, density=DENSITY)
    birth, survive = CONWAY
    code = REMOTE_SRC.format(
        w=W, h=H, gens=GENS, birth=birth, survive=survive, grid=ref.grid
    )
    return code, ref


def _parse_result(stdout: str) -> tuple[int, float] | None:
    """Pull `(population, seconds)` out of the single RESULT line, or None."""
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "RESULT":
            try:
                return int(parts[1]), float(parts[2])
            except ValueError:
                return None
    return None


def main() -> int:
    key = os.environ.get("MOJOJOJO_KEY")
    if not key:
        print(
            "skip: MOJOJOJO_KEY is not set, so there is nothing to authenticate with.\n"
            "  Get a key at https://mojojojo.cc/keys, then:\n"
            "      export MOJOJOJO_KEY=...\n"
            "  Nothing was run and nothing was charged."
        )
        return 0

    mojojojo = _load_sdk()
    if mojojojo is None:
        return 0

    code, ref = _build_payload()

    # `getattr` for the exception types: a module that has `run` but not
    # `AuthError` would otherwise turn a skip into a traceback.
    def _err(name: str) -> type[BaseException]:
        got = getattr(mojojojo, name, None)
        return got if isinstance(got, type) and issubclass(got, BaseException) else ()

    try:
        result = mojojojo.run(code, source="life-art-showcase", timeout_ms=TIMEOUT_MS)
    except _err("AuthError") as e:
        print(
            f"skip: mojojojo rejected the key: {e}\n"
            "  Check MOJOJOJO_KEY, or mint a fresh one at https://mojojojo.cc/keys.\n"
            "  Nothing was charged."
        )
        return 0
    except _err("InsufficientCredits") as e:
        print(
            f"skip: the account is out of credits: {e}\n"
            "  Top up at https://mojojojo.cc/keys. Nothing beyond the rejected\n"
            "  request was charged."
        )
        return 0
    except _err("MojojojoError") as e:
        print(f"error: mojojojo run failed: {e}", file=sys.stderr)
        return 1

    if not result.ok:
        print("error: the remote run did not succeed.", file=sys.stderr)
        if result.stderr.strip():
            print(result.stderr.rstrip(), file=sys.stderr)
        return 1

    parsed = _parse_result(result.stdout)
    if parsed is None:
        print("error: no RESULT line in the remote stdout.", file=sys.stderr)
        if result.stdout.strip():
            print(result.stdout.rstrip(), file=sys.stderr)
        if result.stderr.strip():
            print(result.stderr.rstrip(), file=sys.stderr)
        return 1
    remote_pop, remote_s = parsed

    # Same seed, same loop, same rule, timed here.
    t0 = time.perf_counter()
    local_pop = ref.run(GENS, CONWAY)
    local_ms = (time.perf_counter() - t0) * 1000.0

    if remote_pop != local_pop:
        print(
            f"error: population mismatch after {GENS} generations: "
            f"remote {remote_pop}, local {local_pop}",
            file=sys.stderr,
        )
        return 1

    print(f"| {W}x{H} x {GENS} gens, B3/S23, seed {SEED} | value |")
    print("| --- | --- |")
    print(f"| remote wall ms | {result.wall_ms:.1f} |")
    print(f"| remote billed ms | {result.charge.billed_ms:.1f} |")
    print(f"| local ms | {local_ms:.1f} |")
    print(
        f"| population agreement | yes, {remote_pop} live cells "
        f"(remote self-timed {remote_s:.6f}s) |"
    )
    print()
    print(f"ran on {result.ran_on}, charged ${result.charge.usd:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())