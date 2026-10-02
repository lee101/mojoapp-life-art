"""`python -m mojoapp_life_art` — the command line.

Five subcommands, each of which writes one artifact:

    poster   a still PNG of the final generation, coloured by cell age
    gif      an animated GIF sampled along the run
    ascii    a text card, for terminals and READMEs
    stats    populations and timings, as JSON
    rules    list the built-in rules and palettes

Every one of them takes `--seed` and is reproducible from it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from . import gif, palettes, png, reference
from .simulation import (
    Simulation,
    check_rule,
    par_min_cells,
    rule_notation,
    simd_width,
    workers,
)

ASCII_RAMP = " .:-=+*#%@"


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--width", type=int, default=256, help="grid width (default 256)")
    p.add_argument("--height", type=int, default=256, help="grid height (default 256)")
    p.add_argument("--gens", type=int, default=200, help="generations (default 200)")
    p.add_argument("--seed", type=int, default=1, help="RNG seed (default 1)")
    p.add_argument("--density", type=float, default=0.3, help="initial density (default 0.3)")
    p.add_argument("--rule", default="conway", help="rule name (default conway)")
    p.add_argument("--palette", default=palettes.DEFAULT_PALETTE,
                   help=f"palette name, one of {', '.join(palettes.names())}")
    p.add_argument("-o", "--out", help="output path")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mojoapp_life_art",
        description="Conway and cellular-automata poster generator, "
                    "compiled Mojo core.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m mojoapp_life_art poster --seed 7 --gens 400 -o poster.png\n"
            "  python -m mojoapp_life_art gif --width 192 --height 192 "
            "--gens 240 --stride 4 -o life.gif\n"
            "  python -m mojoapp_life_art ascii --width 64 --height 32 --gens 60\n"
        ),
    )
    p.add_argument("--version", action="store_true", help="print version and exit")
    sub = p.add_subparsers(dest="command")

    q = sub.add_parser("poster", help="write a PNG of the final generation")
    _add_common(q)

    q = sub.add_parser("gif", help="write an animated GIF of the whole run")
    _add_common(q)
    q.add_argument("--stride", type=int, default=4,
                   help="generations between frames (default 4)")
    q.add_argument("--delay", type=int, default=8,
                   help="frame delay in hundredths of a second (default 8)")

    q = sub.add_parser("ascii", help="print a text card of the final generation")
    _add_common(q)

    q = sub.add_parser("stats", help="print a JSON record of the run")
    _add_common(q)

    sub.add_parser("rules", help="list the built-in rules and palettes")
    return p


# ------------------------------------------------------------------- commands
def _prepare(args) -> Simulation:
    if args.width < 3 or args.height < 3:
        raise SystemExit("grid must be at least 3x3")
    if not 0.0 <= args.density <= 1.0:
        raise SystemExit("--density must be between 0 and 1")
    if args.gens < 0:
        raise SystemExit("--gens must not be negative")
    # Resolved here so a typo is a message on stderr, not a KeyError traceback.
    rule = check_rule(args.rule)
    palettes.get(args.palette)
    sim = Simulation(args.width, args.height)
    sim.randomize(seed=args.seed, density=args.density)
    return sim


def cmd_poster(args) -> int:
    sim = _prepare(args)
    rule = check_rule(args.rule)
    t0 = time.perf_counter()
    sim.run(args.gens, rule)
    evolve_s = time.perf_counter() - t0
    rgb, _ = sim.render(args.palette)
    out = args.out or "life-art.png"
    nbytes = png.write(out, rgb)
    _report(out, nbytes, sim, evolve_s)
    return 0


def cmd_gif(args) -> int:
    sim = _prepare(args)
    rule = check_rule(args.rule)
    stride = max(1, args.stride)
    max_frames = 256
    t0 = time.perf_counter()
    _rgb, idx = sim.run_frames(
        args.gens, stride, rule, palette=args.palette, max_frames=max_frames
    )
    evolve_s = time.perf_counter() - t0
    pal_b = palettes.to_bytes(palettes.get(args.palette))
    out = args.out or "life-art.gif"
    nbytes = gif.write(out, idx, pal_b, delay_cs=args.delay)
    _report(out, nbytes, sim, evolve_s, frames=idx.shape[0])
    return 0


def cmd_ascii(args) -> int:
    sim = _prepare(args)
    sim.run(args.gens, check_rule(args.rule))
    _rgb, idx = sim.render(args.palette)
    h, w = idx.shape
    lines = []
    for y in range(h):
        row = "".join(
            ASCII_RAMP[min(len(ASCII_RAMP) - 1, int(idx[y, x]) * len(ASCII_RAMP) // 256)]
            for x in range(w)
        )
        lines.append(row)
    header = (
        f"life-art  rule={args.rule} ({rule_notation(args.rule)})  "
        f"{args.width}x{args.height}  gens={args.gens}  seed={args.seed}  "
        f"palette={args.palette}"
    )
    footer = f"population {sim.population()}  max age {sim.max_age()}"
    print(header)
    print("+" + "-" * w + "+")
    for line in lines:
        print("|" + line + "|")
    print("+" + "-" * w + "+")
    print(footer)
    if args.out:
        with open(args.out, "w") as f:
            f.write(header + "\n")
            f.write("+" + "-" * w + "+\n")
            for line in lines:
                f.write("|" + line + "|\n")
            f.write("+" + "-" * w + "+\n")
            f.write(footer + "\n")
        print(f"wrote {args.out}")
    return 0


def cmd_stats(args) -> int:
    sim = _prepare(args)
    rule = check_rule(args.rule)
    # Sample the population curve at `probe` points. Probing costs an extra
    # population pass each time, so it stays coarse on a big run.
    probe = 16 if args.gens >= 16 else args.gens + 1
    marks = sorted({int(round(args.gens * i / probe)) for i in range(probe + 1)})
    curve = []
    t0 = time.perf_counter()
    done = 0
    for target in marks:
        if target > done:
            sim.run(target - done, rule)
            done = target
        curve.append({"gen": done, "population": sim.population()})
    evolve_s = time.perf_counter() - t0
    record = {
        "width": args.width,
        "height": args.height,
        "gens": args.gens,
        "seed": args.seed,
        "density": args.density,
        "rule": args.rule,
        "rule_notation": rule_notation(args.rule),
        "palette": args.palette,
        "population_curve": curve,
        "final_population": sim.population(),
        "max_age": sim.max_age(),
        "evolve_seconds": round(evolve_s, 6),
        "generations_per_second": (
            round(args.gens / evolve_s, 1) if evolve_s > 0 else None
        ),
        "kernel": {
            "workers": workers(),
            "simd_width": simd_width(),
            "par_min_cells": par_min_cells(),
        },
        "numpy": np.__version__,
    }
    print(json.dumps(record, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(record, f, indent=2)
            f.write("\n")
        print(f"wrote {args.out}")
    return 0


def cmd_rules(_args) -> int:
    print("rules:")
    for name in sorted(reference.RULES):
        print(f"  {name:<12} {reference.RULES[name][0]}")
    print("\npalettes:")
    for name in palettes.names():
        pal = palettes.get(name)
        print(f"  {name:<10} {pal[0]} -> {pal[-1]}")
    print(f"\ndefault palette: {palettes.DEFAULT_PALETTE}")
    return 0


def _report(out: str, nbytes: int, sim: Simulation, evolve_s: float,
            frames: int | None = None) -> None:
    print(
        f"wrote {out}  {nbytes / 1024:.1f} KiB  "
        f"population {sim.population()}  max age {sim.max_age()}  "
        f"{evolve_s * 1000:.1f} ms"
        + (f"  {frames} frames" if frames else "")
    )


COMMANDS = {
    "poster": cmd_poster,
    "gif": cmd_gif,
    "ascii": cmd_ascii,
    "stats": cmd_stats,
    "rules": cmd_rules,
}


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if getattr(args, "version", False):
        from . import __version__

        print(f"mojoapp-life-art {__version__}")
        return 0
    if not args.command:
        p.print_help()
        return 0
    try:
        return COMMANDS[args.command](args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())