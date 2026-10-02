"""Palettes: 256-entry RGB ramps indexed by cell age.

Index 0 is what a cell of age 0 gets, which for a fresh render is the whole
grid — so index 0 is the background and index 255 is the oldest surviving
lineage. That ordering is the point: a Life run is a colour ramp read forwards
in time.

Palettes are named so `life-art --palette ember poster.png` reads like a
command rather than a hex string, and every ramp is built by one function so
they stay in step.
"""

from __future__ import annotations

import colorsys


def _hex_to_rgb(s: str) -> tuple[int, int, int]:
    s = s.lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def _lerp(a, b, t: float) -> tuple[int, int, int]:
    return (
        round(a[0] + (b[0] - a[0]) * t),
        round(a[1] + (b[1] - a[1]) * t),
        round(a[2] + (b[2] - a[2]) * t),
    )


def multi_stop(stops: list[tuple[float, str]], n: int = 256) -> list[tuple[int, int, int]]:
    """A ramp through `(position, "#rrggbb")` stops, positions in [0, 1].

    Stop 0 is index 0 and stop 1 is index n-1, so the endpoints of the ramp
    line up with the endpoints of the age range without an off-by-one at the
    young end.
    """
    if len(stops) < 2:
        raise ValueError("a ramp needs at least two stops")
    pts = sorted((float(pos), _hex_to_rgb(col)) for pos, col in stops)
    out = []
    for i in range(n):
        t = i / (n - 1)
        lo = pts[0]
        hi = pts[-1]
        for j in range(len(pts) - 1):
            if pts[j][0] <= t <= pts[j + 1][0]:
                lo, hi = pts[j], pts[j + 1]
                break
        span = hi[0] - lo[0]
        local = 0.0 if span <= 0 else (t - lo[0]) / span
        out.append(_lerp(lo[1], hi[1], local))
    return out


def hue_sweep(start: float, turns: float, s: float, v: float, n: int = 256) -> list[tuple[int, int, int]]:
    """A full hue cycle: a good default when you want the lineage bands to read
    as distinct colours rather than a single fade."""
    out = []
    for i in range(n):
        h = (start + turns * i / (n - 1)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        out.append((round(r * 255), round(g * 255), round(b * 255)))
    return out


# Every palette is index 0 -> index 255, young -> old.
PALETTES: dict[str, list[tuple[int, int, int]]] = {
    "ember": multi_stop(
        [(0.0, "#0b0a12"), (0.35, "#5b1a2e"), (0.7, "#d1481f"), (1.0, "#ffd98a")]
    ),
    "spectrum": hue_sweep(0.62, 1.0, 0.85, 1.0),
    "ink": multi_stop(
        [(0.0, "#f6f4ef"), (0.5, "#7a7f8a"), (1.0, "#101418")]
    ),
    "terminal": multi_stop(
        [(0.0, "#04140a"), (0.5, "#0f7a3d"), (1.0, "#7dffa0")]
    ),
    "ice": multi_stop(
        [(0.0, "#04070f"), (0.45, "#1c3f7a"), (1.0, "#bfe6ff")]
    ),
    "dusk": multi_stop(
        [(0.0, "#120a1e"), (0.4, "#5b2a6e"), (0.75, "#c04b7a"), (1.0, "#ffc98b")]
    ),
    "mono": multi_stop([(0.0, "#000000"), (1.0, "#ffffff")]),
}

DEFAULT_PALETTE = "ember"


def get(name: str) -> list[tuple[int, int, int]]:
    """Look up a palette by name."""
    try:
        return PALETTES[name]
    except KeyError:
        raise ValueError(
            f"unknown palette {name!r}; choose from {', '.join(sorted(PALETTES))}"
        ) from None


def names() -> list[str]:
    return sorted(PALETTES)


def to_bytes(palette: list[tuple[int, int, int]]) -> bytes:
    """Flatten to the 768-byte RGB table the Mojo renderer indexes."""
    return bytes(v for px in palette for v in px)