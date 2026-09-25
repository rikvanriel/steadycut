"""Automatic framing pitch from the helmet-top edge (hole-1 fix).

The bar elevation is posture-dependent (measured 7 degrees apart on one
ride), so no hand pitch transfers across recordings. What does transfer is
the CRITERION: the dark mass (helmet/bike) tops out at 90% of frame height.
Sweep a few pitches over the window, find each helmet-top edge, interpolate
to 0.90.

Method notes, all earned the hard way:
- position, not amount: dark FRACTION rises monotonically with depth and has
  no peak; the edge row does bracket 0.90.
- scan inside the bike's x-band (parallax NEAR): a tire sliver in the corner
  dilutes below threshold full-width.
- per frame, then a low percentile across frames: presence is intermittent
  (the bike bounces in and out); a time average erases it.
- colour (orange gloves) does NOT work: sunlit foliage matches, and the
  median locks onto static colour (identical 0.749 at pitches 10 deg apart).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DARK_LEVEL = 60
DARK_FRAC = 0.25
EDGE_TARGET = 0.90
EDGE_PCT = 30


def helmet_edge(frames: np.ndarray, x0: float = 0.35, x1: float = 0.65) -> float:
    """Helmet-top edge as a frame fraction for a (N, H, W) grey stack."""
    n, h, w = frames.shape
    xa, xb = int(x0 * w), int(x1 * w)
    edges = []
    for k in range(n):
        rowdark = (frames[k][:, xa:xb] < DARK_LEVEL).mean(axis=1)
        for y in range(h - 1, -1, -1):
            if rowdark[y] < DARK_FRAC:
                edges.append((y + 1) / h)
                break
    return float(np.percentile(edges, EDGE_PCT)) if edges else 1.0


def solve_pitch(pitches: list[float], edges: list[float],
                target: float = EDGE_TARGET) -> float:
    """Interpolate the pitch whose edge lands on target. Edges must bracket
    the target; otherwise returns the nearest end (caller: widen the sweep)."""
    ps = np.array(pitches, dtype=float)
    es = np.array(edges, dtype=float)
    if target <= es.min():
        return float(ps[np.argmin(es)])
    if target >= es.max():
        return float(ps[np.argmax(es)])
    order = np.argsort(ps)
    return float(np.interp(target, es[order], ps[order]))


def measure_framing(source, start: float, duration: float = 3.0,
                    pitches=(-5.0, -12.0, -19.0, -26.0),
                    workdir: str | Path | None = None):
    """Render the pitch ladder, measure each edge, solve. Returns
    (pitch, edges-dict) -- numerics in memory, renders to a temp dir."""
    import subprocess
    import tempfile
    from steadycut.core.pipeline import ClipSpec, render_constant
    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp())
    edges = {}
    for pitch in pitches:
        spec = ClipSpec(source=source, start=start, duration=duration,
                        framing_pitch=float(pitch))
        vid = render_constant(spec, tmp / f"p{abs(pitch):.0f}.mp4")
        col = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(vid), "-vf", "fps=5,scale=960:720",
             "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            capture_output=True, check=True).stdout
        w, h = 960, 720
        n = len(col) // (w * h)
        fr = np.frombuffer(col[:n * w * h], np.uint8).reshape(n, h, w)
        edges[float(pitch)] = helmet_edge(fr)
    pitch = solve_pitch(list(edges), [edges[p] for p in edges])
    return pitch, edges
