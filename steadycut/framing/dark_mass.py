"""The top edge of a dark mass, as a fraction of frame height.

An INSTRUMENT, not a policy. It answers one question -- "how far down the frame
does the dark thing start?" -- for whatever dark thing a policy points it at.
On a helmet mount that is the rider's own helmet and jacket, which is what puts
the bar in the bottom of the frame; another mount may aim it at a wheel, a
frame bag, or nothing at all. The x-band and the target fraction belong to the
policy; the measuring does not.

Method notes, all earned the hard way:
- position, not amount: dark FRACTION rises monotonically with depth and has no
  peak, so it cannot be interpolated to a target; the edge row does bracket one.
- scan inside the subject's x-band (a helmet mount uses parallax.NEAR): a tire
  sliver in the corner dilutes below threshold full-width.
- per frame, then a low percentile across frames: presence is intermittent (the
  bike bounces in and out) and a time average erases it.
- colour does NOT work: orange gloves match sunlit foliage, and the median locks
  onto static colour (identical 0.749 at pitches 10 degrees apart).
"""
from __future__ import annotations

import numpy as np

DARK_LEVEL = 60
DARK_FRAC = 0.25
EDGE_PCT = 30


def top_edge(frames: np.ndarray, x0: float = 0.0, x1: float = 1.0,
             dark_level: int = DARK_LEVEL, dark_frac: float = DARK_FRAC,
             pct: float = EDGE_PCT) -> float:
    """Topmost row of the dark mass in the x-band, as a frame fraction.

    Returns NaN when no frame carried a dark mass at all: "not measurable",
    which the sweep treats as a refusal rather than as a position. Reporting a
    position for a mass that was never found is how a pitch that meets no
    criterion gets rendered.
    """
    n, h, w = frames.shape
    xa, xb = int(x0 * w), int(x1 * w)
    edges = []
    for k in range(n):
        rowdark = (frames[k][:, xa:xb] < dark_level).mean(axis=1)
        if rowdark.max() < dark_frac:
            # Nothing dark in this frame's band at all. Without this check the
            # bottom-up scan below would "find" the frame's last row and report
            # the mass as topping out at the very bottom -- a position for a
            # mass that is not there, which is what a caller then interpolates
            # against.
            continue
        for y in range(h - 1, -1, -1):
            if rowdark[y] < dark_frac:
                edges.append((y + 1) / h)
                break
    if not edges:
        return float("nan")
    return float(np.percentile(edges, pct))
