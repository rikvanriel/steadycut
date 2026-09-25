"""Deriving a framing pitch from rendered frames: the mount-independent half.

A policy says WHAT to measure and where that measurement should land (its
`PitchSearch`); this module does the sweeping -- render the clip at each pitch
in the ladder, measure the policy's criterion in every render, then interpolate
the pitch whose measurement lands on the target.

Nothing here knows what a helmet is, or a bar, or a ski tip. The criterion is a
callable the policy supplies, so a mount whose subject is a wheel, a ski tip, or
nothing at all uses the same sweep with its own measurement, its own target and
its own search range.

Two traps live here because both were found the hard way:

* **A ladder that never reaches the target yields a pitch that satisfies
  nothing.** Interpolation has to clamp to the nearest END, which is a number
  that looks like an answer. `brackets()` reports whether the sweep straddled
  the target, and every caller must check it before trusting the result.
* **The caller's workdir is a fresh temp subdirectory**, so it does not exist
  yet: ffmpeg fails with a bare exit status when the renders have nowhere to
  go, and the whole measurement refuses for a reason that looks like nothing.

A criterion's contract: `criterion(frames) -> float`, where `frames` is an
`(N, H, W)` grey stack and the return value is a position as a FRACTION OF
FRAME HEIGHT. A criterion that cannot measure returns NaN, which this module
treats as "not measurable at that pitch" rather than as a value.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

MEASURE_W, MEASURE_H = 960, 720     # the ladder renders at this size
MEASURE_FPS = 5.0                   # enough frames to take a percentile over


def _workdir(workdir: str | Path | None) -> Path:
    """A workdir that exists: the ladder renders have to have somewhere to go."""
    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp())
    tmp.mkdir(parents=True, exist_ok=True)
    return tmp


def solve_pitch(pitches: list[float], edges: list[float],
                target: float) -> float:
    """Interpolate the pitch whose measurement lands on `target`.

    When the measurements do not straddle the target this returns the nearest
    END -- a pitch that does not meet the criterion. Callers must check
    `brackets()` first; this function cannot refuse, because it is also the
    thing that reports where the sweep stopped.
    """
    ps = np.array(pitches, dtype=float)
    es = np.array(edges, dtype=float)
    finite = np.isfinite(es)
    if not finite.any():
        return float("nan")
    ps, es = ps[finite], es[finite]
    if target <= es.min():
        return float(ps[np.argmin(es)])
    if target >= es.max():
        return float(ps[np.argmax(es)])
    order = np.argsort(ps)
    return float(np.interp(target, es[order], ps[order]))


def brackets(edges, target: float) -> bool:
    """Did the ladder actually straddle the target?

    On one real window a helmet-mount sweep ran from 0.5717 (pitch -26) to
    0.7778 (pitch -5) against a 0.90 target and returned -5.0 as if it had
    solved something: a caller that only asks "did the criterion return a
    number?" cannot tell a solved pitch from one nobody checked.
    """
    es = [e for e in (edges.values() if isinstance(edges, dict) else edges)
          if np.isfinite(e)]
    if not es:
        return False
    return min(es) <= target <= max(es)


def measure_frames(video, w: int = MEASURE_W, h: int = MEASURE_H) -> np.ndarray:
    """Grey frames of a render, subsampled, for a criterion to measure."""
    import subprocess
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video),
         "-vf", f"fps={MEASURE_FPS:g},scale={w}:{h}",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True).stdout
    n = len(raw) // (w * h)
    return np.frombuffer(raw[:n * w * h], np.uint8).reshape(n, h, w)


def solve(source, start: float, search, duration: float = 3.0,
          workdir: str | Path | None = None):
    """Sweep a policy's ladder against its criterion; solve the pitch.

    `search` is a PitchSearch (criterion, target, ladder, label). Returns
    (pitch, info): `info["usable"]` is the thing to check -- False means the
    criterion was unmeasurable at some pitch, or the ladder never straddled the
    target, and `info["reason"]` says which and by how much.
    """
    from steadycut.core.pipeline import ClipSpec, render_constant

    tmp = _workdir(workdir)
    edges: dict[float, float] = {}
    for pitch in search.ladder:
        spec = ClipSpec(source=source, start=start, duration=duration,
                        framing_pitch=float(pitch))
        vid = render_constant(spec, tmp / f"p{abs(pitch):.0f}.mp4")
        edges[float(pitch)] = float(search.criterion(measure_frames(vid)))

    finite = {p: e for p, e in edges.items() if np.isfinite(e)}
    pitch = solve_pitch(list(finite), list(finite.values()), search.target)
    straddled = brackets(edges, search.target)
    if len(finite) < len(edges):
        missing = sorted(p for p, e in edges.items() if not np.isfinite(e))
        reason = (f"the criterion ({search.label}) could not be measured at "
                  f"pitches {missing}")
    elif not straddled:
        lo, hi = min(finite.values()), max(finite.values())
        reason = (f"the criterion ({search.label}) never reached its target: "
                  f"{lo:.2f}..{hi:.2f} of frame height against "
                  f"{search.target:.2f}, so the ladder did not bracket it")
    else:
        reason = None
    return pitch, {"edges": edges, "bracketed": straddled,
                   "usable": not reason, "criterion": search.label,
                   "target": search.target, "reason": reason}
