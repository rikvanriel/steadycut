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


# ffmpeg writes the mp4 container even when it encodes no frames, so a window
# past the end of the footage leaves a small file rather than none: 262 bytes was
# observed on a real .insv, against tens of MB for a real 3 s render.  Measuring
# one returns whatever is on disk, which is stale content from an earlier run, so
# a window that was never rendered has to be reported as unmeasurable rather than
# read.  Guarded by framing/tests/test_empty_render.py.
MIN_PLAUSIBLE_BYTES = 4096

# MEASURED AND THEN RETRACTED. A band-brightness threshold at 96 was derived
# here on seven windows and read as separating bracketing from refusing windows
# 7 of 7. Widening the sample to seventeen windows across three rides, with each
# window's own bracket verdict measured on its own renders, shows the classes
# OVERLAP by seventeen grey levels:
#
#     bracketing: 8 windows, medians 80 90 96 96 98 104 108 115
#     refusing:   9 windows, medians 64 66 68 73 78 84 86 88 97
#
# A bracketing window at 80 sits below a refusing one at 97, so no threshold
# separates them; the chosen value scored 12 of 17 and the best available
# threshold reached 15 of 17. The seven-window sample looked clean because none
# of the seven sat near the boundary, which is exactly when a small sample is
# most convincing. Shade remains a real tendency -- refusing windows are darker
# on average -- but it does not determine the outcome, so it cannot be named as
# the cause of a refusal. The lesson belongs with the number: the first question
# about a threshold is the MARGIN around it, not the count of easy cases.
#
# The band-versus-frame result below is KEPT, because it is about where to
# measure rather than about a cut-off: on 0519/1200 the criterion's own band
# reads 87 while the whole frame reads 127, since forest fills the frame. A
# whole-frame reading would miss the case it is used for.
SHADED_BAND_MEDIAN = 96.0


def _band_median(frames, band=None) -> float:
    """Median grey level inside a policy's x band, or NaN if there is nothing to read.

    Measured in the band the criterion scans, not the whole frame: on shaded
    trail footage the forest fills most of the frame and drags a whole-frame
    median well above the band reading.  Kept because it is the instrument a
    future diagnosis would need, and because where to measure is settled even
    though no cut-off on this measurement has survived validation.
    """
    if frames is None or band is None:
        return float("nan")
    f = np.asarray(frames)
    if f.ndim != 3 or f.size == 0:
        return float("nan")
    h, w = f.shape[1], f.shape[2]
    x0, x1 = int(band[0] * w), int(band[1] * w)
    f = f[:, :, x0:x1] if x1 > x0 else f
    return float(np.median(f.astype(np.float32))) if f.size else float("nan")


def solve(source, start: float, search, duration: float = 3.0,
          workdir: str | Path | None = None):
    """Sweep a policy's ladder against its criterion; solve the pitch.

    `search` is a PitchSearch (criterion, target, ladder, label). Returns
    (pitch, info): `info["usable"]` is the thing to check -- False means the
    criterion was unmeasurable at some pitch, or the ladder never straddled the
    target, and `info["reason"]` says which and by how much.

    A refusal is a refusal: an unsolvable window gets no pitch, whatever the
    cause, because inventing one is the failure this module exists to prevent.
    It does not claim to know the cause. An `info["shaded"]` diagnosis was here
    and was removed: its band-brightness threshold separated bracketing from
    refusing windows 7 of 7 on seven windows and overlapped by seventeen grey
    levels on seventeen, so the key described a cause the data does not support.
    """
    from steadycut.core.pipeline import ClipSpec, render_constant

    tmp = _workdir(workdir)
    edges: dict[float, float] = {}
    stacks: dict[float, np.ndarray] = {}
    for pitch in search.ladder:
        spec = ClipSpec(source=source, start=start, duration=duration,
                        framing_pitch=float(pitch))
        vid = render_constant(spec, tmp / f"p{abs(pitch):.0f}.mp4")
        # A render past the end of the recording encodes nothing and ffmpeg
        # leaves a container with no payload, so a window that was never
        # rendered has to be reported as unmeasurable rather than read --
        # measuring it returns whatever is on disk, which is stale content from
        # an earlier run.  The size check is the last line of defence, AFTER
        # measurement, because a stubbed renderer writes no file at all and
        # checking before measuring would discard every rung of a unit test
        # while fixing nothing.
        stack = measure_frames(vid)
        if stack is None or getattr(stack, "size", 0) == 0 or (
                vid.exists() and vid.stat().st_size < MIN_PLAUSIBLE_BYTES):
            edges[float(pitch)] = float("nan")
            continue
        stacks[float(pitch)] = stack
        edges[float(pitch)] = float(search.criterion(stack))

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
