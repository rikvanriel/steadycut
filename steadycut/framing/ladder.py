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

# Measured 2026-09-28 across 7 windows on three rides: the median brightness of a
# criterion's own x band separates the windows where a dark-mass target is
# reachable from the ones where it is not, 7 of 7 at a threshold of 96 grey
# levels, with bracketing windows averaging 107 and failing windows 89.  The
# failing windows are SHADED, and there the dark mass the criterion finds is the
# forest floor rather than the rider, so no pitch brings it to the target.
#
# The band matters and the whole frame does not: on a shaded window the FOREST
# fills most of the frame and lifts the whole-frame median above the threshold,
# so measuring the whole frame silently loses the signal.  Measured 0519/1200:
# band median 87, whole-frame median 127.  A policy states its band; without one
# the diagnosis declines rather than guessing.
SHADED_BAND_MEDIAN = 96.0


def _band_median(frames, band=None) -> float:
    """Median grey level inside a policy's x band, or NaN if there is nothing to read.

    Measured in the band the criterion scans, not the whole frame: on shaded
    trail footage the forest fills most of the frame and drags a whole-frame
    median above the threshold, which is exactly the case this needs to catch.
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


def diagnose_refusal(search, frames_by_pitch: dict) -> str | None:
    """Explain WHY a sweep failed to bracket, when the band can tell us.

    Two refusals look identical from `edges` alone and need different actions:
    the criterion could not be measured (a canopy window, or a pitch where the
    subject is out of frame), or it measured fine and the target was simply out
    of reach. Measured on real footage, the second case is a SHADING problem --
    the dark mass in a shaded window is the forest floor, not the subject -- and
    a wider ladder cannot fix it, because the pitch that would land the target
    there is one that points at the sky.

    So when the band is dark, say that, and say that re-ranging will not help.
    Returning None means "no better explanation than the generic one", and the
    caller keeps the generic reason.
    """
    band = getattr(search, "x_band", None)
    med = _band_median(next(iter(frames_by_pitch.values()), None), band)
    if not np.isfinite(med):
        return None
    if med >= SHADED_BAND_MEDIAN:
        return None
    return (
        f"the band is dark (median {med:.0f} against {SHADED_BAND_MEDIAN:.0f}), "
        f"so the dark mass being measured is most likely shaded ground rather "
        f"than the subject; a wider pitch ladder will not reach the target "
        f"here, and framing in this window needs the pitch set by eye or by a "
        f"criterion that does not depend on brightness")


def solve(source, start: float, search, duration: float = 3.0,
          workdir: str | Path | None = None):
    """Sweep a policy's ladder against its criterion; solve the pitch.

    `search` is a PitchSearch (criterion, target, ladder, label). Returns
    (pitch, info): `info["usable"]` is the thing to check -- False means the
    criterion was unmeasurable at some pitch, or the ladder never straddled the
    target, and `info["reason"]` says which and by how much.

    `info["shaded"]` is True when the band was dark enough that the unbracketed
    target is an exposure problem rather than a framing one. It is a diagnosis,
    not a licence: the sweep still refuses, because a shaded window genuinely
    has no solution here and inventing a pitch is the failure this module exists
    to prevent.
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
    shaded = False
    span = ""                    # the measured range, for the reason string
    if len(finite) < len(edges):
        missing = sorted(p for p, e in edges.items() if not np.isfinite(e))
        reason = (f"the criterion ({search.label}) could not be measured at "
                  f"pitches {missing}")
    elif not straddled:
        lo, hi = min(finite.values()), max(finite.values())
        span = (f"{lo:.2f}..{hi:.2f} of frame height against "
                f"{search.target:.2f}")
        why = diagnose_refusal(search, stacks)
        shaded = why is not None
        reason = why or (
            f"the criterion ({search.label}) never reached its target: "
            f"{span}, so the ladder did not bracket it")
    else:
        reason = None
    # A diagnosis is APPENDED to the numbers, never substituted for them: the
    # generic reason is what carries the criterion, the measured range and the
    # target, and those are what a reader needs to judge the sweep.  Replacing it
    # with "it is dark" would pass a message and lose the evidence.
    if reason is not None and shaded:
        reason += f" (criterion '{search.label}' reached {span})"
    return pitch, {"edges": edges, "bracketed": straddled,
                   "usable": not reason, "criterion": search.label,
                   "target": search.target, "reason": reason,
                   "shaded": shaded}
