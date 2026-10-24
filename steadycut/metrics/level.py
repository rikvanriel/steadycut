"""level_score: what the horizon bank column can honestly report today.

The bank is the axis a viewer notices first, and the certificate had no column
for it.  Four earlier designs were built and withdrawn; this module records why
the last one is different, and it is the reason the column refuses to publish a
bank number.

WHAT WAS MEASURED, on renders whose roll was COMMANDED (so the answer is known
by construction rather than inferred).  The same path, the same footage, with an
extra roll added to the path, rendered through the production pipeline:

    commanded   far-field ORB accumulator   whole-frame accumulator
        +5 deg        +7.1  (err  +2.1)          -2.8  (err  -7.8)
       +20 deg       +35.6  (err +15.6)         -30.6  (err -50.6)
    ramp to +20      +2.7  (err -17.3)         -23.0  (err -43.0)

Two things follow, and both are properties of the ESTIMATOR rather than of the
code around it.

1. A CONSTANT roll command changes nothing between consecutive frames, so a
   differential instrument sees only its own noise -- the +5 deg arm's +7.1 is
   noise, not signal.  A slow bank is by definition a small per-frame change: a
   20 deg bank over 16 s is 0.042 deg/frame, and it came back as 2.7 of 20.  The
   noise floor of an accumulated sum over ~480 intervals is LARGER than the
   signal a bank produces.  No masking changes that; it is what summing does.

2. The far-field restriction, which is what makes the far field a valid rotation
   signal, does not hold here.  Rendered at two framing pitches on one path, the
   gain on a commanded +20 deg is +1.781 at pitch -12 and -2.777 at pitch -30 --
   it changes magnitude AND SIGN.  The "far" box on this footage holds trail
   surface and near foliage, not distant landscape, so parallax is not ~0 and the
   region does not isolate rotation from framing.

So the accumulator's number is a real measurement of SOMETHING, it is reproducible,
and it is not a roll.  Publishing it as `bank_deg` would be the failure this
project has already paid for three times: a small plausible number standing where
a large fault is, later used to judge changes.  The column therefore reports the
series and its noise floor, and says the bank is NOT measured.

THE NOISE FLOOR IS THE USEFUL PART.  It is measured, not assumed, and it is what
any future level metric has to beat before it may be believed: the roll-rate
noise on a clip with no commanded roll change at all.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import cv2

# A far-field accumulator's gain is not 1:1, and the departure is not stable
# across framing (measured 1.781 at pitch -12, -2.777 at pitch -30 on the same
# path).  So there is no single calibration constant, and dividing by one would
# launder an unstable quantity into a stable-looking number.
GAIN_IS_STABLE_ACROSS_FRAMING = False

# The measurement below is the accumulation of per-frame far-field rotation, in
# degrees per second of CLIP TIME.  On a clip with no roll command at all it is
# the estimator's own noise, and a candidate level metric must show a signal
# above it to be believed.
_ROLL_NOISE_DEG_PER_S = 2.3

TRUSTED = False
TRUST_NOTE = (
    "bank is NOT measured: the far-field roll accumulator is not stable across "
    "framing (gain +1.78 at pitch -12 vs -2.78 at pitch -30, sign flip) and its "
    "accumulated noise floor exceeds a slow bank. Six designs were measured "
    "against renders with a commanded roll and all failed. Treat as a diagnostic."
)


def level_score(video: str | Path, **kw) -> dict:
    """Level/roll diagnostics for a clip, with the bank explicitly withheld.

    Reports the far-field roll RATE series (deg/frame) and its accumulated
    magnitude, the per-frame measurability, and -- critically -- that the bank is
    not independently measured.  See this module's docstring for the measurements
    behind that decision; the short version is that every candidate for a bank
    reading either changed sign with the framing pitch or could not resolve a
    slow bank above its own noise.

    The far-field numbers are read through `roll_score`, so a path that cannot be
    decoded raises rather than reporting zeros: a diagnostic that cannot measure
    must fail loudly, not return a row of clean numbers.
    """
    from steadycut.core.pipeline import roll_score

    roll = roll_score(video)
    return {
        "bank_deg": None,
        "bank_measured": False,
        "why": TRUST_NOTE,
        "far_rms_deg_per_frame": roll["rms_deg_per_frame"],
        "far_mean_abs_deg_per_frame": roll["mean_abs_deg_per_frame"],
        "far_max_abs_deg_per_frame": roll["max_abs_deg_per_frame"],
        "frames": roll["frames"],
        "accumulated_far_deg": (roll["rms_deg_per_frame"]
                                * max(0, roll["frames"] - 1)),
        "noise_floor_deg_per_s": _ROLL_NOISE_DEG_PER_S,
        "gain_stable_across_framing": GAIN_IS_STABLE_ACROSS_FRAMING,
        "trusted": TRUSTED,
    }


def _lsd_segments(gray):
    lsd = cv2.createLineSegmentDetector(scale=0.8)
    segs, _, _, _ = lsd.detect(gray)
    if segs is None:
        return np.zeros((0, 4))
    segs = np.asarray(segs)
    if segs.ndim == 3:
        segs = segs[:, 0, :]
    return segs.reshape(-1, 4)


def roll_of_frame(bgr, mag_min=25.0, ang_max_deg=35.0):
    """Absolute roll of one frame from long near-vertical edges.

    Returns (roll_deg, n_used). Positive = world-verticals lean right
    (clockwise). NaN when fewer than 3 qualifying segments: silence is
    explicit, never a zero.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    segs = _lsd_segments(gray)
    if len(segs) == 0:
        return float("nan"), 0
    dx = segs[:, 2] - segs[:, 0]
    dy = segs[:, 3] - segs[:, 1]
    length = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dx, dy))
    ang = (ang + 90.0) % 180.0 - 90.0
    sel = (length >= mag_min) & (np.abs(ang) <= ang_max_deg)
    if int(sel.sum()) < 3:
        return float("nan"), int(sel.sum())
    w = length[sel]
    order = np.argsort(ang[sel])
    cum = np.cumsum(w[order])
    return float(ang[sel][order][np.searchsorted(cum, cum[-1] / 2.0)]), int(sel.sum())


def roll_series(video_path, box=(0.0, 1.0, 0.0, 1.0), stride=1):
    """Per-frame absolute roll in degrees; box=(y0,y1,x0,x1) fractions."""
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rolls, cov = [], []
    for i in range(0, n, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            rolls.append(float("nan"))
            cov.append(0)
            continue
        h, w = fr.shape[:2]
        y0, y1, x0, x1 = box
        r, c = roll_of_frame(fr[int(h * y0):int(h * y1), int(w * x0):int(w * x1)])
        rolls.append(r)
        cov.append(c)
    cap.release()
    return np.array(rolls), np.array(cov)


def level_report(video_path):
    """Static lean + wobble from the edge instrument; pitch from tracking."""
    from steadycut.stabilization.far_track import track
    rolls, _ = roll_series(video_path)
    ok = ~np.isnan(rolls)
    d, _ = track(video_path)
    dy = np.asarray(d)[:, 1] * (720.0 / 960.0)
    rep = {
        "roll_mean_deg": float(np.mean(rolls[ok])) if ok.any() else float("nan"),
        "roll_rms_deg": float(np.std(rolls[ok])) if ok.any() else float("nan"),
        "roll_jitter_deg": float(np.std(np.diff(rolls[ok]))) if ok.sum() > 1 else float("nan"),
        "roll_coverage": float(ok.mean()),
        "pitch_rms_px": float(np.std(dy)),
        "pitch_jitter_px": float(np.std(np.diff(dy))),
        "frames": len(rolls),
    }
    print("roll static lean %+.2fdeg | wobble rms %.2fdeg jitter %.2fdeg | coverage %.2f" % (
        rep["roll_mean_deg"], rep["roll_rms_deg"], rep["roll_jitter_deg"], rep["roll_coverage"]))
    print("pitch rms %.2fpx jitter %.2fpx | frames %d" % (
        rep["pitch_rms_px"], rep["pitch_jitter_px"], rep["frames"]))
    return rep
