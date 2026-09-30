"""Is this frame a shot worth delivering? Three conditions, none of them a target.

Framing was judged for a long stretch by an angle read off the IMU, and the
rider caught it: "you are judging the framing by the angles used, rather than the
resulting image". The correction is recorded here rather than left implicit,
because it decides what this module is FOR.

**A framing criterion is not a measure.** A pitch band was tried as the framing
judgement and it does not work, for a reason visible in the rider's own labels:
the required pitch depends on the terrain slope, so the mapping from camera pitch
to what is in frame is not monotonic. Across one sampled sheet the judgement went
good at +29 degrees, bad at +40, and good again at +50 -- which is not a band, it
is a terrain-dependent target flattened onto one axis. So nothing here decides
whether a shot is well COMPOSED. That is the image cue's job, and this module
does not touch it.

What this module answers is a different and narrower question: **is this frame a
riding shot at all?** The rider labelled eleven of forty sampled frames with the
same phrase -- "no horizon" -- covering a camera upside down, a camera pointed at
its own body, and a rider bent over a bike on a rack. Those frames have no framing
target, so no target choice helps them and a confident render is the wrong
outcome. This module refuses them and says which condition failed.

THE THREE CONDITIONS, and why each exists:

  * **roll** -- an inverted camera reads PITCH of roughly zero and lands in the
    middle of any sane pitch band. Measured on this material: three frames the
    rider called "upside down and sideways" and one "still putting my helmet on"
    all sat inside the good band on pitch, at roll 134 to 158 degrees. A 45 degree
    limit caught 5 of the 6 not-riding frames.
  * **speed** -- the sixth was at roll 11 degrees and pitch 36, an attitude
    indistinguishable from riding. A bike that is not moving is not riding.
    NOT IMPLEMENTED HERE, and the omission is deliberate rather than an oversight:
    there is no speed in the IMU, because forward speed produces no wheel
    parallax and reads from the streaming trail surface, not from the camera. It
    needs the far-field tracker, so this module REFUSES to guess one.
  * **a visible boundary** -- the frames where the camera is level and pointed at
    the rider's own body have an attitude that looks fine and no ground boundary
    to compose around. NOT IMPLEMENTED HERE either: a degeneracy check for it was
    built, and it fires on 0 of 4 good windows AND 0 of 4 broken ones, so it
    would be a guard that never fires, which is the mirror of a detector that
    underreads and equally harmful -- a green suite that makes the case look
    covered while the failure stays in place.

So this module ships ONE working condition and says plainly what it does not
cover, because a refusal guard that converts a silent wrong answer into a
refusal is worth having and one that claims coverage it lacks is not.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Beyond this much roll the frame is not a level riding shot. Measured: a 45
#: degree limit caught 5 of the 6 frames the rider called not-riding, and the
#: clean ones on the same corpus sit under 35.
ROLL_LIMIT_DEG = 45.0


@dataclass(frozen=True)
class Validity:
    """Per-sample validity, and a reason for every sample that failed."""

    valid: np.ndarray          # bool, per sample
    reason: np.ndarray         # str, per sample; "" when valid
    roll: np.ndarray
    pitch: np.ndarray
    #: The conditions this module did NOT evaluate, so a caller cannot mistake a
    #: pass here for a pass overall.
    unchecked: tuple[str, ...] = ("speed", "visible_boundary")


def assess(attitude, roll_limit_deg: float = ROLL_LIMIT_DEG) -> Validity:
    """Judge a riding shot per sample on the conditions available here.

    `attitude` is an `attitude_vqf.Attitude`. The return is per SAMPLE rather
    than per clip, because a clip can contain both riding frames and the first
    seconds of a rider putting their helmet on, and a single verdict over the
    clip would hide one inside the other.
    """
    roll = np.asarray(attitude.roll, dtype=float)
    pitch = np.asarray(attitude.pitch, dtype=float)
    if roll.shape != pitch.shape:
        raise ValueError(
            f"roll has {roll.shape} samples and pitch has {pitch.shape}")

    valid = np.ones(roll.shape, dtype=bool)
    reason = np.full(roll.shape, "", dtype=object)

    tilted = ~np.isfinite(roll) | (roll > roll_limit_deg)
    valid &= ~tilted
    reason[tilted] = "not-riding-shot: camera is not level"

    unknown = ~np.isfinite(roll) | ~np.isfinite(pitch)
    reason[unknown] = "not-measurable: attitude is not finite here"
    valid &= ~unknown

    return Validity(valid=valid, reason=reason, roll=roll, pitch=pitch)


def summary(attitude, roll_limit_deg: float = ROLL_LIMIT_DEG) -> dict:
    """Clip-level rollup: what fraction was deliverable, and what was not."""
    v = assess(attitude, roll_limit_deg)
    n = int(v.valid.size)
    bad = int((~v.valid).sum())
    out = {
        "samples": n,
        "refused": bad,
        "refused_fraction": (bad / n) if n else float("nan"),
        "roll_limit_deg": roll_limit_deg,
        "max_roll_deg": float(np.nanmax(v.roll)) if n else float("nan"),
        "not_evaluated": v.unchecked,
    }
    if bad:
        kinds = {}
        for r in v.reason[~v.valid]:
            kinds[str(r)] = kinds.get(str(r), 0) + 1
        out["reasons"] = kinds
    return out
