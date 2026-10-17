"""Ski, camera on the rider's helmet: travel-heading framing.

The IMU measures the SKIER'S HEAD, and on skis the head is not the travel:
carve lean rolls the helmet +/-30 deg at turn frequency, and look-arounds
yaw it 90+ deg for seconds (measured Jan26/110-118: 6 s woods-watching
mid-run). A gaze follower shows the woods; a travel holder shows the slope.

Framing criterion: the virtual camera holds the TRAVEL heading -- gaze yaw
on a slow leash (YAW_CORNER), pitch/roll on the production corner. Sustained
turns (a winding cat track turns 150 deg in 20 s) pass the leash; look-arounds
do not. Validated by the residual metric and the eye on the Jan26 whip
window (hold-heading 6.5/2.7 vs follow-gaze 15.5/18, watchable vs whippy).

Level comes from gravity (the shipped levelling, which ski needs most: the
helmet rolls far more than a bike helmet). Pitch is a fixed default for now:
grade-aware pitch (gravity median sees slope grade) is the ground-policy
junction and is explicitly future work, not smuggled in here.

Open (needs footage, not guessing): a slew cap on travel yaw for sustained
looks (30 s kid-watching WILL drag any leash eventually). No cap ships in v1;
the new videos decide whether one is needed and where it sits.
"""
from __future__ import annotations

from steadycut.framing.policies import FramingPolicy, register

# Travel trend: slow on every channel. The head's transients (whips,
# leans, bumps) are rejected on yaw AND roll AND pitch; sustained motion
# (turns, grade changes) passes over tens of seconds. 0.02 Hz needs ~25 s to
# settle, so it can only be validated on windows 30 s and up -- an 8 s clip
# renders one long catch-up transient, which is what the first ski_policy
# attempt measured (and misread). Yaw-only leashing was tried and fails: the
# whip rolls the helmet +-90 deg, and a held yaw with a following roll couples
# through the Euler order into vertical swing (cert roll 1.31 vs raw 0.94).
CORNER_HZ = 0.02
YAW_CORNER_HZ = CORNER_HZ

# Fixed framing pitch until the grade-aware policy lands. -12 matches the
# MTB default the ski renders were validated with; it is a placeholder with
# a name, not a measurement.
PITCH_DEFAULT = -12.0


POLICY = register(FramingPolicy(
    name="ski",
    mount="helmet",
    follows="travel",
    anchor="virtual camera holds the travel heading: sustained turns pass, "
           "look-arounds do not; helmet roll levelled by gravity",
    occluders=("own helmet at the nadir (a FIXED occluder: same sphere region "
               "every frame, so bias the crop away from it rather than "
               "tracking it)",
               "ski tips rising into frame in turns (transient, ignore)"),
    pitch=PITCH_DEFAULT,
    yaw_corner_hz=YAW_CORNER_HZ,
    yaw_adaptive_hz=0.16,  # ~1 s leash inside sustained turns
    corner_hz=CORNER_HZ,
    search=None,                    # grade-aware pitch is future work
    hint="grade-aware framing pitch is not implemented yet; pass --pitch",
    notes="full travel trend at 0.02 Hz (yaw-only leashing tried and "
          "rejected: Euler-order coupling); needs 30 s+ windows to settle; "
          "no slew cap in v1 (needs footage)",
    capabilities=frozenset({"travel-yaw"}),
))
