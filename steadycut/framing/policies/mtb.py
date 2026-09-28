"""MTB, camera on the rider's helmet: the framing policy that exists today.

The camera sits above and behind the helmet shell, so the IMU measures the
RIDER'S HEAD and the rider is also the camera operator: they already look where
they are going, so following their own (heavily low-passed) orientation gives
"looking with the turn" with no estimator at all.

Framing criterion: the bar sits in the BOTTOM ~10% of the frame, which is
equivalent to the dark mass -- the rider's own helmet, ringed by jacket and
shoulders -- topping out at 90% of frame height. That equivalence is what makes
the pitch measurable without a human: the dark-mass edge brackets 0.90, while
dark FRACTION does not (it rises monotonically with depth and never peaks).

The pitch is a PER-RECORDING constant, never carried across rides: the bar's
elevation is posture-dependent (44.5 degrees below the horizon at one moment,
38.6 at another, on the same ride), so a number that holds one recording puts
the bar out of frame on the next. None of that offset is data in this file --
it is what `search` exists to measure.
"""
from __future__ import annotations

from steadycut.framing.policies import FramingPolicy, PitchSearch, register

# Where the bar sits across the frame: parallax.NEAR's x span. Scanning there
# matters because a tire sliver in the corner dilutes below threshold over the
# full width of a frame.
BAND = (0.35, 0.65)

# Where the dark mass should top out, as a fraction of frame height.
TARGET = 0.90


def dark_mass_edge(frames):
    """Where the rider's own dark mass (helmet, jacket) tops out in frame.

    This is the policy's half of the measurement: the sweep in
    framing/ladder.py drives it and knows nothing about helmets.
    """
    from steadycut.framing.dark_mass import top_edge

    return top_edge(frames, x0=BAND[0], x1=BAND[1])


POLICY = register(FramingPolicy(
    name="mtb",
    mount="helmet",
    follows="gaze",
    anchor="bar in the bottom ~10% of frame, i.e. the dark mass (the rider's "
           "own helmet and jacket) topping out at 90% of frame height",
    occluders=("own helmet at the nadir (a FIXED occluder: same sphere region "
               "every frame, so bias the crop away from it rather than "
               "tracking it)",
               "jacket and shoulders ringing the helmet"),
    pitch=None,                     # must be measured per recording
    search=PitchSearch(
        criterion=dark_mass_edge,
        target=TARGET,
        label="dark-mass top edge (own helmet and jacket)",
        x_band=BAND,
    ),
    hint="re-try with --pitch set by eye",
    capabilities=frozenset({"pitch_measured", "gaze_follow"}),
    notes="Rider is the camera operator. Head pitch swings the subject in and "
          "out of frame, so the framing band is 2-D, not yaw-only. The rider's "
          "own arms and shoulders occlude intermittently: occlusion is normal, "
          "not failure. Never carry a pitch across recordings.",
))
