"""MTB, camera on the rider's helmet: the framing policy that exists today.

The camera sits above and behind the helmet shell, so the IMU measures the
RIDER'S HEAD and the rider is also the camera operator: they already look where
they are going, so following their own (heavily low-passed) orientation gives
"looking with the turn" with no estimator at all.

Framing criterion: the bar sits in the BOTTOM ~10% of the frame, which is
equivalent to the dark mass -- the rider's own helmet, ringed by jacket and
shoulders -- topping out at 90% of frame height. That equivalence is what makes
the pitch measurable without a human: the dark-mass edge brackets 0.90 while
dark FRACTION does not (it rises monotonically with depth and never peaks).

The pitch is a PER-RECORDING constant, never carried across rides: the bar's
elevation is posture-dependent (44.5 degrees below the horizon at one moment,
38.6 at another, on the same ride), so a number that holds one recording puts
the bar out of frame on the next.
"""
from __future__ import annotations

from pathlib import Path

from steadycut.framing.policies import FramingPolicy, register


def measure_pitch(source, start: float, workdir: str | Path | None = None):
    """Solve this recording's pitch from the helmet-top edge.

    Returns (pitch, info). `info["usable"]` is False when the dark mass was
    never found at any pitch in the ladder, which means the sweep cannot have
    solved anything -- the caller refuses rather than picking an end.
    """
    from steadycut.framing import autopitch

    pitch, edges = autopitch.measure_framing(source, start, workdir=workdir)
    found = not all(e >= 0.999 for e in edges.values())
    bracketed = autopitch.brackets(edges)
    if not found:
        reason = "no dark mass found at any pitch"
    elif not bracketed:
        # The solver returned an end of the ladder, so the criterion was never
        # met: say so rather than rendering a pitch nobody checked. Measured
        # on a May 26 window the edges ran 0.57..0.78 against a 0.90 target.
        lo, hi = min(edges.values()), max(edges.values())
        reason = (f"the dark-mass edge never reached the framing criterion: "
                  f"{lo:.2f}..{hi:.2f} of frame height against 0.90, so the "
                  f"ladder did not bracket it")
    else:
        reason = None
    return pitch, {
        "edges": edges,
        "usable": found and bracketed,
        "bracketed": bracketed,
        "criterion": "dark-mass top edge at 0.90 of frame height",
        "reason": reason,
    }


POLICY = register(FramingPolicy(
    name="mtb",
    mount="helmet",
    follows="gaze",
    anchor="bar in the bottom ~10% of frame, i.e. the dark mass topping out "
           "at 90% of frame height",
    occluders=("own helmet at the nadir (a FIXED occluder: same sphere region "
               "every frame, so bias the crop away from it rather than tracking it)",
               "jacket and shoulders ringing the helmet"),
    pitch=None,                 # must be measured per recording
    measure=measure_pitch,
    hint="re-try with --pitch set by eye",
    capabilities=frozenset({"pitch_measured", "gaze_follow"}),
    notes="Rider is the camera operator. Head pitch swings the subject in and "
          "out of frame, so the framing band is 2-D, not yaw-only. The rider's "
          "own arms and shoulders occlude intermittently: occlusion is normal, "
          "not failure. Never carry a pitch across recordings.",
))
