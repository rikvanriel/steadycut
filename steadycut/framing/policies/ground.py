"""The ground cue as a framing criterion: a second way for a sweep to bracket.

WHY THIS EXISTS. The dark-mass criterion refuses on about a third of windows and
nobody knows why: the shading explanation was measured, published and then
RETRACTED when the classes turned out to overlap by 17 grey levels. The ground
boundary is the cue for this question -- a gravity-aligned observable in the
image that riding acceleration cannot fake, against roll, where the IMU is the
better signal even with its contamination. Different questions, opposite
answers.

WHAT THE CUE MEASURED, and what it has NOT. Fitted on 19 human-labelled frames
of one window of one recording, it came back at 0.0197 held-out error, and
0.0229 when rebuilt from the profile JSON and re-scored across all 19. That is
the first real number for this material: the cue's own docstring quotes 0.082
from frames whose boundary was known by construction, which is not evidence
about this footage. The cue also predicted all 960 columns of a frame where the
labelling covered 565, so it is finding the boundary rather than reproducing the
marks.

WHAT REMAINS UNTESTED, and it is the whole point: whether the sweep BRACKETS
with it. The cue was fitted at one pitch on one window, and the sweep asks the
criterion to move MONOTONICALLY with pitch and to straddle a target across rungs
it has never seen. That is a different question from "can it find the boundary
in a frame it was fitted on", and it is the question this wiring exists to
answer. A criterion that is flat in pitch cannot bracket at any target, and that
is the outcome to watch for rather than the absolute number.

THE TARGET IS MEASURED, NOT CHOSEN. Hand-labelling 19 frames of 0519/1200 at
pitch -12 put the ground boundary at 0.311 of frame height (p10-p90
0.293-0.329, spread 0.076), so the ground fills the lower ~69% of the frame --
a trail-heavy composition, and the one a rider's own framing produced rather
than one picked to suit the metric. It is still one window on one ride, so it
stays a parameter with a provenance rather than a law.
"""
from __future__ import annotations

import numpy as np

from steadycut.framing.policies import FramingPolicy, PitchSearch, register

# Where the ground should begin, as a fraction of frame height. Measured from
# 19 hand-labelled frames of 0519/1200 at pitch -12: median 0.307, mean 0.311,
# p10-p90 0.293-0.329. One window is not a constant, so this is deliberately
# overridable rather than a second magic number in a module.
GROUND_TARGET = 0.311

# The shipped ladder. Kept the same as the dark-mass policy's so the two
# criteria are compared over the same rungs: a cue that brackets only on a range
# the other never saw would not be evidence that it is better.
LADDER = (-5.0, -12.0, -19.0, -26.0)


def boundary_of(cue, img):
    """The cue's per-column boundary, as one position for a frame.

    The reduction is a MEDIAN over columns, not a mean: the boundary is a curve
    with tails where the forest closes in, and a mean lets those tails pull the
    answer. Median over frames likewise, so one canopy frame in the stack cannot
    set the pitch. NaN when the cue finds nothing, which is the honest answer
    and the case the sweep must be able to refuse on.
    """
    b = np.asarray(cue.boundary(img), dtype=float)
    return float(np.nanmedian(b)) if np.isfinite(b).any() else float("nan")


def ground_cue_search(cue, target: float = GROUND_TARGET,
                      ladder: tuple[float, ...] = LADDER) -> PitchSearch:
    """A `PitchSearch` that brackets on the fitted ground cue.

    The cue is captured in the closure. It is not passed to the criterion and
    not held in a module global, because the sweep's criterion signature is a
    stack of frames and carries nothing else, and a global would make two cues
    in one process fight over which one is being measured.
    """
    def criterion(frames):
        per_frame = [boundary_of(cue, img) for img in frames]
        vals = np.asarray(per_frame, dtype=float)
        return float(np.nanmedian(vals)) if np.isfinite(vals).any() else float("nan")

    return PitchSearch(
        criterion=criterion,
        target=target,
        label="ground boundary (fitted per recording, human-labelled)",
        ladder=tuple(ladder),
        needs_colour=True,
    )


def search_for(source):
    """Build this recording's ground search, or None when there is no cue.

    The cue is PER RECORDING and the policy is not, so the search cannot be a
    module-level constant the way the dark-mass one is -- it has to be built
    from the source. This factory is what the CLI calls once it knows which
    clip it is framing.

    Returning None rather than a search that cannot work is deliberate. A
    policy whose cue is missing has no target, and a sweep with no target
    interpolates to an end of the ladder and calls it a pitch -- which is the
    "-5.0 degrees presented as usable" failure this module exists to prevent.
    The caller turns None into a refusal that says the cue is missing.
    """
    from steadycut.core.profiles import load_for
    profile = load_for(source)
    if profile is None or getattr(profile, "cue", None) is None:
        return None
    return ground_cue_search(profile.cue)


POLICY = register(FramingPolicy(
    name="ground",
    mount="helmet",
    follows="gaze",
    anchor="the ground/vegetation boundary starting at 31% of frame height, "
           "from a cue fitted per recording on hand-labelled frames",
    occluders=("own helmet at the nadir", "forest canopy, which can occlude "
               "the boundary entirely at any pitch"),
    search=search_for,
    hint="run steadycut-label on this recording to fit a cue, or use "
         "--framing mtb with --pitch set by eye",
    notes="a ground cue is fitted per recording, so this policy only works on "
          "recordings someone has labelled; it refuses otherwise rather than "
          "sweeping toward a target it does not have",
    capabilities=frozenset({"needs_fitted_cue"}),
))
