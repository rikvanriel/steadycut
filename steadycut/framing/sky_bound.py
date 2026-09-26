"""How much sky is ALLOWED in the top band -- a bound, not an objective.

Both other anchors were measured wrong, in ways worth keeping visible:

  * the sky share alone is NOT monotone in pitch: "reach 25%" can be satisfied by
    looking up into a canopy gap, which drove the view up the trees and lost the
    bike entirely;
  * the bike anchor inverts when the rider ducks (the machine rises above the
    camera's axis, so "bar at the bottom" asks for the sky), and the dark-mass
    edge cannot be measured in canopy at all.

So the framing decides WHERE the view looks -- the ride's own constant -- and the
sky share only bounds how much sky is permitted in the top band. The correction
is one-sided by construction: it tilts down when there is too much sky and does
nothing otherwise. It cannot trade the trail away, which is what both other
attempts did.

Searching DOWNWARD only is what keeps the result monotone. The band moves away
from the sky as the view drops; searching upward runs into the canopy-gap
ambiguity that broke the earlier attempt, where a gap in the leaves satisfied a
sky target by pointing at the trees.

Measured on rendered files: against a constant pitch in a window where the pitch
policy refuses, this halves the frames over the bound and leaves the descent
untouched.
"""

from __future__ import annotations

import numpy as np

V_FOV = 90.0
TOP_BAND = 0.30          # fraction of frame height counted as the top band
FORWARD = 60.0           # degrees either side of dead ahead
BOUND = 0.25             # sky allowed in the top band
FLOOR = -45.0            # never search below this


def sky_mask(img: np.ndarray) -> np.ndarray:
    """Blue-dominant pixels. Sunlit foliage can read blue-ish, so this is used
    only as a share with a bound -- never as a target to optimise."""
    b = img[..., 0].astype(int)
    g = img[..., 1].astype(int)
    r = img[..., 2].astype(int)
    return (b > r + 12) & (b > g + 4)


def grid(w: int, h: int):
    """Pitch and yaw of every equirect pixel, in degrees."""
    pitch = 90.0 - (np.arange(h)[:, None] + 0.5) * 180.0 / h
    yaw = (np.arange(w)[None, :] + 0.5) * 360.0 / w
    return pitch, yaw


def share_fn(sky: np.ndarray, pitch: np.ndarray, yaw: np.ndarray,
             v_fov: float = V_FOV, top_band: float = TOP_BAND,
             forward: float = FORWARD):
    """Share of the top band that is sky, as a function of view pitch."""
    ahead = np.abs(yaw - 180.0) <= forward
    band = sky & ahead

    def share(view: float) -> float:
        lo = view + v_fov / 2 - top_band * v_fov
        hi = view + v_fov / 2
        sel = band & (pitch >= lo) & (pitch <= hi)
        total = ahead & (pitch >= lo) & (pitch <= hi)
        return float(sel.sum()) / max(1, int(total.sum()))

    return share


def bound_pitch(share, base: float, bound: float = BOUND,
                floor: float = FLOOR, iters: int = 30):
    """Lowest pitch in [floor, base] that meets the bound, else `floor`.

    Returns (pitch, moved). One-sided: never returns a pitch above `base`, so a
    canopy gap cannot pull the view up into the trees. When no pitch down to
    `floor` meets the bound, `floor` is returned -- the caller checks, because
    the honest answer there is "there is more sky than the bound allows and no
    amount of tilting down fixes it", not a silent pass.
    """
    if share(base) <= bound:
        return base, False
    lo, hi = floor, base
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if share(mid) > bound:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi), True
