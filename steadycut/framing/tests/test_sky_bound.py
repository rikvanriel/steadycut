"""The sky bound must be one-sided, and must not be fooled by a canopy gap.

The failure this exists to prevent is measured: treating the sky share as an
OBJECTIVE ("reach 25% sky") is satisfied by looking up into a gap in the leaves,
which drove the view up the trees and lost the bike. So the invariant that
matters is not "the share is low" -- it is "the correction never tilts UP".
A test that only checked the share would pass on the very behaviour that failed.
"""

import numpy as np
import pytest

from steadycut.framing.sky_bound import bound_pitch, grid, share_fn, sky_mask

W, H = 360, 180        # 1 degree per pixel: pitch from +90 (up) to -90 (down)


def mask_from(pitch_lo, pitch_hi, w=W, h=H):
    """All-sky mask between two pitches, nothing elsewhere."""
    pitch, _ = grid(w, h)
    return (pitch >= pitch_lo) & (pitch <= pitch_hi)


def test_share_falls_as_the_view_drops():
    """The binary search assumes this. With sky only high up, tilting down can
    only see less of it."""
    pitch, yaw = grid(W, H)
    share = share_fn(mask_from(30.0, 90.0), pitch, yaw)
    shares = [share(v) for v in range(30, -46, -5)]
    assert all(a >= b for a, b in zip(shares, shares[1:])), shares
    assert shares[0] > 0.1 and shares[-1] == 0.0


def test_does_nothing_when_already_under_the_bound():
    pitch, yaw = grid(W, H)
    share = share_fn(mask_from(-90.0, -70.0), pitch, yaw)   # sky far below
    p, moved = bound_pitch(share, base=0.0, bound=0.25)
    assert moved is False and p == 0.0


def test_tilts_down_to_meet_the_bound():
    pitch, yaw = grid(W, H)
    share = share_fn(mask_from(20.0, 90.0), pitch, yaw)      # sky high up only
    assert share(0.0) > 0.25
    p, moved = bound_pitch(share, base=0.0, bound=0.25)
    assert moved is True
    assert p < 0.0
    assert share(p) <= 0.25


@pytest.mark.parametrize("case", [
    mask_from(0.0, 90.0),        # sky above
    mask_from(-90.0, 90.0),      # all sky: no downward answer exists
    mask_from(45.0, 60.0),       # a band of sky, like a canopy gap
    mask_from(-20.0, 5.0),       # sky right where the view is looking
])
def test_never_tilts_up(case):
    """The invariant, over masks that include the no-solution case: a bound that
    cannot be met by looking down must return the floor, never a pitch above the
    base where the view would climb into the trees."""
    pitch, yaw = grid(W, H)
    share = share_fn(case, pitch, yaw)
    for base in (0.0, -19.0, 20.0, -40.0):
        p, moved = bound_pitch(share, base=base)
        assert p <= base, f"tilted up: base {base}, returned {p}"
        if moved:
            assert p >= -45.0


def test_sky_mask_is_blue_dominance():
    img = np.zeros((4, 4, 3), np.uint8)          # BGR
    img[0, :] = (200, 180, 120)                  # blue sky
    img[1, :] = (60, 90, 120)                    # foliage
    m = sky_mask(img)
    assert m[0].all() and not m[1].any()
