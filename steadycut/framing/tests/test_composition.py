"""Composition corrections: the sign, the hold, and the bound's order of work."""

import numpy as np
import pytest

from steadycut.framing.composition import V_FOV, Correction, correct

FLAT = np.full(64, 0.0)                       # a per-column boundary, all columns


def boundary(frac):
    return np.full(64, frac)


def test_sign_is_a_tilt_down_when_the_ground_shows_too_little():
    """Boundary low in frame (a large fraction) means little ground showing, so
    the view must tilt DOWN. Getting this backwards would point the camera at the
    sky, which is the failure the whole policy exists to avoid."""
    c = correct([boundary(0.60)], base=0.0, smooth_s=0.0)
    want = (0.48 - 0.60) * V_FOV
    assert c.pitch[0] == pytest.approx(want)
    assert c.pitch[0] < 0.0


def test_sign_is_a_tilt_up_when_the_ground_shows_too_much():
    c = correct([boundary(0.30)], base=0.0, smooth_s=0.0)
    assert c.pitch[0] == pytest.approx((0.48 - 0.30) * V_FOV)
    assert c.pitch[0] > 0.0


def test_holds_through_a_gap_and_re_anchors_after():
    seq = [boundary(0.60), np.full(64, np.nan), np.full(64, np.nan), boundary(0.60)]
    c = correct(seq, base=0.0, smooth_s=0.0)
    assert c.held_count == 2
    assert np.isfinite(c.pitch).all()                 # never a guessed row
    assert c.pitch[1] == c.pitch[0] and c.pitch[2] == c.pitch[0]
    assert c.pitch[3] == pytest.approx(c.pitch[0])    # re-anchored, not drifted
    assert list(c.held) == [False, True, True, False]


def test_smoothing_blends_the_re_anchor():
    seq = [boundary(0.60), np.full(64, np.nan), boundary(0.30), boundary(0.30)]
    sharp = correct(seq, base=0.0, smooth_s=0.0)
    smooth = correct(seq, base=0.0, smooth_s=3.0, fps=1.0)   # 3-sample window
    assert smooth.pitch[2] != sharp.pitch[2]
    # smoothing spreads the step over neighbours, so it is the LARGEST step that
    # must shrink, not every elementwise difference
    assert abs(np.diff(smooth.pitch)).max() < abs(np.diff(sharp.pitch)).max()


def test_sky_bound_applies_last_and_only_downward():
    base = 0.0
    c = correct([boundary(0.30)], base=base, smooth_s=0.0)
    assert c.pitch[0] > base                          # the correction wants up
    bounded = correct([boundary(0.30)], base=base, smooth_s=0.0,
                      share=lambda v: 0.60)           # sky is over the bound
    assert bounded.pitch[0] <= c.pitch[0]             # the bound never adds pitch
    assert bounded.pitch[0] == -45.0                  # no downward answer exists
    assert list(bounded.acted) == [True]


def test_no_action_when_the_sky_is_already_under_the_bound():
    c = correct([boundary(0.60)], base=0.0, share=lambda v: 0.0, smooth_s=0.0)
    assert c.acted_count == 0
    assert c.pitch[0] == pytest.approx((0.48 - 0.60) * V_FOV)


def test_a_held_sample_cannot_violate_the_bound():
    """The bound is applied to the held value too, or a canopy gap would be a
    hole in the guarantee."""
    seq = [boundary(0.60), np.full(64, np.nan)]
    c = correct(seq, base=0.0, share=lambda v: 0.60, smooth_s=0.0)
    assert c.held[1]
    assert c.pitch[1] == -45.0 and c.acted[1]


def test_correction_reports_its_counts():
    c = correct([boundary(0.48)], base=-19.0, smooth_s=0.0)
    assert isinstance(c, Correction)
    assert c.held_count == 0 and c.acted_count == 0
    assert c.pitch[0] == pytest.approx(-19.0)         # already on target
