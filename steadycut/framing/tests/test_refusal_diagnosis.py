"""The refusal must report what it MEASURED, and must not name a cause it cannot support.

Two things are pinned here, and the second is the one that matters.

First, a refusal still carries the criterion, the measured range and the target.
That is what a reader can act on.

Second, there is no band-brightness diagnosis, and that is deliberate. One was
built and measured on seven windows, where it separated bracketing from refusing
windows 7 of 7, and it was shipped. Widening the sample to seventeen windows
across three rides showed the classes overlap by seventeen grey levels:

    bracketing: 8 windows, medians 80 90 96 96 98 104 108 115
    refusing:   9 windows, medians 64 66 68 73 78 84 86 88 97

A bracketing window at 80 sits below a refusing one at 97, so no threshold
separates them. The shipped value scored 12 of 17 and the best available
threshold reached 15 of 17. The seven-window sample looked clean because none of
the seven sat near the boundary, which is precisely when a small sample is most
convincing.

The guard below is the point of the file: a threshold must not come back without
someone having measured the margin around it. That is the question this project
keeps not asking, and it is cheaper to encode than to re-learn.
"""
import numpy as np
import pytest

from steadycut.framing.ladder import (SHADED_BAND_MEDIAN, _band_median,
                                     brackets, solve_pitch)
from steadycut.framing.policies import PitchSearch


def _stack(level: int, n: int = 3, h: int = 72, w: int = 96) -> np.ndarray:
    return np.full((n, h, w), level, dtype=np.uint8)


# The measured overlap, kept here so the numbers travel with the guard.
BRACKETING_MEDIANS = [80, 90, 96, 96, 98, 104, 108, 115]
REFUSING_MEDIANS = [64, 66, 68, 73, 78, 84, 86, 88, 97]


def test_the_measured_classes_overlap_so_no_threshold_is_shippable() -> None:
    """The measurement that retracted the diagnosis, asserted so it cannot drift.

    If a future change makes these classes separable, this test is the one to
    update deliberately -- and it should be updated with a MARGIN in hand, not a
    hit count over the easy cases.
    """
    worst_bracketing = min(BRACKETING_MEDIANS)
    worst_refusing = max(REFUSING_MEDIANS)
    assert worst_refusing > worst_bracketing, (
        "the classes no longer overlap; a refusal diagnosis may be viable "
        "again, but only with the margin measured on windows near the boundary")


def test_a_band_brightness_diagnosis_does_not_exist() -> None:
    """No code path may name shading as the cause of a refusal."""
    from steadycut.framing import ladder

    assert not hasattr(ladder, "diagnose_refusal"), (
        "a refusal diagnosis was re-added; measure the margin around any "
        "threshold on windows near the boundary before shipping it")


def test_a_refusal_still_carries_its_numbers() -> None:
    """The measured range and the target are what a reader can act on."""
    edges = {-5.0: 0.66, -12.0: 0.59, -19.0: 0.53, -26.0: 0.47}
    assert not brackets(edges, 0.90)
    # solve_pitch clamps to an end when the target is out of range; the caller
    # is what must check. Confirm the clamp is visible rather than silent.
    assert solve_pitch(list(edges), list(edges.values()), 0.90) in edges


def test_band_median_reads_the_policy_band_not_the_whole_frame() -> None:
    """Kept because where to measure is settled, even though the cut-off is not.

    Measured 0519/1200: band median 87 against whole-frame 127, since forest
    fills the frame. This is the instrument a future diagnosis would need.
    """
    frame = np.full((1, 40, 40), 140, np.uint8)     # bright forest everywhere
    frame[:, :, 14:26] = 70                          # dark band in the middle
    assert np.isnan(_band_median(frame, None))      # no band, no reading
    assert _band_median(frame, (0.0, 1.0)) == pytest.approx(140.0)
    assert _band_median(frame, (0.35, 0.65)) == pytest.approx(70.0)
    assert np.isnan(_band_median(None, (0.0, 1.0)))
    assert _band_median(_stack(80), (0.0, 1.0)) == pytest.approx(80.0)


def test_a_policy_may_declare_the_band_it_scans() -> None:
    """The band is policy data, and a policy that declares none is still valid."""
    s = PitchSearch(criterion=lambda f: 0.5, target=0.9, label="x")
    assert s.x_band is None
    s2 = PitchSearch(criterion=lambda f: 0.5, target=0.9, label="x",
                     x_band=(0.35, 0.65))
    assert s2.x_band == (0.35, 0.65)


def test_the_retracted_threshold_value_is_still_recorded() -> None:
    """So the number quoted in the source comment can be checked against the sample."""
    assert SHADED_BAND_MEDIAN == 96.0
    assert SHADED_BAND_MEDIAN in BRACKETING_MEDIANS


def test_a_refusal_does_not_license_a_pitch() -> None:
    """The invariant that holds whatever the cause turns out to be.

    A window with no solution here must still refuse; a sweep that returned a
    pitch because it had an explanation would be the failure the guard exists to
    prevent.
    """
    assert not brackets({-5.0: 0.5, -12.0: 0.4}, 0.90)
    assert not brackets({}, 0.90)
    assert np.isnan(solve_pitch([], [], 0.90))
