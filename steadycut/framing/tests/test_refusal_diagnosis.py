"""A refusal must say WHICH problem it is, because the two need different fixes.

The sweep refuses on about half its windows, and the refusal used to be the same
sentence either way. Measured on real footage the two cases are different
problems with different answers: an unmeasurable criterion (canopy, or the
subject out of frame) is fixed by looking, while an unreachable target in a
SHADED window is fixed by nothing the ladder can do -- band brightness separates
the two cases 7 of 7 -- and a user told only "did not bracket" has no way to act
on it.

So these tests pin the discrimination, and the control is the part that matters:
a BRIGHT window that fails to bracket must still get the generic reason, or the
diagnosis is just a more confident way of being wrong.
"""
import numpy as np
import pytest

from steadycut.framing.ladder import (SHADED_BAND_MEDIAN, _band_median,
                                     brackets, diagnose_refusal,
                                     solve_pitch)
from steadycut.framing.policies import PitchSearch


def _stack(level: int, n: int = 3, h: int = 72, w: int = 96) -> np.ndarray:
    return np.full((n, h, w), level, dtype=np.uint8)


def _search(**kw):
    """A PitchSearch carrying the band, since the diagnosis reads it."""
    return PitchSearch(criterion=lambda frames: 0.5, target=0.90, label="x",
                      ladder=(-5.0,), **kw)


def test_a_dark_band_is_diagnosed_as_shading() -> None:
    """The measured case: a band at 89 (failing-window mean) names the cause."""
    why = diagnose_refusal(_search(x_band=(0.35, 0.65)),
                           {-5.0: _stack(89)})
    assert why is not None, "a dark band must be recognised as shaded"
    assert "dark" in why
    assert "wider pitch ladder will not reach" in why


def test_a_bright_band_gets_no_shading_claim() -> None:
    """The CONTROL. Brighter than the threshold must fall through to generic.

    Without this, the diagnosis would fire on every refusal and be a more
    confident way of being wrong -- the failure mode this whole project keeps
    meeting.
    """
    s = _search(x_band=(0.35, 0.65))
    assert diagnose_refusal(s, {-5.0: _stack(140)}) is None
    assert diagnose_refusal(s, {-5.0: _stack(SHADED_BAND_MEDIAN)}) is None


def test_the_band_matters_and_the_whole_frame_does_not() -> None:
    """Measured 0519/1200: band median 87, whole-frame median 127.

    A whole-frame median sits well ABOVE the threshold on exactly the shaded
    windows the diagnosis exists to catch, because the forest fills most of the
    frame.  So measuring the whole frame would silence the diagnosis on the very
    case it is for.
    """
    frame = np.full((1, 40, 40), 140, np.uint8)     # bright forest everywhere
    frame[:, :, 14:26] = 70                          # dark band in the middle
    # No band means no reading at all -- a whole-frame fallback would sit at 140
    # here, above the threshold, and miss the dark window it exists to catch.
    assert np.isnan(_band_median(frame, None))
    assert _band_median(frame, (0.0, 1.0)) == pytest.approx(140.0)
    assert _band_median(frame, (0.35, 0.65)) == pytest.approx(70.0)
    s = _search(x_band=(0.35, 0.65))
    assert diagnose_refusal(s, {-5.0: frame}) is not None


def test_a_policy_without_a_band_declines_rather_than_guessing() -> None:
    """No band declared means no band reading, not a whole-frame fallback."""
    assert diagnose_refusal(_search(), {-5.0: _stack(40)}) is None
    assert np.isnan(_band_median(_stack(40), None))


def test_nothing_to_measure_is_not_shading() -> None:
    """An empty sweep is missing data, not a dark window."""
    assert diagnose_refusal(_search(x_band=(0.35, 0.65)), {}) is None


def test_band_median_reads_the_stack() -> None:
    assert _band_median(_stack(80), (0.0, 1.0)) == pytest.approx(80.0)
    assert np.isnan(_band_median(None, (0.0, 1.0)))


def test_the_diagnosis_does_not_license_a_pitch() -> None:
    """A diagnosis is not a solution: the sweep still refuses.

    This is the invariant that holds whatever the cause turns out to be. A
    shaded window genuinely has no solution here, and returning a pitch anyway
    would be the defect the whole refusal guard exists to prevent.
    """
    edges = {-5.0: 0.66, -12.0: 0.59, -19.0: 0.53, -26.0: 0.47}
    assert not brackets(edges, 0.90)
    # solve_pitch clamps to an end when the target is out of range; the caller
    # is what must check. Confirm the clamp is visible rather than silent.
    clamped = solve_pitch(list(edges), list(edges.values()), 0.90)
    assert clamped in edges


def test_diagnosis_is_a_string_the_user_can_act_on() -> None:
    """The message must carry the threshold it compared against.

    A reason that says "it is dark" without saying how dark invites an argument
    about the word "dark" instead of a look at the window.
    """
    why = diagnose_refusal(_search(x_band=(0.35, 0.65)), {-5.0: _stack(60)})
    assert why is not None
    assert str(int(SHADED_BAND_MEDIAN)) in why
    assert "by eye" in why, "a refusal should say what the rider can do about it"
