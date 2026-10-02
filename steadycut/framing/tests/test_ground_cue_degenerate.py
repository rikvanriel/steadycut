"""A cue with no boundary in the frame must SAY SO, not answer from the prior.

Measured on real footage: on 0519a the cue's boundary moves across a 21 degree
ladder (0.286 to 0.348) because there IS a ground/vegetation boundary in the
frame. On 0519b it returned 0.279 at every rung -- the same number to three
decimals -- and looking at the frames explains why: the camera is pointed at the
rider's own body rather than the trail, so there is no ground boundary anywhere
in view. The rider confirmed this by eye.

A constant answer is not a small defect. It is a confident number with nothing
behind it, which is the failure this project has paid for repeatedly: the
262-byte empty render read as stale content, the retracted shading diagnosis
named a cause the data did not support, and `bank_measured: false` exists
because a plausible number outlives the thing that produced it.

THE TEST THIS ENCODES. The cue's features include `row`, a pure vertical ramp
that is IDENTICAL in every frame -- a geometry prior, not evidence. If the only
thing separating ground from not-ground is that ramp, the probability is a
function of row alone and the prediction is the prior's, not the picture's. So
the discriminator is how much the probability varies ACROSS COLUMNS at a fixed
row: image content makes it vary, a prior makes it flat. No threshold on the
answer itself can tell those apart, because a prior-driven answer looks exactly
like a perfectly level horizon.

A real boundary is not required to vary either -- a dead-flat horizon is
legitimate -- so the primary test is the probability spread, and the
constant-answer test is a second, weaker guard.
"""
import numpy as np
import pytest

from steadycut.framing.ground_cue import GroundCue


def _cue():
    """A cue that uses BOTH the image and the row prior, as a fitted one does.

    Two things matter here and the first version of this fixture had neither.
    A zero-weight cue makes the full answer and the prior-only answer identical
    BY CONSTRUCTION, so it would call every frame degenerate. And a cue with no
    weight on `row` produces NO boundary from the prior alone, so the comparison
    has nothing to compare against and every frame is again degenerate. A real
    fitted cue weights both -- the module's own note calls `row` the strongest
    single feature available -- so the fixture does too.
    """
    return GroundCue(weights=np.array([3.0, 0.0, 0.0, 0.0, 6.0, -1.0]),
                     mean=np.zeros(5), scale=np.ones(5))


def _img(half, h=64, w=80):
    """Top `half` rows one colour, bottom rows another: a real boundary."""
    f = np.zeros((h, w, 3), np.uint8)
    f[:half] = (30, 90, 40)
    f[half:] = (120, 110, 100)
    return f


def test_a_frame_with_a_real_boundary_is_not_degenerate() -> None:
    assert _cue().is_degenerate(_img(24)) is None


def test_a_frame_with_no_boundary_is_degenerate() -> None:
    """One flat colour: nothing but the row prior separates the pixels.

    This is the 0519b case -- the rider's own body fills the view -- and with
    zero weights the probability is a constant per row, so nothing in the image
    is doing any work.
    """
    why = _cue().is_degenerate(np.full((64, 80, 3), 90, np.uint8))
    assert why is not None, "a frame with no structure must be reported as such"
    assert "boundary" in why.lower() or "structure" in why.lower()


def test_the_reason_names_the_cause_a_rider_can_act_on() -> None:
    """The message is the product. "no boundary" is only useful if it says what
    to do about it, and what the rider actually needs to hear is that the camera
    may be pointed at the wrong thing."""
    why = _cue().is_degenerate(np.full((64, 80, 3), 90, np.uint8))
    low = why.lower()
    assert "trail" in low or "pointed" in low, (
        f"the reason does not say what to look at: {why!r}")


def test_boundary_or_none_gives_none_rather_than_a_confident_number() -> None:
    cue = _cue()
    assert cue.boundary_or_none(_img(24)) is not None
    assert cue.boundary_or_none(np.full((64, 80, 3), 90, np.uint8)) is None


def test_a_flat_prediction_over_a_real_boundary_is_still_honest() -> None:
    """A smooth sky over a level horizon is UNIFORM ACROSS COLUMNS, and that is
    not a fault.

    An earlier version of this check refused any frame whose boundary came out
    the same in every column. That fired here, on a frame that plainly has a
    boundary, and it would have refused honest footage -- a still lake horizon
    or a fog bank is exactly as uniform as a broken cue. The guard was removed
    rather than loosened: a cue that cancels the image out is already caught by
    the prior comparison, so the second guard was buying nothing but false
    refusals.
    """
    cue = _cue()
    assert cue.is_degenerate(_img(24)) is None, (
        "a level boundary under a smooth sky must not be refused")


def test_degeneracy_is_not_a_caller_setting_it_can_forget() -> None:
    """The default must be the SAFE one: a caller that never asks still gets
    the honest answer, because the unsafe path is the one that ships by
    accident."""
    import inspect

    sig = inspect.signature(_cue().is_degenerate)
    assert "img" in sig.parameters
    assert _cue().is_degenerate(_img(24)) is None
