"""The labelling pass that the ground cue needs, and that nothing implements yet.

`GroundCue.fit()` takes `(image, boundary per column)` pairs. Nothing in the tree
produces the boundary half: the plan called it "the caller's business" and it was
never built, which is the whole reason the cue has never run on real footage. The
cue's own tests fit on SYNTHETIC frames with a boundary known by construction, so
the "within 0.082 of frame height" in its docstring is a synthetic number and has
never been earned on this material.

What is pinned here, in the order the tool can actually fail:

  * A boundary assembled from a few clicks is a PER-COLUMN array, because that is
    the shape `fit()` demands, and it is NaN outside the clicked span. Sparse
    labelling is not a compromise: `fit()` explicitly tolerates unlabelled
    pixels, so a hand-drawn curve over the middle of the frame is a first-class
    input, not a degraded one.
  * ONE click yields no boundary and must be refused. A single point is the shape
    that makes a fit "succeed" while learning nothing, and the difference
    between an all-NaN sample and a two-column one is the difference between an
    honest refusal and a silently wrong cue.
  * Samples survive a save/load round trip EXACTLY, because a labelling pass is
    long enough that losing work to a float format is a real cost.
  * The fitted profile records what it was fitted on and refuses a stale one. The
    profile already carries `fitted_on` with held-out error and a size/mtime
    guard, and this is the test that a cue fitted to one export cannot be quietly
    used against a re-export.
"""
import numpy as np
import pytest
from pathlib import Path

from steadycut.core.profiles import Profile
from steadycut.framing.labeller import (boundary_from_points, samples_from_npz,
                                        samples_to_npz, usable, write_profile)


def _img(h=72, w=96, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


def test_clicks_become_a_per_column_boundary_with_nan_outside() -> None:
    """A few clicks must produce one value PER COLUMN, NaN beyond the span.

    NaN is load-bearing, not padding: `fit()` skips unlabelled pixels, so a
    hand-drawn curve covering the middle of the frame is a valid input. If this
    returned a short array or filled the ends, the fit would either raise or
    quietly learn from columns nobody marked.
    """
    pts = [(20, 30.0), (50, 40.0)]
    b = boundary_from_points(pts, width=96, height=72)
    assert b.shape == (96,)
    assert np.isnan(b[:20]).all(), "columns before the first click must be NaN"
    assert np.isnan(b[51:]).all(), "columns after the last click must be NaN"
    assert not np.isnan(b[20:51]).any(), "the clicked span must be filled"
    # fractions of height, per fit()'s contract -- 30/72 and 40/72
    assert b[20] == pytest.approx(30.0 / 72.0)
    assert b[50] == pytest.approx(40.0 / 72.0)
    assert b[35] == pytest.approx(35.0 / 72.0)


def test_a_single_click_refuses_rather_than_inventing_a_boundary() -> None:
    """One point is not a boundary, and must not be dressed up as one.

    This is the shape that makes a fit report success while learning nothing: a
    flat line drawn through one point is a plausible "the ground is here" and it
    is the single most likely way for a labelling pass to produce a confidently
    wrong cue. Refusing is the honest outcome.
    """
    b = boundary_from_points([(40, 30.0)], width=96, height=72)
    assert np.isnan(b).all(), "one click must yield no boundary at all"
    assert not usable(b), "an all-NaN boundary must be rejected"


def test_boundaries_are_fractions_of_height_not_pixels() -> None:
    """`fit()` documents `boundary` as a FRACTION of height. Pixels would be a
    72x error that still fits and still returns a plausible-looking number."""
    b = boundary_from_points([(10, 36.0), (20, 36.0)], width=96, height=72)
    assert b[10] == pytest.approx(0.5)
    assert (b[10:21] <= 1.0).all() and (b[10:21] >= 0.0).all()


def test_samples_survive_a_save_load_round_trip_exactly() -> None:
    """A labelling pass is long enough that a lossy round trip is a real cost."""
    samples = [(_img(seed=i), boundary_from_points(
        [(10, 20.0), (80, 30.0)], width=96, height=72)) for i in range(3)]
    back = samples_from_npz(samples_to_npz(samples, path="/tmp/_lab_test.npz"))
    assert len(back) == len(samples)
    for (a_img, a_b), (b_img, b_b) in zip(samples, back):
        assert np.array_equal(a_img, b_img)
        np.testing.assert_array_equal(np.isnan(a_b), np.isnan(b_b))   # NaN-safe
        np.testing.assert_allclose(np.nan_to_num(a_b), np.nan_to_num(b_b),
                                   rtol=0, atol=0)


def test_a_boundary_outside_the_frame_is_rejected() -> None:
    """A click above the top edge is a mis-click, not a boundary at -0.4.

    Clamping it into frame would invent a label the person never made, and the
    fit has no way to know.
    """
    with pytest.raises(ValueError):
        boundary_from_points([(10, -5.0), (20, 10.0)], width=96, height=72)
    with pytest.raises(ValueError):
        boundary_from_points([(10, 10.0), (20, 80.0)], width=96, height=72)


def test_a_cue_fitted_on_one_export_is_refused_against_a_re_export() -> None:
    """The profile's own staleness guard, exercised end to end.

    Re-exporting a video changes its size and mtime. A cue fitted on the old
    export must not be applied to the new one without a refit, because the
    features it learned thresholds for were that export's. `Profile.matches()`
    already implements this, so the test pins that the labeller RECORDS the
    source rather than inventing a second mechanism.
    """
    src = Path("/tmp/_lab_guard_src.mp4")
    src.write_bytes(b"first export")
    prof = Profile().record_source(src)
    assert prof.matches(src) is True, "a freshly recorded source must match"

    # a re-export changes the bytes, so size and mtime both move
    src.write_bytes(b"second export, longer than the first")
    assert prof.matches(src) is False, (
        "a re-export must not silently reuse a cue fitted on the old one")

    # and an unverified profile claims nothing at all
    unverified = Profile()
    assert unverified.verified is False
    assert unverified.matches(src) is True, (
        "no recorded claim means 'unchecked', not 'matches' -- these are "
        "different and a caller has to be able to tell them apart")


def test_write_profile_records_the_source_so_the_guard_is_live() -> None:
    """Behaviour, not a grep: write a profile, then re-export the source.

    The first version of this asserted the string "record_source" appeared in
    the module. Replacing the CALL with `pass` left that test green, because the
    word also appears in the docstring explaining why the call matters -- a grep
    cannot tell a call from a comment about a call. This one checks the guard
    actually refuses afterwards, which is the property anyone relies on.
    """
    src = Path("/tmp/_lab_live_src.mp4")
    src.write_bytes(b"export one")
    prof_path = Path("/tmp/_lab_live_profile.json")
    if prof_path.exists():
        prof_path.unlink()

    # A real cue-shaped object, not `object()`: write_profile serialises through
    # Profile.to_dict(), which reads weights/mean/scale. A bare object fails
    # there, which is the honest shape of the contract rather than a missing
    # check -- an unfit cue cannot be written.
    cue = type("Cue", (), {"weights": np.zeros(6), "mean": np.zeros(5),
                           "scale": np.ones(5)})()
    write_profile(prof_path, cue=cue, report={"frames": 4}, source=src)
    prof = Profile.from_dict(__import__("json").loads(prof_path.read_text()))
    assert prof.matches(src) is True, (
        "a cue written without recording its source is reused against any "
        "re-export, which is exactly the silent-reuse this guard exists to stop")

    src.write_bytes(b"export two, different bytes")
    assert prof.matches(src) is False, (
        "a re-export must not match a profile fitted on the previous export")



def test_a_nan_held_out_error_refuses_storage() -> None:
    """The Oct-1 shape must never reach disk again.

    fit_samples reports NaN when the cue predicts nothing finite on the
    held-out frames; store_refusal turns that report into the reason the
    --fit path prints before exiting 2.
    """
    from steadycut.framing.labeller import store_refusal
    bad = {"frames": 7, "train": 6, "held_out": 1,
           "held_out_error": float("nan")}
    reason = store_refusal(bad)
    assert reason is not None and "NaN" in reason
    good = {"frames": 19, "train": 15, "held_out": 4,
            "held_out_error": 0.0197}
    assert store_refusal(good) is None


def test_proposals_draw_green_for_trusted_yellow_for_review() -> None:
    """Trust colors the proposal; the rider still presses the key.

    Green = triple agreement (trace quickly), yellow = inspect first.
    No proposal drawn where the model has nothing finite.
    """
    import numpy as np
    from steadycut.framing.labeller import draw_proposals
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    good = np.full(200, 0.4)
    hook = draw_proposals({3: (good, True), 7: (good, False)})
    assert hook(9, frame, [], 200, 100) is frame  # no proposal: untouched
    g = hook(3, frame, [], 200, 100)
    assert (g[40, ::4] == (0, 200, 0)).all()  # BGR green dots on the line
    y = hook(7, frame, [], 200, 100)
    assert (y[40, ::4] == (0, 215, 255)).all()  # BGR yellow dots
