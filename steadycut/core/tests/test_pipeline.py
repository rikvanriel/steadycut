"""Core pipeline tests: composition logic on synthetic data, no footage."""
import math

import numpy as np

from steadycut.core.pipeline import (
    ClipSpec, SyncResult, Traj, accumulate, align_corrections, bounce_of_traj,
    correlate, gsmooth, plate_scale, residual_correction,
)


def test_plate_scale_matches_measured_values() -> None:
    # 10.67/9.16 at h_fov 120 were measured independently before this helper
    # existed; the helper must reproduce them, not redefine them.
    ppx, ppdy = plate_scale(120.0)
    assert abs(ppx - 10.67) < 0.02
    assert abs(ppdy - 9.16) < 0.02


def test_gsmooth_preserves_constant_and_centers_step() -> None:
    c = np.ones(40)
    assert np.allclose(gsmooth(c, 4.0), 1.0)
    s = np.concatenate([np.zeros(20), np.ones(20)])
    sm = gsmooth(s, 2.0)
    assert abs(sm[19] - 0.5) < 0.15  # symmetric: midpoint stays midpoint


def test_correlate_finds_lag_and_ignores_nan() -> None:
    t = np.arange(200) / 10.0
    a = np.sin(t)
    b = np.sin(t - 0.8)  # same signal, delayed
    assert correlate(a, b) < 0.95  # misaligned reads low...
    b2 = np.sin(t)
    assert correlate(a, b2) > 0.999  # ...aligned reads ~1
    c = b.copy()
    c[:200:2] = np.nan
    assert abs(correlate(a, c) - correlate(a, b)) < 0.05  # missing ~= same value
    assert math.isnan(correlate(np.ones(20), np.ones(20)))  # flat is NaN


def test_accumulate_and_bounce() -> None:
    d = np.zeros((10, 3))
    d[:, 1] = 2.0
    tr = Traj(deltas=d, inliers=np.full(10, 50))
    pos = accumulate(tr)
    assert pos.shape == (11, 3) and pos[0, 1] == 0.0 and pos[-1, 1] == 20.0
    bounce, mean = bounce_of_traj(tr)
    assert bounce == 0.0 and mean == 2.0  # constant motion: no bounce


def test_residual_correction_honours_bound() -> None:
    d = np.zeros((60, 3))
    d[:, 1] = 30.0  # huge drift the smoother keeps
    tr = Traj(deltas=d, inliers=np.full(60, 100))
    corr = residual_correction(tr, sigma=8.0, bound=12.0)
    assert corr.shape[0] == 61  # one per frame
    assert np.abs(corr[:, :2]).max() <= 12.0 + 1e-9


def test_clip_spec_defaults_match_production() -> None:
    s = ClipSpec(source="x.insv", start=0.0, duration=1.0)
    assert s.framing_pitch == -12.0 and s.fov == 120.0
    assert isinstance(SyncResult(lag_ms=-80, r_best=-0.9, r_at_zero=0.4),
                      SyncResult)


def test_align_corrections_preserves_the_index_pairing() -> None:
    """Row j belongs to frame j: a surplus frame is handled at the TAIL only.

    Padding or trimming at the head would shift every correction one frame
    late against its frame, and a correction one frame late injects motion
    instead of removing it -- which is the failure this test is here to catch.
    The trajectory comes from cv2 and the pixels from ffmpeg, and the two
    disagree by one frame on some files, so one-frame slack is expected.
    """
    corr = np.arange(15, dtype=float).reshape(5, 3)
    assert np.array_equal(align_corrections(corr, 5), corr)

    padded = align_corrections(corr, 6)          # one frame more than rows
    assert padded.shape == (6, 3)
    assert np.array_equal(padded[:5], corr), "the head must not shift"
    assert padded[5].tolist() == [0.0, 0.0, 0.0]

    trimmed = align_corrections(corr, 4)         # one row more than frames
    assert trimmed.shape == (4, 3)
    assert np.array_equal(trimmed, corr[:4]), "the head must not shift"


def test_align_corrections_refuses_a_real_mismatch() -> None:
    import pytest
    with pytest.raises(ValueError, match="corrections"):
        align_corrections(np.zeros((4, 3)), 9)


def test_far_field_score_reports_rms_and_jitter_at_render_scale(monkeypatch) -> None:
    """Two numbers, and the tracker-to-render scale conversion, on a known
    series.

    A correction can remove sway without removing jitter or the other way
    round, so the certificate has to report them separately: rms is the spread
    of the per-frame far-field motion, jitter the spread of its frame-to-frame
    difference.
    """
    import pytest
    from steadycut.core import pipeline as P
    deltas = np.zeros((5, 3))
    deltas[:, 1] = [0.0, 100.0, 0.0, 100.0, 0.0]
    traj = Traj(deltas=deltas, inliers=np.full(5, 500))
    monkeypatch.setattr(P, "track_far", lambda video: traj)

    score = P.far_field_score("whatever.mp4", render_size=(960, 720))

    dy = deltas[:, 1] * (720.0 / 960.0)
    assert score["rms_px"] == pytest.approx(float(np.std(dy)))
    assert score["jitter_px"] == pytest.approx(float(np.std(np.diff(dy))))
    assert score["frames"] == 5


def test_the_2d_defaults_are_the_validated_ones() -> None:
    """sigma and bound are measurements; pin them where they live.

    sigma 4.0 won the far-field amplitude sweep (against 8.0 and 2.0) and
    bound 12 px is the honesty limit the warp is clipped to. Both were chosen
    by measuring, so a change should be a re-measurement rather than a nudge.
    """
    import inspect
    from steadycut.core import pipeline as P
    hy = inspect.signature(P.hybrid_correct).parameters
    assert hy["sigma"].default == 4.0
    assert hy["bound"].default == 12.0
    assert hy["zoom"].default == 1.05
    assert inspect.signature(
        P.residual_correction).parameters["sigma"].default == 4.0
