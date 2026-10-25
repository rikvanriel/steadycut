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


def test_correlate_weights_mute_a_loud_lying_stretch() -> None:
    # First half: y matches x (few inliers). Second half: y is loud and
    # unrelated (many samples, zero inliers -- near-field mush the matcher
    # could not lock). Unweighted lets the loud half vote; weighted mutes
    # it and reads the honest half.
    t = np.arange(200) / 10.0
    a = np.sin(t)
    y = np.sin(t)
    y[100:] = 10 * np.sin(3 * t[100:] + 1.0)
    w = np.ones(200)
    w[100:] = 0
    assert correlate(a, y) < 0.9
    assert correlate(a, y, w=w) > 0.999
    assert math.isnan(correlate(a, y, w=np.zeros(200)))  # nothing votes: NaN


def test_correlate_none_weights_are_the_old_path() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=50)
    b = 2 * a + 1
    assert correlate(a, b) == correlate(a, b, w=None)


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


def _osc_traj(amp: float, hz: float = 4.0, n: int = 120) -> Traj:
    # Velocity sine at 30 fps; the smoothed residual is the position sine,
    # so the correction oscillates at ~amp once the mean is removed.
    t = np.arange(n) / 30.0
    d = np.zeros((n, 3))
    d[:, 1] = amp * np.sin(2.0 * math.pi * hz * t)
    return Traj(deltas=d, inliers=np.full(n, 100))


def test_residual_correction_never_rail_bangs() -> None:
    # A +/-25 px oscillation under a 12 px bound: the old hard clip
    # squared it into rail-to-rail teleports of exactly 2*bound per frame
    # (measured: bound-12 output 44.8 px on a 21.2 px input). The rescale
    # must keep every frame-to-frame step well under the teleport size.
    # sigma=8.0 follows almost nothing at 3 Hz, so the correction spans
    # the full oscillation and really exercises the bound (asserted).
    tr = _osc_traj(25.0)
    pos = accumulate(tr)
    raw = np.stack([gsmooth(pos[:, i], 8.0) for i in range(3)], axis=1) - pos
    assert np.abs(raw[:, 1]).max() > 12.0  # bound actually engaged
    corr = residual_correction(tr, sigma=8.0, bound=12.0)
    assert np.abs(corr[:, :2]).max() <= 12.0 + 1e-9
    # A rail-to-rail teleport is 2*bound per frame; the rescaled series
    # must stay under half of that (the old code steps the full 24 here).
    assert np.abs(np.diff(corr[:, 1])).max() < 12.0


def test_residual_correction_identity_when_clean() -> None:
    # A +/-5 px oscillation under a 12 px bound needs no bounding at all:
    # the output must equal the unclipped smoothed residual bit for bit.
    tr = _osc_traj(5.0)
    corr = residual_correction(tr, sigma=0.2, bound=12.0)
    pos = accumulate(tr)
    sm = np.stack([gsmooth(pos[:, i], 0.2) for i in range(3)], axis=1)
    want = (sm - pos) * np.array([960.0 / 1280.0, 720.0 / 960.0, 1.0])
    np.testing.assert_allclose(corr, want, rtol=1e-12)


def test_residual_correction_spikes_keep_hard_clip() -> None:
    # Rare spikes (1 of 120 frames past the bound) take the hard-clip
    # path: rescaling the whole series for an isolated kick would weaken
    # the 119 good frames to accommodate one bad one.
    d = np.zeros((120, 3))
    t = np.arange(120) / 30.0
    # Base ripple small in POSITION (vel 2.0 at 1 Hz ~= +/-9.5 px, under
    # the bound after the 0.75 render scaling); only the kick exceeds it.
    d[:, 1] = 2.0 * np.sin(2.0 * math.pi * 1.0 * t)
    d[60, 1] += 40.0
    d[61, 1] -= 40.0  # out-and-back kick: a transient corr spike
    tr = Traj(deltas=d, inliers=np.full(120, 100))
    corr = residual_correction(tr, sigma=8.0, bound=12.0)
    pos = accumulate(tr)
    sm = np.stack([gsmooth(pos[:, i], 8.0) for i in range(3)], axis=1)
    want = (sm - pos) * np.array([960.0 / 1280.0, 720.0 / 960.0, 1.0])
    np.testing.assert_allclose(corr[:, :2], np.clip(want[:, :2], -12.0, 12.0),
                               rtol=1e-12, atol=1e-12)


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

    Re-measured on real trail footage, which overturned the previous value. Nine
    windows, three rides, at 35/50/65 percent of each: the corrected-over-raw
    far-field ratio was 0.83 at sigma 1, 0.86 at sigma 2, 0.95 at sigma 4, and
    1.37 at the sigma 8 that image-only mode shipped. The sigma 1 setting helped
    on 6 of 6 windows held out of the tuning set.

    The old value came from a synthetic ten-frame jitter sweep. On a trail the
    dominant tracked motion is partly the rider's own machine and swaying
    foliage, so smoothing it hard warps the frame toward a camera path that does
    not exist. A change should still be a re-measurement rather than a nudge.
    """
    import inspect
    from steadycut.core import pipeline as P
    hy = inspect.signature(P.hybrid_correct).parameters
    assert hy["sigma"].default == 1.0
    assert hy["bound"].default == 12.0
    assert hy["zoom"].default == 1.05
    assert inspect.signature(
        P.residual_correction).parameters["sigma"].default == 4.0
