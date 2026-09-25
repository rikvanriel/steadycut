"""Core pipeline tests: composition logic on synthetic data, no footage."""
import math

import numpy as np

from steadycut.core.pipeline import (
    ClipSpec, SyncResult, Traj, accumulate, bounce_of_traj, correlate,
    gsmooth, plate_scale, residual_correction,
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
