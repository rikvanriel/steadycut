"""Synthetic known-answer tests for aim_hold (C6 + C7).

Gate numbers are pinned here BEFORE any bench runs and must never
be retuned after seeing footage content.
"""
import numpy as np

from steadycut.stabilization import aim_hold as A


def _sine(n, fps, hz, amp):
    t = np.arange(n) / fps
    return amp * np.sin(2 * np.pi * hz * t)


def test_pan_followed_jitter_removed():
    # (i) slow pan + fast jitter in the reference: offset must
    # follow the pan (lag < gate) while the mechanism passes jitter
    # through to the correction (offset itself is slow by design;
    # the jitter stays in the gyro path, which we do not touch).
    fps, n = 30.0, 300
    pan = np.linspace(0, 20, n)  # 20 deg over 10 s
    ref = pan + _sine(n, fps, 3.0, 2.0)
    off = A.hold_offset(ref, fps)
    # Offset counter-pans (correlation strongly negative).
    r = float(np.corrcoef(off, pan)[0, 1])
    assert r < -0.95, r
    # No jump above the rate limit anywhere.
    assert float(np.max(np.abs(np.diff(off))) * fps) <= A.RATE_LIMIT + 1e-9


def test_random_walk_refuses():
    # (ii) tracker random-walk injection must refuse: a region whose
    # motion is uncorrelated with the gyro fails consistency.
    rng = np.random.default_rng(0)
    fps, n = 30.0, 300
    walk = np.cumsum(rng.normal(0, 3.0, n))  # px random walk
    gyro = _sine(n, fps, 1.0, 5.0)  # px/s predicted
    ok, r, scale, reason = A.gyro_consistency(
        np.diff(walk, prepend=walk[0]) * fps, np.zeros(n), gyro,
        np.zeros(n))
    assert not ok, (r, scale, reason)


def test_camera_locked_overlay_not_selected():
    # (iii) textured camera-locked overlay: strong salience and
    # inliers, but ~0 raw motion -> consistency scale ~0 -> fail.
    fps, n = 30.0, 300
    rng = np.random.default_rng(1)
    tracked = np.full(n, 0.5) + rng.normal(0, 0.01, n)  # ~static
    gyro = _sine(n, fps, 1.0, 5.0)
    ok, r, scale, reason = A.gyro_consistency(
        tracked, np.zeros(n), gyro, np.zeros(n))
    assert not ok, (r, scale, reason)  # empty or uncorrelated: refused
    # And selection skips it even with great inliers/salience.
    regions = [{"inliers": 2000, "lifetime": 300, "salience": 50.0,
                "resid": 5.0, "consist_ok": False}]
    idx, reason = A.select_reference(regions)
    assert idx is None, (idx, reason)


def test_moving_blob_not_selected():
    # (iv) salient independently-moving blob over weak background:
    # blob fails consistency (uncorrelated), background fails
    # salience -> run refuses.
    fps, n = 30.0, 300
    blob = _sine(n, fps, 0.7, 8.0)
    gyro = _sine(n, fps, 1.3, 5.0)
    ok, _, _, _ = A.gyro_consistency(blob, np.zeros(n), gyro,
                                      np.zeros(n))
    assert not ok
    regions = [
        {"inliers": 900, "lifetime": 300, "salience": 60.0,
         "resid": 6.0, "consist_ok": False},   # the blob
        {"inliers": 400, "lifetime": 300, "salience": 2.0,
         "resid": 0.4, "consist_ok": True},    # weak background
    ]
    idx, reason = A.select_reference(regions)
    assert idx is None, (idx, reason)


def test_long_turn_settles():
    # (v) 90+ deg turn saturates, then settles with no jump above
    # the rate limit.
    fps, n = 30.0, 600
    ref = np.linspace(0, 120, n)  # 120 deg over 20 s
    off = A.hold_offset(ref, fps)
    assert float(np.max(np.abs(off))) <= A.OFFSET_BOUND + 1e-9
    assert float(np.max(np.abs(np.diff(off))) * fps) <= A.RATE_LIMIT + 1e-9
    st = A.settle_time(off, fps)
    assert 0 <= st <= n


def test_refusal_is_bit_identical():
    # (vi) refusal path contributes exactly zero offset.
    fps, n = 30.0, 100
    off = A.hold_offset(np.zeros(n), fps)
    assert bool((off == 0.0).all())


def test_gates_pinned():
    # C7: the numbers below are the contract. Changing any of them
    # requires a plan amendment, not a quiet edit.
    assert (A.INLIER_FLOOR, A.LIFETIME_MIN, A.SALIENCE_FLOOR,
            A.RESID_CAP, A.CONSIST_MIN_R, A.CONSIST_SCALE,
            A.RATE_LIMIT, A.OFFSET_BOUND, A.BLEED_RATE,
            A.SETTLE_T, A.LEASH_TAU) == (
        150, 20, 8.0, 2.0, 0.6, (0.5, 2.0),
        30.0, 10.0, 5.0, 2.0, 2.0)


def test_large_explained_motion_passes():
    # A far region rotating WITH the camera moves several px/frame
    # yet must pass: its motion is gyro-explained, residual ~0.
    fps, n = 30.0, 300
    rng = np.random.default_rng(7)
    gyro = _sine(n, fps, 1.1, 6.0) + rng.normal(0, 0.2, n)
    tracked = gyro * 1.1 + rng.normal(0, 0.3, n)
    ok, r, scale, reason = A.gyro_consistency(tracked, np.zeros(n),
                                             gyro, np.zeros(n))
    assert ok, (r, scale, reason)
    resid = float(np.median(np.abs(tracked - gyro)))
    assert resid <= A.RESID_CAP, resid
    regions = [{"inliers": 800, "lifetime": 300, "salience": 900.0,
                "resid": resid, "consist_ok": ok}]
    idx, reason = A.select_reference(regions)
    assert idx == 0, (idx, reason)
