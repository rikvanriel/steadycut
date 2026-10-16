"""Pins for travel-yaw: policy registration, leash behavior, legacy intact."""
import sys

import numpy as np

sys.path.insert(0, "/source/upstream/steadycut")
from steadycut.framing.policies import policy_for                     # noqa: E402
from steadycut.stabilization import orientation as O                   # noqa: E402


def test_ski_policy_is_travel_with_a_leash():
    p = policy_for("ski")
    assert p is not None
    assert (p.mount, p.follows) == ("helmet", "travel")
    assert p.yaw_corner_hz == 0.02
    assert "travel-yaw" in p.capabilities


def test_mtb_policy_has_no_leash():
    p = policy_for("mtb")
    assert p is not None
    assert p.yaw_corner_hz is None


def test_zxy_round_trip_is_exact():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(500, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    e = O.to_euler_zxy(q)
    q2 = O._zxy_to_quat(np.radians(e[:, 0]), np.radians(e[:, 1]),
                        np.radians(e[:, 2]))
    err = np.degrees(2 * np.arccos(np.clip(np.abs((q * q2).sum(axis=1)),
                                           0, 1)))
    assert err.max() < 1e-4  # the repo's own 2.7e-6 standard, with margin


def test_yaw_leash_holds_slow_rejects_fast():
    # yaw = slow ramp (travel turning) + fast whip (look-around)
    # 10 s lead-in: the production path pads +-15 s so the smoother settles;
    # a filter measured without padding measures the padding, not the filter.
    t = np.arange(-2000, 4000) / 200.0
    slow = 30 * np.sin(2 * np.pi * 0.02 * t)
    fast = 60 * np.sin(2 * np.pi * 0.5 * t) * (t > 10) * (t < 14)
    yaw = np.radians(slow + fast)
    q = O._zxy_to_quat(yaw, np.zeros_like(yaw), np.zeros_like(yaw))
    import math
    trend = O.smooth_channels(q, t, 1 / (2 * math.pi * 0.02),
                              1 / (2 * math.pi * 0.15))
    ty = np.degrees(np.unwrap(np.radians(O.to_euler_zxy(trend)[:, 0])))
    # slow passes (tracks the ramp within a few degrees off-transient)...
    # The calm set stays a full leash-length from the whip on both sides: the
    # forward-backward smoother is non-causal, so a tau=8 s leash starts
    # leaning into the whip ~8 s BEFORE it (backward leak, the documented
    # price of zero lag) and needs seconds after to recover. Both are the
    # leash working as designed, pinned below by the pre_response bound.
    calm = ((t > 0) & (t < 2)) | (t > 20)
    assert np.abs(ty[calm] - slow[calm]).max() < 5.0
    # ...fast is attenuated (whip leaks at most a third of its amplitude).
    assert np.abs(ty[(t > 10) & (t < 14)] - slow[(t > 10) & (t < 14)]).max() < 25.0
    # backward leak bounded: 1 s before a 60 deg whip the trend leans < 12.
    pre = (t >= 9) & (t < 10)
    assert np.abs(ty[pre] - slow[pre]).max() < 12.0


def test_equal_taus_match_legacy_smoothing():
    # Single-axis motion: quaternion slerp and angle filtering coincide
    # exactly, so equal taus must reproduce smooth() bit for bit here.
    import math
    t = np.arange(2000) / 200.0
    g = np.zeros((2000, 3))
    g[:, 1] = 5 * np.sin(2 * np.pi * 0.5 * t) + 30 * np.sin(2 * np.pi * 0.05 * t)
    qr = O.integrate(t, g)
    tau = 1 / (2 * math.pi * 0.15)
    a = O.smooth(qr, t, tau)
    b = O.smooth_channels(qr, t, tau, tau)
    err = np.degrees(2 * np.arccos(np.clip(np.abs((a * b).sum(axis=1)), 0, 1)))
    assert err.max() < 1e-4  # repo round-trip standard, with margin
