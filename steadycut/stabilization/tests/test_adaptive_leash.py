"""Adaptive yaw leash: fast in sustained turns, slow otherwise."""
import numpy as np

from steadycut.stabilization import orientation as O


def _yaw_path(rate_dps, t, tau):
    # yaw angle from a rate profile, then leashed trend (Euler-yaw part of
    # smooth_channels exercised directly)
    y = np.cumsum(np.radians(rate_dps) * np.gradient(t))
    n = len(t)
    # pipeline yaw is about body-Y (v360 order YAW/PITCH/ROLL about Y,X,Z)
    q = np.stack([np.cos(y / 2), np.zeros(n), np.sin(y / 2), np.zeros(n)],
                 -1)
    return O.smooth_channels(q, t, tau, 7.96)


def test_turn_state_holds_glances():
    t = np.arange(0, 30, 0.01)
    rate = np.zeros_like(t)
    rate[1000:1150] = 30.0  # 1.5 s glance blip
    rate[2000:] = 7.0  # sustained turn to end
    m = O.turn_state(t, rate)
    assert not m[500:1500].any(), "glance must not engage"
    assert m[2500:2900].mean() > 0.9, "sustained turn must engage"


def test_adaptive_tracks_turn_ignores_glance():
    t = np.arange(0, 30, 0.01)
    rate = np.zeros_like(t)
    rate[500:650] = 30.0  # 1.5 s, 45 deg glance
    rate[1500:2500] = 7.0  # 10 s, 70 deg sustained turn
    slow_tau = 1.0 / (2 * np.pi * 0.02)
    # trend error vs truth (zero-lag shape keeps constant rates exact;
    # the turn ENTRY transient is what the leash speed governs)
    tr_slow = _yaw_path(rate, t, slow_tau)
    tr_ad = _yaw_path(rate, t, (slow_tau, 1.0))
    # compare yaw-angle trends over the turn (samples 1500:2500)
    from steadycut.stabilization.orientation import to_euler_zxy
    ys = np.unwrap(np.radians(to_euler_zxy(tr_slow)[:, 0]))
    ya = np.unwrap(np.radians(to_euler_zxy(tr_ad)[:, 0]))
    truth = np.cumsum(np.radians(rate) * 0.01)
    es = np.abs(ys[1500:2500] - truth[1500:2500]).max()
    ea = np.abs(ya[1500:2500] - truth[1500:2500]).max()
    assert ea < es, (np.degrees(es), np.degrees(ea))
    # glance response: adaptive must not move more than the slow leash
    # does (same taus outside turns; tolerance for edge effects)
    gs = np.abs(ys[400:800] - truth[400:800]).max()
    ga = np.abs(ya[400:800] - truth[400:800]).max()
    assert ga <= gs + np.radians(3), (np.degrees(gs), np.degrees(ga))
