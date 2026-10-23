"""Travel-holder synthetic gate (IMU level, no video).

Knock shape measured on May19 file 010: fast step (+4 -> +42 deg over
5 s), slow creep (+42 -> +58), long hold (~72-78 deg, ~80 s), snap back
(-115 deg in 1 s). A head-down look (-20..-60 deg, 1-15 s, with return)
must NOT engage. Both streams drive gyro by differentiating the attitude
and accel by rotating world gravity with the full attitude --
`accel_from_quats`, never the roll-only helper (pinned blind below).
"""
import numpy as np
import os
import pytest

from steadycut.stabilization import orientation as O
from steadycut.testing import synthetic_motion as mot

HZ = 100.0


def _attitude_from_elev(a_deg):
    """Body->world quats pitching the forward axis to `a_deg` elevation."""
    a = np.radians(np.asarray(a_deg, float))
    return O.from_rotvec(np.stack([a, np.zeros_like(a),
                                   np.zeros_like(a)], axis=-1))


def _knock_stream():
    t = np.arange(int(170 * HZ)) / HZ
    a = 3.0 * np.sin(2 * np.pi * 0.3 * t) - 24  # head wander about -24
    step = np.clip((t - 20.0) / 5.0, 0, 1) * 66.0       # +4 -> +42, 5 s
    creep = np.clip((t - 25.0) / 35.0, 0, 1) * 16.0      # +42 -> +58
    hold = ((t >= 60.0) & (t < 140.0)) * (74.0 - 58.0)   # hold ~74
    back = np.clip((t - 140.0) / 1.0, 0, 1) * -99.0      # snap to -25
    a = a + step + creep + hold + back
    a[t >= 141.0] = 3.0 * np.sin(2 * np.pi * 0.3 * t[t >= 141.0]) - 25.0
    # Real knock flop: the mount flops +-20 deg in heading and +-25 in
    # roll through the hold (measured May19: heading/roll std 21/25 deg
    # under a 1 s leash). A pitch-only freeze leaves a touring trend;
    # the synthetic must carry the flop or it cannot tell the designs
    # apart.
    flop = (t >= 60.0) & (t < 140.0)
    yaw = np.where(flop, 20.0 * np.sin(2 * np.pi * 2.3 * t),
                   3.0 * np.sin(2 * np.pi * 0.3 * t))
    roll = np.where(flop, 25.0 * np.sin(2 * np.pi * 3.1 * t + 1.0),
                    3.0 * np.sin(2 * np.pi * 0.4 * t))
    q = O._normalise(O.from_rotvec(np.column_stack(
        [np.radians(a), np.radians(yaw), np.radians(roll)])))
    gyro = mot.body_rates(q, 1.0 / HZ)
    return t, q, gyro, mot.accel_from_quats(q)


def _headdown_stream():
    t = np.arange(int(60 * HZ)) / HZ
    a = 3.0 * np.sin(2 * np.pi * 0.3 * t) - 24.0
    dip1 = (t >= 20.0) & (t < 28.0)   # -40 deg for 8 s, with return
    dip2 = (t >= 40.0) & (t < 42.0)   # -55 deg for 2 s
    a[dip1] = -40.0
    a[dip2] = -55.0
    q = _attitude_from_elev(a)
    gyro = mot.body_rates(q, 1.0 / HZ)
    return t, q, gyro, mot.accel_from_quats(q)


def test_knock_engages_and_holds_last_good():
    t, q, _, _ = _knock_stream()
    elev = O.forward_elevation(q)
    assert elev[(t >= 70) & (t < 130)].min() > 50.0  # the hold is a hold
    mask, runs = O.holder_mask(t, elev)
    assert mask.any(), "knock must engage the holder"
    assert mask[(t >= 70) & (t < 130)].all(), "hold interior fully engaged"
    assert not mask[t < 20].any(), "pre-knock riding stays free"
    assert not mask[t >= 150].any(), "restored riding stays free"
    assert any(b - a > 60.0 for a, b in runs), runs
    qt, info = O.smooth_world_trend(t, q, 1.0 / (2 * np.pi * 0.15),
                                    1.0 / (2 * np.pi * 0.02),
                                    1.0 / (2 * np.pi * 0.15))
    assert info["holder_mask"][(t >= 70) & (t < 130)].all()
    # The trend IS the view: it must sit still through the flop, not
    # just engage. A pitch-only freeze leaves heading/roll touring here.
    hold = (t >= 70) & (t < 130)
    assert O.forward_elevation(qt[hold]).std() < 3.0
    # Last-good means pre-knock: the frozen view must read the riding
    # elevation, not the engagement edge (which already carries whip --
    # freezing at the edge is what delivered the Dutch angle).
    pre = np.median(O.forward_elevation(q[(t >= 5) & (t < 18)]))
    assert abs(O.forward_elevation(qt[hold]).mean() - pre) < 8.0


def test_head_down_never_engages():
    t, q, _, _ = _headdown_stream()
    mask, _ = O.holder_mask(t, O.forward_elevation(q))
    assert not mask.any(), "legitimate head-down must not engage"


def test_roll_only_helper_is_blind_to_pitch():
    """Pin the trap: the old helper sees a level camera through a knock."""
    t, q, _, _ = _knock_stream()
    blind = mot.accel_from_attitude(t)  # roll-only, ignores the stream
    # Reads ~level throughout: only its own +-12 deg roll shows (<= ~2.1),
    # never the 70+ deg pitch-over that is the whole event.
    assert np.median(np.abs(blind[:, 1])) < 3.0
    honest = mot.accel_from_quats(q)
    assert np.median(np.linalg.norm(honest, axis=1)) == pytest.approx(9.81)
    # At the hold the honest accel tips with the camera, the blind one not.
    hold = (t >= 70) & (t < 130)
    assert np.median(np.abs(honest[hold, 2])) < 5.0
    assert np.median(np.abs(blind[hold, 2]) - 9.81) < 1.0


def test_vqf_recovers_knock_synthetic():
    """The synthetic IMU pair is honest: VQF reconstructs the profile."""
    vqf = pytest.importorskip("vqf")
    t, q, gyro, accel = _knock_stream()
    dt = 1.0 / HZ
    out = vqf.offlineVQF(np.ascontiguousarray(np.radians(gyro)),
                         np.ascontiguousarray(accel), None, dt)
    rec = np.asarray(out["quat6D"], float)
    err = np.abs(O.forward_elevation(rec)[t > 5.0]
                 - O.forward_elevation(q)[t > 5.0])
    assert np.median(err) < 5.0, np.median(err)


KNOCK_INSV = os.environ.get(
    "STEADYCUT_TEST_INSV",
    "/backup/insta360/VID_20260519_074911_00_010_011-Original/"
    "VID_20260519_074911_00_010.insv")


def _window(t0, dur):
    from steadycut.ingest.telemetry import read_telemetry
    from steadycut.ingest.cameras.registry import identify
    ident = identify(KNOCK_INSV)
    imu = read_telemetry(KNOCK_INSV, profile=ident["profile"])
    m = (imu.time_s >= t0) & (imu.time_s < t0 + dur)
    return (imu.time_s[m], imu.gyro[m], imu.accel[m],
            ident["profile"].axis_map)


@pytest.mark.parametrize("t0, expect_hold", [(365.0, True), (100.0, False)])
def test_holder_on_real_knock_window(t0, expect_hold):
    """Fail-old pin: the May19 knock engages the holder, calm riding not."""
    if not os.path.exists(KNOCK_INSV):
        pytest.skip(f"needs Insta360 fixture file (set STEADYCUT_TEST_INSV)")
    from steadycut.core.pipeline import correction_world_axes
    t, gyro, accel, am = _window(t0, 150.0)
    (_, _, _), info = correction_world_axes(
        t, gyro, accel, am, 1.0 / (2 * np.pi * 0.15),
        1.0 / (2 * np.pi * 0.02), 1.0 / (2 * np.pi * 0.15))
    if expect_hold:
        assert info["holder_seconds"] > 60.0, info["holder_seconds"]
    else:
        assert info["holder_seconds"] == 0.0, info["holder_runs"]
    assert info["gate_med_deg"] < 8.0, info["gate_med_deg"]


def test_gaze_policies_keep_legacy_path():
    """None default = legacy bit for bit: mtb/ski carry no pitch leash."""
    from steadycut.framing.policies import policy_for
    assert policy_for("mtb").pitch_tau_s is None
    assert policy_for("ski").pitch_tau_s is None
    assert policy_for("mtb_travel").pitch_tau_s is not None
