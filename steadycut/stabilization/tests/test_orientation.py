"""Known-answer tests for the quaternion orientation pipeline."""
import numpy as np

from steadycut.stabilization import orientation as o


def test_integration_recovers_a_known_rotation() -> None:
    """90 deg/s about an axis for 1 s must give pi/2 about that axis."""
    t = np.arange(0, 1.0, 1.0 / 1000.0)
    gyro = np.stack([np.zeros_like(t), np.zeros_like(t), np.full_like(t, 90.0)], axis=1)
    q = o.integrate(t, gyro)
    rv = o.to_rotvec(q[-1])
    assert abs(rv[2] - np.pi / 2) < 0.01, f"got {rv}"
    assert abs(rv[0]) < 0.01 and abs(rv[1]) < 0.01, f"leaked into other axes: {rv}"


def test_to_euler_extracts_known_angles() -> None:
    yaw = o.from_rotvec(np.radians(np.array([0.0, 0.0, 30.0])))
    assert np.allclose(o.to_euler(yaw), [30.0, 0.0, 0.0], atol=0.01)
    roll = o.from_rotvec(np.radians(np.array([15.0, 0.0, 0.0])))
    assert np.allclose(o.to_euler(roll), [0.0, 0.0, 15.0], atol=0.01)


def test_smoothing_preserves_a_steady_pan() -> None:
    """A constant-rate rotation is deliberate motion and must come through
    unchanged. If the filter introduced a lag, the correction would be
    non-zero here and the tool would fight the rider's own panning."""
    t = np.arange(0, 4.0, 1.0 / 1000.0)
    gyro = np.stack([np.zeros_like(t), np.zeros_like(t), np.full_like(t, 20.0)], axis=1)
    q = o.integrate(t, gyro)
    smoothed = o.smooth(q, t, 0.05)
    worst = 0.0
    for i in range(200, len(q) - 200, 50):
        rv = o.to_rotvec(o.relative(q[i], smoothed[i]))
        worst = max(worst, float(np.degrees(np.abs(rv)).max()))
    assert worst < 1.0, f"steady pan distorted by {worst:.3f} deg"


def test_smoothing_suppresses_a_sudden_jolt() -> None:
    """A brief high-rate spike must NOT survive into the smoothed orientation,
    otherwise there is nothing for the correction to cancel."""
    t = np.arange(0, 2.0, 1.0 / 1000.0)
    gyro = np.zeros((t.size, 3))
    gyro[500:520, 2] = 400.0          # 20 ms at 400 deg/s
    q = o.integrate(t, gyro)
    smoothed = o.smooth(q, t, 0.05)
    raw_angle = np.degrees(np.abs(o.to_rotvec(q[520])).max())
    smooth_angle = np.degrees(np.abs(o.to_rotvec(smoothed[520])).max())
    assert raw_angle > 6.0, f"jolt should be visible in raw data, got {raw_angle}"
    assert smooth_angle < raw_angle * 0.8, (
        f"jolt not suppressed: raw {raw_angle:.2f}, smoothed {smooth_angle:.2f}")
