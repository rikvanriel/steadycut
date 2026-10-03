"""Known-answer tests for the quaternion orientation pipeline."""
import numpy as np

from steadycut.stabilization import orientation as o
from steadycut.stabilization.orientation import to_euler_zxy, to_euler_zyx


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


# --------------------------------------------------------------- to_euler_zyx
def _zyx_matrix(abc):
    """Rz(a) Ry(b) Rx(c) built explicitly, so no library convention is assumed."""
    a, b, c = np.radians(np.asarray(abc, float)).T
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cc, sc = np.cos(c), np.sin(c)
    z, o = np.zeros_like(a), np.ones_like(a)
    rz = np.stack([np.stack([ca, -sa, z], -1),
                   np.stack([sa, ca, z], -1),
                   np.stack([z, z, o], -1)], -2)
    ry = np.stack([np.stack([cb, z, sb], -1),
                   np.stack([z, o, z], -1),
                   np.stack([-sb, z, cb], -1)], -2)
    rx = np.stack([np.stack([o, z, z], -1),
                   np.stack([z, cc, -sc], -1),
                   np.stack([z, sc, cc], -1)], -2)
    return rz @ ry @ rx


def test_to_euler_zyx_reproduces_the_rotation_it_was_given():
    """Round-trip, not an angle comparison.

    Euler triples are ambiguous -- several triples name the same rotation, and a
    second implementation may legitimately return a different one. Comparing
    angles then reports a 180 degree disagreement for a perfectly correct
    decomposition, which is what happened twice while writing this. Comparing
    the RECONSTRUCTED rotations has no such branch to trip over.
    """
    rng = np.random.default_rng(0)
    q = rng.normal(size=(5000, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)

    src = np.stack([1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2),
                    2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3]),
                    2 * (q[:, 1] * q[:, 3] + q[:, 0] * q[:, 2]),
                    2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3]),
                    1 - 2 * (q[:, 1] ** 2 + q[:, 3] ** 2),
                    2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1]),
                    2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2]),
                    2 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1]),
                    1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)], axis=1).reshape(-1, 3, 3)

    got = _zyx_matrix(to_euler_zyx(q))
    trace = np.einsum("nij,nij->n", src, got)
    ang = np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
    assert ang.max() < 1e-4, f"worst reconstruction error {ang.max():.2e} deg"


def test_to_euler_zyx_separates_the_three_axes():
    for axis, expected in ((3, (30, 0, 0)), (2, (0, 30, 0)), (1, (0, 0, 30))):
        q = np.zeros(4)
        q[0] = np.cos(np.radians(15))
        q[axis] = np.sin(np.radians(15))
        assert np.allclose(to_euler_zyx(q[None, :])[0], expected, atol=1e-9)


def test_to_euler_zyx_is_finite_at_gimbal_lock():
    """b = +-90 deg leaves c undetermined; the result must still be finite."""
    q = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])
    out = to_euler_zyx(np.repeat(q[None, :], 3, axis=0))
    assert np.all(np.isfinite(out))


# --------------------------------------------------------------- to_euler_zxy
def _zxy_matrix(ypy):
    """R_z(r) R_x(p) R_y(y2), built explicitly -- the measured render model.

    Real renders show a (yaw, pitch, roll) command acts on sphere content as
    R_z(roll) R_x(pitch) R_y(-yaw); with y2 = -yaw that product is this. No
    library Euler helper is used here because scipy's lowercase sequences
    compose in the reverse order of their names, and that confusion is what
    the previous two wrong versions of this code were built on.
    """
    y2, p, r = np.radians(np.asarray(ypy, float)).T
    cy, sy = np.cos(y2), np.sin(y2)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    z, o = np.zeros_like(y2), np.ones_like(y2)
    ry = np.stack([np.stack([cy, z, sy], -1),
                   np.stack([z, o, z], -1),
                   np.stack([-sy, z, cy], -1)], -2)
    rx = np.stack([np.stack([o, z, z], -1),
                   np.stack([z, cp, -sp], -1),
                   np.stack([z, sp, cp], -1)], -2)
    rz = np.stack([np.stack([cr, -sr, z], -1),
                   np.stack([sr, cr, z], -1),
                   np.stack([z, z, o], -1)], -2)
    return rz @ rx @ ry


def _quat_matrix(q):
    """Rotation matrix of (w, x, y, z) quaternions, the standard form."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=1).reshape(-1, 3, 3)


def test_to_euler_zxy_reproduces_the_rotation_it_was_given():
    """Round-trip through reconstruction, never an angle comparison.

    Euler triples are ambiguous -- several triples name the same rotation, so
    comparing angles can report 180 degrees of disagreement for a correct
    decomposition. Rebuilding R_z(r) R_x(p) R_y(y2) from the returned triple
    and comparing ROTATIONS has no such branch to trip over.
    """
    rng = np.random.default_rng(0)
    q = rng.normal(size=(5000, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    src = _quat_matrix(q)
    got = _zxy_matrix(to_euler_zxy(q))
    trace = np.einsum("nij,nij->n", src, got)
    ang = np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
    assert ang.max() < 1e-4, f"worst reconstruction error {ang.max():.2e} deg"


def test_to_euler_zxy_keeps_each_channel_on_its_own_axis():
    """A pure Y/X/Z rotation must come back on the matching output slot.

    A single-axis rotation has no order ambiguity, so any movement here is a
    permutation bug, not an order question. The output slots are
    (y2-for-yaw, pitch, roll): a pure Y rotation returns on slot 0, pure X
    on slot 1, pure Z on slot 2.
    """
    for axis, expected in ((2, (30, 0, 0)), (1, (0, 30, 0)), (3, (0, 0, 30))):
        q = np.zeros(4)
        q[0] = np.cos(np.radians(15))
        q[axis] = np.sin(np.radians(15))
        assert np.allclose(to_euler_zxy(q[None, :])[0], expected, atol=1e-9)


def test_to_euler_zxy_is_finite_at_gimbal_lock():
    """pitch = +-90 deg leaves y2/roll inseparable; still must be finite."""
    q = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
    out = to_euler_zxy(np.repeat(q[None, :], 3, axis=0))
    assert np.all(np.isfinite(out))
