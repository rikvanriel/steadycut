"""Synthetic known-answer tests for mesh_warp (reviewer t_9fb047b4 probes).

Gate numbers are the contract (C7-style): changing them needs a plan
amendment, not a quiet edit.
"""
import numpy as np
import pytest

from steadycut.stabilization import mesh_warp as M

scipy = pytest.importorskip("scipy")


def _sine(n, fps, hz, amp):
    t = np.arange(n) / fps
    return amp * np.sin(2 * np.pi * hz * t)


def test_affine_shake_removed():
    # Injected 2Hz shake: the long-tau smoother must NOT follow it
    # (smooth ~= 0), so warping vertices to the smoothed path
    # removes the shake. Short tau would track it (wrong layer).
    fps, n = 30.0, 300
    shake = _sine(n, fps, 2.0, 8.0)
    sm = M.smooth_gauss(shake, fps, tau_s=1.0)
    assert float(np.std(sm)) < 0.3 * float(np.std(shake))
    sm2 = M.smooth_symmetric(shake, fps, tau_s=1.0)
    assert float(np.std(sm2)) < 0.3 * float(np.std(shake))


def test_symmetric_tracks_ramp():
    # The symmetric smoother follows a steady ramp (travel) with
    # ~zero interior lag — the property the mesh depends on.
    fps, n = 30.0, 300
    ramp = np.linspace(0, 300, n)
    sm = M.smooth_symmetric(ramp, fps, tau_s=0.2)
    assert float(np.max(np.abs(sm[30:-30] - ramp[30:-30]))) < 2.0


def test_static_bit_identical():
    # Static input: smoother output equals input; remap is identity.
    fps, n = 30.0, 100
    traj = np.zeros(n)
    assert bool((M.smooth_gauss(traj, fps) == 0.0).all())
    assert bool((M.smooth_l1(traj) == 0.0).all())
    h, w = 36, 48
    mx, my = M.compose_remap(0.0, 0.0, 0.0, 0.0, h, w)
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    assert bool((mx == xs).all() and (my == ys).all())


def test_translation_removed_rotation_untouched():
    # Translation-only motion: after projecting out the (zero)
    # rotation term, the residual carries all of it.
    n = 200
    tx = _sine(n, 30.0, 0.5, 6.0)
    rx, ry = M.project_rotation(tx, np.zeros(n), np.zeros(n),
                                np.zeros(n))
    assert bool((rx == tx).all())
    # Rotation-only motion: projecting it out leaves ~nothing.
    gyro = _sine(n, 30.0, 1.0, 5.0)
    rx2, _ = M.project_rotation(gyro, np.zeros(n), gyro, np.zeros(n))
    assert float(np.std(rx2)) < 1e-9


def test_l1_gauss_agree_on_static_and_track_ramp():
    # Both arms agree on trivial inputs (no solver pathology).
    fps, n = 30.0, 120
    ramp = np.linspace(0, 10, n)
    g = M.smooth_gauss(ramp, fps, tau_s=1.0)
    l1 = M.smooth_l1(ramp, lmbda=10.0)
    # Interior tracks the ramp with bounded lag (exp smoother lags a
    # ramp by slope*tau; the warp inherits that lag, bounded here).
    assert float(np.max(np.abs(g[10:-10] - ramp[10:-10]))) < 2.0
    assert float(np.max(np.abs(g - ramp))) < 3.0
    assert float(np.max(np.abs(l1 - ramp))) < 2.0


def test_near_field_pin():
    # Pinned cells get zero local correction, unpinned keep it.
    h, w = 36, 48
    local = np.full((M.GRID[0] + 1, M.GRID[1] + 1), 5.0)
    pin = np.zeros((h, w), dtype=bool)
    pin[30:, 18:30] = True  # bike/helmet zone
    mx, _ = M.compose_remap(0.0, 0.0, local, local, h, w,
                            pin_mask=pin)
    xs, _ = np.meshgrid(np.arange(w), np.arange(h))
    assert float(mx[33, 24]) == float(xs[33, 24])  # pinned: identity
    assert float(mx[5, 5]) < float(xs[5, 5])  # unpinned: shifted


def test_gates_pinned():
    assert (M.GRID, M.CELL_INLIER_FLOOR, M.VERTEX_LIFETIME_MIN,
            M.SALIENCE_FLOOR, M.DISTORT_CAP, M.SOLVE_RESID_CAP,
            M.COVERAGE_FLOOR, M.ZOOM_BUDGET, M.WARP_CAP, M.MESH_TAU,
            M.L1_LAMBDA) == (
        (10, 10), 40, 20, 8.0, 2.0, 1.5, 0.5, 1.05, 16.0, 0.2, 1.0)


def test_warp_cap_lets_travel_through():
    # A huge steady local drift is clipped, not followed.
    h, w = 36, 48
    big = np.full((M.GRID[0] + 1, M.GRID[1] + 1), 400.0)
    mx, my = M.compose_remap(0.0, 0.0, big, big, h, w)
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    assert float(np.max(np.abs(mx - xs))) <= M.WARP_CAP + 1.0
    assert float(np.max(np.abs(my - ys))) <= M.WARP_CAP + 1.0


def _option_d_setup(seed=0, ny=10, nx=12):
    # Synthetic cells: known depth field, known gyro rates per frame.
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx]
    depth = 0.5 + (xx / max(nx - 1, 1))  # near on the right
    depth = depth / float(np.median(depth))
    plate, dx, dy = M.plate_map(72, 96, 115.0)
    # per-cell plate: sample the pixel map at cell centres
    pc = np.zeros((ny, nx))
    for iy in range(ny):
        for ix in range(nx):
            pc[iy, ix] = plate[int((iy + 0.5) * 72 / ny),
                               int((ix + 0.5) * 96 / nx)]
    return depth, pc, dx, dy


def test_fit_translation_recovers_t_only():
    # Translation-only residual: fit recovers t, small residual.
    ny, nx = 10, 12
    depth, pc, _, _ = _option_d_setup()
    t_true = np.array([3.0, -2.0, 1.5])
    yy, xx = np.mgrid[0:ny, 0:nx]
    rx = (xx - (nx - 1) / 2.0) / (nx - 1)
    ry = (yy - (ny - 1) / 2.0) / (ny - 1)
    dm = depth - float(np.median(depth))
    fx = t_true[0] * dm + t_true[2] * rx * dm
    fy = t_true[1] * dm + t_true[2] * ry * dm
    flow = np.stack([fx, fy], -1)
    rot = np.zeros_like(flow)
    w = np.ones((ny, nx))
    rng = np.random.default_rng(1)
    flow = flow + rng.normal(0, 0.2, flow.shape)
    t, inl, rms = M.fit_frame_translation(flow, rot, depth, w)
    assert abs(t[0] - 3.0) < 0.5 and abs(t[1] + 2.0) < 0.5
    assert abs(t[2] - 1.5) < 0.5
    assert inl > 0.9 and rms < 1.0


def test_fit_translation_leak_gate():
    # Rotation-only residual under a WRONG rotation model (plate off
    # by 15%, swirl omitted - the real plate-error regime): fitted t
    # must stay ~0. Fails if the fit absorbs gyro motion into
    # translation. (With an exact model the fit sees zeros, which
    # would pass vacuously; the gate has to earn its keep.)
    ny, nx = 10, 12
    depth, pc, _, _ = _option_d_setup()
    om = (0.01, -0.008, 0.005)  # rad/frame pitch/yaw/roll
    yy, xx = np.mgrid[0:ny, 0:nx]
    cx = (xx + 0.5) * 96.0 / nx - 47.5
    cy = (yy + 0.5) * 72.0 / ny - 35.5
    vx, vy = M.rotation_field(*om, pc, cx, cy)
    flow = np.stack([vx, vy], -1)
    wx, wy = M.rotation_field(om[0], om[1], 0.0, pc * 1.15, cx, cy)
    rot_bad = np.stack([wx, wy], -1)
    t, inl, rms = M.fit_frame_translation(flow, rot_bad, depth,
                                           np.ones((ny, nx)))
    assert float(np.abs(t).max()) < 0.3


def test_fit_translation_survives_outliers():
    # 20% garbage cells: t still recovered, inlier fraction drops.
    ny, nx = 10, 12
    depth, _, _, _ = _option_d_setup()
    dm = depth - float(np.median(depth))
    flow = np.stack([2.0 * dm, -1.0 * dm], -1)
    rng = np.random.default_rng(2)
    flow = flow + rng.normal(0, 0.1, flow.shape)
    w = np.ones((ny, nx))
    flow[0:2, :] = 30.0  # garbage rows
    t, inl, rms = M.fit_frame_translation(flow, np.zeros_like(flow),
                                           depth, w)
    assert abs(t[0] - 2.0) < 0.5 and abs(t[1] + 1.0) < 0.5
    assert inl < 0.95


def test_fit_translation_uniform_depth_degenerates():
    # Uniform depth: tz column is zero; fit must survive and return
    # tx, ty (tz = 0), never NaN.
    ny, nx = 8, 8
    depth = np.ones((ny, nx))
    dm = depth - 1.0
    flow = np.stack([2.0 * dm, -1.0 * dm], -1)
    t, inl, rms = M.fit_frame_translation(flow, np.zeros_like(flow),
                                           depth, np.ones((ny, nx)))
    assert np.all(np.isfinite(t)) and abs(t[2]) < 1e-9


def test_update_invdepth_resets_on_step():
    # Slow drift is followed; an occlusion step resets the cell.
    d = np.ones((6, 6))
    m = np.full((6, 6), 1.0)
    d1 = M.update_invdepth(d, m)
    assert abs(float(np.median(d1)) - 1.0) < 1e-9
    m2 = np.full((6, 6), 1.0)
    m2[0, 0] = 8.0
    d2 = M.update_invdepth(d1, m2)
    assert d2[0, 0] > 2.0  # reset, not smeared
    assert abs(d2[3, 3] - 1.0) < 0.2  # untouched cells stay


def test_pin_weights_hysteresis():
    # Flicker between the thresholds holds state: a frame-by-frame
    # detector toggle adds no step to the weights (reviewer gate C).
    prev = np.array([0.0, 1.0])
    mid = np.array([0.5, 0.5])
    assert list(M.pin_weights(mid, prev)) == [0.0, 1.0]
    assert list(M.pin_weights(np.array([0.9, 0.1]), prev)) == [1.0, 0.0]
    # toggling the detector every frame INSIDE the deadband holds
    # forever, never steps; dips above `down` never release an engage.
    w = np.zeros(4)
    for t in range(20):
        w = M.pin_weights(np.full(4, 0.4 + 0.2 * (t % 2)), w)
    assert float(np.abs(w).max()) == 0.0
    w = M.pin_weights(np.full(4, 0.9), w)
    for t in range(20):
        w = M.pin_weights(np.full(4, 0.4 + 0.2 * (t % 2)), w)
    assert float(np.abs(w - 1.0).max()) == 0.0
