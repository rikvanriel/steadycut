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
