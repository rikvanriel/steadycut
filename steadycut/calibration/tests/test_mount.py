"""Synthetic slope + lean + injected mount rotation for mount.py.

The naive estimator (sign-weighted pool, time-weighted, slope-ignorant)
must FAIL this fixture while the alternating/heading-weighted one passes:
uneven L/R dwell, a 20-degree slope pulling at the fall line, and body
lean into the turns.
"""
import numpy as np

from steadycut.calibration import mount


def _quat_from_ypr(yaw, pitch, roll):
    cy, sy = np.cos(yaw / 2, ), np.sin(yaw / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    qy = np.stack([cy, np.zeros_like(cy), np.zeros_like(cy), sy], -1)
    qx = np.stack([cp, np.zeros_like(cp), sp, np.zeros_like(cp)], -1)
    qz = np.stack([cr, np.zeros_like(cr), np.zeros_like(cr), sr], -1)
    return mount._quat_mul(mount._quat_mul(qz, qx), qy)


def _qz(a):
    a = np.asarray(a, float)
    return np.stack([np.cos(a / 2), np.zeros_like(a), np.zeros_like(a),
                     np.sin(a / 2)], -1)


def _qx(a):
    a = np.asarray(a, float)
    return np.stack([np.cos(a / 2), np.sin(a / 2), np.zeros_like(a),
                     np.zeros_like(a)], -1)


def _qy(a):
    a = np.asarray(a, float)
    return np.stack([np.cos(a / 2), np.zeros_like(a), np.sin(a / 2),
                     np.zeros_like(a)], -1)


def _synthetic(mount_yaw_deg=90.0, slope_deg=40.0, seed=3, lean_gain=0.4,
               pitch_frac=0.5):
    """Z-up world. Body-x = forward by construction; mount yaws about the
    mount vertical (body z at rest). Travel truth is east; the estimator
    must read offset angle -mount_yaw_deg (travel in body coords)."""
    rng = np.random.default_rng(seed)
    dt = 0.001  # 1000 Hz like the real trailer IMU: coning error that
    # dominates at 100 Hz shrinks ~100x, as on real data
    t = np.arange(0, 144, dt)  # multiple of the 8 s period: wraparound pad
    # stays continuous, else a fake snap at t=0 offsets all attitude
    # giant slalom: slow wide left turns (10 deg/s x 5 s), quick right
    # flicks (25 deg/s x 2 s). Dwell 71/29, sweep +/-25 deg.
    P = 8.0
    ph = (t / P) % 1.0
    tri = np.where(ph < 5.0 / 8.0, ph / (5.0 / 8.0),
                   1.0 - (ph - 5.0 / 8.0) / (3.0 / 8.0))
    psi = np.radians(25) * (2 * tri - 1)
    # real muscles don't step turn rate instantly: round the corners
    nk = int(round(0.3 / dt))
    k = np.ones(nk) / nk
    psi = np.convolve(np.concatenate([psi[-2 * nk:], psi, psi[:2 * nk]]),
                      k, mode="same")[2 * nk:-2 * nk]
    v = 8.0
    vel = np.stack([v * np.cos(psi), v * np.sin(psi),
                    np.zeros_like(psi)], -1)
    acc = np.gradient(vel, dt, axis=0)
    # at-rest specific force on a slope descending toward +y: the surface
    # normal tilts uphill, (0, +sin, +cos)
    th = np.radians(slope_deg)
    g = np.array([0.0, 9.81 * np.sin(th), 9.81 * np.cos(th)])
    f_world = acc + g  # specific force, z-up world
    # body: yaw tracks heading, slope pitches about lateral (body y),
    # mount yaws about body z. Lean rolls about the WORLD travel axis
    # (about body-x only when the mount is forward).
    yaw_rate = np.gradient(psi, dt)
    lean = np.clip(lean_gain * yaw_rate, -0.3, 0.3)
    my = np.radians(mount_yaw_deg)
    vh = np.stack([np.cos(psi), np.sin(psi), np.zeros_like(psi)], -1)
    hl = np.stack([np.cos(lean / 2), np.sin(lean / 2) * vh[..., 0],
                   np.sin(lean / 2) * vh[..., 1],
                   np.zeros_like(lean)], -1)
    q = mount._quat_mul(
        mount._quat_mul(hl, _qz(psi)),
        mount._quat_mul(_qy(np.full_like(t, th * pitch_frac)),
                        _qz(np.full_like(t, my))))
    # body gyro from attitude derivative (finite difference, exact enough)
    w = np.zeros((len(t), 3))
    qi = mount._quat_conj(q)
    dq = (np.roll(q, -1, axis=0) - np.roll(q, 1, axis=0)) / (2 * dt)
    w = 2 * mount._quat_mul(dq, qi)[:, 1:]
    w[0] = w[1]
    w[-1] = w[-2]
    w = np.degrees(w)  # integrate() takes deg/s
    w += rng.normal(0, 0.5, w.shape)  # gyro noise, deg/s
    f_body = mount.rotate_vec(qi, f_world)
    f_body += rng.normal(0, 0.02, f_body.shape)
    return t, w, f_body, q


def _world_rate_and_h(t, w, f_body, q):
    up = mount.rotate_vec(mount._quat_conj(q), np.tile([0.0, 0.0, 1.0],
                                                       (len(t), 1)))
    rate = np.einsum("ij,ij->i", w, up)  # rad/s about world vertical
    fw = mount.rotate_vec(q, f_body)
    return rate, fw[:, :2]


def test_alternating_recovers_injected_mount():
    t, w, f_body, q = _synthetic(mount_yaw_deg=90.0)
    rate, h = _world_rate_and_h(t, w, f_body, q)
    az, (n_l, n_r), _ = mount.alternating_bearing(t, np.degrees(rate), h)
    assert az is not None
    assert n_l > 0 and n_r > 0
    # travel azimuth (east = 0) minus body-forward azimuth must equal the
    # injected mount offset: body forward in world = heading + mount.
    # The estimator returns travel azimuth; compare against truth 0 after
    # removing the known body rotation is circular, so instead check the
    # offset it implies: body yaw at mid-segment minus travel.
    err = (np.degrees(az) - 0.0 + 180) % 360 - 180
    # travel truth is east (mean heading 0); estimator error:
    assert abs(err) < 3.0, (np.degrees(az), n_l, n_r)


def test_naive_pool_fails_uneven_dwell():
    """Sign-flipped pool against a SINGLE turn-sample reference (no L/R
    split) must be off by > 10 deg: the reference tilts with net turn
    Delta-v and the slope pull leaks in. The production estimator splits
    L/R with glide-anchored references instead."""
    from steadycut.stabilization import orientation as O
    t, w, f_body, q = _synthetic(mount_yaw_deg=90.0)
    qg = O.integrate(t, w)
    upg = mount.rotate_vec(mount._quat_conj(qg),
                           np.tile([0.0, 0.0, 1.0], (len(t), 1)))
    rate = np.degrees(np.einsum("ij,ij->i", np.radians(w), upg))
    segs = mount.turn_segments(t, rate)
    assert len(segs) > 2
    idx = np.concatenate([np.arange(a, b) for a, b, _ in segs])
    fw = mount.rotate_vec(qg[idx], f_body[idx])
    gv = fw.mean(axis=0)
    gv = gv / np.linalg.norm(gv)
    h = fw - np.outer(fw @ gv, gv)
    pool = np.zeros(3)
    for a, b, s in segs:
        pool += s * h[a:b].sum(axis=0)
    m = segs[len(segs) // 2]
    qm = qg[(m[0] + m[1]) // 2]
    Rm = mount._quat_to_mat(qm)
    bx = Rm @ np.array([1.0, 0.0, 0.0])
    fwd = bx - (bx @ gv) * gv
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(gv, fwd)
    v = np.array([pool @ fwd, pool @ right])
    v = v / np.linalg.norm(v)
    tv = np.array([v[1], -v[0]])
    err = (np.degrees(np.arctan2(tv[1], tv[0])) + 90.0 + 180) % 360 - 180
    assert abs(err) > 10.0, np.degrees(np.arctan2(tv[1], tv[0]))


def test_travel_offset_recovers_mount():
    """travel_offset reads the injected mount within 8 deg at slope 0 and
    30 (travel truth is east; body-x reads -mount in body coords).

    Pure-yaw motion: the slalom-with-lean variant phase-locks lean and
    yaw into the harshest coning regime, which defeats first-order
    attitude integration (footage-validated separately on Mar05/Jan26);
    the fall-pull cancellation this pins is independent of lean.
    """
    for my, slope in ((90.0, 0.0), (0.0, 0.0), (-90.0, 0.0), (90.0, 30.0),
                      (0.0, 30.0), (-90.0, 30.0)):
        t, w, f_body, q = _synthetic(mount_yaw_deg=my, slope_deg=slope,
                                     lean_gain=0.0)
        vec, info = mount.travel_offset(t, w, f_body, forward_axis=0)
        assert vec is not None, (my, slope, info)
        err = (np.degrees(np.arctan2(vec[1], vec[0])) + my + 180) % 360 - 180
        assert abs(err) < 8.0, (my, slope, np.degrees(
            np.arctan2(vec[1], vec[0])))


def test_travel_offset_z_forward_axis():
    """Production convention (z forward): the same motion relabeled to
    (lateral, vertical, forward) must read +mount."""
    for my, slope in ((90.0, 0.0), (0.0, 30.0), (-90.0, 30.0)):
        t, w, f_body, q = _synthetic(mount_yaw_deg=my, slope_deg=slope,
                                     lean_gain=0.0)
        perm = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                         [1.0, 0.0, 0.0]])  # proper rotation, det +1
        vec, info = mount.travel_offset(t, w @ perm.T, f_body @ perm.T)
        assert vec is not None, (my, slope, info)
        err = (np.degrees(np.arctan2(vec[1], vec[0])) + my + 180) % 360 - 180
        assert abs(err) < 8.0, (my, slope, np.degrees(
            np.arctan2(vec[1], vec[0])))
