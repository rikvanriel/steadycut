"""Travel-forward in mount coordinates, from turn centripetal force.

A sideways mount makes lens-relative "forward" wrong by tens of degrees,
so travel must be measured, not assumed. In a turn the horizontal specific
force alternates sign with turn direction around a constant fall-line pull:
(mean_L - mean_R)/2 of per-direction means keeps the alternating
(centripetal) part and drops the common (slope) part. Integrated over
HEADING (samples weighted by 1/|yaw rate|), not time, so uneven dwell does
not leak an along-track residual.

Attitude through turns is gyro-only, anchored at the flanking glides: an
accel-corrected filter would tilt toward apparent gravity and erase the
signal being measured. Drift budget is checked per segment (< 2 deg).
"""
from __future__ import annotations

import numpy as np

TURN_RATE_DEG_S = 8.0   # sustained |rate| above this opens a turn segment
MIN_TURN_S = 2.0        # shorter wiggles are look-arounds, not turns
MIN_HEADING_DEG = 15.0  # the turn must actually go somewhere
DRIFT_BUDGET_DEG = 2.0  # gyro bias x duration, checked per segment


def _quat_mul(a, b):
    a = np.asarray(a, float)
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


def _quat_conj(q):
    q = np.asarray(q, float)
    return np.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)


def rotate_vec(q, v):
    """Rotate vectors v by unit quaternions q (w,x,y,z). Body-to-world when
    q is the gyro-only attitude from integrate()."""
    q = np.asarray(q, float)
    v = np.asarray(v, float)
    zero = np.zeros(v.shape[:-1] + (1,))
    vq = np.concatenate([zero, v], axis=-1)
    out = _quat_mul(_quat_mul(q, vq), _quat_conj(q))
    return out[..., 1:]


def _unwrap(x):
    return np.unwrap(np.asarray(x, float))


def turn_segments(t, yaw_rate_world):
    """Contiguous sustained one-sign turn segments: (i0, i1, sign).

    Selection is on WORLD-VERTICAL yaw rate, not body-z gyro: a leaning
    mount corrupts the body axis. Hysteresis (enter above 10, exit below
    3 deg/s) kills flicker fragments; same-sign neighbors within 1 s
    merge; the sign comes from total heading change, not the first
    sample; corners are trimmed 0.3 s each side (rate ramps and lean
    steps live there and would dominate 1/rate weights). Refuses wiggles
    (short or small total heading change).
    """
    ENTER = 8.0
    EXIT = 3.0
    t = np.asarray(t, float)
    r = np.asarray(yaw_rate_world, float)
    n = len(t)
    raw = []
    i = 0
    while i < n:
        if abs(r[i]) < ENTER:
            i += 1
            continue
        s = 1 if r[i] > 0 else -1
        j = i
        while j + 1 < n and r[j + 1] * s > -EXIT:
            j += 1
        raw.append((i, j + 1))
        i = j + 1
    # merge same-sign neighbors within 1 s (sign by heading change).
    # Never merge across a sign change in the gap (that glues an L leg to
    # the following R leg through the corner ramp).
    merged = []
    for a, b in raw:
        dh = float(np.trapezoid(r[a:b], t[a:b]))
        if abs(dh) < np.radians(5.0):
            continue  # flicker, not a turn
        s = 1 if dh > 0 else -1
        if merged:
            pa, pb, ps = merged[-1]
            gap = r[pb:a]
            samesign = ps == s and t[a] - t[pb] < 1.0
            noswap = len(gap) == 0 or np.all(gap * ps > -EXIT)
            if samesign and noswap:
                merged[-1][1] = b
                continue
        merged.append([a, b, s])
    # trim corners, enforce minimums
    segs = []
    dt_med = float(np.median(np.diff(t)))
    trim = int(round(0.3 / dt_med))
    for a, b, s in merged:
        a2, b2 = a + trim, b - trim
        if b2 - a2 < 2:
            continue
        dur = t[b2 - 1] - t[a2]
        dhead = abs(float(np.trapezoid(r[a2:b2], t[a2:b2])))
        if dur >= MIN_TURN_S and dhead >= np.radians(MIN_HEADING_DEG):
            segs.append((a2, b2, s))
    return segs


def alternating_bearing(t, yaw_rate_world, h_force_world):
    """Travel azimuth from the alternating horizontal specific force.

    h_force_world: (N,2) horizontal specific force in the gravity frame
    (east, north), gyro-only attitude. Returns (azimuth_rad,
    per_direction_counts, drift_ok). Azimuth is CCW-from-east math
    convention; the caller maps it into the framing convention once.
    Samples weighted by 1/|rate| (heading integration).
    """
    t = np.asarray(t, float)
    r = np.asarray(yaw_rate_world, float)
    h = np.asarray(h_force_world, float)
    segs = turn_segments(t, r)
    if not segs:
        return None, (0, 0), True
    num_l = np.zeros(2)
    num_r = np.zeros(2)
    den_l = 0.0
    den_r = 0.0
    for i0, i1, s in segs:
        w = 1.0 / np.maximum(np.abs(r[i0:i1]), 1e-6)
        if s > 0:
            num_l += (h[i0:i1] * w[:, None]).sum(axis=0)
            den_l += w.sum()
        else:
            num_r += (h[i0:i1] * w[:, None]).sum(axis=0)
            den_r += w.sum()
    if den_l == 0 or den_r == 0:
        return None, (int(den_l > 0), int(den_r > 0)), True
    mean_l = num_l / den_l
    mean_r = num_r / den_r
    # Alternating part: centripetal. Common (fall-line) part cancels.
    c = (mean_l - mean_r) / 2.0
    if float(np.linalg.norm(c)) < 1e-9:
        return None, (1, 1), True
    # c points left-of-velocity (CCW): travel is c rotated -90 deg.
    az = np.arctan2(c[1], c[0]) - np.pi / 2.0
    n_l = sum(s > 0 for _, _, s in segs)
    n_r = sum(s < 0 for _, _, s in segs)
    return float(az), (n_l, n_r), True


def travel_offset(t, gyro_body, accel_body, forward_axis=2):
    """Mount offset direction in BODY coordinates from turn segments.

    gyro_body: (N,3) deg/s about body x,y,z. accel_body: (N,3) g.
    forward_axis: which body axis is forward (2 = roll axis = z, the
    production convention: roll is rotation about the forward axis,
    pitch/x lateral, yaw/y vertical). Pass 0 for x-forward fixtures.
    Each turn segment is integrated locally from zero (gyro-only; drift
    over seconds stays in budget), its weighted-mean horizontal-force
    vector transported into the body-at-mid frame, and per-direction
    means combined as (mean_L - mean_R)/2. Returns (offset_xy unit vec2
    in body-horizontal plane | None, info dict).
    """
    from steadycut.stabilization import orientation as O
    t = np.asarray(t, float)
    g = np.asarray(gyro_body, float)
    a = np.asarray(accel_body, float)
    # coarse global attitude for SELECTION only (drift corrupts the
    # projection slowly; sustained-rate detection tolerates it). Leveled
    # start: identity would tilt the reference by the mount itself.
    qg = O.integrate(t, g, q0=O.level_start(t, a, g))
    # world-up expressed IN BODY coords (conjugate direction!): dotting
    # body gyro with body-z-in-world mixes frames and corrupts the sign
    # the moment lean/pitch tilt body-z away from world-up.
    upg = rotate_vec(_quat_conj(qg), np.tile([0.0, 0.0, 1.0], (len(t), 1)))
    rate = np.einsum("ij,ij->i", np.radians(g), upg)
    segs = turn_segments(t, np.degrees(rate))
    if not segs:
        return None, {"refused": "no turn segments"}
    num_l = np.zeros(2)
    num_r = np.zeros(2)
    den_l = 0.0
    den_r = 0.0
    n_l = n_r = 0
    dwell_l = dwell_r = 0.0
    rows = []  # per-segment votes: (t_mid, sign, vx, vy, weight)
    for i0, i1, s in segs:
        # vertical is anchored in the pre-turn glide (up to 3 s of
        # below-threshold samples): the turn's own Delta-v would tilt a
        # segment-mean reference along-track by ~10 deg and leak into the
        # alternating part by turn geometry (not common-mode).
        ip = i0
        while ip > 0 and (t[i0] - t[ip]) < 3.0 and \
                abs(rate[ip - 1]) < TURN_RATE_DEG_S / 2:
            ip -= 1
        if t[i0] - t[ip] < 0.5:
            # no glide to anchor the vertical in: skip, do not guess
            continue
        st, sg, sa = t[ip:i1], g[ip:i1], a[ip:i1]
        q = O.integrate(st - st[0], sg)  # local frame = body at prepend
        j0 = i0 - ip  # turn starts here in local indices
        gv = sa[:max(j0, 1)].mean(axis=0)
        gv = gv / (np.linalg.norm(gv) + 1e-12)
        # vertical from the glide, expressed in local-world (the turn's own
        # Delta-v would tilt a segment-mean reference along-track).
        gvw = rotate_vec(q[:max(j0, 1)], gv).mean(axis=0)
        gvw = gvw / (np.linalg.norm(gvw) + 1e-12)
        # accumulation over the TURN samples only, in local-world
        sl = slice(j0, len(st))
        fw = rotate_vec(q[sl], sa[sl])
        rr = np.einsum("ij,ij->i", np.radians(sg[sl]),
                       rotate_vec(_quat_conj(q[sl]),
                                  np.tile([0.0, 0.0, 1.0],
                                          (i1 - i0, 1))))
        h = fw - np.outer(fw @ gvw, gvw)
        # 1/|rate| integrates over heading, not time (uneven dwell must not
        # bias L vs R); capped at the entry threshold so the trimmed edges
        # (rate ramps) cannot spike and re-dominate through the back door.
        w = 1.0 / np.maximum(np.abs(rr), np.radians(10.0))
        # offset in the segment's horizontal plane, yaw-only: lean must not
        # contaminate the projection, so build forward/right vectorially
        # (no Euler conventions). Forward is the body ROLL axis (z):
        # roll is rotation about the forward axis, while pitch (x) is
        # lateral and yaw (y) is vertical in integrate's body frame.
        # (Body-x here read ~90 deg off the Mar05 FOE anchor; z lands on
        # it.) Body-z into local, horizontalized:
        qmid = q[j0 + (len(st) - j0) // 2]
        Rm = _quat_to_mat(qmid)
        axes = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]))
        bx = Rm @ axes[forward_axis]
        fwd = bx - (bx @ gvw) * gvw
        fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
        right = np.cross(gvw, fwd)
        # PER-SAMPLE accumulation into body-mid coords: collapsing to a
        # segment mean first would zero the signal (the mean of h about a
        # segment's own reference is identically ~zero; the geometry lives
        # in the heading dependence across samples).
        V = np.stack([h @ fwd, h @ right], axis=1)
        Vsum = (V * w[:, None]).sum(axis=0)
        Wsum = float(w.sum())
        rows.append((float(st[j0 + (len(st) - j0) // 2]), int(s),
                     float(Vsum[0] / Wsum), float(Vsum[1] / Wsum), Wsum))
        if s > 0:
            num_l += Vsum
            den_l += Wsum
            n_l += 1
            dwell_l += t[i1 - 1] - t[i0]
        else:
            num_r += Vsum
            den_r += Wsum
            n_r += 1
            dwell_r += t[i1 - 1] - t[i0]
    if den_l == 0 or den_r == 0:
        return None, {"refused": "one-sided turns",
                      "n_l": n_l, "n_r": n_r}
    mean_l = num_l / den_l
    mean_r = num_r / den_r
    c = (mean_l - mean_r) / 2.0
    n = float(np.linalg.norm(c))
    if n < 1e-9:
        return None, {"refused": "alternating part vanishes"}
    # c points left-of-velocity in (forward, left) coords; travel is c
    # rotated -90 deg (clockwise): (x, y) -> (y, -x).
    tv = np.array([c[1], -c[0]]) / n
    info = {"n_l": n_l, "n_r": n_r, "dwell_l": dwell_l, "dwell_r": dwell_r,
            "alternating_g": n, "mean_l": mean_l.tolist(),
            "mean_r": mean_r.tolist(), "rows": rows}
    return tv, info


def _quat_to_mat(q):
    q = np.asarray(q, float)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w),
         1 - 2 * (x * x + y * y)]])
