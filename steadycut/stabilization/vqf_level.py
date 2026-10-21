"""Time-varying roll leveling from VQF (gravity-leveling rev2, step 2/3).

Split of duties (axis-swamp avoidance):
- VQF supplies the ATTITUDE (gyro-carried, sees through turn
  centripetal via earth-frame lowpass + offline forward-backward).
- `shift_from_gravity` (eye-settled on Jan +66.6 / Feb -104, NOT
  derived) supplies the CONVENTION: per-sample shift from VQF-predicted
  body gravity, same formula as the static path. On straights this
  reduces to the static shift by construction; in turns the prediction
  ( unlike raw accel ) carries no centripetal.
- Freeze through sustained turns (VQF yaw RATE via turn_state with 2.5 s
  entry + backward dilation, raised-cosine rejoin) holds pre-turn level
  through sustained one-way lateral that VQF itself admits. No gyro
  integration anywhere (coning/bias diverge tens of degrees on real
  data); yaw rate is gyro-dominated and drift-free. Documented leak:
  carve with counter-rotated head reads straight (accepted, rare).
No Euler extraction, no hand-derived signs anywhere.

Known limit (pinned in tests): VQF's bias estimator absorbs part of
SLOW oscillations (~25% at 0.05 Hz) — slow rolls under-corrected,
never invented. Eye decides magnitude on footage.
"""
import numpy as np


def quat_to_mat(q):
    """3x3 body->world from (w, x, y, z)."""
    q = np.asarray(q, float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))


def _handed_signs(axis_map):
    """Signed (col, sign) triples with the det-derived handedness fix.

    Shared by level_shifts and bridge_spans: the X4 map frame is
    left-handed (det -1, signs serve the flipped sphere); negate Z iff
    the signed frame has det -1 so VQF gets a right-handed z-up frame.
    """
    p_col, p_sign = axis_map.pitch
    r_col, r_sign = axis_map.roll
    g_col, g_sign = axis_map.yaw
    _perm = [p_col, r_col, g_col]
    _inv = sum(1 for i in range(3) for j in range(i + 1, 3)
               if _perm[i] > _perm[j])
    if p_sign * r_sign * g_sign * (-1) ** _inv < 0:
        g_sign = -g_sign
    return (p_col, p_sign), (r_col, r_sign), (g_col, g_sign)


def _body_up_from_R(R):
    """Unit up-vector in body frame from a body->world matrix.

    VQF's world is z-up; the body-frame components of world +z are the
    third ROW of R (R.T @ z). Pinned against static accel below.
    """
    u = np.asarray(R)[2, :]
    n = float(np.linalg.norm(u))
    return u / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def _carry_up(u0, w_rad, dt, sign):
    """Carry a body-frame up-vector through gyro samples.

    sign=+1 integrates forward (u[k+1] = rot(-w*dt) u[k]: the body
    rotates by +w*dt under a fixed world vector); sign=-1 backward.
    """
    from steadycut.stabilization import orientation as O
    u = np.array(u0, float)
    out = np.zeros((len(w_rad), 3))
    for k in range(len(w_rad)):
        if (sign > 0 and k == 0) or (sign < 0 and k == len(w_rad) - 1):
            out[k] = u
            continue
        j = k - 1 if sign > 0 else k
        dq = O.from_rotvec((-sign * w_rad[j] * dt[j])[None, :])[0]
        qv = np.concatenate([[0.0], u])
        u = O._mul(O._mul(dq, qv), O._conj(dq))[1:]
        u = u / max(float(np.linalg.norm(u)), 1e-12)
        out[k] = u
    if sign < 0:
        # built backward from the post anchor: reverse to time order
        out = out[::-1]
    return out


def _best_anchor(score, edge, side, dt, n):
    """Lowest-score 1 s window within 5 s of a span edge.

    The adjacent second is often transitional (turn exit/entry); the
    anchor should be the second VQF actually settled in. Returns
    (center_index, score) or (None, inf).
    """
    wlen = int(round(1.0 / dt))
    span = int(round(5.0 / dt))
    if side < 0:
        lo, hi = max(0, edge - span), edge
    else:
        lo, hi = edge, min(n, edge + span)
    best, bq = None, float("inf")
    for s in range(lo, max(lo, hi - wlen)):
        q = float(score[s:s + wlen].mean())
        if q < bq:
            best, bq = s + wlen // 2, q
    return best, bq


def bridge_spans(t, gyr, acc, spans, R, axis_map, dt, flips=()):
    """Gyro-carry the world-fixed up reference through frozen spans.

    Anchors at turn-transition FLIPS (yaw sign changes): lateral-g
    passes through zero there, so median measured accel over +-0.3 s
    is true gravity with no VQF attitude in the path. (Anchoring via
    R[c].T @ g_world inherits VQF's local tilt error ~10-17 deg under
    sustained lateral: VQF absorbs lateral as gravity.) Each sliver
    carries forward from its pre-flip and backward from its post-flip,
    cross-faded (traverse with known endpoints). One flip: one-way
    carry. No flip: innov-anchor fallback. Returns (synth, n_b, n_u).
    Unbridgeable spans keep the old constant hold.
    """
    (p_col, p_sign), (r_col, r_sign), (g_col,
                                       g_sign) = _handed_signs(axis_map)
    body_of = [(p_col, p_sign), (r_col, r_sign), (g_col, g_sign)]
    w = np.radians(np.asarray(gyr, float))  # raw columns
    # body-frame rate in the VQF-built frame order
    wb = np.column_stack([s * w[:, c] for c, s in body_of])
    yaw_w = wb[:, 2]
    # per-sample VQF innovation: angle between VQF body-up and measured
    # accel (up-positive frame). Only used for the no-flip fallback.
    acc_vqf = np.column_stack([s * np.asarray(acc, float)[:, c]
                               for c, s in body_of])
    an = np.linalg.norm(acc_vqf, axis=1, keepdims=True)
    up_v = np.asarray(R)[:, 2, :]
    innov = np.degrees(np.arccos(np.clip(
        (up_v * (acc_vqf / np.maximum(an, 1e-12))).sum(axis=1),
        -1.0, 1.0)))
    au = acc_vqf / np.maximum(an, 1e-12)

    def _flip_up(k):
        # narrow window + SYMMETRY gate: the flip must be a clean
        # zero-crossing (1 s means on both sides disagree in sign,
        # both clearing rate) so lateral cancels in the median. Rate
        # through the instant is irrelevant — a fast clean crossing
        # cancels, a slow muddy one does not. Returns (up, ok).
        hw = max(1, int(round(0.15 / dt)))
        lo, hi = max(0, k - hw), min(len(t), k + hw)
        m = np.median(au[lo:hi], axis=0)
        n = float(np.linalg.norm(m))
        u = m / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])
        w1 = max(1, int(round(1.0 / dt)))
        lm = float(np.degrees(yaw_w[max(0, k - w1):k]).mean())
        rm = float(np.degrees(yaw_w[k:min(len(t), k + w1)]).mean())
        ok = (lm > 6.0 and rm < -6.0) or (lm < -6.0 and rm > 6.0)
        return u, ok

    synth = {}
    n_b = n_u = 0
    n = len(t)
    for a, b in spans:
        # cap the carry: gyro coning/drift through violent whip bounds
        # honest bridging to ~tens of seconds; longer spans keep the
        # hold (counted unbridged, may trip refusal).
        if (b + 1 - a) * dt > 20.0:
            n_u += 1
            continue
        _fa = [f for f in flips if f <= a
                 and a - f < int(round(5.0 / dt))]
        _fb = [f for f in flips if f >= b
                 and f - b < int(round(5.0 / dt))]
        fa = max(_fa) if _fa else None
        fb = min(_fb) if _fb else None
        if fa is not None or fb is not None:
            # symmetry gate lives inside _flip_up (clean zero-crossing
            # cancels lateral in the median); no rate bar here.
            U1 = U2 = None
            if fa is not None:
                ua, oka = _flip_up(fa)
                if oka:
                    seg = np.arange(fa, b + 1)
                    full = _carry_up(ua, wb[seg],
                                     np.gradient(t[seg]), +1)
                    U1 = full[a - fa:]
            if fb is not None:
                ub, okb = _flip_up(fb)
                if okb:
                    seg = np.arange(a, fb + 1)
                    full = _carry_up(ub, wb[seg],
                                     np.gradient(t[seg]), -1)
                    U2 = full[:b + 1 - a]
            if U1 is None and U2 is None:
                # gate-rejected flips (muddy crossings): fall through
                # to the innov-anchor path below instead of refusing —
                # the gate working is not missing gravity. Only count
                # unbridged when nothing anchors at all.
                pass
            else:
                if U1 is not None and U2 is not None:
                    wgt = np.linspace(0, 1, b + 1 - a)[:, None]
                    U = (1 - wgt) * U1 + wgt * U2
                else:
                    U = U1 if U1 is not None else U2
                assert U is not None
                U = U / np.maximum(
                    np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
                G = np.zeros((b + 1 - a, 3))
                for i, (cc, s) in enumerate(body_of):
                    G[:, cc] = s * (9.81 * U[:, i])
                synth[(a, b)] = G
                n_b += 1
                continue
        # settled = gate-calm AND directly slow: the gate's entry delay
        # leaves the first second calm while already spinning, so an
        # anchor there would carry a corrupt reference (test b pin).
        # Search the lowest-innovation nearby second (adjacent is
        # transitional). Accept under 10 deg mean disagreement.
        ai, aq = _best_anchor(innov, a, -1, dt, n)
        bi, bq = _best_anchor(innov, b + 1, +1, dt, n)
        pre_ok = ai is not None and aq < 10.0
        post_ok = bi is not None and bq < 10.0
        if pre_ok:
            assert ai is not None
            u0 = _body_up_from_R(R[ai])
            seg = np.arange(ai, b + 1)
            U = _carry_up(u0, wb[seg], np.gradient(t[seg]), +1)
            U = U[a - ai:]
        elif post_ok:
            assert bi is not None
            u0 = _body_up_from_R(R[bi])
            seg = np.arange(a, bi + 1)
            U = _carry_up(u0, wb[seg], np.gradient(t[seg]), -1)
            U = U[:b + 1 - a]
        else:
            n_u += 1
            continue
        # synth raw-column accel from carried up, in the pipeline's
        # up-positive convention (same linear map as gvec above:
        # gvec[:, col] = up_i / sign; scale is ratio-free in atan2)
        G = np.zeros((b + 1 - a, 3))
        for i, (c, s) in enumerate(body_of):
            G[:, c] = s * (9.81 * U[:, i])
        synth[(a, b)] = G
        n_b += 1
    return synth, n_b, n_u


def resample_shifts(path_t, imu_t, shifts, start):
    """Shifts onto ControlPoint timestamps. path_t is CLIP-RELATIVE
    (imu_time_s - start + sync offset); imu_t is absolute. Mixing the
    clocks parks every point left of xp[0] and applies one constant
    from before the window — subtract start here.
    """
    import numpy as _np
    pt = _np.asarray(path_t, float)
    xp = _np.asarray(imu_t, float) - float(start)
    return _np.interp(pt, xp, _np.asarray(shifts, float),
                      left=shifts[0], right=shifts[-1])


def level_shifts(t, gyro, accel, axis_map, sel=None):
    """Per-sample roll-slot shifts (deg) from VQF-predicted body gravity.

    Returns (shifts, info); info carries turn_frac/n_turns for the cert
    (rev2: duty-cycle visibility) and the p95 shift size.

    sel: optional boolean mask scoping turn_frac/refusal to the delivery
    window. Pass WIDE arrays (attitude needs context: q0 from a violent
    10 s window poisons the whole prediction) with sel on the window;
    freeze spans are computed wide so rejoins at window edges are sane.
    """
    import vqf
    from steadycut.stabilization.level import tilt_from_gravity
    t = np.asarray(t, float)
    gyro = np.asarray(gyro, float)
    accel = np.asarray(accel, float)
    dt = float(np.median(np.diff(t)))
    p_col, p_sign = axis_map.pitch
    r_col, r_sign = axis_map.roll
    g_col, g_sign = axis_map.yaw  # gravity lives in the vendor yaw column
    # Handedness fix (review R2): shared helper below derives it; see
    # _handed_signs. Per-body safe: a det +1 map keeps its signs;
    # |roll|>90 still refuses below.
    (p_col, p_sign), (r_col, r_sign), (g_col,
                                       g_sign) = _handed_signs(axis_map)

    def _build(ps, rs, gs):
        g = np.radians(np.column_stack((
            ps * gyro[:, p_col], rs * gyro[:, r_col],
            gs * gyro[:, g_col])))
        a = np.column_stack((
            ps * accel[:, p_col], rs * accel[:, r_col],
            gs * accel[:, g_col]))
        return g, a

    gyr, acc = _build(p_sign, r_sign, g_sign)
    out = vqf.offlineVQF(gyr, acc, None, dt)
    R = quat_to_mat(np.asarray(out["quat6D"], float))  # body -> world
    # absurdity guard: a bike/ski camera never rides upside-down; if the
    # winning branch still reads |roll| > 90 deg the frame is inverted
    # (map signs are renderer-frame) — refuse, do not ship it.
    _w, _x, _y, _z = np.asarray(out["quat6D"], float).T
    _roll = np.degrees(np.arctan2(2 * (_w * _x + _y * _z),
                                  1 - 2 * (_x * _x + _y * _y)))
    # body components of world-up: v_b with R v_b = e_z, i.e. row 2 of R
    up_body = np.einsum("...ji,...j->...i", R, np.array([0.0, 0.0, 1.0]))
    # VQF frame (X,Y,Z) = signed (pitch, roll, gravity) columns -> raw
    gvec = np.empty_like(up_body)
    gvec[:, p_col] = up_body[:, 0] / p_sign
    gvec[:, r_col] = up_body[:, 1] / r_sign
    gvec[:, g_col] = up_body[:, 2] / g_sign
    raw = np.array([tilt_from_gravity(gvec[i], axis_map)[1]
                    for i in range(len(t))])
    raw_pitch = np.array([tilt_from_gravity(gvec[i], axis_map)[0]
                          for i in range(len(t))])
    w, x, y, z = np.asarray(out["quat6D"], float).T
    # freeze gate: SUSTAINED body yaw rate (turn_state: 2.5 s entry, so
    # spikes and glances never enter), backward-dilated for the entry.
    # Keyed on BODY-column yaw, not VQF yaw: under lateral-g VQF's own
    # attitude absorbs rotation into tilt error and its yaw never
    # sustains (measured: body legs +-20 deg/s for 10 s each read mush
    # in VQF yaw). Mount roll hides part of world yaw from the helmet
    # column (~40% visible), but real turns still clear 6 deg/s by far;
    # glance rejection via the 2.5 s sustain is unchanged.
    # No gyro integration anywhere: coning and bias make a predicted-up
    # innov diverge tens of degrees on real data (measured), while yaw
    # RATE is gyro-dominated and drift-free. Documented leaks (review R2):
    # a carve with counter-rotated head reads straight (accepted, rare);
    # sub-threshold rate with real lateral leaks (5.5 dps + 2.0 m/s2
    # leaks ~8.6 of ~11.5 deg — wide GS turns live there; gate on
    # v*omega when speed exists, else accept).
    import steadycut.stabilization.orientation as _O
    _wb = np.column_stack([s * gyro[:, c]
                           for c, s in [(p_col, p_sign), (r_col, r_sign),
                                         (g_col, g_sign)]])
    yaw_w = _wb[:, 2]
    back = int(round(2.5 / dt))
    turn_m = _O.turn_state(t, yaw_w, smooth_s=2.0)
    # dilate each contiguous span backward (entry coverage), offline luxury
    idx = np.flatnonzero(turn_m)
    if len(idx):
        cuts = np.flatnonzero(np.diff(idx) > 1)
        starts = np.concatenate([[idx[0]], idx[cuts + 1]])
        for s in starts:
            turn_m[max(0, s - back):s] = True
    frozen = turn_m
    # hold pre-freeze value through each span, raised-cosine rejoin
    shifts = raw.copy()
    pitch_shifts = raw_pitch.copy()
    n = len(t)
    i = 0
    spans = []
    while i < n:
        if not frozen[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and frozen[j + 1]:
            j += 1
        spans.append((i, j))
        i = j + 1
    # split spans at genuine direction flips (slalom legs): a span
    # covering L-then-R would merge into one long carry that trips the
    # 20 s cap. Cut where 1 s means on both sides disagree in sign and
    # both clear 6 deg/s (whip flicker near zero never cuts).
    _sm = np.convolve(yaw_w, np.ones(max(1, int(round(2.0 / dt))))
                      / max(1, int(round(2.0 / dt))), mode="same")
    _w1 = max(1, int(round(1.0 / dt)))
    _split = []
    _flips: list = []
    for a, b in spans:
        _cands = []
        for k in range(a + _w1, b - _w1 + 1):
            _lm = _sm[k - _w1:k].mean()
            _rm = _sm[k:k + _w1].mean()
            if _lm > 6.0 and _rm < -6.0 or _lm < -6.0 and _rm > 6.0:
                _cands.append((k, abs(_lm) + abs(_rm)))
        # non-maximum suppression (>=3 s apart): one cut per flip,
        # strongest opposition wins; whip flicker never qualifies.
        # Cuts double as zero-lateral anchor instants (flip = edge
        # change, lateral passes through zero) for bridge_spans.
        _cands.sort(key=lambda c: -c[1])
        _cuts = [a]
        for k, _ in _cands:
            if all(abs(k - c) > 3 * _w1 for c in _cuts) and k - a > _w1:
                _cuts.append(k)
        _cuts = sorted(_cuts) + [b + 1]
        _flips.extend(_cuts[1:-1])
        for s0, s1 in zip(_cuts[:-1], _cuts[1:]):
            if s1 - 1 >= s0:
                _split.append((s0, s1 - 1))
    spans = _split
    # anchor-bridge: gyro-carry the world-fixed up through frozen
    # spans, anchored at flip instants (bridge_spans docstring);
    # unbridgeable spans keep the constant hold (no new failure mode)
    bridged, n_b, n_u = bridge_spans(t, gyro, accel, spans, R,
                                     axis_map, dt, flips=_flips)
    for a, b in spans:
        G = bridged.get((a, b))
        if G is not None:
            # gyro-carried world reference through the turn: per-sample
            # (pitch, roll) tilts from the carried up-vector via the
            # settled formula. Start-blend from the previous final value
            # so adjacent spans join continuously.
            car = np.array([tilt_from_gravity(gv, axis_map)[1]
                            for gv in G])
            carp = np.array([tilt_from_gravity(gv, axis_map)[0]
                             for gv in G])
            m0 = min(a, int(round(2.0 / dt)), len(car))
            if m0 > 1 and a > 0:
                wgt = 0.5 - 0.5 * np.cos(np.pi * np.arange(m0) / m0)
                car[:m0] = (1 - wgt) * shifts[a - 1] + wgt * car[:m0]
                carp[:m0] = (1 - wgt) * pitch_shifts[a - 1] + wgt * carp[:m0]
            shifts[a:b + 1] = car
            pitch_shifts[a:b + 1] = carp
            continue
        # unbridgeable (no settled anchor either side): constant hold
        lo = max(0, a - int(round(1.0 / dt)))
        pre = raw[lo:a] if a > lo else np.array([])
        hold = float(np.median(pre)) if len(pre) else (
            raw[b + 1] if b + 1 < n else float(np.median(raw)))
        prep = raw_pitch[lo:a] if a > lo else np.array([])
        holdp = float(np.median(prep)) if len(prep) else (
            raw_pitch[b + 1] if b + 1 < n else float(np.median(raw_pitch)))
        shifts[a:b + 1] = hold
        pitch_shifts[a:b + 1] = holdp
        m = min(n - b - 1, int(round(2.0 / dt)))
        if m > 1 and b + 1 < n:
            wgt = 0.5 - 0.5 * np.cos(np.pi * np.arange(m) / m)
            shifts[b + 1:b + 1 + m] = ((1 - wgt) * hold + wgt * raw[b + 1:b + 1 + m])
            pitch_shifts[b + 1:b + 1 + m] = (
                (1 - wgt) * holdp + wgt * raw_pitch[b + 1:b + 1 + m])
    # level reference carries no shake: the gyro path owns fast motion.
    # Zero-phase lowpass at level timescales (vibration-speed noise in
    # VQF attitude otherwise lands straight in the roll slots: measured
    # 24 deg/frame pump on Mar05 from +-8 deg unsmoothed shifts).
    from steadycut.stabilization.orientation import _smooth_1d
    shifts = _smooth_1d(shifts, t, 3.0)
    pitch_shifts = _smooth_1d(pitch_shifts, t, 3.0)
    if sel is None:
        lo, hi = 0, n
    else:
        idx = np.flatnonzero(np.asarray(sel, bool))
        lo, hi = (int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, 0)
    frozen_n = sum(min(b, hi - 1) - max(a, lo) + 1
                   for a, b in spans if b >= lo and a < hi)
    scope = float(max(1, hi - lo))
    turn_frac = float(frozen_n / scope)
    info = {"turn_frac": turn_frac,
            "n_turns": len(spans),
            "n_bridged": n_b,
            "n_unbridged": n_u,
            "shift_p95": float(np.percentile(np.abs(shifts[lo:hi]), 95))
            if hi > lo else 0.0,
            "pitch_p95": float(np.percentile(np.abs(pitch_shifts[lo:hi]), 95))
            if hi > lo else 0.0,
            "raw_p95": float(np.percentile(np.abs(raw[lo:hi]), 95))
            if hi > lo else 0.0,
            "rest_frac": float(np.mean(out["restDetected"]))}
    if turn_frac > 0.5 and n_u > 0:
        # open-loop most of the run with spans no anchor reaches:
        # refuse rather than pump. Caller falls back to static.
        # Fully-bridged windows apply: each span carries its own
        # anchors, so drift does not chain across spans.
        info["refused"] = (f"turn_frac {turn_frac:.2f} > 0.50 with "
                           f"{n_u} unbridged spans: no gravity reference "
                           "in this window")
    if abs(float(np.median(_roll))) > 90.0:
        info["refused"] = ("roll median %.0f deg: inverted attitude" %
                           float(np.median(_roll)))
    return shifts, pitch_shifts, info
