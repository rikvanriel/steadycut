"""Content hold for aim wander: heading-relative yaw offset from trackable background.

Two-layer design (reviewer t_d2d608e7): the gyro trend owns heading,
this module owns a small yaw-only content offset that holds still
whatever the gyro cannot see (slow aim wander). It NEVER drives the
view by itself: no passing region = refusal = today's path exactly.

Reference selection is by GYRO CONSISTENCY, not trackability: a
candidate region's high-band raw-view motion must match the
far-field flow the gyro predicts. Camera-fixed parts (bars, arms,
seam) show ~0 raw motion and fail; near-field shows excess parallax
and fail; independently moving subjects fail. Position is measured
by registering each frame to a keyframe (re-anchor on loss), never
by cumulative sum (which random-walks).

Gates (C7 — pinned here first, never retuned after seeing footage):
  INLIER_FLOOR = 150, LIFETIME_MIN = 20 frames, SALIENCE_FLOOR = 8.0
  (L2 color variance), RESID_CAP = 2.0 px/frame median residual,
  CONSIST_MIN_R = 0.6, CONSIST_SCALE = (0.5, 2.0),
  RATE_LIMIT = 30.0 deg/s, OFFSET_BOUND = 10.0 deg,
  BLEED_RATE = 5.0 deg/s, SETTLE_T = 2.0 s, LEASH_TAU = 2.0 s.
"""

import numpy as np

# Gate numbers, pinned in tests before any bench runs (C7).
# RESID is residual drift: median |tracked - gyro-predicted|
# velocity. Absolute frame motion is NOT gated (a far region
# rotating with the camera moves several px/frame and must pass);
# only motion the gyro cannot explain counts.
INLIER_FLOOR = 150
LIFETIME_MIN = 20
SALIENCE_FLOOR = 8.0
RESID_CAP = 2.0
CONSIST_MIN_R = 0.6
CONSIST_SCALE = (0.5, 2.0)
RATE_LIMIT = 30.0
OFFSET_BOUND = 10.0
BLEED_RATE = 5.0
SETTLE_T = 2.0
LEASH_TAU = 2.0


def _highpass(x, fps, cutoff_hz=0.5):
    """First-difference high-pass (removes wander, keeps shake)."""
    return np.diff(x, prepend=x[0])


def gyro_consistency(tracked_vx, tracked_vy, gyro_vx, gyro_vy):
    """Check a region's motion against gyro-predicted far-field flow.

    All inputs are per-frame pixel velocities (already divided by dt).
    Returns (ok, r, scale, reason). Camera-fixed regions fail on
    scale (~0); near-field fails on scale (>>1); subjects fail on r.
    """
    t = np.asarray(tracked_vx) + 1j * np.asarray(tracked_vy)
    g = np.asarray(gyro_vx) + 1j * np.asarray(gyro_vy)
    t -= np.median(t.real) + 1j * np.median(t.imag)
    g -= np.median(g.real) + 1j * np.median(g.imag)
    denom = float(np.sqrt((np.abs(t) ** 2).sum() * (np.abs(g) ** 2).sum()))
    if denom <= 0:
        return False, 0.0, 0.0, "empty-motion"
    r = float((t.real * g.real + t.imag * g.imag).sum() / denom)
    # Sign-agnostic: axis->pixel sign conventions vary, but only
    # rotation-locked content mirrors the gyro either way.
    r = abs(r)
    scale = float(np.sqrt((np.abs(t) ** 2).sum() / (np.abs(g) ** 2).sum()))
    if r < CONSIST_MIN_R:
        return False, r, scale, "uncorrelated"
    if not (CONSIST_SCALE[0] <= scale <= CONSIST_SCALE[1]):
        return False, r, scale, "scale-off"
    return True, r, scale, "ok"


def select_reference(regions):
    """Pick the hold reference from scored candidate regions.

    regions: list of dicts with keys inliers (median), lifetime
    (frames), salience, drift (median px/frame), consist_ok (bool
    from gyro_consistency). Returns (index, reason). Refusal is a
    normal outcome, never an error.
    """
    best, best_key = None, None
    for i, r in enumerate(regions):
        if r["inliers"] < INLIER_FLOOR:
            continue
        if r["lifetime"] < LIFETIME_MIN:
            continue
        if r["salience"] < SALIENCE_FLOOR:
            continue
        if r["resid"] > RESID_CAP:
            continue
        if not r["consist_ok"]:
            continue
        key = (r["inliers"], r["salience"])
        if best_key is None or key > best_key:
            best, best_key = i, key
    if best is None:
        return None, "no-region-passes"
    return best, "ok"


def hold_offset(ref_yaw, fps, bound=OFFSET_BOUND,
                rate_limit=RATE_LIMIT, bleed=BLEED_RATE):
    """Leashed counter-pan with saturation-then-bleed (C4).

    ref_yaw: per-frame reference heading (deg, keyframe-registered).
    Returns per-frame yaw offset (deg): -leashed drift, rate-limited,
    saturating at bound, then bleeding to 0 (never a frozen offset,
    never a saccade).
    """
    ref_yaw = np.asarray(ref_yaw, dtype=float)
    n = len(ref_yaw)
    a = 1 - np.exp(-1 / (fps * LEASH_TAU))
    slow = np.zeros(n)
    for i in range(1, n):
        slow[i] = slow[i - 1] + a * (ref_yaw[i] - slow[i - 1])
    raw = -(slow - slow[0])
    max_step = rate_limit / fps
    out = np.zeros(n)
    for i in range(1, n):
        step = np.clip(raw[i] - out[i - 1], -max_step, max_step)
        out[i] = out[i - 1] + step
        if abs(out[i]) > bound:
            # Bleed back toward zero instead of freezing (C4).
            back = np.sign(out[i]) * bleed / fps
            out[i] -= back
            if abs(out[i]) > bound:
                out[i] = np.sign(out[i]) * bound
    return out


def settle_time(offset, fps, thresh_deg=1.0):
    """Frames after the last saturation until |offset| < thresh."""
    over = np.nonzero(np.abs(offset) >= OFFSET_BOUND)[0]
    if len(over) == 0:
        return 0
    last = over[-1]
    under = np.nonzero(np.abs(offset[last:]) < thresh_deg)[0]
    if len(under) == 0:
        return len(offset) - last
    return int(under[0])
