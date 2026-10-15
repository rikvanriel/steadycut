"""Corridor flow: stereo DIS + confidence (Task 4.1), single-map correction.

Framing (measured, load-bearing): the two lenses share a rigid baseline at
the same instant, so gyro sees ZERO inter-lens motion. DIS solves the stereo
disparity; gyro priors TEMPORAL flow across frames and RS skew only — never
the parallax. Signature carries (gyro_series, frame_times) for the temporal
term; snap alignment follows reframe.py:282-296.

Belt: equirect corridor strips in SHARED sphere coords (caller aligns B).
Locally rotation-free since both views are sphere-registered; the residual
is the disparity itself. Disparity pin (disparity_pin.py, Jan26/112):
median 0, p10 -28.7, max |.| 33 px — the stage must cover +-35 px, so the
old 6 px synthetic pin is retired.

Correction is SINGLE-map (review catch: "maps" plural risks double
correction): warp B's samples toward A, alpha from flow confidence.
"""
from __future__ import annotations

import numpy as np

FLOW_RANGE_PX = 35.0  # measured pin; DIS stage must cover this


def corridor_pair(eqA, eqB, seam_x, halfw=96):
    """Crop both strips. Caller guarantees shared sphere coords."""
    x0 = max(0, seam_x - halfw)
    x1 = min(eqA.shape[1], seam_x + halfw)
    return eqA[:, x0:x1], eqB[:, x0:x1], x0


def flow_strip(A, B, preset=None):
    """Dense A->B flow (float32, px). B_corrected(x) = B(x + flow(x))."""
    cv2 = __import__("cv2")
    dis = cv2.DISOpticalFlow_create(preset or cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    gA = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    gB = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
    flow = dis.calc(gA, gB, None)
    return np.asarray(flow)


def flow_confidence(A, B, flow, win=9, valid=None):
    """Confidence 0..1 per pixel: local texture strength in A (flat snow has
    no correspondence to find — low confidence forces the hard-switch
    fallback downstream, never a hallucinated warp)."""
    cv2 = __import__("cv2")
    g = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    ty = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.blur(np.hypot(tx, ty), (win, win))
    # Fixed scale, never per-image percentiles (self-normalization would
    # promote the noisiest snow pixel to 1.0): trunk gradients run 50-200
    # DN/px, snow noise ~5. REF_GRAD = 25 DN/px separates them.
    conf = np.clip(mag / 25.0, 0, 1)
    if valid is not None:
        conf = np.where(valid, conf, 0.0)  # no data, no correspondence
    return conf.astype(np.float32)


def joint_valid_strip(seam_x, halfw, fov_a=185.0, fov_b=183.5, h=1024,
                      w=2048):
    """Joint-valid mask for a sphere-coordinate corridor strip, in V360
    OUTPUT frame (measured: track0/yaw0 centers output-1024, track1/yaw180
    centers output-0; lens_geometry's internal az=0 sits at output-1024, so
    its validity is 180 deg off here — this function does NOT reuse it).

    Analytic incidence vs half-FOV per lens, no black-pixel thresholds
    (scene confounds those). The overlap is only ~24 px wide at fitted
    FOVs — strips wider than that are mostly one-sided, and DIS matched
    against black edges there (measured blue-smear failure)."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    az = (xs / w) * 2 * np.pi
    el = (0.5 - ys / h) * np.pi
    dx = np.cos(el) * np.sin(az)
    dy = np.sin(el)
    dz = np.cos(el) * np.cos(az)
    # output-frame axes: track0 (fov_a) at az=pi (x=1024), track1 at az=0.
    inc_a = np.arccos(np.clip(-dz, -1, 1))
    inc_b = np.arccos(np.clip(dz, -1, 1))
    va = inc_a <= np.radians(fov_a) / 2
    vb = inc_b <= np.radians(fov_b) / 2
    x0 = max(0, seam_x - halfw)
    x1 = min(w, seam_x + halfw)
    return (va & vb)[:, x0:x1]


def warp_strip(B, flow):
    """Resample B by the flow field (still-level validation + tests)."""
    cv2 = __import__("cv2")
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(B, xs + flow[:, :, 0], ys + flow[:, :, 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def compose_corrected_B(eqB, seam_x, halfw, flow, conf):
    """Confidence-gated flow correction of the full B equirect (Task 4.2).

    B_corr = remap(B, grid + flow * conf * feather): full warp at the seam
    where confidence is high, decaying to unwarped B at the corridor edge
    (continuity with the uncorrected field outside). Zero confidence rows
    return B bit-identically — the violent-motion fallback is structural,
    not a branch. flow/conf are strip-arrays; embedding is full-frame.
    """
    cv2 = __import__("cv2")
    H, W = eqB.shape[:2]
    h, w = flow.shape[:2]
    x0 = max(0, seam_x - halfw)
    full_f = np.zeros((H, W, 2), np.float32)
    full_c = np.zeros((H, W), np.float32)
    full_f[:, x0:x0 + w] = flow
    full_c[:, x0:x0 + w] = conf
    xs = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)
    feather = np.zeros((H, W), np.float32)
    d = np.abs(xs - seam_x) / halfw
    feather = np.where(d < 1, 0.5 * (1 + np.cos(np.pi * d)), 0.0)
    strength = (full_c * feather)[:, :, None]
    ys = np.arange(H, dtype=np.float32)[:, None].repeat(W, axis=1)
    return cv2.remap(eqB, xs + full_f[:, :, 0] * strength[:, :, 0],
                     ys + full_f[:, :, 1] * strength[:, :, 0],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
