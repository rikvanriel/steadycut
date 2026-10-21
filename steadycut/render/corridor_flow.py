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


def sift_field(A, B, band_h=40, min_n=3, dx_lo=8, dx_hi=150,
               max_dy=10, seam_x=None, scan_w=200, search_w=150):
    """Sparse SIFT displacement field A->B (float32, px) + confidence.

    NCC plateaus on forest; descriptors vote clean displacements
    (measured trunk: 13/15 good matches at +50/+55 px). Per row-band
    median dx over ratio-tested matches (dy-bounded); bands with <min_n
    matches get 0 displacement (compose falls back to plain blend there).
    Returns (field, conf) over the input pair shape; conf = min(1, n/8).
    Covers copies outside the joint too (warp moves B's copy onto A's —
    selection cannot help there: both territories must show).
    Votes must STRADDLE the seam (A-point left-adjacent, B-point
    right-adjacent): forest tree-spacing fakes displacements anywhere."""
    cv2 = __import__("cv2")
    gA = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    gB = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
    H, W = gA.shape
    field = np.zeros((H, W, 2), np.float32)
    conf = np.zeros((H, W), np.float32)
    sift = cv2.SIFT_create()
    for by0 in range(0, H, band_h):
        by1 = min(H, by0 + band_h)
        kA, dA = sift.detectAndCompute(gA[by0:by1], None)
        kB, dB = sift.detectAndCompute(gB[by0:by1], None)
        if dA is None or dB is None or len(kA) < 2 or len(kB) < 2:
            continue
        dxs, bxs = [], []
        for pair in cv2.BFMatcher().knnMatch(dA, dB, k=2):
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance >= 0.75 * second.distance:
                continue
            ax = kA[first.queryIdx].pt[0]
            bx = kB[first.trainIdx].pt[0]
            dx = bx - ax
            dy = (kB[first.trainIdx].pt[1] - kA[first.queryIdx].pt[1])
            if not (dx_lo <= dx <= dx_hi and abs(dy) <= max_dy):
                continue
            if seam_x is not None:
                # straddle: A-point left-adjacent, B-point right-adjacent
                if not (seam_x - scan_w <= ax <= seam_x
                        and seam_x <= bx <= seam_x + search_w):
                    continue
            dxs.append(dx)
            bxs.append(bx)
        if len(dxs) < min_n:
            continue
        # x-localized: warp only around the matched copy span (median bx
        # +- 80, cosine falloff). Full-row shifts would misalign innocent
        # content elsewhere in the corridor.
        dx_med = float(np.median(dxs))
        cx0 = float(np.median(bxs))
        xs = np.arange(W, dtype=np.float32)
        win = np.clip(1 - np.abs(xs - cx0) / 80.0, 0, 1)
        win = 0.5 - 0.5 * np.cos(np.clip(win, 0, 1) * np.pi)
        field[by0:by1, :] = (dx_med, 0.0)
        conf[by0:by1, :] = win[None, :] * min(1.0, len(dxs) / 8.0)
    return field, conf


def block_flow(A, B, bw=48, bh=96, sx=24, sy=48, rx=96, ry=8,
               min_score=0.75):
    """Block-match A->B flow (float32, px): B_corrected(x) = B(x + flow(x)).

    DIS hallucinates on repetitive trunks at 37 px (measured 55.5 px median
    vs 37 truth, vetoed 0.0 conf). Big translated blocks with a mutual gate
    instead: each block must match back within 4 px. Returns (flow, conf)
    over the input pair shape; conf = mutual-match score or 0. Unmatched
    (bland/occluded) blocks stay 0 — compose falls back to plain blend.
    Search is wide-horizontal (parallax) x narrow-vertical (residue)."""
    cv2 = __import__("cv2")
    gA = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gB = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gA.shape
    flow = np.zeros((H, W, 2), np.float32)
    conf = np.zeros((H, W), np.float32)
    for cy in range(bh // 2, H - bh // 2, sy):
        for cx in range(bw // 2, W - bw // 2, sx):
            t = gA[cy - bh // 2:cy + bh // 2, cx - bw // 2:cx + bw // 2]
            if float(t.std()) < 8:
                continue
            x0, x1 = max(0, cx - rx), min(W, cx + rx + bw)
            y0, y1 = max(0, cy - ry), min(H, cy + ry + bh)
            if x1 - x0 < bw or y1 - y0 < bh:
                continue  # clipped search smaller than template: skip
            s = gB[y0:y1, x0:x1]
            r = cv2.matchTemplate(s, t, cv2.TM_CCOEFF_NORMED)
            _, best, _, ml = cv2.minMaxLoc(r)
            if best < min_score:
                continue
            mx = x0 + ml[0] + bw // 2 - cx
            my = y0 + ml[1] + bh // 2 - cy
            # mutual gate: B's match must point back within 4 px
            tb = gB[cy + my - bh // 2:cy + my + bh // 2,
                    cx + mx - bw // 2:cx + mx + bw // 2]
            if tb.shape != (bh, bw):
                continue
            sa = gA[max(0, cy - ry):cy + ry + bh,
                    max(0, cx - rx):cx + rx + bw]
            if sa.shape[0] < bh or sa.shape[1] < bw:
                continue
            rb = cv2.matchTemplate(sa, tb, cv2.TM_CCOEFF_NORMED)
            _, best2, _, ml2 = cv2.minMaxLoc(rb)
            jx = max(0, cx - rx) + ml2[0] + bw // 2 - (cx + mx)
            jy = max(0, cy - ry) + ml2[1] + bh // 2 - (cy + my)
            if abs(jx) > 4 or abs(jy) > 4 or best2 < min_score:
                continue
            flow[cy - sy // 2:cy + sy // 2,
                 cx - sx // 2:cx + sx // 2] = (mx, my)
            conf[cy - sy // 2:cy + sy // 2,
                 cx - sx // 2:cx + sx // 2] = min(best, best2)
    return flow, conf


def flow_strip(A, B, preset=None):
    """Dense A->B flow (float32, px). B_corrected(x) = B(x + flow(x))."""
    cv2 = __import__("cv2")
    dis = cv2.DISOpticalFlow_create(preset or cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    # DIS = Dense Inverse Search (Kroeger et al. 2016, see docs/sources.md).
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


def joint_valid_strip(seam_x, halfw, fov_a=195.9, fov_b=197.5, h=1024,
                      w=2048):
    """Joint-valid mask for a sphere-coordinate corridor strip — moved to
    sphere_maps (analytic geometry's home); kept here as a re-export."""
    from steadycut.render.sphere_maps import joint_valid_strip as _jv
    return _jv(seam_x, halfw, fov_a, fov_b, h, w)


def warp_strip(B, flow):
    """Resample B by the flow field (still-level validation + tests)."""
    cv2 = __import__("cv2")
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(B, xs + flow[:, :, 0], ys + flow[:, :, 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def dest_src_window(W, dest, src, ramp=12):
    """Flat-top confidence window spanning destination -> source.

    compose/warp_strip sample B[x + flow] AT the destination x, so the
    window must be ~1 at the destination (A-copy) and cover through the
    source (B-copy). A source-centered window silently never fires where
    needed (track +50 warp did nothing) — pinned by
    test_window_must_cover_destination below.
    """
    xs = np.arange(W, dtype=np.float32)
    lo, hi = min(dest, src) - 5, max(dest, src) + 5
    v = np.clip((xs - (lo - ramp)) / ramp, 0, 1) * np.clip(
        ((hi + ramp) - xs) / ramp, 0, 1)
    return 0.5 - 0.5 * np.cos(np.clip(v, 0, 1) * np.pi)


def blend_scaled_flow(flow, alpha):
    """Scale the flow correction by the blend weight: flow * (1 - alpha).

    alpha is the A-side blend weight per pixel. Outside the transition
    (alpha 0 or 1) the correction is zero by construction, so a warp
    cannot tear single-lens territory at the band edges.
    """
    f = np.asarray(flow, dtype=np.float32)
    a = np.asarray(alpha, dtype=np.float32)
    if a.ndim == 2:
        a = a[:, :, None]
    factor = np.where((a > 0) & (a < 1), 1.0 - a, 0.0)
    return (f * factor).astype(np.float32)


def fuse_left_right_flow(flow_lr, flow_rl):
    """Sign-consistent L/R flow fusion.

    Per component: where the left flow is nonnegative take the larger
    of (L, -R), otherwise the smaller. Disagreement collapses toward
    the conservative signed value instead of averaging opposites.
    """
    left = np.asarray(flow_lr, dtype=np.float32)
    right = np.asarray(flow_rl, dtype=np.float32)
    neg_r = -right
    return np.where(left >= 0, np.maximum(left, neg_r),
                    np.minimum(left, neg_r)).astype(np.float32)


def temporal_median_flow(flows):
    """Temporal median across a deque of flow fields. A single-frame
    outlier cannot survive the median.
    """
    stack = np.stack([np.asarray(f, dtype=np.float32) for f in flows])
    return np.median(stack, axis=0).astype(np.float32)


def compose_corrected_B(eqB, seam_x, halfw, flow, conf, alpha=None):
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
    if alpha is not None:
        flow = blend_scaled_flow(flow, alpha)
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
