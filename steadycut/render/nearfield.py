"""Near-field double repair (production): detect, displace, warp/trim/patch.

Geometry the instruments proved on real doubles (trunk, lift-gap, track
junction): a near copy co-occurs with A's copy in the joint, wider than
any seam placement can dodge. Repair per span, in this order:

1. displace: SIFT votes inside the span rows give dx (B-copy minus
   A-copy). Confident votes only (count + spread gates).
2. warp: move B's copy onto A's with a dest->src flat-top window
   (dest_src_window: compose samples AT the destination).
3. trim: invalidate B's span where the joint covers it (union preserved).
4. patch: two-band (A content, A->B tone) where the joint covers it —
   warp smears washed snow, patch singles it.
5. merge with ACTUAL eroded validity (cv2.distanceTransform on each
   render's black edge): analytic FOV validity never saw the real black
   boundary or its blue resampling fringe.

auto_repair() runs detection (NCC + SIFT spans) then repair, span by
span. Auto acts only with a SIFT-confirmed displacement; detection
without one is cert-logged skipped-needs-eye (acting on a lookalike
would swap in the other lens's content). Explicit spans (operator
measured) may still trim/patch without dx. Every action is gated; a
clean pair returns the plain merge untouched.
"""
from __future__ import annotations

import numpy as np

ERODE_PX = 12       # fringe width at a render's black edge (measured ~8-12)
SIFT_MIN_N = 5      # min votes for a trusted displacement
SIFT_MAX_IQR = 12   # max dx spread (px) for a trusted displacement


def _gray(img):
    cv2 = __import__("cv2")
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def span_displacement(eqA, eqB, rect, min_dx=8, max_dx=150, max_dy=10):
    """Median SIFT dx of B-copies inside rect (y0,y1,x0,x1), or None.

    Matches A-side keypoints against B-side keypoints in the span rows;
    keeps ratio-tested matches with min_dx <= dx <= max_dx, |dy| small.
    Returns (dx, n) with n votes, or (None, n) when untrusted.
    """
    cv2 = __import__("cv2")
    gA = _gray(eqA).astype(np.uint8)
    gB = _gray(eqB).astype(np.uint8)
    y0, y1, x0, x1 = rect
    H, W = gA.shape
    y0, y1 = max(0, y0), min(H, y1)
    ax0, ax1 = max(0, x0 - 200), x1
    bx0, bx1 = max(0, x0 - 20), min(W, x1 + 150)
    a = gA[y0:y1, ax0:ax1]
    b = gB[y0:y1, bx0:bx1]
    sift = cv2.SIFT_create()
    kA, dA = sift.detectAndCompute(a, None)
    kB, dB = sift.detectAndCompute(b, None)
    if dA is None or dB is None or len(kA) < 2 or len(kB) < 2:
        return None, 0
    dxs = []
    for pair in cv2.BFMatcher().knnMatch(dA, dB, k=2):
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance >= 0.75 * second.distance:
            continue
        ax = kA[first.queryIdx].pt[0] + ax0
        bx = kB[first.trainIdx].pt[0] + bx0
        dy = kB[first.trainIdx].pt[1] - kA[first.queryIdx].pt[1]
        dx = bx - ax
        if min_dx <= dx <= max_dx and abs(dy) <= max_dy:
            dxs.append(dx)
    n = len(dxs)
    if n < SIFT_MIN_N:
        return None, n
    q25, med, q75 = np.percentile(dxs, (25, 50, 75))
    if q75 - q25 > SIFT_MAX_IQR:
        return None, n
    return float(med), n


def lens_validity(h, w, fov_a=185.0, fov_b=183.5):
    """Per-lens output-frame validity (axis A at x=w/2, B at x=0).

    Renders show pixels past the calibrated FOV (unrectified rim:
    misplaced, magnified, blue). Masking to the analytic model forces
    those directions onto the other lens, which sees them well inside
    its calibration. Returns (validA, validB).
    """
    ys, xs = np.mgrid[0:h, 0:w].astype(float)
    az = (xs / w) * 2 * np.pi
    el = (0.5 - ys / h) * np.pi
    dz = np.cos(el) * np.cos(az)
    incA = np.arccos(np.clip(-dz, -1, 1))
    incB = np.arccos(np.clip(dz, -1, 1))
    return (incA <= np.radians(fov_a) / 2, incB <= np.radians(fov_b) / 2)


def eroded_validity(gray, thresh=8, erode_px=ERODE_PX):
    """Pixels safely inside a render's coverage (fringe excluded)."""
    cv2 = __import__("cv2")
    mask = (np.asarray(gray) > thresh).astype(np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    return dist > erode_px


def repair_spans(eqA, eqB, wgt, spans, fov_a=185.0, fov_b=183.5,
                 halfw=150):
    """Merge eqA/eqB under weight wgt (1 = B) with spans repaired.

    spans: list of (rect, dx, seam_x) with rect = (y0, y1, x0, x1) the
    detected B-span and dx the measured displacement (B-copy minus
    A-copy); dx None = trim/patch only, no warp. Returns the merged
    uint8 image. Pure function (testable).
    """
    cv2 = __import__("cv2")
    from steadycut.render.corridor_flow import (compose_corrected_B,
                                                dest_src_window)
    from steadycut.render.dup_trim import apply_trim
    from steadycut.render.sphere_maps import joint_valid_strip
    A = np.asarray(eqA)
    B = np.asarray(eqB)
    H, W = A.shape[:2]
    F = np.zeros((H, W, 2), np.float32)
    C = np.zeros((H, W), np.float32)
    trim_rects = []
    # dest (A-copy center) and window from the measured dx; the trim
    # covers exactly the detected span (joint-clamped at detection, so
    # A covers: trimming wider would risk holes where A is blind).
    for (y0, y1, x0, x1), dx, _sx in spans:
        y0c, y1c = max(0, y0), min(H, y1)
        if dx is not None:
            dest = (x0 + x1) / 2 - dx
            F[y0c:y1c] = (float(dx), 0.0)
            C[y0c:y1c] = dest_src_window(W, dest, x1)[None, :]
            n = min(12, (y1c - y0c) // 2)
            if n > 0:
                r = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
                     ).astype(np.float32)
                C[y0c:y0c + n] *= r[:, None]
                C[y1c - n:y1c] *= r[::-1][:, None]
        trim_rects.append((y0c, y1c, x0, x1))
    # corrected B per seam corridor, spliced (corridors don't overlap;
    # flow is zero outside them so splicing is exact).
    modelA, modelB = lens_validity(H, W, fov_a, fov_b)
    Bc = B.copy()
    validBc = eroded_validity(_gray(B)) & modelB
    for sx in sorted({s[2] for s in spans} | {1536}):
        x0, x1 = sx - halfw, sx + halfw
        Bc[:, x0:x1] = compose_corrected_B(
            B, sx, halfw, F[:, x0:x1], C[:, x0:x1])[:, x0:x1]
        validBc[:, x0:x1] = (
            compose_corrected_B(
                ((eroded_validity(_gray(B)) & modelB).astype(np.float32)
                 * 255).astype(np.uint8),
                sx, halfw, F[:, x0:x1], C[:, x0:x1])[:, x0:x1] > 127)
    validA = eroded_validity(_gray(A)) & modelA
    w2 = apply_trim(np.asarray(wgt, dtype=float), trim_rects)
    w2 = np.where(~validA, 1.0, w2)
    w2 = np.where(~validBc, 0.0, w2)
    w3 = w2[:, :, None]
    A64 = A.astype(float)
    merged = Bc.astype(float) * w3 + A64 * (1 - w3)
    # two-band patch per span where the joint covers (A content, A->B
    # tone). Per-row joint clamp — never paint A where A is blind. Tone
    # refs sampled from single-coverage wings, masked by validity.
    gA = _gray(A).astype(float)
    gB = _gray(B).astype(float)
    onlyA = validA & ~validBc
    onlyB = validBc & ~validA
    for (y0, y1, x0, x1), _dx, sx in spans:
        jv = joint_valid_strip(sx, halfw, fov_a, fov_b, H, W)
        jx0 = sx - halfw
        for yy in range(max(0, y0), min(H, y1)):
            xs = np.where(jv[yy])[0] + jx0
            if len(xs) < 24:
                continue
            run = xs[(xs >= x0 - 10) & (xs <= x1 + 10)]
            if len(run) < 24:
                continue
            ax0, ax1 = int(run[0]), int(run[-1]) + 1
            ra = onlyA[yy, max(0, sx - 120):sx - 40]
            rb = onlyB[yy, sx + 40:min(W, sx + 120)]
            if ra.sum() < 8 or rb.sum() < 8:
                continue
            la = float(np.median(gA[yy, max(0, sx - 120):sx - 40][ra]))
            lb = float(np.median(gB[yy, sx + 40:min(W, sx + 120)][rb]))
            gain = lb / max(la, 1)
            if not 0.7 <= gain <= 1.4:
                # wings span different content (e.g. dark trees vs snow):
                # the tone refs are invalid, skip the row (warp covers).
                continue
            t = np.linspace(0, 1, ax1 - ax0)
            tone = ((1 - t) + t * gain)[:, None]
            merged[yy, ax0:ax1] = A64[yy, ax0:ax1] * tone
    return np.clip(merged, 0, 255).astype(np.uint8)


def load_spans(path):
    """Operator spans file (grid-probe measured) -> repair_spans spans.

    JSON list of {"rect": [y0,y1,x0,x1], "dx": float|null, "seam": int}.
    General mechanism (no tuned data in the repo): measure separations
    off a gridded still, pass the file, get exact warps.
    """
    import json
    with open(path) as f:
        items = json.load(f)
    return [(tuple(it["rect"]), it.get("dx"), int(it.get("seam", 1536)))
            for it in items]


def auto_repair(eqA, eqB, wgt, fov_a=185.0, fov_b=183.5,
                seams=(512, 1536)):
    """Detect double spans and repair them; gated, cert-logged.

    Returns (merged, cert) with cert['spans'] listing per-span dicts
    (rect, dx, votes, action: warp+trim+patch | trim+patch | skipped).
    A clean pair returns the plain merge with no spans.
    """
    from steadycut.render.dup_trim import (find_duplicates,
                                           find_duplicates_sift)
    H, W = np.asarray(eqA).shape[:2]
    found = []
    for sx in seams:
        for r in find_duplicates(eqA, eqB, seams=(sx,)):
            found.append((sx, r))
        for r in find_duplicates_sift(eqA, eqB, seams=(sx,)):
            found.append((sx, r))
    cert = {"spans": []}
    if not found:
        w3 = np.asarray(wgt, dtype=float)[:, :, None]
        merged = (np.asarray(eqB).astype(float) * w3
                  + np.asarray(eqA).astype(float) * (1 - w3))
        return np.clip(merged, 0, 255).astype(np.uint8), cert
    fixed = []
    for sx, rect in found:
        dx, n = span_displacement(eqA, eqB, rect)
        if dx is None:
            # detection without displacement: acting (trim or patch)
            # would swap in the other lens's content on a mere
            # lookalike (forest gross-similarity passes NCC tiers).
            # Log for eye measurement, change nothing.
            cert["spans"].append({"rect": [int(v) for v in rect],
                                  "dx": None, "votes": int(n),
                                  "action": "skipped-needs-eye",
                                  "seam": sx})
            continue
        fixed.append((rect, dx, sx))
        cert["spans"].append({"rect": [int(v) for v in rect], "dx": dx,
                              "votes": int(n), "action": "warp+trim+patch",
                              "seam": sx})
    if not fixed:
        w3 = np.asarray(wgt, dtype=float)[:, :, None]
        merged = (np.asarray(eqB).astype(float) * w3
                  + np.asarray(eqA).astype(float) * (1 - w3))
        return np.clip(merged, 0, 255).astype(np.uint8), cert
    return repair_spans(eqA, eqB, wgt, fixed, fov_a, fov_b), cert
