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


def eroded_validity(gray, thresh=8, erode_px=ERODE_PX, bright_keep=100,
                    color=None, sat_max=25.0):
    """Pixels safely inside a render's coverage (fringe excluded).

    Blind erosion also kills bright content at the FOV edge (a yellow
    sign's middle 8 px: both lenses erode it, the merge shows a gap that
    reads as a duplicate/split). Bright pixels are real content, not
    black-smear (smear is a dark mix), so they are restored: validity is
    eroded-mask OR bright. Dark smear stays dead; the tone stages correct
    what remains. When color (BGR) is given, bright-keep additionally
    requires near-gray: the coverage-edge resampling fringe is bright
    AND saturated (B-R 56-99 DN at the rim), while real bright content
    (sign, snow) is near-neutral. Saturated bright edge pixels stay
    dead so the other lens (which sees them well inside its coverage)
    wins them outright.
    """
    cv2 = __import__("cv2")
    g = np.asarray(gray)
    mask = (g > thresh).astype(np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    keep = (dist > erode_px) | (g > bright_keep)
    if color is not None:
        c = np.asarray(color, dtype=np.float32)
        if c.ndim == 3 and c.shape[2] == 3:
            # SIGNED B-R: the coverage-edge resampling fringe is B>R
            # (measured B-R +56..+99 at the rim). Yellow content is
            # R-B (negative) and must survive at the edge — abs()
            # would kill the sign the same way (finding-24 regression).
            bluer = c[:, :, 0] - c[:, :, 2]
            keep = keep & ((dist > erode_px) | (bluer <= sat_max))
    return keep


def residual_level(ea, eb, wgt, validA=None, validB=None, max_dn=8.0,
                   feather_frac=0.07, rows_frac=0.03, sigma_mid=6.0,
                   sigma_scene_frac=0.03, var_thresh=400.0,
                   gray_sat=15.0):
    """Studio-style residual step removal at the cut (globalSeamLeveling).

    Multiband softens a DC step into a wider gradient the eye still reads
    as a line on flat sky/snow (Mach bands); only equalization removes it.
    Splitting B-minus-A half/half under-corrects one-sided errors (a rim
    ring living in lens A only) and dents the innocent lens, so this
    attributes per lens: S is the row-smoothed scene (sigma_scene) from
    the flat-masked mean, and each lens is pulled toward S by its own
    mid-frequency deviation (sigma_mid keeps 16 px rings, drops 2 px
    noise; scene slopes live in S and are never touched). Steps, creases,
    and rings are all caught whatever their shape. Each side is guarded
    on the raw dev first (blurring first would smear content under the
    gate), the correction itself is smoothed, and applied
    under its own smoothstep feather (flat AT the seam, so the seam is
    smooth, relaxing to no-change feather_px out). Returns corrected
    (ea, eb) float32.
    """
    cv2 = __import__("cv2")
    A = np.asarray(ea, dtype=np.float32)
    B = np.asarray(eb, dtype=np.float32)
    h, w = A.shape[:2]
    w1 = np.asarray(wgt, dtype=float)
    joint = (w1 > 0.02) & (w1 < 0.98)
    if not joint.any():
        return A, B, {}
    nch = A.shape[2] if A.ndim == 3 else 1
    if validA is None:
        validA = (A.mean(axis=2) if nch == 3 else A) > 8
    if validB is None:
        validB = (B.mean(axis=2) if nch == 3 else B) > 8
    vA = np.asarray(validA).astype(bool)
    vB = np.asarray(validB).astype(bool)
    both = vA & vB
    feather_px = feather_frac * w  # scene scales are relative: the same
    sigma_rows = rows_frac * h  # seam works at any sphere size; only the
    sigma_scene = sigma_scene_frac * w  # noise scale (sigma_mid) is absolute
    mean = (A + B) / 2
    mg = mean.mean(axis=2) if mean.ndim == 3 else mean
    mu = cv2.blur(mg, (15, 15))
    var = cv2.blur((mg - mu) ** 2, (15, 15))
    use = (joint & (var < var_thresh))
    # scene: smoothed mean over usable seeds OUTSIDE the joint, interpolated
    # across it. Seeds inside the joint would absorb the very seam-coherent
    # tilt being removed (mirror-symmetric lens coatings tint both sides
    # the same way: common-mode, invisible to a joint-seeded S, but the eye
    # reads the resulting chroma swing as a line).
    use_scene = ((var < var_thresh) & ~joint & both)
    xs = np.arange(w)
    S = np.zeros((h, w, nch), np.float32)
    for c in range(nch):
        mc = mean[:, :, c] if mean.ndim == 3 else mean
        filled = mc.copy()
        for y in range(h):
            m = use_scene[y]
            if m.sum() < 5:
                filled[y] = 0.0
            elif m.sum() < w:
                filled[y] = np.interp(xs, xs[m], mc[y][m])
        kh = int(sigma_scene * 6) | 1
        filled = cv2.GaussianBlur(filled, (kh, 1), sigma_scene, 0)
        khr = int(sigma_rows * 6) | 1
        filled = cv2.GaussianBlur(filled, (1, khr), 0, sigma_rows)
        S[:, :, c] = filled
    seam_x = np.argmin(np.abs(w1 - 0.5), axis=1)
    xg = np.arange(w)[None, :].repeat(h, axis=0)
    u = np.clip(np.abs(xg - seam_x[:, None]).astype(float)
                / feather_px, 0, 1)
    f = (1 + np.cos(np.pi * u)) / 2
    fA = np.where((xg <= seam_x[:, None]) & (u < 1), f, 0.0)
    fB = np.where((xg > seam_x[:, None]) & (u < 1), f, 0.0)
    info = {}
    # correction lives near the joint only (dilated): far-field flat pixels
    # differ from smoothed-S by real scene shading, which must never move
    kdil = int(feather_px / 4) | 1
    near = cv2.dilate(joint.astype(np.uint8), np.ones((kdil, kdil),
                                                      np.uint8)) > 0
    # chroma flatten lives inside the per-lens loop below (needs fside).
    for tag, img, fside in (("A", A, fA), ("B", B, fB)):
        if nch == 3:
            # chroma flatten on near-gray pixels under this lens's feather:
            # the residual tint line (coating chroma tilt, v360 edge
            # resampling) lives in B-R on sky/snow, where desaturation is
            # invisible; saturated content keeps its color (texture masks
            # any line there anyway). fside is smooth: no desat cliff.
            # Seam-coherent admission (same surgery as the mag guard):
            # sky tint at the seam runs 19-32 DN saturated — the gray
            # gate is blind to it. Tint that is a function of x (shared
            # across rows) is admitted at any saturation via its
            # row-median B-R profile; LOCALIZED saturation (jacket,
            # function of y) keeps the gate. Global tint (flat halves)
            # is gain's job: admit only above the far-field level.
            im3 = img.reshape(h, w, nch)
            sat = np.abs(im3[:, :, 0] - im3[:, :, 2])
            lum = im3.mean(axis=2, keepdims=True)
            tint = im3[:, :, 0] - im3[:, :, 2]
            tprof = np.median(np.where(near, tint, np.nan), axis=0)
            tprof = np.where(np.isnan(tprof), 0.0, tprof)
            u2c = np.clip(np.abs(xg - seam_x[:, None]).astype(float)
                          / (2 * feather_px), 0, 1)
            far_mc = (u2c >= 1.0) & both
            if far_mc.any():
                tfar = np.median(np.where(far_mc, tint, np.nan))
                tfar = 0.0 if np.isnan(tfar) else float(tfar)
            else:
                tfar = 0.0
            tloc = tprof - tfar
            coh = np.abs(tloc)[None, :] > gray_sat
            g = (np.maximum(np.clip((gray_sat - sat) / gray_sat, 0, 1),
                            np.where(coh, 1.0, 0.0))
                 * fside)[:, :, None]
            img[:] = im3 * (1 - g) + lum * g
        dev = img.reshape(h, w, nch) - S
        # guard FIRST on raw dev (blurring first would smear big content
        # edges under the gate): refuse, then smooth the correction itself.
        # Seam-coherent admission: a defect that is a function of x
        # (rim ring, flare — shared across rows) is admitted at any
        # magnitude via its row-median x-profile; CONTENT is a function
        # of y and stays under the max_dn gate. A bar (content) varies
        # row to row, so its median profile is ~0 and it still refuses.
        mag = np.abs(dev).max(axis=2)
        prof = np.median(np.where(near[:, :, None], dev, np.nan),
                         axis=0)
        prof = np.where(np.isnan(prof), 0.0, prof)
        # localize: a GLOBAL offset (flat halves, gain mismatch) is
        # measure_gain's job, not this stage's — admit only the part
        # of the profile that EXCEEDS its far-field level. A rim ring
        # decays away from the seam; a global step does not. Far =
        # beyond 2x feather (the near dilation reaches ~feather/4,
        # the feather itself to 1x — 2x is safely outside both).
        u2 = np.clip(np.abs(xg - seam_x[:, None]).astype(float)
                     / (2 * feather_px), 0, 1)
        far_m = (u2 >= 1.0) & both
        if not far_m.any():
            # no far-field evidence (tiny frame, giant feather): cannot
            # localize — fall back to the magnitude gate (old behavior).
            profloc = np.zeros_like(prof)
        else:
            far = np.median(np.where(far_m[:, :, None], dev, np.nan),
                            axis=(0, 1))
            far = np.where(np.isnan(far), 0.0, far)
            profloc = prof - far[None, :]
        resid = dev - profloc[None, :, :]
        rmag = np.abs(resid).max(axis=2)
        weight = (((mag <= max_dn) | (rmag <= max_dn)) & near).astype(
            np.float32)
        # normalized convolution: smooth the correction without dragging
        # refused content-rows into clean sky (plain blur would smear a
        # tree row's vetoed pixels across +-sigma_rows of sky)
        num = dev * weight[:, :, None]
        km = int(sigma_mid * 6) | 1
        num = cv2.GaussianBlur(num, (km, km), sigma_mid, 0)
        den = cv2.GaussianBlur(weight, (km, km), sigma_mid, 0)
        khr = int(sigma_rows * 6) | 1
        num = cv2.GaussianBlur(num, (1, khr), 0, sigma_rows)
        den = cv2.GaussianBlur(den, (1, khr), 0, sigma_rows)
        den3 = np.maximum(den, 1e-3)[:, :, None]
        fade = np.clip((den - 0.2) / 0.3, 0, 1)[:, :, None]
        corr = np.where(den3 > 0.2, num / den3, 0.0) * fade
        info[tag] = float(np.abs(corr).max())
        if nch == 3:
            img -= corr * fside[:, :, None]
        else:
            img -= (corr * fside[:, :, None]).reshape(h, w)
    return A, B, info


def edge_flatten(ea, eb, wgt, validA=None, validB=None, max_dn=30.0,
                 anchor_px=50, feather_frac=0.07, sigma_cols=10.0,
                 sigma_rows=30.0, var_thresh=400.0):
    """Per-lens seam-band anomaly removal (runs before residual_level).

    residual_level pulls both lenses toward their common mean S, so a
    chroma ramp living in ONE lens's edge (Jan26 A: 26 DN B-R ramp from
    lateral CA + v360 black-smear at the FOV edge) is half-absorbed into
    S and half-refused by the guard — the visible tint line survives.
    Instead each lens is referenced to ITSELF: ref is a coarse scene ramp
    between flat-pixel medians anchored each side of the seam band, and
    the anomaly is the smoothed remainder (anything seam-narrow: ramps,
    rings, fringes). Guarded (|anomaly| <= max_dn, content keeps its
    edges) and applied under the lens's own smoothstep feather. Returns
    corrected (ea, eb) float32 + info dict.
    """
    cv2 = __import__("cv2")
    A = np.asarray(ea, dtype=np.float32)
    B = np.asarray(eb, dtype=np.float32)
    h, w = A.shape[:2]
    w1 = np.asarray(wgt, dtype=float)
    joint = (w1 > 0.02) & (w1 < 0.98)
    if not joint.any():
        return A, B, {}
    nch = A.shape[2] if A.ndim == 3 else 1
    if validA is None:
        validA = (A.mean(axis=2) if nch == 3 else A) > 8
    if validB is None:
        validB = (B.mean(axis=2) if nch == 3 else B) > 8
    vA = np.asarray(validA).astype(bool)
    vB = np.asarray(validB).astype(bool)
    # seed-cleanliness != blend-validity: the v360 black-smear + CA ramp
    # corrupts ~30 px inside the nominal edge, so measurement seeds erode
    # deep (clean pixels only) while the blend keeps the wide joint
    def _deep(v, px=30):
        dd = cv2.distanceTransform(v.astype(np.uint8), cv2.DIST_L2, 3)
        return dd > px
    sA, sB = _deep(vA), _deep(vB)
    mg = ((A + B) / 2).mean(axis=2) if nch == 3 else (A + B) / 2
    mu = cv2.blur(mg, (15, 15))
    flat = (cv2.blur((mg - mu) ** 2, (15, 15)) < var_thresh)
    seam_x = np.argmin(np.abs(w1 - 0.5), axis=1)
    xg = np.arange(w)[None, :].repeat(h, axis=0)
    u = np.clip(np.abs(xg - seam_x[:, None]).astype(float)
                / (feather_frac * w), 0, 1)
    f = (1 + np.cos(np.pi * u)) / 2
    # overlapping (NOT split at the seam): each lens's smear tail extends
    # past the seam column into the other side, and only its own
    # reference can remove it. Both converge (anomalies gone, shared scene
    # stays); residual_level afterward removes any tone offset between refs.
    fA = np.where(u < 1, f, 0.0)
    fB = np.where(u < 1, f, 0.0)
    info = {}
    for tag, img, fside, vv, ss in (("A", A, fA, vA, sA),
                                    ("B", B, fB, vB, sB)):
        im3 = img.reshape(h, w, nch)
        ano = np.zeros_like(im3)
        xs = np.arange(w)
        for y in range(h):
            # anchors and seeds must be CLEAN pixels this lens saw: the
            # ~30 px inside its FOV edge carry black-smear + CA (exactly
            # what is being measured), and seeding on them blinds the fit.
            # The correction itself still applies there (via the feather).
            m = flat[y] & ss[y]
            if m.sum() < 50:
                continue
            for c in range(nch):
                col = im3[y, :, c]
                # coarse scene reference: medians on flat pixels each side
                # of the seam band, ramped across it (scene gradients pass
                # through; seam-narrow anomalies stick out against the ramp)
                sig = seam_x[y]
                mL = m & (xs < sig - anchor_px)
                mR = m & (xs > sig + anchor_px)
                a0 = np.median(col[mL]) if mL.sum() > 20 else np.nan
                a1 = np.median(col[mR]) if mR.sum() > 20 else np.nan
                if np.isnan(a0) and np.isnan(a1):
                    continue
                if np.isnan(a0):
                    a0 = a1
                if np.isnan(a1):
                    a1 = a0
                idx = np.where(m)[0]
                ref = a0 + (a1 - a0) * np.clip(
                    (idx - (sig - anchor_px)) / (2 * anchor_px), 0, 1)
                # reference extends across the full row (interp over the
                # gap): the anomaly is measured everywhere FLAT (the ramp
                # pixels themselves are flat sky) and zeroed on texture.
                # Seeding-only-inside-clean would blind the fit to exactly
                # what it must remove.
                ref_full = np.interp(xs, idx, ref)
                fm = flat[y]
                ano[y, fm, c] = col[fm] - ref_full[fm]
        km = int(sigma_cols * 6) | 1
        # guard FIRST on raw anomaly, then normalized convolution (blurred
        # weight in the denominator keeps refused content-rows from
        # dragging into clean sky)
        nch_e = ano.shape[2]
        mag = np.abs(ano).max(axis=2)
        weight = (mag <= max_dn).astype(np.float32)
        num = ano * weight[:, :, None]
        num = cv2.GaussianBlur(num, (km, km), sigma_cols, 0)
        den = cv2.GaussianBlur(weight, (km, km), sigma_cols, 0)
        khr = int(sigma_rows * 6) | 1
        num = cv2.GaussianBlur(num, (1, khr), 0, sigma_rows)
        den = cv2.GaussianBlur(den, (1, khr), 0, sigma_rows)
        den3 = np.maximum(den, 1e-3)[:, :, None]
        fade = np.clip((den - 0.2) / 0.3, 0, 1)[:, :, None]
        corr = np.where(den3 > 0.2, num / den3, 0.0) * fade
        info[tag] = float(np.abs(corr).max())
        if nch == 3:
            img -= corr * fside[:, :, None]
        else:
            img -= (corr * fside).reshape(h, w)
    return A, B, info


def multiband_merge(A, Bc, wgt, validA, validBc, bands=5):
    """Studio-style blend: tone across octaves, detail at the cut.

    Partition masks over the union (each side wins where the other is
    blind — a weight-side mask alone leaves black ticks where the warp
    reaches past validity). Single-band ramps keep a tone step; the
    multiband stack smooths it while aligned detail survives. Masks are
    the FEATHERED weights (not binary): hard mask edges ring in the
    Laplacian pyramid (bright halo + hairline core with texture continuing
    across — exactly the visible seam band).
    """
    cv2 = __import__("cv2")
    w = np.asarray(wgt, dtype=float)
    vA = np.asarray(validA).astype(bool)
    vB = np.asarray(validBc).astype(bool)
    mA = np.where(vA, (1.0 - w) * 255, 0).astype(np.uint8)
    mB = np.where(vB, w * 255, 0).astype(np.uint8)
    # where only one side is valid it must win outright (else black ticks)
    mA[~vB] = np.where(vA[~vB], 255, 0)
    mB[~vA] = np.where(vB[~vA], 255, 0)
    # soften partition cliffs (eroded-validity edges are binary): a hard
    # mask step leaves a 1-2 px hairline core even when tone matches
    mA = cv2.GaussianBlur(mA, (0, 0), 2, 0)
    mB = cv2.GaussianBlur(mB, (0, 0), 2, 0)
    h, W = mA.shape
    bl = cv2.detail.MultiBandBlender()
    bl.setNumBands(bands)
    bl.prepare((0, 0, W - 1, h - 1))
    bl.feed(np.asarray(A, dtype=np.int16), mA, (0, 0))
    bl.feed(np.asarray(Bc, dtype=np.int16), mB, (0, 0))
    res, _ = bl.blend(None, None)
    return np.clip(res, 0, 255).astype(np.uint8)


def repair_spans(eqA, eqB, wgt, spans, fov_a=185.0, fov_b=183.5,
                 halfw=150, reroute=True, multiband=True):
    """Merge eqA/eqB under weight wgt (1 = B) with spans repaired.

    spans: list of (rect, dx, seam_x) with rect = (y0, y1, x0, x1) the
    detected B-span and dx the measured displacement (B-copy minus
    A-copy); dx None = trim/patch only, no warp. Returns the merged
    uint8 image. Pure function (testable).

    reroute: route again on the repaired pair. Warp moves content after
    routing, invalidating the seam placement (a fringe shown by the old
    cut hides behind the new one); the second route is cheap numpy.
    Falls back to the input weight when routing fails.
    multiband: Studio-style octave blend over partition masks instead
    of the single-band ramp (kills the tone step, keeps detail).
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
            # x-restricted: F/C rows are full-width per span, so a warp
            # tree span sharing rows with a trim-only skier span would
            # move the skier's B-copy out from under its trim rect.
            # Confine the flow to the span's own columns.
            win = dest_src_window(W, dest, x1)
            xs0, xs1 = max(0, x0 - 12), min(W, x1 + 12)
            F[y0c:y1c, xs0:xs1, 0] = float(dx)
            Crow = np.zeros(W, np.float32)
            Crow[xs0:xs1] = win[xs0:xs1]
            C[y0c:y1c] = np.maximum(C[y0c:y1c],
                                    np.broadcast_to(Crow[None, :],
                                                    (y1c - y0c, W)))
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
    # SAME validity the merge uses (color-gated): repair must agree
    # with the merge about coverage, else trim is silently undone at
    # :522 (B comes back where validA fails) or fringe is pasted back
    # by the patch loop below.
    validBc = eroded_validity(_gray(B), color=B) & modelB
    for sx in sorted({s[2] for s in spans} | {1536}):
        x0, x1 = sx - halfw, sx + halfw
        Bc[:, x0:x1] = compose_corrected_B(
            B, sx, halfw, F[:, x0:x1], C[:, x0:x1])[:, x0:x1]
        validBc[:, x0:x1] = (
            compose_corrected_B(
                ((eroded_validity(_gray(B), color=B) & modelB).astype(
                    np.float32) * 255).astype(np.uint8),
                sx, halfw, F[:, x0:x1], C[:, x0:x1])[:, x0:x1] > 127)
    validA = eroded_validity(_gray(A), color=A) & modelA
    w2 = apply_trim(np.asarray(wgt, dtype=float), trim_rects)
    if reroute:
        try:
            from steadycut.render.seam_route import route_both
            from steadycut.render.sphere_maps import routed_weight
            # ALWAYS both seams: routing 512-only leaves 0 ramp px at
            # 1536 instead of 21 (measured hard cut). Extra spans only
            # make the single-seam bug more common.
            seams = sorted(set([s[2] for s in spans] + [512, 1536]))
            polys = route_both(A.astype(np.uint8), Bc.astype(np.uint8),
                               seams=tuple(seams), halfw=48, width=W,
                               fov_a=fov_a, fov_b=fov_b)
            w2, _ = routed_weight(W, H, 4.0, fov_a, fov_b,
                                  polylines=polys)
            w2 = apply_trim(np.asarray(w2, dtype=float), trim_rects)
        except Exception:
            w2 = apply_trim(np.asarray(wgt, dtype=float), trim_rects)
    w2 = np.where(~validA, 1.0, w2)
    w2 = np.where(~validBc, 0.0, w2)
    A64 = A.astype(float)
    if multiband:
        merged = multiband_merge(A, Bc, w2, validA, validBc).astype(float)
    else:
        w3 = w2[:, :, None]
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
            # mask with the merge's validA: the model-only joint strip
            # would paste A's rim fringe back in after the trim. Skip
            # rows where A is not really valid there.
            if float(validA[yy, ax0:ax1].mean()) < 0.9:
                continue
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
    from steadycut.render.dup_trim import (Span, count_instances,
                                           find_duplicates_detailed,
                                           find_duplicates_sift,
                                           parallax_budget,
                                           ramp_duplication)
    from steadycut.render.sphere_maps import joint_valid_strip
    H, W = np.asarray(eqA).shape[:2]
    modelA, _ = lens_validity(H, W, fov_a, fov_b)
    gA = np.asarray(eqA)
    validA = (eroded_validity(_gray(gA), color=gA) & modelA)
    found = []
    for sx in seams:
        for sp in find_duplicates_detailed(eqA, eqB, seams=(sx,)):
            found.append((sx, sp))
        for r in find_duplicates_sift(eqA, eqB, seams=(sx,)):
            found.append((sx, r))
    cert = {"spans": []}
    if not found:
        # passthrough (same reason as below): caller owns the img.
        return None, cert
    fixed = []
    lat_lo, lat_hi = int(80 * H / 1024), int(950 * H / 1024)
    for sx, item in found:
        rect = tuple(item[:4])
        if isinstance(item, Span):
            det = item
        else:
            det = None
        dx, n = span_displacement(eqA, eqB, rect)
        if dx is not None:
            # T1 parallax-budget veto: can only refuse, never admit.
            y0, y1 = int(rect[0]), int(rect[1])
            jv0 = joint_valid_strip(sx, 100, fov_a, fov_b, H, W)
            jw0 = int(np.median([jv0[yy].sum()
                                 for yy in range(max(0, y0),
                                                 min(H, y1))]))
            if abs(dx) > parallax_budget(y0, y1, jw0, H, W):
                cert["spans"].append({"rect": [int(v) for v in rect],
                                      "dx": dx, "votes": int(n),
                                      "action": "skipped-needs-eye",
                                      "gate_refused":
                                      "dx-exceeds-parallax-budget",
                                      "seam": sx})
                continue
            fixed.append((rect, dx, sx))
            cert["spans"].append({"rect": [int(v) for v in rect],
                                  "dx": dx, "votes": int(n),
                                  "action": "warp+trim+patch",
                                  "seam": sx})
            continue
        # trim-only without displacement: five gates, first failure
        # recorded; SIFT rects (no Span details) always skip here
        # (gate 5 = mutual NCC only; SIFT+dx lives on the warp path).
        refused = None
        y0, y1, x0, x1 = (int(v) for v in rect)
        if det is None or not det.mutual:
            refused = "gate5-corroboration"
        else:
            jv = joint_valid_strip(sx, 100, fov_a, fov_b, H, W)
            x0s = max(0, sx - 100)
            sup0, sup1 = max(0, det.sup0), min(W, det.sup1)
            jfracs = []
            for yy in range(max(0, y0), min(H, y1)):
                jrow = set((np.where(
                    jv[yy, max(0, sup0 - x0s):min(
                        jv.shape[1], sup1 - x0s)])[0] + x0s).tolist())
                jfracs.append(len(set(range(sup0, sup1)) & jrow)
                              / max(1, sup1 - sup0))
            jfrac = float(np.median(jfracs)) if jfracs else 0.0
            if jfrac < 0.8:
                refused = "gate1-in-joint-frac"
            else:
                rows = np.arange(max(0, y0), min(H, y1))
                cols = slice(max(0, sup0), min(W, sup1))
                lane = validA[max(0, y0):min(H, y1), cols].mean(axis=1) \
                    if len(rows) else np.array([0.0])
                ok = lane >= 0.9
                n_ok = int(ok.sum())
                contiguous = n_ok and (np.where(ok)[0].ptp() + 1 == n_ok)
                if not (n_ok >= 0.9 * len(rows) and contiguous):
                    refused = "gate2-validA-rows"
                else:
                    na, nb, kind, conf = count_instances(
                        eqA, eqB, det.tx, det.tx + int(det.disp or 0),
                        sup1, rows)
                    if kind != "double":
                        refused = "gate3-instance-count"
                    elif sup1 - sup0 > 2 * np.median([
                            int(jv[yy].sum()) for yy in rows
                            ]) or y0 < lat_lo or y1 > lat_hi:
                        refused = "gate4-extent-latitude"
        if refused is not None:
            # W2: gate-1 refusals (copy wider than joint) fall through
            # to the NCC-disp warp branch: trim would chop, but warp
            # moves the copy. warp_consistency (W1) must pass ANDed,
            # plus branch-local gate 1w (residual support outside
            # joint+feather within taper budget; edge step is covered
            # by the runtime post-trim gate).
            if refused == "gate1-in-joint-frac" and det is not None \
                    and det.mutual and det.disp:
                from steadycut.render.dup_trim import warp_consistency
                w1 = warp_consistency(eqA, eqB, det, sx) or ""
                if w1 == "":
                    jv2 = joint_valid_strip(sx, 100, fov_a, fov_b, H, W)
                    ry0, ry1 = max(0, y0), min(H, y1)
                    jwidths = [int(jv2[yy].sum())
                               for yy in range(ry0, ry1)]
                    jw = int(np.median(jwidths)) if jwidths else 0
                    if abs(float(det.disp)) > parallax_budget(
                            y0, y1, jw, H, W):
                        refused = "dx-exceeds-parallax-budget"
                    elif (det.sup1 - det.sup0) - jw > 32:
                        refused = "gate1w-residual"
                    else:
                        fixed.append((rect, float(det.disp), sx))
                        cert["spans"].append({
                            "rect": [int(v) for v in rect],
                            "dx": float(det.disp), "votes": int(n),
                            "action": "warp+trim+patch",
                            "dx_source": "ncc-disp",
                            "gate_refused": None, "seam": sx})
                        continue
                if w1 != "":
                    refused = "w1-" + w1
            cert["spans"].append({"rect": [int(v) for v in rect],
                                  "dx": None, "votes": int(n),
                                  "action": "skipped-needs-eye",
                                  "gate_refused": refused,
                                  "seam": sx})
            continue
        fixed.append((rect, None, sx))
        cert["spans"].append({"rect": [int(v) for v in rect],
                              "dx": None, "votes": int(n),
                              "action": "trim-only",
                              "gate_refused": None, "seam": sx})
    # overlap veto: a trim candidate overlapping a warp span or a
    # dxless SIFT span is evaluated on the union: any skip vetoes
    # the whole. (NCC+SIFT report the same copy separately; one
    # trimming while the other skips/warps splits the object.)
    warp_rects = [f[0] for f in fixed if f[1] is not None]
    sift_rects = [tuple(s[:4]) for _, s in found
                  if not isinstance(s, Span)]
    def _overlap(a, b):
        ax0, ay0, ax1, ay1 = a[2], a[0], a[3], a[1]
        bx0, by0, bx1, by1 = b[2], b[0], b[3], b[1]
        return (min(ax1, bx1) > max(ax0, bx0)
                and min(ay1, by1) > max(ay0, by0))
    kept, vetoed = [], 0
    for (rect, dx, sx) in fixed:
        if dx is None and any(_overlap(rect, w) for w in warp_rects
                              + sift_rects):
            vetoed += 1
            for s in cert["spans"]:
                if (s["action"] == "trim-only"
                        and s["rect"] == [int(v) for v in rect]):
                    s["action"] = "skipped-needs-eye"
                    s["gate_refused"] = "gate5-overlap-veto"
        else:
            kept.append((rect, dx, sx))
    fixed = kept
    if not fixed:
        # passthrough: the caller (stitch.py) owns the multiband img;
        # a single-band merge here would OVERWRITE its
        # validity/residual_level work on every skipped frame.
        return None, cert
    merged = repair_spans(eqA, eqB, wgt, fixed, fov_a, fov_b)
    # runtime post-trim gate: B-score to baseline (+-2 DN tol) AND
    # step <= pre-trim per trim span, else revert that span and
    # re-run once. Logged in cert; A4 histogram reads it.
    plain = (np.asarray(eqB).astype(float)
             * np.asarray(wgt, dtype=float)[:, :, None]
             + np.asarray(eqA).astype(float)
             * (1 - np.asarray(wgt, dtype=float)[:, :, None]))
    gA0 = _gray(np.asarray(eqA))
    gB0 = _gray(np.asarray(eqB))
    bad = []
    for (rect, dx, sx) in fixed:
        if dx is not None:
            continue
        y0, y1, x0, x1 = rect
        pre = ramp_duplication(plain[:, :, 0] if plain.ndim == 3
                               else plain, gA0, gB0, sx)
        post = ramp_duplication(_gray(merged), gA0, gB0, sx)
        key = next((k for k in post
                    if k[0] <= (y0 + y1) // 2 < k[1]), None)
        preB = pre[key][1] if key else 1.0
        postB = post[key][1] if key else 1.0
        # step = across-seam L-R difference in the span rows (the
        # visible line), pre vs post.
        def _step(img):
            img = np.asarray(img, float)
            if img.ndim == 3:
                img = img.mean(axis=2)
            L = img[y0:y1, max(0, sx - 30):max(0, sx - 5)]
            R = img[y0:y1, min(W, sx + 5):min(W, sx + 30)]
            return float(abs(R.mean() - L.mean())) if L.size and R.size \
                else 0.0
        step_pre, step_post = _step(plain), _step(merged)
        for s in cert["spans"]:
            if s["action"] == "trim-only" and s["rect"] == [int(v)
                                                            for v in rect]:
                s["pre_B"], s["post_B"] = round(preB, 3), round(
                    postB, 3)
                s["pre_step"], s["post_step"] = round(step_pre, 2), \
                    round(step_post, 2)
        if not (postB <= 0.8 and step_post <= step_pre + 2.0):
            bad.append((rect, dx, sx))
            for s in cert["spans"]:
                if s["action"] == "trim-only" and s["rect"] == [int(v)
                                                                for v in rect]:
                    s["action"] = "skipped-needs-eye"
                    s["gate_refused"] = "runtime-post-trim"
    if bad:
        fixed = [f for f in fixed if f not in bad]
        if not fixed:
            return None, cert
        merged = repair_spans(eqA, eqB, wgt, fixed, fov_a, fov_b)
    return merged, cert
