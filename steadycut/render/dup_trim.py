"""Per-location duplicate trim (production): the ramp must never blend across
duplicated content.

Proven failure: a near-field trunk's B-copy (63 px wide) co-occurs with A's
copy in the joint; every seam placement cuts through B's copy and the ramp
mixes both into a translucent overlay (both-copies-in-ramp 0.88+). Routing
cannot dodge it (no placement clears a 63 px copy in a 46 px joint); blending
cannot average it away. Only selection works: invalidate B's duplicated span
so the ramp blends A with A.

Method: on the shared-coords pair, template windows from the A-side near the
seam are matched into B's shown region. A sharp high match at a displaced
offset = the same structure served twice = trim B's span (feathered). Guards:
inside the joint only (union preserved: A covers where B is trimmed),
textured templates only, big structured templates (forest gross-similarity
plateaus at ~0.81 over small windows; true copies hold 0.86+ over 32x120),
peak-sharpness margin against runner-up.
"""
from __future__ import annotations

import numpy as np
from typing import NamedTuple


class Span(NamedTuple):
    """A detected duplicate span with its detection internals.

    Rect-first: s[0:4] == (y0, y1, x0, x1), so every existing 4-tuple
    unpacker keeps working. Extra fields: mutual (back-match gate),
    fwd (forward NCC score), tx (A-side template x), disp (bx0-tx),
    sup0/sup1 (copy-support columns: NCC matched-window union, SIFT
    undilated keypoint hull). Merged mutual = AND (a forward-only
    grown row must never corroborate via OR).
    """
    y0: int
    y1: int
    x0: int
    x1: int
    mutual: bool
    fwd: float | None
    tx: int
    disp: float | None
    sup0: int
    sup1: int

SEAMS = (512, 1536)
HALFW = 100          # joint corridor half-width (validity clamp domain)
SCAN_W = 200         # A-side template scan width (near-field pairs separate
                     # up to ~100 px; the corridor alone misses the A-copy)
SEARCH_W = 150       # B-side search half-width (very-near copies sit deep)
TEMPLATE_W = 32      # match window width (big: rejects gross-similarity)
TEMPLATE_H = 120     # match window height
ROW_STRIDE = 20
DUP_SCORE = 0.84     # NCC floor for a duplicated span (raw pair; trunk 0.955)
DUP_SCORE_FWD = 0.90 # forward-only tier: different-scale copies (trunk B
                     # 63 px vs A 30 px) fail the mutual gate; the 0.90 bar
                     # keeps forest plateaus (<=0.83 measured) out.
BACK_TOL = 4         # mutual-match tolerance (px): B-copy must match back
MIN_OFFSET = 8       # copy center right of seam+ramp: genuinely B-shown
FEATHER = 6          # trim edge feather (px)


def _gray(img):
    cv2 = __import__("cv2")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def find_duplicates(eqA, Bal, seams=SEAMS, halfw=HALFW,
                    fov_a=185.0, fov_b=183.5):
    """Returns list of (y0, y1, x0, x1) trim rects in SHARED sphere coords:
    B spans shown twice must be invalidated. Empty = clean.
    eqA/Bal: full equirects (BGR uint8) in shared coords (v360 outputs
    align: lens0@yaw0 with lens1@yaw180, no roll)."""
    cv2 = __import__("cv2")
    from steadycut.render.sphere_maps import joint_valid_strip
    gA = _gray(eqA)
    gB = _gray(Bal)
    H, W = gA.shape
    rects = []
    for sx in seams:
        x0s = max(0, sx - halfw)
        jv = joint_valid_strip(sx, halfw, fov_a, fov_b, H, W)
        for th, stride in ((TEMPLATE_H, ROW_STRIDE), (60, 10)):
            for y0 in range(0, H - th, stride):
                y1 = y0 + th
                for tx in range(sx - SCAN_W, sx - TEMPLATE_W, stride):
                    rects.extend(_scan_band(
                        gA, gB, cv2, jv, sx, x0s, tx, y0, y1))
    return _merge_rects(rects)


def find_duplicates_detailed(eqA, Bal, seams=SEAMS, halfw=HALFW,
                             fov_a=185.0, fov_b=183.5):
    """find_duplicates with Span details (mutual, fwd, tx, disp,
    support). Zero behavior change to detection itself; the merge
    ANDs mutual across constituents. Callers needing rects keep
    using find_duplicates."""
    cv2 = __import__("cv2")
    from steadycut.render.sphere_maps import joint_valid_strip
    gA = _gray(eqA)
    gB = _gray(Bal)
    H, W = gA.shape
    spans = []
    for sx in seams:
        x0s = max(0, sx - halfw)
        jv = joint_valid_strip(sx, halfw, fov_a, fov_b, H, W)
        for th, stride in ((TEMPLATE_H, ROW_STRIDE), (60, 10)):
            for y0 in range(0, H - th, stride):
                y1 = y0 + th
                for tx in range(sx - SCAN_W, sx - TEMPLATE_W, stride):
                    spans.extend(_scan_band(
                        gA, gB, cv2, jv, sx, x0s, tx, y0, y1))
    return _merge_spans(spans)


def _scan_band(gA, gB, cv2, jv, sx, x0s, tx, y0, y1,
               dup_floor=DUP_SCORE, fwd_floor=DUP_SCORE_FWD,
               x_prior=None, x_tol=30):
    """One template position: forward match into B, mutual/tier-2 gate,
    joint clamp (rows then cols). dup_floor lowers for growth (0.70:
    continuation of a known double, position-priored); x_prior constrains
    the copy center near the seed (copies move smoothly). Returns rects."""
    t = gA[y0:y1, tx:tx + TEMPLATE_W].astype(np.float32)
    if float(t.std()) < 8:
        return []  # bland: nothing to double
    found = []
    s0 = sx - SEARCH_W
    s = gB[y0:y1, s0:sx + SEARCH_W].astype(np.float32)
    r = cv2.matchTemplate(s, t, cv2.TM_CCOEFF_NORMED)
    flat = r[0]
    i = int(flat.argmax())
    best = float(flat[i])
    bx0 = s0 + i
    if best < dup_floor or bx0 + TEMPLATE_W // 2 < sx + MIN_OFFSET:
        return []  # weak, or copy center inside the ramp anyway
    if x_prior is not None and abs(bx0 - x_prior) > x_tol:
        return []  # growth: copy must continue near the seed
    if bx0 < x0s or bx0 + TEMPLATE_W > x0s + jv.shape[1]:
        return []  # outside the clamp domain: never trim blind
    # mutual gate: B's copy must match back to the A window.
    # Lookalikes match one way (plateau); true copies are mutual.
    # Tier 2: different-scale copies (near-field magnification)
    # fail mutual but clear 0.90 forward (forest <= 0.83).
    tb = gB[y0:y1, bx0:bx0 + TEMPLATE_W].astype(np.float32)
    sa = gA[y0:y1, sx - SCAN_W:sx + SCAN_W].astype(np.float32)
    rb = cv2.matchTemplate(sa, tb, cv2.TM_CCOEFF_NORMED)[0]
    j = int(rb.argmax())
    back = float(rb[j])
    mutual = (abs((sx - SCAN_W + j) - tx) <= BACK_TOL
              and back >= dup_floor)
    if not mutual and best < fwd_floor:
        return []
    # clamp to the joint: trimming B where only B covers would
    # open a hole (union preserved: A covers the kept rows).
    # Run-based like the SIFT path (a 32 px template never sits wholly
    # inside the ~28 px joint; require >=16 valid-col runs instead).
    bx1 = bx0 + TEMPLATE_W
    cols = slice(max(0, bx0 - x0s), min(jv.shape[1], bx1 - x0s))
    sub = jv[y0:y1, cols]
    col_ok = sub.all(axis=0)
    xs = np.where(col_ok)[0]
    if len(xs) == 0:
        return []
    # contiguous runs in absolute coords
    runs, rs, re = [], xs[0], xs[0]
    for v in xs[1:]:
        if v == re + 1:
            re = v
        else:
            runs.append((rs, re)); rs, re = v, v
    runs.append((rs, re))
    base = max(0, bx0 - x0s)
    disp = float(bx0 - tx)
    for rs, re in runs:
        if re + 1 - rs < 16:
            continue
        ax0, ax1 = x0s + base + rs, x0s + base + re + 1
        # rows where this run holds throughout (rs/re are sub-relative)
        rsub = jv[y0:y1, base + rs:base + re + 1]
        rows_ok = rsub.all(axis=1)
        ys = np.where(rows_ok)[0]
        if len(ys) < 20:
            continue
        found.append(Span(y0 + int(ys[0]), y0 + int(ys[-1]) + 1, ax0,
                          ax1, mutual, best, tx, disp, bx0, bx1))
    return found


def find_duplicates_sift(eqA, eqB, seams=SEAMS, y_lo=0, y_hi=1024,
                         fov_a=185.0, fov_b=183.5, min_dx=MIN_OFFSET,
                         max_dx=150, max_dy=10, min_matches=3, dilate=16):
    """SIFT-correspondence duplicate spans. NCC plateaus on forest (every
    window matches everywhere ~0.8); descriptors punch through rim
    rendering differences and vote a clean displacement (measured: 13/15
    good matches at +50/+55 px on the trunk). Matches with MIN_OFFSET <=
    dx <= 150, |dy| <= 10 cluster into B-side spans (dilated, merged,
    joint-clamped like NCC detection). y_lo/y_hi bound the search band."""
    cv2 = __import__("cv2")
    from steadycut.render.sphere_maps import joint_valid_strip
    gA = eqA if eqA.ndim == 2 else cv2.cvtColor(eqA, cv2.COLOR_BGR2GRAY)
    gB = eqB if eqB.ndim == 2 else cv2.cvtColor(eqB, cv2.COLOR_BGR2GRAY)
    H, W = gA.shape
    sift = cv2.SIFT_create()
    rects = []
    for sx in seams:
        x0s = max(0, sx - HALFW)
        jv = joint_valid_strip(sx, HALFW, fov_a, fov_b, H, W)
        # per-band matching: full-frame keypoint pools drown the ratio
        # test (forest descriptors all alike); 200-row bands discriminate.
        for by0 in range(y_lo, min(y_hi, H), 200):
            by1 = min(y_hi, by0 + 200)
            pts = _sift_band(sift, cv2, gA, gB, sx, by0, by1,
                             min_dx, max_dx, max_dy)
            # cluster B-side points: sort by y, group gaps <= 40 rows
            pts.sort(key=lambda p: p[1])
            groups, cur = [], []
            for p in pts:
                if cur and p[1] - cur[-1][1] > 40:
                    groups.append(cur); cur = []
                cur.append(p)
            if cur:
                groups.append(cur)
            for g in groups:
                if len(g) < min_matches:
                    continue
                xs = [p[0] for p in g]
                ys = [p[1] for p in g]
                # undilated keypoint hull = copy support (dilate=16 adds
                # 32 px; gates consume the hull, not the grown rect).
                hx0, hx1 = int(min(xs)), int(max(xs))
                hy0, hy1 = int(min(ys)), int(max(ys))
                x0, x1 = hx0 - dilate, hx1 + dilate
                y0, y1 = hy0 - dilate, hy1 + dilate
                # joint clamp (union preserved)
                cols = slice(max(0, x0 - x0s), min(jv.shape[1], x1 - x0s))
                sub = jv[max(0, y0):min(H, y1), cols]
                ok = sub.all(axis=1)
                ys_ok = np.where(ok)[0]
                if len(ys_ok) < 16:
                    continue
                ky0 = max(0, y0) + int(ys_ok[0])
                ky1 = max(0, y0) + int(ys_ok[-1]) + 1
                col_ok = jv[ky0:ky1, cols].all(axis=0)
                xs_ok = np.where(col_ok)[0]
                if len(xs_ok) < 16:
                    continue
                base = max(0, x0 - x0s)
                rects.append((ky0, ky1, x0s + base + int(xs_ok[0]),
                              x0s + base + int(xs_ok[-1]) + 1))
    return _merge_rects(rects)


def _sift_band(sift, cv2, gA, gB, sx, by0, by1,
               min_dx, max_dx, max_dy):
    """One band: SIFT match A-side vs B-side, keep displaced copies."""
    a = gA[by0:by1, sx - SCAN_W:sx]
    b = gB[by0:by1, sx - SEARCH_W:sx + SEARCH_W]
    kA, dA = sift.detectAndCompute(a, None)
    kB, dB = sift.detectAndCompute(b, None)
    pts = []
    if dA is None or dB is None or len(kA) < 2 or len(kB) < 2:
        return pts
    for pair in cv2.BFMatcher().knnMatch(dA, dB, k=2):
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance >= 0.75 * second.distance:
            continue
        ax = kA[first.queryIdx].pt[0] + sx - SCAN_W
        ay = kA[first.queryIdx].pt[1] + by0
        bx = kB[first.trainIdx].pt[0] + sx - SEARCH_W
        by = kB[first.trainIdx].pt[1] + by0
        dx, dy = bx - ax, by - ay
        if min_dx <= dx <= max_dx and abs(dy) <= max_dy:
            pts.append((bx, by))
    return pts


def _merge_rects(rects):
    """Merge x-overlapping, row-adjacent rects (copies span bands)."""
    spans = [r if isinstance(r, Span) else Span(*r, False, None, 0, None,
                                                r[2], r[3]) for r in rects]
    return [tuple(s[:4]) for s in _merge_spans(spans)]


def _merge_spans(spans):
    """Merge Spans: rect union as _merge_rects, mutual ANDed (a
    forward-only grown row must never corroborate via OR), support
    unioned, tx/disp kept from the max-fwd constituent."""
    if not spans:
        return []
    rs = sorted(spans)
    out = [rs[0]]
    for s in rs[1:]:
        p = out[-1]
        if s.y0 <= p.y1 + ROW_STRIDE and s.x0 <= p.x1 and s.x1 >= p.x0:
            f = s if (s.fwd or 0) >= (p.fwd or 0) else p
            out[-1] = Span(min(p.y0, s.y0), max(p.y1, s.y1),
                           min(p.x0, s.x0), max(p.x1, s.x1),
                           p.mutual and s.mutual, f.fwd, f.tx, f.disp,
                           min(p.sup0, s.sup0), max(p.sup1, s.sup1))
        else:
            out.append(s)
    return out


def count_instances(lensA_img, lensB_img, tx, bx0, bx1, rows):
    """Finding-25 instance count across the seam, both lenses.

    Window spans tx..bx1 in BOTH lenses (not the clamped rect).
    Counts content runs in the column-mean profile (a trunk/skier
    reads as a level shift coherent down the rows; noise does not).
    Hole test: middle columns bland in both lenses while outer
    thirds carry content = halves with nothing in the middle
    (naive counting reports (1,1) and would WRONGLY pass trim).
    Returns (na, nb, kind, conf) with kind in double/pair/hole/skip;
    low-confidence or bland-everywhere -> skip (abstention, never
    condemnation).
    """
    cv2 = __import__("cv2")
    A = np.asarray(lensA_img)
    B = np.asarray(lensB_img)
    A = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY) if A.ndim == 3 else A
    B = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY) if B.ndim == 3 else B
    rs = list(rows)
    if not rs:
        return 0, 0, "skip", 0.0
    y0, y1 = max(0, min(rs)), min(A.shape[0], max(rs) + 1)
    x0, x1 = max(0, tx), min(A.shape[1], bx1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0, 0, "skip", 0.0

    def runs(g):
        prof = g[y0:y1, x0:x1].astype(np.float32).mean(axis=0)
        bg = float(np.median(prof))
        mad = float(np.median(np.abs(prof - bg))) + 1e-6
        hot = np.abs(prof - bg) > 3 * mad
        out, start = [], None
        for i, v in enumerate(hot):
            if v and start is None:
                start = i
            elif not v and start is not None:
                if i - start >= 3:
                    out.append((start, i))
                start = None
        if start is not None and len(hot) - start >= 3:
            out.append((start, len(hot)))
        strength = float(np.abs(prof - bg).max() / mad)
        return out, strength

    ra, ca = runs(A)
    rb, cb = runs(B)
    na, nb = len(ra), len(rb)
    conf = min(1.0, (ca + cb) / 12.0)
    # hole: middle third bland in both, outer thirds structured.
    w3 = max(1, (x1 - x0) // 3)
    midA = A[y0:y1, x0 + w3:x0 + 2 * w3]
    midB = B[y0:y1, x0 + w3:x0 + 2 * w3]
    outA = np.concatenate([A[y0:y1, x0:x0 + w3].ravel(),
                           A[y0:y1, x0 + 2 * w3:x1].ravel()])
    outB = np.concatenate([B[y0:y1, x0:x0 + w3].ravel(),
                           B[y0:y1, x0 + 2 * w3:x1].ravel()])
    if (float(midA.std()) < 8 and float(midB.std()) < 8
            and float(outA.std()) >= 8 and float(outB.std()) >= 8):
        return na, nb, "hole", conf
    if na == 0 and nb == 0:
        return na, nb, "skip", 0.0
    if conf < 0.25:
        return na, nb, "skip", conf
    if na == 1 and nb == 1:
        # full-extent match: A copy at tx matches B at bx0 over the
        # whole support, not one edge (else halves pass as double).
        return na, nb, "double", conf
    return na, nb, "pair", conf


def apply_trim(weight, rects, feather=FEATHER):
    """weight: (H, W) float blend weight (1 = B). Zero B over trim rects
    with cosine-feathered edges. Returns the trimmed weight."""
    w = weight.copy().astype(float)
    yy, xx = np.mgrid[0:w.shape[0], 0:w.shape[1]].astype(float)
    for y0, y1, x0, x1 in rects:
        fx = np.clip((xx - (x0 - feather)) / feather, 0, 1) * np.clip(
            ((x1 + feather) - xx) / feather, 0, 1)
        fy = np.clip((yy - (y0 - feather)) / feather, 0, 1) * np.clip(
            ((y1 + feather) - yy) / feather, 0, 1)
        keep = np.clip(fx * fy, 0, 1)
        w *= (1.0 - keep)
    return w


def ramp_duplication(merged_gray, eqA_gray, eqB_gray, seam_x,
                     band=20, tpos=30):
    """Gate metric, per vertical third: is each lens's copy present in the
    merge's ramp zone? Templates come from the UNBLENDED lenses (A-side
    copy, B-side copy); the ramp from the merge. Returns {(y0,y1):
    (scoreA, scoreB)}. Post-trim reads A high / B at baseline (B gone);
    a double reads high on BOTH. Templates from the merge would be
    self-defeating (post-trim both sides are A-derived)."""
    cv2 = __import__("cv2")
    g = np.asarray(merged_gray, dtype=np.float32)
    H, _ = g.shape
    out = {}
    for y0, y1 in ((0, H // 3), (H // 3, 2 * H // 3), (2 * H // 3, H)):
        y = slice(y0, y1)
        tA = np.asarray(eqA_gray[y, seam_x - tpos - band:seam_x - tpos],
                        dtype=np.float32)
        tB = np.asarray(eqB_gray[y, seam_x + tpos:seam_x + tpos + band],
                        dtype=np.float32)
        ramp = g[y, seam_x - band // 2:seam_x + band // 2]
        sA = float(cv2.matchTemplate(ramp, tA, cv2.TM_CCOEFF_NORMED).max())
        sB = float(cv2.matchTemplate(ramp, tB, cv2.TM_CCOEFF_NORMED).max())
        out[(y0, y1)] = (sA, sB)
    return out


def grow_rects(eqA, eqB, seeds, fov_a=185.0, fov_b=183.5,
               reach=200, dup_floor=0.70):
    """Grow seeded copies into neighboring rows with sensitive bars (0.70:
    continuation of a known double) constrained by position prior (copy
    center within 40 px of the seed: copies move smoothly). Joint-clamped
    like detection (union preserved). Returns grown rects (seeds incl.)."""
    cv2 = __import__("cv2")
    from steadycut.render.sphere_maps import joint_valid_strip
    gA = eqA if eqA.ndim == 2 else cv2.cvtColor(eqA, cv2.COLOR_BGR2GRAY)
    gB = eqB if eqB.ndim == 2 else cv2.cvtColor(eqB, cv2.COLOR_BGR2GRAY)
    gA = gA.astype(np.float32)
    gB = gB.astype(np.float32)
    H, W = gA.shape
    jv_cache = {}
    grown = [tuple(s) for s in seeds]
    for y0, y1, x0, x1 in seeds:
        sx = 512 if x0 < 1024 else 1536
        if sx not in jv_cache:
            x0s = max(0, sx - HALFW)
            jv_cache[sx] = (joint_valid_strip(sx, HALFW, fov_a, fov_b,
                                              H, W), x0s)
        jv, x0s = jv_cache[sx]
        for lo, hi in ((max(0, y0 - reach), y0), (y1, min(H, y1 + reach))):
            for ty0 in range(lo, hi - 60, 10):
                ty1 = ty0 + 60
                for tx in range(sx - SCAN_W, sx - TEMPLATE_W, 20):
                    grown.extend(_scan_band(
                        gA, gB, cv2, jv, sx, x0s, tx, ty0, ty1,
                        dup_floor=dup_floor, fwd_floor=dup_floor,
                        x_prior=x0, x_tol=40))
    return _merge_rects(grown)
