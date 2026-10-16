"""Per-clip FOV fit in the serving map builder (Task 1.2).

Grid search through sphere_maps.build_maps — the maps that serve — scoring
corridor sharpness (edge continuity across the seam) + photometric step at
BOTH seams, equator and mid-latitude. A fit done in any other renderer is
void (review catch), so this module owns build+score together.

Model honesty: pure equidistant, centered, untilted lenses. The mismatch
budget is stated in test_fov_fit (transfer proof), not hidden here.
"""
from __future__ import annotations

import numpy as np

from steadycut.render.sphere_maps import build_maps

GRID_A = (180.0, 181.0, 182.2, 183.5, 185.0, 186.0)
GRID_B = (176.0, 177.0, 178.5, 180.6, 182.0, 183.5)


def ghost_score(img):
    """Worst cross-seam self-match (duplication potential).

    For each seam: template just left, search just right (and vice versa);
    the max NCC at NONZERO offset. A trunk shown by both lenses scores
    ~0.9; continuous snow scores low. This is the term that trims margin:
    narrowing a lens's served FOV drops its far projection out of coverage
    and the repeat vanishes. Binned to 0.05 — finer differences are noise.
    """
    cv2 = __import__("cv2")
    import numpy as np
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    worst = 0.0
    for s in (w // 4, 3 * w // 4):
        for y0 in (h // 4, h // 2):
            band = g[y0:y0 + h // 4]
            t = band[:, s - 30:s - 5]
            if t.std() < 6:
                continue  # bland: nothing to double, no penalty
            r = cv2.matchTemplate(band[:, s - 5:s + 60], t,
                                  cv2.TM_CCOEFF_NORMED)
            r[:, :8] = -1  # ignore the zero-offset (continuity) peak
            worst = max(worst, float(r.max()))
    return worst


def render_pair(A, B, fov_a, fov_b, feather=1.0, w=1024, h=512):
    """Blend sphere thumbnail from two lens frames via the serving maps."""
    cv2 = __import__("cv2")
    (max_a, may_a), (max_b, may_b), wgt, _, _ = build_maps(
        w, h, fov_a, fov_b, feather_deg=feather)
    pa = cv2.remap(A, max_a, may_a, cv2.INTER_LANCZOS4,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    pb = cv2.remap(B, max_b, may_b, cv2.INTER_LANCZOS4,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    w3 = wgt[:, :, None].astype(np.float64)
    return np.clip(pa.astype(np.float64) * w3
                   + pb.astype(np.float64) * (1 - w3), 0, 255).astype(np.uint8)


def score_sphere(img):
    """(edge_ratio, step): continuity across both seams vs baseline.

    edge_ratio < 1 means the seam cuts less than average content — aligned
    content continues, misaligned content cuts. step is the worst plateau
    difference across the four seam crossings (2 seams x 2 bands).
    """
    cv2 = __import__("cv2")
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    seams = [int(w / 4), int(3 * w / 4)]
    sw = max(8, w // 128)  # corridor half-width scales with size
    bands = [(h // 2 - h // 6, h // 2 + h // 6),
             (h // 4 - h // 8, h // 4 + h // 8)]
    seam_e, steps = [], []
    base = []
    for y0, y1 in bands:
        for s in seams:
            seam_e.append(gx[y0:y1, s - sw:s + sw].mean())
        base.append(gx[y0:y1, w // 8:2 * w // 8].mean())
        base.append(gx[y0:y1, 6 * w // 8:7 * w // 8].mean())
        med = np.median(img[y0:y1].astype(np.float32), axis=0)
        d = np.abs(med[1:] - med[:-1]).sum(axis=1)
        for s in seams:
            seg = d[max(0, s - sw):s + sw]
            k = int(np.argmax(seg)) + max(0, s - sw)
            steps.append(float(np.abs(med[k + 1] - med[k]).sum()))
    ratio = float(np.mean(seam_e) / (np.mean(base) + 1e-6))
    return ratio, max(steps)


def _full_score(A, B, fa, fb, w=1024, h=512):
    """(ghost_bin, edge_ratio, step): duplication first (0.05 bins — finer
    is noise), then continuity, then tone. The ghost term is what trims
    margin: it prefers served FOVs under which far projections drop out."""
    img = render_pair(A, B, fa, fb, w=w, h=h)
    ratio, step = score_sphere(img)
    return round(ghost_score(img) / 0.05), ratio, step


def _refine(A, B, fa, fb, w=1024, h=512):
    """Hill-climb refinement: grid steps (1.0-1.5 deg) cannot support a 0.5
    deg recovery pin without it (review catch). Tries +-step on each axis
    per round, keeps the best, shrinks the step when nothing improves."""
    best_sc = _full_score(A, B, fa, fb, w=w, h=h)
    step = 1.0
    while step >= 0.25:
        improved = False
        for dfa in (-step, 0.0, step):
            for dfb in (-step, 0.0, step):
                if dfa == 0.0 and dfb == 0.0:
                    continue
                sc = _full_score(A, B, fa + dfa, fb + dfb, w=w, h=h)
                if sc < best_sc:
                    best_sc, fa, fb = sc, fa + dfa, fb + dfb
                    improved = True
        if not improved:
            step /= 2.0
    return fa, fb, best_sc


def fit(A, B, fa_grid=GRID_A, fb_grid=GRID_B, w=1024, h=512):
    """Grid + refinement. Returns (fov_a, fov_b, (ghost, edge, step), table).

    table rows: (fa, fb, ghost_bin, edge_ratio, step) for every grid point.
    """
    table = []
    for fa in fa_grid:
        for fb in fb_grid:
            table.append((fa, fb) + _full_score(A, B, fa, fb, w=w, h=h))
    table.sort(key=lambda r: (r[2], r[3], r[4]))
    fa, fb = table[0][0], table[0][1]
    rfa, rfb, rsc = _refine(A, B, fa, fb, w=w, h=h)
    return rfa, rfb, rsc, table
