"""Focus of expansion from translational flow on the sphere.

After gyro derotation only translational flow remains: all flow vectors
diverge from the focus of expansion (FOE) along great circles, and
converge at the antipodal focus of contraction (FOC). RANSAC over
flow→direction constraints finds the FOE; the divergence SIGN along each
great circle disambiguates FOE from FOC (a pure-direction fit can return
a 180-degree flip — pinned by test).

Conventions here are internal (azimuth CCW from +x, elevation from the
equator); the mapping to lens/body frames is pinned once, at validation
(Mar05 ~ 0, Jan26 ~ -90), not derived from paper.
"""
from __future__ import annotations

import numpy as np

RANSAC_ITERS = 400
RANSAC_TOL_RAD = np.radians(2.0)


def pix_to_dir(xs, ys, w, h):
    """Equirect pixels to unit direction vectors."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    az = (xs / (w - 1) - 0.5) * 2 * np.pi
    el = (0.5 - ys / (h - 1)) * np.pi
    ce = np.cos(el)
    return np.stack([ce * np.cos(az), ce * np.sin(az), np.sin(el)], -1)


def flow_to_rays(xs, ys, us, vs, w, h):
    """Each flow sample as (base direction, tangent direction of motion)."""
    p = pix_to_dir(xs, ys, w, h)
    # pixel displacement -> angular displacement (equirect is linear)
    daz = (np.asarray(us, float) / (w - 1)) * 2 * np.pi
    del_ = -(np.asarray(vs, float) / (h - 1)) * np.pi
    az = (xs / (w - 1) - 0.5) * 2 * np.pi
    el = (0.5 - ys / (h - 1)) * np.pi
    # east and north tangent basis
    east = np.stack([-np.sin(az), np.cos(az), np.zeros_like(az)], -1)
    north = np.stack([-np.sin(el) * np.cos(az), -np.sin(el) * np.sin(az),
                      np.cos(el)], -1)
    t = east * daz[..., None] + north * del_[..., None]
    n = np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    return p, t / n


def essential_travel(p1, p2, seed=0):
    """Travel direction from two ray sets via the essential matrix.

    No derotation needed: rotation and translation are fit jointly, so
    gyro sync/axis-map errors cannot bias the bearing (they only remove
    pairs from the consensus). Rays go through a virtual pinhole about
    the +x (lens) axis; output is a unit vector in lens-sphere coords.
    Returns (unit vector | None, inlier fraction, R estimate).
    """
    cv2 = __import__("cv2")
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    front = p1[:, 0] > 0.15
    f1, f2 = p1[front], p2[front]
    if len(f1) < 8:
        return None, 0.0, None
    q1 = np.stack([f1[:, 1] / f1[:, 0], f1[:, 2] / f1[:, 0]], -1)
    q2 = np.stack([f2[:, 1] / f2[:, 0], f2[:, 2] / f2[:, 0]], -1)
    E, mask = cv2.findEssentialMat(q1, q2, focal=1.0, pp=(0.0, 0.0),
                                   method=cv2.RANSAC, threshold=0.002)
    if E is None or mask is None:
        return None, 0.0, None
    _, R, tvec, m2 = cv2.recoverPose(E, q1, q2, focal=1.0, pp=(0.0, 0.0),
                                     mask=mask)
    # E-frame (X_E, Y_E, Z_E) = (lens Y, lens Z, lens X): map back.
    t = tvec.ravel() / (np.linalg.norm(tvec.ravel()) + 1e-12)
    out = np.array([t[2], t[0], t[1]])
    return out / np.linalg.norm(out), float(m2.sum()) / len(f1), R


def rotation_agreement(R_ess, R_gyro, tol_deg=3.0):
    """Do the vision-fit rotation and the gyro delta agree? Guards the
    pure-rotation degeneracy (stopped camera): E fits garbage t there,
    but its R still matches the gyro, so pair with translation floor."""
    R_ess = np.asarray(R_ess, float)
    R_gyro = np.asarray(R_gyro, float)
    d = R_ess @ R_gyro.T
    ang = np.degrees(np.arccos(np.clip((np.trace(d) - 1) / 2, -1, 1)))
    return bool(ang < tol_deg), float(ang)


def foe_ransac(p, t, iters=RANSAC_ITERS, tol=RANSAC_TOL_RAD, seed=0):
    """RANSAC FOE with FOE/FOC disambiguation. Returns (unit vector,
    inlier fraction). Signs the candidate by divergence: flow must point
    AWAY from the FOE along each inlier's great circle (toward it = FOC).
    """
    rng = np.random.default_rng(seed)
    p = np.asarray(p, float)
    t = np.asarray(t, float)
    n = len(p)
    if n < 3:
        return None, 0.0
    nv = np.cross(p, t)
    nv = nv / (np.linalg.norm(nv, axis=1, keepdims=True) + 1e-12)
    best, best_in = None, np.zeros(n, bool)
    idx = np.arange(n)
    for _ in range(iters):
        pick = rng.choice(idx, 3, replace=False)
        try:
            _, _, vt = np.linalg.svd(nv[pick], full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        cand = vt[-1] / (np.linalg.norm(vt[-1]) + 1e-12)
        resid = np.abs(nv @ cand)
        inl = resid < tol
        if inl.sum() > best_in.sum():
            best, best_in = cand, inl
    if best is None or best_in.sum() < 3:
        return None, 0.0
    # refine on inliers, then disambiguate by divergence sign
    _, _, vt = np.linalg.svd(nv[best_in], full_matrices=False)
    foe = vt[-1] / (np.linalg.norm(vt[-1]) + 1e-12)
    away = np.einsum("ij,ij->i", t[best_in],
                     foe[None, :] - p[best_in] *
                     (p[best_in] @ foe)[..., None])
    if float(np.median(away)) < 0:
        foe = -foe
    return foe, float(best_in.mean())
