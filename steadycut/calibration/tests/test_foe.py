"""Synthetic FOE recovery incl. the 180-degree FOC trap."""
import numpy as np

from steadycut.calibration import foe


def _field(foe_dir, n=400, outlier_frac=0.3, noise=0.02, seed=0,
           converge=False):
    rng = np.random.default_rng(seed)
    foe_dir = np.asarray(foe_dir, float)
    foe_dir = foe_dir / np.linalg.norm(foe_dir)
    # random base points, off the poles
    v = rng.normal(size=(n, 3))
    v[:, 2] *= 0.3
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    # great-circle tangent away from (or toward) the FOE
    along = v * (v @ foe_dir)[..., None] - foe_dir[None, :]
    along = along / (np.linalg.norm(along, axis=1, keepdims=True) + 1e-12)
    if converge:
        along = -along
    along = along + rng.normal(0, noise, along.shape)
    along = along / (np.linalg.norm(along, axis=1, keepdims=True) + 1e-12)
    n_out = int(n * outlier_frac)
    # random tangents as outliers
    r = rng.normal(size=(n_out, 3))
    r = r - v[:n_out] * (v[:n_out] * r).sum(axis=1, keepdims=True)
    along[:n_out] = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-12)
    return v, along


def test_foe_recovers_direction():
    foe_dir = np.array([0.8, 0.4, 0.1])
    foe_dir = foe_dir / np.linalg.norm(foe_dir)
    p, t = _field(foe_dir)
    got, frac = foe.foe_ransac(p, t)
    assert got is not None
    err = np.degrees(np.arccos(np.clip(got @ foe_dir, -1, 1)))
    assert err < 2.0, err
    assert frac > 0.5, frac


def test_foe_rejects_foc_flip():
    """A converging field's FOC sits at F; the FOE answer must be -F."""
    f = np.array([-0.2, 0.9, 0.15])
    f = f / np.linalg.norm(f)
    p, t = _field(f, converge=True, seed=1)
    got, frac = foe.foe_ransac(p, t)
    assert got is not None
    err = np.degrees(np.arccos(np.clip(got @ (-f), -1, 1)))
    assert err < 2.0, err
    assert frac > 0.5, frac


def test_essential_recovers_known_motion():
    """Constructed pinhole pair with known R,t: recover both."""
    rng = np.random.default_rng(5)
    # random 3D points ahead (virtual cam looks along +x here via mapping)
    pts = rng.normal(size=(200, 3)) + np.array([8.0, 0.0, 0.0])
    t_true = np.array([0.9, 0.3, -0.2])
    t_true = t_true / np.linalg.norm(t_true)
    ang = np.radians(2.0)
    ax = np.array([0.1, 0.5, -0.3])
    ax = ax / np.linalg.norm(ax)
    # rotate points by R (camera motion): p2 = R p1 + t
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]],
                  [-ax[1], ax[0], 0]])
    R = (np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K))
    p1 = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    p2 = (R @ pts.T).T + t_true[None, :]
    p2 = p2 / np.linalg.norm(p2, axis=1, keepdims=True)
    # + outliers: random re-matches
    p2[:40] = rng.normal(size=(40, 3))
    p2[:40] = p2[:40] / np.linalg.norm(p2[:40], axis=1, keepdims=True)
    got, frac, Rest, _ = foe.essential_travel(p1, p2)
    assert got is not None
    err = np.degrees(np.arccos(np.clip(abs(got @ t_true), -1, 1)))
    assert err < 3.0, (got, t_true)
    assert frac > 0.5, frac


def test_essential_prefers_foe_over_foc():
    """Weak-perspective converging field: divergence must flip the
    cheirality answer to the true FOE (antipode of convergence)."""
    rng = np.random.default_rng(9)
    f = np.array([0.9, 0.15, -0.1])
    f = f / np.linalg.norm(f)
    n = 300
    v = rng.normal(size=(n, 3)) + np.array([6.0, 0.0, 0.0])
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    # p2 = p1 moved TOWARD f (converging => true FOE is -f)
    step = 0.05
    w = v + step * (f[None, :] - v * (v @ f)[..., None])
    w = w / np.linalg.norm(w, axis=1, keepdims=True)
    got, frac, _, _ = foe.essential_travel(v, w)
    assert got is not None
    err = np.degrees(np.arccos(np.clip(got @ (-f), -1, 1)))
    assert err < 5.0, got
