"""Harmonic alignment fit (Step 2, report-only): residual dx around the
seam decomposed into const + cos/sin of seam angle.

Model: dx(s) = c0 + c1*cos(s) + s1*sin(s), s = seam angle. Constant =
relative yaw; 1/rev = relative tilt/roll. Fit by closed-form least
squares on gated static constraints; parallax-heavy spans downweighted
by the T1 budget (weight = min(1, budget/dx)). Output is a report dict,
never served.
"""
from __future__ import annotations

import numpy as np


def design_matrix(angles: np.ndarray) -> np.ndarray:
    """[1, cos(s), sin(s)] columns."""
    a = np.asarray(angles, float)
    return np.stack([np.ones_like(a), np.cos(a), np.sin(a)], axis=1)


def harmonic_fit(angles, dx, weights=None) -> dict:
    """Closed-form weighted LSQ. Returns coeffs + residual + per-point fit."""
    A = design_matrix(np.asarray(angles, float))
    y = np.asarray(dx, float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    Aw = A * w[:, None]
    coef, *_ = np.linalg.lstsq(Aw, y * w, rcond=None)
    pred = A @ coef
    resid = y - pred
    return {'c0': float(coef[0]), 'c1': float(coef[1]),
            's1': float(coef[2]), 'rms': float(np.sqrt((resid ** 2).mean())),
            'resid': resid, 'pred': pred}
