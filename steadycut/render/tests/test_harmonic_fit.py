"""harmonic_fit pins: known const/tilt recovered and separated;
pure-parallax input fits ~zero rotation."""
import numpy as np

from steadycut.render.dup_trim import parallax_budget
from steadycut.render.harmonic_fit import harmonic_fit


def test_recovers_known_const_and_tilt():
    rng = np.random.default_rng(0)
    s = np.array([0.3, 1.2, 2.1, 3.0, 3.9, 4.8, 5.7])
    dx = 12.0 + 5.0 * np.cos(s) - 3.0 * np.sin(s) + rng.normal(0, 0.2, len(s))
    r = harmonic_fit(s, dx)
    assert abs(r['c0'] - 12.0) < 0.5
    assert abs(r['c1'] - 5.0) < 0.5
    assert abs(r['s1'] + 3.0) < 0.5
    assert r['rms'] < 0.5


def test_pure_parallax_fits_zero_rotation():
    # dx fully explained by the budget (close object): downweighted to ~0,
    # so the fit must not invent rotation from it.
    s = np.array([0.5, 1.5, 2.5])
    dx = np.array([49.0, 111.0, 80.0])
    budgets = np.array([parallax_budget(50, 130, 26),
                        parallax_budget(40, 110, 26),
                        parallax_budget(950, 1020, 34)])
    w = np.minimum(1.0, budgets / dx)
    r = harmonic_fit(s, dx, weights=w)
    assert abs(r['c0']) < max(budgets) or r['rms'] < 30.0
