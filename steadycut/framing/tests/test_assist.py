"""Assist triage math without models: agreement, slope filter, topmost.

Model inference itself is untested here (external weights, needs the `ml`
extra and a network on first use); everything around it -- the selection
logic that decides what the rider sees -- is pinned by these tests on
synthetic arrays.
"""
import numpy as np

from steadycut.framing.assist import AGREE, agreement, median_boundary
from steadycut.framing.assist import slope_keep, topmost


def test_topmost_reads_first_true_pixel_per_column() -> None:
    m = np.zeros((10, 4), dtype=bool)
    m[3, 0] = True
    m[7, 2] = True
    out = topmost(m, 10, 4)
    assert out[0] == 0.3
    assert np.isnan(out[1])
    assert out[2] == 0.7
    assert np.isnan(out[3])


def test_agreement_passes_coincident_boundaries() -> None:
    x = np.full(100, 0.31)
    assert agreement(x, x + 0.01, x - 0.01) <= AGREE


def test_agreement_rejects_a_divergent_third_model() -> None:
    x = np.full(100, 0.31)
    assert agreement(x, x + 0.01, x + 0.20) > AGREE


def test_agreement_refuses_without_common_cover() -> None:
    x = np.full(100, 0.31)
    y = np.full(100, np.nan)
    assert agreement(x, x, y) != agreement(x, x, y)  # NaN


def test_slope_filter_drops_trunk_dives() -> None:
    # A flat boundary with one vertical dive: the dive columns go.
    b = np.full(100, 0.3)
    b[50:55] = np.linspace(0.3, 0.9, 5)  # 0.6/5 per col = 86px/col at 720p
    keep = slope_keep(b, 720)
    assert keep[:50].all() and keep[56:].all()
    assert not keep[51:56].any()  # dive down and steep return up


def test_slope_filter_compares_pixels_not_fractions() -> None:
    # The unit bug that kept everything: 10px/col must fail a 60deg gate.
    b = np.full(100, 0.3)
    b[50] = 0.3 + 10.0 / 720.0
    keep = slope_keep(b, 720)
    assert not keep[50]


def test_median_boundary_ignores_nan() -> None:
    a = np.array([0.3, np.nan, 0.5])
    b = np.array([0.32, 0.31, np.nan])
    out = median_boundary(a, b)
    assert out[0] == 0.31
    assert out[1] == 0.31
    assert out[2] == 0.5
