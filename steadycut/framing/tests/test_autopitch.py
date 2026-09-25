"""Autopitch tests: edge finding and interpolation on synthetic frames."""
import numpy as np

from steadycut.framing.autopitch import helmet_edge, solve_pitch


def _stack(edge_frac, n=8, h=100, w=160, dark_below=True):
    fr = np.full((n, h, w), 180, dtype=np.uint8)
    if dark_below:
        fr[:, int(edge_frac * h):, 40:120] = 20
    return fr


def test_edge_found_in_band() -> None:
    e = helmet_edge(_stack(0.90), x0=0.2, x1=0.8)
    assert abs(e - 0.90) < 0.05


def test_no_dark_mass_returns_bottom() -> None:
    e = helmet_edge(np.full((6, 100, 160), 180, dtype=np.uint8))
    assert e == 1.0


def test_solve_interpolates_and_clamps() -> None:
    got = solve_pitch([-5.0, -12.0, -19.0], [1.0, 0.93, 0.83], target=0.90)
    assert -19.0 < got < -12.0
    assert solve_pitch([-5.0, -12.0], [0.95, 0.93], target=0.90) == -12.0
    assert solve_pitch([-5.0, -12.0], [0.80, 0.70], target=0.90) == -5.0
