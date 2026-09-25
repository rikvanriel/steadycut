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


def test_the_workdir_is_created_when_missing(tmp_path) -> None:
    """The caller passes a fresh temp subdirectory, so it does NOT exist yet.

    ffmpeg fails with a bare exit status when the output directory is absent,
    which made the whole automatic framing path refuse the first time it was
    exercised without an explicit --pitch.
    """
    from steadycut.framing.autopitch import _workdir
    target = tmp_path / "framing" / "nested"
    assert not target.exists()
    out = _workdir(target)
    assert out == target
    assert out.is_dir()


def test_brackets_tells_a_solved_ladder_from_a_clipped_one() -> None:
    """solve_pitch returns an END of the ladder when the target is never met.

    A caller that only asks "was an edge found anywhere?" cannot tell a solved
    pitch from one nobody checked -- which is how a -5.0 pitch got reported as
    usable on a window whose edges never got past 0.78.
    """
    from steadycut.framing.autopitch import brackets
    # Real numbers from a May 26 window: 0.5717 (pitch -26) .. 0.7778 (-5).
    assert brackets({-26.0: 0.5717, -12.0: 0.7003, -5.0: 0.7778}) is False
    assert brackets({-26.0: 0.5717, -12.0: 0.91, -5.0: 0.7778}) is True
    assert brackets({-5.0: 0.9999}) is False
    assert brackets({}) is False
