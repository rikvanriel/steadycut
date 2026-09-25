"""The dark-mass instrument: an edge brackets a target, an amount never does.

The instrument is mount-independent, so nothing here mentions a helmet -- but
the numbers are the MKTB case's, because that is where they were measured.
"""
import numpy as np

from steadycut.framing.dark_mass import top_edge


def _stack(edge_frac: float, n: int = 8, h: int = 100, w: int = 160):
    """Frames with a dark mass below `edge_frac`, across x 40:120 of 160."""
    fr = np.full((n, h, w), 180, dtype=np.uint8)
    fr[:, int(edge_frac * h):, 40:120] = 20
    return fr


def test_the_edge_is_found_inside_the_band() -> None:
    e = top_edge(_stack(0.90), x0=0.2, x1=0.8)
    assert abs(e - 0.90) < 0.05


def test_a_mass_outside_the_band_is_not_claimed() -> None:
    """The band is the subject's x span: a sliver elsewhere must not set it."""
    assert np.isnan(top_edge(_stack(0.90), x0=0.85, x1=0.95))


def test_no_dark_mass_is_not_a_position() -> None:
    """NaN, not a value at the bottom of frame.

    Reporting a position for a mass that was never found is how a pitch that
    meets no criterion gets rendered: the sweep would interpolate against a
    number that means "nothing here".
    """
    assert np.isnan(top_edge(np.full((6, 100, 160), 180, dtype=np.uint8)))


def test_a_mass_present_in_only_some_frames_is_still_measured() -> None:
    """Presence is intermittent (the bike bounces in and out), so the edge is
    read per frame and taken at a low percentile rather than averaged."""
    frames = _stack(0.90)
    frames[:4] = 180                      # four frames with nothing visible
    e = top_edge(frames, x0=0.2, x1=0.8)
    assert abs(e - 0.90) < 0.05
