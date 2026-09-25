"""Metrics package tests: estimators on synthetic ground truth.

The skill requires every metric validated against a synthetic signal whose
ground truth is genuinely known. Phase correlation on a shifted noise field
and rotation_between on identical frames are the two cheapest such checks;
both run anywhere with no footage.
"""
import numpy as np

from steadycut.metrics.parallax import shift as pshift
from steadycut.metrics.rotate import fit_rotation, rotation_between


def _noise(h=180, w=320, seed=7):
    rng = np.random.default_rng(seed)
    return (rng.random((h, w)) * 255).astype(np.float32)


def test_phase_shift_recovers_integer_translation() -> None:
    # Convention (used consistently by all callers): shift(a, b) returns the
    # shift that aligns b onto a, i.e. the negative of the content motion.
    a = _noise()
    dx, dy = 9, -14
    b = np.roll(a, shift=(dy, dx), axis=(0, 1))
    gx, gy = pshift(a, b)
    assert abs(gx + dx) < 1.0
    assert abs(gy + dy) < 1.0


def test_identical_frames_have_no_rotation() -> None:
    a = _noise().astype(np.uint8)
    got = rotation_between(a, a)
    assert got is None or abs(got) < 0.5


def test_fit_rotation_needs_three_patches() -> None:
    import math
    assert math.isnan(fit_rotation(np.zeros((2, 2)), np.zeros((2, 2))))
