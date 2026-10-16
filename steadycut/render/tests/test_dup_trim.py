"""Pins for per-location duplicate trim: clean, double, apply, clamp."""
import numpy as np

from steadycut.render.dup_trim import (apply_trim, find_duplicates,
                                        ramp_duplication)


def _textured_pair(dx=0, h=256, w=2048, seed=7):
    rng = np.random.default_rng(seed)
    A = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    B = np.zeros_like(A)
    if dx == 0:
        B = A.copy()
    else:
        # B-side double: the same content sits dx px RIGHT in B
        B[:, dx:] = A[:, :w - dx]
    return A, B


def test_identical_pair_is_clean():
    A, B = _textured_pair(dx=0)
    assert find_duplicates(A, B) == []


def test_bland_pair_is_clean():
    A = np.full((256, 2048, 3), 180, np.uint8)
    assert find_duplicates(A, A.copy()) == []


def test_shifted_pair_reports_b_span():
    dx = 60
    A, B = _textured_pair(dx=dx)
    rects = find_duplicates(A, B)
    assert rects, "a 60 px-shifted pair must report duplicated B spans"
    assert any(x0 >= 1536 for _, _, x0, _ in rects), rects
    for y0, y1, x0, x1 in rects:
        assert y1 - y0 >= 20 and x1 - x0 >= 16


def test_apply_trim_zeroes_span_keeps_rest():
    w = np.ones((256, 2048), np.float32)
    out = apply_trim(w, [(100, 200, 1540, 1600)])
    assert out[150, 1570] < 0.01
    assert out[150, 1400] == 1.0 and out[50, 1570] == 1.0


def _edge_pair():
    """Foreign-blob double: A is background grating; B pastes a
    vertically-coherent foreign strip over it (the B-copy). Tall like a
    trunk, so every vertical third reads the same."""
    import cv2
    H, W = 256, 2048
    xs = np.arange(W)
    G = (150 + 40 * np.sin(2 * np.pi * xs / 10)).astype(np.uint8)
    A = np.tile(G, (H, 1))
    row = cv2.GaussianBlur(
        np.random.default_rng(3).integers(0, 255, (1, 90)).astype(np.uint8),
        (0, 0), 3)
    B = A.copy()
    B[:, 1530:1620] = np.tile(row, (H, 1))
    A3 = np.stack([A, A, A], axis=-1)
    B3 = np.stack([B, B, B], axis=-1)
    return A3, B3


def test_ramp_gate_reads_double_then_single():
    H = 256
    A, B = _edge_pair()
    import cv2
    gA = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    gB = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
    double = gA.copy()
    double[:, 1536:] = gB[:, 1536:]      # ramp zone carries both copies
    both = ramp_duplication(double, gA, gB, 1536, band=16, tpos=24)
    (sA, sB), = [v for (y0, y1), v in both.items() if y0 == H // 3]
    single = ramp_duplication(gA, gA, gB, 1536, band=16, tpos=24)
    (sA2, sB2), = [v for (y0, y1), v in single.items() if y0 == H // 3]
    assert sB > 0.5, sB                  # B-copy in ramp reads high
    assert sB2 < 0.3, sB2                # A-only ramp: B at baseline
    assert sA > 0.2 and sA2 > 0.2        # A continues in either way
