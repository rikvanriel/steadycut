"""Pins for seam routing (Task 3.1): API shape, bland preference, bounds."""
import numpy as np

from steadycut.render.seam_route import route_both, route_seam


def _pair(busy_side="right"):
    """A bland left half both eyes share; right half textured in A only."""
    h, w = 200, 256
    A = np.full((h, w, 3), 128, np.uint8)
    B = np.full((h, w, 3), 128, np.uint8)
    rng = np.random.default_rng(7)
    tex = (rng.random((h, w // 2, 3)) * 255).astype(np.uint8)
    if busy_side == "right":
        A[:, w // 2:] = tex
    else:
        A[:, :w // 2] = tex
    return A, B


def test_routes_through_bland_region():
    A, B = _pair("right")
    poly = route_seam(A, B, 128, halfw=100)
    # Right half differs (high cut cost), left is identical (zero cost):
    # the seam must live in the bland left half.
    assert (poly < 128).mean() > 0.9, poly[::40]


def test_output_stays_inside_the_strip():
    rng = np.random.default_rng(3)
    A = (rng.random((150, 300, 3)) * 255).astype(np.uint8)
    B = (rng.random((150, 300, 3)) * 255).astype(np.uint8)
    poly = route_seam(A, B, 150, halfw=60)
    assert poly.min() >= 90 and poly.max() <= 209


def test_uniform_content_returns_a_valid_polyline():
    A = np.full((100, 200, 3), 128, np.uint8)
    poly = route_seam(A, A.copy(), 100, halfw=50)
    # Any cut costs zero here; the pin is validity + bounds, not position.
    assert np.isfinite(poly).all()
    assert poly.min() >= 50 and poly.max() <= 149


def test_route_both_keys_both_seams():
    A = np.full((100, 2048, 3), 128, np.uint8)
    B = A.copy()
    out = route_both(A, B, seams=(512, 1536), halfw=64)
    assert set(out) == {512, 1536}
    assert all(len(v) == 100 for v in out.values())
