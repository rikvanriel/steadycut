"""Pins for seam routing v2 (Tasks 3.1, 3.2b): avoid content on both sides,
clamp to joint-valid, never thread between doubled projections."""
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
    assert np.isfinite(poly).all()
    assert poly.min() >= 50 and poly.max() <= 149


def test_route_both_keys_both_seams():
    A = np.full((100, 2048, 3), 128, np.uint8)
    B = A.copy()
    out = route_both(A, B, seams=(512, 1536), halfw=64)
    assert set(out) == {512, 1536}
    assert all(len(v) == 100 for v in out.values())


def test_never_threads_between_doubled_projections():
    """Regression for the +37 px trunk ghost: A-bar at [35,45], B-bar at
    [72,82] (same trunk, parallactic pair). A seam between them (46..71)
    shows BOTH copies; the router must land outside that zone."""
    h, w = 160, 128
    A = np.full((h, w, 3), 128, np.uint8)
    B = np.full((h, w, 3), 128, np.uint8)
    A[:, 35:45] = 40
    B[:, 72:82] = 40
    poly = route_seam(A, B, 64, halfw=60, fov_a=300.0, fov_b=300.0)
    ghost_zone = (poly > 46) & (poly < 71)
    assert ghost_zone.mean() < 0.2, poly[::20]


def test_empty_validity_falls_back_to_meridian():
    A = np.full((80, 200, 3), 128, np.uint8)
    B = A.copy()
    poly = route_seam(A, B, 100, halfw=50, fov_b=10.0, fov_a=10.0)
    assert (poly == 100).all()
