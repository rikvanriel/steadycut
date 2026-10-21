"""Pins for the analytic sphere maps (Task 0.0): crossings, validity, poles.

Fail-old: the geq mask's defect is pinned as data (row 256 crossed at
293/1752) and must fail the gate; the new mask must pass it.
"""
import numpy as np

from steadycut.render.sphere_maps import blend_weight, lens_geometry


def _crossings(row):
    return np.where(np.diff(np.sign(row - 0.5)))[0].tolist()


def test_seam_crossings_within_2px_at_every_latitude():
    wgt, _ = blend_weight(2048, 1024, 4.0)
    for row in (128, 256, 384, 512, 640, 768, 896):
        got = _crossings(wgt[row])
        assert any(abs(c - 512) <= 2 for c in got), (row, got)
        assert any(abs(c - 1536) <= 2 for c in got), (row, got)


def _passes_gate(crossings):
    return (any(abs(c - 512) <= 2 for c in crossings)
            and any(abs(c - 1536) <= 2 for c in crossings))


def test_gate_rejects_the_old_geq_defect():
    """The retired mask crossed at 293/1752 on row 256 (~220 px off). A gate
    that passes that mask gates nothing."""
    assert _passes_gate([511, 1536])
    assert not _passes_gate([293, 1752])


def test_poles_get_a_switch_not_a_full_row_blend():
    wgt, _ = blend_weight(2048, 1024, 4.0)
    # Pole row: mostly decided pixels, narrow transition — never a full row
    # of 0.5 (that was rim-vs-rim smear across the whole row).
    trans = ((wgt[0] > 0.01) & (wgt[0] < 0.99)).sum()
    assert trans < 256, trans
    assert wgt[0].min() == 0.0 and wgt[0].max() == 1.0


def test_validity_never_blends_black():
    wgt, valid = blend_weight(2048, 1024, 4.0)
    g = lens_geometry(2048, 1024)
    # Where A is invalid the weight is 0; where B is invalid it is 1.
    assert (wgt[~g["valid_a"]] == 0.0).all()
    assert (wgt[~g["valid_b"]] == 1.0).all()
    assert valid.mean() > 0.99


def test_ramp_shape_seam_center_half_ownership():
    wgt, _ = blend_weight(2048, 1024, 4.0)
    r = wgt[512]
    assert abs(r[512] - 0.5) < 0.02
    assert r[0] == 1.0 and r[1024] == 0.0
    width = int(((r > 0.01) & (r < 0.99)).sum() // 2)
    assert 4 <= width <= 26, width  # 4deg ramp = 22.8px; wider overlap
    # realizes the full ramp (old geometry clipped it via incidence cap)


def test_routed_meridian_matches_blend_weight_bytes():
    from steadycut.render.sphere_maps import routed_weight
    w0, _ = blend_weight(2048, 1024, 4.0)
    w1, _ = routed_weight(2048, 1024, 4.0, polylines=None)
    assert np.array_equal((w0 * 255).astype(np.uint8),
                          (w1 * 255).astype(np.uint8))


def test_routed_deviation_recenters_within_coverage():
    from steadycut.render.sphere_maps import routed_weight
    poly = {512: np.full(1024, 516.0), 1536: np.full(1024, 1536.0)}
    wgt, _ = routed_weight(2048, 1024, 4.0, polylines=poly)
    assert abs(wgt[512, 516] - 0.5) < 0.02
    # outside any lens's coverage the blind lens is hard-snapped, not blended
    # (coverage edge at 558 with serving FOVs; was 552 at 182 deg)
    poly_far = {512: np.full(1024, 558.0), 1536: np.full(1024, 1536.0)}
    wgt2, _ = routed_weight(2048, 1024, 4.0, polylines=poly_far)
    assert wgt2[512, 558] == 0.0


def test_full_globe_ok_hard_bar_and_joint():
    from steadycut.render.sphere_maps import full_globe_ok
    # served pairs: union full (stabilization can sweep anywhere)
    assert full_globe_ok(185.0, 183.5) == (True, 8.5)
    assert full_globe_ok(182.2, 180.6)[0] is True
    # below 360: true holes — refused, never served
    assert full_globe_ok(170.0, 170.0) == (False, -20.0)
    # over-narrow fit trim: union full but joint 2 deg < 4 deg ramp — the
    # blend ramp eats past coverage (validity snapping hardens into cut +
    # smear, eye-REJECTED). Narrow joints cert-flag; holes refuse.
