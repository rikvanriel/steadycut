"""Pins for corridor flow (Task 4.1): measured-range recovery, zero, snow."""
import numpy as np

from steadycut.render.corridor_flow import (FLOW_RANGE_PX, flow_confidence,
                                            flow_strip, warp_strip)


def _shifted_pair(dx=20.0, h=128, w=192):
    rng = np.random.default_rng(11)
    tex = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    B = np.zeros_like(tex)
    B[:, :w - int(dx)] = tex[:, int(dx):]  # B = A shifted left by dx
    return tex, B


def test_recovers_measured_range_shift():
    A, B = _shifted_pair(20.0)
    f = flow_strip(A, B)
    h, w = f.shape[:2]
    got = np.median(f[h // 4:3 * h // 4, 20:-20, 0])
    assert 15.0 <= abs(got) <= 25.0, got
    assert FLOW_RANGE_PX >= 33.0  # the Jan26 pin, not the retired 6 px


def test_identical_gives_zero_field():
    rng = np.random.default_rng(5)
    A = (rng.random((128, 192, 3)) * 255).astype(np.uint8)
    f = flow_strip(A, A.copy())
    assert np.median(np.abs(f)) < 1.0, np.median(np.abs(f))


def test_flat_snow_forces_low_confidence():
    snow = np.full((128, 192, 3), 180, np.uint8)
    snow = (snow.astype(float)
            + np.random.default_rng(9).normal(0, 1.5, snow.shape)).clip(
                0, 255).astype(np.uint8)
    f = flow_strip(snow, snow.copy())
    c = flow_confidence(snow, snow.copy(), f)
    assert np.median(c) < 0.35, np.median(c)


def test_warp_with_plus_flow_reduces_residual():
    A, B = _shifted_pair(20.0)
    f = flow_strip(A, B)
    base = np.abs(A.astype(float) - B.astype(float)).mean()
    W = warp_strip(B, f)  # +flow: measured on real corridors (30.8 -> 19.4);
    assert np.abs(A.astype(float) - W.astype(float)).mean() < base  # -flow worsens


def test_compose_zero_flow_is_bit_identical():
    rng = np.random.default_rng(21)
    eqB = (rng.random((256, 512, 3)) * 255).astype(np.uint8)
    h, w = 256, 128
    from steadycut.render.corridor_flow import compose_corrected_B
    out = compose_corrected_B(eqB, 256, 64,
                              np.zeros((h, w, 2), np.float32),
                              np.ones((h, w), np.float32))
    assert np.array_equal(out, eqB)


def test_compose_zero_confidence_falls_back_to_B():
    rng = np.random.default_rng(22)
    eqB = (rng.random((256, 512, 3)) * 255).astype(np.uint8)
    h, w = 256, 128
    from steadycut.render.corridor_flow import compose_corrected_B
    flow = (rng.random((h, w, 2)).astype(np.float32) - 0.5) * 60
    out = compose_corrected_B(eqB, 256, 64, flow,
                              np.zeros((h, w), np.float32))
    assert np.array_equal(out, eqB)  # violent-motion fallback: no warp at all


def _double_pair(dx=50, h=64, w=512):
    """Synthetic double: B carries the same bar dx px right of A's."""
    A = np.full((h, w, 3), 180, np.uint8)
    B = np.full((h, w, 3), 180, np.uint8)
    A[:, 200:220] = 60
    B[:, 200 + dx:220 + dx] = 60
    return A, B


def test_window_must_cover_destination():
    """Span field -> window -> compose -> merge; the double must collapse.

    Guards the destination-vs-source window bug: a source-centered window
    leaves the double standing (fails), a dest->src window collapses it.
    """
    from steadycut.render.corridor_flow import (compose_corrected_B,
                                                dest_src_window)
    dx = 50
    A, B = _double_pair(dx)
    h, w = A.shape[:2]
    flow = np.zeros((h, w, 2), np.float32)
    flow[:, :] = (float(dx), 0.0)
    dest, src = 210, 210 + dx

    def residual(conf):
        x0 = dest - 128
        Bc = compose_corrected_B(B, dest, 128, flow[:, x0:x0 + 256],
                                 conf[:, x0:x0 + 256])
        M = np.where(np.arange(w)[None, :, None] < dest - 30, A, Bc)
        strip = M[:, dest - 30:dest + 30].astype(float)
        ref = A[:, dest - 30:dest + 30].astype(float)
        return np.abs(strip - ref).mean()

    good = dest_src_window(w, dest, src, ramp=12).astype(np.float32)
    good = np.broadcast_to(good[None, :], (h, w)).copy()
    assert residual(good) < 15.0, residual(good)
    # the old shape: source-centered, destination outside -> double stands
    xs = np.arange(w, dtype=np.float32)
    bad = (0.5 - 0.5 * np.cos(
        np.clip(1 - np.abs(xs - src) / 35, 0, 1) * np.pi)).astype(np.float32)
    bad = np.broadcast_to(bad[None, :], (h, w)).copy()
    assert residual(bad) > 15.0, residual(bad)


def test_blend_scaled_flow_zero_outside_transition():
    from steadycut.render.corridor_flow import blend_scaled_flow
    flow = np.full((2, 4, 2), 40.0, np.float32)
    alpha = np.array([[0.0, 0.25, 0.75, 1.0]] * 2, np.float32)
    out = blend_scaled_flow(flow, alpha)
    assert np.all(out[:, 0] == 0) and np.all(out[:, 3] == 0)
    assert np.allclose(out[:, 1, 0], 30.0)
    assert np.allclose(out[:, 2, 0], 10.0)


def test_fuse_left_right_flow_is_sign_consistent():
    from steadycut.render.corridor_flow import fuse_left_right_flow
    left = np.array([[[10.0, -10.0]]], np.float32)
    right = np.array([[[-30.0, 30.0]]], np.float32)
    out = fuse_left_right_flow(left, right)
    assert out[0, 0, 0] == 30.0
    assert out[0, 0, 1] == -30.0


def test_temporal_median_flow_rejects_outlier():
    from steadycut.render.corridor_flow import temporal_median_flow
    flows = [np.full((2, 2, 2), v, np.float32) for v in (4.0, 5.0, 100.0)]
    assert np.all(temporal_median_flow(flows) == 5.0)


def test_compose_alpha_scales_correction():
    # A-side alpha: full correction mid-blend, none outside it.
    from steadycut.render.corridor_flow import compose_corrected_B
    H, Wd = 64, 512
    B = np.zeros((H, Wd, 3), np.uint8)
    B[:, 262:] = 200
    flow = np.zeros((H, 128, 2), np.float32)
    flow[:, :] = (40.0, 0.0)
    conf = np.ones((H, 128), np.float32)
    alpha = np.zeros((H, 128), np.float32)
    alpha[:, 48:80] = np.linspace(0.05, 0.95, 32, dtype=np.float32)
    out = compose_corrected_B(B, 256, 64, flow, conf, alpha=alpha)
    edge = out[:, 256 - 64:256 - 48].astype(float)
    mid = out[:, 256 - 8:256 + 8].astype(float)
    assert np.abs(edge - B[:, 256 - 64:256 - 48].astype(float)).mean() < 1.0
    assert np.abs(mid.mean() - 200) < 60  # B content pulled in
