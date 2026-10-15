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
