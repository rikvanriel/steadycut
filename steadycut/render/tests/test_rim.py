"""Pins for rim-blue correction: ring removed, center kept."""
import numpy as np


def _ringed(h=256, w=256, ring=40.0):
    rng = np.random.default_rng(4)
    base = np.full((h, w, 3), 170, np.uint8).astype(float)
    base += rng.normal(0, 2, base.shape)
    ys, xs = np.mgrid[0:h, 0:w]
    r = np.sqrt((xs - w / 2) ** 2 + (ys - h / 2) ** 2) / (w / 2)
    base[:, :, 0] += np.clip((r - 0.85) / 0.15, 0, 1) * ring
    return np.clip(base, 0, 255).astype(np.uint8), r


def test_rim_ring_removed_center_kept():
    from steadycut.render.rim import rim_blue_correct
    img, r = _ringed()
    out = rim_blue_correct(img).astype(float)
    exc = out[:, :, 0] - out[:, :, 2]
    assert abs(np.median(exc[(r > 0.5) & (r < 0.6)])) < 6.0
    assert np.median(exc[r > 0.95]) < 12.0


def test_flat_field_is_untouched():
    from steadycut.render.rim import rim_blue_correct
    rng = np.random.default_rng(5)
    flat = (np.full((128, 128, 3), 170.0) + rng.normal(0, 1.5, (128, 128, 3)))
    flat = np.clip(flat, 0, 255).astype(np.uint8)
    out = rim_blue_correct(flat).astype(float)
    assert np.abs(out - flat.astype(float)).mean() < 1.5
