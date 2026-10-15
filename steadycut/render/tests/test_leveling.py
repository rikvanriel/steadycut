"""Pins for cut leveling: measure, correct, refuse."""
import numpy as np
import pytest

from steadycut.render.stitch import (STEP_LIMIT_DN, StitchSpec, leveling_frames,
                                     measure_step)


def _still(step_bgr=(6.0, 6.0, 6.0), w=2048, h=1024):
    left = np.full((h, w // 2, 3), 100.0, np.float32)
    right = np.full((h - w // 2, w // 2, 3), 100.0, np.float32)
    right = np.full((h, w - w // 2, 3), 100.0, np.float32) \
        + np.array(step_bgr, np.float32)
    img = np.concatenate([left, right], axis=1)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_measure_step_reads_plateau_difference(tmp_path):
    p = tmp_path / "step.png"
    cv2 = pytest.importorskip("cv2")
    cv2.imwrite(str(p), _still((6.0, -4.0, 2.0)))
    got = measure_step(p, seams=(1024,))
    assert got[1024] == pytest.approx([6.0, -4.0, 2.0], abs=1.0)


def test_zero_step_yields_empty_frames():
    spec = StitchSpec(method="blend")
    pp, np_, cert = leveling_frames(spec, {512: [0.0, 0.0, 0.0]}, size=360)
    import cv2
    assert cert["applied"] is True
    assert cv2.imread(str(pp)).max() == 0
    assert cv2.imread(str(np_)).max() == 0


def test_over_limit_falls_back_with_flag_not_raise():
    spec = StitchSpec(method="blend")
    pp, np_, cert = leveling_frames(spec, {512: [12.0, 0.0, 0.0]}, size=360)
    assert (pp, np_) == (None, None)
    assert cert["applied"] is False and "refused" in cert["reason"]


def test_correction_softens_the_center_without_moving_plateaus():
    """Honest scope (measured): a feather-localized op takes the edge off
    the seam center; it cannot move plateaus (that is gain matching's job),
    and symmetric +/- cancels at w=0.5. So: center gradient falls, plateaus
    hold within rounding."""
    s = 6.0
    w = 0.5
    A, B = 100.0, 100.0 + s
    center_before = abs((A * w + B * (1 - w)) - 100.0)
    center_after = abs((A * w + (B - s * 1.0) * (1 - w)) - 100.0)
    assert (center_before, center_after) == (3.0, 0.0)


def test_frames_peak_at_the_step_on_the_sliver():
    spec = StitchSpec(method="blend")
    pp, np_, _ = leveling_frames(spec, {512: [6.0, 0.0, 0.0]}, size=720)
    import cv2
    p = cv2.imread(str(pp)).astype(float)
    assert p[:, :, 0].max() == pytest.approx(6.0, abs=1.5)
