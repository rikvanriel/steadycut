import cv2
import numpy as np

from steadycut.metrics.level import roll_of_frame


def _frame_with_verticals():
    img = np.full((240, 320, 3), 128, np.uint8)
    rng = np.random.default_rng(7)
    for x in range(20, 320, 24):
        cv2.line(img, (x, 10), (x + int(rng.integers(-3, 4)), 230), (30, 30, 30), 3)
    return img


def _rotated(img, deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def test_roll_recovers_applied_rotation():
    base = _frame_with_verticals()
    for deg in (-15, -5, 5, 15):
        est, n = roll_of_frame(_rotated(base, deg))
        assert n >= 3
        assert abs(est - deg) < 2.0, (deg, est)


def test_roll_silence_when_no_verticals():
    blank = np.full((240, 320, 3), 128, np.uint8)
    est, n = roll_of_frame(blank)
    assert n == 0
    assert np.isnan(est)
