"""The image-only (L3) stabiliser on a synthetic clip with KNOWN motion.

Frames carry a slow ramp plus a fast 5 Hz vertical jitter. The bounded 2D
pass keeps the slow part and must remove the fast one, so the corrected
far-bounce must beat the raw one -- the same acceptance the second pass of
the gyro pipeline already meets. This is the component a no-IMU camera
would rely on as its ONLY stabilisation.
"""
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from steadycut.core.pipeline import stabilize_image_only

W, H, N, FPS = 1280, 960, 36, 30.0


def _write_moving_noise(path: Path) -> None:
    rng = np.random.default_rng(7)
    base = rng.integers(0, 255, (H, W), dtype=np.uint8)
    enc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}",
         "-framerate", f"{FPS:g}", "-i", "-",
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", str(path)],
        stdin=subprocess.PIPE)
    for i in range(N):
        dy = 0.5 * i + 6.0 * math.sin(2 * math.pi * 5.0 * i / FPS)
        m = np.float32([[1, 0, 0], [0, 1, dy]])
        frame = cv2.warpAffine(base, m, (W, H), borderMode=cv2.BORDER_REPLICATE)
        enc.stdin.write(frame.tobytes())
    enc.stdin.close()
    assert enc.wait() == 0


def test_image_only_removes_the_fast_jitter(tmp_path: Path) -> None:
    src = tmp_path / "moving_noise.mp4"
    _write_moving_noise(src)
    out = tmp_path / "stabilised.mp4"
    res = stabilize_image_only(src, out, start=0.0, duration=N / FPS,
                               workdir=tmp_path / "work")
    assert res["output"] == out and out.exists()
    assert res["inliers_med"] > 20, f"tracking too weak: {res}"
    assert res["corrected_bounce"] < 0.5 * res["raw_bounce"], (
        f"jitter not removed: raw {res['raw_bounce']:.2f} -> "
        f"corrected {res['corrected_bounce']:.2f}")


def test_image_only_refuses_a_missing_source(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        stabilize_image_only(tmp_path / "nope.mp4", tmp_path / "o.mp4",
                             0.0, 1.0)
