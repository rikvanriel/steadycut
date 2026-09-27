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


def test_image_only_reduces_jitter_without_destroying_the_ramp(tmp_path: Path) -> None:
    """The pass must remove shake WITHOUT flattening the intended motion.

    The earlier version of this test asserted the corrected bounce fell below
    half the raw, which was a property of sigma 8 rather than of the pass. Real
    trail footage then showed sigma 8 is actively harmful there: across nine
    windows on three rides the corrected-over-raw ratio was 1.37, i.e. it made
    the far field worse every time, because the dominant tracked motion on a
    trail is the rider's own machine and swaying foliage rather than camera
    shake. A synthetic 5 Hz jitter on a static scene is not that material, and
    tuning to it is how the wrong default got shipped.

    So the property worth keeping is the one that holds for both: the high
    frequency component goes down, and the low frequency ramp survives. A pass
    that flattens the ramp is smoothing away the ride.
    """
    src = tmp_path / "moving_noise.mp4"
    _write_moving_noise(src)
    out = tmp_path / "stabilised.mp4"
    res = stabilize_image_only(src, out, start=0.0, duration=N / FPS,
                               workdir=tmp_path / "work")
    assert res["output"] == out and out.exists()
    assert res["inliers_med"] > 20, f"tracking too weak: {res}"
    assert res["corrected_bounce"] < res["raw_bounce"], (
        f"jitter not reduced at all: raw {res['raw_bounce']:.2f} -> "
        f"corrected {res['corrected_bounce']:.2f}")

    # the intended motion must survive: the clip is a steady 0.5 px/frame ramp
    # under the jitter, so the delivered trajectory still has to climb
    from steadycut.core.pipeline import far_field_motion
    import numpy as np
    dy = far_field_motion(str(out))
    assert dy.mean() > 0.0, "the ramp was flattened, so the ride was smoothed away"
    assert abs(dy.mean()) > 0.15, (
        f"ramp nearly gone (mean {dy.mean():.3f} px/frame): the pass is "
        f"removing intended motion, not shake")


def test_image_only_refuses_a_missing_source(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        stabilize_image_only(tmp_path / "nope.mp4", tmp_path / "o.mp4",
                             0.0, 1.0)
