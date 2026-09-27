"""roll_score must recover a KNOWN roll rate, and read zero on a still clip.

The certificate carried two far-field numbers (both translation of the
landscape) and no roll at all, so the axis a viewer notices first had no column.
This is the check that the new column measures what it says, on a synthetic clip
whose ground truth is genuinely known: every frame is the same textured image
rotated by a fixed number of degrees, so the per-frame in-plane rotation is
known exactly.

The aspect ratio is the discriminating part, and it is why the clip here is 4:3
rather than the metric's 16:9 default. A non-uniform rescale squeezes angles in
the ratio of the scale factors, so a 960x720 clip measured at 960x540 reads about
25 percent low -- a plausible number for the wrong quantity. The tolerance is
tight enough that the default-size behaviour FAILS it.
"""
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from steadycut.core.pipeline import roll_score

W, H, N, FPS = 640, 480, 24, 30.0     # 4:3, deliberately not the metric's 16:9
PER_FRAME_DEG = 0.40


def _textured(h: int, w: int) -> np.ndarray:
    """Corners, not noise: the estimator needs real features to fit a rotation."""
    rng = np.random.default_rng(11)
    img = np.full((h, w), 30, np.uint8)
    for _ in range(500):
        x, y = int(rng.integers(0, w - 40)), int(rng.integers(0, h - 40))
        bw, bh = (int(rng.integers(6, 40)), int(rng.integers(6, 40)))
        cv2.rectangle(img, (x, y), (x + bw, y + bh),
                      int(rng.integers(70, 255)), -1)
    return img


def _write_rolling_clip(path: Path, deg_per_frame: float) -> None:
    base = _textured(H, W)
    enc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}",
         "-framerate", f"{FPS:g}", "-i", "-",
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", str(path)],
        stdin=subprocess.PIPE)
    for i in range(N):
        m = cv2.getRotationMatrix2D((W / 2, H / 2), deg_per_frame * i, 1.0)
        frame = cv2.warpAffine(base, m, (W, H),
                               borderMode=cv2.BORDER_REPLICATE)
        enc.stdin.write(frame.tobytes())
    enc.stdin.close()
    assert enc.wait() == 0


def test_roll_score_recovers_a_known_rate_and_reads_zero_when_still(
        tmp_path: Path) -> None:
    still = tmp_path / "still.mp4"
    rolling = tmp_path / "rolling.mp4"
    _write_rolling_clip(still, 0.0)
    _write_rolling_clip(rolling, PER_FRAME_DEG)

    s = roll_score(still)
    r = roll_score(rolling)

    # Control: a clip that does not roll must read ~0, or the column is noise.
    assert s["mean_abs_deg_per_frame"] < 0.1, s
    # And the rolling clip must recover the commanded rate. Measured: 0.393
    # against a commanded 0.400 (1.75 percent) at the clip's own size, and
    # 0.343 (14 percent low) at the metric's 16:9 default. The tolerance is
    # 5 percent, which passes the first and FAILS the second -- the aspect
    # ratio is the whole point of the check, so the bound has to sit between
    # the two readings and not above both.
    got = r["mean_abs_deg_per_frame"]
    assert abs(got - PER_FRAME_DEG) < 0.05 * PER_FRAME_DEG, (
        f"roll rate {got:.3f} vs commanded {PER_FRAME_DEG} "
        f"(0.343 means the clip was measured at the wrong aspect)")


def test_roll_score_reports_nan_rather_than_guessing_on_no_clip(
        tmp_path: Path) -> None:
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"\x00" * 64)
    with pytest.raises(Exception):
        roll_score(empty)
