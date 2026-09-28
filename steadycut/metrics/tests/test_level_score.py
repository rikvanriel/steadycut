"""level_score must never publish a bank it cannot defend, and must fail loudly.

The column exists because the horizon bank had no column at all, and six
measurement designs were built and withdrawn before it.  The test that matters
here is therefore not "does it read a number" but "does it REFUSE to": a level
column that under-reports is worse than no column, because it puts a small
plausible number where a large fault is, and later changes get judged by it.

Two of these tests need no footage at all, and that is deliberate. The refusal
is a CONTRACT, and a contract that could only be checked on real footage would
be untestable on the machines where it matters most. The one test that does need
pixels builds its own clip, and it is there to guard the finding the whole
analysis rests on.
"""
import subprocess
from pathlib import Path

import cv2
import numpy as np

from steadycut.metrics.level import (GAIN_IS_STABLE_ACROSS_FRAMING, TRUSTED,
                                     level_score)

W, H, N, FPS = 320, 240, 10, 30.0


def _clip(path: Path, deg_per_frame: float = 0.0) -> Path:
    """A short 4:3 clip: same texture every frame, rotated by a known step."""
    rng = np.random.default_rng(3)
    img = np.full((H, W), 30, np.uint8)
    for _ in range(300):
        x, y = int(rng.integers(0, W - 30)), int(rng.integers(0, H - 30))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(6, 30)),
                                    y + int(rng.integers(6, 30))),
                      int(rng.integers(70, 255)), -1)
    enc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}",
         "-framerate", f"{FPS:g}", "-i", "-",
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", str(path)],
        stdin=subprocess.PIPE)
    for i in range(N):
        m = cv2.getRotationMatrix2D((W / 2, H / 2), deg_per_frame * i, 1.0)
        enc.stdin.write(cv2.warpAffine(img, m, (W, H),
                                       borderMode=cv2.BORDER_REPLICATE).tobytes())
    enc.stdin.close()
    assert enc.wait() == 0
    return path


def test_level_score_declines_to_publish_a_bank() -> None:
    """The contract: say 'not measured' rather than emit a small number.

    If a future change makes the bank measurable this is the test to update
    deliberately -- never by relaxing the assertion to let a new number through.
    """
    from steadycut.metrics.level import TRUST_NOTE

    assert "NOT measured" in TRUST_NOTE
    assert TRUSTED is False


def test_noise_floor_exceeds_a_slow_bank() -> None:
    """A future level metric has to clear a number, so the number is published.

    This is the finding that retired every accumulator design: a bank spread
    over a clip is a small per-second quantity, and the accumulated sum's noise
    floor is larger than it.
    """
    from steadycut.metrics.level import _ROLL_NOISE_DEG_PER_S

    assert _ROLL_NOISE_DEG_PER_S > 0.0
    # 20 deg over 16 s is the fastest bank the plan cared about, and it is
    # already below the floor.
    assert 20.0 / 16.0 < _ROLL_NOISE_DEG_PER_S


def test_gain_is_recorded_as_unstable_across_framing() -> None:
    """The measured reason the far field cannot stand in for a bank.

    Gain +1.781 at pitch -12 and -2.777 at pitch -30 on one path: a magnitude
    change AND a sign flip, so no single calibration constant exists.
    """
    assert GAIN_IS_STABLE_ACROSS_FRAMING is False
    assert TRUSTED is False


def test_rotation_between_is_still_exact_on_a_known_rotation() -> None:
    """Guard the instrument the analysis rests on.

    The finding that retired every accumulator was that the ESTIMATOR is exact
    (a known rotation comes back to ~0 deg) and the ACCUMULATION is what fails.
    If this ever fails, the root cause recorded in level.py is wrong and the
    analysis needs redoing rather than re-tuning.
    """
    from steadycut.metrics.rotate import rotation_between

    p = _clip(Path("/tmp/level_known.mp4"), 0.0)
    cap = cv2.VideoCapture(str(p))
    ok, f = cap.read()
    cap.release()
    assert ok
    m = cv2.getRotationMatrix2D((W / 2, H / 2), 8.0, 1.0)
    rot = cv2.warpAffine(f, m, (W, H), borderMode=cv2.BORDER_REPLICATE)
    got = rotation_between(f.astype(np.float32), rot.astype(np.float32))
    assert got is not None
    # reads -8.0 by this project's sign convention; the magnitude is the point
    assert abs(abs(got) - 8.0) < 0.5


def test_level_score_on_a_real_clip_still_refuses_the_bank(tmp_path: Path) -> None:
    """With pixels, the far-field numbers are real and the bank is still None.

    The two halves of that sentence are the module's whole position: report what
    is measured, refuse what is not.
    """
    p = _clip(tmp_path / "roll.mp4", 0.4)
    d = level_score(p)
    assert d["bank_deg"] is None
    assert d["bank_measured"] is False
    assert d["frames"] > 0
    assert d["far_rms_deg_per_frame"] >= 0.0
    assert d["noise_floor_deg_per_s"] > 0.0
