"""far_track: exactly one row per frame interval, always.

The row count is load-bearing. `residual_correction` accumulates the rows into
per-frame positions, and `warp_video` applies row j to frame j; a dropped row
shifts every later correction one frame late against its frame, which injects
motion instead of removing it. A frame whose features are too sparse to match
used to drop its row, so this test feeds the tracker frames that cannot be
matched at all (uniform grey, which ORB finds nothing in) and asserts the
count still comes out as frames - 1.
"""
import cv2
import numpy as np

from steadycut.stabilization import far_track

SIZE = (320, 240)


def _write_video(path, n_noise_a=8, n_blank=2, n_noise_b=8) -> int:
    """Noise, then featureless frames, then noise again; returns the count."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, SIZE)
    assert vw.isOpened(), "no mp4v writer available to build the fixture"
    rng = np.random.default_rng(0)
    total = 0
    for _ in range(n_noise_a):
        vw.write(rng.integers(0, 255, (SIZE[1], SIZE[0], 3), dtype=np.uint8))
        total += 1
    for _ in range(n_blank):
        vw.write(np.full((SIZE[1], SIZE[0], 3), 128, np.uint8))
        total += 1
    for _ in range(n_noise_b):
        vw.write(rng.integers(0, 255, (SIZE[1], SIZE[0], 3), dtype=np.uint8))
        total += 1
    vw.release()
    return total


def _frame_count(path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n


def test_a_frame_with_no_features_still_gets_a_row(tmp_path) -> None:
    path = tmp_path / "blank_frames.mp4"
    written = _write_video(path)
    n = _frame_count(path)
    assert n == written, "fixture must round-trip its frame count"

    deltas, inliers = far_track.track(str(path))

    assert len(deltas) == n - 1, (
        f"{len(deltas)} rows for {n} frames: a frame that cannot be matched "
        "must still emit a zero-motion row, or every later correction lands "
        "one frame late")
    assert (inliers == 0).sum() >= 2, (
        "the featureless frames should show up as zero-inlier rows")
