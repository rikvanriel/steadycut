"""A generated equirectangular scene whose markers are at exact known lat/lon.

Markers, not a texture: each is a point whose position is *known*, so the test
needs arithmetic rather than correlation. A texture would need feature matching
and would fail hardest on exactly the frames under test -- the ones with the
largest correction, where the markers land near the frame edge and blur.

The trap this module exists to avoid: a frame stack is `(N, ROWS, COLS)`.
Repeating a 2-D image along axis 0 yields one tall strip, which ffmpeg accepts,
reports the right width and height for, and renders as a silently black smear --
so a test built on it measures nothing and passes.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

COLS, ROWS = 4096, 2048
MARKER_R = 6
LAT_RANGE = range(-60, 61, 20)
LON_RANGE = range(-160, 180, 20)


def marker_grid() -> list[tuple[float, float]]:
    """(lat, lon) pairs that map to DISTINCT, FULLY DRAWN pixels.

    The grid deliberately stops at -160/+160. Both -180 and +180 land on
    column 0, and `cv2.circle` clips rather than wraps, so a marker there is
    drawn as a half-disc whose centroid sits ~2 px inside the frame. Every
    seam-column marker is then off its known pixel by the same 1.98 px, which
    looks like a scene-encoder error and is not one. Keeping the grid off the
    seam removes the artefact instead of tuning a tolerance around it.
    """
    return [(float(la), float(lo))
            for la in LAT_RANGE
            for lo in LON_RANGE]


def latlon_to_pixel(lat: float, lon: float) -> tuple[int, int]:
    x = int((lon + 180.0) / 360.0 * COLS) % COLS
    y = int((90.0 - lat) / 180.0 * ROWS)
    return x, y


def build_frame() -> np.ndarray:
    img = np.zeros((ROWS, COLS), np.uint8)
    for la, lo in marker_grid():
        x, y = latlon_to_pixel(la, lo)
        cv2.circle(img, (x, y), MARKER_R, 255, -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def make_scene(path: str | Path, frames: int = 60, fps: int = 30) -> Path:
    """Write the equirect scene and return its path.

    The content is static -- the camera's motion is the subject, and a moving
    scene would be a second, uncontrolled variable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stack = np.repeat(build_frame()[None, :, :], frames, axis=0)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{COLS}x{ROWS}",
         "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p", str(path)],
        input=stack.tobytes(), check=True)
    return path


def read_frame(path: str | Path, index: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise RuntimeError(f"could not decode frame {index} of {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)