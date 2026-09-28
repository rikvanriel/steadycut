"""A pitch sweep must not REPORT a window that was never rendered.

A delegated 9-window sweep reported numbers for windows at 1350 s and 1500 s of
a recording that is only 1256.8 s long. Those renders produced an EMPTY file --
ffmpeg logged "Output file is empty, nothing was encoded" and left 262 bytes --
and the readings that came back were stale content from an earlier run, not
measurements. Two of the nine "windows" were fiction, and the numbers they
produced were confidently wrong.

The failure is silent because a table cell holding a plausible float is
indistinguishable from a measured one. The rule is therefore stated where it can
be enforced: a render smaller than MIN_PLAUSIBLE_BYTES never measured anything
and must be reported as missing, never interpolated over or averaged in.

The size threshold is reproduced here against the real encoder rather than
assumed, because the guard is only as good as the number it compares against.
"""
import subprocess
from pathlib import Path

import pytest

# ffmpeg writes the container even when it encodes no frames, so a window past
# the end of the footage leaves a small file rather than no file.  262 bytes was
# the observed size on a real 30-minute .insv; anything under a few KB is a
# container with no payload.
MIN_PLAUSIBLE_BYTES = 4096


def _encode(src: Path, dst: Path, ss: float, t: float) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{ss:g}", "-i", str(src),
         "-t", f"{t:g}", "-c:v", "libx264", "-preset", "ultrafast", str(dst)],
        check=True)
    return dst


def is_measurement(path: Path) -> bool:
    """Did this render carry frames, or is it a container with no payload?"""
    try:
        return path.stat().st_size >= MIN_PLAUSIBLE_BYTES
    except OSError:
        return False


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    src = tmp_path_factory.mktemp("src") / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc=size=320x240:rate=15:duration=10",
         "-c:v", "libx264", "-preset", "ultrafast", str(src)], check=True)
    return src


def test_a_window_past_the_end_encodes_nothing(clip: Path, tmp_path: Path) -> None:
    """The fact behind the fiction: 3 s requested 30 s into a 10 s clip.

    This is the exact shape of the bad sweep. If ffmpeg ever stops producing an
    empty file here, the guard has lost its reason to exist and should be
    revisited rather than kept as superstition.
    """
    out = _encode(clip, tmp_path / "past.mp4", ss=30.0, t=3.0)
    assert not is_measurement(out), (
        f"a window past the end produced {out.stat().st_size} bytes of real "
        f"content, so the empty-render guard is testing a condition that no "
        f"longer occurs")


def test_a_real_window_is_large_enough_to_measure(clip: Path, tmp_path: Path) -> None:
    """The control: an in-range window must clear the threshold.

    Without this, a threshold that rejected every render would satisfy the test
    above while silently discarding every genuine measurement -- the failure
    mode this whole module exists to prevent, one level up.
    """
    out = _encode(clip, tmp_path / "real.mp4", ss=1.0, t=3.0)
    assert is_measurement(out), (
        f"a valid window produced {out.stat().st_size} bytes, so "
        f"MIN_PLAUSIBLE_BYTES is set too high and would drop real readings too")


def test_a_missing_render_is_not_a_measurement(tmp_path: Path) -> None:
    """An absent file is missing, not zero."""
    assert not is_measurement(tmp_path / "never_written.mp4")
