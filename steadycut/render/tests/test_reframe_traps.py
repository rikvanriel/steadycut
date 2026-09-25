"""Regression tests for pitch delivery at yaw 180 and the fisheye stack order.

The two faults met at yaw 180: v360 clears yaw, pitch and roll on every sendcmd
command unless reset_rot is set, so a pitch command there collapsed the render
back to the yaw 0 view and the pitch looked ignored, and stacking the two
fisheye tracks the wrong way round puts the direction of travel exactly at yaw
180. Both cost several rounds of work, so both are worth a test.
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from steadycut.stabilization import path as P
from steadycut.render.reframe import build_command_file, render
from steadycut.ingest.footage import footage_root

S = (f"{footage_root()}/"
     "VID_20260526_093530_00_010_012-Original/VID_20260526_093530_00_010.insv")


def _pixels(video):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video),
         "-vf", "scale=160:90,format=gray", "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw[:160 * 90], np.uint8).astype(np.int16).reshape(90, 160)


def _render(tmp, name, **angles):
    out = Path(tmp) / f"{name}.mp4"
    render(S, out, [P.PathPoint(t=0.0, fov=100.0, **angles)],
           start=400.0, duration=0.05, size=(320, 180), preset="ultrafast")
    return _pixels(out)


@pytest.mark.skipif(not Path(S).exists(), reason="source footage not present")
def test_pitch_is_honoured_at_yaw_180() -> None:
    """The pitch must matter at yaw 180 as well as at yaw 0.

    This test used to assert the OPPOSITE, documenting a "v360 degeneracy" that
    cost several rounds: at exactly +-180 a 60 degree pitch change altered the
    image not at all, and a yaw offset workaround was built around it.

    The cause was not a degeneracy in the projection. v360's process_command()
    zeroes yaw, pitch and roll on every sendcmd command unless reset_rot is set,
    so the pitch command was silently clearing the yaw=180 and the render fell
    back to the yaw 0 view - which made the pitch look ignored, since both
    renders then showed the same thing. render() now passes reset_rot=1.

    If this test fails again, reset_rot has been lost from the filtergraph.
    """
    with tempfile.TemporaryDirectory() as tmp:
        worked = np.abs(_render(tmp, "a", pitch=-60.0)
                        - _render(tmp, "b", pitch=0.0)).mean()
        honoured = np.abs(_render(tmp, "c", yaw=180.0, pitch=-60.0)
                          - _render(tmp, "d", yaw=180.0, pitch=0.0)).mean()
    assert worked > 5.0, f"pitch should matter at yaw 0, got {worked:.2f}"
    assert honoured > 1.0, (
        f"pitch should now be honoured at yaw 180, got {honoured:.2f}. "
        "Check that render() still passes reset_rot=1 to v360.")


@pytest.mark.skipif(not Path(S).exists(), reason="source footage not present")
def test_stream_order_puts_forward_at_yaw_zero() -> None:
    """The bike must be visible looking down at yaw 0, not at yaw 180.

    With the correct stack order, pitching down 40 degrees at yaw 0 brings the
    bicycle into frame: the bottom of the view fills with the rider's own helmet
    and the near ground, so the lower half of the frame differs strongly from the
    horizon view. With the wrong order that direction was the rear view.
    """
    with tempfile.TemporaryDirectory() as tmp:
        level = _render(tmp, "lvl", pitch=0.0)
        down = _render(tmp, "dwn", pitch=-40.0)
    # Compare CONTENT in the lower half, not mean brightness: a tilt replaces
    # what is visible there rather than simply brightening or darkening it.
    content = np.abs(down[45:, :] - level[45:, :]).mean()
    assert content > 8.0, (
        f"pitching down should change the lower half substantially, got {content:.2f}")
