"""End-to-end checks on the DELIVERED frame. Everything else in the suite is
synthetic, and that is exactly how three real failures got through.

The framing and stabilisation stacks are covered by unit tests on arrays: the
ground cue, the sky bound, the composition correction, the path optimiser, the
dark-mass edge, the far-field tracker. None of them renders. The one test file
that measures real pixels checks two projection traps and nothing else, and the
heaviest file in the suite is a trailer parser against a synthesised payload.

So three failures reached real footage unopposed:

  * the ground cue measures as a per-recording CONSTANT (a 0.016-wide boundary
    band over 922 frames of a 30 minute ride), which every synthetic test passes
    because synthetic frames do have a consistent boundary;
  * the path optimiser flattened a real path to a CONSTANT when its sky-share
    function reported 0.93 against a 0.25 bound. Every metric that counted
    improvement said success: frames over the bound fell from 51 to 8. Only the
    largest frame-to-frame step showed the gyro path had been thrown away;
  * the equirect extraction read 0.519 sky where the front view read 0.068 for
    the same ride, so the bound was fed a number that described nothing.

Each of the tests below fails on the state that let those through. They need the
footage and skip without it, matching test_reframe_traps.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from steadycut.core.pipeline import ClipSpec, build_gyro_path, render_path
from steadycut.framing.path_optimizer import Constraints, solve
from steadycut.framing.sky_bound import grid, share_fn, sky_mask
from steadycut.ingest.cameras import registry
from steadycut.ingest.footage import footage_root

SRC = Path(f"{footage_root()}/"
           "VID_20260519_074911_00_010_011-Original/VID_20260519_074911_00_010.insv")
# t=0..62s is the longest sky-heavy run in the recording (peak share 0.662), so it
# is where a framing correction has the most to do.
START, DUR = 0.0, 6.0
PITCH = -19.0
W, H = 320, 180

needs_footage = pytest.mark.skipif(not SRC.exists(),
                                   reason="source footage not present")


def _spec(dur=DUR, start=START, pitch=PITCH) -> ClipSpec:
    ident = registry.identify(str(SRC))
    return ClipSpec(source=str(SRC), start=start, duration=dur,
                    framing_pitch=pitch, profile=ident.get("profile"))


def _grey(video: Path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vf", "scale=160:90,"
         "format=gray", "-f", "rawvideo", "-"], capture_output=True,
        check=True).stdout
    n = len(raw) // (160 * 90)
    return np.frombuffer(raw, np.uint8)[: n * 160 * 90].reshape(n, 90, 160)


@needs_footage
def test_the_optimiser_preserves_the_gyro_path_on_real_footage(tmp_path):
    """The flattening guard, on real data with a real share function.

    The unit test proves it with a share function over the bound at every pitch.
    This proves the share function BUILT FROM FOOTAGE cannot drive the optimiser
    into that state, which is the case that actually happened: a bogus equirect
    read 0.93, the one-sided search ran every sample to the floor, and the render
    kept every metric that counts improvement while discarding the stabilisation.
    """
    spec = _spec()
    base_pts = build_gyro_path(spec, stamp_offset_ms=40.0)
    base = np.array([p.pitch for p in base_pts])
    times = np.array([p.t for p in base_pts])
    assert np.abs(np.diff(base)).max() > 0.05, "the base path must vary to test this"

    # a share function built from the front view, as the bound is meant to be
    fr = tmp_path / "front.png"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(START), "-i", str(SRC), "-t", "1",
         "-filter_complex", "[0:v:0]v360=input=equirect:id_fov=110:output=flat:"
         "w=640:h=480:yaw=0", "-frames:v", "1", str(fr)],
        capture_output=True, check=True)
    img = cv2.imread(str(fr))
    assert img is not None
    mask = sky_mask(img)
    share = share_fn(mask, *grid(img.shape[1], img.shape[0]), v_fov=90.0)

    out = solve(Constraints(base=base, times=times, sky_share=share, iters=200))
    assert np.abs(np.diff(out)).max() > 0.5 * np.abs(np.diff(base)).max(), (
        f"path collapsed: largest step {np.abs(np.diff(out)).max():.3f} vs "
        f"base {np.abs(np.diff(base)).max():.3f}")
    assert out.std() > 0.5 * base.std(), (
        f"path flattened: std {out.std():.3f} vs base {base.std():.3f}")


@needs_footage
def test_the_sky_measure_is_self_consistent(tmp_path):
    """A 110 degree front view cannot hold more sky than the whole sphere.

    The equirect extraction read 0.519 while the front view read 0.068 for the
    same footage, and every number derived from the first was meaningless. The
    two must at least order correctly, or one of them is not measuring sky.
    """
    out = tmp_path / "eq.png"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(START), "-i", str(SRC), "-t", "1",
         "-filter_complex", "[0:v:0]v360=input=equirect:id_fov=110:output=e:"
         f"w=640:h=320", "-frames:v", "1", str(out)],
        capture_output=True, check=True)
    eq = cv2.imread(str(out))
    assert eq is not None
    sphere = float(sky_mask(eq).mean())

    fr = tmp_path / "front.png"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(START), "-i", str(SRC), "-t", "1",
         "-filter_complex", "[0:v:0]v360=input=equirect:id_fov=110:output=flat:"
         "w=640:h=480:yaw=0", "-frames:v", "1", str(fr)],
        capture_output=True, check=True)
    front = cv2.imread(str(fr))
    assert front is not None
    f = float(sky_mask(front).mean())

    assert f <= sphere + 0.02, (
        f"front view {f:.3f} holds more sky than the whole sphere {sphere:.3f}: "
        f"one of the two projections is not measuring sky")
    assert sphere < 0.95, f"equirect is {sphere:.3f} sky, which is not forest"


@needs_footage
def test_a_framing_correction_changes_the_delivered_frame(tmp_path):
    """A correction that claims to have moved the path must move the pixels.

    Ties the optimiser to the render rather than to its own model: solve a path,
    render both it and the base, and require the frames to differ. A correction
    that reports movement but renders identically is not a correction, and
    nothing in the unit suite can see that.
    """
    spec = _spec()
    base_pts = build_gyro_path(spec, stamp_offset_ms=40.0)
    base = np.array([p.pitch for p in base_pts])
    times = np.array([p.t for p in base_pts])

    # a target the base path does not already meet, so the solve must move
    out = solve(Constraints(base=base, times=times, boundary=base.mean() * 0 + 0.62,
                            iters=100, comp_w=4.0))
    shift = np.abs(out - base).max()
    assert shift > 1.0, f"the solve barely moved: {shift:.3f} deg"

    from steadycut.render.reframe import ControlPoint
    moved = [ControlPoint(t=p.t, yaw=p.yaw, pitch=float(v), roll=p.roll,
                          fov=p.fov) for p, v in zip(base_pts, out)]
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    render_path(spec, base_pts, a, preset="ultrafast")
    render_path(spec, moved, b, preset="ultrafast")
    fa, fb = _grey(a), _grey(b)
    assert len(fa) and len(fb)
    diff = float(np.abs(fa[:min(len(fa), len(fb))].astype(int)
                        - fb[:min(len(fa), len(fb))].astype(int)).mean())
    assert diff > 1.0, (
        f"the path moved {shift:.1f} deg but the frames differ by {diff:.2f} "
        f"grey levels: the render is not following the path")
