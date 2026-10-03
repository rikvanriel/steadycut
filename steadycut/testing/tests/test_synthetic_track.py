"""Task 1: the tracker, tested against a path it did not build.

A synthetic scene rendered along an ANALYTIC sinusoid path (no pipeline, no
telemetry): every frame's rotation is known exactly, so each tracked feature's
displacement has an exact prediction -- back-project its pixel to a ray,
rotate the ray by the known inter-frame rotation, re-project. Any disagreement
is the tracker's, and there is nowhere else for it to hide.
"""
import numpy as np
import pytest

from steadycut.render.reframe import ControlPoint, render
from steadycut.testing import synthetic_check as chk
from steadycut.testing import synthetic_scene as scn
from steadycut.testing import synthetic_track as trk
from steadycut.testing import synthetic_view as view
from steadycut.testing.synthetic_profile import (
    FPS, H_FOV_DEG, OUT_H, OUT_W,
)

V_FOV = H_FOV_DEG * OUT_H / OUT_W
DURATION = 2.0
FRAMES = int(DURATION * FPS)


def analytic_path():
    out = []
    for k in range(FRAMES):
        t = k / FPS
        out.append(ControlPoint(
            t=t,
            pitch=float(20.0 * np.sin(2 * np.pi * 0.5 * t)),
            yaw=float(12.0 * np.sin(2 * np.pi * 0.3 * t)),
            roll=float(6.0 * np.sin(2 * np.pi * 0.4 * t)),
            fov=H_FOV_DEG))
    return out


@pytest.fixture(scope="module")
def render_two(tmp_path_factory):
    d = tmp_path_factory.mktemp("track")
    scene = scn.make_scene(d / "s.mp4", frames=FRAMES + 2, fps=FPS)
    pts = analytic_path()
    out = d / "p.mp4"
    render(scene, out, pts, size=(OUT_W, OUT_H), input_projection="equirect",
           input_fov=360.0, dual_stream=False, start=0.0, duration=DURATION,
           preset="ultrafast")
    frames = [scn.read_frame(out, k) for k in range(FRAMES)]
    return frames, pts


def test_tracked_displacement_matches_the_known_path(render_two):
    frames, pts = render_two
    pt = np.array([p.t for p in pts])
    pyr = np.array([(p.yaw, p.pitch, p.roll) for p in pts])
    errs, n = [], 0
    for k in range(0, FRAMES - 1, 10):
        tracks = trk.track(frames[k], frames[k + 1])
        if len(tracks) == 0:
            continue
        m0 = view.v360_matrix(*np.array(
            [np.interp(k / FPS, pt, pyr[:, i]) for i in range(3)]))
        m1 = view.v360_matrix(*np.array(
            [np.interp((k + 1) / FPS, pt, pyr[:, i]) for i in range(3)]))
        rel = m1 @ np.linalg.inv(m0)
        for (x, y, dx, dy) in tracks:
            ray = trk.backproject(x, y, H_FOV_DEG, V_FOV, OUT_W, OUT_H)
            pred = rel @ ray
            px, py, ok = chk.project(pred[None, :], H_FOV_DEG, V_FOV,
                                     OUT_W, OUT_H)
            if not ok[0]:
                continue
            errs.append(np.hypot(px[0] - (x + dx), py[0] - (y + dy)))
            n += 1
    assert n > 50, f"only {n} tracks to judge"
    print(f"\n  {n} tracks, median error {np.median(errs):.2f}px, "
          f"p95 {np.percentile(errs, 95):.2f}px")
    assert np.percentile(errs, 95) < 2.0
