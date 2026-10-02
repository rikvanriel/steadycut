"""Task 4: the calibration gate -- the point where the real renderer enters.

Everything else in this package is arithmetic. This test is the first that
decodes an actual ffmpeg/v360 render, and it exists to answer one question
before any other result is believed:

    do the projection conventions used by the expectation match the ones the
    renderer actually implements?

A static, unrotated view puts the marker nearest the view axis at the centre of
the frame. If the analytic prediction and the decoded pixels disagree, the
disagreement is a CONVENTION error -- in `dirs_from_latlon`, in the axis order,
in the FOV convention -- not a pipeline defect, and every number measured
downstream would be attributed to the wrong cause.

This gate has already earned its place once: it reported a 48.6 degree error
that turned out to be the test selecting the marker FURTHEST from the view axis
(`argmax` of the x-y norm) instead of the NEAREST (`argmax` of z). Without the
gate that would have been reported as a renderer defect.

The gate is 2.0 px and is frozen. It currently measures well under 1 px; the
margin is there so a sub-pixel regression is visible without the test becoming
brittle to H.264.
"""
import cv2
import numpy as np
import pytest

from steadycut.render.reframe import ControlPoint, render
from steadycut.testing import synthetic_check as chk
from steadycut.testing import synthetic_scene as scn
from steadycut.testing.synthetic_profile import FPS, H_FOV_DEG, OUT_H, OUT_W

FRAMES = 12


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    return scn.make_scene(tmp_path_factory.mktemp("cal") / "sphere.mp4",
                          frames=FRAMES, fps=FPS)


@pytest.fixture(scope="module")
def zero_render(scene, tmp_path_factory):
    out = tmp_path_factory.mktemp("calout") / "zero.mp4"
    path = [ControlPoint(t=i / FPS, pitch=0.0, yaw=0.0, roll=0.0,
                         fov=H_FOV_DEG) for i in range(FRAMES)]
    cmd = render(scene, out, path, size=(OUT_W, OUT_H),
                 input_projection="equirect", input_fov=360.0,
                 dual_stream=False, start=0.0, duration=FRAMES / FPS,
                 preset="ultrafast")
    assert out.exists(), f"no output written; ffmpeg said: {' '.join(cmd[-4:])}"
    return out


def test_render_produces_a_decodable_frame_with_markers(zero_render):
    frame = scn.read_frame(zero_render, 0)
    assert frame.shape[:2] == (OUT_H, OUT_W)
    found = chk.detect_markers(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    assert len(found) > 10, "render decoded with no markers; check the input stage"


def test_marker_nearest_the_view_axis_lands_at_the_frame_centre(zero_render):
    """The convention gate. Fails loudly on a convention error, by design."""
    grid = scn.marker_grid()
    dirs = chk.dirs_from_latlon([la for la, _ in grid], [lo for _, lo in grid])
    half_v = chk.half_v_fov(H_FOV_DEG, OUT_W, OUT_H)
    px, py, in_front = chk.project(dirs, H_FOV_DEG, half_v, OUT_W, OUT_H)

    # Nearest the view axis is the LARGEST z. argmax of the x-y norm selects
    # the furthest instead, which reads as a ~48 degree error and is not one.
    near = int(np.argmax(dirs[:, 2]))
    assert in_front[near]

    frame = scn.read_frame(zero_render, 0)
    found = chk.detect_markers(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    assert len(found), "no markers detected"

    err = np.hypot(found[:, 0] - px[near], found[:, 1] - py[near])
    offset = float(err.min())
    print(f"\n  calibration offset {offset:.2f} px "
          f"({offset * chk.deg_per_px(half_v, OUT_H):.4f} deg)")
    assert offset < chk.CAL_GATE_PX, (
        f"calibration offset {offset:.2f} px exceeds the "
        f"{chk.CAL_GATE_PX} px gate -- this is a CONVENTION error in the "
        f"expectation, not a pipeline defect")


@pytest.mark.xfail(
    reason="expectation model is a rectilinear pinhole; v360's perspective "
           "output is azimuthal (vf_v360.c:3283) and its coverage depends on "
           "v_fov, which render() does not pass. strict, so fixing the model "
           "fails the suite instead of leaving an expected failure nobody "
           "re-reads.",
    strict=True,
)
def test_surrounding_markers_also_agree(zero_render):
    """Not just the centre: every in-frame marker must land where predicted.

    A single centred marker can agree by luck of symmetry. The whole grid
    agreeing is what pins the axis convention and the FOV convention at once.

    KNOWN FAILING, and left failing on purpose. The expectation still uses a
    rectilinear pinhole model, while v360's perspective output is an azimuthal
    mapping (vf_v360.c:3283, `perspective_to_xyz`) whose coverage depends on
    `h = 1 + v_fov` -- and `render()` passes only `h_fov`, so the resulting
    field of view has not been established. 27 markers are actually visible in
    the render; the pinhole model puts 53-62 in frame. `strict=True` so that
    fixing the model turns this into an XPASS and fails the suite, rather than
    leaving a permanently-expected failure that nobody re-reads.
    """
    grid = scn.marker_grid()
    dirs = chk.dirs_from_latlon([la for la, _ in grid], [lo for _, lo in grid])
    half_v = chk.half_v_fov(H_FOV_DEG, OUT_W, OUT_H)
    px, py, in_front = chk.project(dirs, H_FOV_DEG, half_v, OUT_W, OUT_H)

    frame = scn.read_frame(zero_render, 0)
    found = chk.detect_markers(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    d = chk.match(found, np.column_stack([px, py]), size=(OUT_W, OUT_H))
    assert len(d) >= 10, f"only {len(d)} predictions were in frame"
    print(f"\n  {len(d)} markers matched, median error "
          f"{np.median(d):.2f} px, p95 {np.percentile(d, 95):.2f} px")
    assert np.percentile(d, 95) < chk.CAL_GATE_PX