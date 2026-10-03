"""Tasks 1-3: the pieces, tested against arithmetic rather than against the pipeline.

These run in about a second and need no ffmpeg, so a failure here points at the
test's own maths rather than at a render. Task 4 is where the renderer enters.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from steadycut.testing import synthetic_check as chk
from steadycut.testing import synthetic_motion as mot
from steadycut.testing import synthetic_scene as scn


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    return scn.make_scene(tmp_path_factory.mktemp("s") / "sphere.mp4", frames=4)


# ---------------------------------------------------------------- Task 1
def test_scene_encodes_and_markers_survive(scene):
    """A scene whose markers did not survive measures nothing and passes."""
    frame = scn.read_frame(scene, 0)
    assert frame.shape == (scn.ROWS, scn.COLS)
    assert frame.max() > 200, "scene decoded black; check the frame-stack shape"


def test_detected_markers_match_their_latlon_pixels(scene):
    frame = scn.read_frame(scene, 0)
    found = chk.detect_markers(cv2_bgr(frame))
    assert len(found) >= len(scn.marker_grid()) - 4  # a few merge at the seam
    worst = 0.0
    for la, lo in scn.marker_grid():
        x, y = scn.latlon_to_pixel(la, lo)
        d = np.linalg.norm(found - np.array([x, y]), axis=1).min()
        worst = max(worst, d)
    assert worst <= 1.5, f"marker off its known pixel by {worst:.2f}"


def cv2_bgr(gray):
    import cv2
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------- Task 2
def test_gyro_integrates_back_to_the_attitude():
    """The property the whole test rests on.

    Round-tripped through the pipeline's OWN `integrate` rather than a
    hand-rolled integrator: an earlier version of this test integrated on the
    left (`step * acc`) while `integrate` composes on the right
    (`q = _mul(q, deltas[i])`), and the mismatch compounded into a 179.97 degree
    "drift" that was entirely the test's own convention. If the synthetic IMU
    cannot drive the production integrator back to the intended attitude, every
    later expectation is measured against the wrong reference.
    """
    from steadycut.stabilization import orientation as O

    t = np.arange(4000) / 1000.0
    q = mot.quats(t)
    gyro = mot.body_rates(q, 0.001)
    got = O.integrate(t, gyro)
    worst = 0.0
    for i in range(1, len(q)):
        d = O.to_rotvec(O._normalise(O._mul(O._conj(got[i]), q[i])))
        worst = max(worst, float(np.degrees(np.abs(d).max())))
    assert worst < 0.01, f"integrate round-trip drift {worst:.4f} deg"


def test_quaternions_are_unit():
    q = mot.quats(np.array([0.0, 0.5, 1.0]))
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0)
    assert not np.any(np.isnan(q)), "zero-norm quaternion at t=0"


# ---------------------------------------------------------------- Task 3
def test_projection_puts_the_view_axis_at_the_centre():
    px, py, ok = chk.project(np.array([[0.0, 0.0, 1.0]]), 120.0, 90.0, 960, 720)
    assert ok[0]
    assert px[0] == pytest.approx(480.0, abs=1e-9)
    assert py[0] == pytest.approx(360.0, abs=1e-9)


def test_projection_reaches_the_frame_edge_at_half_v_fov():
    hfov = 60.0
    d = np.array([[np.sin(np.radians(hfov)), 0.0, np.cos(np.radians(hfov))]])
    px, _, _ = chk.project(d, 2 * hfov, 90.0, 960, 720)
    assert px[0] == pytest.approx(960.0, abs=1e-6)


def test_points_behind_the_camera_are_marked_not_mirrored():
    px, _, ok = chk.project(np.array([[0.0, 0.0, -1.0]]), 120.0, 90.0, 960, 720)
    assert not ok[0]


def test_match_drops_predictions_at_the_frame_edge():
    observed = np.array([[480.0, 360.0]])
    pred = np.array([[3.0, 360.0], [480.0, 360.0]])
    d = chk.match(observed, pred, size=(960, 720))
    assert len(d) == 1, "a clipped marker must not enter the error distribution"