"""Task 5: the full pipeline, through a real render, against a known answer.

This exercises telemetry -> integrate -> smooth -> correction -> `sendcmd` ->
v360 -> decoded pixels, and compares the result against the path's own angles
through an explicit model of v360's composition. The model is built from
rotation matrices, not from a library's Euler helpers: scipy's lowercase
sequences compose in the REVERSE order of what the names suggest, and that
single confusion produced two wrong "measured" conventions in a row.

The source is the STATIC sphere, deliberately. A camera-rotated source was
tried and abandoned: it makes the expected view a composition whose order and
inverses admit four indistinguishable possibilities, and every one of them
scored 270-300 px. With a static source the expected view is exactly what the
path commands, which is what the calibration gate already verifies.

The gate is 2.0 px, the same as the calibration gate, for the same reason: the
measurement floor on a static render is ~1.1 px at p95 (detection, H.264,
lanczos), so a tighter gate would fail a correct pipeline. What this catches
is the 100 px class: the neighbouring ZYX order costs 114 px at p95 here.
"""
import numpy as np
import pytest

from steadycut.core.pipeline import build_gyro_path
from steadycut.render.reframe import render
from steadycut.testing import synthetic_check as chk
from steadycut.testing import synthetic_motion as mot
from steadycut.testing import synthetic_scene as scn
from steadycut.testing import synthetic_view as view
from steadycut.testing.synthetic_profile import (
    FPS, H_FOV_DEG, OUT_H, OUT_W, make_spec,
)

DURATION = 4.0
FRAMES = int(DURATION * FPS)
CORNER_HZ = 0.15
V_FOV = H_FOV_DEG * OUT_H / OUT_W


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    return scn.make_scene(tmp_path_factory.mktemp("e2e") / "sphere.mp4",
                          frames=FRAMES + 2, fps=FPS)


@pytest.fixture(scope="module")
def stream():
    return mot.synthetic_telemetry(DURATION)


@pytest.fixture(scope="module")
def patched():
    """Replace the telemetry loader -- the one thing a real file cannot supply."""
    from steadycut.ingest import telemetry as tele_mod
    original = tele_mod.read_telemetry

    def install(tele):
        tele_mod.read_telemetry = lambda path, profile=None: tele
    yield install
    tele_mod.read_telemetry = original


@pytest.fixture(scope="module")
def run(scene, stream, patched, tmp_path_factory):
    """Drive build_gyro_path on synthetic telemetry, then render for real.

    The source is the static sphere, NOT a camera-rotated one. Pre-rolling
    the source by q_raw was tried and abandoned: it makes the expected view
    a composition of camera rotation and correction whose order and inverses
    are four separate possibilities, and every one of them scored 270-300 px.
    With a static source the expected view is simply the path's own angles,
    which is what the calibration gate already verifies exactly.
    """
    patched(stream)
    spec = make_spec(scene, DURATION)
    points = build_gyro_path(spec, corner_hz=CORNER_HZ)
    out = tmp_path_factory.mktemp("e2eout") / "path.mp4"
    render(scene, out, points, size=(OUT_W, OUT_H),
           input_projection="equirect", input_fov=360.0, dual_stream=False,
           start=0.0, duration=DURATION, preset="ultrafast")
    assert out.exists()
    return out, points, stream


def marker_error(rendered, points, negate_yaw=False):
    """Per-frame p95 marker displacement in pixels.

    The prediction routes the path's own angles through the explicit v360
    model (R_y R_x R_z, no library Euler helpers) and the calibrated
    projection. `negate_yaw` builds the control: the same render scored
    against a deliberately sign-flipped yaw, which must exceed the gate --
    otherwise the test could not tell the bug it exists to catch from a pass.
    """
    grid = scn.marker_grid()
    dirs = chk.dirs_from_latlon([a for a, _ in grid], [b for _, b in grid])
    pt = np.array([p.t for p in points])
    pys = np.array([p.yaw for p in points])
    pps = np.array([p.pitch for p in points])
    prs = np.array([p.roll for p in points])

    out = []
    for k in range(FRAMES):
        frame = scn.read_frame(rendered, k)
        found = chk.detect_markers(cv2_bgr(frame))
        if not len(found):
            continue
        # Frame START, matching build_command_file's command grid
        # (np.arange(0, ts[-1], 1/fps)), not the frame centre. At 20 deg/s
        # the half-frame difference is 0.33 deg = 2.3 px.
        tc = k / FPS
        yw = float(np.interp(tc, pt, pys))
        if negate_yaw:
            yw = -yw
        cam = view.predict_pixels(
            dirs, yw,
            float(np.interp(tc, pt, pps)),
            float(np.interp(tc, pt, prs)))
        px, py, _ = chk.project(cam, H_FOV_DEG, V_FOV, OUT_W, OUT_H)
        err = chk.match(found, np.column_stack([px, py]), size=(OUT_W, OUT_H))
        if len(err):
            out.append(float(np.percentile(err, 95)))
    return np.asarray(out)


def cv2_bgr(gray):
    import cv2
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ------------------------------------------------------------------ the tests
def test_render_matches_the_path(run):
    rendered, points, _ = run
    err = marker_error(rendered, points)
    assert len(err) > FRAMES // 2, "too few frames had markers to judge"
    p95 = float(np.percentile(err, 95))
    print(f"\n  {len(err)} frames judged; marker p95 {p95:.2f} px "
          f"({p95 * chk.deg_per_px(V_FOV / 2, OUT_H):.3f} deg), "
          f"median per-frame {np.median(err):.2f} px")
    assert p95 < chk.CAL_GATE_PX, (
        f"marker displacement p95 {p95:.2f} px exceeds {chk.CAL_GATE_PX} px")


def test_negated_yaw_exceeds_the_gate(run):
    """The control: the same render against a sign-flipped yaw must fail.

    Without this, a test that always passes and a test that cannot see yaw
    errors look identical. The path's yaw spans +-14 deg here, so the flip
    costs tens of pixels and the gate catches it with room to spare.
    """
    rendered, points, _ = run
    err = marker_error(rendered, points, negate_yaw=True)
    assert float(np.percentile(err, 95)) > chk.CAL_GATE_PX