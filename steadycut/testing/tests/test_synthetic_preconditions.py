"""Task 0: preconditions that must hold before any synthetic render is trusted.

Five asserts, no ffmpeg. Each one kills a specific misattribution -- a
mismatch here shows up later as rotation error that is not in the rotation.

    1. the axis map copy has not drifted from the registry
    2. start=0.0 lands on the generated clip's frame grid, so `render`'s
       start-snap (reframe.py:271-276) is a no-op
    3. `resolve_input` really resolves to equirect/360/single rather than
       falling back to the X4-era "dfisheye" default
    4. the synthetic accel passes the loader's gravity gate
    5. the synthetic rate passes the loader's 10x rate gate

Gates 4 and 5 are qualified: `read_telemetry` IS `load`, so the synthetic seam
replaces it and those gates never execute. They are asserted anyway, because the
thing they would catch -- a synthetic gyro that drifts -- is invisible in every
later number if it slips through.
"""
import subprocess

import numpy as np
import pytest

from steadycut.ingest.telemetry import GRAVITY_NOMINAL, GRAVITY_TOLERANCE
from steadycut.render.reframe import resolve_input, snap_to_frame
from steadycut.testing import synthetic_motion, synthetic_scene
from steadycut.testing.synthetic_profile import (
    AXIS, IMU_HZ, OUT_H, OUT_W, SYNTH_PROFILE, make_spec, registry_axis_map,
)


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    d = tmp_path_factory.mktemp("scene")
    return synthetic_scene.make_scene(d / "sphere.mp4", frames=6)


def test_axis_map_has_not_drifted_from_the_registry():
    """An explicit copy of the measured map rots silently; this makes it loud."""
    stored = registry_axis_map()
    if stored is None:
        pytest.skip("no X4 entry in the registry; nothing to compare against")
    assert AXIS == stored


def test_start_zero_lands_on_the_frame_grid(scene):
    """render() shifts every path point by the sub-frame phase of `start`.

    With a non-zero phase the expectation and the render disagree by that
    phase, which reads as rotation error. Assert the shift is absent rather
    than assuming it: the generated clip's grid is a property of how it was
    written, not of the intent.
    """
    assert snap_to_frame(0.0, scene) == 0.0


def test_resolve_input_is_equirect_not_the_x4_default():
    """A leaked fisheye default misattributes itself as a projection error."""
    proj, fov, dual, tyaw, order = resolve_input(
        None, "equirect", 360.0, False, None)
    assert proj == "equirect"
    assert float(fov) == 360.0
    assert dual is False
    assert order == "single"


def test_synthetic_accel_would_pass_the_gravity_gate():
    t, gyro, accel = synthetic_motion.synthetic_stream(2.0)
    median = float(np.median(np.linalg.norm(accel, axis=1)))
    assert abs(median - GRAVITY_NOMINAL) < GRAVITY_TOLERANCE


def test_synthetic_rate_would_pass_the_rate_gate():
    nominal = SYNTH_PROFILE.imu_rate_nominal_hz
    assert nominal
    t, _, _ = synthetic_motion.synthetic_stream(2.0)
    rate = 1.0 / float(np.median(np.diff(t)))
    assert nominal / 10.0 <= rate <= nominal * 10.0


def test_spec_carries_the_seven_fields_build_gyro_path_reads():
    """build_gyro_path reads seven off ClipSpec; this is the regression net."""
    spec = make_spec(scene, 2.0)
    for attr in ("source", "start", "duration", "framing_pitch", "fov",
                 "profile", "size"):
        assert getattr(spec, attr) is not None or attr == "profile"
    assert spec.profile.axis_map is AXIS
    assert spec.size == (OUT_W, OUT_H)