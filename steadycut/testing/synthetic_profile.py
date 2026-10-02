"""Synthetic camera constants for the end-to-end reframe test.

`build_gyro_path` takes a `ClipSpec`, not a source path plus telemetry, so the
synthetic driver supplies a real `ClipSpec` and replaces only the telemetry
loader. That is the right seam: telemetry is the one thing that cannot be
synthesised by pointing the pipeline at a real file, whereas `spec.start`,
`spec.duration`, `spec.fov` and `spec.framing_pitch` are ordinary fields that
happen to be read on the same code path.

See steadycut/testing/synthetic_scene.py for the video and
synthetic_motion.py for the IMU that this profile has to stay consistent with.
"""
from __future__ import annotations

import dataclasses

from steadycut.core.pipeline import ClipSpec
from steadycut.ingest.cameras.insta360_x4 import PROFILE as X4
from steadycut.ingest.cameras.profiles import AxisMap

# The measured X4 axis map: yaw on raw column 1, pitch on 2, roll on 0, every
# sign -1. Constructed explicitly because the map is attached per body by the
# registry, so a bare PROFILE has `axis_map=None` and every lookup dies on
# `NoneType.pitch`. tests/test_synthetic_preconditions.py asserts this against
# the registry so the copy cannot rot into a stale convention.
AXIS = AxisMap(yaw=(1, -1), pitch=(2, -1), roll=(0, -1))

# The synthetic IMU runs at the X4's nominal 1000 Hz, which the loader's 10x
# rate gate expects. The gate does not run on this path -- read_telemetry is
# `load`, so replacing it skips every gate in `load` -- but keeping the field
# correct means the telemetry stays valid if the seam ever stops bypassing it.
SYNTH_PROFILE = dataclasses.replace(
    X4,
    axis_map=AXIS,
    imu_rate_nominal_hz=1000.0,
)

H_FOV_DEG = 120.0
OUT_W, OUT_H = 960, 720
IMU_HZ = 1000.0
FPS = 30


def make_spec(scene_path, duration: float) -> ClipSpec:
    """A ClipSpec over a synthetic scene.

    `framing_pitch` is 0.0 rather than the production -12 default: the scene
    markers are symmetric about the equator, so a framing offset would bias
    every expected pixel position for no reason under test.
    """
    return ClipSpec(
        source=scene_path,
        start=0.0,
        duration=duration,
        framing_pitch=0.0,
        fov=H_FOV_DEG,
        size=(OUT_W, OUT_H),
        profile=SYNTH_PROFILE,
    )


def registry_axis_map():
    """The X4's stored axis map from the user's registry, or None.

    Read from the real store rather than a fixture so that re-measuring a body
    shows up as a test failure. Returns None when the registry has no X4 entry,
    which is a legitimate state -- the comparison is then skipped rather than
    pinned to whatever this checkout happens to have.
    """
    try:
        from steadycut.ingest.cameras import registry
        return registry.axis_map_for("Insta360 X4")
    except Exception:                      # registry absent or unreadable
        return None