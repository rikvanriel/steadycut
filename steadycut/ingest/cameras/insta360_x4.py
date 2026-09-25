"""Insta360 X4 profile: model-wide facts only. Every value is measured.

The gyro axis map is deliberately NOT here. It belongs to one camera BODY, and
the one measured on our X4 (yaw=column 1, pitch=column 2, roll=column 0, each
winner unambiguous at |r| ~ 0.83; signs negative because the rear-first stack
order rotates the stitched sphere 180 degrees against the camera axes) is
stored in the user's registry instead, keyed by serial number. On first use of
any body -- ours included, on a fresh install -- the map is measured from that
body's own footage and written to ~/.config/steadycut/cameras.json.

Lens: the makernotes protobuf carries per-lens half-FOVs 91.077/90.291 and
91.065/90.300 degrees, i.e. full per-lens FOVs of 182.15/180.58; the mean of
the four estimates, 181.37, is what the renderer uses. The two lenses
differ by 1.6 degrees and v360 takes one id_fov, so a residual seam is
expected at the sides.

Framing ceilings: 4:3 at h_fov 120 and 9:16 at h_fov 51 are the settled
deliverables; 16:9 was rejected because holding that vertical coverage needs
h_fov 149, whose edge smearing is visible to the eye.

Sync offset is per FILE (-80 ms on one file, -20 ms on another) and is measured
per render, never stored as a camera constant.
"""
from steadycut.ingest.cameras.profiles import (
    CameraProfile, LensGeometry, register,
)

PROFILE = register(CameraProfile(
    model="Insta360 X4",
    telemetry_format="insv_trailer",
    axis_map=None,                 # measured per body; see the module docstring
    lens=LensGeometry(
        projection="dfisheye",
        lenses=2,
        lens_fov_deg=181.37,      # mean of the four makernotes estimates
        stack_order="rear_first",  # [0:v:1][0:v:0]; wrong order = sphere at 180
        travel_yaw=0.0,            # normal mount: travel already sits at yaw 0
    ),
    imu_rate_nominal_hz=1000.0,    # measured 999.1-999.3 Hz on both bodies
    max_useful_hfov={"4:3": 120.0, "9:16": 51.0},
    capabilities=frozenset({"reframe", "gyro_stabilize"}),
    prestabilized_default=False,   # 360 modes never bake stabilisation in
    notes="Gravity medians on real files: 9.65 and 10.22 m/s^2 (pre-roll head "
          "load inflates |a| slightly; median is robust).",
))
