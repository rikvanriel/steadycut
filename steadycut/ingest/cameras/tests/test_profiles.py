"""Camera profile tests: a profile is model-wide facts, not measurements.

The axis map belongs to a camera BODY and lives in the user's registry, so the
model profile must not carry one -- an unmeasured body has to be measured, not
handed another camera's numbers.
"""
from steadycut.ingest.cameras.insta360_x4 import PROFILE as X4
from steadycut.ingest.cameras.profiles import MODEL_PROFILES, profile_for


def test_x4_profile_is_registered_without_a_measured_map() -> None:
    assert MODEL_PROFILES["Insta360 X4"] is X4
    # Measure-first: the map is per body and stored in the user's config, so
    # the model profile carries None and the registry attaches the measured
    # one. A non-None map here would be a silently shared X4 constant.
    assert X4.axis_map is None


def test_x4_geometry_carries_the_stitch_facts() -> None:
    lens = X4.lens
    assert lens.projection == "dfisheye" and lens.lenses == 2
    assert lens.stack_order == "rear_first"
    # Mean of the four makernotes estimates in the trailer.
    assert abs(lens.lens_fov_deg - 181.37) < 1e-9


def test_unknown_model_returns_none() -> None:
    assert profile_for("Camera That Does Not Exist") is None
    assert profile_for(None) is None


def test_x4_capabilities_and_framing_ceilings() -> None:
    assert "gyro_stabilize" in X4.capabilities
    assert X4.prestabilized_default is False
    assert X4.max_useful_hfov == {"4:3": 120.0, "9:16": 51.0}


def test_register_refuses_duplicates() -> None:
    import pytest
    from steadycut.ingest.cameras.profiles import CameraProfile, register
    with pytest.raises(ValueError):
        register(X4)  # already registered by the camera file's import
