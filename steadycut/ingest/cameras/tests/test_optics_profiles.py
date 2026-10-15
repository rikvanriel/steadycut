"""Pins for registry optics profiles (Task 1.1): schema, refusal, lifecycle."""
import pytest

from steadycut.ingest.cameras import registry as R
from steadycut.ingest.cameras import store


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    """Point the user's config and cache directories at a temp dir."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path


PROF = {"fov_a": 183.5, "fov_b": 183.5, "deflare_poly": [5.39, -7.2, 9.36,
        -0.85], "gain_default": [1.016, 1.029, 1.043], "label": "glass",
        "fw": "v1"}


def test_unregistered_body_has_no_optics(cfg) -> None:
    assert R.optics_for("NEVER-SEEN-0000") == []


def test_legacy_entry_without_optics_returns_empty(cfg) -> None:
    R.record_measured("S1", model="Insta360 X4", fw="v1",
                      axis_map={"yaw": [1, -1], "pitch": [2, -1],
                                "roll": [0, -1]})
    assert R.optics_for("S1") == []  # axis map path untouched by optics


def test_missing_and_model_keys_are_refused(cfg) -> None:
    assert R.optics_for("missing") == []
    assert R.optics_for("") == []
    assert R.optics_for("model:Insta360 X4") == []
    with pytest.raises(ValueError):
        R.record_optics("missing", PROF, serial_known=True)
    with pytest.raises(ValueError):
        R.record_optics("model:Insta360 X4", PROF, serial_known=True)
    with pytest.raises(ValueError):
        R.record_optics("S9", PROF, serial_known=False)


def test_record_select_confirm_lifecycle(cfg) -> None:
    R.record_measured("S2", model="Insta360 X4", fw="v1")
    entry = R.record_optics("S2", dict(PROF), serial_known=True)
    ids = [p["id"] for p in entry["optics"]]
    assert ids == ["S2-cand-1"]
    assert R.optics_for("S2")[-1]["status"] == "candidate"
    # near-duplicate FOVs dedup instead of appending.
    entry2 = R.record_optics("S2", dict(PROF, fov_a=183.6), serial_known=True)
    assert [p["id"] for p in entry2["optics"]] == ["S2-cand-1"]
    # distinct cover appends namespaced.
    entry3 = R.record_optics("S2", dict(PROF, fov_a=182.2, fov_b=180.6,
                                        label="bare"), serial_known=True)
    assert [p["id"] for p in entry3["optics"]] == ["S2-cand-1", "S2-cand-2"]
    R.confirm_optics("S2", "S2-cand-1", "clipA.insv")
    got = R.optics_for("S2")
    assert [p["status"] for p in got] == ["confirmed", "candidate"]
    assert got[0]["confirmed_by"] == ["clipA.insv"]


def test_firmware_mismatch_hides_optics(cfg) -> None:
    R.record_measured("S3", model="Insta360 X4", fw="v1")
    R.record_optics("S3", dict(PROF), serial_known=True)
    assert len(R.optics_for("S3", "v1")) == 1
    assert R.optics_for("S3", "v2") == []  # same rule as the axis map


def test_describe_reports_optics_ids(cfg) -> None:
    R.record_measured("S4", model="Insta360 X4", fw="v1",
                      axis_map={"yaw": [1, -1], "pitch": [2, -1],
                                "roll": [0, -1]})
    R.record_optics("S4", dict(PROF), serial_known=True)
    assert R.describe("S4")["optics"] == ["S4-cand-1"]
