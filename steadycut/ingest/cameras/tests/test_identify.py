"""identify(): trailer first, then sniffing, then a hint.

A body whose axis map has been measured resolves to "ok" with the map attached;
a body with no measurement resolves to "needs_measure", which the CLI turns
into an automatic measurement on the user's own footage. Refusals stay for the
cases no measurement can fix (an unsupported model, an unsupported container).
The trailer branch is also exercised against a real .insv when footage is
present (STEADYCUT_FOOTAGE-gated, like the renderer tests).
"""
from pathlib import Path

import pytest

from steadycut.ingest.cameras import registry as R
from steadycut.ingest.footage import footage_root

S = (f"{footage_root()}/"
     "VID_20260115_122221_00_003_004-Original/VID_20260115_122221_00_003.insv")

X4_AXIS = {"yaw": [1, -1], "pitch": [2, -1], "roll": [0, -1]}


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    """Keep every measurement in a temp dir, never in the user's config."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path


def test_hint_needs_a_measurement_until_one_is_stored(cfg) -> None:
    r = R.identify("whatever.bin", hint="Insta360 X4")
    assert r["status"] == "needs_measure" and r["via"] == "hint"
    assert r["profile"].model == "Insta360 X4"
    R.record_measured("Insta360 X4", axis_map=X4_AXIS)
    r = R.identify("whatever.bin", hint="Insta360 X4")
    assert r["status"] == "ok" and r["profile"].axis_map is not None


def test_hint_for_an_unknown_model_refuses(cfg) -> None:
    r = R.identify("whatever.bin", hint="Nope 9000")
    assert r["status"] == "refused" and r["profile"] is None
    assert "profiles" in r["action"]


def test_unidentifiable_file_refuses(tmp_path: Path, cfg) -> None:
    f = tmp_path / "plain.mp4"
    f.write_bytes(b"\x00" * 512)
    r = R.identify(f)
    assert r["status"] == "refused" and r["via"] is None
    assert "--camera" in r["action"]


def test_sniffers_on_canned_ffprobe_output() -> None:
    assert R._has_gpmf({"streams": [{"codec_name": "gpmd"}]}) is True
    assert R._has_gpmf({"streams": [{"codec_name": "h265"}]}) is False
    assert R._is_dji({"format": {"tags": {"make": "DJI"}}}) is True
    assert R._is_dji({"format": {"tags": {"make": "GoPro"}}}) is False


def _fake_camera(monkeypatch, payload):
    monkeypatch.setattr(R, "file_camera", lambda path, reseed=False: payload)


def test_trailer_body_without_a_measurement_asks_for_one(monkeypatch,
                                                         cfg) -> None:
    _fake_camera(monkeypatch, {"serial": "NEW-SERIAL-1", "known": False,
                               "model": "Insta360 X4", "fw": "v1.9.21"})
    r = R.identify("clip.insv")
    assert r["status"] == "needs_measure"
    assert r["serial"] == "NEW-SERIAL-1" and r["measure_key"] == "NEW-SERIAL-1"
    assert "cameras.json" in r["action"]


def test_trailer_body_with_a_measurement_resolves(monkeypatch, cfg) -> None:
    R.record_measured("IBMYA2410M3EB5", model="Insta360 X4", fw="v1.9.21",
                      axis_map=X4_AXIS)
    _fake_camera(monkeypatch, {"serial": "IBMYA2410M3EB5", "known": True,
                               "model": "Insta360 X4", "fw": "v1.9.21"})
    r = R.identify("clip.insv")
    assert r["status"] == "ok" and r["via"] == "trailer"
    assert r["profile"].axis_map.yaw == (1, -1)


def test_trailer_after_a_firmware_update_remeasures(monkeypatch, cfg) -> None:
    R.record_measured("IBMYA2410M3EB5", fw="v1.9.21", axis_map=X4_AXIS)
    _fake_camera(monkeypatch, {"serial": "IBMYA2410M3EB5", "known": True,
                               "model": "Insta360 X4", "fw": "v1.10.0"})
    r = R.identify("clip.insv")
    assert r["status"] == "needs_measure"
    assert "firmware changed" in r["reason"]


def test_trailer_known_body_without_profile_refuses(monkeypatch, cfg) -> None:
    _fake_camera(monkeypatch, {"serial": "XYZ-1", "known": True,
                               "model": "Insta360 X9", "fw": "v1"})
    r = R.identify("clip.insv")
    assert r["status"] == "refused" and "no camera profile" in r["reason"]


def test_gpmf_container_refuses_with_milestone_hint(tmp_path: Path,
                                                    monkeypatch, cfg) -> None:
    f = tmp_path / "gopro.mp4"
    f.write_bytes(b"\x00" * 512)
    monkeypatch.setattr(R, "_has_gpmf", lambda probe: True)
    r = R.identify(f)
    assert r["status"] == "refused" and r["via"] == "gpmf"
    assert "milestone D" in r["action"]


@pytest.mark.skipif(not Path(S).exists(), reason="source footage not present")
def test_trailer_branch_on_a_real_insv(cfg) -> None:
    R.record_measured("IBMYA2410M3EB5", model="Insta360 X4", axis_map=X4_AXIS)
    r = R.identify(S)
    assert r["status"] == "ok" and r["via"] == "trailer"
    assert r["model"] == "Insta360 X4"
    assert r["profile"].model == "Insta360 X4"
    assert r["serial"] == "IBMYA2410M3EB5"
    assert r["profile"].axis_map.yaw == (1, -1)
