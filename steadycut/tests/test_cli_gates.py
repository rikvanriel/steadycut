"""CLI capability gates: the camera ladder refuses with named next steps.

The gates are driven by a fake identify() (monkeypatched at the registry), so
no invented camera profile is needed -- the gate logic is the component under
test, and its inputs are the profile fields themselves.
"""
from pathlib import Path

import pytest

from steadycut.steadycut import build_parser, main
from steadycut.ingest.cameras.insta360_x4 import PROFILE as X4


def _ok_ident(profile):
    return {"status": "ok", "via": "hint", "serial": None,
            "model": profile.model, "profile": profile}


def _patch_identify(monkeypatch, ident):
    from steadycut.ingest.cameras import registry as R
    monkeypatch.setattr(R, "identify", lambda path, hint=None, reseed=False:
                        ident)


def _variant(**overrides):
    from dataclasses import replace

    from steadycut.ingest.cameras.profiles import AxisMap
    # The map the registry would attach for a measured body; the gates under
    # test are about capabilities, so a usable map keeps them independent.
    fields = dict(model="Test Cam", capabilities=X4.capabilities,
                  prestabilized_default=False,
                  axis_map=AxisMap(yaw=(1, -1), pitch=(2, -1), roll=(0, -1)))
    fields.update(overrides)
    return replace(X4, **fields)


def test_parser_accepts_camera_and_image_only() -> None:
    a = build_parser().parse_args(
        ["f.mp4", "--start", "1", "--dur", "2", "--camera", "Insta360 X4",
         "--image-only"])
    assert a.camera == "Insta360 X4" and a.image_only is True


def test_unidentifiable_file_refuses(tmp_path: Path) -> None:
    fake = tmp_path / "plain.mp4"
    fake.write_bytes(b"\x00" * 512)
    rc = main([str(fake), "--start", "0", "--dur", "1",
               "--out", str(tmp_path / "o.mp4")])
    assert rc == 2
    assert not (tmp_path / "o.mp4").exists()  # no clip claimed


def test_no_imu_camera_refuses_and_offers_image_only(tmp_path, capsys,
                                                     monkeypatch) -> None:
    _patch_identify(monkeypatch, _ok_ident(
        _variant(capabilities=frozenset({"reaim"}), telemetry_format="none")))
    rc = main([str(tmp_path / "x.mp4"), "--start", "0", "--dur", "1"])
    assert rc == 2
    assert "--image-only" in capsys.readouterr().out


def test_prestabilized_camera_refuses(tmp_path, capsys, monkeypatch) -> None:
    _patch_identify(monkeypatch, _ok_ident(
        _variant(prestabilized_default=True)))
    rc = main([str(tmp_path / "x.mp4"), "--start", "0", "--dur", "1"])
    assert rc == 2
    assert "stabilisation OFF" in capsys.readouterr().out


def test_unmeasured_axis_map_refuses_with_the_next_step(tmp_path, capsys,
                                                        monkeypatch) -> None:
    """An ok identification with no map is still refused, not guessed."""
    _patch_identify(monkeypatch, _ok_ident(_variant(axis_map=None)))
    rc = main([str(tmp_path / "x.mp4"), "--start", "0", "--dur", "1"])
    assert rc == 2
    assert "measured axis map" in capsys.readouterr().out


def test_first_use_measures_the_body_and_stores_it(tmp_path, capsys,
                                                   monkeypatch) -> None:
    """needs_measure is answered by measuring, not by telling the user to."""
    from dataclasses import replace

    from steadycut.calibration import discover as D
    from steadycut.ingest.cameras import registry as R
    from steadycut.ingest.cameras.profiles import AxisMap

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    measured = AxisMap(yaw=(1, -1), pitch=(2, -1), roll=(0, -1))
    calls = {"n": 0}

    def fake_identify(path, hint=None, reseed=False):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"status": "needs_measure", "via": "trailer",
                    "serial": "NEW-SERIAL", "model": "Insta360 X4",
                    "profile": X4, "measure_key": "NEW-SERIAL", "fw": "v1",
                    "reason": "Insta360 X4 has no measured axis map",
                    "action": "measured automatically", "source": str(path)}
        return {"status": "ok", "via": "trailer", "serial": "NEW-SERIAL",
                "model": "Insta360 X4", "measure_key": "NEW-SERIAL",
                "fw": "v1", "profile": replace(X4, axis_map=measured),
                "source": str(path)}

    monkeypatch.setattr(R, "identify", fake_identify)
    monkeypatch.setattr(D, "discover_axis_map", lambda *a, **k: {
        "ok": True, "axis_map": {"yaw": [1, -1], "pitch": [2, -1],
                                 "roll": [0, -1]},
        "corr": {"yaw": [0.1, -0.8, 0.2], "pitch": [0.2, 0.1, -0.8],
                 "roll": [-0.8, 0.2, 0.1]},
        "start": 3.0, "strength": 0.8, "fps": 59.94, "frames": 120,
        "attempts": []})

    rc = main([str(tmp_path / "x.insv"), "--start", "3", "--dur", "2"])
    out = capsys.readouterr().out
    assert "measuring this body's gyro axes" in out
    assert "measured axis map yaw=axis1-1" in out
    stored = R.store.entry("NEW-SERIAL")
    assert stored is not None and stored["axis_map"]["yaw"] == [1, -1]
    assert stored["fw"] == "v1"
    # The gate was passed and the pipeline continued; it can only refuse later
    # because the fake file cannot render.
    assert rc == 2


def test_image_only_runs_without_telemetry(tmp_path, capsys, monkeypatch) -> None:
    """The L3 path must not touch the telemetry stage at all."""
    from steadycut.core import pipeline
    _patch_identify(monkeypatch, _ok_ident(
        _variant(capabilities=frozenset({"image_only"}),
                 telemetry_format="none")))
    calls = []
    monkeypatch.setattr(pipeline, "stabilize_image_only",
                        lambda *args, **kw: calls.append(args) or {
                            "output": Path(kw.get("output", "o.mp4")),
                            "raw_bounce": 10.0, "corrected_bounce": 5.0,
                            "inliers_med": 300})
    out = tmp_path / "o.mp4"
    rc = main([str(tmp_path / "x.mp4"), "--start", "0", "--dur", "1",
               "--image-only", "--out", str(out)])
    assert rc == 0
    assert len(calls) == 1  # reached the 2D pass, never a telemetry refusal
    assert "image-only far bounce raw 10.00 -> corrected 5.00" \
        in capsys.readouterr().out
