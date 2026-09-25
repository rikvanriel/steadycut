"""Ingest package tests: trailer walk + camera identity, no footage needed."""
import struct
from pathlib import Path

from steadycut.ingest.insv_trailer import (
    IMU_RECORD, MAGIC, NotInsta360Error, read_imu, read_trailer,
)
from steadycut.ingest.cameras.registry import describe


def _write_insv(path: Path, records, padding: int = 0) -> None:
    body = b"\x00" * 4096
    trailer = b""
    for record_id, payload in records:
        trailer += payload + struct.pack("<HI", record_id, len(payload))
        trailer += b"\x00" * padding
    footer = struct.pack("<II", len(trailer) + len(MAGIC), 3) + MAGIC
    path.write_bytes(body + trailer + footer)


def test_trailer_roundtrip(tmp_path: Path) -> None:
    samples = [(0, (0.0, 0.0, 1.0), (1.0, 2.0, 3.0)),
               (10, (0.2, 0.0, 1.0), (2.0, 3.0, 4.0))]
    payload = b"".join(IMU_RECORD.pack(t, *a, *g) for t, a, g in samples)
    path = tmp_path / "clip.insv"
    _write_insv(path, [(0x0300, payload)])
    imu = read_imu(path, read_trailer(path))
    assert [s["t_ms"] for s in imu] == [0, 10]
    assert imu[0]["gyro"] == (1.0, 2.0, 3.0)


def test_non_insv_rejected(tmp_path: Path) -> None:
    path = tmp_path / "plain.mp4"
    path.write_bytes(b"\x00" * 512)
    try:
        read_trailer(path)
    except NotInsta360Error:
        return
    raise AssertionError("expected NotInsta360Error")


def test_registry_measure_then_use(monkeypatch, tmp_path) -> None:
    """A body resolves once its measurement is stored, and not before."""
    from steadycut.ingest.cameras import registry as R
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    new = describe("SOME-NEW-CAMERA")
    assert new["known"] is False
    assert "action" in new  # measure-first, never inherit silently
    R.record_measured("SOME-NEW-CAMERA", model="Insta360 X4",
                      axis_map={"yaw": [1, -1], "pitch": [2, -1],
                                "roll": [0, -1]})
    known = describe("SOME-NEW-CAMERA")
    assert known["known"] is True
    assert known["model"] == "Insta360 X4"
