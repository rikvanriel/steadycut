"""Tests for the pieces that can be checked without camera-original footage.

The trailer reader is exercised against a synthesised trailer rather than a
real recording, so these run anywhere; the properties they pin down (walking
backwards from the magic, tolerating the zero padding X4 inserts, unpacking
56-byte IMU records) are exactly the ones that broke when parsers written for
older models met X4 files.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from steadycut.ingest.insv_trailer import (
    IMU_RECORD,
    MAGIC,
    NotInsta360Error,
    read_imu,
    read_trailer,
)
from steadycut.render.reframe import ControlPoint, build_command_file


def _imu_payload(samples: list[tuple[int, tuple[float, float, float], tuple[float, float, float]]]) -> bytes:
    return b"".join(
        IMU_RECORD.pack(t, *accel, *gyro) for t, accel, gyro in samples
    )


def _write_insv(path: Path, records: list[tuple[int, bytes]], padding: int = 0) -> None:
    """Build a file whose tail mimics an Insta360 trailer.

    Records are laid out payload-then-header so that a reader walking
    backwards from the footer meets each header before its payload, which is
    the arrangement the real format uses.
    """
    body = b"\x00" * 4096  # stand-in for the MP4 that precedes the trailer
    trailer = b""
    for record_id, payload in records:
        trailer += payload + struct.pack("<HI", record_id, len(payload))
        trailer += b"\x00" * padding

    # The size field spans the records and the magic, but not the version and
    # size words; read_trailer subtracts those 8 bytes to find the start.
    footer = struct.pack("<II", len(trailer) + len(MAGIC), 3) + MAGIC
    path.write_bytes(body + trailer + footer)


def test_reads_records_and_imu(tmp_path: Path) -> None:
    samples = [
        (0, (0.0, 0.0, 1.0), (1.0, 2.0, 3.0)),
        (5, (0.1, 0.0, 1.0), (1.5, 2.5, 3.5)),
        (10, (0.2, 0.0, 1.0), (2.0, 3.0, 4.0)),
    ]
    path = tmp_path / "clip.insv"
    _write_insv(path, [(0x0101, b"makernotes"), (0x0300, _imu_payload(samples))])

    trailer = read_trailer(path)
    assert set(trailer.records) == {0x0101, 0x0300}
    assert trailer.records[0x0300].name == "imu"

    imu = read_imu(path, trailer)
    assert [s["t_ms"] for s in imu] == [0, 5, 10]
    assert imu[0]["gyro"] == (1.0, 2.0, 3.0)
    assert imu[-1]["accel"] == pytest.approx((0.2, 0.0, 1.0))


def test_tolerates_padding_between_records(tmp_path: Path) -> None:
    """X4 separates records with zero padding; a naive walker desynchronises."""
    samples = [(0, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0))]
    path = tmp_path / "padded.insv"
    _write_insv(
        path,
        [(0x0101, b"makernotes"), (0x0300, _imu_payload(samples))],
        padding=7,
    )

    trailer = read_trailer(path)
    assert 0x0300 in trailer.records
    assert len(read_imu(path, trailer)) == 1


def test_rejects_file_without_magic(tmp_path: Path) -> None:
    path = tmp_path / "plain.mp4"
    path.write_bytes(b"\x00" * 8192)
    with pytest.raises(NotInsta360Error):
        read_trailer(path)


def test_missing_imu_record_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "noimu.insv"
    _write_insv(path, [(0x0101, b"makernotes")])
    assert read_imu(path) == []


def test_gps_lives_at_0x0700_on_x4(tmp_path: Path) -> None:
    path = tmp_path / "gps.insv"
    _write_insv(path, [(0x0700, b"gpsdata"), (0x0300, _imu_payload([]))])
    trailer = read_trailer(path)
    assert trailer.records[0x0700].name == "gps"


def test_command_file_ramps_between_control_points() -> None:
    commands = build_command_file(
        [ControlPoint(t=0.0, yaw=-45.0), ControlPoint(t=2.0, yaw=45.0)]
    )
    # Absolute per-frame commands, which is the delivery that WORKS. The interval
    # lerp form was the default for a long time and is broken: it spun a 5 s view
    # by about 150 degrees while the same path delivered per frame held steady.
    # See build_command_file's docstring for the measurement.
    assert "[expr]" not in commands
    assert "lerp(" not in commands
    lines = [l for l in commands.splitlines() if "v360 yaw" in l]
    assert len(lines) > 50                    # one per frame, ~30 fps over 2 s
    times = [float(l.split()[0]) for l in lines]
    vals = [float(l.split()[-1].rstrip(";")) for l in lines]
    assert times == sorted(times)
    assert abs(vals[0] - -45.0) < 0.01        # begins at the first control point
    # Frame cadence stops just BEFORE the final control point, so the last emitted
    # value is short of it by one frame's worth of ramp: 90 degrees over 2 s at
    # ~30 fps is about 1.5 degrees per frame.
    assert abs(vals[-1] - 45.0) < 2.0
    assert vals[-1] > vals[0]                 # it ramps, rather than stepping


def test_per_frame_delivery_is_the_default() -> None:
    """The delivery that works must be the one you get without asking."""
    import inspect
    sig = inspect.signature(build_command_file)
    assert sig.parameters["per_frame"].default is True


def test_lerp_delivery_remains_selectable() -> None:
    """Kept reachable so the broken mode can be demonstrated, never defaulted to."""
    commands = build_command_file(
        [ControlPoint(t=0.0, yaw=-45.0), ControlPoint(t=2.0, yaw=45.0)],
        per_frame=False)
    assert "lerp(" in commands


def test_single_control_point_is_static() -> None:
    commands = build_command_file([ControlPoint(t=0.0, yaw=30.0)])
    assert "v360 yaw 30.0000" in commands
    assert commands.count("v360 yaw") == 1


def test_empty_path_rejected() -> None:
    with pytest.raises(ValueError):
        build_command_file([])


def test_negative_times_are_dropped_from_the_command_file():
    """Telemetry starts before the first frame, so paths carry negative times.

    Emitting those makes intervals like "-1.000000--0.950000", whose leading
    minus collides with the interval separator and which ffmpeg rejects.
    """
    from steadycut.render.reframe import ControlPoint, build_command_file

    path = [ControlPoint(t=-1.0, yaw=5.0), ControlPoint(t=0.0, yaw=0.0),
            ControlPoint(t=2.0, yaw=10.0)]
    commands = build_command_file(path)
    times = [float(l.split()[0]) for l in commands.splitlines() if l.strip()]
    assert times, "no commands emitted"
    assert all(t >= 0.0 for t in times), "a negative time survived"
    assert min(times) == 0.0


def test_path_with_only_negative_times_is_rejected():
    from steadycut.render.reframe import ControlPoint, build_command_file
    import pytest

    with pytest.raises(ValueError):
        build_command_file([ControlPoint(t=-2.0)])
