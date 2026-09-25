"""Telemetry dispatch and the scale gates.

The gates exist because a mis-scaled parser (g instead of m/s^2, rad/s instead
of deg/s) produces plausible-looking numbers that poison a whole calibration;
the gravity median catches them cheaply. Real X4 medians are 9.65 and 10.22
m/s^2 -- inside the gate -- so the gate must accept those and refuse grossly
wrong ones.
"""
import numpy as np
import pytest

from steadycut.ingest.cameras.insta360_x4 import PROFILE as X4
from steadycut.ingest.telemetry import (
    NoTelemetryError, Telemetry, check_scales, load, read_telemetry,
)

# The measured mean accel vector from path.py's gravity note, |a| = 9.59.
_BASE = np.array([3.3, 8.71, 2.29])
_BASE_N = float(np.linalg.norm(_BASE))


def _tel(accel_mag=9.81, rate_hz=1000.0) -> Telemetry:
    t = np.arange(0.0, 2.0, 1.0 / rate_hz)
    n = len(t)
    return Telemetry(
        time_s=t,
        gyro=np.zeros((n, 3)),
        accel=np.tile(_BASE * (accel_mag / _BASE_N), (n, 1)),
        model="test",
    )


def test_gravity_gate_accepts_the_measured_range() -> None:
    check_scales(_tel(accel_mag=9.6451))   # real file, Jan15 body
    check_scales(_tel(accel_mag=10.2236))  # real file, Feb21 body


def test_gravity_gate_refuses_a_mis_scale() -> None:
    with pytest.raises(ValueError, match="scale"):
        check_scales(_tel(accel_mag=4.9))    # half scale: g vs m/s^2 class
    with pytest.raises(ValueError, match="scale"):
        check_scales(_tel(accel_mag=57.3))   # deg/s vs rad/s class


def test_rate_gate_uses_the_profile_nominal() -> None:
    check_scales(_tel(rate_hz=999.1), X4)         # the real X4 rate
    with pytest.raises(ValueError, match="rate"):
        check_scales(_tel(rate_hz=50.0), X4)      # 20x off nominal


def test_dispatch_none_format_refuses() -> None:
    no_imu = X4.__class__(
        model="Test No-IMU", telemetry_format="none", axis_map=None,
        lens=X4.lens, imu_rate_nominal_hz=None, max_useful_hfov={},
    )
    with pytest.raises(NoTelemetryError, match="no embedded IMU"):
        load("whatever.bin", no_imu)


def test_dispatch_future_formats_refuse_loudly() -> None:
    with pytest.raises(NotImplementedError, match="gpmf"):
        load("whatever.bin", X4.__class__(
            model="Test GPMF", telemetry_format="gpmf", axis_map=None,
            lens=X4.lens, imu_rate_nominal_hz=None, max_useful_hfov={}))
    with pytest.raises(NoTelemetryError, match="unknown telemetry format"):
        load("whatever.bin", X4.__class__(
            model="Test Junk", telemetry_format="floppy", axis_map=None,
            lens=X4.lens, imu_rate_nominal_hz=None, max_useful_hfov={}))


def test_read_telemetry_is_the_legacy_name_of_load() -> None:
    assert read_telemetry is load


def test_parse_cache_hits_per_file(tmp_path, monkeypatch) -> None:
    """The sync scan re-reads telemetry ~29 times per run; the parse cache
    must turn that into ONE parse per file (the OOM lesson)."""
    import os
    import time

    from steadycut.ingest.telemetry import insv as insv_mod
    calls = []

    def fake_load(path):
        calls.append(1)
        return _tel()

    monkeypatch.setattr(insv_mod, "load", fake_load)
    f = tmp_path / "c.insv"
    f.write_bytes(b"\x00" * 2048)
    a = load(f)
    b = load(f)
    assert len(calls) == 1 and a is b
    os.utime(f, (time.time(), time.time() + 5))  # file changed: re-parse
    load(f)
    assert len(calls) == 2
