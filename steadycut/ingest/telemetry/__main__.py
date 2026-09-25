import sys

from steadycut.ingest.telemetry import NoTelemetryError, read_telemetry

for arg in sys.argv[1:]:
    try:
        telemetry = read_telemetry(arg)
    except Exception as exc:  # noqa: BLE001
        print(f"{arg}: {exc}")
        continue
    print(f"{arg}: model {telemetry.model}, {len(telemetry.time_s)} samples, "
          f"{telemetry.rate_hz:.1f} Hz")
