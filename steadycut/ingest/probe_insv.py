"""Identify what an Insta360 file actually contains.

The file extension cannot be trusted: started over its HTTP API rather than
the shutter button, an X4 writes genuine dual-fisheye content into a file
named .mp4.  Anything that dispatches on the extension then treats it as
flat video and silently discards the second lens and the IMU trailer.  The
check that matters is the stream layout, so that is what this probes.

Two modes are recorded in-camera and cannot be undone afterwards -- Me Mode
bakes in both stabilisation and stitching, and the single-lens/FOV+ modes are
flat video rather than 360 -- so both are rejected here instead of failing
later in the pipeline.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from steadycut.ingest.insv_trailer import NotInsta360Error, read_trailer


@dataclass
class VideoStream:
    index: int
    codec: str
    width: int
    height: int
    frames: int
    duration: float
    fps: float


@dataclass
class Probe:
    path: Path
    video: list[VideoStream]
    audio_channels: int | None
    trailer_records: dict[int, str] = field(default_factory=dict)
    has_imu: bool = False
    has_gps: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def is_dual_fisheye(self) -> bool:
        """True when the file holds two equal square video streams."""
        if len(self.video) != 2:
            return False
        first, second = self.video
        return (
            first.width == first.height
            and (first.width, first.height) == (second.width, second.height)
        )

    @property
    def usable(self) -> bool:
        return self.is_dual_fisheye and not self.errors


def _ffprobe(path: Path) -> dict:
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found on PATH")

    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _fps(stream: dict) -> float:
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    try:
        num, _, den = rate.partition("/")
        den_value = float(den) if den else 1.0
        return float(num) / den_value if den_value else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def probe(path: str | Path) -> Probe:
    path = Path(path)
    data = _ffprobe(path)

    video: list[VideoStream] = []
    audio_channels: int | None = None

    for stream in data.get("streams", []):
        kind = stream.get("codec_type")
        if kind == "video":
            video.append(
                VideoStream(
                    index=stream.get("index", -1),
                    codec=stream.get("codec_name", "?"),
                    width=int(stream.get("width") or 0),
                    height=int(stream.get("height") or 0),
                    frames=int(stream.get("nb_frames") or 0),
                    duration=float(stream.get("duration") or 0.0),
                    fps=_fps(stream),
                )
            )
        elif kind == "audio" and audio_channels is None:
            audio_channels = int(stream.get("channels") or 0)

    result = Probe(path=path, video=video, audio_channels=audio_channels)

    try:
        trailer = read_trailer(path)
        result.trailer_records = {r.id: r.name for r in trailer.records.values()}
        result.has_imu = 0x0300 in trailer.records
        result.has_gps = 0x0700 in trailer.records
    except NotInsta360Error:
        result.errors.append("no Insta360 trailer: not camera-original media")
    except OSError as exc:
        result.errors.append(f"could not read trailer: {exc}")

    if len(video) == 1:
        result.errors.append(
            "single video stream: single-lens/FOV+ or an already-stitched export, "
            "not a 360 recording this tool can reframe"
        )
    elif len(video) == 0:
        result.errors.append("no video streams")
    elif not result.is_dual_fisheye:
        shape = ", ".join(f"{s.width}x{s.height}" for s in video)
        result.errors.append(f"expected two equal square streams, found {shape}")

    if result.is_dual_fisheye and not result.has_imu:
        result.errors.append(
            "dual-fisheye but no IMU record: horizon levelling and gaze-follow "
            "are unavailable"
        )

    return result


def describe(result: Probe) -> str:
    lines = [f"file          {result.path.name}"]
    for stream in result.video:
        lines.append(
            f"  video[{stream.index}]   {stream.codec} {stream.width}x{stream.height} "
            f"{stream.fps:.2f} fps  {stream.duration:.1f}s"
        )
    if result.audio_channels is not None:
        lines.append(f"  audio       {result.audio_channels} ch")

    if result.trailer_records:
        names = ", ".join(
            f"{name}({rid:#06x})" for rid, name in sorted(result.trailer_records.items())
        )
        lines.append(f"  trailer     {names}")

    lines.append(f"  dual fisheye: {'yes' if result.is_dual_fisheye else 'no'}")
    lines.append(f"  imu:          {'yes' if result.has_imu else 'no'}")
    lines.append(f"  gps:          {'yes' if result.has_gps else 'no'}")

    for message in result.errors:
        lines.append(f"  ERROR       {message}")

    lines.append(f"  usable:       {'yes' if result.usable else 'no'}")
    return "\n".join(lines)


def _main() -> None:
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("usage: probe_insv.py <file>")
    print(describe(probe(sys.argv[1])))


if __name__ == "__main__":
    _main()
