"""Clip-picker job logic. Qt-free: no PySide6 import anywhere in here.

The panel state translates to argparse dest names (the ONE vocabulary the
CLI parses), then either into a versioned envelope for `--job` or into a
literal argv for Copy CLI. Both paths agree on every envelope key, so the
rendered file is identical either way.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROXY_SIZE = (960, 720)
PROXY_RENDERER = "reframe-1"
PROXY_SPAN_S = 60.0

STAGES = (
    "identify",
    "framing",
    "stitch_setup",
    "sync",
    "path+bake",
)


@dataclass
class GuiState:
    """Everything the Render button needs. Defaults mirror the CLI parser."""

    source: str = ""
    start: float = 0.0
    dur: float = 15.0
    out: str = ""
    framing: str = "mtb"
    tall: bool = False
    pitch: float | None = None
    yaw_bias: float = 0.0
    pitch_aim: float | None = None
    stitch: str = "legacy"
    preset: str = "medium"
    no_2d: bool = False
    no_level: bool = False
    level_vqf: bool = False
    repair: bool = False
    repair_spans: str | None = None
    camera: str | None = None
    corner: float | None = None
    yaw_corner: float | None = None
    image_only: bool = False
    yaw_bias_track: str | None = None


def state_to_render_args(state: GuiState) -> dict:
    """Panel state as argparse dest names (the envelope vocabulary)."""
    return {
        "source": state.source,
        "start": float(state.start),
        "dur": float(state.dur),
        "out": state.out,
        "tall": state.tall,
        "framing": state.framing,
        "pitch": state.pitch,
        "no_2d": state.no_2d,
        "no_level": state.no_level,
        "stitch": state.stitch,
        "preset": state.preset,
        "repair": state.repair,
        "repair_spans": state.repair_spans,
        "camera": state.camera,
        "corner": state.corner,
        "yaw_corner": state.yaw_corner,
        "image_only": state.image_only,
        "yaw_bias": float(state.yaw_bias),
        "pitch_aim": state.pitch_aim,
        "level_vqf": state.level_vqf,
        "yaw_bias_track": state.yaw_bias_track,
    }


def fill_state_from_args(state: GuiState, args: dict) -> GuiState:
    """Load-job: fill the panel from an envelope's args (unknown keys stay)."""
    for key, value in args.items():
        if hasattr(state, key):
            setattr(state, key, value)
    return state


def write_job(state: GuiState, envelope_path: str | Path,
              base=None) -> dict:
    """Write the versioned envelope the render subprocess consumes."""
    from steadycut.job import write_envelope

    return write_envelope(envelope_path, state_to_render_args(state),
                          base=base)


def cli_entry() -> list[str]:
    """How to launch the CLI: same interpreter, module path, no PATH luck."""
    return [sys.executable, "-m", "steadycut.steadycut"]


def job_argv(state: GuiState, envelope_path: str | Path) -> list[str]:
    """Argv for the render subprocess: window on argv, rest via envelope.

    source/start/dur stay on argv and must match the envelope (the backend
    refuses a stale job instead of rendering the wrong window); --out is
    explicit so the destination is visible in the process list.
    """
    return [
        *cli_entry(),
        state.source, "--start", repr(float(state.start)),
        "--dur", repr(float(state.dur)),
        "--job", str(envelope_path),
        "--out", state.out,
    ]


def literal_argv(state: GuiState) -> list[str]:
    """Copy-CLI command: the same render without `--job`, for pasting.

    Only non-default flags are spelled out, so the command stays readable;
    omitted flags are parser defaults, which equal the state defaults, so
    the namespace (and the rendered file) is unchanged.
    """
    argv = [*cli_entry(), state.source,
            "--start", repr(float(state.start)),
            "--dur", repr(float(state.dur))]
    if state.out:
        argv += ["--out", state.out]
    if state.framing != "mtb":
        argv += ["--framing", state.framing]
    if state.pitch is not None:
        argv += ["--pitch", repr(float(state.pitch))]
    if state.tall:
        argv.append("--tall")
    if state.no_2d:
        argv.append("--no-2d")
    if state.no_level:
        argv.append("--no-level")
    if state.level_vqf:
        argv.append("--level-vqf")
    if state.image_only:
        argv.append("--image-only")
    if state.stitch != "legacy":
        argv += ["--stitch", state.stitch]
    if state.preset != "medium":
        argv += ["--preset", state.preset]
    if state.repair:
        argv.append("--repair")
    if state.repair_spans:
        argv += ["--repair-spans", state.repair_spans]
    if state.camera:
        argv += ["--camera", state.camera]
    if state.corner is not None:
        argv += ["--corner", repr(float(state.corner))]
    if state.yaw_corner is not None:
        argv += ["--yaw-corner", repr(float(state.yaw_corner))]
    if state.yaw_bias:
        argv += ["--yaw-bias", repr(float(state.yaw_bias))]
    if state.pitch_aim is not None:
        argv += ["--pitch-aim", repr(float(state.pitch_aim))]
    if state.yaw_bias_track:
        argv += ["--yaw-bias-track", state.yaw_bias_track]
    return argv


def literal_shell(state: GuiState) -> str:
    """Copy-CLI text with the venv binary name when it reads better."""
    argv = literal_argv(state)
    if shutil.which("steadycut") is not None:
        argv = ["steadycut", *argv[3:]]
    return shlex.join(argv)


def extract_refusals(log_text: str) -> list[tuple[str, str]]:
    """(msg, hint) pairs from `REFUSE|` lines; the CLI prints both."""
    lines = [ln[len("REFUSE| "):] for ln in log_text.splitlines()
             if ln.startswith("REFUSE| ")]
    pairs = []
    for i in range(0, len(lines) - 1, 2):
        pairs.append((lines[i], lines[i + 1]))
    if len(lines) % 2:
        pairs.append((lines[-1], ""))
    return pairs


def stage_of(line: str) -> str | None:
    """Map a `CLI|` log line to a render stage, for the progress label."""
    if line.startswith("CLI| time "):
        return line[len("CLI| time "):].split()[0]
    if line.startswith("CLI| camera"):
        return "identify"
    if line.startswith("CLI| framing"):
        return "framing"
    if line.startswith("CLI| sync"):
        return "sync"
    if line.startswith("CLI| wrote"):
        return "done"
    return None


def stage_fraction(stage: str | None) -> float | None:
    """Determinate progress from the stage order; None before/after."""
    if stage in (None, "identify"):
        return None
    if stage == "done":
        return 1.0
    try:
        return (STAGES.index(stage) + 1) / (len(STAGES) + 1)
    except ValueError:
        return None


def proxy_cache_dir() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME",
                               Path.home() / ".cache")) / "steadycut" / "proxy"


def proxy_paths(fingerprint: str) -> tuple[Path, Path]:
    """(video, manifest) cache paths for one frozen proxy identity."""
    base = proxy_cache_dir() / fingerprint
    return base.with_suffix(".mp4"), base.with_suffix(".manifest.json")


def current_proxy_manifest(source: str, camera: dict | None,
                           stitch: str) -> dict:
    """The manifest a fresh proxy for this source would carry."""
    from steadycut.job import proxy_manifest

    return proxy_manifest(source, camera, stitch, PROXY_RENDERER)


def proxy_is_fresh(manifest_path: str | Path, source: str,
                   camera: dict | None, stitch: str) -> bool:
    """True when the cached proxy's manifest matches this source."""
    import json

    try:
        cached = json.loads(Path(manifest_path).read_text())
    except (OSError, ValueError):
        return False
    want = current_proxy_manifest(source, camera, stitch)
    return cached.get("fingerprint") == want["fingerprint"]


def proxy_span(total_s: float) -> tuple[float, float]:
    """Frozen proxy window: the first minute (or the whole short file)."""
    dur = min(PROXY_SPAN_S, max(1.0, total_s))
    return 0.0, dur


def default_out_dir(source: str) -> Path:
    """Where render products go: beside the media when writable.

    Footage often lives on a read-only mount (SD-card dumps, archives),
    and the envelope, log, and certificate all land next to `--out` —
    so an unwritable media dir falls back to ~/Videos/steadycut, then
    the working directory. Always explicit on argv either way.
    """
    parent = Path(source).parent
    if os.access(parent, os.W_OK):
        return parent
    videos = Path.home() / "Videos" / "steadycut"
    try:
        videos.mkdir(parents=True, exist_ok=True)
        if os.access(videos, os.W_OK):
            return videos
    except OSError:
        pass
    return Path.cwd()


def default_out(source: str, start: float, tall: bool = False) -> str:
    """Render destination; always explicit on argv."""
    stem = Path(source).stem
    tag = "tall" if tall else "wide"
    return str(default_out_dir(source) / f"{stem}_{int(start)}s_{tag}.mp4")


def fmt_time(seconds: float) -> str:
    """Absolute media time as MM:SS.t for the panel fields."""
    seconds = max(0.0, seconds)
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def parse_time(text: str) -> float:
    """Panel input back to seconds: `372.5`, `06:12.5`, or `6:12`."""
    text = text.strip()
    if ":" in text:
        minute, rest = text.split(":", 1)
        return int(minute) * 60 + float(rest)
    return float(text)


def media_total(source: str) -> float:
    """Full media duration, probed from the file itself."""
    from steadycut.job import media_duration

    return media_duration(source)


def identify_camera(source: str, hint: str | None = None) -> dict:
    """Camera gate result for the top bar (lazy: registry import here)."""
    from steadycut.ingest.cameras.registry import identify

    return identify(source, hint=hint)


def frozen_pitch(source: str, start: float, profile) -> tuple[float, dict]:
    """One fixed pitch for the proxy: stored policy pitch, else gravity.

    Never a followed path: the proxy shows WHERE the clip is, not HOW the
    framing moves. Falls back to -12.0 with usable=False when gravity is
    untrusted on this window, so the caller can say so on the preview tag.
    """
    from steadycut.core.profiles import load_for
    from steadycut.framing import ladder

    stored = load_for(source)
    if stored is not None and stored.pitch is not None:
        return stored.pitch, {"layer": "stored", "usable": True}
    pitch, info = ladder.gravity_fallback(source, start, profile)
    if pitch is not None:
        return pitch, info
    return -12.0, {"layer": "constant", "usable": False,
                   "reason": info.get("reason")}


def check_state(state: GuiState, duration_fn=None) -> tuple[bool, str]:
    """Preflight without rendering: existence, bounds, snapping.

    Always probes through `media_total` above, so the panel refusal
    matches what the render subprocess would do.
    """
    from steadycut.job import check_window

    return check_window(state.source, state.start, state.dur,
                        duration_fn=duration_fn or media_total)


def tail_bytes(path: str | Path, limit: int = 65536) -> str:
    """Last `limit` bytes of a log file, decoded loosely."""
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - limit))
        return fh.read().decode("utf-8", "replace")


def fingerprint_of(state: GuiState) -> str:
    """Short hash of the render args behind the current panel state."""
    from steadycut.job import fingerprint

    return fingerprint(state_to_render_args(state))
