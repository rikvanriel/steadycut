"""Clip-job envelope, fingerprint, window check, proxy, lockfile.

The GUI and the CLI render the same bytes because both go through this
module: the GUI writes an envelope, the CLI reads it with --job, and the
fingerprint is computed in exactly one place. Anything that changes the
delivered pixels changes the fingerprint; anything that does not (aim
override display, aspect label, review state) lives outside it.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

ENVELOPE_VERSION = 1

# argparse dest names the envelope may carry. Keys outside this set are
# refused: a stale GUI writing a renamed flag must fail loudly, not render
# a clip whose certificate claims flags it never saw.
ENVELOPE_KEYS = (
    "source start dur out tall framing pitch no_2d no_level stitch preset "
    "repair repair_spans camera corner yaw_corner image_only yaw_bias "
    "pitch_aim level_vqf yaw_bias_track"
).split()


def source_identity(source) -> dict:
    """Identity of the media file: path, size, mtime.

    Size+mtime catch a re-exported file under a stable name; the GUI also
    records the camera body separately once identify() has run.
    """
    st = os.stat(source)
    return {"path": str(Path(source).resolve()),
            "size": st.st_size, "mtime": st.st_mtime}


def fingerprint(args: dict) -> str:
    """Short stable hash of everything that changes the delivered pixels.

    The ONE place fingerprints are computed: the GUI's freshness check,
    the proxy manifest, and the certificate all call this. Render args are
    keyed by argparse dest name so `--job` and literal argv agree.
    """
    canon = {k: args.get(k) for k in sorted(args)}
    blob = json.dumps(canon, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def write_envelope(path, args: dict, base=None) -> dict:
    """Write a versioned envelope; returns the envelope dict.

    `base` records the project root the GUI opened, so a moved project
    directory still resolves its media; the fingerprint pins the absolute
    path, size, and mtime so a silently replaced file cannot pass as the
    same source.
    """
    render = {k: args[k] for k in ENVELOPE_KEYS if k in args}
    env = {"version": ENVELOPE_VERSION,
           "base": str(base) if base is not None else None,
           "source": source_identity(args["source"]),
           "provenance": {"writer": "steadycut-gui",
                          "envelope_keys": sorted(render)},
           "args": render,
           "fingerprint": fingerprint({**render,
                                       "source": source_identity(
                                           args["source"])})}
    Path(path).write_text(json.dumps(env, indent=1))
    return env


def load_envelope(path) -> dict:
    """Load and validate an envelope; raises ValueError on any mismatch."""
    env = json.loads(Path(path).read_text())
    if env.get("version") != ENVELOPE_VERSION:
        raise ValueError(
            f"envelope version {env.get('version')} != {ENVELOPE_VERSION}")
    args = env.get("args", {})
    unknown = sorted(set(args) - set(ENVELOPE_KEYS))
    if unknown:
        raise ValueError(f"envelope carries unknown keys: {unknown}")
    for need in ("source", "start", "dur"):
        if need not in args:
            raise ValueError(f"envelope missing required key: {need}")
    return env


def apply_envelope(args_ns, env: dict, argv=None) -> None:
    """Merge an envelope into parsed CLI args, in place.

    source/start/dur stay on argv and must match the envelope: a job file
    pointing at another window refuses instead of rendering the wrong
    clip. Every other envelope value fills a flag the user did not pass
    explicitly; an explicit flag wins even when it repeats the default
    (value comparison cannot tell those apart, so presence on argv is the
    test). Raises ValueError on a stale job.
    """
    eargs = env["args"]
    if Path(args_ns.source).resolve() != Path(eargs["source"]).resolve():
        raise ValueError(
            f"job is for {eargs['source']}, argv has {args_ns.source}")
    for k in ("start", "dur"):
        if abs(float(getattr(args_ns, k)) - float(eargs[k])) > 1e-6:
            raise ValueError(
                f"job {k}={eargs[k]}, argv {k}={getattr(args_ns, k)}")
    ident = env.get("source")
    if ident is not None:
        try:
            cur = source_identity(eargs["source"])
        except OSError:
            raise ValueError(f"job source not found: {eargs['source']}")
        if (cur["size"] != ident.get("size")
                or cur["mtime"] != ident.get("mtime")):
            raise ValueError(
                f"job source changed since envelope "
                f"(size/mtime mismatch): {eargs['source']}")
    explicit = set()
    if argv is not None:
        from steadycut.steadycut import build_parser
        probe_p = build_parser()
        for a in probe_p._actions:
            if a.dest != "help":
                a.default = argparse.SUPPRESS
        try:
            probe_ns, _ = probe_p.parse_known_args(argv)
            explicit = set(vars(probe_ns)) & set(eargs)
        except SystemExit:
            explicit = set()
    for k, v in eargs.items():
        if k in ("source", "start", "dur") or k in explicit:
            continue
        setattr(args_ns, k, v)


def media_duration(source) -> float:
    """Duration of the first video stream, probed from the file itself."""
    from steadycut.ingest.probe_insv import probe
    res = probe(source)
    for stream in res.video:
        if stream.duration > 0:
            return float(stream.duration)
    raise ValueError(f"no stream duration for {source}")


def snap_window(source, start: float, dur: float,
                duration_fn=None) -> tuple[float, float]:
    """Frame-snapped [start, end): start down, end exclusive.

    Both land on the source's own frame grid (probed, never assumed:
    30000/1001 is not 29.97). A window that snaps to nothing raises
    instead of rendering zero frames.
    """
    from steadycut.render.reframe import snap_to_frame
    total = (duration_fn or media_duration)(source)
    s = snap_to_frame(max(0.0, start), source)
    e = snap_to_frame(min(start + dur, total), source)
    if e <= s:
        raise ValueError(
            f"window [{start}, {start + dur}) snaps to nothing "
            f"(media {total:.2f}s)")
    return s, e - s


def check_window(source, start: float, dur: float,
                 duration_fn=None) -> tuple[bool, str]:
    """Validate a clip window without rendering. Returns (ok, reason)."""
    if not Path(source).exists():
        return False, f"source not found: {source}"
    if dur <= 0:
        return False, f"non-positive duration: {dur}"
    if start < 0:
        return False, f"negative start: {start}"
    try:
        total = (duration_fn or media_duration)(source)
    except Exception as exc:
        return False, f"cannot probe media: {exc}"
    if start >= total:
        return False, f"start {start:.2f}s past media end {total:.2f}s"
    if start + dur > total:
        return False, (f"window end {start + dur:.2f}s past media end "
                       f"{total:.2f}s")
    try:
        snap_window(source, start, dur, duration_fn=duration_fn)
    except ValueError as exc:
        return False, str(exc)
    return True, "ok"


def proxy_manifest(source, camera: dict | None, stitch: str,
                   renderer: str) -> dict:
    """Identity key for a frozen proxy: what rebuilds it, nothing else.

    Aim and aspect are deliberately OUT: the proxy is one frozen framing
    and the aspect is a crop overlay, so neither invalidates the file.
    Anything here changes the proxy pixels; anything not here does not.
    """
    ident = source_identity(source)
    return {"version": ENVELOPE_VERSION,
            "source": ident,
            "camera": camera or {},
            "stitch": stitch,
            "renderer": renderer,
            "fingerprint": fingerprint({"source": ident, "camera": camera or {},
                                        "stitch": stitch,
                                        "renderer": renderer})}


def build_proxy(source, start: float, dur: float, dst,
                pitch: float, size=(960, 720)) -> Path:
    """Render a frozen constant-framing proxy for scrubbing.

    One fixed pitch for the whole window (gravity fallback or the stored
    policy pitch -- never a followed path): the proxy shows WHERE the clip
    is, not HOW the framing moves. Linear cold-open: no gyro, no 2D, no
    certificate -- first frame out in seconds, not minutes.
    """
    from steadycut.core.pipeline import ClipSpec, render_constant
    spec = ClipSpec(source=source, start=start, duration=dur,
                    framing_pitch=pitch, fov=120.0, size=size,
                    profile=None, stitch=None)
    return render_constant(spec, dst, preset="ultrafast")


@contextmanager
def claim_out(out):
    """Lockfile beside the output: two renders never publish one file.

    Yields the staged temp path; the caller renders there, then publishes
    with job.publish-equivalent rename. Stale locks (dead PID) are
    reclaimed; a live holder raises instead of overwriting.

    The lockfile is never unlinked while held: an unlink-while-locked
    orphan lets a second process lock the dead inode while a third locks
    the fresh path, so two renders both believe they hold the lock.
    After flock, the open file is re-statted against the path and the
    claim retries on an inode mismatch, so only the holder of the live
    path proceeds.
    """
    out = Path(out)
    lock = out.with_name(out.name + ".lock")
    for _ in range(100):
        with open(lock, "a+") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                fh.seek(0)
                raise ValueError(
                    f"output locked by another render: {fh.read().strip()}")
            try:
                live = os.stat(lock)
            except OSError:
                continue  # unlinked between open and stat: retry
            if (live.st_ino != os.fstat(fh.fileno()).st_ino
                    or live.st_dev != os.fstat(fh.fileno()).st_dev):
                continue  # we locked an orphan: retry on the live path
            staged = Path(tempfile.mkdtemp(prefix="steadycut-job-")) / out.name
            fh.truncate(0)
            fh.write(f"{os.getpid()}\n")
            fh.flush()
            try:
                yield staged
            finally:
                shutil.rmtree(staged.parent, ignore_errors=True)
            return
    raise ValueError(f"output locked by another render: {lock}")
