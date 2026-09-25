"""Camera registry: measured per-body data, stored OUTSIDE the source tree.

Every .insv trailer carries Metadata with serial_number, camera_type,
fw_version and a per-unit calibration block ("offset"). Measured constants
that belong to one camera body -- at minimum the gyro axis map, which is what
turns a recording into a usable stabilisation -- are kept in the user's config
file (see store.py), keyed by serial number. They are NOT in the source:
an rpm/deb install ships a read-only tree, and a value discovered on first use
has to outlive a package upgrade.

Nothing here guesses. A body with no stored measurement returns
status "needs_measure", and the CLI measures it from the user's own footage
(steadycut.calibration.discover) and stores the result -- so first use of a
new camera is automatic rather than a request to run scripts.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from steadycut.ingest.cameras import store


def axis_map_for(key: str, fw: str | None = None):
    """The stored axis map for a body, or None when it must be (re)measured.

    Keyed by serial number, so every file from one body reuses the single
    measurement taken for it -- there is no per-file re-measurement, and no
    per-file copy of the same numbers.

    A stored measurement records the firmware it was taken under. A firmware
    update can change the gyro's axis order or sign convention (and the
    trailer's per-unit calibration), so a map measured under a different
    firmware is treated as absent and the body is re-measured once. That costs
    one short window and buys the difference between a correct correction and
    a silently inverted one.
    """
    from steadycut.ingest.cameras.profiles import AxisMap

    data = store.entry(key) or {}
    raw = data.get("axis_map")
    if not raw:
        return None
    stored_fw = data.get("fw")
    if fw is not None and stored_fw and stored_fw != fw:
        return None
    try:
        return AxisMap(
            yaw=tuple(raw["yaw"]), pitch=tuple(raw["pitch"]),
            roll=tuple(raw["roll"]))
    except (KeyError, TypeError, ValueError):
        return None


def stale_reason(key: str, fw: str | None) -> str | None:
    """Why a stored measurement cannot be reused, when that is the reason."""
    data = store.entry(key) or {}
    stored_fw = data.get("fw")
    if data.get("axis_map") and fw and stored_fw and stored_fw != fw:
        return (f"firmware changed from {stored_fw} to {fw}: the gyro axis "
                "convention may have moved, so the axes are re-measured once")
    return None


def describe(serial: str) -> dict:
    """A camera's entry from the user's registry, or a measure-first stub.

    Usable means the stored map parses: a truncated or hand-edited entry is
    reported as not usable rather than trusted because it looks present.
    """
    data = store.entry(serial)
    if data and axis_map_for(serial) is not None:
        return {"serial": serial, "known": True, **data}
    if data:
        return {"serial": serial, "known": False, **data,
                "action": ("this body is registered but has no usable axis map "
                           "yet; it is measured automatically on first use")}
    return {"serial": serial, "known": False,
            "action": ("not measured yet; the stabiliser measures this body "
                       "from its own footage on first use and stores it in "
                       f"{store.registry_path()}")}


def record_measured(key: str, *, model: str | None = None,
                    fw: str | None = None, axis_map: dict | None = None,
                    measured: dict | None = None, note: str | None = None
                    ) -> dict:
    """Store a measurement (or the fact that one is missing) for a body.

    `key` is the trailer serial number, or the model name for files that carry
    no trailer. Only non-None fields are written, so a later measurement can
    add detail without clearing what is already known.
    """
    fields = {}
    if model is not None:
        fields["model"] = model
    if fw is not None:
        fields["fw"] = fw
    if axis_map is not None:
        fields["axis_map"] = axis_map
    if measured is not None:
        fields["measured"] = measured
    if note is not None:
        fields["note"] = note
    return store.upsert(key, fields)


def _cache_key(path) -> str:
    from pathlib import Path as P
    st = P(path).stat()
    return f"{P(path).resolve()}|{st.st_size}|{int(st.st_mtime)}"


def file_camera(path, reseed: bool = False) -> dict:
    """Identify the camera that recorded a file, from its trailer.

    The full .telemetry() dump materializes ~2M per-sample dicts and gets
    OOM-killed on a loaded box, so identity metadata is cached per
    (path, size, mtime) after the first successful parse. Pass reseed=True
    (or delete the cache file) to force a fresh parse.
    """
    key = _cache_key(path)
    if not reseed:
        hit = store.load_file_cache().get(key)
        if hit:
            entry = describe(hit.get("serial_number", "missing"))
            entry.update({k: hit.get(k) for k in ("model", "fw")})
            entry["offset_len"] = hit.get("offset_len", 0)
            entry["cached"] = True
            return entry
    import telemetry_parser
    md = telemetry_parser.Parser(str(path)).telemetry()[0]["Default"]["Metadata"]
    meta = {"serial_number": md.get("serial_number", "missing"),
            "model": md.get("camera_type"), "fw": md.get("fw_version"),
            "offset_len": len(md.get("offset") or [])}
    store.store_file_cache(key, meta)
    entry = describe(meta["serial_number"])
    entry.update({k: meta[k] for k in ("model", "fw")})
    entry["offset_len"] = meta["offset_len"]
    entry["cached"] = False
    return entry


if __name__ == "__main__":
    import json
    import sys
    try:
        print(json.dumps(file_camera(sys.argv[1]), indent=1))
    except Exception as exc:  # renders carry no trailer: not a camera file
        print(json.dumps({"file": sys.argv[1], "known": False,
                          "reason": f"no camera trailer ({exc})"}))


# --- identification: trailer first, then container sniffing, then a hint ---

def _probe_streams(path) -> dict:
    """ffprobe the container (streams + format tags), {} on any failure."""
    import json
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, check=True).stdout
        return json.loads(out)
    except Exception:
        return {}


def _has_gpmf(probe: dict) -> bool:
    """A GoPro GPMF telemetry track: data stream named gpmd."""
    for stream in probe.get("streams", []):
        if (stream.get("codec_name") == "gpmd"
                or "gpmd" in str(stream.get("codec_tag_string", ""))):
            return True
    return False


def _is_dji(probe: dict) -> bool:
    """DJI metadata in the container tags."""
    tags = (probe.get("format", {}) or {}).get("tags", {}) or {}
    return any("dji" in str(value).lower() for value in tags.values())


def _measured(source, key, profile, via, serial, model, fw=None) -> dict:
    """Attach the stored axis map to a profile, or ask for a measurement.

    Returns either a usable identification (status "ok", profile carrying the
    measured axis map) or status "needs_measure", which the CLI turns into an
    automatic measurement on the user's own footage.
    """
    am = axis_map_for(key, fw)
    base = {"via": via, "serial": serial, "model": model,
            "measure_key": key, "source": str(source), "file_model": model,
            "fw": fw}
    if am is not None:
        return {"status": "ok", "profile": replace(profile, axis_map=am), **base}
    return {**base, "status": "needs_measure", "profile": profile,
            "reason": stale_reason(key, fw) or (
                f"{model} has no measured axis map for this body ({key})"),
            "action": ("the axis map is measured automatically from this "
                       "file's own gyro and image motion, then stored in "
                       f"{store.registry_path()}")}


def identify(path, hint: str | None = None, reseed: bool = False) -> dict:
    """Identify the camera behind a file, and its profile.

    Order: camera trailer (serial-keyed, cached identity parse), then
    container sniffing (GPMF, DJI), then an explicit --camera hint. Refuses
    with format-aware instructions at every dead end -- never guesses a
    profile. Statuses: "ok" carries a usable profile (measured axis map
    attached); "needs_measure" carries everything needed to measure one;
    "refused" carries the next step.
    """
    from steadycut.ingest.cameras.profiles import profile_for

    # 1. camera trailer (Insta360 family; cached identity parse).
    try:
        cam = file_camera(path, reseed=reseed)
    except Exception:
        cam = None
    if cam is not None:
        serial = cam.get("serial")
        model = cam.get("model")
        profile = profile_for(model)
        if profile is None:
            return {"status": "refused", "via": "trailer", "serial": serial,
                    "model": model, "profile": None,
                    "reason": (f"{model} ({serial}) is identified but has no "
                               "camera profile"),
                    "action": ("this model is not supported yet: add its "
                               "model-wide profile (lens geometry, stack "
                               "order, capabilities) under "
                               "steadycut/ingest/cameras/")}
        return _measured(path, serial or model, profile, "trailer", serial,
                         model, cam.get("fw"))

    # 2. container sniffing for known-but-unsupported families.
    probe = _probe_streams(path)
    if _has_gpmf(probe):
        return {"status": "refused", "via": "gpmf", "serial": None,
                "model": None, "profile": None,
                "reason": "GoPro GPMF telemetry detected, no loader yet",
                "action": "the GPMF loader lands at plan milestone D; until "
                          "then this file is out of scope"}
    if _is_dji(probe):
        return {"status": "refused", "via": "dji", "serial": None,
                "model": None, "profile": None,
                "reason": "DJI metadata detected, no loader yet",
                "action": "the DJI loader lands at plan milestone D; until "
                          "then this file is out of scope"}

    # 3. explicit hint: no serial to key the measurement by, so the model
    #    name is the key.
    if hint:
        profile = profile_for(hint)
        if profile is not None:
            return _measured(path, hint, profile, "hint", None, hint)
        return {"status": "refused", "via": "hint", "serial": None,
                "model": hint, "profile": None,
                "reason": f"no camera profile for --camera {hint}",
                "action": "profiles live in steadycut/ingest/cameras/"}

    return {"status": "refused", "via": None, "serial": None, "model": None,
            "profile": None,
            "reason": (f"{Path(path).name}: no camera trailer and no known "
                       "container signature"),
            "action": ("pass --camera MODEL if you know the body; otherwise "
                       "this format is unsupported")}
