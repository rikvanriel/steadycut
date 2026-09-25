"""Measure a camera body's own numbers from its footage.

The first time steadycut sees a camera body it needs that body's gyro axis map,
which is not printed on any datasheet and, being a hardware convention, is
asserted rather than derived. So it is measured on the body itself: render a
short constant-framing view of one of the user's own files, measure the image
motion between frames, and correlate each gyro column against the motion it is
supposed to explain:

    horizontal image motion -> yaw
    vertical image motion   -> pitch
    in-plane rotation       -> roll

The column with the largest |r| wins, and the sign of that correlation gives
the direction. The sign convention is measured, not paper-derived: on the X4
every channel correlates negatively (|r| ~ 0.84) and the mapping that
stabilises is the negative one, because the rear-first fisheye stack order
rotates the stitched sphere 180 degrees against the camera axes. That is why
no map is ever copied between models -- a second model is measured this way,
which is exactly what this module does.

Refusal is part of the contract. A window too smooth to correlate (|r| below
MIN_ABS_R on any channel, or two channels claiming the same column) would store
a map that stabilises the wrong way, which is worse than no stabilisation at
all. Discovery then reports failure and says what to do instead.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

MIN_ABS_R = 0.5
DEFAULT_CANDIDATES = (0.0, 120.0, 240.0)   # offsets after the requested start


def probe_frames(source, start: float, duration: float, workdir: Path,
                 size=(960, 540), preset="ultrafast"):
    """Render a constant view and return image motion, roll, gyro and fps.

    A constant control point means no stabilisation is applied, so the frame
    pairs carry the camera's own motion -- which is the signal being measured.
    """
    from steadycut.metrics import rotate, shake
    from steadycut.ingest.telemetry import read_telemetry
    from steadycut.render.reframe import render, source_fps
    from steadycut.stabilization.path import PathPoint

    clip = Path(workdir) / "_discover_base.mp4"
    render(source, clip, [PathPoint(t=0.0)], start=start, duration=duration,
           size=size, preset=preset)

    frames = shake.frames(str(clip))
    motion = np.array([shake.shift(frames[i], frames[i + 1])
                       for i in range(len(frames) - 1)])
    roll = np.asarray(rotate.rotation_series(str(clip))[0])[:len(motion)]

    fps = source_fps(str(source))
    imu = read_telemetry(str(source))
    frame_t = start + np.arange(len(motion)) / fps
    gyro = np.column_stack([
        np.interp(frame_t, imu.time_s, imu.gyro[:, a]) for a in range(3)
    ])
    return motion, roll, gyro, fps


def _corr(a, b) -> float:
    """Pearson correlation, defining the degenerate case as 0.

    A constant series (a window with no motion) has no correlation to report,
    and numpy's corrcoef returns NaN for it -- which would slip through the
    |r| gate below, since every comparison against NaN is False. Returning 0
    makes a motionless window a refusal, which is what it should be.
    """
    a = np.asarray(a, dtype=float) - np.mean(a)
    b = np.asarray(b, dtype=float) - np.mean(b)
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def infer_map(motion, roll, gyro):
    """Correlate each image motion against each gyro column.

    Returns (axis_map, corr): `axis_map` is {channel: [column, sign]} and
    `corr` keeps every correlation coefficient, so the caller can refuse a
    weak measurement and a reader can see what was actually measured. Split
    from the rendering so the inference is testable against a known answer.
    """
    axis_map, corr = {}, {}
    for channel, series in (("yaw", motion[:, 0]),
                            ("pitch", motion[:, 1]),
                            ("roll", roll)):
        rs = [_corr(gyro[:, a], series) for a in range(3)]
        best = int(np.argmax(np.abs(rs)))
        corr[channel] = rs
        axis_map[channel] = [best, -1 if rs[best] < 0 else 1]
    return axis_map, corr


def _strength(axis_map, corr) -> float:
    """The weakest channel's |r| at its chosen column: the floor that matters.

    A map is only as good as its worst channel, because a single wrong column
    or sign wrecked by low correlation is what the stabiliser would apply.
    """
    return min(abs(corr[ch][axis_map[ch][0]]) for ch in corr)


def _weak(axis_map, corr) -> str | None:
    """The reason this measurement must not be stored, or None if it may."""
    for channel, rs in corr.items():
        col, _ = axis_map[channel]
        r = rs[col]
        if not np.isfinite(r) or abs(r) < MIN_ABS_R:
            return (f"{channel} correlates at only {abs(r):.2f} "
                    f"(needs {MIN_ABS_R})")
    cols = [tuple(v) for v in axis_map.values()]
    if len({c[0] for c in cols}) < 3:
        return ("two channels claim the same gyro column: "
                f"{ {k: v[0] for k, v in axis_map.items()} }")
    return None


def discover_axis_map(source, start: float, duration: float = 2.0,
                      workdir=None, candidates=DEFAULT_CANDIDATES) -> dict:
    """Measure this body's axis map from its own footage, or say why not.

    Tries the requested window first, then the later offsets in `candidates`,
    because the window a user asks for may be a smooth piece of trail: a
    measurement needs the camera to actually have moved. The strongest window
    that passes the |r| gate is kept, and each attempt is reported so a
    failure says which windows were tried rather than just "failed".
    """
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(workdir) if workdir is not None else Path(tmp)
        base.mkdir(parents=True, exist_ok=True)
        attempts, best = [], None
        for offset in candidates:
            window = start + offset
            try:
                motion, roll, gyro, fps = probe_frames(
                    source, window, duration, base)
            except Exception as exc:      # past the end of the clip, etc.
                attempts.append({"start": window, "skipped":
                                 f"{type(exc).__name__}: {exc}"})
                continue
            if len(motion) < 8:
                attempts.append({"start": window,
                                 "skipped": "too few frames"})
                continue
            axis_map, corr = infer_map(motion, roll, gyro)
            strength = _strength(axis_map, corr)
            attempts.append({"start": window, "strength": strength,
                             "corr": corr})
            reason = _weak(axis_map, corr)
            if reason is None and (best is None or strength > best["strength"]):
                best = {"axis_map": axis_map, "corr": corr, "start": window,
                        "strength": strength, "fps": fps,
                        "frames": int(len(motion))}
        if best is None:
            return {"ok": False, "attempts": attempts,
                    "reason": ("no window carried enough motion to measure "
                               "this body's axes"),
                    "action": ("re-run on a section where the camera is "
                               "moving and shaking (a descending trail, not "
                               "a smooth roll-out)")}
        return {"ok": True, "attempts": attempts, **best}
