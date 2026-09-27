"""Driving code: the pipeline owns intermediates, not /tmp files.

Every earlier investigation script passed numerics through .npy files and
temp videos by hand. This module is the single place that composition lives:
rendering, tracking, correlating and warping call each other with arrays and
small dataclasses. Videos are still files (they are large), but they travel
as explicit arguments and every numeric result is returned in memory.

Conventions shared here so tools agree:
- tracker plate scale is derived from the field of view, never hardcoded.
- smoothing is zero-phase (symmetric kernel, edge-padded): the acausal rule.
- framing is stated on every measurement (the far-field metric depends on it).
"""
from __future__ import annotations

import math
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from steadycut.ingest.cameras.profiles import CameraProfile


@dataclass(frozen=True)
class ClipSpec:
    """What to render: source file, window, framing. No correction in here.

    `profile` is the camera model's profile when known; every stage that needs
    a camera constant (axis map, lens geometry) reads it from here, so the X4
    is not baked into the stages themselves.

    `framing_pitch`'s default is a placeholder for tests and synthetic specs,
    NOT a production framing: the pitch is measured per recording by the
    framing policy for the mount (steadycut.framing.policies) and must never be
    carried across rides.
    """
    source: str | Path
    start: float
    duration: float
    framing_pitch: float = -12.0
    fov: float = 120.0
    size: tuple[int, int] = (960, 720)
    profile: "CameraProfile | None" = None


@dataclass(frozen=True)
class Traj:
    """Per-frame far-field motion: dx, dy (pixels at tracker scale), dtheta."""
    deltas: np.ndarray          # (N, 3) float
    inliers: np.ndarray         # (N,) int
    fps: float | None = None    # probed per source; never a camera constant


@dataclass(frozen=True)
class SyncResult:
    """Gyro-stamp offset measured on one file+window. See evaluate --sync-scan
    for the sign convention; confirm on a second window before using."""
    lag_ms: float
    r_best: float
    r_at_zero: float
    curve: tuple = ()
    inliers_med: float = 0.0


def plate_scale(h_fov: float, w: int = 1280, h: int = 960) -> tuple[float, float]:
    """Pixels per degree at tracker scale for a flat v360 output.

    v360 takes a horizontal fov and derives the vertical from the frame shape,
    so both scales derive from h_fov (validated: 10.67/9.15 at h_fov 120).
    """
    ppx = w / h_fov
    v_fov = 2.0 * math.degrees(math.atan(math.tan(math.radians(h_fov / 2)) * h / w))
    return ppx, h / v_fov


def gsmooth(x: np.ndarray, sigma_frames: float) -> np.ndarray:
    """Zero-phase Gaussian smoothing of a 1-D series (symmetric, edge-padded)."""
    r = max(1, int(3 * sigma_frames))
    k = np.arange(-r, r + 1)
    w = np.exp(-0.5 * (k / sigma_frames) ** 2)
    w /= w.sum()
    return np.convolve(np.pad(x, (r,), mode="edge"), w, mode="valid")[:len(x)]


def accumulate(traj: Traj) -> np.ndarray:
    """Per-frame positions, frame 0 at the origin: (N+1, 3)."""
    pos = np.zeros((len(traj.deltas) + 1, 3))
    pos[1:] = np.cumsum(traj.deltas, axis=0)
    return pos


def correlate(a: np.ndarray, b: np.ndarray) -> float:
    """Finite-safe Pearson r; NaN where either side is missing or flat."""
    m = np.isfinite(a) * np.isfinite(b)
    if m.sum() < 8:
        return float("nan")
    x, y = a[m], b[m]
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def render_constant(spec: ClipSpec, dst: str | Path,
                   preset: str = "ultrafast") -> Path:
    """Render with a fixed framing pitch: the no-correction baseline."""
    from steadycut.render.reframe import ControlPoint, render as _render
    _render(spec.source, dst, [ControlPoint(t=0.0, fov=spec.fov,
                                            pitch=spec.framing_pitch)],
            start=spec.start, duration=spec.duration, size=spec.size,
            preset=preset,
            geometry=spec.profile.lens if spec.profile is not None else None)
    return Path(dst)


def track_far(video: str | Path) -> Traj:
    """Far-field ORB+RANSAC trajectory of a rendered clip, in memory."""
    from steadycut.stabilization.far_track import track
    deltas, inl = track(str(video))
    return Traj(deltas=np.asarray(deltas, dtype=float),
                inliers=np.asarray(inl, dtype=int))


def gyro_pitch_per_frame(source: str | Path, t0: float, n: int,
                         fps: float | None = None,
                         axis_map=None) -> np.ndarray:
    """Mean gyro pitch (deg per frame) in each frame interval from t0.

    The pitch column comes from the camera profile's axis map (X4 default).
    fps is probed from the source when not given -- deriving it from a module
    constant silently mis-bins 60 fps files.
    """
    from steadycut.ingest.telemetry import read_telemetry
    if fps is None:
        from steadycut.render.reframe import source_fps
        fps = source_fps(str(source))
    if axis_map is None:
        raise ValueError(
            "no measured axis map: pass the one the camera registry loaded "
            "for this body (registry.identify returns it); an unknown body is "
            "measured from its own footage on first use")
    col = axis_map.pitch[0]
    imu = read_telemetry(str(source))
    out = []
    for i in range(n):
        tc = t0 + (i + 0.5) / fps
        m = (imu.time_s >= tc - 0.5 / fps) * (imu.time_s < tc + 0.5 / fps)
        out.append(imu.gyro[m, col].mean() / fps if m.sum() else np.nan)
    return np.array(out)


def measure_sync(spec: ClipSpec, lags_ms=range(-200, 61, 10),
                workdir: str | Path | None = None) -> SyncResult:
    """Gyro-stamp offset for one file+window, returned -- not written.

    Renders a constant-framing baseline to a temp dir (kept only if workdir
    is given), tracks the far field, and correlates vertical image motion
    against gyro pitch across lags. Peak |r| wins; r at 0 is reported so a
    caller can tell unobservable from aligned.
    """
    from steadycut.ingest.telemetry import read_telemetry  # noqa: F401 (warms cache)
    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp())
    tmp.mkdir(parents=True, exist_ok=True)
    video = tmp / "sync_raw.mp4"
    render_constant(spec, video)
    traj = track_far(video)
    _, ppdy = plate_scale(spec.fov)
    dy = traj.deltas[:, 1] / ppdy  # deg per frame
    curve = []
    axis_map = spec.profile.axis_map if spec.profile is not None else None
    for lag in lags_ms:
        pr = gyro_pitch_per_frame(spec.source, spec.start + lag / 1000.0,
                                  len(dy), axis_map=axis_map)
        curve.append((float(lag), correlate(pr, dy)))
    best = max(curve, key=lambda kv: abs(kv[1]))
    r0 = dict(curve).get(0, float("nan"))
    return SyncResult(lag_ms=best[0], r_best=best[1], r_at_zero=r0,
                      curve=tuple(curve),
                      inliers_med=float(np.median(traj.inliers)))


def build_gyro_path(spec: ClipSpec, stamp_offset_ms: float = 0.0,
                    corner_hz: float = 0.15, drift_ppm: float = 0.0,
                    drift_ref_s: float = 0.0):
    """Stabilisation path for a clip: gyro correction + framing, thinned.

    corner_hz 0.15 is the MEASURED production default (see correction_axes);
    it lives here as well as on the CLI flag so that a tool calling this
    directly evaluates production rather than a configuration nobody runs.

    stamp_offset_ms shifts the gyro stamps (the per-file sync calibration):
    path value at video time V uses gyro stamped V-offset... precisely, the
    stamps move by +offset, so V reads the sample stamped V-offset. A measured
    lag of -80ms (gyro(t-80) explains frame t) therefore wants +80 here.

    drift_ppm handles IMU/video clock drift (measured -100ppm on one file:
    lag grows -30ms per 300s). The stamp shift becomes
    offset + drift_ppm*1e-6*(V - drift_ref_s)*1000 ms. Fit per file with
    fit_drift(); do not copy across files.

    STATUS: UNVALIDATED -- a direct offset sweep at the far end of the file
    (+60/+80/+100/+120ms) puts the minimum at +80..+100, indistinguishable
    from the +80 that wins 600s earlier, while the scan trend predicted +130
    (which scores worse than raw). So the scan trend does NOT transfer to the
    render; the true offset is constant within measurement noise (<=35ppm)
    and the trend's mechanism is unidentified. Do NOT pass nonzero drift_ppm
    until a render verifies it.
    """
    from steadycut.ingest.telemetry import read_telemetry
    from steadycut.stabilization import orientation as O
    from steadycut.stabilization.path import PathPoint, thin
    imu = read_telemetry(str(spec.source), profile=spec.profile)
    base = imu.time_s - spec.start
    shift = (stamp_offset_ms
             + drift_ppm * 1e-6 * (base + spec.start - drift_ref_s) * 1000.0)
    t = base + shift / 1000.0
    mask = (t >= -15) * (t <= spec.duration + 15)
    tw, gw = t[mask], imu.gyro[mask]
    tau = 1.0 / (2.0 * math.pi * corner_hz)
    axis_map = spec.profile.axis_map if spec.profile is not None else None
    qy, qp, qr = O.correction_axes(tw, gw, tau, axis_map)
    from steadycut.render.reframe import ControlPoint
    from steadycut.stabilization.path import PathPoint, thin
    raw = [PathPoint(t=float(tt), yaw=float(y),
                     pitch=float(p) + spec.framing_pitch, roll=float(r),
                     fov=spec.fov)
           for tt, y, p, r in zip(tw, qy, qp, qr)]
    # PathPoint and ControlPoint share the shape; thin selects by time only,
    # and the render boundary takes ControlPoint. Unifying the two types is
    # recorded debt, not done here.
    return [ControlPoint(t=q.t, yaw=q.yaw, pitch=q.pitch, roll=q.roll,
                         fov=q.fov) for q in thin(raw, 0.05)]


def fit_drift(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Fit lag(t) scan points, return STAMP-SHIFT parameters for build_gyro_path.

    points: (file_time_s, lag_ms) from sync scans spread over the file.
    The scan reports lags (gyro(t+lag) explains frame t); the path builder
    takes shifts (add to stamps), which are the negation. This returns the
    negated fit directly: (shift0_ms at ref, shift_ppm), ref = first point.
    Needs 2+ spread points; clustered points make the slope noise.
    """
    if len(points) < 2:
        raise ValueError("drift fit needs 2+ spread sync points")
    ref = points[0][0]
    xs = np.array([p[0] - ref for p in points])
    ys = np.array([p[1] for p in points])
    a = np.stack([np.ones_like(xs), xs], axis=1)
    sol, *_ = np.linalg.lstsq(a, ys, rcond=None)
    return float(-sol[0]), float(-sol[1] * 1e6 / 1000.0)


def render_path(spec: ClipSpec, points, dst: str | Path,
               preset: str = "ultrafast") -> Path:
    """Render a clip along a correction path."""
    from steadycut.render.reframe import render as _render
    _render(spec.source, dst, points, start=spec.start,
            duration=spec.duration, size=spec.size, preset=preset,
            geometry=spec.profile.lens if spec.profile is not None else None)
    return Path(dst)


def far_field_motion(video: str | Path,
                     render_size: tuple[int, int] = (960, 720),
                     track_size: tuple[int, int] = (1280, 960)) -> np.ndarray:
    """Per-frame FAR-field vertical motion at RENDER scale, in pixels.

    The far field can only move when the camera ROTATES -- there is no parallax
    at a hundred metres -- so this is the landscape number, and the only one a
    rotation correction can move at all. Measured with the ORB + RANSAC
    estimator: on foliage phase correlation reads r ~= -0.18 where this reads
    ~= -0.9, which is why the estimator matters and not just the region.
    """
    traj = track_far(video)
    return traj.deltas[:, 1] * (render_size[1] / track_size[1])


def far_field_score(video: str | Path, **kw) -> dict:
    """RMS and frame-to-frame jitter of the far field, in render pixels.

    Two numbers because they are two faults: RMS is how far the landscape
    moves (sway and drift), jitter is how much it shakes between frames. A
    correction can remove one without the other, and a single blended "bounce"
    number cannot say which happened.
    """
    dy = far_field_motion(video, **kw)
    if dy.size < 2:
        return {"rms_px": float("nan"), "jitter_px": float("nan"),
                "frames": int(dy.size)}
    return {"rms_px": float(np.std(dy)),
            "jitter_px": float(np.std(np.diff(dy))),
            "frames": int(dy.size)}


def bounce_of_traj(traj: Traj) -> tuple[float, float]:
    """(bounce, y-mean) straight from a trajectory, no video needed."""
    dy = traj.deltas[:, 1]
    return float(np.diff(dy).std()), float(np.abs(dy).mean())


def residual_correction(traj: Traj, sigma: float = 4.0,
                        bound: float = 12.0,
                        render_size: tuple[int, int] = (960, 720),
                        track_size: tuple[int, int] = (1280, 960)) -> np.ndarray:
    """Bounded 2D residual correction per frame: (N, dx, dy, dtheta) at render
    scale. Smoothed residual minus residual, clipped to the honesty bound."""
    pos = accumulate(traj)
    sm = np.stack([gsmooth(pos[:, i], sigma) for i in range(3)], axis=1)
    corr = sm - pos
    sx = render_size[0] / track_size[0]
    corr[:, 0] *= sx
    corr[:, 1] *= render_size[1] / track_size[1]
    corr[:, :2] = np.clip(corr[:, :2], -bound, bound)
    return corr


def align_corrections(corr: np.ndarray, n_frames: int) -> np.ndarray:
    """One correction per frame, from a per-interval trajectory.

    Correction row j belongs to frame j, so the index pairing is the property
    to preserve -- never the length for its own sake.

    The trajectory is counted through cv2 and the pixels come from ffmpeg's
    rawvideo pipe, and on some files those two disagree by one frame at the END
    (measured: six files agreed exactly, others differ). The frames both agree
    on keep their own row; a surplus frame gets no correction, and a surplus
    correction row is dropped. Padding or trimming at the head instead would
    shift every correction one frame late against its frame, and a correction
    one frame late injects motion rather than removing it. Any mismatch beyond
    one frame is a real inconsistency and raises.
    """
    if len(corr) == n_frames:
        return corr
    if len(corr) == n_frames - 1:
        return np.vstack([corr, np.zeros((1, corr.shape[1]))])
    if len(corr) == n_frames + 1:
        return corr[:-1]
    raise ValueError(f"frames {n_frames} != corrections {len(corr)}")


def warp_video(src_video: str | Path, dst_video: str | Path,
               corr: np.ndarray, size: tuple[int, int] = (960, 720),
               zoom: float = 1.05, fps: float | None = None) -> Path:
    """Apply a per-frame (dx, dy, dtheta) correction with a margin zoom.

    The zoom hides the sampling borders large shifts expose (5% covers ~24px
    horizontal / ~18px vertical at 960x720); it costs field of view, stated
    here rather than hidden. fps is probed from the source when not given.

    One correction per frame is an invariant rather than a convenience: the
    tracker emits one row per frame interval so that row j belongs to frame j,
    and the length check is what caught that being violated (a frame that could
    not be matched used to drop its row, shifting every later correction one
    frame late).
    """
    w, h = size
    if fps is None:
        from steadycut.render.reframe import source_fps
        fps = source_fps(str(src_video))
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src_video), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    n = len(raw) // (w * h * 3)
    fr = np.frombuffer(raw[:n * w * h * 3], np.uint8).reshape(n, h, w, 3)
    corr = align_corrections(corr, n)
    enc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-framerate", f"{fps:.6f}".rstrip("0").rstrip(".") or "30",
         "-i", "-", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
         str(dst_video)], stdin=subprocess.PIPE)
    assert enc.stdin is not None
    cx, cy = w / 2, h / 2
    for i in range(n):
        dx, dy, th = (float(corr[i, 0]), float(corr[i, 1]),
                      float(corr[i, 2]))
        m = cv2.getRotationMatrix2D((cx, cy), th, zoom)
        m[0, 2] += dx
        m[1, 2] += dy
        enc.stdin.write(cv2.warpAffine(fr[i], m, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)
                        .tobytes())
    enc.stdin.close()
    enc.wait()
    if enc.returncode != 0:
        raise RuntimeError(f"encode failed rc={enc.returncode}")
    return Path(dst_video)


def hybrid_correct(src_video: str | Path, dst_video: str | Path,
                   sigma: float = 4.0, bound: float = 12.0,
                   zoom: float = 1.05) -> Traj:
    """Bounded 2D residual pass over an already gyro-stabilised clip.

    sigma 4.0 is measured, not chosen by taste: on four windows across two
    clips it gave the best far-field amplitude in every arm of a sweep
    (sigma 8/4/2 against corner 0.5/0.15/0.05), while sigma 2 started trading
    amplitude away for jitter it did not win back. Tracks the far field,
    smooths the residual, warps by the clipped difference. Returns the
    residual trajectory (in memory) the warp removed.
    """
    traj = track_far(src_video)
    corr = residual_correction(traj, sigma=sigma, bound=bound)
    warp_video(src_video, dst_video, corr, zoom=zoom)
    return traj


def stabilize_image_only(source: str | Path, output: str | Path,
                         start: float, duration: float, *,
                         sigma: float = 8.0, bound: float = 12.0,
                         zoom: float = 1.05, preset: str = "ultrafast",
                         workdir: str | Path | None = None) -> dict:
    """Image-only stabilisation for footage with no usable telemetry.

    The bounded 2D residual pass IS the stabiliser here: cut the window
    (input seek, snapped to the frame grid), track the far field, warp by the
    smoothed-minus-raw difference with the margin zoom. No reframe, no gyro.
    Returns the raw and corrected far-bounce -- so the caller can state
    whether the pass helped on THIS window -- and the median inlier count as
    the tracking-strength certificate.
    """
    from steadycut.render.reframe import snap_to_frame, source_fps, source_size
    tmp = (Path(workdir) if workdir
           else Path(tempfile.mkdtemp(prefix="steadycut-l3-")))
    tmp.mkdir(parents=True, exist_ok=True)
    fps = source_fps(str(source))
    cut = tmp / "window.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-ss", f"{snap_to_frame(start, source):g}", "-t", f"{duration:g}",
         "-i", str(source), "-an",
         "-c:v", "libx264", "-preset", preset, "-crf", "18", str(cut)],
        check=True)
    traj = track_far(cut)
    corr = residual_correction(traj, sigma=sigma, bound=bound)
    warp_video(cut, output, corr, size=source_size(cut), zoom=zoom, fps=fps)
    fixed = track_far(output)
    raw_bounce, _ = bounce_of_traj(traj)
    fixed_bounce, _ = bounce_of_traj(fixed)
    # Keep the better of the two. Measured on real trail footage, from the middle
    # of three rides at three fractions of each: the pass made the far-field
    # residual WORSE on 9 of 9 windows, by 3 to 51 percent. The pass is tuned
    # against a clean 5 Hz synthetic jitter, and singletrack is not that: parallax
    # between canopy and trail, a rider in the near field, and foliage that moves
    # for reasons unrelated to the camera.
    #
    # So when it does not help, deliver the raw cut rather than the degraded one.
    # A warning is not a fallback: the previous behaviour warned and still shipped
    # the worse file, so the default output of this mode was the bad one.
    kept = "corrected"
    if fixed_bounce >= raw_bounce:
        import shutil
        shutil.copyfile(cut, output)
        kept = "raw"
    return {"output": Path(output),
            "raw_bounce": raw_bounce,
            "corrected_bounce": fixed_bounce,
            "kept": kept,
            "inliers_med": float(np.median(traj.inliers))}


@dataclass
class ClipResult:
    """Everything a finished clip leaves behind, numerically."""
    video: Path
    sync: SyncResult | None = None
    residual_traj: Traj | None = None
    far_bounce: float = float("nan")
    far_ymean: float = float("nan")
    notes: list = field(default_factory=list)
