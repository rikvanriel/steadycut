"""Evaluate camera-path policies against measured image motion.

A policy is a function (t, gyro, accel) -> camera path. Policies are therefore
swappable: a different orientation source, a different filtering choice or a
different framing rule can be compared without touching the renderer.
"""
from pathlib import Path
import numpy as np

from steadycut.stabilization import path as P
from steadycut.metrics import rotate
from steadycut.metrics import shake


FOV = 100.0


def policies() -> dict:
    """Named policies, each mapping telemetry to a camera path."""
    def raw(t, gyro, accel):
        return [P.PathPoint(t=0.0, fov=FOV)]

    def stab(t, gyro, accel):
        pts = P.build_path(t, gyro, None, cutoff_s=0.5)
        return P.thin(pts, 0.05)

    def level(t, gyro, accel):
        pts = P.build_path(t, gyro, accel, cutoff_s=1e6, level_roll=True)
        return P.thin(pts, 0.05)

    def level_neg(t, gyro, accel):
        pts = P.build_path(t, gyro, accel, cutoff_s=1e6, level_roll=True,
                           roll_gain=-1.0)
        return P.thin(pts, 0.05)

    def quat(t, gyro, accel):
        """Quaternion orientation, smoothed in orientation space, applied as a
        relative rotation. Filtering Euler axes independently is a documented
        anti-pattern.

        The time constant must be the one the pipeline actually uses, or this
        measures a configuration nobody runs. It is derived from the measured
        optimum corner frequency, not guessed: an
        exponential with time constant tau has a corner at 1/(2*pi*tau), so a
        time constant reused from a moving-average window cancels far too much.
        A previous 0.3 here was exactly that mistake, about 0.5 Hz, and made this
        policy look worse than raw while the production path improves.
        """
        from steadycut.stabilization import orientation
        tau = 1.0 / (2.0 * np.pi * 3.18)
        yaw, pitch, roll = orientation.correction_axes(t, gyro, tau)
        pts = [P.PathPoint(t=float(a), yaw=float(b), pitch=float(c),
                           roll=float(d), fov=FOV)
               for a, b, c, d in zip(t, yaw, pitch, roll)]
        return P.thin(pts, 0.05)

    def quat_neg(t, gyro, accel):
        from steadycut.stabilization import orientation
        tau = 1.0 / (2.0 * np.pi * 3.18)
        yaw, pitch, roll = orientation.correction_axes(t, gyro, tau)
        pts = [P.PathPoint(t=float(a), yaw=-float(b), pitch=-float(c),
                           roll=-float(d), fov=FOV)
               for a, b, c, d in zip(t, yaw, pitch, roll)]
        return P.thin(pts, 0.05)

    def stab_level(t, gyro, accel):
        pts = P.build_path(t, gyro, accel, cutoff_s=0.5, level_roll=True)
        return P.thin(pts, 0.05)

    def _one_channel(which, sign):
        """Drive ONE correction axis, so each is judged by its OWN metric.

        The rule from the tuning notes: with all three live, one bad channel swamps two
        good ones and the combined reading cannot locate the fault. This re-establishes
        the sign and scale per axis now that reset_rot=1 means commands actually survive,
        which they did not when the original per-channel numbers were measured.
        """
        def fn(t, gyro, accel, _w=which, _s=sign):
            from steadycut.stabilization import orientation
            tau = 1.0 / (2.0 * np.pi * 3.18)
            axes = dict(zip(("yaw", "pitch", "roll"),
                            orientation.correction_axes(t, gyro, tau)))
            zero = np.zeros_like(t)
            vals = {k: (_s * axes[k] if k == _w else zero)
                    for k in ("yaw", "pitch", "roll")}
            pts = [P.PathPoint(t=float(a), yaw=float(vals["yaw"][i]),
                               pitch=float(vals["pitch"][i]),
                               roll=float(vals["roll"][i]), fov=FOV)
                   for i, a in enumerate(t)]
            return P.thin(pts, 0.05)
        return fn

    pol = {"raw": raw, "stab": stab, "level": level, "level-1": level_neg,
           "quat": quat, "quat-neg": quat_neg, "stab+level": stab_level}
    for which in ("yaw", "pitch", "roll"):
        for sign, tag in ((1.0, ""), (-1.0, "-neg")):
            pol[f"{which}{tag}"] = _one_channel(which, sign)
    return pol


def metrics(video: Path) -> tuple:
    """Residual shake (translation) and residual roll (rotation)."""
    f = shake.frames(str(video))
    d = np.array([shake.shift(f[i], f[i + 1]) for i in range(len(f) - 1)])
    sh = float(np.linalg.norm(np.diff(d, axis=0), axis=1).std())
    rate, _ = rotate.rotation_series(str(video))
    good = rate[np.isfinite(rate)]
    return sh, float(good.std()) if good.size else float("nan")


def run(source: str, start: float, duration: float, outdir: Path,
        names=None) -> None:
    from steadycut.render.reframe import render
    from steadycut.ingest.telemetry import read_telemetry

    imu = read_telemetry(source)
    win = (imu.time_s >= start - 1.0) & (imu.time_s <= start + duration + 1.0)
    t = imu.time_s[win] - start
    gyro = imu.gyro[win]
    accel = imu.accel[win]

    chosen = policies()
    if names:
        chosen = {k: v for k, v in chosen.items() if k in names}

    print(f"{'policy':<12} {'shake':>8} {'roll std':>10}")
    for name, fn in chosen.items():
        video = outdir / f"eval_{name.replace('+', '_')}.mp4"
        path = fn(t, gyro, accel)
        render(source, video, path, start=start, duration=duration,
               size=(960, 540), preset="ultrafast")
        sh, roll = metrics(video)
        print(f"{name:<12} {sh:>8.3f} {roll:>10.4f}")


def axis_metrics(video: Path) -> tuple:
    """Residual motion PER AXIS: horizontal, vertical, rotation.

    Each correction channel exists to improve exactly one of these, so a channel must be
    judged by its own axis. `metrics()` above reports a combined translation figure and a
    rotation figure; that is the right summary for comparing whole policies, but it
    cannot attribute a fault to a channel.
    """
    f = shake.frames(str(video))
    d = np.array([shake.shift(f[i], f[i + 1]) for i in range(len(f) - 1)])
    rate, _ = rotate.rotation_series(str(video))
    good = rate[np.isfinite(rate)]
    return (float(np.diff(d[:, 0]).std()),
            float(np.diff(d[:, 1]).std()),
            float(good.std()) if good.size else float("nan"))


def per_channel(source: str, start: float, duration: float, outdir: Path) -> None:
    """Render each correction channel alone and report its OWN axis.

    Read it like this: yaw is judged by dx, pitch by dy, roll by rotation. The correct
    sign improves its axis and the opposite sign degrades it; if both signs read alike,
    the correction is not correlated with the target at all, which is a different fault
    from a wrong sign.
    """
    from steadycut.render.reframe import render
    from steadycut.ingest.telemetry import read_telemetry

    imu = read_telemetry(source)
    win = (imu.time_s >= start - 1.0) & (imu.time_s <= start + duration + 1.0)
    t = imu.time_s[win] - start
    gyro, accel = imu.gyro[win], imu.accel[win]

    pol = policies()
    order = ["raw", "yaw", "yaw-neg", "pitch", "pitch-neg", "roll", "roll-neg"]
    print(f"{'channel':<11} {'dx std':>9} {'dy std':>9} {'rot std':>9}   judged by")
    judging = {"raw": "-", "yaw": "dx", "yaw-neg": "dx", "pitch": "dy",
               "pitch-neg": "dy", "roll": "rot", "roll-neg": "rot"}
    for name in order:
        video = outdir / f"ch_{name.replace('-', '_')}.mp4"
        render(source, video, pol[name](t, gyro, accel), start=start,
               duration=duration, size=(960, 540), preset="ultrafast")
        dx, dy, rot = axis_metrics(video)
        print(f"{name:<11} {dx:>9.3f} {dy:>9.3f} {rot:>9.4f}   {judging[name]}")


def cutoff_sweep(source: str, start: float, duration: float, outdir: Path,
                 corners=(1.0, 2.0, 3.18, 5.0, 8.0)) -> None:
    """Sweep the correction corner frequency against residual motion per axis.

    The BAND is settled by the gyro rate spectrum (knee ~2 Hz, <3% of rate energy above
    10 Hz) and was never in question. What is in question is the OPTIMUM, because every
    previous sweep was rendered through a delivery that silently discarded the commands,
    so those numbers described a configuration the code did not produce.

    From the tuning notes: the cutoff is a shake-rejection FREQUENCY, not a smoothing
    time; the control-point spacing must stay finer than the shortest period being
    corrected; and both residual metrics must improve, since a policy that fixes one
    while wrecking the other is not an improvement.
    """
    from steadycut.render.reframe import render
    from steadycut.ingest.telemetry import read_telemetry
    from steadycut.stabilization import orientation

    imu = read_telemetry(source)
    win = (imu.time_s >= start - 1.0) & (imu.time_s <= start + duration + 1.0)
    t = imu.time_s[win] - start
    gyro, accel = imu.gyro[win], imu.accel[win]

    print("SWEEP| " + f"{'corner Hz':>10} {'dx std':>9} {'dy std':>9} {'rot std':>9}")
    raw = None
    for hz in (None,) + tuple(corners):
        if hz is None:
            path = [P.PathPoint(t=0.0, fov=FOV)]          # the no-change baseline
            label = "raw"
        else:
            tau = 1.0 / (2.0 * np.pi * hz)
            yaw, pitch, roll = orientation.correction_axes(t, gyro, tau)
            path = P.thin([P.PathPoint(t=float(a), yaw=float(b), pitch=float(c),
                                       roll=float(d), fov=FOV)
                           for a, b, c, d in zip(t, yaw, pitch, roll)], 0.05)
            label = f"{hz:.2f}"
        video = outdir / f"sweep_{label.replace('.', '_')}.mp4"
        render(source, video, path, start=start, duration=duration,
               size=(960, 540), preset="ultrafast")
        dx, dy, rot = axis_metrics(video)
        if hz is None:
            raw = (dx, dy, rot)
        print(f"SWEEP| {label:>10} {dx:>9.3f} {dy:>9.3f} {rot:>9.4f}")
    if raw:
        print(f"SWEEP| raw-baseline dx {raw[0]:.3f}  dy {raw[1]:.3f}  rot {raw[2]:.4f}")


def far_field_sweep(source: str, start: float, duration: float, outdir: Path,
                    corners=(2.0, 1.0, 0.5, 0.2, 0.1, 0.05)) -> None:
    """Sweep the corner against FAR-FIELD motion: the landscape, i.e. the trees.

    Why this is the right metric for "steady the landscape". For a far-field object the
    image motion is purely the CAMERA'S ROTATION - there is no parallax at a hundred
    metres - and a 360 camera can cancel rotation exactly. So far-field motion is the
    clean measure of how well the rotation channels are working, and it is NOT what the
    whole-frame metrics show: those are dominated by the near field (the bike), which
    moves mostly by parallax and which the rider has explicitly said may bounce.

    `build_path_quat` has no gain knob, so the correction is already complete ABOVE the
    corner; what is left in the trees is motion BELOW it. Lowering the corner therefore
    steadies the landscape further, and the rider's steer (follow the direction of
    travel, not every head turn) is exactly what permits going lower.

    Regions are parallax.py's own NEAR and FAR boxes, reused rather than redefined.
    """
    from steadycut.render.reframe import render
    from steadycut.ingest.telemetry import read_telemetry
    from steadycut.stabilization import orientation
    from steadycut.metrics import parallax

    imu = read_telemetry(source)
    win = (imu.time_s >= start - 1.0) & (imu.time_s <= start + duration + 1.0)
    t = imu.time_s[win] - start
    gyro = imu.gyro[win]

    print(f"FARFIELD| {'corner Hz':>10} {'far y-mean':>11} {'far bounce':>11}"
          f" {'near y-mean':>12}")

    def measure(video):
        f = parallax.frames(str(video), 1280, 960)
        fy, ny = [], []
        for i in range(len(f) - 1):
            fy.append(parallax.shift(parallax.crop(f[i], parallax.FAR),
                                     parallax.crop(f[i + 1], parallax.FAR))[1])
            ny.append(parallax.shift(parallax.crop(f[i], parallax.NEAR),
                                     parallax.crop(f[i + 1], parallax.NEAR))[1])
        fy, ny = np.array(fy), np.array(ny)
        return float(np.abs(fy).mean()), float(np.diff(fy).std()), float(np.abs(ny).mean())

    for hz in (None,) + tuple(corners):
        if hz is None:
            path = [P.PathPoint(t=0.0, fov=FOV)]
            label = "raw"
        else:
            tau = 1.0 / (2.0 * np.pi * hz)
            yaw, pitch, roll = orientation.correction_axes(t, gyro, tau)
            path = P.thin([P.PathPoint(t=float(a), yaw=float(b), pitch=float(c),
                                       roll=float(d), fov=FOV)
                           for a, b, c, d in zip(t, yaw, pitch, roll)], 0.05)
            label = f"{hz:.2f}"
        video = outdir / f"far_{label.replace('.', '_')}.mp4"
        render(source, video, path, start=start, duration=duration,
               size=(960, 540), preset="ultrafast")
        fmean, fbounce, nmean = measure(video)
        print(f"FARFIELD| {label:>10} {fmean:>11.3f} {fbounce:>11.3f} {nmean:>12.3f}")


def sync_scan(source: str, start: float, duration: float, outdir: Path,
              lags_ms=range(-200, 61, 10), framing_pitch=-12.0) -> float:
    """Measure the gyro-stamp offset for one file and window (Gap-2 instrument).

    Single implementation lives in steadycut.core.pipeline.measure_sync; this keeps
    the CLI output stable (SYNCSCAN| lines) while the composition lives in one
    place. Sign convention, framing dependence and the second-window rule are
    documented on measure_sync.
    """
    from steadycut.core.pipeline import ClipSpec, measure_sync
    res = measure_sync(ClipSpec(source=source, start=start, duration=duration,
                                framing_pitch=framing_pitch),
                       lags_ms=lags_ms, workdir=outdir)
    print(f"SYNCSCAN| tracked steps, inliers med {res.inliers_med:.0f} "
          f"(framing pitch {framing_pitch})")
    print(f"SYNCSCAN| best lag {res.lag_ms:.0f}ms r={res.r_best:+.3f}  "
          f"(r at 0: {res.r_at_zero:+.3f})")
    print("SYNCSCAN| " + " ".join(f"{lag:+.0f}:{r:+.2f}"
                                  for lag, r in res.curve))
    return float(res.lag_ms)


def main() -> None:
    import argparse
    import tempfile

    ap = argparse.ArgumentParser(description="Compare camera-path policies")
    ap.add_argument("source")
    ap.add_argument("--start", type=float, default=400.0)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--per-channel", action="store_true",
                    help="drive each correction channel alone: the sign/scale re-check")
    ap.add_argument("--cutoff-sweep", action="store_true",
                    help="sweep the correction corner frequency against residual motion")
    ap.add_argument("--far-field-sweep", action="store_true",
                    help="sweep the corner against FAR-FIELD (landscape) motion")
    ap.add_argument("--sync-scan", action="store_true",
                    help="measure the gyro-stamp offset for this file+window")
    a = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        if a.per_channel:
            per_channel(a.source, a.start, a.duration, Path(tmp))
        elif a.cutoff_sweep:
            cutoff_sweep(a.source, a.start, a.duration, Path(tmp))
        elif a.far_field_sweep:
            far_field_sweep(a.source, a.start, a.duration, Path(tmp))
        elif a.sync_scan:
            sync_scan(a.source, a.start, a.duration, Path(tmp))
        else:
            run(a.source, a.start, a.duration, Path(tmp), a.only)


if __name__ == "__main__":
    main()
