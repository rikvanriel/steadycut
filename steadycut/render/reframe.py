"""Render a reframed flat video from a camera path.

The path is a sparse list of control points -- time, yaw, pitch, roll, fov.
`sendcmd` drives `v360`'s runtime parameters, and the path is resampled to one
absolute command per frame before being written out.

Expressions are deliberately not used.  `lerp(from, to, TI)` does not deliver
the intended value: with TI as interval seconds it overshoots, and normalising
by the interval length still disagreed with the equivalent constant orientation
by 35.5 deg at the midpoint of a 0->10 deg ramp.  Absolute commands are
delivered faithfully, so resolution comes from resampling rather than from
interpolation inside the filter.

Two details are easy to get wrong.  The commands must come from a file: passing
them inline through -vf means the shell mangles the quoting.  And `sendcmd` has
to sit before `v360` in the chain, since it forwards commands to the filter that
follows it.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

FPS = 29.97

_FPS_CACHE: dict = {}


def source_fps(source) -> float:
    """Exact frame rate of the source, cached.

    Probed rather than assumed: 30000/1001 is not 29.97, and at these clip
    lengths the difference is small but the frame grid it defines is what the
    seek has to land on.
    """
    key = str(source)
    if key not in _FPS_CACHE:
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=r_frame_rate",
                 "-of", "default=noprint_wrappers=1", str(source)],
                capture_output=True, text=True, check=True).stdout
            value = out.strip().split("=")[-1]
            num, _, den = value.partition("/")
            _FPS_CACHE[key] = float(num) / float(den) if den else float(num)
        except Exception:
            _FPS_CACHE[key] = FPS
    return _FPS_CACHE[key]


def snap_to_frame(t: float, source) -> float:
    """Nearest frame boundary at or before `t`, on the source's own grid."""
    fps = source_fps(source)
    return math.floor(t * fps + 1e-9) / fps


_SIZE_CACHE: dict = {}


def source_size(source) -> tuple[int, int]:
    """Frame size of the first video stream, cached.

    The image-only path decodes rawvideo and reshapes by the frame's own
    (w, h); assuming the render size instead breaks on any other geometry --
    a truncated reshape that fails far from the cause.
    """
    key = str(source)
    if key not in _SIZE_CACHE:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0:s=x", str(source)],
            capture_output=True, text=True, check=True).stdout
        w, _, h = out.strip().partition("x")
        _SIZE_CACHE[key] = (int(w), int(h))
    return _SIZE_CACHE[key]


# v360's id_fov is the DIAGONAL field of view, and a circular fisheye is
# inscribed in a square frame, so the frame diagonal spans sqrt(2) times the
# angle the circle does. Computed in ONE place, here.
DIAGONAL_FOV = math.sqrt(2.0)
X4_LENS_FOV_DEG = 181.37   # legacy default: the X4 makernotes mean

STACK_ORDERS = {
    "rear_first": "[0:v:1][0:v:0]hstack=inputs=2,",
    "front_first": "[0:v:0][0:v:1]hstack=inputs=2,",
}


def resolve_input(geometry=None, input_projection=None, input_fov=None,
                  dual_stream=None, travel_yaw=None):
    """Resolve the input stage of the reproject filtergraph.

    Priority: an explicit kwarg wins over the camera profile, and the X4-era
    defaults stand when neither is given (so old callers render unchanged).
    Returns (projection, id_fov, dual_stream, travel_yaw, stack_order) where
    stack_order is a key of STACK_ORDERS or "single".

    Every returned value must end up IN the filtergraph -- a parameter that
    appears only in a signature does nothing (the input_fov lesson), which is
    what test_geometry asserts.
    """
    if geometry is not None:
        proj = (input_projection if input_projection is not None
                else geometry.projection)
        fov = (input_fov if input_fov is not None
               else DIAGONAL_FOV * geometry.lens_fov_deg)
        dual = (dual_stream if dual_stream is not None
                else geometry.lenses == 2)
        tyaw = travel_yaw if travel_yaw is not None else geometry.travel_yaw
        order = geometry.stack_order if dual else "single"
    else:
        proj = input_projection if input_projection is not None else "dfisheye"
        fov = (input_fov if input_fov is not None
               else DIAGONAL_FOV * X4_LENS_FOV_DEG)
        dual = dual_stream if dual_stream is not None else True
        tyaw = travel_yaw if travel_yaw is not None else 0.0
        order = "rear_first" if dual else "single"
    return proj, fov, dual, tyaw, order


@dataclass(frozen=True)
class ControlPoint:
    """One knot of the camera path.  Angles in degrees, time in seconds."""

    t: float
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    fov: float = 100.0


def build_command_file(path: list[ControlPoint], per_frame: bool = True,
                       fps: float = 29.97) -> str:
    """Render control points as sendcmd commands.

    Two delivery modes, and the default matters:

    `per_frame=True` (the default) emits an ABSOLUTE command at frame cadence. This is
    the one that works.

    `per_frame=False` emits one interval per consecutive pair with `lerp(from, to, TI)`.
    **This mode is broken and must not be used.** Measured on a 5 s clip whose path swung
    only 1.65 / 10.05 / 2.99 degrees in yaw / pitch / roll over the whole window: the
    interval-lerp file spun the view about 150 degrees, swinging from looking straight
    down at the rider's helmet to looking up at the forest canopy, while the identical
    path delivered as absolute per-frame commands held the framing steady across the same
    three moments. So the fault is the delivery, not the path, the renderer or the stitch.
    `TI` evidently does not behave as an interval position across chained intervals. The
    earlier note here recorded that switching to absolute commands "made stabilisation
    worse" and reverted it - that comparison was made against this broken baseline and is
    therefore meaningless.

    One consequence worth stating: every multi-point render made while this mode was in
    use is suspect, including the stabilisation policy comparisons. Re-measure them on
    the per-frame delivery before trusting any of those numbers.
    """
    if not path:
        raise ValueError("camera path is empty")

    points = sorted(path, key=lambda p: p.t)

    # Telemetry starts slightly before the first video frame, so a path built
    # straight from it carries negative times.  Emitting those produces
    # intervals like "-1.000000--0.950000", and the leading minus collides
    # with the interval separator, which ffmpeg rejects outright.
    points = [p for p in points if p.t >= 0.0]
    if not points:
        raise ValueError("camera path has no control point at or after t=0")

    if len(points) == 1:
        only = points[0]
        return (
            f"{only.t:.6f} v360 yaw {only.yaw:.4f};\n"
            f"{only.t:.6f} v360 pitch {only.pitch:.4f};\n"
            f"{only.t:.6f} v360 roll {only.roll:.4f};\n"
            f"{only.t:.6f} v360 h_fov {only.fov:.4f};\n"
        )

    ts = np.array([p.t for p in points], dtype=float)
    if per_frame:
        lines = []
        times = np.arange(0.0, ts[-1] + 1e-9, 1.0 / fps)
        for name, attr in (("yaw", "yaw"), ("pitch", "pitch"),
                           ("roll", "roll"), ("h_fov", "fov")):
            vals = np.interp(times, ts,
                             np.array([getattr(p, attr) for p in points], dtype=float))
            lines.extend(f"{tt:.6f} v360 {name} {vv:.4f};" for tt, vv in zip(times, vals))
        return "\n".join(lines) + "\n"

    lines = []
    for start, end in zip(points, points[1:]):
        if end.t <= start.t:
            continue
        span = f"{start.t:.6f}-{end.t:.6f}"
        for name, a, b in (
            ("yaw", start.yaw, end.yaw),
            ("pitch", start.pitch, end.pitch),
            ("roll", start.roll, end.roll),
            ("h_fov", start.fov, end.fov),
        ):
            lines.append(f"{span} [expr] v360 {name} 'lerp({a:.4f},{b:.4f},TI)';")
    return "\n".join(lines) + "\n"


def render(
    source: str | Path,
    output: str | Path,
    path: list[ControlPoint],
    size: tuple[int, int] = (1920, 1080),
    input_projection: str | None = None,
    input_fov: float | None = None,
    crf: int = 18,
    preset: str = "medium",
    duration: float | None = None,
    start: float | None = None,
    dual_stream: bool | None = None,
    travel_yaw: float | None = None,
    sphere_size: tuple[int, int] = (2048, 1024),
    geometry=None,
    dry_run: bool = False,
) -> list[str]:
    """Reproject `source` along `path` and encode to `output`.

    The input stage (projection, id_fov, stream stacking, mount yaw) comes
    from `geometry` -- the camera profile's LensGeometry -- with explicit
    kwargs overriding it and the X4-era defaults standing in for callers that
    pass neither. See resolve_input.

    Camera-original `.insv` keeps the two fisheyes in separate video tracks,
    so they must be stacked side by side before `v360` sees them; only
    pre-stitched files carry both in one frame (dual_stream False).

    `travel_yaw` is where the direction of travel sits in the sphere, in v360
    coordinates, for a camera NOT mounted the usual way round. It defaults to 0
    because the fisheye stack order already puts travel at yaw 0 for a normally
    mounted camera. Set it to 180 for a camera mounted the other way; the sphere
    is then reoriented so that framing and tilt behave the same either way. See
    the note in the body for why the sphere is rotated rather than the view.

    Returns the ffmpeg argument list, so a caller can inspect or re-run the
    exact command.  With dry_run set nothing is executed.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")

    source, output = Path(source), Path(output)
    width, height = size
    proj, fov, dual, tyaw, order = resolve_input(
        geometry, input_projection, input_fov, dual_stream, travel_yaw)

    # Snap the seek to the source's frame grid and re-reference the path to it.
    #
    # -ss is an INPUT seek, so the first output frame is the one CONTAINING
    # `start`, while a path built from telemetry has t=0 at `start` itself. The
    # gap is the sub-frame phase of `start`, and it is a real misalignment of the
    # correction against the video: measured on far-field motion, a start with
    # phase 0.287 of a frame gave a correction WORSE than no correction at all,
    # while a start with phase 0.006 gave 32 percent better. Snapping makes the
    # alignment deterministic, and shifting the path by the same amount keeps the
    # correction referenced to the frame that actually comes out first.
    if start is not None:
        snapped = snap_to_frame(start, source)
        if snapped != start:
            delta = start - snapped
            path = [replace(p, t=p.t + delta) for p in path]
            start = snapped

    # The static filter options need a point inside the clip, since a path
    # built from telemetry can begin slightly before the first frame.
    start_points = sorted((p for p in path if p.t >= 0.0), key=lambda p: p.t)
    if not start_points:
        raise ValueError("camera path has no control point at or after t=0")
    first = start_points[0]

    commands = build_command_file(path, fps=source_fps(str(source)))
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".cmd", delete=False, encoding="utf-8"
    )
    with handle:
        handle.write(commands)
    command_file = Path(handle.name)

    # sendcmd feeds the filter downstream of it, so it must precede v360.
    #
    # The input fisheye field of view is passed through as id_fov. Each lens
    # sees MORE than 180 degrees, so their images overlap; leaving v360 at its
    # 180 degree default draws that overlap twice and leaves a seam and a dark
    # curved region where the two are joined.
    #
    # The default is 256.5 degrees, which is sqrt(2) times the calibrated lens
    # field of view, and the sqrt(2) is v360's convention rather than a fudge:
    # id_fov is the DIAGONAL field of view, and the fisheye circle is inscribed
    # in a square frame, so the frame diagonal spans sqrt(2) times the angle the
    # circle does. The camera's own calibration reports per-lens fields of view of
    # 182.15 and 180.58 degrees (two independent calibration strings in the
    # makernotes agree to within 0.02 degrees), against a mean of 181.37.
    #
    # Confirmed by confirm_stitch_fov.py, which measures the black fraction of
    # the lower frame: 0.50 at 180 degrees, 0.14 at 205, 0.048 at 256.5. The old
    # value leaves half the lower frame black because the stitcher reaches past
    # the lens circle into the void around it; the visible artefact is the lens
    # edge itself, with its chromatic fringe. The two lenses need slightly
    # DIFFERENT values and v360 takes one, so some asymmetry remains.
    reproject = (
        f"sendcmd=f='{command_file}',"
        f"v360=input={proj}:output=flat:id_fov={fov}"
        f":h_fov={first.fov}:v_fov={first.fov * height / width}"
        f":w={width}:h={height}"
        f":yaw={first.yaw}:pitch={first.pitch}:roll={first.roll}"
        # reset_rot=1 is REQUIRED here, not cosmetic. v360's process_command()
        # zeroes yaw, pitch and roll on EVERY sendcmd command unless reset_rot is
        # set, so with the default a command file that changes more than one angle
        # per timestamp has each value wiped by the command that follows it, and
        # the result depends on which command lands last. Left at the default, a
        # pitch correction never survives alongside a yaw or roll one.
        f":reset_rot=1:interp=lanczos"
    )

    if tyaw:
        # Rotate the SPHERE so the direction of travel lands at yaw 0, rather
        # than offsetting the view. The view offset cannot work: if the camera is
        # mounted the other way round, travel sits at yaw 180, and that is exactly
        # where v360's pitch is degenerate, so offsetting the view would leave the
        # one direction we care about permanently unable to tilt. Reorienting the
        # sphere first means the framing yaw stays near 0 and tilt keeps working.
        #
        # Two passes rather than one, because a single v360 call cannot both
        # reorient the sphere and project from it. travel_yaw is 0 for a camera
        # mounted the usual way, so the single-pass path below is the normal one
        # and this costs nothing unless it is actually needed.
        stage_one = (
            f"v360=input={proj}:output=equirect:id_fov={fov}"
            f":w={sphere_size[0]}:h={sphere_size[1]}"
            f":yaw={-tyaw}:interp=lanczos"
        )
        stage_two = reproject.replace(f"input={proj}:", "input=equirect:")
        stage_two = stage_two.replace(f"id_fov={fov}", "id_fov=0")
        reproject = f"{stage_one}[s];[s]{stage_two}"

    if dual:
        # The two fisheye tracks sit in the same container at the same size;
        # placing them side by side reproduces what v360's dfisheye input
        # expects.  Audio is taken from the first stream only.
        # The two fisheye streams are stacked in the order the stitcher expects:
        # rear lens first, front lens second (the profile's stack_order). With
        # the source order (0 then 1) the sphere comes out rotated 180 degrees,
        # which puts the direction of travel at yaw 180 - exactly where v360's
        # pitch is degenerate, so the view cannot be tilted up or down at all.
        # Measured: pitch has a full effect at yaw 0 (mean pixel difference
        # 70.5), partial at 170 (39.3) and 179 (26.6), and NONE at exactly
        # +-180 (0.25). Swapping the inputs puts forward at yaw 0 and the
        # bicycle is visible when looking down.
        filtergraph = f"{STACK_ORDERS[order]}{reproject}[v]"
        args = ["ffmpeg", "-hide_banner", "-y"]
        if start is not None:
            args += ["-ss", f"{snap_to_frame(start, source):g}"]
        if duration is not None:
            args += ["-t", f"{duration:g}"]
        args += [
            "-i", str(source),
            "-filter_complex", filtergraph,
            "-map", "[v]",
            "-map", "0:a:0?",
        ]
    else:
        filtergraph = reproject
        args = ["ffmpeg", "-hide_banner", "-y"]
        if start is not None:
            args += ["-ss", f"{start:g}"]
        if duration is not None:
            args += ["-t", f"{duration:g}"]
        args += ["-i", str(source), "-vf", filtergraph]

    args += [
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        # Stereo is captured in the wearer's frame while the virtual camera
        # turns independently, so the audio is copied rather than rotated to
        # match a heading it never had.
        "-c:a", "aac",
        "-b:a", "192k",
        str(output),
    ]

    if dry_run:
        command_file.unlink(missing_ok=True)
        return args

    try:
        subprocess.run(args, check=True)
    finally:
        command_file.unlink(missing_ok=True)
    return args


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Reframe 360 footage along a camera path")
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--fov", type=float, default=100.0)
    parser.add_argument(
        "--sweep",
        type=float,
        default=None,
        metavar="DEGREES",
        help="demo path: yaw sweeps this far across the clip",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.sweep is not None:
        span = args.duration or 10.0
        path = [
            ControlPoint(t=0.0, yaw=-args.sweep / 2, fov=args.fov),
            ControlPoint(t=span, yaw=args.sweep / 2, fov=args.fov),
        ]
    else:
        path = [ControlPoint(t=0.0, fov=args.fov)]

    command = render(
        args.source,
        args.output,
        path,
        duration=args.duration,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(" ".join(command))


if __name__ == "__main__":
    _main()
