"""Key frames for choosing a field of view, in both aspect ratios.

The user wants fragments where the handlebars sit at the bottom of the frame, for
both vertical (shorts) and widescreen viewing, and wants the field of view explored
with still frames before committing to video.

The pitch is DERIVED rather than swept. The bar sits about 44.5 degrees below the
horizon in this mount's frame, measured once: at h_fov 100 (v_fov 56.25) a pitch of
-22 placed the bar about 90 percent of the way down the frame. Requiring that same
fraction for any other vertical field of view gives

    pitch = -44.5 + 0.8 * (v_fov / 2)

which reproduces the measured -22 exactly at v_fov 56.25, so the rule is checked
against the one known-good value rather than assumed. That is the point: one frame
and a geometry gives every other field of view's pitch, instead of a two-dimensional
sweep of field of view against pitch.

Aspect matters because v360 is given a HORIZONTAL field of view and the vertical is
derived from the frame shape. For a portrait frame the same vertical framing implies
a much narrower horizontal view, which is exactly what makes a short look different
rather than merely smaller.
"""
import os
import subprocess
import tempfile
from pathlib import Path

from steadycut.stabilization import path as P
from steadycut.ingest.footage import clip
from steadycut.render.reframe import render

# The recording is an argument, never a constant: see main(). MOMENT is the
# moment this study was pointed at (the top 2 s by accel deviation on the ride
# it was written for), and it comes in the same way.
MOMENT = float(os.environ.get("STEADYCUT_MOMENT", 0.0))
# The bar's elevation below the horizon is NOT a constant of the mount: it moves
# with the rider's posture. Measured -44.5 degrees at t=300 s, where h_fov 100 with
# pitch -22 put the bar 90 percent down the frame; reading the render at t=724.5 s
# the bar sits near the middle at that same pitch, i.e. about -28 degrees, because
# the rider is more upright at that moment. So it is a parameter, measured per
# moment, rather than a constant baked into the rule.
BAR_ELEVATION = float(os.environ.get("STEADYCUT_BAR_ELEV", -44.5))
BAR_FRACTION = float(os.environ.get("STEADYCUT_BAR_FRAC", 0.9))

# (label, width, height, vertical field of view to show)
CASES = [
    ("widescreen", 960, 540, [("narrow", 45.0), ("medium", 56.25), ("wide", 67.5)]),
    ("vertical",   540, 960, [("narrow", 56.0), ("medium", 70.0), ("wide", 85.0)]),
]


def pitch_for(v_fov):
    return BAR_ELEVATION + (2.0 * BAR_FRACTION - 1.0) * (v_fov / 2.0)


def main(source=None, moment=None):
    """Usage: python -m steadycut.framing.keyframe_fov <file.insv> [moment_s]

    `source` is required: this studies the framing of one moment of one
    recording, and which recording that is belongs to the caller.
    """
    import sys
    if source is None:
        if len(sys.argv) < 2:
            print("usage: keyframe_fov.py <file.insv> [moment_s]",
                  file=sys.stderr)
            return 2
        source = sys.argv[1]
    moment = float(sys.argv[2]) if (moment is None and len(sys.argv) > 2) \
        else (moment if moment is not None else MOMENT)
    tmp = Path(tempfile.mkdtemp())
    for name, w, h, options in CASES:
        tiles = []
        for label, v_fov in options:
            h_fov = v_fov * w / h                  # renderer derives v from h
            pitch = pitch_for(v_fov)
            out = tmp / f"{name}_{label}.mp4"
            render(S, out, [P.PathPoint(t=0.0, fov=h_fov, yaw=0.0, pitch=pitch)],
                   start=MOMENT, duration=0.1, size=(w, h), preset="ultrafast")
            png = tmp / f"{name}_{label}.png"
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(out),
                 "-vf", f"drawtext=text='{label} h_fov {h_fov:.0f} v_fov {v_fov:.0f}"
                        f" pitch {pitch:.1f}':x=8:y=8:fontsize=20:fontcolor=white:"
                        f"box=1:boxcolor=black@0.7,"
                        f"drawgrid=w={w//10}:h={h//10}:t=1:c=red@0.25",
                 "-frames:v", "1", str(png)], check=True)
            tiles.append(png)
            print(f"{name:11s} {label:7s} h_fov {h_fov:6.1f}  v_fov {v_fov:5.1f}"
                  f"  pitch {pitch:6.1f}")
        files = [str(p) for p in tiles]
        args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        for f in files:
            args += ["-i", f]
        args += ["-filter_complex", f"hstack=inputs={len(files)}", "-frames:v", "1",
                 "-update", "1", f"/source/upstream/steadycut/fov_{name}.png"]
        subprocess.run(args, check=True)
        print(f"wrote fov_{name}.png")


if __name__ == "__main__":
    main()
