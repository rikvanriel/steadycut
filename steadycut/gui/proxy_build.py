"""Build ONE proxy segment in an isolated process (group-killable).

Invoked by steadycut.gui.proxy.default_build_fn, one interpreter per
segment: spawn/seek cost is the `a` in the measured wall, and a crashed
ffmpeg can never take the GUI thread with it. Atomic publish: render to
a temp name in the same dir, os.replace into place, sidecar written
LAST (a sidecar without a renamed mp4 never validates, so a crash
mid-publish reads as a miss, not a corrupt hit).

Qt-free, like the rest of the proxy package.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def build(args) -> int:
    from steadycut.job import build_proxy

    # Single source of truth is the frame window: times derive from it on
    # the source's rational fps, so the parent can never skew -ss vs span.
    num, _, den = args.fps.partition("/")
    fps = float(num) / float(den)
    start_s = (args.frame_start / fps) if fps else 0.0
    dur_s = (args.nframes / fps) if fps else 0.0
    out = Path(args.out)
    tmp = Path(args.tmp)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        build_proxy(args.source, start_s, dur_s, str(tmp),
                    float(args.pitch), size=(args.width, args.height),
                    nframes=args.nframes, video_only=True)
        os.replace(tmp, out)
        # Sidecar last: the listing-plus-sidecars index only trusts a
        # segment whose sidecar exists, so this order is the commit.
        sidecar = json.loads(args.sidecar_json)
        Path(args.sidecar).write_text(json.dumps(sidecar, indent=1))
    except Exception as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        print(f"segment build failed: {exc}", file=sys.stderr)
        return 1
    print(f"ok {out.name} {args.nframes}f")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build one proxy segment.")
    ap.add_argument("--source", required=True)
    ap.add_argument("--frame-start", type=int, required=True)
    ap.add_argument("--nframes", type=int, required=True)
    ap.add_argument("--fps", required=True,
                    help="source fps as num/den (rational, never rounded)")
    ap.add_argument("--pitch", type=float, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--sidecar-json", required=True)
    args = ap.parse_args(argv)
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
