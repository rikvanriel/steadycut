# SteadyCut

Free and open-source action-cam editing for Linux: gyro-stabilised reframing
of 360° and single-lens action-camera footage.

A virtual camera follows the rider's intent — the landscape stays level and
the subject stays framed — using the camera's own embedded motion telemetry.
No vendor SDK: the pipeline reads camera-original files directly and renders
through ffmpeg.

## Status

Alpha. Works end to end on Insta360 X4 footage (4:3 and 9:16 reframed clips
with a numeric certificate); the camera-abstraction layer supports adding
other cameras (GoPro, Insta360 X5, DJI, SJCAM-class) as measured profiles.
Single-lens "image-only" stabilisation exists but its parameters are not yet
tuned for raw footage.

## Install (development)

    git clone <repo>
    cd steadycut
    python3.11 -m venv .venv && . .venv/bin/activate
    uv pip install -e .        # or: pip install -e .
    # ffmpeg with an HEVC decoder is required (e.g. RPM Fusion's full build)

## Use

    steadycut clip.insv --start 30 --dur 3                 # 4:3 landscape
    steadycut clip.insv --start 30 --dur 3 --tall          # 9:16
    steadycut clip.insv --start 30 --dur 3 --corner 0.15   # steadier, lags gaze
    steadycut clip.mp4  --start 30 --dur 3 --image-only    # no-telemetry mode

Every run writes a `.cert.json` certificate: measured framing, sync offset,
and raw-vs-corrected far-field bounce, so a render can be judged by numbers,
not by eye alone.

## Layout

    steadycut/           # the installable package
      ingest/            #   telemetry (per-format), trailer, camera profiles
      render/            #   the ffmpeg v360 filtergraph
      stabilization/     #   quaternion orientation -> camera path
      framing/           #   framing measurements and policies
      metrics/           #   far-field tracking, shake/rotation metrics
      calibration/       #   per-camera calibration and first-use discovery
      harness/           #   the policy comparison harness
      core/              #   pipeline composition
      steadycut.py       #   the entry point

## Cameras are measured, never assumed

The code carries only what is model-wide and knowable in advance: lens
geometry, projection, stack order, capabilities, framing ceilings. Everything
that belongs to a particular camera BODY — above all the gyro axis map, which
is a hardware convention that no datasheet states — is measured from that
body's own footage and stored per serial number in the user's config:

    ~/.config/steadycut/cameras.json

The first time a body is used, `steadycut` measures its axis map from the file
being rendered (correlating each gyro column against the image motion it
explains) and stores the result, so this happens once per camera rather than
once per clip. The measurement records the firmware it was taken under, and a
firmware update — which can move the axis convention — triggers one
re-measurement.

A measurement that cannot be trusted is refused rather than stored: a window
too smooth to correlate, or two channels claiming the same gyro column, fails
with the windows that were tried and what to do instead. Unsupported models
and formats refuse with instructions. Nothing is ever inherited from another
camera.

## Licence

GPL-3.0-or-later. The runtime dependencies are permissively licensed
(numpy BSD-3, opencv-python-headless Apache-2.0, telemetry-parser
MIT OR Apache-2.0) and ffmpeg is invoked as an external program, so no
third-party copyleft code is linked into this source.
