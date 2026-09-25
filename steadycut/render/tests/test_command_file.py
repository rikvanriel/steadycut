"""build_command_file: the sendcmd delivery that the renderer consumes.

Uncovered until now. Per-frame ABSOLUTE commands are the delivery that works
(interval-lerp is broken and must stay non-default); telemetry-built paths
carry negative times that ffmpeg rejects, so they must be dropped; a single
point must still emit every channel.
"""
from steadycut.render.reframe import ControlPoint, build_command_file


def test_per_frame_absolute_commands_span_the_clip() -> None:
    pts = [ControlPoint(t=0.0, yaw=0.0, pitch=-10.0, fov=100.0),
           ControlPoint(t=2.0, yaw=10.0, pitch=-10.0, fov=100.0)]
    txt = build_command_file(pts, fps=10.0)
    lines = txt.strip().splitlines()
    assert len(lines) == 21 * 4          # 21 frames x 4 channels
    yaws = [line for line in lines if " v360 yaw " in line]
    assert yaws[0].endswith(" yaw 0.0000;")
    assert yaws[-1].endswith(" yaw 10.0000;")


def test_negative_times_dropped_not_emitted() -> None:
    """Telemetry starts before the first video frame; a leading minus collides
    with the interval separator and ffmpeg rejects the WHOLE render."""
    pts = [ControlPoint(t=-1.0, yaw=5.0, fov=100.0),
           ControlPoint(t=0.0, yaw=0.0, fov=100.0)]
    txt = build_command_file(pts, fps=10.0)
    assert "-1.000000" not in txt
    assert "1.000000--" not in txt


def test_single_point_emits_every_channel() -> None:
    txt = build_command_file([ControlPoint(t=0.0, yaw=3.0, fov=90.0)], fps=10.0)
    for channel in ("yaw", "pitch", "roll", "h_fov"):
        assert f"v360 {channel} " in txt


def test_empty_path_refuses() -> None:
    try:
        build_command_file([])
    except ValueError:
        return
    raise AssertionError("empty path must refuse, not emit an empty file")
