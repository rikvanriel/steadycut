"""build_command_file: the sendcmd delivery that the renderer consumes.

Uncovered until now. Per-frame ABSOLUTE commands are the delivery that works
(interval-lerp is broken and must stay non-default); telemetry-built paths
carry negative times that ffmpeg rejects, so they must be dropped; a single
point must still emit every channel.
"""
from steadycut.render.reframe import (
    ControlPoint,
    build_command_file,
    lerp_bias_anchors,
    unwrap_deg_series,
    wrap_deg,
)


def _channel_values(txt, channel):
    """Emitted per-frame values for one channel, in file order."""
    vals = []
    for line in txt.strip().splitlines():
        if f" v360@view {channel} " in line:
            vals.append(float(line.rsplit(" ", 1)[-1].rstrip(";")))
    return vals


def _circ_step(a, b):
    """Short-way distance between two wrapped degree values."""
    return min(abs(b - a), 360.0 - abs(b - a))


def test_per_frame_absolute_commands_span_the_clip() -> None:
    pts = [ControlPoint(t=0.0, yaw=0.0, pitch=-10.0, fov=100.0),
           ControlPoint(t=2.0, yaw=10.0, pitch=-10.0, fov=100.0)]
    txt = build_command_file(pts, fps=10.0)
    lines = txt.strip().splitlines()
    assert len(lines) == 21 * 4          # 21 frames x 4 channels
    yaws = [line for line in lines if " v360@view yaw " in line]
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
        assert f"v360@view {channel} " in txt


def test_empty_path_refuses() -> None:
    try:
        build_command_file([])
    except ValueError:
        return
    raise AssertionError("empty path must refuse, not emit an empty file")


def test_seam_crossing_pass_takes_the_short_way() -> None:
    """A front-to-behind follow drives the view yaw through +-180.

    Characterization test: the per-frame loop already unwrapped at
    base -- this pins the short-way behavior while the helpers move
    into one shared home. Knots at 170 and -170 (i.e. 190 wrapped):
    must cross the seam the short way -- continuous wrapped values, no
    jump of more than a few degrees between consecutive commands, and
    nothing outside v360's [-180, 180] range. Interpolating the wrapped
    knots directly would emit a ~358-degree violent spin instead.
    """
    pts = [ControlPoint(t=0.0, yaw=170.0, fov=100.0),
           ControlPoint(t=2.0, yaw=-170.0, fov=100.0)]
    yaws = _channel_values(build_command_file(pts, fps=10.0), "yaw")
    assert len(yaws) == 21
    assert all(-180.0 <= y <= 180.0 for y in yaws), yaws
    steps = [_circ_step(a, b) for a, b in zip(yaws, yaws[1:])]
    assert max(steps) < 5.0, max(steps)
    # Short way: total travel is ~20 deg through +-180, not ~340 back.
    assert 15.0 < sum(steps) < 25.0, sum(steps)
    assert any(abs(y) > 179.0 for y in yaws), "must actually cross the seam"


def test_single_point_wraps_every_angle_channel() -> None:
    """A lone knot beyond the seam must still emit in-range commands."""
    txt = build_command_file(
        [ControlPoint(t=0.0, yaw=190.0, pitch=200.0, roll=-190.0)], fps=10.0)
    got = {c: _channel_values(txt, c)[0] for c in ("yaw", "pitch", "roll")}
    assert got["yaw"] == -170.0, got                        # 190 -> -170
    assert got["pitch"] == -160.0, got                      # 200 -> -160
    assert got["roll"] == 170.0, got                        # -190 -> 170
    assert all(-180.0 <= v <= 180.0 for v in got.values()), got


def test_static_graph_wraps_the_seam_crossing_first_point() -> None:
    """render()'s static v360 options are validated like commands.

    A follow that starts beyond the seam (yaw 190) must not reach the
    filtergraph unwrapped: v360 refuses out-of-range statics outright.
    """
    from steadycut.render.reframe import render
    args = render("dummy.insv", "out.mp4",
                  [ControlPoint(t=0.0, yaw=190.0, pitch=-10.0, fov=100.0),
                   ControlPoint(t=2.0, yaw=-170.0, fov=100.0)],
                  dry_run=True)
    fg = next(a for a in args if "v360@view" in a)
    assert "yaw=190" not in fg, fg
    assert ":yaw=-170.0:" in fg, fg


def test_follow_bias_anchors_unwrap_before_lerping(tmp_path) -> None:
    """The --yaw-bias-track follow lerps the UNWRAPPED anchors.

    Anchors at 170 and -170 mean 170 -> 190 (a +20 follow through the
    seam). This goes through the real CLI path (apply_yaw_bias_track
    over a JSON track file): mid-window bias must read 180, the short
    way. Feeding the same helper the wrapped anchors gives 0.0, the
    ~340-deg long way the old inline lerp emitted -- so reverting the
    unwrap in the CLI path fails this test.
    """
    import json

    import numpy as np

    from steadycut.steadycut import apply_yaw_bias_track
    track = tmp_path / "bias.json"
    track.write_text(json.dumps([[0.0, 170.0], [10.0, -170.0]]))
    pts = [ControlPoint(t=0.0, yaw=0.0, fov=100.0),
           ControlPoint(t=5.0, yaw=0.0, fov=100.0),
           ControlPoint(t=10.0, yaw=0.0, fov=100.0)]
    got, n = apply_yaw_bias_track(pts, str(track))
    assert n == 2
    assert [p.yaw for p in got] == [170.0, 180.0, 190.0]
    # The helper lerps raw numbers: wrapped anchors take the long way.
    wrapped = [(0.0, 170.0), (10.0, -170.0)]
    assert lerp_bias_anchors(wrapped, 5.0) == 0.0
    assert float(np.interp(5.0, [0.0, 10.0],
                           unwrap_deg_series([170.0, -170.0]))) == 180.0


def test_wrap_deg_passes_integers_and_fractions_through() -> None:
    """In-range values reach the argv text bit-unchanged.

    The modulo form is not an identity in floating point (0.1 became
    0.09999999999999432), so non-integer first-point angles changed the
    emitted text. The identity short-circuit keeps scalars and arrays
    exact, including the -180 edge and -0.0's sign.
    """
    import math

    import numpy as np
    assert wrap_deg(0.1) == 0.1
    assert wrap_deg(-22.3) == -22.3
    assert math.copysign(1.0, wrap_deg(-0.0)) == -1.0
    assert wrap_deg(179.999) == 179.999
    assert wrap_deg(-180.0) == -180.0
    assert wrap_deg(180.0) == -180.0
    assert wrap_deg(190.0) == -170.0
    got = wrap_deg(np.array([0.1, -22.3, 190.0, -180.0]))
    assert list(got) == [0.1, -22.3, -170.0, -180.0]
