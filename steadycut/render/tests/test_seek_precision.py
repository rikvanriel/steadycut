"""-ss/-t must keep full precision past ~1000 s.

f"{x:g}" keeps 6 significant digits, so at 30000/1001 fps
snap_to_frame(1500) = 1499.9985 was emitted as "-ss 1500" -- 0.045 frame
AFTER the snapped frame, so input seek dropped that frame. The ffmpeg args
must parse back to at most the snapped frame time, and to the same frame
index. Fails on the old :g formatting (which emits the bare "1500").
"""
import math

import pytest

import steadycut.render.reframe as reframe
from steadycut.render.reframe import ControlPoint, format_ts, render

FPS = 30000 / 1001
START = 1500.0
SNAPPED = math.floor(START * FPS + 1e-9) / FPS
DURATION = math.floor(60.0 * FPS + 1e-9) / FPS  # 59.99327, also lossy under :g


@pytest.fixture(autouse=True)
def _fps(monkeypatch):
    monkeypatch.setattr(reframe, "source_fps", lambda source: FPS)


def _opt(args, flag):
    return args[args.index(flag) + 1]


def _assert_seek(args):
    ss = float(_opt(args, "-ss"))
    assert ss <= SNAPPED, f"-ss {ss} is past the snapped frame {SNAPPED}"
    assert math.floor(ss * FPS + 1e-9) == math.floor(SNAPPED * FPS + 1e-9), (
        f"-ss {ss} selects a different frame than snapped {SNAPPED}")
    assert float(_opt(args, "-t")) == DURATION


def _path():
    return [ControlPoint(t=0.0, fov=100.0)]


def test_format_ts_round_trips_snapped_frame():
    assert float(format_ts(SNAPPED)) == SNAPPED
    assert f"{SNAPPED:g}" != format_ts(SNAPPED)  # old code emitted "1500"


def test_dual_stream_seek_lands_on_snapped_frame(tmp_path):
    args = render("src.insv", tmp_path / "out.mp4", _path(),
                  start=START, duration=DURATION, dry_run=True)
    _assert_seek(args)


def test_single_stream_seek_is_snapped(tmp_path):
    # The single-stream branch used to pass the raw start unsnapped.
    args = render("src.insv", tmp_path / "out.mp4", _path(),
                  start=START, duration=DURATION,
                  dual_stream=False, dry_run=True)
    _assert_seek(args)


def test_blend_audio_seek_lands_on_snapped_frame(tmp_path):
    from steadycut.render.stitch import StitchSpec
    args = render("src.insv", tmp_path / "out.mp4", _path(),
                  start=START, duration=DURATION,
                  stitch=StitchSpec(method="blend"), dry_run=True)
    _assert_seek(args)


def test_bake_lens_tracks_seek_is_snapped(tmp_path, monkeypatch):
    import steadycut.render.stitch as stitch
    seen = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        return None

    monkeypatch.setattr("subprocess.run", fake_run)
    spec = stitch.StitchSpec(method="blend", deflare_scale=(1.0, 1.0))
    stitch.bake_lens_tracks("src.insv", spec, stitch.GAIN_DEFAULT,
                            START, DURATION, tmp_path)
    _assert_seek(seen["args"])
