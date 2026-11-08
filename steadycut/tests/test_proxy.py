"""Segmented proxy: frame grid, cache policy, background builder.

Qt-free: no PySide6 import here (the blocked-import test below proves
proxy.py serves the base install). Real-media boundary checks at the
end are skipped when the footage is absent, so the suite stays green
anywhere; STEADYCUT_TEST_INSV points at a 30-min .insv to run them.
"""

import json
import os
import subprocess
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path

import pytest

import steadycut
from steadycut.gui import proxy as PX

FPS = Fraction(30000, 1001)
N = 899  # round(30 s * 30000/1001)
TOTAL = 53920


def _store(tmp_path, n=N, total=TOTAL, pitch=-12.0, cap=PX.DEFAULT_CAP_BYTES):
    root = tmp_path / "fp"
    root.mkdir()
    return PX.SegmentStore(root=root, fps=FPS, frames_per_segment=n,
                           total_frames=total, pitch=pitch, source="src.insv",
                           cap_bytes=cap)


def _publish(store, i):
    video, sidecar = store.seg_paths(i)
    video.write_bytes(b"fake-segment-%d" % i)
    sidecar.write_text(json.dumps(store.expected(i)))


def test_segment_count_is_a_frame_constant():
    assert PX.segment_frame_count(30.0, FPS) == 899
    assert PX.segment_frame_count(30.0, Fraction(30000, 1001)) == 899
    assert PX.n_segments(TOTAL, N) == 60
    assert PX.n_segments(N, N) == 1
    assert PX.n_segments(N + 1, N) == 2


def test_segment_spans_clip_the_tail():
    assert PX.segment_span(0, N, TOTAL) == (0, 899)
    assert PX.segment_span(1, N, TOTAL) == (899, 1798)
    assert PX.segment_span(59, N, TOTAL) == (53041, 53920)
    with pytest.raises(IndexError):
        PX.segment_span(60, N, TOTAL)


def test_frame_time_is_full_precision():
    # repr round-trips: the emitted -ss lands on the snapped frame, never
    # a dust epsilon past it (the :g formatting bug this replaces).
    t = PX.frame_time(899, FPS)
    assert repr(float(Fraction(899, 1) / FPS)) == repr(t)
    assert abs(t - 899 * 1001 / 30000) < 1e-9


def test_time_to_segment_mapping_round_trips():
    assert PX.segment_at_time(0.0, FPS, N) == 0
    assert PX.segment_at_time(29.99, FPS, N) == 0
    assert PX.segment_at_time(30.0, FPS, N) == 1
    assert PX.segment_at_time(750.0, FPS, N) == 25
    s, d, c = PX.segment_window(25, N, FPS, TOTAL)
    assert c == N and abs((s + d) - PX.frame_time(25 * N + N, FPS)) < 1e-6


def test_sidecar_identity_rejects_foreign_pixels():
    good = PX.expected_sidecar(3, N, FPS, -12.0, (960, 720), TOTAL)
    assert PX.sidecar_matches(dict(good), dict(good))
    for key, bad in (("pitch", -11.0), ("size", [480, 270]),
                     ("fps", "25/1"), ("span_frames", [0, 899]),
                     ("version", 999)):
        other = dict(good)
        other[key] = bad
        assert not PX.sidecar_matches(other, good), key
    assert not PX.sidecar_matches({}, good)


def test_store_scan_is_the_directory_listing(tmp_path):
    store = _store(tmp_path)
    assert store.scan() == {}
    assert not store.is_present(0)
    _publish(store, 0)
    _publish(store, 2)
    assert set(store.scan()) == {0, 2}
    assert store.is_present(0) and not store.is_present(1)
    # A sidecar from another segment set never validates ...
    (tmp_path / "other").mkdir(exist_ok=True)
    other = _store(tmp_path / "other", pitch=-11.0)
    video, sidecar = store.seg_paths(4)
    video.write_bytes(b"x")
    sidecar.write_text(json.dumps(other.expected(4)))
    assert not store.is_present(4)
    # ... and neither does a sidecar without a video file.
    lone, lone_side = store.seg_paths(5)
    lone_side.write_text(json.dumps(store.expected(5)))
    assert not store.is_present(5)
    store.clean_temps()


def test_pins_cover_window_playhead_preload_and_caps(tmp_path):
    store = _store(tmp_path)
    pins = store.pins(playhead_i=25, in_i=20, out_i=22,
                      open_segments=(30,))
    assert {20, 21, 22, 25, 26, 30} <= pins
    assert store.pins(playhead_i=59) <= set(range(60))  # clipped, no 60
    assert 60 not in store.pins(playhead_i=59)


def test_builder_pins_player_open_segments(tmp_path):
    # Open caps are unevictable: the player holds the current segment
    # plus preload across boundary crossings, and evicting under an open
    # cap turns a cached seek into a "building" miss. The app refreshes
    # the set from player.open_segments() on its poll loop.
    store = _store(tmp_path)
    b = PX.SegmentBuilder(store, build_fn=lambda *a: None, settle_s=0.0)
    b.request_playhead(25 * 899 / float(FPS))
    b.note_open_segments((24,))
    _, pins = b._targets()
    assert {24, 25, 26} <= pins


def test_only_demand_evicts(tmp_path):
    store = _store(tmp_path)
    for i in range(6):
        _publish(store, i)
        store.seg_paths(i)[0].write_bytes(b"0" * (200 * 1024))
    store.cap_bytes = 500 * 1024
    keep = store.pins(playhead_i=0, in_i=0, out_i=0)
    evicted = store.enforce_cap(keep, rank=[5, 4, 3, 2, 1])
    assert evicted == [5, 4, 3, 2]  # farthest first, pins survive
    assert store.is_present(0) and not store.is_present(5)
    # Pins are never touched even when the dir stays over cap.
    store.cap_bytes = 1
    assert store.enforce_cap(keep) == []


def test_adopt_or_delete_legacy(tmp_path):
    store = _store(tmp_path)
    legacy = tmp_path / "legacy.mp4"
    assert store.adopt_or_delete_legacy(legacy) == "absent"
    legacy.write_bytes(b"not-a-video")
    assert store.adopt_or_delete_legacy(legacy) == "deleted"
    assert not legacy.exists()


def test_legacy_with_audio_is_deleted(tmp_path):
    subprocess.run(["ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc=s=160x120:r=10:d=1",
                    "-f", "lavfi", "-i", "sine=f=440:d=1",
                    "-c:v", "libx264", "-c:a", "aac",
                    str(tmp_path / "legacy.mp4")], check=True)
    store = _store(tmp_path, n=300, total=300, pitch=-12.0)
    out = store.adopt_or_delete_legacy(tmp_path / "legacy.mp4")
    assert out == "deleted"


def test_pitch_persists_across_sessions(tmp_path):
    calls = []

    def gravity():
        calls.append(1)
        return -13.5, {"layer": "gravity", "usable": True}

    p1, _ = PX.load_or_resolve_pitch(tmp_path, "src.insv", gravity=gravity)
    p2, info2 = PX.load_or_resolve_pitch(tmp_path, "src.insv",
                                          gravity=gravity)
    assert (p1, p2) == (-13.5, -13.5)
    assert info2["layer"] == "cache"  # second open: no re-measure
    assert calls == [1]
    assert json.loads((tmp_path / PX.PITCH_FILE).read_text())["pitch"] == -13.5


def _fps10_store(tmp_path, n=30, total=300, pitch=-12.0,
                 cap=PX.DEFAULT_CAP_BYTES):
    root = tmp_path / "fp"
    root.mkdir(exist_ok=True)
    return PX.SegmentStore(root=root, fps=Fraction(10, 1),
                           frames_per_segment=n, total_frames=total,
                           pitch=pitch, source="src.insv", cap_bytes=cap)


def _fake_builder(store, settle=0.05, delay=0.0):
    built = []
    started = threading.Event()

    def build(store_, i, builder=None):
        started.set()
        if delay:
            time.sleep(delay)
        _publish(store_, i)
        built.append(i)

    b = PX.SegmentBuilder(store, build_fn=build, settle_s=settle)
    return b, built, started


def _wait_for(pred, timeout=15.0):
    deadline = time.time() + timeout
    while not pred() and time.time() < deadline:
        time.sleep(0.05)
    assert pred()


def test_demand_jumps_the_queue(tmp_path):
    store = _fps10_store(tmp_path)
    b, built, _ = _fake_builder(store)
    b.request_playhead(9.0 * 3)  # segment 9 of 10 (fps 10, n 30)
    b.start()  # request before start: deterministic, no fill race
    try:
        _wait_for(lambda: 9 in built)
        assert built[0] == 9  # demand first, not fill from 0
    finally:
        b.stop()


def test_debounce_coalesces_rapid_seeks(tmp_path):
    store = _fps10_store(tmp_path)
    starts = []

    def build(store_, i, builder=None):
        starts.append((i, time.monotonic()))
        _publish(store_, i)
        return None

    b = PX.SegmentBuilder(store, build_fn=build, settle_s=0.3)
    b.request_playhead(0.0)
    for k in range(1, 5):
        time.sleep(0.05)
        b.request_playhead(float(k * 3))  # segments 0..4
    last_request = time.monotonic()  # stamped at the final request
    b.start()
    try:
        _wait_for(lambda: len(starts) > 0)
        b.stop()
        # One build started, of the latest playhead only, after settling.
        assert starts[0][0] == 4, starts
        assert starts[0][1] - last_request >= 0.3, starts
    finally:
        b.stop()


def test_fill_runs_playhead_outward_and_never_evicts(tmp_path):
    store = _fps10_store(tmp_path)
    _publish(store, 0)
    b, built, _ = _fake_builder(store, settle=0.0)
    store.cap_bytes = 1  # over cap: fill must wait, demand still builds
    b.start()
    try:
        time.sleep(1.0)
        assert built == []  # fill never evicts to make room
        store.cap_bytes = PX.DEFAULT_CAP_BYTES
        _wait_for(lambda: len(built) >= 4)
        assert built[:4] == [1, 2, 3, 4]  # nearest-first from playhead 0
    finally:
        b.stop()


def test_pause_during_render(tmp_path):
    store = _store(tmp_path, n=30, total=300)
    entered = threading.Event()
    release = threading.Event()

    def slow(store_, i, builder=None):
        entered.set()
        assert release.wait(timeout=10)
        _publish(store_, i)

    b = PX.SegmentBuilder(store, build_fn=slow, settle_s=0.0)
    b.start()
    try:
        assert entered.wait(timeout=15)  # fill started
        b.set_rendering(True)  # in-flight finishes anyway
        release.set()
        deadline = time.time() + 10
        while not store.is_present(0) and time.time() < deadline:
            time.sleep(0.05)
        assert store.is_present(0)
        time.sleep(0.6)
        assert set(store.scan()) == {0}  # nothing new starts mid-render
        b.set_rendering(False)
        deadline = time.time() + 10
        while len(store.scan()) < 2 and time.time() < deadline:
            time.sleep(0.05)
        assert len(store.scan()) >= 2
    finally:
        b.stop()


def test_stop_drops_pending_and_joins(tmp_path):
    store = _store(tmp_path, n=30, total=300)
    b, built, _ = _fake_builder(store, settle=60.0)  # never settles
    b.request_playhead(5.0)  # queued ...
    b.start()
    time.sleep(0.5)  # ... but the playhead never settles, so no build
    thread = b._thread
    b.stop()
    assert built == []
    assert not thread.is_alive()


def _jpg(color=127):
    import cv2
    import numpy as np

    img = np.full((27, 48, 3), color, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return bytes(buf)


def test_thumb_stream_and_contact_sheet(tmp_path):
    import cv2

    grabs = []

    def fake_grab(source, t, pitch, size):
        grabs.append(t)
        return _jpg(color=int(t) % 256)

    total = 100.0
    times = PX.thumb_times(total)
    assert len(times) == 10 and times[0] == 5.0
    strip = PX.ThumbStrip("src.insv", times, -12.0,
                          sheet=tmp_path / PX.THUMB_SHEET,
                          index=tmp_path / PX.THUMB_INDEX,
                          grab_fn=fake_grab, workers=2)
    strip.start()
    deadline = time.time() + 15
    while len(grabs) < len(times) and time.time() < deadline:
        time.sleep(0.05)
    strip.stop()
    got = [e for e in strip.drain() if e[0] == "thumb"]
    assert len(got) == len(times)  # every slot streams in
    strip.save_sheet()
    assert (tmp_path / PX.THUMB_SHEET).exists()
    # Reopen: the cached sheet gives instant thumbs, no grabs.
    grabs.clear()
    strip2 = PX.ThumbStrip("src.insv", times, -12.0,
                           sheet=tmp_path / PX.THUMB_SHEET,
                           index=tmp_path / PX.THUMB_INDEX,
                           grab_fn=fake_grab, workers=2)
    assert len(strip2.cells) == len(times)
    strip2.start()
    strip2.stop()
    assert grabs == []
    assert len([e for e in strip2.drain() if e[0] == "thumb"]) == len(times)
    # A different framing never matches the cached sheet.
    strip3 = PX.ThumbStrip("src.insv", times, -11.0,
                           sheet=tmp_path / PX.THUMB_SHEET,
                           index=tmp_path / PX.THUMB_INDEX,
                           grab_fn=fake_grab, workers=2)
    assert strip3.cells == {}


def test_player_hold_last_frame_and_eof(tmp_path):
    cv2 = pytest.importorskip("cv2")
    store = _fps10_store(tmp_path)
    subprocess.run(["ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc=s=160x120:r=10:d=3",
                    "-frames:v", "30", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(tmp_path / "fp" / "x.mp4")],
                   check=True)
    src = tmp_path / "fp" / "x.mp4"
    for i in (0, 2):
        video, sidecar = store.seg_paths(i)
        import shutil

        shutil.copy(src, video)
        sidecar.write_text(json.dumps(store.expected(i)))
    player = PX.SegmentPlayer(store)
    try:
        frame, status = player.frame_at(1.0)
        assert status == "ok" and frame is not None
        held = frame
        # Segment 1 is unbuilt: hold the last frame, report building.
        frame, status = player.frame_at(4.0)
        assert status == "building" and frame is held
        # POS_FRAMES seeks move within a segment.
        import numpy as np

        f05, _ = player.frame_at(0.5)
        f15, _ = player.frame_at(1.5)
        assert not np.array_equal(f05, f15)
        # Past the end: eof.
        _, status = player.frame_at(31.0)
        assert status == "eof"
        assert player.open_segments() == (0,)
    finally:
        player.close()
    assert player.open_segments() == ()


def test_render_exact_frames_video_only_dry_run(monkeypatch):
    from steadycut.render import reframe
    from steadycut.render.reframe import ControlPoint, render

    monkeypatch.setattr(reframe, "source_fps", lambda s: 29.97)
    path = [ControlPoint(t=0.0, fov=100.0, pitch=-12.0)]
    args = render("src.mp4", "out.mp4", path, duration=30.0, start=0.0,
                  frames=899, include_audio=False, dry_run=True)
    assert "-frames:v" in args and args[args.index("-frames:v") + 1] == "899"
    assert "-an" in args
    assert not any(a.startswith("0:a") for a in args)
    assert "-c:a" not in args
    # Defaults are byte-identical to before: audio mapped, slack cap.
    args = render("src.mp4", "out.mp4", path, duration=30.0, start=0.0,
                  dry_run=True)
    assert "0:a:0?" in args and "-c:a" in args and "-an" not in args


def test_proxy_modules_import_without_qt(tmp_path):
    root = Path(steadycut.__file__).parent.parent
    blocker = tmp_path / "sitecustomize.py"
    blocker.write_text(
        "import sys\n"
        "class _BlockQt:\n"
        "    def find_module(self, name, path=None):\n"
        "        return self if name == 'PySide6' else None\n"
        "    def load_module(self, name):\n"
        "        raise ImportError('Qt blocked for hygiene check')\n"
        "sys.meta_path.insert(0, _BlockQt())\n")
    code = ("import steadycut.gui.proxy, steadycut.gui.proxy_build, "
            "steadycut.gui.backend, steadycut.gui.runner; "
            "print('qt-free ok')")
    env = dict(os.environ, PYTHONPATH=str(tmp_path) + os.pathsep + str(root))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "qt-free ok" in proc.stdout


REAL_INSV = os.environ.get(
    "STEADYCUT_TEST_INSV",
    "/backup/insta360/VID_20260221_150751_00_010_012-Original"
    "/VID_20260221_150751_00_010.insv")
needs_media = pytest.mark.skipif(
    not Path(REAL_INSV).exists(),
    reason="boundary gate needs a 30-min camera-original .insv")


def _psnr(a, b) -> float:
    import numpy as np

    mse = float(((a.astype(float) - b.astype(float)) ** 2).mean())
    return float("inf") if mse == 0 else float(
        20 * np.log10(255.0 / np.sqrt(mse)))


def _read_all(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


@needs_media
def test_segment_boundaries_index_exact(tmp_path):
    """Segment edges are exact source frames (PSNR, index-based).

    A "no visible jump" check passes a duplicated frame, so this test
    compares against the RIGHT frames: segment i's last frame and i+1's
    first must match source frames k-1 and k rendered as one 2-frame
    window. A duplicate or skip shows the WRONG content (~20 dB here);
    encode noise between two correct renders sits near ~38 dB.
    """
    import cv2  # noqa: F811

    from steadycut.job import build_proxy
    from steadycut.render.reframe import source_fps  # noqa: F401 (warms cache)

    store_dir = Path(os.environ.get(
        "STEADYCUT_TEST_SEGMENTS",
        str(Path.home() / ".cache" / "steadycut" / "test-segments")))
    fps = PX.fps_ratio(REAL_INSV)
    assert fps == Fraction(30000, 1001)
    n = PX.segment_frame_count(30.0, fps)
    assert n == 899
    store = PX.open_store(store_dir, REAL_INSV, -12.0)
    assert store.frames_per_segment == n
    assert store.count() == 60

    def build(i):
        if store.is_present(i):
            return
        video, sidecar = store.seg_paths(i)
        tmp = video.with_name(f"{PX.TMP_PREFIX}{video.name}")
        r = subprocess.run(
            [sys.executable, "-m", "steadycut.gui.proxy_build",
             "--source", REAL_INSV, "--frame-start", str(i * n),
             "--nframes", str(PX.segment_span(i, n, store.total_frames)[1]
                              - PX.segment_span(i, n, store.total_frames)[0]),
             "--fps", f"{fps.numerator}/{fps.denominator}",
             "--pitch", "-12.0", "--width", "960", "--height", "720",
             "--out", str(video), "--tmp", str(tmp),
             "--sidecar", str(sidecar),
             "--sidecar-json", json.dumps(store.expected(i))],
            capture_output=True, text=True, cwd=Path(
                steadycut.__file__).parent.parent, timeout=1800)
        assert r.returncode == 0, r.stderr[-3000:]
        assert store.is_present(i)

    for i in (0, 1, 25):
        build(i)
    for i in (0, 1, 25):
        video, _ = store.seg_paths(i)
        assert len(_read_all(video)) == n  # -frames:v exact, not the +5 slack
        probes = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, check=True)
        assert probes.stdout.strip().split() == ["video"]  # AAC would skew edges

    refs = {}
    for name, f0 in (("r898", 898), ("r22474", 22474)):
        dst = tmp_path / f"{name}.mp4"
        build_proxy(REAL_INSV, PX.frame_time(f0, fps),
                    float(Fraction(2, 1) / fps), str(dst), -12.0,
                    nframes=2, video_only=True)
        refs[name] = _read_all(dst)
        assert len(refs[name]) == 2

    s0 = _read_all(store.seg_paths(0)[0])
    s1 = _read_all(store.seg_paths(1)[0])
    s25 = _read_all(store.seg_paths(25)[0])
    right_0 = _psnr(s0[-1], refs["r898"][0])
    right_1 = _psnr(s1[0], refs["r898"][1])
    right_25 = _psnr(s25[0], refs["r22474"][1])
    print(f"\nboundary PSNR: i=1 last {right_0:.1f} first {right_1:.1f}, "
          f"i=25 first {right_25:.1f}")
    assert right_0 >= 30.0
    assert right_1 >= 30.0
    assert right_25 >= 30.0


def test_playback_cursor_loops_and_holds():
    cur = PX.PlaybackCursor(play_t=10.0, in_t=10.0, out_t=25.0)
    t, tag = cur.advance(0.066, "ok")
    assert t == pytest.approx(10.066) and tag == ""
    cur.play_t = 24.99
    t, _ = cur.advance(0.066, "ok")
    assert t == 10.0  # in-window loop wraps at out_t, never at p0
    cur.play_t = 20.0
    t, tag = cur.advance(0.066, "building")
    assert (t, tag) == (20.0, "building …")  # unbuilt gap holds the frame
