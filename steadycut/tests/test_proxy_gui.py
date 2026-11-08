"""Offscreen smoke: open a real .insv through the segmented proxy wiring.

Real probe/identify/pitch/gyro/telemetry; segment builds and thumb
grabs are fast fakes (a tiny lavfi mp4 + synthetic JPEGs) so the test
proves the WIRING (store ready, strip span, seek, building tag, event
plumbing, loop tick) without waiting on 30 s renders. The render path
itself is covered by the boundary test in test_proxy.py.

Skipped when the footage is absent, so the suite stays green anywhere;
STEADYCUT_TEST_INSV points at a 30-min .insv to run it.
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

PySide6 = pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from steadycut.gui import proxy as PX  # noqa: E402
from steadycut.gui.app import Main  # noqa: E402

REAL_INSV = os.environ.get(
    "STEADYCUT_TEST_INSV",
    "/backup/insta360/VID_20260221_150751_00_010_012-Original"
    "/VID_20260221_150751_00_010.insv")
needs_media = pytest.mark.skipif(
    not Path(REAL_INSV).exists(),
    reason="smoke needs a 30-min camera-original .insv")


@pytest.fixture(scope="module")
def app(qapp):
    return qapp


def _wait_for(pred, timeout=120.0):
    deadline = time.time() + timeout
    while not pred() and time.time() < deadline:
        QApplication.processEvents()
        time.sleep(0.1)
    assert pred()


@needs_media
def test_open_seek_build_tick_offscreen(app, tmp_path, monkeypatch):
    w = Main()
    w.show()
    try:
        # Tiny stand-in segment file: decodable, so the player exercises
        # the real POS_FRAMES path against fake-built segments.
        (tmp_path / "mini").mkdir()
        mini = tmp_path / "mini" / "m.mp4"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc=s=160x120:r=10:d=3",
             "-frames:v", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             str(mini)], check=True, timeout=120)

        def fake_build(store, i, builder=None):
            video, sidecar = store.seg_paths(i)
            shutil.copy(mini, video)
            sidecar.write_text(json.dumps(store.expected(i)))

        import cv2
        import numpy as np

        def fake_grab(source, t, pitch, size):
            img = np.full((27, 48, 3), int(t) % 256, dtype=np.uint8)
            ok, buf = cv2.imencode(".jpg", img)
            assert ok
            return bytes(buf)

        real_builder = PX.SegmentBuilder
        real_strip = PX.ThumbStrip
        monkeypatch.setattr(
            PX, "SegmentBuilder",
            lambda store: real_builder(store, build_fn=fake_build,
                                       settle_s=0.05))
        monkeypatch.setattr(
            PX, "ThumbStrip",
            lambda source, times, pitch, **kw: real_strip(
                source, times, pitch, grab_fn=fake_grab, workers=2, **kw))

        # Isolate the segment cache: the default root is the real
        # ~/.cache, and a previous run's segments would make the
        # "unbuilt gap" assertion below fail on a second run.
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        w.open_media(REAL_INSV)
        _wait_for(lambda: w.store is not None)
        assert w.total > 1700.0  # full 30-min file, not the 60 s window
        assert (w.strip.p0, w.strip.p1) == (0.0, w.total)
        _wait_for(lambda: len(w.gyro.times) > 1000)  # full-file gyro lane
        _wait_for(lambda: len(w.thumb_qimages) > 5)  # streaming thumbs
        assert w.strip.thumbs, "strip shows streamed thumbs"
        seg = PX.segment_at_time(10.0, w.store.fps,
                                 w.store.frames_per_segment)
        _wait_for(lambda: w.store.is_present(seg), timeout=180)
        # Seek onto a built segment: moving preview, no building tag.
        # (t=0.5: the fake stand-in segment holds 30 frames, so the seek
        # stays inside its content; the path is the real POS_FRAMES one.)
        w._on_seek(0.5)
        QApplication.processEvents()
        assert w.prev.frame is not None
        assert "building" not in w.prev.overlay
        # Seek onto an unbuilt gap: hold + tag, never a hang.
        far = 1500.0
        assert not w.store.is_present(PX.segment_at_time(
            far, w.store.fps, w.store.frames_per_segment))
        w._on_seek(far)
        QApplication.processEvents()
        assert "building" in w.prev.overlay
        # Loop tick inside the built window advances the playhead.
        w._on_seek(0.5)
        w.playing = True
        t0 = w.play_t
        w._on_tick()
        w.playing = False
        assert w.play_t >= t0
    finally:
        w._close_proxy_workers()
        if w.loader is not None and w.loader.isRunning():
            w.loader.wait(15000)
        w.close()
