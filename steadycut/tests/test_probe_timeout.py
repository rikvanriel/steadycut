"""ffprobe hangs on the media-probe path must fail fast, not block Open.

Camera-original .insv occasionally stalls ffprobe; the GUI open path
(media_total -> probe, fps_ratio, total_source_frames) had no timeout, so
one stall hung Open forever with no message. Every ffprobe on that path
now carries an explicit timeout and surfaces "cannot probe <file>".

These tests put a stub `ffprobe` that sleeps on PATH and assert each
probe entry point gives up within its timeout instead of blocking behind
the stub. No real footage needed: the stub never reads its argument.
"""

import os
import stat
import time
from fractions import Fraction

import pytest

TIMEOUT = 1.0
STUB_SLEEP_S = 30
# Generous upper bound: the stub sleeps 30 s, so anything under this proves
# the timeout fired instead of the stub finishing.
BOUND_S = 20.0


@pytest.fixture
def hung_ffprobe(tmp_path, monkeypatch):
    """A stub `ffprobe` that hangs: sleeps without reading its argument."""
    stub = tmp_path / "ffprobe"
    stub.write_text("#!/bin/sh\nsleep %d\n" % STUB_SLEEP_S)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    return stub


def test_ffprobe_probe_times_out(hung_ffprobe) -> None:
    from steadycut.ingest import probe_insv

    start = time.monotonic()
    with pytest.raises(RuntimeError, match="cannot probe"):
        probe_insv._ffprobe("clip.insv", timeout=TIMEOUT)
    assert time.monotonic() - start < BOUND_S


def test_fps_ratio_times_out(hung_ffprobe) -> None:
    from steadycut.gui import proxy as PX

    start = time.monotonic()
    with pytest.raises(ValueError, match="cannot probe"):
        PX.fps_ratio("ratio-clip.insv", timeout=TIMEOUT)
    assert time.monotonic() - start < BOUND_S


def test_total_source_frames_times_out_without_fallback(hung_ffprobe) -> None:
    """Timeout raises at once: no fallback into another (slow) probe."""
    from steadycut.gui import proxy as PX

    start = time.monotonic()
    with pytest.raises(ValueError, match="cannot probe"):
        PX.total_source_frames("frames-clip.insv", Fraction(30000, 1001),
                               timeout=TIMEOUT)
    assert time.monotonic() - start < BOUND_S


def test_source_fps_times_out_to_fallback(hung_ffprobe) -> None:
    """reframe keeps its probe-failure fallback (29.97 grid), but bounded."""
    from steadycut.render import reframe

    start = time.monotonic()
    assert reframe.source_fps("reframe-clip.insv",
                              timeout=TIMEOUT) == reframe.FPS
    assert time.monotonic() - start < BOUND_S


def test_adopt_or_delete_legacy_times_out(hung_ffprobe, tmp_path) -> None:
    """The legacy-adopt ffprobe carries the same timeout: a hung probe
    deletes (never adopts) instead of blocking Open behind the stall."""
    from steadycut.gui import proxy as PX

    store = PX.SegmentStore(root=tmp_path / "fp", fps=Fraction(30000, 1001),
                            frames_per_segment=899, total_frames=899,
                            pitch=-12.0, source="src.insv")
    legacy = tmp_path / "legacy.mp4"
    legacy.write_bytes(b"not-a-video")
    start = time.monotonic()
    assert store.adopt_or_delete_legacy(legacy, timeout=TIMEOUT) == "deleted"
    assert time.monotonic() - start < BOUND_S
    assert not legacy.exists()
