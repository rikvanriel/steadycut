"""Offscreen Main-window smoke: construction, panel sync, Copy CLI text.

Requires the `[gui]` extra; skipped in the base install. Offscreen, so
no display server is needed and no footage is touched.
"""

import os

import pytest

PySide6 = pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from steadycut.gui import backend as B  # noqa: E402
from steadycut.gui.app import Main  # noqa: E402


@pytest.fixture(scope="module")
def app(qapp):
    return qapp


def test_window_builds_with_no_media(app):
    w = Main()
    assert w.state.source == ""
    assert "no media" in w.valid_lbl.text()
    assert w.render_btn.isEnabled()
    assert not w.cancel_btn.isEnabled()
    w.close()


def test_in_out_edits_drive_state_and_copy_cli(app):
    w = Main()
    w.state.source = "clip.insv"
    w.e_in.setText("00:10.0")
    w.e_out.setText("00:25.0")
    w._on_edits()
    assert (w.state.start, w.state.dur) == (10.0, 15.0)
    assert "playhead 00:10.0" in w.tc.text()
    w.on_copy()
    cmd = QApplication.clipboard().text()
    assert "clip.insv" in cmd and "--start 10.0" in cmd
    assert "--dur 15.0" in cmd and "--job" not in cmd
    w.close()


def test_aim_drag_sets_manual_override_and_reset_clears(app):
    w = Main()
    w.state.source = "clip.insv"
    w.proxy_pitch = -12.0
    assert "policy owns it" in w.aimlbl.text()
    w._on_aim_drag(0.0, 0.0)  # press anchor
    w._on_aim_drag(20.0, -10.0)  # right 20px, up 10px
    assert w.state.yaw_bias == pytest.approx(1.0)
    assert w.state.pitch == pytest.approx(-11.5)
    assert "manual" in w.aimv.text()
    w._on_reset_aim()
    assert (w.state.yaw_bias, w.state.pitch) == (0.0, None)
    w.close()


def test_refusal_banner_shows_msg_and_next_step(app):
    w = Main()
    w.show()
    assert not w.refusal.isVisible()
    w._refuse("window end past media end", "pick a window inside the file")
    assert w.refusal.isVisible()
    assert "past media end" in w.refusal.text()
    assert "inside the file" in w.refusal.text()
    w.close()


def test_fingerprint_tracks_panel_changes(app):
    w = Main()
    w.state.source = "clip.insv"
    w._panel_to_state()
    w._refresh_meta()
    before = B.fingerprint_of(w.state)
    assert before in w.fp_lbl.text()
    w.framing.setCurrentText("ski")
    assert B.fingerprint_of(w.state) != before
    assert "modified since render" in w.fp_lbl.text() or \
        "unrendered" in w.fp_lbl.text()
    w.close()


def test_render_preflight_refuses_without_launching(app):
    w = Main()
    w.show()
    w.state.source = "missing.insv"
    w.e_in.setText("00:10.0")
    w.e_out.setText("00:25.0")
    w.on_render()
    assert w.proc is None  # no subprocess launched
    assert w.refusal.isVisible()
    assert "not found" in w.refusal.text()
    w.close()


def test_strip_click_seeks_and_handle_press_starts_drag(app):
    # Regression: QPointF.x/y are methods in Qt6; the unparenthesised
    # form passed a bound method into arithmetic and every click died
    # in mousePressEvent. Offscreen tests never clicked, so it shipped.
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    from steadycut.gui.widgets import FilmStrip
    strip = FilmStrip()
    strip.resize(800, 90)
    strip.show()
    strip.set_span(0.0, 100.0)
    strip.set_window(50.0, 10.0, 90.0)

    got = []
    strip.seeked.connect(lambda t: got.append(t))
    QTest.mouseClick(strip, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     QPoint(400, 30))
    assert got and abs(got[-1] - 50.0) < 1.0  # mid-strip seeks mid-span

    strip._drag = None
    in_x = int(strip.t2x(10.0)) + 1
    QTest.mousePress(strip, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     QPoint(in_x, 30))
    assert strip._drag == "in"
    QTest.mouseRelease(strip, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                       QPoint(in_x, 30))
    assert strip._drag is None
    strip.close()


def test_preview_paints_empty_and_framed(app):
    # Media-free: both paintEvent branches (overlay text with no frame,
    # framed image with the proxy tag) run offscreen without footage.
    from PySide6.QtGui import QImage

    from steadycut.gui.widgets import Preview
    w = Preview()
    w.resize(320, 240)
    w.show()
    app.processEvents()
    img = QImage(64, 48, QImage.Format.Format_RGB32)
    img.fill(0x112233)
    w.proxy_note = "proxy 720p"
    w.show_frame(img)
    app.processEvents()
    assert w.frame is not None
    w.close()


def test_preview_drag_emits_pixel_delta(app):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QImage
    from PySide6.QtTest import QTest

    from steadycut.gui.widgets import Preview
    w = Preview()
    img = QImage(64, 48, QImage.Format.Format_RGB32)
    img.fill(1)
    w.show_frame(img)
    w.resize(320, 240)
    w.show()
    got = []
    w.dragged.connect(lambda dx, dy: got.append((dx, dy)))
    QTest.mousePress(w, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, QPoint(160, 120))
    QTest.mouseMove(w, QPoint(180, 110))
    QTest.mouseRelease(w, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, QPoint(180, 110))
    assert got == [(20.0, -10.0)]
    w.close()


def test_filmstrip_paints_thumbs_and_drags_both_handles(app):
    # Media-free: thumb paint branch, both handle drags (in/out ranged
    # emits), and the empty-checker branch.
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QImage
    from PySide6.QtTest import QTest

    from steadycut.gui.widgets import FilmStrip

    class _Pos:
        def __init__(self, x):
            self._x = x

        def position(self):
            outer = self

            class _P:
                def x(self):
                    return outer._x

                def y(self):
                    return 30.0

            return _P()

    def _pos_event(x):
        return _Pos(x)

    s = FilmStrip()
    s.resize(800, 90)
    thumbs = []
    for i in range(8):
        img = QImage(80, 54, QImage.Format.Format_RGB32)
        img.fill(i + 1)
        thumbs.append((float(i * 10), img))
    s.thumbs = thumbs
    s.set_span(0.0, 80.0)
    s.set_window(40.0, 10.0, 70.0)
    s.show()
    app.processEvents()
    got = []
    s.seeked.connect(lambda t: got.append(t))
    # A move with no active drag is ignored (guard branch).
    s._drag = None
    s.mouseMoveEvent(_pos_event(200.0))
    QTest.mouseClick(s, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, QPoint(400, 30))
    assert got  # empty-area click seeks
    ranged = []
    s.ranged.connect(lambda a, b: ranged.append((a, b)))
    in_x = int(s.t2x(10.0)) + 1
    QTest.mousePress(s, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, QPoint(in_x, 30))
    assert s._drag == "in"
    QTest.mouseMove(s, QPoint(in_x + 40, 30))
    QTest.mouseRelease(s, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, QPoint(in_x + 40, 30))
    assert ranged and s._drag is None
    ranged.clear()
    out_x = int(s.t2x(70.0)) - 1
    QTest.mousePress(s, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, QPoint(out_x, 30))
    assert s._drag == "out"
    QTest.mouseMove(s, QPoint(out_x - 40, 30))
    QTest.mouseRelease(s, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, QPoint(out_x - 40, 30))
    assert ranged and s._drag is None
    s.close()
    # No thumbs: the fallback checker branch.
    s2 = FilmStrip()
    s2.resize(800, 90)
    s2.set_span(0.0, 100.0)
    s2.set_window(50.0, 10.0, 90.0)
    s2.show()
    app.processEvents()
    s2.close()


def test_gyrostrip_paints_empty_and_series(app):
    # Media-free: the waiting-text branch and the polyline branch with
    # a synthetic series, plus the setter paths.
    from steadycut.gui.widgets import GyroStrip
    g = GyroStrip()
    g.resize(800, 48)
    g.show()
    app.processEvents()
    times = [float(i) * 0.1 for i in range(200)]
    values = [float(i % 10) - 5.0 for i in range(200)]
    g.set_series(times, values)
    g.set_window(5.0, 0.0, 20.0)
    app.processEvents()
    assert g.times and g.values
    g.close()
