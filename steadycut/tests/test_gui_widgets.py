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
def app():
    application = QApplication([])
    yield application


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
