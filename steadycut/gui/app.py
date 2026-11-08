"""Clip-picker main window: preview + job panel + transport.

PySide6 lives only in this file and widgets.py, both under the `[gui]`
extra. The CLI never imports them; the translation to argv/envelope and
the process-group runner live in backend.py/runner.py (Qt-free).
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QAction, QImage
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton,
                               QSplitter, QVBoxLayout, QWidget)

from steadycut.gui import backend as B
from steadycut.gui import proxy as PX
from steadycut.gui import runner as R
from steadycut.gui.widgets import FilmStrip, GyroStrip, Preview

PRESETS = ("medium", "fast", "ultrafast", "slow")


def qimage_from_bgr(frame: np.ndarray, width: int = 0) -> QImage:
    """BGR OpenCV frame to a detached RGB QImage, optionally scaled."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    if width and w != width:
        img = img.scaledToWidth(width,
                                Qt.TransformationMode.SmoothTransformation)
    return img


class LoaderThread(QThread):
    """Open-file background work: probe, identify, store, gyro, thumbs.

    The proxy is segmented (one 30 s video-only file per segment in the
    per-source cache dir), so open only resolves the store (pitch, grid,
    legacy adopt/delete) and streams full-file gyro; segment builds and
    thumbnail grabs run in the proxy package's own workers, polled by
    the main window.
    """

    note = Signal(str)
    store_ready = Signal(object, dict, float, dict, float)
    telemetry_ready = Signal(object, object)
    failed = Signal(str, str)
    proxy_only = False

    def __init__(self, source: str, camera_hint: str | None,
                 stitch: str, camera: dict | None = None,
                 profile=None, total: float = 0.0,
                 proxy_only: bool = False, parent=None):
        super().__init__(parent)
        self.source = source
        self.camera_hint = camera_hint
        self.stitch = stitch
        self.camera = camera
        self.profile = profile
        self._total_attr = total
        self._proxy_only = proxy_only

    def run(self):
        try:
            self._run()
        except Exception as exc:  # keep the window alive, show the hint
            self.failed.emit(f"open failed ({exc})",
                             "is this a camera-original file?")

    def _run(self):
        if not self._proxy_only:
            self.note.emit("probing media …")
            total = B.media_total(self.source)
            self.note.emit("identifying camera …")
            ident = B.identify_camera(self.source, self.camera_hint)
            if ident["status"] != "ok":
                self.failed.emit(ident.get("reason", "unidentified body"),
                                 ident.get("action", "see the CLI gates"))
                return
            profile = ident.get("profile")
            camera = {"model": ident.get("model"),
                      "serial": ident.get("serial")}
            manifest = B.current_proxy_manifest(self.source, camera,
                                                self.stitch)
            self._camera = camera
            self._ident = ident
            self._profile = ident.get("profile")
            self._total = total
        else:
            assert self._camera is not None, "proxy rebuild needs a camera"
            camera = self._camera
            manifest = B.current_proxy_manifest(self.source, camera,
                                                self.stitch)
            profile = self.profile
            total = self._total_attr
        self.note.emit("resolving proxy store …")
        root = B.proxy_cache_dir() / manifest["fingerprint"]
        pitch, info = PX.load_or_resolve_pitch(root, self.source, profile)
        store = PX.open_store(root, self.source, pitch)
        self._store = store
        self._pitch, self._pitch_info = pitch, info
        legacy = B.proxy_paths(manifest["fingerprint"])[0]
        adopted = store.adopt_or_delete_legacy(legacy)
        if adopted == "adopted":
            self.note.emit("adopted legacy proxy as segment 0 …")
        self.store_ready.emit(store, manifest, float(pitch),
                              dict(info), float(total))
        if self._proxy_only:
            return
        self.note.emit("loading gyro …")
        self._emit_gyro()
        self.note.emit("proxy store ready …")

    def _emit_gyro(self):
        from steadycut.ingest.telemetry import read_telemetry

        profile = getattr(self, "_profile", None) or self.profile
        imu = read_telemetry(str(self.source), profile=profile)
        t = np.asarray(imu.time_s, dtype=float)
        mag = np.linalg.norm(np.asarray(imu.gyro, dtype=float),
                             axis=1)  # telemetry gyro is already deg/s
        step = max(1, len(t) // 1200)
        self.telemetry_ready.emit(t[::step].tolist(), mag[::step].tolist())


class Main(QMainWindow):
    """Top bar, preview + job panel, filmstrip, gyro, transport."""

    DEG_PER_PX = 0.05

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Steadycut clip-picker")
        self.resize(1100, 720)
        self.state = B.GuiState()
        self.job_state: B.GuiState | None = None
        self.total = 0.0
        self.store: PX.SegmentStore | None = None
        self.builder: PX.SegmentBuilder | None = None
        self.thumbs: PX.ThumbStrip | None = None
        self.player: PX.SegmentPlayer | None = None
        self.cursor = PX.PlaybackCursor()
        self.thumb_qimages: dict[float, object] = {}
        self.thumbs_saved = 0
        self.proxy_pitch = -12.0
        self.camera: dict | None = None
        self.profile = None
        self.ident_label = "no media"
        self.fps = 29.97
        self.play_t = 0.0
        self.playing = False
        self.out_dirty = False
        self.rendered_fp: str | None = None
        self.proc = None
        self.proc_fh = None
        self.log_reader = None
        self.log_path: Path | None = None
        self.loader: LoaderThread | None = None
        self._drag_anchor: tuple | None = None
        self._build()
        self._tick = QTimer(self)
        self._tick.setInterval(66)
        self._tick.timeout.connect(self._on_tick)
        self._poll = QTimer(self)
        self._poll.setInterval(120)
        self._poll.timeout.connect(self._poll_job)
        self._proxypoll = QTimer(self)
        self._proxypoll.setInterval(200)
        self._proxypoll.timeout.connect(self._poll_proxy)

    # -- layout ---------------------------------------------------------
    def _build(self):
        from steadycut.framing.policies import policy_names

        self.topbar = QLabel("no media — File > Open media (.insv)")
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.addWidget(self.topbar)

        mid = QHBoxLayout()
        left = QVBoxLayout()
        self.prev = Preview(self)
        self.prev.dragged.connect(self._on_aim_drag)
        left.addWidget(self.prev, 1)
        self.aimlbl = QLabel("aim: unset → policy owns it")
        self.aimlbl.setStyleSheet("color:#333")
        self.aimlbl.setWordWrap(True)
        left.addWidget(self.aimlbl)
        reset_row = QHBoxLayout()
        self.reset_aim = QPushButton("Reset aim")
        self.reset_aim.clicked.connect(self._on_reset_aim)
        reset_row.addWidget(self.reset_aim)
        reset_row.addStretch(1)
        left.addLayout(reset_row)
        mid.addLayout(left, 1)

        job = QGroupBox("Job")
        jl = QVBoxLayout(job)
        form = QFormLayout()
        self.e_in = QLineEdit("00:10.0")
        self.e_out = QLineEdit("00:25.0")
        self.e_in.editingFinished.connect(self._on_edits)
        self.e_out.editingFinished.connect(self._on_edits)
        form.addRow("in", self.e_in)
        form.addRow("out", self.e_out)
        self.framing = QComboBox()
        self.framing.addItems(policy_names())
        self.framing.setCurrentText(self.state.framing)
        self.framing.currentTextChanged.connect(self._on_option)
        form.addRow("framing", self.framing)
        self.aspect = QComboBox()
        self.aspect.addItems(["4:3 — 960x720", "9:16 tall"])
        self.aspect.currentIndexChanged.connect(self._on_aspect)
        form.addRow("aspect", self.aspect)
        self.stitch = QComboBox()
        self.stitch.addItems(["legacy", "blend"])
        self.stitch.currentTextChanged.connect(self._on_stitch)
        form.addRow("stitch", self.stitch)
        self.preset = QComboBox()
        self.preset.addItems(list(PRESETS))
        self.preset.currentTextChanged.connect(self._on_option)
        form.addRow("preset", self.preset)
        self.no_2d = QCheckBox("skip 2D pass")
        self.no_2d.toggled.connect(self._on_option)
        form.addRow("", self.no_2d)
        self.sync_lbl = QLabel("sync: auto (measured at render)")
        form.addRow("sync", self.sync_lbl)
        self.aimv = QLabel("yaw — · pitch — (unset → policy)")
        self.aimv.setWordWrap(True)
        form.addRow("aim", self.aimv)
        self.valid_lbl = QLabel("validity: no media")
        self.valid_lbl.setWordWrap(True)
        form.addRow("validity", self.valid_lbl)
        self.fp_lbl = QLabel("fingerprint: —")
        form.addRow("fingerprint", self.fp_lbl)
        jl.addLayout(form)
        self.e_destination = QLineEdit()
        self.e_destination.setPlaceholderText("output .mp4 (auto)")
        self.e_destination.editingFinished.connect(self._on_out_edit)
        jl.addWidget(self.e_destination)
        jl.addStretch(1)
        self.render_btn = QPushButton("Render clip")
        self.render_btn.clicked.connect(self.on_render)
        jl.addWidget(self.render_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.on_cancel)
        self.cancel_btn.setEnabled(False)
        jl.addWidget(self.cancel_btn)
        self.copy_btn = QPushButton("Copy CLI command")
        self.copy_btn.clicked.connect(self.on_copy)
        jl.addWidget(self.copy_btn)
        mid.addWidget(job)
        outer.addLayout(mid, 1)

        self.strip = FilmStrip(self)
        self.strip.seeked.connect(self._on_seek)
        self.strip.ranged.connect(self._on_range)
        outer.addWidget(self.strip)
        self.gyro = GyroStrip(self)
        outer.addWidget(self.gyro)

        trow = QHBoxLayout()
        self.play = QPushButton("▶")
        self.play.setFixedWidth(40)
        self.play.clicked.connect(self.toggle)
        trow.addWidget(self.play)
        self.tc = QLabel("playhead --")
        trow.addWidget(self.tc)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        trow.addWidget(self.progress, 1)
        self.status_lbl = QLabel("no media")
        trow.addWidget(self.status_lbl)
        outer.addLayout(trow)

        self.refusal = QLabel()
        self.refusal.setWordWrap(True)
        self.refusal.setStyleSheet(
            "color:#7a1010; background:#fde8e8; padding:6px")
        self.refusal.setVisible(False)
        outer.addWidget(self.refusal)
        self.logview = QPlainTextEdit()
        self.logview.setReadOnly(True)
        self.logview.setMaximumBlockCount(400)
        self.logview.setPlaceholderText("render log appears here")
        self.logview.setMinimumHeight(90)
        outer.addWidget(self.logview)

        menu = self.menuBar().addMenu("File")
        open_act = QAction("Open media…", self)
        open_act.triggered.connect(self._on_open_dialog)
        menu.addAction(open_act)
        job_act = QAction("Load job JSON…", self)
        job_act.triggered.connect(self._on_load_job)
        menu.addAction(job_act)

    # -- open / loader --------------------------------------------------
    def _on_open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open media", "", "Insta360 (*.insv);;All (*)")
        if path:
            self.open_media(path)

    def open_media(self, path: str, out: str | None = None,
                   hint: str | None = None):
        if self.loader is not None and self.loader.isRunning():
            self._status("loader busy — wait for the proxy, then retry")
            return
        self._stop_playback()
        self._close_proxy_workers()
        self.state = B.GuiState(source=path)
        self.job_state = None
        self.rendered_fp: str | None = None
        self.out_dirty = out is not None
        if out:
            self.state.out = out
        self.store = None
        self.camera = None
        self.profile = None
        self.thumb_qimages = {}
        self.thumbs_saved = 0
        self.refusal.setVisible(False)
        self.logview.clear()
        self.prev.show_frame(None)
        self.prev.overlay = "loading …"
        self.prev.update()
        self.strip.thumbs = []
        self.gyro.set_series([], [])
        self._status(f"opening {Path(path).name} …")
        stitch = self.stitch.currentText()
        self.loader = LoaderThread(path, hint, stitch)
        self.loader.note.connect(self._status)
        self.loader.store_ready.connect(self._on_store_ready)
        self.loader.telemetry_ready.connect(self._on_gyro)
        self.loader.failed.connect(self._on_open_failed)
        self.loader.start()

    def _on_open_failed(self, msg: str, hint: str):
        self._status("open refused")
        self._refuse(msg, hint)
        self.prev.overlay = "no media"
        self.prev.update()

    def _close_proxy_workers(self):
        """Stop fill/thumbs/player: group-kill builds, join, release caps."""
        if self.builder is not None:
            self.builder.stop(cancel_in_flight=True)
            self.builder = None
        if self.thumbs is not None:
            self.thumbs.stop()
            self.thumbs = None
        if self.player is not None:
            self.player.close()
            self.player = None
        if hasattr(self, "_proxypoll"):
            self._proxypoll.stop()

    def _on_store_ready(self, store, manifest: dict, pitch: float,
                        info: dict, total: float):
        loader = self.loader
        self.store = store
        self.proxy_pitch = pitch
        self.total = total
        self.fps = float(store.fps)
        if loader is not None and not loader._proxy_only:
            self.camera = loader.camera
            self.profile = getattr(loader, "_profile", None)
            ident = loader._ident
            self.ident_label = (
                f"{ident.get('model')} via {ident.get('via')}")
            in_t = min(10.0, max(0.0, total - 1.0))
            out_t = min(in_t + 15.0, total)
            self.strip.set_span(0.0, total)
            self._set_window(in_t, out_t, in_t)
            self.e_destination.setText(
                self.state.out or B.default_out(
                    self.state.source, in_t, self.state.tall))
        elif getattr(self, "_pending_window", None) is not None:
            in_t, dur, play_t = self._pending_window
            self._pending_window = None
            self.strip.set_span(0.0, total)
            self._set_window(in_t, in_t + dur,
                             min(max(play_t, in_t), in_t + dur))
        usable = info.get("usable", True)
        self.prev.proxy_note = ("proxy · 720p · ready" if usable
                                else "proxy · 720p · pitch fallback")
        self.player = PX.SegmentPlayer(store)
        self.builder = PX.SegmentBuilder(store)
        self.builder.request_window(self.state.start,
                                    self.state.start + self.state.dur)
        self.builder.request_playhead(self.play_t)
        self.builder.start()
        self.thumbs = PX.ThumbStrip(
            self.state.source, PX.thumb_times(total), pitch,
            sheet=store.root / PX.THUMB_SHEET,
            index=store.root / PX.THUMB_INDEX)
        self.thumbs.start()
        self._proxypoll.start()
        self._show_at(self.play_t)
        self._refresh_meta()
        self._status("proxy store ready — building playhead segment …")

    def _poll_proxy(self):
        """Drain segment/thumb worker events into the strip and preview."""
        if self.builder is not None:
            if self.player is not None:
                self.builder.note_open_segments(
                    self.player.open_segments())
            for kind, i, ok, detail in self.builder.drain():
                if kind != "segment":
                    continue
                if not ok:
                    self._status(f"segment {i} failed ({detail})")
                    continue
                if self.store is not None and i == PX.segment_at_time(
                        self.play_t, self.store.fps,
                        self.store.frames_per_segment):
                    self._show_at(self.play_t)
        if self.thumbs is not None:
            fresh = 0
            for event in self.thumbs.drain():
                if event[0] != "thumb":
                    continue
                _, t, jpg = event
                img = QImage.fromData(jpg, "JPG")
                if img.isNull():
                    continue
                self.thumb_qimages[float(t)] = img
                fresh += 1
            if fresh:
                self.strip.thumbs = sorted(
                    ((t, img) for t, img in self.thumb_qimages.items()),
                    key=lambda kv: kv[0])
                self.strip.update()
                if len(self.thumb_qimages) - self.thumbs_saved >= 10:
                    self.thumbs.save_sheet()
                    self.thumbs_saved = len(self.thumb_qimages)
        if self.builder is None and self.thumbs is None:
            self._proxypoll.stop()

    def _on_gyro(self, times, values):
        self.gyro.set_series(times, values)
        self.gyro.set_window(self.play_t, self.state.start,
                             self.state.start + self.state.dur)

    def _on_stitch(self, text: str):
        self._panel_to_state()
        self._refresh_meta()
        if (self.store is None or self.camera is None
                or self.loader is not None and self.loader.isRunning()):
            return
        self._status("stitch changed — reopening proxy store …")
        play_t, in_t, dur = self.play_t, self.state.start, self.state.dur
        self._close_proxy_workers()
        self.loader = LoaderThread(
            self.state.source, None, text, camera=self.camera,
            profile=self.profile, total=self.total, proxy_only=True)
        self.loader.note.connect(self._status)
        self.loader.store_ready.connect(self._on_store_ready)
        self.loader.failed.connect(self._on_open_failed)
        self.loader.start()
        self._pending_window = (in_t, dur, play_t)

    def _status(self, text: str):
        self.status_lbl.setText(text)

    def _refuse(self, msg: str, hint: str):
        self.refusal.setText(f"REFUSED: {msg} — next: {hint}")
        self.refusal.setVisible(True)

    def _clear_refusal(self):
        self.refusal.setVisible(False)

    # -- panel <-> state ------------------------------------------------
    def _panel_to_state(self):
        s = self.state
        try:
            s.start = B.parse_time(self.e_in.text())
            s.dur = max(0.5, B.parse_time(self.e_out.text()) - s.start)
        except ValueError:
            pass
        s.framing = self.framing.currentText()
        s.tall = self.aspect.currentIndex() == 1
        s.stitch = self.stitch.currentText()
        s.preset = self.preset.currentText()
        s.no_2d = self.no_2d.isChecked()
        dest = self.e_destination.text().strip()
        s.out = dest or B.default_out(s.source, s.start, s.tall)
        self.e_destination.setText(s.out)

    def _set_window(self, in_t: float, out_t: float, play_t: float):
        self.e_in.setText(B.fmt_time(in_t))
        self.e_out.setText(B.fmt_time(out_t))
        self.play_t = play_t
        self.state.start = in_t
        self.state.dur = max(0.5, out_t - in_t)
        if self.state.source and not self.out_dirty:
            self.e_destination.setText(
                B.default_out(self.state.source, in_t, self.state.tall))
        self.strip.set_window(play_t, in_t, out_t)
        self.gyro.set_window(play_t, in_t, out_t)
        self.tc.setText(f"playhead {B.fmt_time(play_t)}  |  "
                        f"out {B.fmt_time(out_t)}")
        self._refresh_meta()

    def _on_option(self, *args):
        self._panel_to_state()
        self._refresh_meta()

    def _on_aspect(self, *args):
        self._panel_to_state()
        if not self.out_dirty:
            self.e_destination.setText(
                B.default_out(self.state.source, self.state.start,
                              self.state.tall))
            self._panel_to_state()
        self._refresh_meta()

    def _on_out_edit(self):
        self.out_dirty = bool(self.e_destination.text().strip())
        self._panel_to_state()
        self._refresh_meta()

    def _on_edits(self):
        try:
            in_t = B.parse_time(self.e_in.text())
            out_t = B.parse_time(self.e_out.text())
        except ValueError:
            self._set_window(self.state.start,
                             self.state.start + self.state.dur, self.play_t)
            return
        end = self.total if self.total > 0 else None
        if end is None:
            # No media open: keep the typed window (ordering only); bounds
            # apply once the probe reports the real duration.
            in_t = max(0.0, in_t)
            out_t = max(out_t, in_t + 0.5)
        else:
            end = max(0.5, end)
            in_t = min(max(in_t, 0.0), end - 0.5)
            out_t = min(max(out_t, in_t + 0.5), end)
        self._set_window(in_t, out_t, min(max(self.play_t, in_t), out_t))
        if self.builder is not None:
            self.builder.request_window(in_t, out_t)
        self._show_at(self.play_t)

    def _refresh_meta(self):
        self._panel_to_state()
        fp = B.fingerprint_of(self.state)
        if self.rendered_fp is None:
            self.fp_lbl.setText(f"{fp} · unrendered")
        elif fp == self.rendered_fp:
            self.fp_lbl.setText(f"{fp} · current")
        else:
            self.fp_lbl.setText(f"{fp} · modified since render")
        ok, reason = B.check_state(
            self.state, duration_fn=(lambda s: self.total)
            if self.total else None)
        if not self.state.source:
            self.valid_lbl.setText("validity: no media")
        elif ok:
            self.valid_lbl.setText("● gates pass")
        else:
            self.valid_lbl.setText(f"○ {reason}")
        total = f" · {self.total:.0f}s" if self.total else ""
        self.topbar.setText(
            f"{Path(self.state.source).name if self.state.source else '—'}"
            f"{total} · {self.ident_label}")
        self._refresh_aim()

    def _refresh_aim(self):
        yaw, pitch = self.state.yaw_bias, self.state.pitch
        if not yaw and pitch is None:
            self.aimlbl.setText("aim: unset → policy owns it")
            self.aimv.setText("yaw — · pitch — (policy)")
        else:
            how = (f"pitch {pitch:+.1f}° (manual)" if pitch is not None
                   else "pitch auto")
            self.aimlbl.setText(f"aim override: yaw-bias {yaw:+.1f}°, {how}")
            self.aimv.setText(f"yaw {yaw:+.1f}° · {how}")

    def _on_aim_drag(self, dx_px: float, dy_px: float):
        if self._drag_anchor is None:
            self._drag_anchor = (self.state.yaw_bias,
                                 (self.state.pitch if self.state.pitch
                                  is not None else self.proxy_pitch),
                                 dx_px, dy_px)
            return
        yaw0, pitch0, x0, y0 = self._drag_anchor
        self.state.yaw_bias = round(yaw0 + (dx_px - x0) * self.DEG_PER_PX, 2)
        self.state.pitch = round(pitch0 - (dy_px - y0) * self.DEG_PER_PX, 1)
        self._refresh_aim()
        self._refresh_meta()

    def _on_reset_aim(self):
        self._drag_anchor = None
        self.state.yaw_bias = 0.0
        self.state.pitch = None
        self._refresh_aim()
        self._refresh_meta()

    def mouseReleaseEvent(self, ev):
        self._drag_anchor = None
        super().mouseReleaseEvent(ev)

    # -- preview / transport (segmented) -------------------------------
    def _show_at(self, t: float):
        """Show the proxy frame for media time t (POS_FRAMES, never MSEC).

        Unbuilt gaps hold the last frame with a "building …" tag instead
        of hanging the UI; demand for the missing segment is already
        queued by _on_seek.
        """
        if self.player is None or self.store is None:
            return
        frame, status = self.player.frame_at(t)
        if frame is None:
            self.prev.overlay = (f"{Path(self.state.source).name} · "
                                 f"{B.fmt_time(t)} · building …")
            self.prev.show_frame(None)
            return
        if status == "building":
            self.prev.overlay = (f"{Path(self.state.source).name} · "
                                 f"{B.fmt_time(t)} · building …")
        else:
            self.prev.overlay = (f"{Path(self.state.source).name} · "
                                 f"{B.fmt_time(t)}")
        self.prev.show_frame(qimage_from_bgr(frame))

    def _on_seek(self, t: float):
        self.play_t = t
        self._set_window(self.state.start,
                         self.state.start + self.state.dur, t)
        if self.store is not None and not self.store.is_present(
                PX.segment_at_time(t, self.store.fps,
                                   self.store.frames_per_segment)):
            self.prev.overlay = (f"{Path(self.state.source).name} · "
                                 f"{B.fmt_time(t)} · building …")
            self.prev.update()
            if self.builder is not None:
                self.builder.request_playhead(t)
        self._show_at(t)

    def _on_range(self, in_t: float, out_t: float):
        self._set_window(in_t, out_t,
                         min(max(self.play_t, in_t), out_t))
        if self.builder is not None:
            self.builder.request_window(in_t, out_t)
        self._show_at(self.play_t)

    def toggle(self):
        if self.player is None:
            return
        self.playing = not self.playing
        self.play.setText("⏸" if self.playing else "▶")
        if self.playing:
            self._tick.start()
        else:
            self._tick.stop()

    def _stop_playback(self):
        self.playing = False
        self.play.setText("▶")
        self._tick.stop()

    def _on_tick(self):
        if not self.playing or self.player is None or self.store is None:
            return
        in_t = self.state.start
        out_t = self.state.start + self.state.dur
        self.cursor.play_t, self.cursor.in_t, self.cursor.out_t = \
            self.play_t, in_t, out_t
        frame, status = self.player.frame_at(
            min(self.play_t + self._tick.interval() / 1000.0, out_t))
        if status != "ok":
            # Unbuilt gap inside [in, out]: hold the last frame, never
            # wrap to p0 and never hang the UI. The demand for this
            # segment is already queued; playback resumes when it lands.
            self.prev.overlay = (f"{Path(self.state.source).name} · "
                                 f"{B.fmt_time(self.play_t)} · building …")
            self.prev.update()
            return
        play_t, _ = self.cursor.advance(
            self._tick.interval() / 1000.0, status)
        self.play_t = play_t
        self._set_window(in_t, out_t, self.play_t)
        self._show_at(self.play_t)

    def keyPressEvent(self, ev):
        from PySide6.QtWidgets import QLineEdit as _LE

        if isinstance(QApplication.focusWidget(), _LE):
            super().keyPressEvent(ev)
            return
        step = 1.0 / self.fps
        if ev.key() == Qt.Key.Key_I:
            self._set_window(min(self.play_t, self.state.start
                                 + self.state.dur - 0.5),
                             self.state.start + self.state.dur, self.play_t)
        elif ev.key() == Qt.Key.Key_O:
            self._set_window(self.state.start,
                             max(self.play_t, self.state.start + 0.5),
                             self.play_t)
        elif ev.key() == Qt.Key.Key_Space:
            self.toggle()
        elif ev.key() == Qt.Key.Key_Left:
            self._on_seek(max(0.0, self.play_t - step))
        elif ev.key() == Qt.Key.Key_Right:
            self._on_seek(min(self.total or self.play_t,
                              self.play_t + step))
        else:
            super().keyPressEvent(ev)

    # -- render ---------------------------------------------------------
    def on_render(self):
        self._panel_to_state()
        ok, reason = B.check_state(self.state)
        if not ok:
            self._refuse(reason, "adjust in/out, then Render again")
            return
        if self.proc is not None and self.proc.poll() is None:
            self._status("a render is already running")
            return
        self._clear_refusal()
        self.job_state = B.GuiState(**vars(self.state))
        out = Path(self.state.out)
        env_path = out.with_suffix(".job.json")
        base = Path(self.state.source).parent
        B.write_job(self.job_state, env_path, base=base)
        self.log_path = out.with_suffix(".render.log")
        self.logview.clear()
        self.logview.appendPlainText(
            f"$ {' '.join(B.job_argv(self.job_state, env_path))}")
        self.proc, self.proc_fh = R.start_job(
            B.job_argv(self.job_state, env_path), self.log_path)
        self.log_reader = open(self.log_path, "rb")
        self.render_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        if self.builder is not None:
            # Pause fill during renders: stop starting new segments, let
            # the in-flight one finish (bounded by segment size).
            self.builder.set_rendering(True)
        self._status("rendering …")
        self._poll.start()

    def _poll_job(self):
        if self.proc is None:
            return
        chunk = self.log_reader.read() if self.log_reader else b""
        if chunk:
            text = chunk.decode("utf-8", "replace")
            self.logview.appendPlainText(text.rstrip())
            for line in text.splitlines():
                stage = B.stage_of(line)
                if stage == "done":
                    self.progress.setRange(0, 100)
                    self.progress.setValue(100)
                elif stage is not None:
                    frac = B.stage_fraction(stage)
                    if frac is None:
                        self.progress.setRange(0, 0)
                    else:
                        self.progress.setRange(0, 100)
                        self.progress.setValue(int(frac * 100))
                    self._status(f"rendering: {stage} …")
        if self.proc.poll() is None:
            return
        self._poll.stop()
        self._finish_job(self.proc.poll())

    def _finish_job(self, rc: int):
        rest = self.log_reader.read() if self.log_reader else b""
        if rest:
            self.logview.appendPlainText(
                rest.decode("utf-8", "replace").rstrip())
        if self.log_reader:
            self.log_reader.close()
            self.log_reader = None
        if self.proc_fh:
            self.proc_fh.close()
            self.proc_fh = None
        log_text = B.tail_bytes(self.log_path) if self.log_path else ""
        out = Path(self.job_state.out) if self.job_state else None
        if rc == 0:
            self.rendered_fp = B.fingerprint_of(self.job_state)
            cert_path = out.with_suffix(".cert.json") if out else None
            extra = (f" · certificate {cert_path.name}"
                     if cert_path and cert_path.exists() else "")
            self._status(f"wrote {out}{extra}")
            self._refresh_cert_sync(cert_path)
        else:
            refusals = B.extract_refusals(log_text)
            if refusals:
                msg, hint = refusals[-1]
                self._refuse(msg, hint or "see the render log below")
            else:
                self._refuse(f"render exited {rc}",
                             "see the render log below")
            self._status(f"render failed (exit {rc})")
        self._refresh_meta()
        self.proc = None
        self.progress.setVisible(False)
        self.render_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if self.builder is not None:
            self.builder.set_rendering(False)

    def _refresh_cert_sync(self, cert_path):
        try:
            cert = json.loads(Path(cert_path).read_text())
        except (OSError, ValueError, TypeError):
            return
        sync = cert.get("sync", {})
        if sync:
            self.sync_lbl.setText(
                f"sync: {sync.get('offset_applied', 0):+.0f}ms "
                f"r {sync.get('r', 0):+.2f} (rendered)")

    def on_cancel(self):
        if self.proc is None or self.proc.poll() is not None:
            return
        self.cancel_btn.setEnabled(False)
        self._status("cancelling (TERM the group) …")
        threading.Thread(target=self._cancel_blocking,
                         daemon=True).start()

    def _cancel_blocking(self):
        outcome = R.cancel(self.proc, grace_s=5.0)
        self._status(f"cancelling … ({outcome})")

    def on_copy(self):
        self._panel_to_state()
        cmd = B.literal_shell(self.state)
        QApplication.clipboard().setText(cmd)
        self.logview.appendPlainText(f"$ {cmd}")
        self._status("copied equivalent CLI command")

    def _on_load_job(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load job JSON", "", "Job (*.json)")
        if not path:
            return
        from steadycut.job import load_envelope

        try:
            env = load_envelope(path)
        except ValueError as exc:
            QMessageBox.warning(self, "Bad job envelope",
                                f"{exc}\nRe-export the job from the GUI.")
            return
        self._panel_to_state()
        B.fill_state_from_args(self.state, env["args"])
        src = self.state.source
        self.framing.setCurrentText(self.state.framing)
        self.aspect.setCurrentIndex(1 if self.state.tall else 0)
        self.stitch.setCurrentText(self.state.stitch)
        self.preset.setCurrentText(self.state.preset)
        self.no_2d.setChecked(self.state.no_2d)
        self.e_destination.setText(self.state.out)
        if src and Path(src).exists() and self.store is None:
            self.open_media(src)
        else:
            self._set_window(self.state.start,
                             self.state.start + self.state.dur,
                             self.state.start)
            self._refresh_meta()


def main(argv=None) -> int:
    """Entry point: `steadycut-gui [source] [--out]`."""
    import argparse

    ap = argparse.ArgumentParser(description="Steadycut clip-picker")
    ap.add_argument("source", nargs="?", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    app = QApplication(sys.argv[:1])
    w = Main()
    w.show()
    if args.source:
        w.open_media(args.source, out=args.out)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
