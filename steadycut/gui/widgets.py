"""Display widgets: preview, filmstrip, gyro rate.

Display plus gestures only: the main window owns state, the CLI owns
pixels. Times here are absolute media seconds; the filmstrip maps its
span [p0, p1] to pixels and back.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget


class Preview(QWidget):
    """Current proxy frame; drag re-aims (yaw right, pitch up)."""

    dragged = Signal(float, float)  # dx_px, dy_px from the press anchor

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame: QImage | None = None
        self.overlay = "no media"
        self.proxy_note = ""
        self._anchor = None
        self.setMinimumHeight(240)

    def show_frame(self, image: QImage | None) -> None:
        self.frame = image
        self.update()

    def mousePressEvent(self, ev):
        if (ev.button() == Qt.MouseButton.LeftButton
                and self.frame is not None):
            self._anchor = (ev.position().x(), ev.position().y())

    def mouseMoveEvent(self, ev):
        if self._anchor is not None:
            x0, y0 = self._anchor
            self.dragged.emit(ev.position().x() - x0, ev.position().y() - y0)

    def mouseReleaseEvent(self, ev):
        self._anchor = None

    def paintEvent(self, ev):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#05070c"))
        if self.frame is None:
            p.setPen(QColor("#8a8f98"))
            p.drawText(12, h // 2, self.overlay)
        else:
            scaled = self.frame.scaled(
                w, h, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            p.drawImage((w - scaled.width()) // 2,
                        (h - scaled.height()) // 2, scaled)
            p.setPen(QPen(QColor("#ff5a5a"), 1))
            p.drawLine(w // 2, int(h * 0.44), w // 2, int(h * 0.56))
            p.drawLine(int(w * 0.46), h // 2, int(w * 0.54), h // 2)
            p.setFont(QFont("", 10))
            p.fillRect(6, 4, 200, 20, QColor(0, 0, 0, 160))
            p.setPen(QColor("#e6e6e6"))
            p.drawText(10, 18, self.overlay)
            if self.proxy_note:
                tag_w = p.fontMetrics().horizontalAdvance(self.proxy_note)
                tag_w = min(tag_w + 12, w - 12)
                p.fillRect(w - 6 - tag_w, 4, tag_w, 20,
                           QColor(0, 0, 0, 160))
                p.setPen(QColor("#8ab4f8"))
                p.drawText(w - tag_w, 4, tag_w - 6, 20,
                           Qt.AlignmentFlag.AlignRight
                           | Qt.AlignmentFlag.AlignVCenter,
                           self.proxy_note)


class FilmStrip(QWidget):
    """Full-timeline thumbnails with draggable in/out handles + playhead."""

    seeked = Signal(float)
    ranged = Signal(float, float)  # in_t, out_t

    def __init__(self, parent=None):
        super().__init__(parent)
        self.p0, self.p1 = 0.0, 1.0
        self.play_t = 0.0
        self.in_t, self.out_t = 0.0, 1.0
        self.thumbs: list = []  # (t, QImage) across the span
        self._drag = None
        self.setMinimumHeight(72)

    def set_span(self, p0: float, p1: float) -> None:
        self.p0, self.p1 = p0, max(p1, p0 + 0.01)
        self.update()

    def set_window(self, play_t: float, in_t: float,
                   out_t: float) -> None:
        self.play_t, self.in_t, self.out_t = play_t, in_t, out_t
        self.update()

    def x2t(self, x: float) -> float:
        return self.p0 + x / max(1, self.width()) * (self.p1 - self.p0)

    def t2x(self, t: float) -> float:
        return (t - self.p0) / (self.p1 - self.p0) * self.width()

    def paintEvent(self, ev):
        p = QPainter(self)
        w, h = self.width(), self.height()
        th = h - 18
        if self.thumbs:
            cell = w / len(self.thumbs)
            for i, (_, img) in enumerate(self.thumbs):
                p.drawImage(int(i * cell), 0,
                            img.scaled(
                                int(cell) + 1, th,
                                Qt.AspectRatioMode.IgnoreAspectRatio,
                                Qt.TransformationMode.SmoothTransformation),
                            0, 0, -1, -1)
        else:
            for i in range(16):
                c = QColor("#3d4a2c") if i % 2 else QColor("#2c3e50")
                p.fillRect(int(i * w / 16), 0, int(w / 16) + 1, th, c)
        p.fillRect(int(self.t2x(self.in_t)), 0,
                   int(self.t2x(self.out_t) - self.t2x(self.in_t)), h,
                   QColor(47, 111, 237, 70))
        p.setPen(QPen(QColor("#2f6fed"), 2))
        p.drawLine(int(self.t2x(self.in_t)), 0,
                   int(self.t2x(self.in_t)), h)
        p.drawLine(int(self.t2x(self.out_t)), 0,
                   int(self.t2x(self.out_t)), h)
        p.setPen(QPen(QColor("#ff5a5a"), 2))
        p.drawLine(int(self.t2x(self.play_t)), 0,
                   int(self.t2x(self.play_t)), h)
        p.setPen(QColor("#888"))
        p.setFont(QFont("", 8))
        span = self.p1 - self.p0
        step = 10.0 if span > 30 else 5.0 if span > 12 else 1.0
        tick = (int(self.p0 / step) + 1) * step
        while tick < self.p1:
            p.drawText(int(self.t2x(tick)) + 2, h - 4,
                       f"{int(tick // 60):02d}:{tick % 60:04.1f}")
            tick += step

    def mousePressEvent(self, ev):
        x = ev.position().x()
        if abs(x - self.t2x(self.in_t)) < 6:
            self._drag = "in"
        elif abs(x - self.t2x(self.out_t)) < 6:
            self._drag = "out"
        else:
            t = min(self.out_t, max(self.in_t, self.x2t(x)))
            self.seeked.emit(t)
        self.update()

    def mouseMoveEvent(self, ev):
        if not self._drag:
            return
        v = self.x2t(ev.position().x())
        if self._drag == "in":
            self.ranged.emit(min(v, self.out_t - 0.5), self.out_t)
        else:
            self.ranged.emit(self.in_t, max(v, self.in_t + 0.5))

    def mouseReleaseEvent(self, ev):
        self._drag = None


class GyroStrip(QWidget):
    """Raw gyro magnitude over the proxy span; window shade + playhead."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.times: list = []
        self.values: list = []
        self.in_t, self.out_t, self.play_t = 0.0, 1.0, 0.0
        self.setMinimumHeight(48)

    def set_series(self, times, values) -> None:
        self.times = list(times)
        self.values = list(values)
        self.update()

    def set_window(self, play_t: float, in_t: float,
                   out_t: float) -> None:
        self.play_t, self.in_t, self.out_t = play_t, in_t, out_t
        self.update()

    def paintEvent(self, ev):
        from PySide6.QtGui import QPolygonF
        from PySide6.QtCore import QRectF

        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#0b0d10"))
        if not self.times:
            p.setPen(QColor("#5a5e66"))
            p.drawText(12, h // 2 + 4, "gyro: loading telemetry …")
            return
        t0, t1 = self.times[0], self.times[-1]
        span = max(t1 - t0, 1e-6)
        vmax = max(1.0, max(abs(v) for v in self.values) * 1.1)

        def x_of(t):
            return (t - t0) / span * w

        p.fillRect(int(x_of(self.in_t)), 0,
                   int(x_of(self.out_t) - x_of(self.in_t)), h,
                   QColor(47, 111, 237, 40))
        pts = QPolygonF()
        for t, v in zip(self.times, self.values):
            y = h / 2 - v / vmax * (h / 2 - 2)
            pts.append(QRectF(x_of(t), y, 1, 1).center())
        p.setPen(QPen(QColor("#2f6fed"), 1))
        p.drawPolyline(pts)
        p.setPen(QPen(QColor("#ff5a5a"), 1))
        p.drawLine(0, h // 2, w, h // 2)
        p.setPen(QPen(QColor("#ff5a5a"), 2))
        p.drawLine(int(x_of(self.play_t)), 0,
                   int(x_of(self.play_t)), h)
