"""Custom widgets: two-handle range slider and the live image view."""
from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QSpinBox, QWidget

from . import theme


class RangeSlider(QWidget):
    """Slider with two handles. With wrap=True the lower value may exceed the upper
    one, selecting the range through zero (needed for red hues)."""

    rangeChanged = Signal(int, int)
    MARGIN = 8

    def __init__(self, maximum: int, wrap: bool = False, parent=None):
        super().__init__(parent)
        self.maximum, self.wrap = maximum, wrap
        self.low, self.high = 0, maximum
        self._stops: list[QColor] = [QColor("#222"), QColor("#ddd")]
        self._dragging: str | None = None
        self.setMinimumHeight(22)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)

    def sizeHint(self) -> QSize:
        return QSize(160, 22)

    def set_stops(self, colors: list[QColor]) -> None:
        self._stops = colors
        self.update()

    def set_range(self, low: int, high: int) -> None:
        self.low, self.high = self._normalized(low, high)
        self.update()

    def _normalized(self, low: int, high: int) -> tuple[int, int]:
        low = max(0, min(self.maximum, low))
        high = max(0, min(self.maximum, high))
        if not self.wrap and low > high:
            low, high = high, low
        return low, high

    def _x_of(self, value: int) -> float:
        span = max(1, self.width() - 2 * self.MARGIN)
        return self.MARGIN + span * value / self.maximum

    def _value_at(self, x: float) -> int:
        span = max(1, self.width() - 2 * self.MARGIN)
        return max(0, min(self.maximum, round((x - self.MARGIN) * self.maximum / span)))

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        track = QRectF(self.MARGIN, self.height() / 2 - 4, self.width() - 2 * self.MARGIN, 8)
        gradient = QLinearGradient(track.left(), 0, track.right(), 0)
        for index, color in enumerate(self._stops):
            gradient.setColorAt(index / max(1, len(self._stops) - 1), color)
        painter.setOpacity(1.0 if self.isEnabled() else 0.35)
        painter.setPen(Qt.NoPen)
        painter.setBrush(gradient)
        painter.drawRoundedRect(track, 4, 4)

        # Darken what is outside the selected range.
        painter.setBrush(QColor(10, 13, 16, 190))
        low_x, high_x = self._x_of(self.low), self._x_of(self.high)
        if self.low <= self.high:
            painter.drawRect(QRectF(track.left(), track.top(), low_x - track.left(), track.height()))
            painter.drawRect(QRectF(high_x, track.top(), track.right() - high_x, track.height()))
        else:
            painter.drawRect(QRectF(high_x, track.top(), low_x - high_x, track.height()))

        painter.setBrush(QColor(theme.TEXT))
        painter.setPen(QPen(QColor(theme.BG), 2))
        for x in (low_x, high_x):
            painter.drawEllipse(QPointF(x, self.height() / 2), 6.5, 6.5)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        x = event.position().x()
        self._dragging = "low" if abs(x - self._x_of(self.low)) <= abs(x - self._x_of(self.high)) else "high"
        self._move_to(x)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            self._move_to(event.position().x())

    def mouseReleaseEvent(self, _event) -> None:
        self._dragging = None

    def _move_to(self, x: float) -> None:
        value = self._value_at(x)
        low, high = self.low, self.high
        if self._dragging == "low":
            low = value if self.wrap else min(value, high)
        else:
            high = value if self.wrap else max(value, low)
        if (low, high) != (self.low, self.high):
            self.low, self.high = low, high
            self.update()
            self.rangeChanged.emit(low, high)


class RangeRow(QWidget):
    """Label, range slider and two spin boxes kept in sync."""

    changed = Signal()

    def __init__(self, label: str, maximum: int, wrap: bool = False, tooltip: str = "", parent=None):
        super().__init__(parent)
        self.slider = RangeSlider(maximum, wrap)
        self.low_spin, self.high_spin = self._spin(maximum), self._spin(maximum)
        name = QLabel(label)
        name.setFixedWidth(14)
        name.setToolTip(tooltip)
        self.slider.setToolTip(tooltip)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(name)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.low_spin)
        layout.addWidget(self.high_spin)
        self.slider.rangeChanged.connect(self._from_slider)
        self.low_spin.valueChanged.connect(self._from_spins)
        self.high_spin.valueChanged.connect(self._from_spins)

    @staticmethod
    def _spin(maximum: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setObjectName("compact")
        spin.setRange(0, maximum)
        spin.setAlignment(Qt.AlignCenter)
        spin.setFixedWidth(44)
        return spin

    def values(self) -> tuple[int, int]:
        return self.slider.low, self.slider.high

    def set_values(self, low: int, high: int) -> None:
        for spin, value in ((self.low_spin, low), (self.high_spin, high)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self.slider.set_range(low, high)

    def _from_slider(self, low: int, high: int) -> None:
        self.set_values(low, high)
        self.changed.emit()

    def _from_spins(self) -> None:
        low, high = self.low_spin.value(), self.high_spin.value()
        if not self.slider.wrap and low > high:
            if self.sender() is self.low_spin:
                high = low
            else:
                low = high
        self.set_values(low, high)
        self.changed.emit()


class HistoryChart(QWidget):
    """One series: objects found over the last minute, as a step line (counts are whole numbers)."""

    MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM = 30, 12, 26, 20

    def __init__(self, window_seconds: float = 60.0, parent=None):
        super().__init__(parent)
        self.window_seconds = window_seconds
        self._title = ""
        self._points: list[tuple[float, int]] = []  # (seconds ago, count), oldest first
        self._hover_x: float | None = None
        self.setMouseTracking(True)
        self.setMinimumSize(240, 130)

    def set_series(self, title: str, points: list[tuple[float, int]]) -> None:
        self._title, self._points = title, points
        self.update()

    def _plot(self) -> QRectF:
        return QRectF(
            self.MARGIN_LEFT, self.MARGIN_TOP,
            max(1, self.width() - self.MARGIN_LEFT - self.MARGIN_RIGHT),
            max(1, self.height() - self.MARGIN_TOP - self.MARGIN_BOTTOM),
        )

    def _x(self, plot: QRectF, ago: float) -> float:
        return plot.right() - plot.width() * min(ago, self.window_seconds) / self.window_seconds

    def point_at(self, x: float) -> tuple[float, int] | None:
        """The sample nearest to a horizontal position; None if there are no samples."""
        if not self._points:
            return None
        plot = self._plot()
        return min(self._points, key=lambda p: abs(self._x(plot, p[0]) - x))

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        plot = self._plot()
        muted, text = QColor(theme.MUTED), QColor(theme.TEXT)

        painter.setPen(text)
        painter.drawText(QRectF(0, 0, self.width(), self.MARGIN_TOP - 6), Qt.AlignLeft | Qt.AlignVCenter, self._title)

        top_value = max((count for _, count in self._points), default=0)
        top_value = max(1, top_value)
        painter.setPen(QPen(QColor(theme.BORDER), 1))
        for y in (plot.top(), plot.bottom()):
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
        painter.setPen(muted)
        for value, y in ((top_value, plot.top()), (0, plot.bottom())):
            painter.drawText(QRectF(0, y - 9, self.MARGIN_LEFT - 6, 18), Qt.AlignRight | Qt.AlignVCenter, str(value))
        bottom = QRectF(plot.left(), plot.bottom() + 2, plot.width(), self.MARGIN_BOTTOM - 2)
        painter.drawText(bottom, Qt.AlignLeft | Qt.AlignVCenter, f"{self.window_seconds:.0f} s temu")
        painter.drawText(bottom, Qt.AlignRight | Qt.AlignVCenter, "teraz")

        if not self._points:
            painter.drawText(plot, Qt.AlignCenter, "Brak danych. Naciśnij Start.")
            return

        y_of = lambda count: plot.bottom() - plot.height() * count / top_value
        line = QPolygonF()
        for index, (ago, count) in enumerate(self._points):
            x = self._x(plot, ago)
            if index:
                line.append(QPointF(x, y_of(self._points[index - 1][1])))
            line.append(QPointF(x, y_of(count)))
        pen = QPen(QColor(theme.ACCENT), 2)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawPolyline(line)

        if self._hover_x is not None:
            ago, count = self.point_at(self._hover_x)
            x, y = self._x(plot, ago), y_of(count)
            painter.setPen(QPen(muted, 1))
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.setPen(QPen(QColor(theme.BG), 2))
            painter.setBrush(QColor(theme.ACCENT))
            painter.drawEllipse(QPointF(x, y), 4, 4)
            painter.setPen(text)
            label = f"{ago:.0f} s temu: {count}"
            painter.drawText(QRectF(0, 0, self.width() - 4, self.MARGIN_TOP - 6), Qt.AlignRight | Qt.AlignVCenter, label)

    def mouseMoveEvent(self, event) -> None:
        self._hover_x = event.position().x()
        self.update()

    def leaveEvent(self, _event) -> None:
        self._hover_x = None
        self.update()


class FrameView(QWidget):
    """Shows the frame scaled to fit and reports positions in image coordinates."""

    imageClicked = Signal(int, int)
    imageHovered = Signal(int, int)
    hoverLeft = Signal()
    areaSelected = Signal(int, int, int, int)  # x, y, width, height in image pixels

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selecting = False
        self._drag_start: QPoint | None = None
        self._drag_now: QPoint | None = None
        self._pixmap: QPixmap | None = None
        self._image_size = QSize()
        self._title = "Brak obrazu"
        self._hint = ""
        self.setMouseTracking(True)
        self.setMinimumSize(320, 220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_placeholder(self, title: str, hint: str = "") -> None:
        self._pixmap = None
        self._image_size = QSize()
        self._title, self._hint = title, hint
        self.update()

    def set_image(self, image: QImage) -> None:
        self._pixmap = QPixmap.fromImage(image)
        self._image_size = image.size()
        self.update()

    def set_selecting(self, active: bool) -> None:
        """In selecting mode a mouse drag marks an area of the image."""
        self.selecting = active
        self._drag_start = self._drag_now = None
        self.update()

    def _clamped(self, pos: QPoint) -> QPoint:
        rect = self.target_rect()
        return QPoint(min(max(pos.x(), rect.left()), rect.right()), min(max(pos.y(), rect.top()), rect.bottom()))

    def target_rect(self) -> QRect:
        if self._image_size.isEmpty():
            return QRect()
        size = self._image_size.scaled(self.size(), Qt.KeepAspectRatio)
        return QRect(
            (self.width() - size.width()) // 2, (self.height() - size.height()) // 2,
            size.width(), size.height(),
        )

    def to_image(self, pos: QPoint) -> tuple[int, int] | None:
        """Map a widget position to image pixels; None outside the image."""
        rect = self.target_rect()
        if rect.isEmpty() or not rect.contains(pos):
            return None
        x = (pos.x() - rect.left()) * self._image_size.width() // rect.width()
        y = (pos.y() - rect.top()) * self._image_size.height() // rect.height()
        return (min(x, self._image_size.width() - 1), min(y, self._image_size.height() - 1))

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(theme.BG))
        if self._pixmap is not None:
            # Keep pixels crisp when enlarging, so single pixels stay easy to pick.
            enlarging = self.target_rect().width() >= self._image_size.width()
            painter.setRenderHint(QPainter.SmoothPixmapTransform, not enlarging)
            painter.drawPixmap(self.target_rect(), self._pixmap)
            if self._drag_start is not None and self._drag_now is not None:
                painter.setPen(QPen(QColor(theme.ACCENT_HOVER), 1, Qt.DashLine))
                painter.setBrush(QColor(47, 181, 160, 50))
                painter.drawRect(QRect(self._drag_start, self._drag_now).normalized())
            return
        painter.setPen(QColor(theme.TEXT))
        title_font = painter.font()
        title_font.setPixelSize(17)
        title_font.setWeight(title_font.Weight.DemiBold)
        painter.setFont(title_font)
        centre_y = self.height() // 2
        painter.drawText(QRect(16, centre_y - 34, self.width() - 32, 28), Qt.AlignCenter, self._title)
        hint_font = painter.font()
        hint_font.setPixelSize(13)
        hint_font.setWeight(hint_font.Weight.Normal)
        painter.setFont(hint_font)
        painter.setPen(QColor(theme.MUTED))
        painter.drawText(
            QRect(32, centre_y, self.width() - 64, 70),
            Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, self._hint,
        )

    def mouseMoveEvent(self, event) -> None:
        if self._drag_start is not None:
            self._drag_now = self._clamped(event.position().toPoint())
            self.update()
        point = self.to_image(event.position().toPoint())
        if point is None:
            self.hoverLeft.emit()
        else:
            self.imageHovered.emit(*point)

    def leaveEvent(self, _event) -> None:
        self.hoverLeft.emit()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        point = self.to_image(event.position().toPoint())
        if point is None:
            return
        if self.selecting:
            self._drag_start = self._drag_now = event.position().toPoint()
        else:
            self.imageClicked.emit(*point)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or self._drag_start is None:
            return
        first = self.to_image(self._drag_start)
        last = self.to_image(self._clamped(event.position().toPoint()))
        self._drag_start = self._drag_now = None
        self.update()
        if first is not None and last is not None:
            left, top = min(first[0], last[0]), min(first[1], last[1])
            width, height = abs(first[0] - last[0]) + 1, abs(first[1] - last[1]) + 1
            self.areaSelected.emit(left, top, width, height)
