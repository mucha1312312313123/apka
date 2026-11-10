"""Colours, stylesheet and the programmatically drawn application icon."""
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPalette, QPen, QPixmap

BG = "#14181d"
PANEL = "#1b2127"
RAISED = "#242b33"
HOVER = "#2d3640"
BORDER = "#313a44"
TEXT = "#e6eaee"
MUTED = "#9aa5b1"
ACCENT = "#2fb5a0"
ACCENT_HOVER = "#3cc9b3"
ACCENT_TEXT = "#07211d"
DANGER = "#e5604d"
WARNING = "#e8b04a"
INFO = "#5ba7e6"

STATE_COLORS = {
    "idle": MUTED,
    "running": ACCENT,
    "paused": INFO,
    "waiting": WARNING,
    "error": DANGER,
}

STYLESHEET = f"""
* {{ font-family: "Segoe UI"; font-size: 13px; color: {TEXT}; }}
QMainWindow, QDialog, QMessageBox {{ background: {BG}; }}
QWidget#panel {{ background: {PANEL}; }}
QLabel {{ background: transparent; }}
QLabel[role="heading"] {{ font-size: 11px; font-weight: 600; color: {MUTED}; letter-spacing: 1px; }}
QLabel[role="muted"] {{ color: {MUTED}; }}
QLabel#banner {{ background: #3a2f18; color: {WARNING}; padding: 7px 12px; border-bottom: 1px solid #5a4620; }}

QMenuBar {{ background: {PANEL}; border-bottom: 1px solid {BORDER}; }}
QMenuBar::item {{ background: transparent; padding: 5px 10px; }}
QMenuBar::item:selected {{ background: {HOVER}; }}
QMenu {{ background: {RAISED}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 6px 26px 6px 14px; border-radius: 4px; }}
QMenu::item:selected {{ background: {HOVER}; }}
QMenu::item:disabled {{ color: #5f6a75; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 6px; }}

QPushButton, QToolButton {{
    background: {RAISED}; border: 1px solid {BORDER}; border-radius: 5px; padding: 6px 12px;
}}
QPushButton:hover, QToolButton:hover {{ background: {HOVER}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled, QToolButton:disabled {{ color: #5f6a75; background: {PANEL}; }}
QToolButton:checked {{ background: #17423c; border-color: {ACCENT}; color: {ACCENT_HOVER}; }}
QPushButton:focus, QToolButton:focus {{ border-color: {ACCENT}; }}
QToolButton#primary {{
    background: {ACCENT}; border-color: {ACCENT}; color: {ACCENT_TEXT}; font-weight: 600; padding: 6px 18px;
}}
QToolButton#primary:hover {{ background: {ACCENT_HOVER}; }}
QToolButton#primary:checked {{ background: {DANGER}; border-color: {DANGER}; color: #ffffff; }}

QLineEdit, QSpinBox, QComboBox, QPlainTextEdit {{
    background: {BG}; border: 1px solid {BORDER}; border-radius: 5px; padding: 5px 7px;
    selection-background-color: {ACCENT}; selection-color: {ACCENT_TEXT};
}}
/* Hide the arrows by zero width: NoButtons with a stylesheet collapses the text field to 1 px. */
QSpinBox::up-button, QSpinBox::down-button {{ width: 0; border: 0; }}
QSpinBox QLineEdit, QComboBox QLineEdit {{ background: transparent; border: 0; padding: 0 6px; }}
QSpinBox#compact QLineEdit {{ padding: 0; }}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled {{ color: #5f6a75; }}
QComboBox QAbstractItemView {{
    background: {RAISED}; border: 1px solid {BORDER};
    selection-background-color: {HOVER}; outline: 0;
}}

QTreeWidget, QTableWidget, QListWidget {{
    background: {BG}; border: 1px solid {BORDER}; border-radius: 5px;
    gridline-color: {BORDER}; outline: 0; alternate-background-color: {PANEL};
}}
QTreeWidget::item, QListWidget::item {{ padding: 5px 2px; }}
QTreeWidget::item:selected, QTableWidget::item:selected, QListWidget::item:selected {{
    background: #17423c; color: {TEXT};
}}
QHeaderView::section {{
    background: {PANEL}; color: {MUTED}; border: 0; border-bottom: 1px solid {BORDER}; padding: 5px 7px;
}}
QTableCornerButton::section {{ background: {PANEL}; border: 0; }}

QTabWidget::pane {{ border: 0; border-top: 1px solid {BORDER}; background: {PANEL}; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 7px 14px; border-bottom: 2px solid transparent; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom-color: {ACCENT}; }}

QCheckBox {{ background: transparent; spacing: 6px; }}
QCheckBox::indicator {{ width: 13px; height: 13px; border: 1px solid {MUTED}; border-radius: 3px; background: {BG}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QStatusBar {{ background: {PANEL}; border-top: 1px solid {BORDER}; color: {MUTED}; }}
QStatusBar::item {{ border: 0; }}
QStatusBar QLabel {{ color: {MUTED}; padding: 0 8px; }}
QScrollArea {{ border: 0; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle {{ background: {BORDER}; border-radius: 4px; min-height: 24px; min-width: 24px; }}
QScrollBar::handle:hover {{ background: #4a5663; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QToolTip {{ background: {RAISED}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px; }}
"""


def apply_palette(app) -> None:
    """Native parts (file dialogs, checkbox marks) follow the dark theme too."""
    palette = QPalette()
    roles = {
        QPalette.Window: BG, QPalette.WindowText: TEXT, QPalette.Base: BG,
        QPalette.AlternateBase: PANEL, QPalette.Text: TEXT, QPalette.Button: RAISED,
        QPalette.ButtonText: TEXT, QPalette.Highlight: ACCENT,
        QPalette.HighlightedText: ACCENT_TEXT, QPalette.ToolTipBase: RAISED,
        QPalette.ToolTipText: TEXT, QPalette.PlaceholderText: MUTED,
    }
    for role, color in roles.items():
        palette.setColor(role, QColor(color))
    app.setPalette(palette)


def app_icon() -> QIcon:
    """A viewfinder mark, drawn in code so the app ships no binary assets."""
    icon = QIcon()
    for size in (16, 32, 64, 256):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(ACCENT))
        painter.drawRoundedRect(QRectF(0, 0, size, size), size * 0.2, size * 0.2)
        painter.setPen(QPen(QColor(ACCENT_TEXT), max(1.5, size * 0.09)))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QRectF(size * 0.27, size * 0.27, size * 0.46, size * 0.46))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(ACCENT_TEXT))
        dot = size * 0.09
        painter.drawEllipse(QRectF(size / 2 - dot, size / 2 - dot, dot * 2, dot * 2))
        painter.end()
        icon.addPixmap(pixmap)
    return icon
