"""Main application window."""
from __future__ import annotations

import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRegularExpression, QSettings, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QAction, QColor, QDesktopServices, QIcon, QImage, QKeySequence, QPixmap, QRegularExpressionValidator,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QColorDialog, QComboBox, QFileDialog, QFormLayout, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QSizePolicy,
    QPushButton, QScrollArea, QSlider, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QToolButton,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from ..calibrate import range_from_sample
from ..capture import ImageSource, WindowSource, foreground_window, list_windows, write_png
from ..detector import (
    SHAPE_ANY, SHAPE_BAR, SHAPE_CIRCLE, SHAPE_DOT, SHAPE_OTHER, SHAPE_RECTANGLE, SHAPE_TEMPLATE, ColorRule,
    TemplateRule, template_is_usable,
)
from ..plugin import load_plugins
from ..profile import Profile, ProfileError, load_profile, save_profile
from ..automation import (
    EVENT_ACTION, EVENT_TASK_DONE, EVENT_TASK_FAILED, EVENT_TASK_START, MODE_ARMED, MODE_OFF, MODE_SIMULATE, Decision, Task,
)
from ..events import EVENT_APPEARED, EVENT_DISAPPEARED, EVENT_MOVING, EVENT_STATE_OFF, EVENT_STATE_ON, EVENT_STOPPED, Event
from ..history import DetectionHistory
from ..paths import DATA_ROOT, bundled_files, prepare_user_files
from ..states import MODE_ALL, StateChange, StateRule, format_duration, missing_items, rename_item
from . import theme
from .automation_dialogs import STEP_LABELS, DecisionDialog, TaskDialog
from .state_dialog import OPERATOR_LABELS, StateDialog
from .widgets import FrameView, HistoryChart, RangeRow
from .worker import MODE_DETECTIONS, MODE_MASK, FramePacket, VisionWorker

ROOT = DATA_ROOT
APP_NAME = "OSRS Color Vision"
DEFAULT_TITLE = "Old School RuneScape"
MAX_TABLE_ROWS = 100
MAX_HISTORY_ROWS = 200
PICK_WINDOW_SECONDS = 3
WORKER_STOP_TIMEOUT_MS = 5000
MIN_CROP_SIDE = 50
MAX_EVENT_ROWS = 500
HISTORY_WINDOW_SECONDS = 60.0
EVENT_LABELS = {
    EVENT_APPEARED: "Pojawienie",
    EVENT_DISAPPEARED: "Zniknięcie",
    EVENT_MOVING: "Ruch",
    EVENT_STOPPED: "Zatrzymanie",
    EVENT_STATE_ON: "Stan: start",
    EVENT_STATE_OFF: "Stan: koniec",
    EVENT_TASK_START: "Zadanie: start",
    EVENT_TASK_DONE: "Zadanie: koniec",
    EVENT_TASK_FAILED: "Zadanie: błąd",
    EVENT_ACTION: "Akcja",
}
EVENT_GROUPS = {
    "Obiekty": {EVENT_APPEARED, EVENT_DISAPPEARED},
    "Ruch": {EVENT_MOVING, EVENT_STOPPED},
    "Stany": {EVENT_STATE_ON, EVENT_STATE_OFF},
    "Zadania": {EVENT_TASK_START, EVENT_TASK_DONE, EVENT_TASK_FAILED, EVENT_ACTION},
}
TABLE_REFRESH_SECONDS = 0.2
RULE_NAME_PATTERN = r"[A-Za-z0-9_ \-]{1,40}"
SHAPE_LABELS = {
    SHAPE_ANY: "Dowolny",
    SHAPE_BAR: "Pasek",
    SHAPE_RECTANGLE: "Prostokąt",
    SHAPE_CIRCLE: "Koło",
    SHAPE_DOT: "Kropka",
    SHAPE_OTHER: "Inny",
    SHAPE_TEMPLATE: "Wzorzec",
}


def heading(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setProperty("role", "heading")
    return label


def muted(text: str, wrap: bool = True) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "muted")
    label.setWordWrap(wrap)
    return label


def swatch_icon(bgr: tuple[int, int, int]) -> QIcon:
    pixmap = QPixmap(12, 12)
    pixmap.fill(QColor(bgr[2], bgr[1], bgr[0]))
    return QIcon(pixmap)


def picture_pixmap(image: np.ndarray, max_side: int) -> QPixmap:
    """BGR picture scaled to fit, with hard pixel edges so small icons stay readable."""
    height, width = image.shape[:2]
    qimage = QImage(image.data, width, height, 3 * width, QImage.Format_BGR888).copy()
    return QPixmap.fromImage(qimage).scaled(max_side, max_side, Qt.KeepAspectRatio, Qt.FastTransformation)


class WindowCombo(QComboBox):
    """Editable window picker; the list is refreshed each time it is opened."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.NoInsert)
        self.setMinimumWidth(230)
        self.setToolTip("Tytuł okna klienta. Wpisz go albo wybierz z listy otwartych okien.")

    def showPopup(self) -> None:
        current = self.currentText()
        self.blockSignals(True)
        self.clear()
        try:
            windows = [w for w in list_windows() if not w.title.startswith(APP_NAME)]
        except RuntimeError:
            windows = []
        for window in sorted(windows, key=lambda w: w.title.casefold()):
            self.addItem(window.title, window.hwnd)
        self.setEditText(current)
        self.blockSignals(False)
        super().showPopup()


class MainWindow(QMainWindow):
    def __init__(self, settings_path: Path | None = None, plugins_dir: Path | None = None, profile_path: Path | None = None):
        super().__init__()
        self.settings = QSettings(str(settings_path or ROOT / "settings.ini"), QSettings.IniFormat)
        self.plugins_dir = plugins_dir or ROOT / "plugins"
        self.profile = Profile()
        self.profile_path: Path | None = None
        self.dirty = False
        self.worker: VisionWorker | None = None
        self.last_packet: FramePacket | None = None
        self.image_path: str | None = None
        self._loading = False
        self._last_table_refresh = 0.0
        self._state = "idle"
        self._recut_target: str | None = None
        self._auto_paused = False
        self.history = DetectionHistory()
        self._history_rule: str | None = None
        self._abandoned_workers: list[VisionWorker] = []
        self.window_hwnd: int | None = None  # chosen exactly; otherwise found by title
        self.crop: tuple[int, int, int, int] | None = None
        self._selecting_crop = False
        self._source_info: dict | None = None
        self._pick_seconds_left = 0
        self._pick_timer = QTimer(self)
        self._pick_timer.setInterval(1000)
        self._pick_timer.timeout.connect(self._on_pick_tick)

        self.setWindowIcon(theme.app_icon())
        self.resize(1320, 840)
        self.setMinimumSize(1240, 640)

        self._build_actions()
        self._build_menu()
        self._build_ui()
        self._build_statusbar()

        self.window_combo.setEditText(self.settings.value("window_title", DEFAULT_TITLE))
        self.crop = self._stored_crop()
        self.window_combo.editTextChanged.connect(self._on_window_text_changed)
        self.window_combo.activated.connect(self._on_window_chosen)
        self.fps_spin.setValue(int(self.settings.value("fps", 20)))
        geometry = self.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

        self._history_timer = QTimer(self)
        self._history_timer.setInterval(1000)
        self._history_timer.timeout.connect(self._refresh_history)
        self._history_timer.start()

        self._refresh_plugins()
        self._open_initial_profile(profile_path)
        self._set_state("idle")
        self._sync_actions()

    # ---------------------------------------------------------------- construction

    def _action(self, text: str, handler, shortcut=None, checkable=False, tip: str = "") -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        action.setCheckable(checkable)
        if tip:
            action.setToolTip(f"{tip} ({QKeySequence(shortcut).toString()})" if shortcut is not None else tip)
        (action.toggled if checkable else action.triggered).connect(handler)
        return action

    def _build_actions(self) -> None:
        self.new_action = self._action("Nowy profil", self._new_profile, QKeySequence.New)
        self.open_action = self._action("Otwórz profil…", self._open_profile_dialog, QKeySequence.Open)
        self.save_action = self._action("Zapisz", self._save, QKeySequence.Save)
        self.save_as_action = self._action("Zapisz jako…", self._save_as, QKeySequence.SaveAs)
        self.image_action = self._action("Analizuj obraz z pliku…", self._open_image)
        self.quit_action = self._action("Zakończ", self.close, "Alt+F4")

        self.start_action = self._action("▶  Start", self._toggle_run, "F5", True, "Uruchom lub zatrzymaj wykrywanie")
        self.pause_action = self._action("Pauza", self._toggle_pause, "F6", True, "Zatrzymaj obraz na bieżącej klatce")
        self.pipette_action = self._action("Pipeta", self._toggle_pipette, "F7", True, "Kliknij kolor na obrazie, aby go pobrać")
        self.mask_action = self._action("Maska", self._push_view, "F8", True, "Pokaż tylko piksele pasujące do wybranej reguły")
        self.snapshot_action = self._action("Zapisz klatkę", self._save_snapshot, "F9", tip="Zapisz czystą klatkę do pliku PNG")
        self.template_action = self._action("Wytnij wzorzec", self._toggle_template_cut, "F10", True, "Zaznacz myszą obiekt na obrazie, aby szukać podobnych")
        self.pick_window_action = self._action("Wskaż okno", self._start_window_pick, "F2", tip="Kliknij okno gry w ciągu 3 sekund, a program je zapamięta")
        self.crop_action = self._action("Obszar", self._toggle_crop_select, "F3", True, "Zaznacz myszą część obrazu do przechwytywania, np. samą grę")
        self.debug_action = self._action("Debug", self._push_view, "F11", True, "Pokaż numery obiektów, pewność, ślady ruchu, regiony i czasy na podglądzie")

        self.help_action = self._action("Jak zacząć", self._show_help, "F1")
        self.plugins_folder_action = self._action("Otwórz folder pluginów", self._open_plugins_folder)
        self.about_action = self._action(f"O programie", self._show_about)

    def _build_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&Plik")
        file_menu.addActions([self.new_action, self.open_action, self.save_action, self.save_as_action])
        file_menu.addSeparator()
        file_menu.addActions([self.image_action, self.snapshot_action])
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)
        run_menu = bar.addMenu("&Wykrywanie")
        run_menu.addActions([self.pick_window_action, self.crop_action])
        run_menu.addSeparator()
        run_menu.addActions([self.start_action, self.pause_action])
        run_menu.addSeparator()
        run_menu.addActions([self.pipette_action, self.mask_action, self.template_action, self.debug_action])
        help_menu = bar.addMenu("Pomo&c")
        help_menu.addActions([self.help_action, self.plugins_folder_action])
        help_menu.addSeparator()
        help_menu.addAction(self.about_action)

    def _tool_button(self, action: QAction, object_name: str = "") -> QToolButton:
        button = QToolButton()
        button.setDefaultAction(action)
        button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        if object_name:
            button.setObjectName(object_name)
        return button

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_toolbar())
        self.banner = QLabel()
        self.banner.setObjectName("banner")
        self.banner.setWordWrap(True)
        self.banner.hide()
        root.addWidget(self.banner)

        self.view = FrameView()
        self.view.imageClicked.connect(self._on_image_clicked)
        self.view.imageHovered.connect(self._on_image_hovered)
        self.view.areaSelected.connect(self._on_area_selected)
        self.view.hoverLeft.connect(lambda: self.pixel_label.setText(""))
        self._show_start_placeholder()

        center = QSplitter(Qt.Vertical)
        center.addWidget(self.view)
        center.addWidget(self._build_bottom_tabs())
        center.setStretchFactor(0, 1)
        center.setStretchFactor(1, 0)
        center.setSizes([600, 210])
        center.setChildrenCollapsible(False)

        main = QSplitter(Qt.Horizontal)
        main.addWidget(self._build_rules_panel())
        main.addWidget(center)
        main.addWidget(self._build_editor_panel())
        main.setStretchFactor(0, 0)
        main.setStretchFactor(1, 1)
        main.setStretchFactor(2, 0)
        main.setSizes([290, 690, 340])
        main.setChildrenCollapsible(False)
        root.addWidget(main, 1)
        self.setCentralWidget(central)

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("panel")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        layout.addWidget(muted("Okno klienta", wrap=False))
        self.window_combo = WindowCombo()
        layout.addWidget(self.window_combo, 1)
        layout.addWidget(self._tool_button(self.pick_window_action))
        layout.addWidget(self._tool_button(self.crop_action))
        self.start_button = self._tool_button(self.start_action, "primary")
        layout.addWidget(self.start_button)
        layout.addSpacing(12)
        layout.addWidget(self._tool_button(self.pause_action))
        layout.addWidget(self._tool_button(self.pipette_action))
        layout.addWidget(self._tool_button(self.mask_action))
        layout.addWidget(self._tool_button(self.template_action))
        layout.addWidget(self._tool_button(self.debug_action))
        layout.addWidget(self._tool_button(self.snapshot_action))
        layout.addSpacing(12)
        layout.addWidget(muted("FPS", wrap=False))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setToolTip("Docelowa liczba analizowanych klatek na sekundę")
        self.fps_spin.setFixedWidth(68)
        self.fps_spin.valueChanged.connect(self._on_fps_changed)
        layout.addWidget(self.fps_spin)
        return bar

    def _build_rules_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("panel")
        panel.setMinimumWidth(270)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(heading("Reguły i wzorce"))
        self.rules_tree = QTreeWidget()
        self.rules_tree.setColumnCount(2)
        self.rules_tree.setHeaderLabels(["Nazwa", "Ilość"])
        self.rules_tree.setRootIsDecorated(False)
        self.rules_tree.setUniformRowHeights(True)
        self.rules_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.rules_tree.header().setSectionResizeMode(1, QHeaderView.Fixed)
        self.rules_tree.header().setStretchLastSection(False)
        self.rules_tree.header().resizeSection(1, 48)
        self.rules_tree.setIconSize(QPixmap(16, 16).size())
        self.rules_tree.setToolTip("Odznacz pozycję, aby ją wyłączyć bez usuwania.")
        self.rules_tree.itemSelectionChanged.connect(self._on_rule_selected)
        self.rules_tree.itemChanged.connect(self._on_rule_checked)
        layout.addWidget(self.rules_tree, 1)
        add_row = QHBoxLayout()
        add_row.setSpacing(6)
        self.add_rule_button = QPushButton("Dodaj kolor")
        self.add_rule_button.clicked.connect(self._add_rule)
        add_row.addWidget(self.add_rule_button, 1)
        cut_button = self._tool_button(self.template_action)
        cut_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        add_row.addWidget(cut_button, 1)
        layout.addLayout(add_row)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.duplicate_rule_button = QPushButton("Duplikuj")
        self.duplicate_rule_button.clicked.connect(self._duplicate_rule)
        self.delete_rule_button = QPushButton("Usuń")
        self.delete_rule_button.clicked.connect(self._delete_rule)
        row.addWidget(self.duplicate_rule_button, 1)
        row.addWidget(self.delete_rule_button, 1)
        layout.addLayout(row)
        return panel

    def _build_editor_panel(self) -> QWidget:
        content = QWidget()
        content.setObjectName("panel")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)
        self.editor_heading = heading("Edytor reguły")
        layout.addWidget(self.editor_heading)

        self.editor_empty = muted("Wybierz pozycję z listy albo dodaj nową regułę koloru lub wzorzec.")
        layout.addWidget(self.editor_empty)

        self.editor = QWidget()
        form = QVBoxLayout(self.editor)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(10)

        fields = QFormLayout()
        fields.setLabelAlignment(Qt.AlignLeft)
        fields.setVerticalSpacing(8)
        self.name_edit = QLineEdit()
        self.name_edit.setValidator(QRegularExpressionValidator(QRegularExpression(RULE_NAME_PATTERN)))
        self.name_edit.editingFinished.connect(self._on_name_edited)
        self.region_combo = QComboBox()
        self.region_combo.setToolTip("Część obrazu, w której reguła szuka koloru")
        self.region_combo.currentIndexChanged.connect(self._apply_editor)
        self.color_button = QPushButton()
        self.color_button.setToolTip("Kolor ramki na podglądzie")
        self.color_button.clicked.connect(self._choose_draw_color)
        fields.addRow("Nazwa", self.name_edit)
        fields.addRow("Region", self.region_combo)
        fields.addRow("Kolor ramki", self.color_button)
        form.addLayout(fields)
        form.addWidget(self._build_color_section())
        form.addWidget(self._build_template_section())
        layout.addWidget(self.editor)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        scroll.setMinimumWidth(310)
        return scroll

    def _build_color_section(self) -> QWidget:
        self.color_section = QWidget()
        form = QVBoxLayout(self.color_section)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(10)
        form.addWidget(heading("Zakres koloru"))
        self.h_row = RangeRow("H", 179, wrap=True, tooltip="Odcień. Min większe od maks oznacza zakres przez czerwień.")
        self.s_row = RangeRow("S", 255, tooltip="Nasycenie: od szarości do czystego koloru")
        self.v_row = RangeRow("V", 255, tooltip="Jasność: od czerni do pełnej jasności")
        self.h_row.slider.set_stops([QColor.fromHsv(angle % 360, 255, 255) for angle in range(0, 361, 30)])
        for row in (self.h_row, self.s_row, self.v_row):
            row.changed.connect(self._apply_editor)
            form.addWidget(row)
        form.addWidget(self._tool_button(self.pipette_action))
        form.addWidget(muted("Włącz pipetę i kliknij kolor na obrazie. Pauza ułatwia trafienie w ruchomy obiekt."))

        form.addWidget(heading("Filtrowanie obiektów"))
        filters = QFormLayout()
        filters.setVerticalSpacing(8)
        self.shape_combo = QComboBox()
        for shape in (SHAPE_ANY, SHAPE_BAR, SHAPE_RECTANGLE, SHAPE_CIRCLE, SHAPE_DOT, SHAPE_OTHER):
            self.shape_combo.addItem(SHAPE_LABELS[shape], shape)
        self.shape_combo.setToolTip("Zostawia tylko obiekty o tym kształcie. Rozpoznany kształt widać w zakładce Wykrycia.")
        self.shape_combo.currentIndexChanged.connect(self._apply_editor)
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(1, 10_000_000)
        self.min_area_spin.setSuffix(" px")
        self.min_area_spin.setToolTip("Mniejsze obiekty są ignorowane")
        self.max_area_spin = QSpinBox()
        self.max_area_spin.setRange(0, 10_000_000)
        self.max_area_spin.setSuffix(" px")
        self.max_area_spin.setSpecialValueText("bez limitu")
        self.max_area_spin.setToolTip("Większe obiekty są ignorowane")
        self.merge_spin = QSpinBox()
        self.merge_spin.setRange(0, 25)
        self.merge_spin.setSuffix(" px")
        self.merge_spin.setSpecialValueText("wyłączone")
        self.merge_spin.setToolTip("Łączy fragmenty jednego obiektu rozdzielone małą przerwą")
        self.min_confidence_spin = QSpinBox()
        self.min_confidence_spin.setRange(0, 100)
        self.min_confidence_spin.setSuffix(" %")
        self.min_confidence_spin.setSpecialValueText("bez limitu")
        self.min_confidence_spin.setToolTip(
            "Odrzuca obiekty słabo pasujące do koloru. Pewność mówi, jak dobrze obiekt pasuje do reguły, a nie jak prawdopodobne jest, że to właściwa rzecz."
        )
        for spin in (self.min_area_spin, self.max_area_spin, self.merge_spin, self.min_confidence_spin):
            spin.valueChanged.connect(self._apply_editor)
        filters.addRow("Min. pewność", self.min_confidence_spin)
        filters.addRow("Kształt", self.shape_combo)
        filters.addRow("Min. pole", self.min_area_spin)
        filters.addRow("Maks. pole", self.max_area_spin)
        filters.addRow("Łączenie", self.merge_spin)
        form.addLayout(filters)
        return self.color_section

    def _build_template_section(self) -> QWidget:
        self.template_section = QWidget()
        form = QVBoxLayout(self.template_section)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(10)
        form.addWidget(heading("Wzorzec"))
        self.template_preview = QLabel()
        self.template_preview.setAlignment(Qt.AlignCenter)
        self.template_preview.setMinimumHeight(110)
        form.addWidget(self.template_preview)
        self.template_size_label = muted("", wrap=False)
        self.template_size_label.setAlignment(Qt.AlignCenter)
        form.addWidget(self.template_size_label)
        recut = QPushButton("Wytnij ponownie")
        recut.setToolTip("Zastąp ten obraz nowym fragmentem podglądu")
        recut.clicked.connect(self._recut_template)
        form.addWidget(recut)

        form.addWidget(heading("Dopasowanie"))
        fields = QFormLayout()
        fields.setVerticalSpacing(8)
        threshold_row = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(50, 100)
        self.threshold_slider.setToolTip("Im wyżej, tym wierniejsze musi być podobieństwo do wzorca")
        self.threshold_label = QLabel()
        self.threshold_label.setFixedWidth(40)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        threshold_row.addWidget(self.threshold_slider, 1)
        threshold_row.addWidget(self.threshold_label)
        self.max_results_spin = QSpinBox()
        self.max_results_spin.setRange(1, 200)
        self.max_results_spin.setToolTip("Największa liczba zgłaszanych trafień")
        self.max_results_spin.valueChanged.connect(self._apply_editor)
        fields.addRow("Podobieństwo", threshold_row)
        fields.addRow("Maks. trafień", self.max_results_spin)
        form.addLayout(fields)
        form.addWidget(muted(
            "Wzorce najlepiej działają na elementach o stałym wyglądzie, np. przedmiotach i ikonach. "
            "Modele 3D zmieniają wygląd przy obracaniu, więc mogą nie zostać znalezione."
        ))
        return self.template_section

    def _build_bottom_tabs(self) -> QWidget:
        self.tabs = QTabWidget()
        self.tabs.setMinimumHeight(150)
        self.detections_table = QTableWidget(0, 9)
        self.detections_table.setHorizontalHeaderLabels(["Reguła", "Kształt", "X", "Y", "Szer.", "Wys.", "Pole", "Środek", "Pewność"])
        self.detections_table.verticalHeader().hide()
        self.detections_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.detections_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.detections_table.setShowGrid(False)
        self.detections_table.horizontalHeader().setDefaultSectionSize(68)
        self.detections_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tabs.addTab(self.detections_table, "Wykrycia")
        self.objects_tab = self._build_objects_tab()
        self.tabs.addTab(self.objects_tab, "Obiekty")
        self.events_tab = self._build_events_tab()
        self.tabs.addTab(self.events_tab, "Zdarzenia")
        self.history_tab = self._build_history_tab()
        self.tabs.addTab(self.history_tab, "Historia")
        self.states_tab = self._build_states_tab()
        self.tabs.addTab(self.states_tab, "Stany")
        self.automation_tab = self._build_automation_tab()
        self.tabs.addTab(self.automation_tab, "Automat")
        self.tabs.addTab(self._build_regions_tab(), "Regiony")
        self.tabs.addTab(self._build_plugins_tab(), "Pluginy")
        self.tabs.addTab(self._build_log_tab(), "Dziennik")
        return self.tabs

    @staticmethod
    def _plain_table(headers: list[str], stretch_column: int = 0) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().hide()
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.setShowGrid(False)
        table.horizontalHeader().setDefaultSectionSize(72)
        table.horizontalHeader().setSectionResizeMode(stretch_column, QHeaderView.Stretch)
        return table

    def _build_objects_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 8, 10, 8)
        top = QHBoxLayout()
        top.addWidget(muted("Maks. przesunięcie", wrap=False))
        self.track_distance_spin = QSpinBox()
        self.track_distance_spin.setRange(1, 2000)
        self.track_distance_spin.setSuffix(" px")
        self.track_distance_spin.setFixedWidth(80)
        self.track_distance_spin.setToolTip("O ile pikseli obiekt może się przesunąć między klatkami, aby nadal był tym samym obiektem")
        top.addWidget(self.track_distance_spin)
        top.addWidget(muted("Uznaj za zniknięty po", wrap=False))
        self.track_lost_spin = QSpinBox()
        self.track_lost_spin.setRange(0, 600_000)
        self.track_lost_spin.setSingleStep(100)
        self.track_lost_spin.setSuffix(" ms")
        self.track_lost_spin.setFixedWidth(90)
        self.track_lost_spin.setToolTip("Tak długo niewidoczny obiekt zachowuje swój numer, np. gdy na chwilę coś go zasłoni")
        top.addWidget(self.track_lost_spin)
        top.addStretch(1)
        for spin in (self.track_distance_spin, self.track_lost_spin):
            spin.valueChanged.connect(self._on_tracking_settings_changed)
        layout.addLayout(top)
        self.objects_table = self._plain_table(["ID", "Reguła", "X", "Y", "Prędkość", "Czas", "Pewność", "Status"], stretch_column=1)
        self.objects_table.horizontalHeader().resizeSection(0, 50)
        layout.addWidget(self.objects_table)
        return tab

    def _build_events_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 8, 10, 8)
        top = QHBoxLayout()
        top.addWidget(muted("Pokaż:", wrap=False))
        self.event_filters: dict[str, QCheckBox] = {}
        for group in EVENT_GROUPS:
            box = QCheckBox(group)
            box.setChecked(True)
            box.toggled.connect(self._apply_event_filter)
            self.event_filters[group] = box
            top.addWidget(box)
        top.addStretch(1)
        export = QPushButton("Eksportuj CSV")
        export.clicked.connect(self._export_events)
        clear = QPushButton("Wyczyść")
        clear.clicked.connect(self._clear_events)
        top.addWidget(export)
        top.addWidget(clear)
        layout.addLayout(top)
        self.events_table = self._plain_table(["Czas", "Typ", "Źródło", "Opis"], stretch_column=3)
        self.events_table.horizontalHeader().resizeSection(1, 100)
        self.events_table.horizontalHeader().resizeSection(2, 120)
        layout.addWidget(self.events_table)
        return tab

    def _build_history_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(10, 8, 10, 8)
        left = QVBoxLayout()
        top = QHBoxLayout()
        top.addWidget(muted("Ostatnia minuta", wrap=False))
        top.addStretch(1)
        export = QPushButton("Eksportuj CSV")
        export.clicked.connect(self._export_samples)
        clear = QPushButton("Wyczyść")
        clear.clicked.connect(self._clear_history)
        top.addWidget(export)
        top.addWidget(clear)
        left.addLayout(top)
        self.history_table = self._plain_table(["Reguła", "Teraz", "Średnio", "Maks.", "Pewność", "Łącznie"])
        self.history_table.setToolTip("Kliknij regułę, aby zobaczyć jej wykres. Łącznie to liczba różnych obiektów od wyczyszczenia historii.")
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.history_table.setSelectionMode(QAbstractItemView.SingleSelection)
        for column, width in ((1, 50), (2, 62), (3, 50), (4, 68), (5, 62)):
            self.history_table.horizontalHeader().resizeSection(column, width)
        self.history_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.history_table.itemSelectionChanged.connect(self._on_history_row_selected)
        left.addWidget(self.history_table)
        layout.addLayout(left, 3)
        self.history_chart = HistoryChart(HISTORY_WINDOW_SECONDS)
        layout.addWidget(self.history_chart, 2)
        return tab

    def _build_automation_tab(self) -> QWidget:
        tab = QWidget()
        root = QVBoxLayout(tab)
        root.setContentsMargins(10, 8, 10, 8)
        top = QHBoxLayout()
        top.addWidget(muted("Tryb", wrap=False))
        self.automation_combo = QComboBox()
        self.automation_combo.addItem("Wyłączona", MODE_OFF)
        self.automation_combo.addItem("Symulacja (bez klikania)", MODE_SIMULATE)
        self.automation_combo.addItem("Prawdziwe sterowanie", MODE_ARMED)
        self.automation_combo.activated.connect(self._on_automation_mode_chosen)
        top.addWidget(self.automation_combo)
        top.addWidget(muted("F12 zatrzymuje natychmiast, także gdy aktywne jest okno gry.", wrap=False))
        top.addStretch(1)
        root.addLayout(top)

        columns = QHBoxLayout()
        self.decisions_table, decision_buttons = self._automation_list(
            ["Decyzja", "Prior.", "Zadanie"], self._add_decision, self._edit_decision, self._delete_decision,
        )
        self.decisions_table.itemChanged.connect(self._on_decision_checked)
        self.tasks_table, task_buttons = self._automation_list(
            ["Zadanie", "Kroki"], self._add_task, self._edit_task, self._delete_task,
        )
        self.run_task_button = QPushButton("Uruchom")
        self.run_task_button.setToolTip("Uruchom zaznaczone zadanie teraz, w wybranym trybie")
        self.run_task_button.clicked.connect(self._run_selected_task)
        task_buttons.insertWidget(0, self.run_task_button)
        for table, buttons in ((self.decisions_table, decision_buttons), (self.tasks_table, task_buttons)):
            column = QVBoxLayout()
            column.addWidget(table)
            column.addLayout(buttons)
            columns.addLayout(column, 1)
        root.addLayout(columns)
        return tab

    def _automation_list(self, headers, on_add, on_edit, on_delete):
        table = self._plain_table(headers, stretch_column=len(headers) - 1)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.horizontalHeader().resizeSection(0, 130)
        table.itemDoubleClicked.connect(lambda _item: on_edit())
        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        for text, handler in (("Dodaj", on_add), ("Edytuj", on_edit), ("Usuń", on_delete)):
            button = QPushButton(text)
            button.clicked.connect(handler)
            buttons.addWidget(button)
        buttons.addStretch(1)
        return table, buttons

    def _build_states_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(10, 8, 10, 8)
        left = QVBoxLayout()
        self.states_table = QTableWidget(0, 4)
        self.states_table.setHorizontalHeaderLabels(["Stan", "Teraz", "Od", "Warunki"])
        self.states_table.verticalHeader().hide()
        self.states_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.states_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.states_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.states_table.setShowGrid(False)
        header = self.states_table.horizontalHeader()
        header.setDefaultSectionSize(70)
        header.resizeSection(0, 140)
        header.resizeSection(1, 92)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        self.states_table.itemChanged.connect(self._on_state_checked)
        self.states_table.itemDoubleClicked.connect(lambda _item: self._edit_state())
        self.states_table.itemSelectionChanged.connect(self._sync_state_buttons)
        left.addWidget(self.states_table)
        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        add = QPushButton("Dodaj")
        add.clicked.connect(self._add_state)
        self.edit_state_button = QPushButton("Edytuj")
        self.edit_state_button.clicked.connect(self._edit_state)
        self.duplicate_state_button = QPushButton("Duplikuj")
        self.duplicate_state_button.clicked.connect(self._duplicate_state)
        self.delete_state_button = QPushButton("Usuń")
        self.delete_state_button.clicked.connect(self._delete_state)
        for button in (add, self.edit_state_button, self.duplicate_state_button, self.delete_state_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        left.addLayout(buttons)
        layout.addLayout(left, 3)

        right = QVBoxLayout()
        right.addWidget(heading("Historia zmian"))
        self.states_history = QListWidget()
        self.states_history.setSelectionMode(QAbstractItemView.NoSelection)
        right.addWidget(self.states_history)
        layout.addLayout(right, 1)
        return tab

    def _build_regions_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 8, 10, 8)
        top = QHBoxLayout()
        top.addWidget(muted("Układ odniesienia", wrap=False))
        self.ref_w_spin, self.ref_h_spin = QSpinBox(), QSpinBox()
        for spin in (self.ref_w_spin, self.ref_h_spin):
            spin.setRange(1, 10000)
            spin.setFixedWidth(76)
            spin.setToolTip("Rozmiar okna, dla którego podano współrzędne. Przy innym rozmiarze regiony są skalowane.")
            spin.valueChanged.connect(self._on_reference_changed)
            top.addWidget(spin)
        top.addStretch(1)
        self.add_region_button = QPushButton("Dodaj region")
        self.add_region_button.clicked.connect(self._add_region)
        self.delete_region_button = QPushButton("Usuń region")
        self.delete_region_button.clicked.connect(self._delete_region)
        top.addWidget(self.add_region_button)
        top.addWidget(self.delete_region_button)
        layout.addLayout(top)
        self.regions_table = QTableWidget(0, 5)
        self.regions_table.setHorizontalHeaderLabels(["Nazwa", "X", "Y", "Szerokość", "Wysokość"])
        self.regions_table.verticalHeader().hide()
        self.regions_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.regions_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.regions_table.setShowGrid(False)
        self.regions_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.regions_table.itemChanged.connect(self._on_region_edited)
        layout.addWidget(self.regions_table)
        return tab

    def _build_plugins_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 8, 10, 8)
        top = QHBoxLayout()
        top.addWidget(muted("Zaznaczone pluginy działają po naciśnięciu Start."), 1)
        self.refresh_plugins_button = QPushButton("Odśwież")
        self.refresh_plugins_button.clicked.connect(self._refresh_plugins)
        folder = QPushButton("Otwórz folder")
        folder.clicked.connect(self._open_plugins_folder)
        top.addWidget(self.refresh_plugins_button)
        top.addWidget(folder)
        layout.addLayout(top)
        self.plugins_list = QListWidget()
        self.plugins_list.itemChanged.connect(self._on_plugin_checked)
        layout.addWidget(self.plugins_list)
        return tab

    def _build_log_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 8, 10, 8)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        layout.addWidget(self.log_view)
        return tab

    def _build_statusbar(self) -> None:
        bar = self.statusBar()
        bar.setSizeGripEnabled(True)
        self.state_label = QLabel()
        self.source_label = QLabel()
        self.fps_label = QLabel()
        self.size_label = QLabel()
        self.pixel_label = QLabel()
        self.active_states_label = QLabel()
        self.active_states_label.setToolTip("Stany aktywne w tej chwili")
        bar.addWidget(self.state_label)
        bar.addWidget(self.source_label)
        bar.addWidget(self.active_states_label)
        self.task_label = QLabel()
        bar.addWidget(self.task_label, 1)
        bar.addPermanentWidget(self.pixel_label)
        bar.addPermanentWidget(self.size_label)
        bar.addPermanentWidget(self.fps_label)

    # ---------------------------------------------------------------- small helpers

    def _warn(self, title: str, text: str) -> None:
        QMessageBox.warning(self, title, text)

    def _confirm(self, title: str, text: str) -> bool:
        return QMessageBox.question(self, title, text) == QMessageBox.Yes

    def _show_start_placeholder(self) -> None:
        self.view.set_placeholder(
            "Wybierz okno klienta i naciśnij Start",
            "Okno gry nie może być zminimalizowane ani przykryte, bo program odczytuje jego obraz z ekranu. "
            "Ustaw ten program obok klienta.",
        )

    def append_log(self, text: str) -> None:
        for line in text.rstrip("\n").splitlines():
            self.log_view.appendPlainText(f"{time.strftime('%H:%M:%S')}  {line}")

    def _set_state(self, state: str, message: str = "") -> None:
        self._state = state
        names = {"idle": "Zatrzymano", "running": "Działa", "paused": "Pauza", "waiting": "Czekam na okno", "error": "Błąd"}
        self.state_label.setText(f'<span style="color:{theme.STATE_COLORS[state]}">●</span> {names[state]}')
        self._refresh_banner(message if state in ("waiting", "error") else "")
        if state == "error" and message:
            self.append_log(message)

    def _refresh_banner(self, message: str = "") -> None:
        if not message and self.last_packet is not None and self.last_packet.covered and self._state == "running":
            message = "Inne okno przykrywa klienta, więc obraz może zawierać jego fragment. Odsłoń okno gry."
        if self._automation_mode() == MODE_ARMED and self.worker is not None:
            warning = "PRAWDZIWE STEROWANIE WŁĄCZONE. Program może klikać w oknie klienta. F12 zatrzymuje."
            message = f"{warning} {message}" if message else warning
        if not message and self.worker is not None and self._source_info and self._source_info["whole_screen"]:
            message = (
                f"Przechwytywane okno „{self._source_info['title']}” zajmuje prawie cały ekran. "
                "Jeśli to nie gra, użyj Wskaż okno. Jeśli gra jest zmaksymalizowana, użyj Obszar, aby wybrać jej fragment."
            )
        self.banner.setText(message)
        self.banner.setVisible(bool(message))

    def _sync_actions(self) -> None:
        running = self.worker is not None
        has_frame = self.last_packet is not None
        self.start_action.blockSignals(True)
        self.start_action.setChecked(running)
        self.start_action.blockSignals(False)
        self.start_action.setText("■  Stop" if running else "▶  Start")
        self.pause_action.setEnabled(running)
        if not running:
            self.pause_action.blockSignals(True)
            self.pause_action.setChecked(False)
            self.pause_action.blockSignals(False)
        self.snapshot_action.setEnabled(has_frame)
        self.pipette_action.setEnabled(has_frame)
        self.template_action.setEnabled(has_frame)
        self.crop_action.setEnabled(self.crop is not None or (has_frame and running))
        self.crop_action.blockSignals(True)
        self.crop_action.setChecked(self.crop is not None or self._selecting_crop)
        self.crop_action.blockSignals(False)
        self.crop_action.setText("Cały obraz" if self.crop is not None else "Obszar")
        if not has_frame and self.pipette_action.isChecked():
            self.pipette_action.setChecked(False)
        self.mask_action.setEnabled(self._current_rule() is not None)
        self.window_combo.setEnabled(not running)
        self.plugins_list.setEnabled(not running)
        self.refresh_plugins_button.setEnabled(not running)
        has_item = self._current_item() is not None
        self.duplicate_rule_button.setEnabled(has_item)
        self.delete_rule_button.setEnabled(has_item)
        if self.image_path:
            source = f"Plik: {Path(self.image_path).name}"
        elif running and self._source_info:
            info = self._source_info
            source = f"Okno: {info['title']} ({info['width']}×{info['height']})"
        else:
            source = f"Okno: {self.window_combo.currentText()}"
        self.source_label.setText(source)
        if not running:
            self.fps_label.setText("")

    def _update_title(self) -> None:
        name = self.profile_path.name if self.profile_path else "Nowy profil"
        self.setWindowTitle(f"{APP_NAME} – {name}{' •' if self.dirty else ''}")

    def _mark_dirty(self) -> None:
        self.dirty = True
        self._update_title()
        if self.worker is not None:
            self.worker.set_profile(self.profile)

    # ---------------------------------------------------------------- profiles

    def _open_initial_profile(self, explicit: Path | None) -> None:
        stored = self.settings.value("profile")
        candidates = [explicit, Path(stored) if stored else None, ROOT / "profiles" / "default.json"]
        for path in candidates:
            if path is not None and Path(path).exists():
                try:
                    self._set_profile(load_profile(path), Path(path))
                    return
                except ProfileError as exc:
                    self._warn("Nie można wczytać profilu", str(exc))
        self._set_profile(Profile(), None)

    def _set_profile(self, profile: Profile, path: Path | None) -> None:
        self.profile, self.profile_path, self.dirty = profile, path, False
        if path is not None:
            self.settings.setValue("profile", str(path))
        self._rebuild_regions()
        self._rebuild_region_combo()
        self._loading = True
        self.track_distance_spin.setValue(profile.track_max_distance)
        self.track_lost_spin.setValue(profile.track_lost_ms)
        self._loading = False
        self._rebuild_rules(profile.items[0].name if profile.items else None)
        self._rebuild_states()
        self._rebuild_automation()
        self._update_title()
        if self.worker is not None:
            self.worker.set_profile(self.profile)

    def _maybe_save(self) -> bool:
        """Return True if it is safe to discard the current profile."""
        if not self.dirty:
            return True
        answer = QMessageBox.question(
            self, "Niezapisane zmiany", "Profil ma niezapisane zmiany. Czy chcesz je zapisać?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
        )
        if answer == QMessageBox.Save:
            return self._save()
        return answer == QMessageBox.Discard

    def _new_profile(self) -> None:
        if self._maybe_save():
            layout = load_profile(ROOT / "profiles" / "default.json") if (ROOT / "profiles" / "default.json").exists() else Profile()
            self._set_profile(Profile(layout.reference_size, layout.regions, []), None)

    def _open_profile_dialog(self) -> None:
        if not self._maybe_save():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Otwórz profil", str(ROOT / "profiles"), "Profile (*.json)")
        if path:
            try:
                self._set_profile(load_profile(path), Path(path))
            except ProfileError as exc:
                self._warn("Nie można wczytać profilu", str(exc))

    def _save(self) -> bool:
        if self.profile_path is None:
            return self._save_as()
        return self._write_profile(self.profile_path)

    def _save_as(self) -> bool:
        start = str(self.profile_path or ROOT / "profiles" / "profil.json")
        path, _ = QFileDialog.getSaveFileName(self, "Zapisz profil", start, "Profile (*.json)")
        return self._write_profile(Path(path)) if path else False

    def _write_profile(self, path: Path) -> bool:
        try:
            save_profile(self.profile, path)
        except OSError as exc:
            self._warn("Nie można zapisać profilu", str(exc))
            return False
        self.profile_path, self.dirty = path, False
        self.settings.setValue("profile", str(path))
        self._update_title()
        self.statusBar().showMessage(f"Zapisano {path.name}", 3000)
        return True

    # ---------------------------------------------------------------- rules

    def _current_item(self) -> ColorRule | TemplateRule | None:
        """The selected list entry: a colour rule or a template."""
        items = self.rules_tree.selectedItems()
        if not items:
            return None
        name = items[0].data(0, Qt.UserRole)
        return next((item for item in self.profile.items if item.name == name), None)

    def _current_rule(self) -> ColorRule | None:
        item = self._current_item()
        return item if isinstance(item, ColorRule) else None

    def _rebuild_rules(self, select: str | None) -> None:
        self._loading = True
        self.rules_tree.clear()
        for entry in self.profile.items:
            item = QTreeWidgetItem([entry.name, "–"])
            item.setData(0, Qt.UserRole, entry.name)
            is_template = isinstance(entry, TemplateRule)
            item.setToolTip(0, f"{entry.name} (wzorzec)" if is_template else f"{entry.name} (kolor)")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked if entry.enabled else Qt.Unchecked)
            item.setIcon(0, QIcon(picture_pixmap(entry.image, 16)) if is_template else swatch_icon(entry.draw_bgr))
            item.setTextAlignment(1, Qt.AlignRight | Qt.AlignVCenter)
            self.rules_tree.addTopLevelItem(item)
            if entry.name == select:
                item.setSelected(True)
                self.rules_tree.setCurrentItem(item)
        self._loading = False
        self._on_rule_selected()

    def _on_rule_selected(self) -> None:
        if self._loading:
            return
        entry = self._current_item()
        self.editor.setVisible(entry is not None)
        self.editor_empty.setVisible(entry is None)
        if entry is not None:
            self._load_editor(entry)
        if self._current_rule() is None and self.mask_action.isChecked():
            self.mask_action.setChecked(False)
        self._push_view()
        self._sync_actions()

    def _load_editor(self, entry: ColorRule | TemplateRule) -> None:
        self._loading = True
        self.name_edit.setText(entry.name)
        self.region_combo.setCurrentIndex(max(0, self.region_combo.findData(entry.region)))
        self._paint_color_button(entry.draw_bgr)
        is_template = isinstance(entry, TemplateRule)
        self.color_section.setVisible(not is_template)
        self.template_section.setVisible(is_template)
        self.editor_heading.setText(("Edytor wzorca" if is_template else "Edytor reguły").upper())
        if is_template:
            self.template_preview.setPixmap(picture_pixmap(entry.image, 100))
            self.template_size_label.setText(f"{entry.image.shape[1]}×{entry.image.shape[0]} px")
            self.threshold_slider.setValue(round(entry.threshold * 100))
            self.threshold_label.setText(f"{round(entry.threshold * 100)}%")
            self.max_results_spin.setValue(entry.max_results)
        else:
            self.h_row.set_values(entry.lower[0], entry.upper[0])
            self.s_row.set_values(entry.lower[1], entry.upper[1])
            self.v_row.set_values(entry.lower[2], entry.upper[2])
            self.shape_combo.setCurrentIndex(max(0, self.shape_combo.findData(entry.shape)))
            self.min_area_spin.setValue(entry.min_area)
            self.max_area_spin.setValue(entry.max_area or 0)
            self.merge_spin.setValue(entry.merge_px)
            self.min_confidence_spin.setValue(round(entry.min_confidence * 100))
            self._update_gradients()
        self._loading = False

    def _paint_color_button(self, bgr) -> None:
        color = QColor(bgr[2], bgr[1], bgr[0])
        self._draw_bgr = tuple(bgr)
        text = "#000000" if color.lightness() > 128 else "#ffffff"
        self.color_button.setText(color.name())
        self.color_button.setStyleSheet(f"background:{color.name()}; color:{text}; border-color:{color.name()};")

    def _update_gradients(self) -> None:
        low, high = self.h_row.values()
        hue = (low + high) // 2 if low <= high else ((low + high + 180) // 2) % 180
        self.s_row.slider.set_stops([QColor.fromHsv(hue * 2, 0, 255), QColor.fromHsv(hue * 2, 255, 255)])
        self.v_row.slider.set_stops([QColor.fromHsv(hue * 2, 255, 0), QColor.fromHsv(hue * 2, 255, 255)])

    def _apply_editor(self, *_args) -> None:
        entry = self._current_item()
        if self._loading or entry is None:
            return
        if isinstance(entry, TemplateRule):
            self._replace_rule(entry, replace(
                entry,
                region=self.region_combo.currentData(),
                threshold=self.threshold_slider.value() / 100,
                max_results=self.max_results_spin.value(),
                draw_bgr=self._draw_bgr,
            ))
            return
        h, s, v = self.h_row.values(), self.s_row.values(), self.v_row.values()
        self._replace_rule(entry, replace(
            entry,
            lower=(h[0], s[0], v[0]),
            upper=(h[1], s[1], v[1]),
            region=self.region_combo.currentData(),
            shape=self.shape_combo.currentData(),
            min_area=self.min_area_spin.value(),
            max_area=self.max_area_spin.value() or None,
            merge_px=self.merge_spin.value(),
            min_confidence=self.min_confidence_spin.value() / 100,
            draw_bgr=self._draw_bgr,
        ))
        self._update_gradients()

    def _on_threshold_changed(self, value: int) -> None:
        self.threshold_label.setText(f"{value}%")
        self._apply_editor()

    def _replace_rule(self, old, new) -> None:
        collection = self.profile.templates if isinstance(old, TemplateRule) else self.profile.rules
        index = next(i for i, entry in enumerate(collection) if entry.name == old.name)
        collection[index] = new
        self._mark_dirty()

    def _on_rule_checked(self, item: QTreeWidgetItem, column: int) -> None:
        if self._loading or column != 0:
            return
        name = item.data(0, Qt.UserRole)
        rule = next((entry for entry in self.profile.items if entry.name == name), None)
        enabled = item.checkState(0) == Qt.Checked
        if rule is not None and rule.enabled != enabled:
            self._replace_rule(rule, replace(rule, enabled=enabled))

    def _on_name_edited(self) -> None:
        rule = self._current_item()
        if rule is None or self._loading:
            return
        name = self.name_edit.text().strip()
        if name == rule.name:
            return
        if not name or any(entry.name == name for entry in self.profile.items):
            self.name_edit.setText(rule.name)
            self._warn("Nieprawidłowa nazwa", "Nazwa nie może być pusta ani powtarzać innej reguły lub wzorca.")
            return
        self.profile.states = rename_item(self.profile.states, rule.name, name)
        self.profile.decisions = [
            replace(d, conditions=tuple(replace(c, item=name) if c.item == rule.name else c for c in d.conditions))
            for d in self.profile.decisions
        ]
        self.profile.tasks = [
            replace(t, steps=tuple(replace(s, target=name) if s.target == rule.name else s for s in t.steps))
            for t in self.profile.tasks
        ]
        self._replace_rule(rule, replace(rule, name=name))
        self._rebuild_rules(name)
        self._rebuild_states()
        self._rebuild_automation()

    def _unique_rule_name(self, base: str) -> str:
        existing = {entry.name for entry in self.profile.items}
        if base not in existing:
            return base
        number = 2
        while f"{base}_{number}" in existing:
            number += 1
        return f"{base}_{number}"

    def _add_rule(self) -> ColorRule:
        rule = ColorRule(self._unique_rule_name("nowa_regula"), (0, 100, 100), (10, 255, 255))
        self.profile.rules.append(rule)
        self._mark_dirty()
        self._rebuild_rules(rule.name)
        self.name_edit.setFocus()
        self.name_edit.selectAll()
        return rule

    def _duplicate_rule(self) -> None:
        entry = self._current_item()
        if entry is not None:
            copy = replace(entry, name=self._unique_rule_name(f"{entry.name}_kopia"))
            (self.profile.templates if isinstance(entry, TemplateRule) else self.profile.rules).append(copy)
            self._mark_dirty()
            self._rebuild_rules(copy.name)

    def _delete_rule(self) -> None:
        entry = self._current_item()
        if entry is None:
            return
        users = [s.name for s in self.profile.states if any(c.item == entry.name for c in s.conditions)]
        warning = f"\n\nKorzystają z niej stany: {', '.join(users)}. Ich warunki przestaną być spełniane." if users else ""
        if self._confirm("Usuń", f"Czy usunąć „{entry.name}”?{warning}"):
            collection = self.profile.templates if isinstance(entry, TemplateRule) else self.profile.rules
            collection[:] = [other for other in collection if other.name != entry.name]
            self._mark_dirty()
            self._rebuild_rules(self.profile.items[0].name if self.profile.items else None)
            self._rebuild_states()

    # ---------------------------------------------------------------- automation

    def _automation_mode(self) -> str:
        return self.automation_combo.currentData()

    def _set_automation_mode(self, mode: str) -> None:
        self.automation_combo.setCurrentIndex(self.automation_combo.findData(mode))
        if self.worker is not None:
            self.worker.set_automation_mode(mode)
        self._refresh_banner(self.banner.text() if self._state in ("waiting", "error") else "")

    def _confirm_armed(self) -> bool:
        return self._confirm(
            "Prawdziwe sterowanie",
            "Program zacznie sam klikać i naciskać klawisze w oknie klienta.\n\n"
            "Używaj tego tylko w wersji offline. Na serwerach Jagex automatyzacja jest zabroniona i grozi banem.\n\n"
            "F12 zatrzymuje sterowanie natychmiast. Czy włączyć?",
        )

    def _on_automation_mode_chosen(self, _index: int = 0) -> None:
        mode = self._automation_mode()
        if mode == MODE_ARMED and (self.image_path is not None or not self._confirm_armed()):
            if self.image_path is not None:
                self._warn("Prawdziwe sterowanie", "Sterowanie działa tylko z oknem klienta, a teraz analizowany jest plik.")
            mode = MODE_OFF
        self._set_automation_mode(mode)
        self.append_log(f"Automatyzacja: {self.automation_combo.currentText()}")

    def _on_automation_stopped(self, reason: str) -> None:
        self._set_automation_mode(MODE_OFF)
        self.append_log(f"Automatyzacja wyłączona: {reason}")
        self.statusBar().showMessage(f"Automatyzacja wyłączona: {reason}", 8000)

    def _rebuild_automation(self) -> None:
        table = self.decisions_table
        table.blockSignals(True)
        table.setRowCount(len(self.profile.decisions))
        for row, decision in enumerate(self.profile.decisions):
            name = QTableWidgetItem(decision.name)
            name.setData(Qt.UserRole, decision.name)
            name.setFlags(name.flags() | Qt.ItemIsUserCheckable)
            name.setCheckState(Qt.Checked if decision.enabled else Qt.Unchecked)
            table.setItem(row, 0, name)
            table.setItem(row, 1, QTableWidgetItem(str(decision.priority)))
            table.setItem(row, 2, QTableWidgetItem(decision.task))
        table.blockSignals(False)
        self.tasks_table.setRowCount(len(self.profile.tasks))
        for row, task in enumerate(self.profile.tasks):
            name = QTableWidgetItem(task.name)
            name.setData(Qt.UserRole, task.name)
            self.tasks_table.setItem(row, 0, name)
            steps = ", ".join(STEP_LABELS[step.kind] for step in task.steps)
            summary = QTableWidgetItem(steps)
            summary.setToolTip(steps)
            self.tasks_table.setItem(row, 1, summary)

    def _selected_name(self, table: QTableWidget) -> str | None:
        row = table.currentRow()
        if row < 0 or not table.selectedItems() or table.item(row, 0) is None:
            return None
        return table.item(row, 0).data(Qt.UserRole)

    def _current_task(self) -> Task | None:
        name = self._selected_name(self.tasks_table)
        return next((t for t in self.profile.tasks if t.name == name), None)

    def _current_decision(self) -> Decision | None:
        name = self._selected_name(self.decisions_table)
        return next((d for d in self.profile.decisions if d.name == name), None)

    def _ask_task(self, task: Task | None) -> Task | None:
        taken = {t.name for t in self.profile.tasks} - ({task.name} if task else set())
        dialog = TaskDialog(task, [item.name for item in self.profile.items], taken, self)
        return dialog.result_task if dialog.exec() else None

    def _ask_decision(self, decision: Decision | None) -> Decision | None:
        taken = {d.name for d in self.profile.decisions} - ({decision.name} if decision else set())
        dialog = DecisionDialog(
            decision, [t.name for t in self.profile.tasks], [s.name for s in self.profile.states],
            [item.name for item in self.profile.items], taken, self,
        )
        return dialog.result_decision if dialog.exec() else None

    def _automation_changed(self) -> None:
        self._mark_dirty()
        self._rebuild_automation()

    def _add_task(self) -> None:
        task = self._ask_task(None)
        if task is not None:
            self.profile.tasks.append(task)
            self._automation_changed()

    def _edit_task(self) -> None:
        task = self._current_task()
        edited = self._ask_task(task) if task else None
        if edited is not None:
            self.profile.tasks[self.profile.tasks.index(task)] = edited
            if edited.name != task.name:
                self.profile.decisions = [
                    replace(d, task=edited.name) if d.task == task.name else d for d in self.profile.decisions
                ]
            self._automation_changed()

    def _delete_task(self) -> None:
        task = self._current_task()
        if task is None:
            return
        users = [d.name for d in self.profile.decisions if d.task == task.name]
        if users:
            self._warn("Zadanie jest używane", f"Zadania „{task.name}” używają decyzje: {', '.join(users)}. Najpierw je zmień lub usuń.")
            return
        if self._confirm("Usuń zadanie", f"Czy usunąć zadanie „{task.name}”?"):
            self.profile.tasks.remove(task)
            self._automation_changed()

    def _add_decision(self) -> None:
        if not self.profile.tasks:
            self._warn("Brak zadań", "Decyzja uruchamia zadanie, więc najpierw dodaj zadanie.")
            return
        decision = self._ask_decision(None)
        if decision is not None:
            self.profile.decisions.append(decision)
            self._automation_changed()

    def _edit_decision(self) -> None:
        decision = self._current_decision()
        edited = self._ask_decision(decision) if decision else None
        if edited is not None:
            self.profile.decisions[self.profile.decisions.index(decision)] = edited
            self._automation_changed()

    def _delete_decision(self) -> None:
        decision = self._current_decision()
        if decision is not None and self._confirm("Usuń decyzję", f"Czy usunąć decyzję „{decision.name}”?"):
            self.profile.decisions.remove(decision)
            self._automation_changed()

    def _on_decision_checked(self, item: QTableWidgetItem) -> None:
        if item.column() != 0:
            return
        name, enabled = item.data(Qt.UserRole), item.checkState() == Qt.Checked
        for index, decision in enumerate(self.profile.decisions):
            if decision.name == name and decision.enabled != enabled:
                self.profile.decisions[index] = replace(decision, enabled=enabled)
                self._mark_dirty()

    def _run_selected_task(self) -> None:
        task = self._current_task()
        if task is None:
            self._warn("Uruchom zadanie", "Najpierw zaznacz zadanie.")
        elif self.worker is None or self._automation_mode() == MODE_OFF:
            self._warn("Uruchom zadanie", "Naciśnij Start i wybierz tryb Symulacja albo Prawdziwe sterowanie.")
        else:
            self.worker.request_task(task.name)

    # ---------------------------------------------------------------- objects, events, history

    def _on_tracking_settings_changed(self) -> None:
        if not self._loading:
            self.profile.track_max_distance = self.track_distance_spin.value()
            self.profile.track_lost_ms = self.track_lost_spin.value()
            self._mark_dirty()

    def _update_objects_table(self, tracks: list) -> None:
        table = self.objects_table
        table.setRowCount(min(len(tracks), MAX_TABLE_ROWS))
        for row, track in enumerate(tracks[:MAX_TABLE_ROWS]):
            score = "" if track.score is None else f"{track.score:.0%}"
            status = ("W ruchu" if track.moving else "Stoi") if track.visible else "Niewidoczny"
            values = (
                f"#{track.id}", track.rule, track.center[0], track.center[1], f"{track.speed:.0f} px/s",
                format_duration(track.age), score, status,
            )
            for column, value in enumerate(values):
                table.setItem(row, column, QTableWidgetItem(str(value)))

    def _event_group_shown(self, kind: str) -> bool:
        return any(kind in kinds and self.event_filters[group].isChecked() for group, kinds in EVENT_GROUPS.items())

    def _on_event(self, event: Event) -> None:
        self.history.add_event(event)
        table = self.events_table
        table.insertRow(0)
        values = (
            time.strftime("%H:%M:%S", time.localtime(event.timestamp)),
            EVENT_LABELS.get(event.kind, event.kind), event.source, event.message,
        )
        for column, value in enumerate(values):
            table.setItem(0, column, QTableWidgetItem(value))
        table.item(0, 1).setData(Qt.UserRole, event.kind)
        table.setRowHidden(0, not self._event_group_shown(event.kind))
        if table.rowCount() > MAX_EVENT_ROWS:
            table.removeRow(table.rowCount() - 1)

    def _apply_event_filter(self) -> None:
        table = self.events_table
        for row in range(table.rowCount()):
            table.setRowHidden(row, not self._event_group_shown(table.item(row, 1).data(Qt.UserRole)))

    def _clear_events(self) -> None:
        self.events_table.setRowCount(0)
        self.history.events.clear()

    def _clear_history(self) -> None:
        self.history.clear()
        self._clear_events()
        self._history_rule = None
        self._refresh_history(force=True)

    def _ask_save_path(self, title: str, name: str) -> str:
        path, _ = QFileDialog.getSaveFileName(self, title, str(ROOT / name), "Plik CSV (*.csv)")
        return path

    def _export(self, write, title: str, base_name: str) -> None:
        path = self._ask_save_path(title, f"{base_name}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        if not path:
            return
        try:
            rows = write(path)
        except OSError as exc:
            self._warn("Nie można zapisać pliku", str(exc))
            return
        self.statusBar().showMessage(f"Zapisano {Path(path).name}, wierszy: {rows}", 4000)

    def _export_events(self) -> None:
        self._export(self.history.export_events_csv, "Eksport zdarzeń", "zdarzenia")

    def _export_samples(self) -> None:
        self._export(self.history.export_samples_csv, "Eksport historii wykryć", "historia")

    def _on_history_row_selected(self) -> None:
        row = self.history_table.currentRow()
        selected = self.history_table.selectedItems()
        if row >= 0 and selected:
            self._history_rule = self.history_table.item(row, 0).text()
            self._refresh_chart()

    def _refresh_chart(self) -> None:
        rule = self._history_rule
        points = self.history.series(rule, HISTORY_WINDOW_SECONDS, time.time()) if rule else []
        self.history_chart.set_series(f"{rule}: liczba obiektów" if rule else "Liczba obiektów", points)

    def _refresh_history(self, force: bool = False) -> None:
        """Runs once a second; skipped unless someone is looking at the tab."""
        if not force and self.tabs.currentWidget() is not self.history_tab:
            return
        rules = self.history.rules()
        if self._history_rule not in rules:
            self._history_rule = rules[0] if rules else None
        table = self.history_table
        table.blockSignals(True)
        table.setRowCount(len(rules))
        now = time.time()
        for row, rule in enumerate(rules):
            stats = self.history.stats(rule, HISTORY_WINDOW_SECONDS, now)
            score = "" if stats.mean_score is None else f"{stats.mean_score:.0%}"
            values = (rule, stats.current, f"{stats.average:.1f}", stats.maximum, score, stats.total_objects)
            for column, value in enumerate(values):
                table.setItem(row, column, QTableWidgetItem(str(value)))
            if rule == self._history_rule:
                table.setCurrentCell(row, 0)
                table.selectRow(row)
        table.blockSignals(False)
        self._refresh_chart()

    # ---------------------------------------------------------------- states

    def _state_summary(self, state: StateRule) -> str:
        joiner = " i " if state.mode == MODE_ALL else " lub "
        text = joiner.join(f"{c.item}: {OPERATOR_LABELS[c.op]} {c.value}" for c in state.conditions)
        missing = missing_items(state, (item.name for item in self.profile.items))
        return f"{text}  ⚠ brak: {', '.join(missing)}" if missing else text

    def _rebuild_states(self) -> None:
        table = self.states_table
        selected = self._current_state()
        table.blockSignals(True)
        table.setRowCount(len(self.profile.states))
        for row, state in enumerate(self.profile.states):
            name = QTableWidgetItem(state.name)
            name.setData(Qt.UserRole, state.name)
            name.setFlags(name.flags() | Qt.ItemIsUserCheckable)
            name.setCheckState(Qt.Checked if state.enabled else Qt.Unchecked)
            table.setItem(row, 0, name)
            table.setItem(row, 1, QTableWidgetItem("–"))
            table.setItem(row, 2, QTableWidgetItem("–"))
            summary = self._state_summary(state)
            summary_item = QTableWidgetItem(summary)
            summary_item.setToolTip(summary)
            table.setItem(row, 3, summary_item)
            if selected is not None and state.name == selected.name:
                table.selectRow(row)
        table.blockSignals(False)
        self._sync_state_buttons()
        self._show_state_values(self.last_packet.states if self.last_packet and self.worker else {})

    def _current_state(self) -> StateRule | None:
        row = self.states_table.currentRow()
        selected = self.states_table.selectedItems()
        if row < 0 or not selected or self.states_table.item(row, 0) is None:
            return None
        name = self.states_table.item(row, 0).data(Qt.UserRole)
        return next((s for s in self.profile.states if s.name == name), None)

    def _sync_state_buttons(self) -> None:
        has_state = self._current_state() is not None
        for button in (self.edit_state_button, self.duplicate_state_button, self.delete_state_button):
            button.setEnabled(has_state)

    def _ask_state(self, state: StateRule | None) -> StateRule | None:
        """Show the editor; returns the edited state, or None if cancelled."""
        taken = {s.name for s in self.profile.states} - ({state.name} if state else set())
        dialog = StateDialog(state, [item.name for item in self.profile.items], taken, self)
        return dialog.result_rule if dialog.exec() else None

    def _add_state(self) -> None:
        state = self._ask_state(None)
        if state is not None:
            self.profile.states.append(state)
            self._mark_dirty()
            self._rebuild_states()

    def _edit_state(self) -> None:
        state = self._current_state()
        if state is None:
            return
        edited = self._ask_state(state)
        if edited is not None:
            self.profile.states[self.profile.states.index(state)] = edited
            self._mark_dirty()
            self._rebuild_states()

    def _duplicate_state(self) -> None:
        state = self._current_state()
        if state is None:
            return
        taken = {s.name for s in self.profile.states}
        name, number = f"{state.name} kopia", 2
        while name in taken:
            name, number = f"{state.name} kopia {number}", number + 1
        self.profile.states.append(replace(state, name=name))
        self._mark_dirty()
        self._rebuild_states()

    def _delete_state(self) -> None:
        state = self._current_state()
        if state is not None and self._confirm("Usuń stan", f"Czy usunąć stan „{state.name}”?"):
            self.profile.states.remove(state)
            self._mark_dirty()
            self._rebuild_states()

    def _on_state_checked(self, item: QTableWidgetItem) -> None:
        if item.column() != 0:
            return
        name = item.data(Qt.UserRole)
        enabled = item.checkState() == Qt.Checked
        for index, state in enumerate(self.profile.states):
            if state.name == name and state.enabled != enabled:
                self.profile.states[index] = replace(state, enabled=enabled)
                self._mark_dirty()

    def _show_state_values(self, statuses: dict) -> None:
        table = self.states_table
        table.blockSignals(True)
        for row in range(table.rowCount()):
            status = statuses.get(table.item(row, 0).data(Qt.UserRole))
            now_item, since_item = table.item(row, 1), table.item(row, 2)
            if status is None:
                now_text, since_text, color = "–", "–", theme.MUTED
            else:
                now_text = "Aktywny" if status.active else "Nieaktywny"
                since_text = format_duration(status.duration)
                color = theme.ACCENT_HOVER if status.active else theme.MUTED
            if now_item.text() != now_text:
                now_item.setText(now_text)
                now_item.setForeground(QColor(color))
            if since_item.text() != since_text:
                since_item.setText(since_text)
        table.blockSignals(False)
        active = [name for name, status in statuses.items() if status.active]
        self.active_states_label.setText(f"Stany: {', '.join(active)}" if active else "")

    def _on_game_state_changed(self, change: StateChange) -> None:
        event = "start" if change.active else f"koniec po {format_duration(change.previous_duration)}"
        text = f"{time.strftime('%H:%M:%S', time.localtime(change.timestamp))}  {change.name}: {event}"
        self.states_history.insertItem(0, text)
        while self.states_history.count() > MAX_HISTORY_ROWS:
            self.states_history.takeItem(self.states_history.count() - 1)
        self.append_log(f"Stan {change.name}: {event}")

    def _choose_draw_color(self) -> None:
        bgr = self._draw_bgr
        color = QColorDialog.getColor(QColor(bgr[2], bgr[1], bgr[0]), self, "Kolor ramki")
        if color.isValid():
            self._paint_color_button((color.blue(), color.green(), color.red()))
            self._apply_editor()
            items = self.rules_tree.selectedItems()
            if items and self._current_rule() is not None:
                self._loading = True
                items[0].setIcon(0, swatch_icon(self._draw_bgr))
                self._loading = False

    # ---------------------------------------------------------------- regions

    def _rebuild_regions(self) -> None:
        self._loading = True
        for spin, value in zip((self.ref_w_spin, self.ref_h_spin), self.profile.reference_size):
            spin.setValue(value)
        self.regions_table.setRowCount(len(self.profile.regions))
        for row, (name, rect) in enumerate(self.profile.regions.items()):
            item = QTableWidgetItem(name)
            item.setData(Qt.UserRole, name)
            self.regions_table.setItem(row, 0, item)
            for col, value in enumerate(rect, start=1):
                self.regions_table.setItem(row, col, QTableWidgetItem(str(value)))
        self._loading = False

    def _rebuild_region_combo(self) -> None:
        self._loading = True
        current = self.region_combo.currentData()
        self.region_combo.clear()
        self.region_combo.addItem("Cały obraz", None)
        for name in self.profile.regions:
            self.region_combo.addItem(name, name)
        self.region_combo.setCurrentIndex(max(0, self.region_combo.findData(current)))
        self._loading = False

    def _regions_changed(self) -> None:
        self._rebuild_region_combo()
        self._mark_dirty()

    def _on_reference_changed(self) -> None:
        if not self._loading:
            self.profile.reference_size = (self.ref_w_spin.value(), self.ref_h_spin.value())
            self._mark_dirty()

    def _on_region_edited(self, item: QTableWidgetItem) -> None:
        if self._loading:
            return
        row = item.row()
        old_name = self.regions_table.item(row, 0).data(Qt.UserRole)
        name = self.regions_table.item(row, 0).text().strip()
        valid = bool(name) and (name == old_name or name not in self.profile.regions)
        try:
            rect = tuple(int(self.regions_table.item(row, col).text()) for col in range(1, 5))
            valid = valid and rect[0] >= 0 and rect[1] >= 0 and rect[2] >= 1 and rect[3] >= 1
        except ValueError:
            valid = False
        if not valid:
            # Rebuilt after the signal returns: the table is still committing the edit.
            QTimer.singleShot(0, self._rebuild_regions)
            self._warn(
                "Nieprawidłowy region",
                "Nazwa musi być niepowtarzalna, X i Y nie mogą być ujemne, a szerokość i wysokość muszą wynosić co najmniej 1.",
            )
            return
        # Rebuilding the dict keeps the order of regions when one is renamed.
        self.profile.regions = {
            (name if key == old_name else key): (rect if key == old_name else value)
            for key, value in self.profile.regions.items()
        }
        if name != old_name:
            self.profile.rules = [replace(r, region=name) if r.region == old_name else r for r in self.profile.rules]
            self.profile.templates = [replace(t, region=name) if t.region == old_name else t for t in self.profile.templates]
            self.regions_table.item(row, 0).setData(Qt.UserRole, name)
        self._regions_changed()
        entry = self._current_item()
        if entry is not None:
            self._load_editor(entry)

    def _add_region(self) -> None:
        number, name = 1, "region_1"
        while name in self.profile.regions:
            number += 1
            name = f"region_{number}"
        self.profile.regions[name] = (0, 0, 100, 100)
        self._rebuild_regions()
        self._regions_changed()

    def _delete_region(self) -> None:
        row = self.regions_table.currentRow()
        if row < 0:
            self._warn("Usuń region", "Najpierw zaznacz region w tabeli.")
            return
        name = self.regions_table.item(row, 0).data(Qt.UserRole)
        users = [entry.name for entry in self.profile.items if entry.region == name]
        if users:
            self._warn("Region jest używany", f"Region „{name}” jest używany przez: {', '.join(users)}. Najpierw zmień region tych reguł.")
            return
        del self.profile.regions[name]
        self._rebuild_regions()
        self._regions_changed()

    # ---------------------------------------------------------------- plugins

    def _disabled_plugins(self) -> set[str]:
        return {name for name in str(self.settings.value("disabled_plugins", "")).split(",") if name}

    def _refresh_plugins(self) -> None:
        disabled = self._disabled_plugins()
        self.plugins_list.blockSignals(True)
        self.plugins_list.clear()
        for plugin in load_plugins(self.plugins_dir):
            item = QListWidgetItem(plugin.name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked if plugin.name in disabled else Qt.Checked)
            self.plugins_list.addItem(item)
        self.plugins_list.blockSignals(False)

    def _on_plugin_checked(self, _item: QListWidgetItem) -> None:
        disabled = [
            self.plugins_list.item(i).text()
            for i in range(self.plugins_list.count())
            if self.plugins_list.item(i).checkState() != Qt.Checked
        ]
        self.settings.setValue("disabled_plugins", ",".join(disabled))

    def _open_plugins_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.plugins_dir)))

    # ---------------------------------------------------------------- running

    def _toggle_run(self, checked: bool) -> None:
        if checked:
            self._start_worker()
        else:
            self._stop_worker()

    def _start_worker(self) -> None:
        if self.worker is not None:
            return
        title = self.window_combo.currentText().strip()
        if self.image_path is None and not title:
            self._warn("Brak okna", "Wpisz tytuł okna klienta albo wybierz je z listy.")
            self._sync_actions()
            return
        self.settings.setValue("window_title", title)
        path, hwnd = self.image_path, self.window_hwnd
        factory = (lambda: ImageSource(path)) if path else (lambda: WindowSource(title, hwnd))
        disabled = self._disabled_plugins()
        plugins = [p for p in load_plugins(self.plugins_dir) if p.name not in disabled]
        self.worker = VisionWorker(factory, self.profile, plugins, self.fps_spin.value())
        self.worker.packetReady.connect(self._on_packet)
        self.worker.stateChanged.connect(self._on_worker_state)
        self.worker.gameStateChanged.connect(self._on_game_state_changed)
        self.worker.eventDetected.connect(self._on_event)
        self.worker.automationStopped.connect(self._on_automation_stopped)
        self.worker.sourceInfoChanged.connect(self._on_source_info)
        self.worker.set_automation_mode(self._automation_mode())
        self.worker.set_crop(self.crop)
        self.worker.sampleReady.connect(self.history.add_sample)
        self.worker.finished.connect(self._on_worker_finished)
        self._push_view()
        self.append_log(f"Start: {path or title}, pluginy: {[p.name for p in plugins] or 'brak'}")
        self.worker.start()
        self._set_state("waiting")
        self._sync_actions()

    def _stop_worker_and_wait(self) -> None:
        """Stop now, so that a new worker can be started straight away.

        The finished signal must be disconnected first. Otherwise it arrives later, while
        the new worker runs, and would be taken for that worker's end: the running thread
        would be destroyed, which takes the whole program down.
        """
        worker = self.worker
        if worker is None:
            return
        worker.finished.disconnect(self._on_worker_finished)
        worker.requestInterruption()
        if not worker.wait(WORKER_STOP_TIMEOUT_MS):
            # Stuck, e.g. in a plugin. A running QThread must never be deleted, so keep it.
            worker.setParent(self)
            self.append_log("Poprzedni wątek jeszcze się nie zakończył")
            self.worker, abandoned = None, worker
            self._abandoned_workers.append(abandoned)
            self.start_action.setEnabled(True)
            self._sync_actions()
            return
        self._on_worker_finished()

    def _stop_worker(self) -> None:
        if self.worker is not None:
            self.worker.requestInterruption()
            self.start_action.setEnabled(False)

    def _on_worker_finished(self) -> None:
        worker, self.worker = self.worker, None
        if worker is not None:
            worker.deleteLater()
        self.start_action.setEnabled(True)
        if self._state != "error":
            self._set_state("idle")
        self.append_log("Zatrzymano")
        self._show_state_values({})
        self.objects_table.setRowCount(0)
        # Control never carries over to the next run: it must be switched on again on purpose.
        self.automation_combo.setCurrentIndex(self.automation_combo.findData(MODE_OFF))
        self._source_info = None
        if self._selecting_crop:
            self.crop_action.setChecked(False)
        self._refresh_banner(self.banner.text() if self._state == "error" else "")
        self._sync_actions()

    def _on_worker_state(self, state: str, message: str) -> None:
        self._set_state(state, message)

    def _on_packet(self, packet: FramePacket) -> None:
        if self.worker is None:
            return
        first = self.last_packet is None
        self.last_packet = packet
        self.view.set_image(packet.image)
        self.fps_label.setText(f"{packet.fps:.0f} FPS")
        height, width = packet.frame.shape[:2]
        self.size_label.setText(f"{width}×{height}")
        self._refresh_banner(self.banner.text() if self._state in ("waiting", "error") else "")
        now = time.perf_counter()
        if now - self._last_table_refresh >= TABLE_REFRESH_SECONDS:
            self._last_table_refresh = now
            self._update_counts(packet)
            self._show_state_values(packet.states)
            self.task_label.setText(f"Zadanie: {packet.task}" if packet.task else "")
            if self.tabs.currentWidget() is self.objects_tab:
                self._update_objects_table(packet.tracks)
            if self.tabs.currentWidget() is self.detections_table:
                self._update_table(packet)
        if first:
            self._sync_actions()
        self.worker.acknowledge_packet()

    def _update_counts(self, packet: FramePacket) -> None:
        self._loading = True
        for index in range(self.rules_tree.topLevelItemCount()):
            item = self.rules_tree.topLevelItem(index)
            found = packet.detections.get(item.data(0, Qt.UserRole))
            text = "–" if found is None else str(len(found))
            if item.text(1) != text:
                item.setText(1, text)
        self._loading = False

    def _update_table(self, packet: FramePacket) -> None:
        rows = [d for found in packet.detections.values() for d in found][:MAX_TABLE_ROWS]
        self.detections_table.setRowCount(len(rows))
        for row, d in enumerate(rows):
            score = "" if d.score is None else f"{d.score:.0%}"
            values = (
                d.rule, SHAPE_LABELS.get(d.shape, d.shape), d.x, d.y, d.width, d.height, d.area,
                f"{d.center[0]}, {d.center[1]}", score,
            )
            for col, value in enumerate(values):
                self.detections_table.setItem(row, col, QTableWidgetItem(str(value)))

    def _toggle_pause(self, paused: bool) -> None:
        if self.worker is not None:
            self.worker.set_paused(paused)

    def _toggle_pipette(self, active: bool) -> None:
        if active and self.template_action.isChecked():
            self.template_action.setChecked(False)
        self.view.setCursor(Qt.CrossCursor if active else Qt.ArrowCursor)

    def _toggle_template_cut(self, active: bool) -> None:
        if active and self._selecting_crop:
            self.crop_action.setChecked(False)
        self.view.set_selecting(active)
        if active:
            if self.pipette_action.isChecked():
                self.pipette_action.setChecked(False)
            self.view.setCursor(Qt.CrossCursor)
            # A still picture makes it possible to mark a moving object precisely.
            if self.worker is not None and not self.pause_action.isChecked():
                self._auto_paused = True
                self.pause_action.setChecked(True)
            self.statusBar().showMessage("Zaznacz obiekt na obrazie: przeciągnij myszą od rogu do rogu.")
            return
        self.view.setCursor(Qt.ArrowCursor)
        self.statusBar().clearMessage()
        self._recut_target = None
        if self._auto_paused:
            self._auto_paused = False
            self.pause_action.setChecked(False)

    def _recut_template(self) -> None:
        entry = self._current_item()
        if not isinstance(entry, TemplateRule):
            return
        if self.last_packet is None:
            self._warn("Brak obrazu", "Najpierw naciśnij Start, aby pojawił się obraz do wycięcia.")
            return
        self.template_action.setChecked(True)
        self._recut_target = entry.name

    def _on_area_selected(self, x: int, y: int, width: int, height: int) -> None:
        if self.last_packet is None:
            return
        if self._selecting_crop:
            self._apply_selected_crop(x, y, width, height)
            return
        frame = self.last_packet.frame
        picture = frame[y:y + height, x:x + width].copy()
        if not template_is_usable(picture):
            self._warn(
                "Nie można użyć tego fragmentu",
                "Zaznacz obszar o wyraźnym wyglądzie, co najmniej 4×4 piksele. Jednolity kolor nie nadaje się na wzorzec, do niego użyj reguły koloru.",
            )
            return
        captured = (frame.shape[1], frame.shape[0])
        target = next((t for t in self.profile.templates if t.name == self._recut_target), None)
        if target is not None:
            self._replace_rule(target, replace(target, image=picture, captured_size=captured))
            name = target.name
        else:
            name = self._unique_rule_name("wzorzec")
            self.profile.templates.append(TemplateRule(name, picture, captured_size=captured))
            self._mark_dirty()
        self.template_action.setChecked(False)
        self._rebuild_rules(name)
        if target is None:
            self.name_edit.setFocus()
            self.name_edit.selectAll()

    def _push_view(self, *_args) -> None:
        if self.worker is not None:
            mode = MODE_MASK if self.mask_action.isChecked() else MODE_DETECTIONS
            entry = self._current_item()
            self.worker.set_view(mode, entry.name if entry else None, self.debug_action.isChecked())

    def _on_fps_changed(self, value: int) -> None:
        self.settings.setValue("fps", value)
        if self.worker is not None:
            self.worker.set_fps(value)

    def _on_window_text_changed(self, _text: str) -> None:
        self.image_path = None
        self.window_hwnd = None
        self._clear_crop()
        self._sync_actions()

    def _on_window_chosen(self, _index: int = 0) -> None:
        """Picked from the list: remember this exact window, not just its title."""
        self.window_hwnd = self.window_combo.currentData()
        self._sync_actions()

    def _start_window_pick(self) -> None:
        self._pick_seconds_left = PICK_WINDOW_SECONDS
        self.pick_window_action.setEnabled(False)
        self._show_pick_prompt()
        self._pick_timer.start()

    def _show_pick_prompt(self) -> None:
        self.statusBar().showMessage(f"Kliknij teraz okno gry. Zostało sekund: {self._pick_seconds_left}")

    def _on_pick_tick(self) -> None:
        self._pick_seconds_left -= 1
        if self._pick_seconds_left > 0:
            self._show_pick_prompt()
            return
        self._pick_timer.stop()
        self.pick_window_action.setEnabled(True)
        self._finish_window_pick(foreground_window())

    def _finish_window_pick(self, window) -> None:
        self.statusBar().clearMessage()
        if window is None:
            self._warn("Nie wskazano okna", "Na czas odliczania kliknij okno gry, a nie ten program ani pulpit.")
            return
        was_running = self.worker is not None
        self._stop_worker_and_wait()
        self.window_combo.setEditText(window.title)  # also clears the old selection
        self.window_hwnd = window.hwnd
        self.statusBar().showMessage(f"Wybrano okno: {window.title}", 5000)
        if was_running:
            self._start_worker()
        self._sync_actions()

    def _on_source_info(self, info: dict | None) -> None:
        self._source_info = info
        self._refresh_banner(self.banner.text() if self._state in ("waiting", "error") else "")
        self._sync_actions()

    def _stored_crop(self) -> tuple[int, int, int, int] | None:
        try:
            values = tuple(int(v) for v in str(self.settings.value("crop", "")).split(","))
        except ValueError:
            return None
        return values if len(values) == 4 else None

    def _store_crop(self) -> None:
        self.settings.setValue("crop", ",".join(str(v) for v in self.crop) if self.crop else "")

    def _clear_crop(self) -> None:
        self.crop = None
        self._store_crop()
        if self.worker is not None:
            self.worker.set_crop(None)

    def _toggle_crop_select(self, active: bool) -> None:
        if self.crop is not None:
            # The button reads "Cały obraz" while a crop is set: it returns to the whole window.
            self._clear_crop()
            self.statusBar().showMessage("Przechwytywany jest znów cały obraz okna", 4000)
            self._sync_actions()
            return
        self._selecting_crop = active
        self.view.set_selecting(active)
        if active:
            if self.template_action.isChecked():
                self.template_action.setChecked(False)
            if self.pipette_action.isChecked():
                self.pipette_action.setChecked(False)
            self.view.set_selecting(True)
            self.view.setCursor(Qt.CrossCursor)
            if self.worker is not None and not self.pause_action.isChecked():
                self._auto_paused = True
                self.pause_action.setChecked(True)
            self.statusBar().showMessage("Zaznacz grę na obrazie: przeciągnij myszą od rogu do rogu.")
            return
        self.view.setCursor(Qt.ArrowCursor)
        self.statusBar().clearMessage()
        if self._auto_paused:
            self._auto_paused = False
            self.pause_action.setChecked(False)

    def _apply_selected_crop(self, x: int, y: int, width: int, height: int) -> None:
        if width < MIN_CROP_SIDE or height < MIN_CROP_SIDE:
            self._warn("Za mały obszar", f"Zaznacz obszar o wymiarach co najmniej {MIN_CROP_SIDE}×{MIN_CROP_SIDE} pikseli.")
            return
        self.crop_action.setChecked(False)  # ends selecting and the automatic pause
        self.crop = (x, y, width, height)
        self._store_crop()
        if self.worker is not None:
            self.worker.set_crop(self.crop)
        self._sync_actions()

    def _open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Wybierz obraz", str(ROOT), "Obrazy (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.use_image(path)

    def use_image(self, path: str) -> None:
        """Analyse a saved image instead of the client window."""
        self._stop_worker_and_wait()
        self.image_path = path
        self.crop = None  # a crop chosen for a window does not apply to a saved picture
        self._start_worker()

    def _save_snapshot(self) -> None:
        if self.last_packet is None:
            return
        name = f"klatka_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path, _ = QFileDialog.getSaveFileName(self, "Zapisz klatkę", str(ROOT / name), "Obraz PNG (*.png)")
        if path:
            try:
                write_png(path, self.last_packet.frame)
                self.statusBar().showMessage(f"Zapisano {Path(path).name}", 3000)
            except (OSError, ValueError) as exc:
                self._warn("Nie można zapisać klatki", str(exc))

    # ---------------------------------------------------------------- image interaction

    def _on_image_hovered(self, x: int, y: int) -> None:
        if self.last_packet is None:
            return
        frame = self.last_packet.frame
        if y >= frame.shape[0] or x >= frame.shape[1]:
            return
        pixel = frame[y:y + 1, x:x + 1]
        b, g, r = (int(v) for v in pixel[0, 0])
        h, s, v = (int(v) for v in cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0])
        self.pixel_label.setText(f"x {x}, y {y}   RGB {r} {g} {b}   HSV {h} {s} {v}")

    def _on_image_clicked(self, x: int, y: int) -> None:
        if not self.pipette_action.isChecked() or self.last_packet is None:
            return
        rule = self._current_rule() or self._add_rule()
        hsv = cv2.cvtColor(self.last_packet.frame, cv2.COLOR_BGR2HSV)
        lower, upper = range_from_sample(hsv, x, y)
        new = replace(rule, lower=lower, upper=upper)
        self._replace_rule(rule, new)
        self._load_editor(new)
        self.pipette_action.setChecked(False)
        self.statusBar().showMessage(f"Pobrano kolor do reguły {new.name}", 3000)

    # ---------------------------------------------------------------- dialogs and shutdown

    def _show_help(self) -> None:
        QMessageBox.information(
            self, "Jak zacząć",
            "<b>1. Wybierz okno</b><br>Rozwiń listę „Okno klienta” albo naciśnij Wskaż okno (F2) i w ciągu 3 sekund kliknij grę. "
            "Okno nie może być zminimalizowane ani przykryte. Gdy gra zajmuje cały ekran, po Start użyj Obszar (F3) i zaznacz jej fragment.<br><br>"
            "<b>2. Naciśnij Start (F5)</b><br>Obraz i wykryte obiekty pojawią się w środkowej części.<br><br>"
            "<b>3. Ustaw kolor</b><br>Wybierz regułę, naciśnij Pipeta (F7) i kliknij kolor na obrazie. Zakres popraw suwakami. "
            "Maska (F8) pokazuje dokładnie, co pasuje do reguły.<br><br>"
            "<b>4. Zawęź wyniki kształtem</b><br>W regule wybierz kształt, np. Pasek, aby odrzucić inne obiekty w tym samym kolorze.<br><br>"
            "<b>5. Rozpoznawaj konkretne rzeczy</b><br>Naciśnij Wytnij wzorzec (F10) i przeciągnij myszą po obiekcie, np. przedmiocie w ekwipunku. "
            "Program znajdzie miejsca, które wyglądają tak samo.<br><br>"
            "<b>6. Śledź stany</b><br>W zakładce Stany określ warunki, np. „W walce”, gdy widać pasek HP. "
            "Program pokaże, czy stan jest aktywny i jak długo trwa.<br><br>"
            "<b>7. Obserwuj obiekty</b><br>Zakładki Obiekty, Zdarzenia i Historia pokazują numery obiektów, co się z nimi działo i wykres z ostatniej minuty. "
            "Debug (F11) nakłada te informacje na podgląd.<br><br>"
            "<b>8. Automatyzuj (tylko offline)</b><br>W zakładce Automat utwórz zadania i decyzje. Najpierw sprawdź je w trybie Symulacja. "
            "F12 zatrzymuje prawdziwe sterowanie.<br><br>"
            "<b>9. Zapisz profil</b> (Ctrl+S). Wzorce, stany i zadania zapisują się w tym samym pliku.<br><br>"
            "Skróty: F2 Wskaż okno, F3 Obszar, F5 Start/Stop, F6 Pauza, F7 Pipeta, F8 Maska, F9 Zapis klatki, F10 Wycinanie wzorca, F11 Debug.",
        )

    def _show_about(self) -> None:
        QMessageBox.about(
            self, f"O programie",
            f"<b>{APP_NAME}</b><br>Wykrywanie obiektów po kolorze w oknie klienta.<br><br>"
            "Program czyta obraz z ekranu i nie modyfikuje klienta. Myszą i klawiaturą steruje tylko po włączeniu "
            "trybu Prawdziwe sterowanie, przeznaczonego do wersji offline.",
        )

    def closeEvent(self, event) -> None:
        if not self._maybe_save():
            event.ignore()
            return
        if self.worker is not None:
            self.worker.requestInterruption()
            self.worker.wait(3000)
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("window_title", self.window_combo.currentText())
        self.settings.sync()
        event.accept()


class LogStream(QObject):
    """Sends print() output, including from plugins, to the log tab."""

    text = Signal(str)

    def __init__(self):
        super().__init__()
        self._buffer = ""

    def write(self, data: str) -> int:
        self._buffer += data
        if "\n" in self._buffer:
            complete, self._buffer = self._buffer.rsplit("\n", 1)
            self.text.emit(complete)
        return len(data)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


def run_app() -> int:
    prepare_user_files(DATA_ROOT, bundled_files())
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    theme.apply_palette(app)
    app.setStyleSheet(theme.STYLESHEET)
    window = MainWindow()
    window.setAttribute(Qt.WA_DeleteOnClose, False)

    stream = LogStream()
    stream.text.connect(window.append_log)
    sys.stdout = stream
    sys.stderr = stream
    if bundled_files() is not None:
        window.append_log(f"Profile, pluginy i ustawienia są w folderze: {DATA_ROOT}")

    def on_exception(kind, value, trace) -> None:
        details = "".join(traceback.format_exception(kind, value, trace))
        log_path = ROOT / "colorvision_error.log"
        try:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n{details}")
        except OSError:
            pass
        window.append_log(details)
        QMessageBox.critical(window, "Nieoczekiwany błąd", f"{value}\n\nSzczegóły zapisano w pliku:\n{log_path}")

    sys.excepthook = on_exception
    window.show()
    return app.exec()
