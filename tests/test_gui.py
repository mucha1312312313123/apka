import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, QPoint, QSize, Qt, QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from colorvision.capture import write_png
from colorvision.detector import ColorRule
from colorvision.gui import theme
from colorvision.gui.main_window import MainWindow
from colorvision.gui.widgets import FrameView, RangeSlider
from colorvision.plugin import process_frame
from colorvision.profile import Profile, load_profile, save_profile

GREEN = (0, 255, 0)


@pytest.fixture(scope="session")
def app():
    application = QApplication.instance() or QApplication([])
    application.setStyleSheet(theme.STYLESHEET)
    return application


def synthetic_frame():
    frame = np.full((503, 765, 3), (40, 70, 90), np.uint8)
    frame[100:105, 200:225] = GREEN
    frame[50:53, 600:603] = (0, 255, 255)
    return frame


def spin(app, condition, timeout_ms=5000):
    """Process events until the condition holds or the time runs out."""
    loop = QEventLoop()
    timer = QTimer()
    timer.timeout.connect(lambda: condition() and loop.quit())
    timer.start(10)
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    timer.stop()
    return condition()


@pytest.fixture
def window(app, tmp_path):
    profile_path = tmp_path / "profile.json"
    save_profile(
        Profile(
            (765, 503),
            {"game_view": (4, 4, 512, 334), "minimap": (550, 4, 210, 160)},
            [
                ColorRule("green", (55, 200, 150), (65, 255, 255), region="game_view", min_area=12),
                ColorRule("yellow", (25, 180, 180), (35, 255, 255), region="minimap", min_area=3, merge_px=0),
            ],
        ),
        profile_path,
    )
    win = MainWindow(settings_path=tmp_path / "s.ini", plugins_dir=tmp_path / "no_plugins", profile_path=profile_path)
    win.warnings = []
    win._warn = lambda title, text: win.warnings.append(title)
    win._confirm = lambda title, text: True
    yield win
    if win.worker is not None:
        win.worker.requestInterruption()
        win.worker.wait(3000)
    win.dirty = False
    win.close()


def select(window, name):
    for i in range(window.rules_tree.topLevelItemCount()):
        item = window.rules_tree.topLevelItem(i)
        if item.data(0, Qt.UserRole) == name:
            window.rules_tree.clearSelection()
            item.setSelected(True)
            return
    raise AssertionError(name)


def test_disabled_rule_is_not_detected():
    profile = Profile(rules=[ColorRule("g", (55, 200, 200), (65, 255, 255), enabled=False)])
    assert process_frame(synthetic_frame(), profile).detections == {}


def test_enabled_flag_round_trip(tmp_path):
    path = tmp_path / "p.json"
    save_profile(Profile(rules=[ColorRule("g", (1, 2, 3), (4, 5, 6), enabled=False)]), path)
    assert load_profile(path).rules[0].enabled is False


def test_range_slider_wrap_rules(app):
    plain = RangeSlider(255)
    plain.set_range(200, 100)
    assert (plain.low, plain.high) == (100, 200)
    hue = RangeSlider(179, wrap=True)
    hue.set_range(175, 3)
    assert (hue.low, hue.high) == (175, 3)


def test_frame_view_maps_positions_to_image_pixels(app):
    view = FrameView()
    view.resize(400, 400)
    image = QImage(QSize(200, 100), QImage.Format_RGB888)
    view.set_image(image)
    assert view.target_rect().height() == 200  # letterboxed, centred vertically
    assert view.to_image(QPoint(200, 50)) is None  # black bar
    assert view.to_image(QPoint(0, 100)) == (0, 0)
    assert view.to_image(QPoint(399, 299)) == (199, 99)


def test_number_fields_have_room_for_their_text(window, app):
    """Regression: inherited padding squeezed the inner line edit and hid the digits."""
    window.show()
    app.processEvents()
    select(window, "green")
    app.processEvents()
    spins = [window.h_row.low_spin, window.s_row.high_spin, window.fps_spin, window.min_area_spin]
    for spin in spins:
        edit = spin.lineEdit()
        assert edit.height() >= edit.fontMetrics().height(), spin
        assert edit.width() >= edit.fontMetrics().horizontalAdvance("255"), spin
        assert spin.rect().contains(edit.geometry()), spin


def test_editor_edit_marks_dirty_and_saves(window):
    select(window, "green")
    assert window.editor.isHidden() is False
    assert window.dirty is False
    window.h_row.low_spin.setValue(50)
    assert window.dirty is True
    assert window.windowTitle().endswith("•")
    assert window._save() is True
    assert window.dirty is False
    assert load_profile(window.profile_path).rules[0].lower[0] == 50


def test_loading_a_rule_does_not_mark_dirty(window):
    select(window, "yellow")
    select(window, "green")
    assert window.dirty is False


def test_rule_can_be_renamed_but_not_to_a_duplicate(window):
    select(window, "green")
    window.name_edit.setText("yellow")
    window._on_name_edited()
    assert window.warnings == ["Nieprawidłowa nazwa"]
    assert window.name_edit.text() == "green"
    window.name_edit.setText("zielony")
    window._on_name_edited()
    assert [r.name for r in window.profile.rules] == ["zielony", "yellow"]
    assert window._current_rule().name == "zielony"


def test_add_duplicate_delete_rule(window):
    window._add_rule()
    window._add_rule()
    assert [r.name for r in window.profile.rules][-2:] == ["nowa_regula", "nowa_regula_2"]
    window._duplicate_rule()
    assert window.profile.rules[-1].name == "nowa_regula_2_kopia"
    window._delete_rule()
    assert "nowa_regula_2_kopia" not in [r.name for r in window.profile.rules]


def test_unchecking_rule_disables_it(window):
    item = window.rules_tree.topLevelItem(0)
    item.setCheckState(0, Qt.Unchecked)
    assert window.profile.rules[0].enabled is False
    assert window.dirty is True


def test_region_in_use_cannot_be_deleted_and_rename_updates_rules(window, app):
    window.regions_table.setCurrentCell(0, 0)
    window._delete_region()
    assert window.warnings == ["Region jest używany"]
    assert "game_view" in window.profile.regions
    window.regions_table.item(0, 0).setText("gra")
    assert window.profile.rules[0].region == "gra"
    assert list(window.profile.regions)[0] == "gra"


def test_invalid_region_value_is_reverted(window, app):
    window.regions_table.item(0, 3).setText("abc")
    assert window.warnings == ["Nieprawidłowy region"]
    assert spin(app, lambda: window.regions_table.item(0, 3).text() == "512")
    assert window.profile.regions["game_view"] == (4, 4, 512, 334)


def test_worker_analyses_image_and_pipette_samples_colour(window, app, tmp_path):
    image = tmp_path / "klatka.png"
    write_png(image, synthetic_frame())
    window.use_image(str(image))
    assert spin(app, lambda: window.last_packet is not None and window.rules_tree.topLevelItem(0).text(1) == "1")
    assert window.rules_tree.topLevelItem(1).text(1) == "1"
    assert window.snapshot_action.isEnabled()
    assert window.source_label.text() == "Plik: klatka.png"

    # Editing while running reaches the worker: narrow the range so green stops matching.
    select(window, "green")
    window.h_row.set_values(100, 110)
    window._apply_editor()
    assert spin(app, lambda: window.rules_tree.topLevelItem(0).text(1) == "0")

    # The pipette restores a matching range from the clicked pixel and switches itself off.
    window.pipette_action.setChecked(True)
    window._on_image_clicked(210, 102)
    assert window.pipette_action.isChecked() is False
    assert spin(app, lambda: window.rules_tree.topLevelItem(0).text(1) == "1")

    window.start_action.setChecked(False)
    assert spin(app, lambda: window.worker is None)
    assert window.start_action.text().endswith("Start")


def test_pause_keeps_processing_edits_on_held_frame(window, app, tmp_path):
    image = tmp_path / "k.png"
    write_png(image, synthetic_frame())
    window.use_image(str(image))
    assert spin(app, lambda: window.last_packet is not None)
    window.pause_action.setChecked(True)
    assert spin(app, lambda: window._state == "paused")
    select(window, "green")
    window.h_row.set_values(100, 110)
    window._apply_editor()
    assert spin(app, lambda: window.rules_tree.topLevelItem(0).text(1) == "0")


def test_missing_window_reports_waiting_state(window, app):
    window.window_combo.setEditText("OKNO_KTOREGO_NA_PEWNO_NIE_MA_12345")
    window._start_worker()
    assert spin(app, lambda: window.banner.text().startswith("Czekam"))
    assert window.state_label.text().endswith("Czekam na okno")
    window.start_action.setChecked(False)
    assert spin(app, lambda: window.worker is None)
    assert window._state == "idle"
    assert window.banner.isHidden()


def test_plugin_receives_frames_and_can_be_disabled(window, app, tmp_path):
    folder = tmp_path / "plugins"
    folder.mkdir()
    marker = tmp_path / "marker.txt"
    (folder / "probe.py").write_text(
        "from pathlib import Path\nfrom colorvision import Plugin\n"
        "class Probe(Plugin):\n    name = 'probe'\n"
        f"    def on_frame(self, result):\n        Path({str(marker)!r}).write_text('x')\n",
        encoding="utf-8",
    )
    window.plugins_dir = folder
    window._refresh_plugins()
    assert window.plugins_list.count() == 1

    image = tmp_path / "k.png"
    write_png(image, synthetic_frame())
    window.plugins_list.item(0).setCheckState(Qt.Unchecked)
    window.use_image(str(image))
    assert spin(app, lambda: window.last_packet is not None)
    assert not marker.exists()
    window.start_action.setChecked(False)
    assert spin(app, lambda: window.worker is None)

    window.plugins_list.item(0).setCheckState(Qt.Checked)
    window.use_image(str(image))
    assert spin(app, lambda: marker.exists())
