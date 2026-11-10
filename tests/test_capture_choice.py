import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from colorvision.capture import ImageSource, WindowInfo, covers_a_monitor, intersect, list_windows, write_png


def test_intersect():
    assert intersect((0, 0, 10, 10), (5, 5, 10, 10)) == (5, 5, 5, 5)
    assert intersect((0, 0, 10, 10), (20, 20, 5, 5)) is None


@pytest.mark.parametrize(
    "rect, expected",
    [
        ((0, 0, 1920, 1080), True),  # fullscreen
        ((0, 0, 1920, 1040), True),  # maximised above the taskbar
        ((100, 100, 765, 503), False),  # a normal game window
        ((1920, 0, 1920, 1080), True),  # fullscreen on the second monitor
        ((1000, 0, 1900, 1080), False),  # large, but split between monitors
    ],
)
def test_covers_a_monitor(rect, expected):
    monitors = [(0, 0, 1920, 1080), (1920, 0, 1920, 1080)]
    assert covers_a_monitor(rect, monitors) is expected


def test_image_source_crops(tmp_path):
    image = np.zeros((100, 200, 3), np.uint8)
    image[10:20, 30:50] = (0, 255, 0)
    path = tmp_path / "k.png"
    write_png(path, image)
    source = ImageSource(path)
    assert source.read().shape == (100, 200, 3)
    source.set_crop((30, 10, 20, 10))
    part = source.read()
    assert part.shape == (10, 20, 3)
    assert (part == (0, 255, 0)).all()
    source.set_crop((500, 500, 20, 10))
    assert source.read().shape == (100, 200, 3), "a crop outside the picture is ignored"
    source.set_crop(None)
    assert source.read().shape == (100, 200, 3)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_desktop_is_not_offered_as_a_window():
    titles = {window.title for window in list_windows()}
    has_desktop = "Program Manager" in titles
    del titles  # never let a failure print the titles of the user's windows
    assert not has_desktop


# ------------------------------------------------------------------- interface

pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from colorvision.detector import ColorRule  # noqa: E402
from colorvision.gui import main_window as main_window_module  # noqa: E402
from colorvision.gui import theme  # noqa: E402
from colorvision.gui.main_window import MainWindow  # noqa: E402
from colorvision.profile import Profile, save_profile  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    application.setStyleSheet(theme.STYLESHEET)
    return application


def spin(condition, timeout_ms=5000):
    loop = QEventLoop()
    timer = QTimer()
    timer.timeout.connect(lambda: condition() and loop.quit())
    timer.start(10)
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    timer.stop()
    return condition()


def make_window(tmp_path, name="s.ini"):
    path = tmp_path / "profile.json"
    save_profile(Profile(rules=[ColorRule("green", (50, 200, 150), (70, 255, 255))]), path)
    win = MainWindow(settings_path=tmp_path / name, plugins_dir=tmp_path / "none", profile_path=path)
    win.warnings = []
    win._warn = lambda title, text: win.warnings.append(title)
    win._confirm = lambda title, text: True
    return win


@pytest.fixture
def window(app, tmp_path):
    win = make_window(tmp_path)
    win.show()
    app.processEvents()
    yield win
    if win.worker is not None:
        win.worker.requestInterruption()
        win.worker.wait(3000)
    win.dirty = False
    win.close()


class FakeWindowSource:
    """Stands in for a captured window of the given size, recording what it was asked for."""

    created = []

    def __init__(self, title, hwnd=None):
        self.title, self.hwnd = title, hwnd
        self.crop = None
        FakeWindowSource.created.append(self)

    def set_crop(self, crop):
        self.crop = crop

    def read(self):
        frame = np.full((1080, 1920, 3), (40, 70, 90), np.uint8)
        frame[500:510, 900:940] = (0, 255, 0)
        if self.crop:
            x, y, w, h = self.crop
            return frame[y:y + h, x:x + w]
        return frame

    def is_covered(self):
        return False

    def describe(self):
        frame = self.read()
        return {
            "title": "Okno testowe", "width": frame.shape[1], "height": frame.shape[0],
            "whole_screen": self.crop is None,
        }

    def close(self):
        pass


@pytest.fixture
def fake_source(monkeypatch):
    FakeWindowSource.created = []
    monkeypatch.setattr(main_window_module, "WindowSource", FakeWindowSource)
    return FakeWindowSource


def start(window):
    window.window_combo.setEditText("Okno testowe")
    window._start_worker()
    assert spin(lambda: window.last_packet is not None)


def test_whole_screen_capture_is_explained_to_the_user(window, fake_source):
    start(window)
    assert spin(lambda: "prawie cały ekran" in window.banner.text())
    assert "Okno testowe (1920×1080)" in window.source_label.text()


def test_selecting_an_area_crops_the_capture_and_removes_the_warning(window, fake_source):
    start(window)
    assert spin(lambda: "prawie cały ekran" in window.banner.text())
    window.crop_action.setChecked(True)
    assert window.pause_action.isChecked(), "the picture holds still for selecting"
    window._on_area_selected(800, 400, 765, 503)
    assert window.crop == (800, 400, 765, 503)
    assert not window.pause_action.isChecked()
    assert window.crop_action.text() == "Cały obraz"
    assert spin(lambda: window.last_packet.frame.shape[:2] == (503, 765))
    assert spin(lambda: "prawie cały ekran" not in window.banner.text())
    assert spin(lambda: window.rules_tree.topLevelItem(0).text(1) == "1")
    assert spin(lambda: window.last_packet.detections["green"][0].x == 100), "coordinates start at the crop"
    assert spin(lambda: "(765×503)" in window.source_label.text())


def test_whole_image_button_cancels_the_crop(window, fake_source):
    start(window)
    window.crop_action.setChecked(True)
    window._on_area_selected(800, 400, 765, 503)
    assert spin(lambda: window.last_packet.frame.shape[0] == 503)
    window.crop_action.trigger()
    assert window.crop is None
    assert window.crop_action.text() == "Obszar"
    assert spin(lambda: window.last_packet.frame.shape[0] == 1080)


def test_too_small_area_is_refused(window, fake_source):
    start(window)
    window.crop_action.setChecked(True)
    window._on_area_selected(10, 10, 20, 20)
    assert window.warnings == ["Za mały obszar"]
    assert window.crop is None
    assert window._selecting_crop, "the user can try again"
    window.crop_action.setChecked(False)
    assert not window.pause_action.isChecked()


def test_crop_selection_does_not_create_a_template(window, fake_source):
    start(window)
    window.crop_action.setChecked(True)
    window._on_area_selected(800, 400, 765, 503)
    assert window.profile.templates == []


def test_template_and_crop_selection_exclude_each_other(window, fake_source):
    start(window)
    window.crop_action.setChecked(True)
    window.template_action.setChecked(True)
    assert not window._selecting_crop
    window._on_area_selected(895, 495, 50, 20)
    assert [t.name for t in window.profile.templates] == ["wzorzec"]
    assert window.crop is None


def test_crop_is_remembered_for_the_next_session(app, tmp_path, fake_source):
    first = make_window(tmp_path, "pamiec.ini")
    first.crop = (1, 2, 300, 400)
    first._store_crop()
    first.settings.sync()
    first.dirty = False
    first.close()
    second = make_window(tmp_path, "pamiec.ini")
    assert second.crop == (1, 2, 300, 400)
    second.dirty = False
    second.close()


def test_invalid_stored_crop_is_ignored(window):
    window.settings.setValue("crop", "a,b,c,d")
    assert window._stored_crop() is None
    window.settings.setValue("crop", "1,2")
    assert window._stored_crop() is None


def test_typing_another_title_forgets_the_window_and_crop(window):
    window.window_hwnd = 123
    window.crop = (0, 0, 100, 100)
    window.window_combo.setEditText("Inne okno")
    assert window.window_hwnd is None
    assert window.crop is None


def test_window_chosen_from_the_list_is_used_exactly(window, fake_source):
    window.window_combo.blockSignals(True)
    window.window_combo.addItem("Gra", 777)
    window.window_combo.setCurrentIndex(window.window_combo.findData(777))
    window.window_combo.blockSignals(False)
    window._on_window_chosen()
    window._start_worker()
    assert spin(lambda: fake_source.created and window.last_packet is not None)
    assert fake_source.created[-1].hwnd == 777


def test_pick_window_with_countdown(window, fake_source, monkeypatch):
    monkeypatch.setattr(main_window_module, "foreground_window", lambda: WindowInfo(555, "Wskazana gra"))
    window._start_window_pick()
    assert not window.pick_window_action.isEnabled()
    assert "Kliknij" in window.statusBar().currentMessage()
    for _ in range(main_window_module.PICK_WINDOW_SECONDS):
        window._on_pick_tick()
    assert window.pick_window_action.isEnabled()
    assert window.window_combo.currentText() == "Wskazana gra"
    assert window.window_hwnd == 555
    window._start_worker()
    assert spin(lambda: fake_source.created and window.last_packet is not None)
    assert fake_source.created[-1].hwnd == 555


def test_pick_window_refuses_own_window_or_desktop(window, monkeypatch):
    monkeypatch.setattr(main_window_module, "foreground_window", lambda: None)
    window._start_window_pick()
    for _ in range(main_window_module.PICK_WINDOW_SECONDS):
        window._on_pick_tick()
    assert window.warnings == ["Nie wskazano okna"]
    assert window.window_hwnd is None


def test_pick_while_running_restarts_on_the_new_window(window, fake_source):
    start(window)
    window._finish_window_pick(WindowInfo(888, "Nowa gra"))
    assert spin(lambda: window.worker is not None and fake_source.created[-1].hwnd == 888)
    assert window.window_combo.currentText() == "Nowa gra"


def test_old_worker_ending_does_not_touch_the_new_one(window, fake_source, app):
    """Regression: the old worker's late finished signal destroyed the new, running thread."""
    start(window)
    first = window.worker
    window._finish_window_pick(WindowInfo(888, "Nowa gra"))
    second = window.worker
    assert second is not None and second is not first
    spin(lambda: False, timeout_ms=200)  # let every queued signal arrive
    assert window.worker is second
    assert second.isRunning()


def test_opening_an_image_while_running_keeps_the_new_worker(window, tmp_path):
    frame = np.zeros((50, 50, 3), np.uint8)
    path = tmp_path / "k.png"
    write_png(path, frame)
    window.use_image(str(path))
    assert spin(lambda: window.last_packet is not None)
    first = window.worker
    window.use_image(str(path))
    second = window.worker
    assert second is not first
    spin(lambda: False, timeout_ms=200)
    assert window.worker is second
    assert second.isRunning()


def test_stuck_worker_is_kept_instead_of_deleted(window, fake_source, monkeypatch):
    start(window)
    worker = window.worker
    monkeypatch.setattr(worker, "wait", lambda _ms=None: False)
    window._stop_worker_and_wait()
    assert window.worker is None
    assert worker in window._abandoned_workers
    monkeypatch.undo()
    worker.wait(3000)


def test_stop_during_selection_ends_it(window, fake_source):
    start(window)
    window.crop_action.setChecked(True)
    window.start_action.setChecked(False)
    assert spin(lambda: window.worker is None)
    assert not window._selecting_crop
    assert not window.view.selecting
