import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest

from colorvision.detector import (
    SHAPE_BAR, SHAPE_CIRCLE, SHAPE_DOT, SHAPE_OTHER, SHAPE_RECTANGLE, SHAPE_TEMPLATE, ColorRule, TemplateRule,
    classify_shape, detect_rule, match_template, template_is_usable,
)
from colorvision.plugin import process_frame
from colorvision.profile import Profile, ProfileError, load_profile, save_profile

GREEN = (0, 255, 0)
ANY_GREEN = {"lower": (55, 200, 200), "upper": (65, 255, 255), "merge_px": 0}


def icon(seed: int = 1, size: int = 16) -> np.ndarray:
    """A small textured picture, unlike the plain background used in the tests."""
    return np.random.default_rng(seed).integers(0, 256, (size, size, 3), np.uint8)


def scene(*positions, size=(503, 765), picture=None) -> np.ndarray:
    frame = np.full((*size, 3), (40, 62, 74), np.uint8)
    picture = icon() if picture is None else picture
    for x, y in positions:
        frame[y:y + picture.shape[0], x:x + picture.shape[1]] = picture
    return frame


@pytest.mark.parametrize(
    "width, height, area, expected",
    [
        (30, 5, 150, SHAPE_BAR),
        (5, 30, 150, SHAPE_BAR),
        (40, 40, 1600, SHAPE_RECTANGLE),
        (40, 40, 1257, SHAPE_CIRCLE),
        (3, 3, 9, SHAPE_DOT),
        (40, 40, 300, SHAPE_OTHER),
    ],
)
def test_classify_shape(width, height, area, expected):
    assert classify_shape(width, height, area) == expected


def test_real_drawn_shapes_are_recognised():
    frame = np.zeros((200, 400, 3), np.uint8)
    frame[20:25, 20:60] = GREEN
    frame[60:110, 20:70] = GREEN
    cv2.circle(frame, (150, 80), 30, GREEN, -1)
    frame[20:23, 300:303] = GREEN
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    shapes = {d.shape for d in detect_rule(hsv, ColorRule("g", min_area=1, **ANY_GREEN))}
    assert shapes == {SHAPE_BAR, SHAPE_RECTANGLE, SHAPE_CIRCLE, SHAPE_DOT}


def test_shape_filter_keeps_only_that_shape():
    frame = np.zeros((200, 400, 3), np.uint8)
    frame[20:25, 20:60] = GREEN  # a bar
    frame[60:110, 20:70] = GREEN  # a same-coloured square, which the bar rule must reject
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    found = detect_rule(hsv, ColorRule("g", shape=SHAPE_BAR, **ANY_GREEN))
    assert [(d.x, d.y, d.shape) for d in found] == [(20, 20, SHAPE_BAR)]


def test_unknown_shape_is_rejected():
    with pytest.raises(ValueError):
        ColorRule("g", (0, 0, 0), (1, 1, 1), shape="trojkat")


def test_template_finds_every_copy_exactly_once():
    template = TemplateRule("item", icon())
    found = match_template(scene((100, 50), (300, 200), (500, 400)), template)
    assert sorted((d.x, d.y) for d in found) == [(100, 50), (300, 200), (500, 400)]
    assert all(d.score > 0.99 and d.shape == SHAPE_TEMPLATE for d in found)
    assert found[0].center == (found[0].x + 8, found[0].y + 8)


def test_template_ignores_a_different_picture():
    frame = scene((100, 50), picture=icon(seed=2))
    assert match_template(frame, TemplateRule("item", icon(seed=1))) == []


def test_template_respects_region_and_result_limit():
    frame = scene((100, 50), (300, 200))
    inside = match_template(frame, TemplateRule("item", icon()), (250, 150, 200, 200))
    assert [(d.x, d.y) for d in inside] == [(300, 200)]
    assert len(match_template(frame, TemplateRule("item", icon(), max_results=1))) == 1


def test_template_threshold_separates_close_and_exact_copies():
    noisy = icon().astype(int) + np.random.default_rng(9).integers(-60, 61, (16, 16, 3))
    frame = scene((100, 50), picture=np.clip(noisy, 0, 255).astype(np.uint8))
    assert len(match_template(frame, TemplateRule("item", icon(), threshold=0.5))) == 1
    assert match_template(frame, TemplateRule("item", icon(), threshold=0.99)) == []


def test_template_scales_with_the_window():
    picture = cv2.resize(icon(size=8), (32, 32), interpolation=cv2.INTER_NEAREST)
    frame = scene((100, 50), picture=picture)
    big = cv2.resize(frame, (765 * 2, 503 * 2), interpolation=cv2.INTER_AREA)
    found = match_template(big, TemplateRule("item", picture, captured_size=(765, 503)))
    assert [(d.x, d.y, d.width, d.height) for d in found] == [(200, 100, 64, 64)]


def test_template_larger_than_search_area_finds_nothing():
    assert match_template(scene(), TemplateRule("item", icon()), (0, 0, 5, 5)) == []


def test_flat_pictures_are_not_usable():
    flat = np.full((20, 20, 3), 90, np.uint8)
    assert not template_is_usable(flat)
    assert not template_is_usable(np.full((20, 20, 3), (40, 62, 74), np.uint8)), "flat colour, channels differ"
    assert not template_is_usable(icon(size=3))
    assert template_is_usable(icon())
    assert match_template(scene(), TemplateRule("flat", flat)) == []


def test_process_frame_skips_disabled_templates():
    frame = scene((100, 50))
    profile = Profile(templates=[TemplateRule("on", icon()), TemplateRule("off", icon(), enabled=False)])
    result = process_frame(frame, profile)
    assert len(result.detections["on"]) == 1
    assert "off" not in result.detections


def test_templates_survive_a_save_and_load(tmp_path):
    path = tmp_path / "p.json"
    picture = icon()
    save_profile(
        Profile(
            regions={"inv": (0, 0, 10, 10)},
            rules=[ColorRule("g", (1, 2, 3), (4, 5, 6), shape=SHAPE_BAR)],
            templates=[TemplateRule("item", picture, threshold=0.75, region="inv", max_results=3, captured_size=(800, 600))],
        ),
        path,
    )
    loaded = load_profile(path)
    assert loaded.rules[0].shape == SHAPE_BAR
    template = loaded.templates[0]
    assert np.array_equal(template.image, picture)
    assert (template.threshold, template.region, template.max_results, template.captured_size) == (0.75, "inv", 3, (800, 600))


def test_profile_without_new_fields_still_loads(tmp_path):
    path = tmp_path / "old.json"
    path.write_text('{"rules": [{"name": "x", "lower": [0, 0, 0], "upper": [1, 1, 1]}]}', encoding="utf-8")
    loaded = load_profile(path)
    assert loaded.templates == []
    assert loaded.rules[0].shape == "any"


@pytest.mark.parametrize(
    "extra",
    [
        '"templates": [{"name": "t", "png_base64": "to nie jest obraz"}]',
        '"templates": [{"name": "t", "png_base64": "aGVsbG8="}]',
        '"templates": [{"name": "t"}]',
    ],
)
def test_damaged_templates_are_reported(tmp_path, extra):
    path = tmp_path / "bad.json"
    path.write_text("{" + extra + "}", encoding="utf-8")
    with pytest.raises(ProfileError):
        load_profile(path)


def test_rule_and_template_cannot_share_a_name(tmp_path):
    path = tmp_path / "p.json"
    save_profile(Profile(rules=[ColorRule("x", (1, 2, 3), (4, 5, 6))], templates=[TemplateRule("x", icon())]), path)
    with pytest.raises(ProfileError):
        load_profile(path)


# ------------------------------------------------------------------- interface

PySide6 = pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, QPoint, Qt, QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from colorvision.capture import write_png  # noqa: E402
from colorvision.gui import theme  # noqa: E402
from colorvision.gui.main_window import MainWindow  # noqa: E402


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


@pytest.fixture
def window(app, tmp_path):
    path = tmp_path / "profile.json"
    save_profile(Profile((765, 503), {"inv": (0, 0, 400, 400)}, [ColorRule("green", (55, 200, 150), (65, 255, 255))]), path)
    win = MainWindow(settings_path=tmp_path / "s.ini", plugins_dir=tmp_path / "none", profile_path=path)
    win.warnings = []
    win._warn = lambda title, text: win.warnings.append(title)
    win._confirm = lambda title, text: True
    win.resize(1320, 840)
    win.show()
    app.processEvents()
    yield win
    if win.worker is not None:
        win.worker.requestInterruption()
        win.worker.wait(3000)
    win.dirty = False
    win.close()


def start_on(window, tmp_path, frame):
    image = tmp_path / "klatka.png"
    write_png(image, frame)
    window.use_image(str(image))
    assert spin(lambda: window.last_packet is not None)


def select(window, name):
    for i in range(window.rules_tree.topLevelItemCount()):
        item = window.rules_tree.topLevelItem(i)
        if item.data(0, Qt.UserRole) == name:
            window.rules_tree.clearSelection()
            item.setSelected(True)
            return
    raise AssertionError(name)


def row_of(window, name):
    return next(
        window.rules_tree.topLevelItem(i)
        for i in range(window.rules_tree.topLevelItemCount())
        if window.rules_tree.topLevelItem(i).data(0, Qt.UserRole) == name
    )


def test_cutting_a_template_with_the_mouse(window, app, tmp_path):
    start_on(window, tmp_path, scene((100, 50), (300, 200)))
    window.template_action.setChecked(True)
    assert window.pause_action.isChecked(), "cutting freezes the picture"

    rect = window.view.target_rect()
    scale = rect.width() / 765
    to_widget = lambda x, y: QPoint(rect.left() + round(x * scale), rect.top() + round(y * scale))
    QTest.mousePress(window.view, Qt.LeftButton, pos=to_widget(100, 50))
    QTest.mouseMove(window.view, to_widget(116, 66))
    QTest.mouseRelease(window.view, Qt.LeftButton, pos=to_widget(115, 65))
    app.processEvents()

    assert [t.name for t in window.profile.templates] == ["wzorzec"]
    assert not window.template_action.isChecked()
    assert not window.pause_action.isChecked(), "the automatic pause is undone"
    assert window._current_item().name == "wzorzec"
    assert window.template_section.isVisibleTo(window.editor)
    assert not window.color_section.isVisibleTo(window.editor)
    assert window.dirty
    assert spin(lambda: row_of(window, "wzorzec").text(1) == "2")


def test_flat_area_is_refused_with_a_message(window, tmp_path):
    start_on(window, tmp_path, scene())
    window.template_action.setChecked(True)
    window._on_area_selected(0, 0, 30, 30)
    assert window.warnings == ["Nie można użyć tego fragmentu"]
    assert window.profile.templates == []
    assert window.template_action.isChecked(), "the user can try again"


def test_pause_set_by_the_user_is_kept_after_cutting(window, tmp_path):
    start_on(window, tmp_path, scene((100, 50)))
    window.pause_action.setChecked(True)
    window.template_action.setChecked(True)
    window._on_area_selected(100, 50, 16, 16)
    assert window.pause_action.isChecked()


def test_template_editor_changes_settings_and_saves(window, tmp_path):
    start_on(window, tmp_path, scene((100, 50)))
    window.template_action.setChecked(True)
    window._on_area_selected(100, 50, 16, 16)
    window.threshold_slider.setValue(70)
    window.max_results_spin.setValue(4)
    window.region_combo.setCurrentIndex(window.region_combo.findData("inv"))
    template = window.profile.templates[0]
    assert (template.threshold, template.max_results, template.region) == (0.7, 4, "inv")
    assert window._save()
    assert load_profile(window.profile_path).templates[0].threshold == 0.7


def test_recut_replaces_the_picture_and_keeps_settings(window, tmp_path):
    start_on(window, tmp_path, scene((100, 50)))
    window.template_action.setChecked(True)
    window._on_area_selected(100, 50, 16, 16)
    window.threshold_slider.setValue(70)
    window._recut_template()
    window._on_area_selected(100, 50, 8, 8)
    template = window.profile.templates[0]
    assert template.image.shape[:2] == (8, 8)
    assert template.threshold == 0.7
    assert len(window.profile.templates) == 1
    assert window._recut_target is None


def test_template_rename_duplicate_delete_and_name_clash(window, tmp_path):
    start_on(window, tmp_path, scene((100, 50)))
    window.template_action.setChecked(True)
    window._on_area_selected(100, 50, 16, 16)
    window.name_edit.setText("green")
    window._on_name_edited()
    assert window.warnings == ["Nieprawidłowa nazwa"]
    window.name_edit.setText("miecz")
    window._on_name_edited()
    assert window.profile.templates[0].name == "miecz"
    window._duplicate_rule()
    assert [t.name for t in window.profile.templates] == ["miecz", "miecz_kopia"]
    window._delete_rule()
    assert [t.name for t in window.profile.templates] == ["miecz"]
    assert [r.name for r in window.profile.rules] == ["green"]


def test_mask_is_not_offered_for_templates(window, tmp_path):
    start_on(window, tmp_path, scene((100, 50)))
    select(window, "green")
    window.mask_action.setChecked(True)
    window.template_action.setChecked(True)
    window._on_area_selected(100, 50, 16, 16)
    assert not window.mask_action.isChecked()
    assert not window.mask_action.isEnabled()
    select(window, "green")
    assert window.mask_action.isEnabled()


def test_region_used_by_template_is_protected_and_renamed(window, tmp_path):
    start_on(window, tmp_path, scene((100, 50)))
    window.template_action.setChecked(True)
    window._on_area_selected(100, 50, 16, 16)
    window.region_combo.setCurrentIndex(window.region_combo.findData("inv"))
    window.regions_table.setCurrentCell(0, 0)
    window._delete_region()
    assert window.warnings == ["Region jest używany"]
    window.regions_table.item(0, 0).setText("plecak")
    assert window.profile.templates[0].region == "plecak"


def test_shape_choice_in_editor_filters_live(window, tmp_path, app):
    frame = np.zeros((503, 765, 3), np.uint8)
    frame[20:25, 20:60] = GREEN
    frame[60:110, 20:70] = GREEN
    start_on(window, tmp_path, frame)
    select(window, "green")
    assert spin(lambda: row_of(window, "green").text(1) == "2")
    window.shape_combo.setCurrentIndex(window.shape_combo.findData(SHAPE_BAR))
    assert window.profile.rules[0].shape == SHAPE_BAR
    assert spin(lambda: row_of(window, "green").text(1) == "1")


def test_table_shows_shape_and_similarity(window, tmp_path, app):
    start_on(window, tmp_path, scene((100, 50)))
    window.template_action.setChecked(True)
    window._on_area_selected(100, 50, 16, 16)
    window.tabs.setCurrentIndex(0)

    def row_ready():
        table = window.detections_table
        return any(table.item(r, 0) and table.item(r, 0).text() == "wzorzec" for r in range(table.rowCount()))

    assert spin(row_ready)
    table = window.detections_table
    row = next(r for r in range(table.rowCount()) if table.item(r, 0).text() == "wzorzec")
    assert table.item(row, 1).text() == "Wzorzec"
    assert table.item(row, 8).text() == "100%"


def test_plugin_can_add_its_own_detections(window, app, tmp_path):
    folder = tmp_path / "plugins"
    folder.mkdir()
    (folder / "model.py").write_text(
        "from colorvision import Detection, Plugin\n"
        "class Model(Plugin):\n    name = 'model'\n"
        "    def on_frame(self, result):\n"
        "        result.detections['npc'] = [Detection('npc', 1, 2, 3, 4, 12, (2, 4))]\n",
        encoding="utf-8",
    )
    window.plugins_dir = folder
    window._refresh_plugins()
    window.tabs.setCurrentIndex(0)
    start_on(window, tmp_path, scene())
    table = window.detections_table
    assert spin(lambda: any(table.item(r, 0) and table.item(r, 0).text() == "npc" for r in range(table.rowCount())))
