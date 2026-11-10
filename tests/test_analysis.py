import csv
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest

from colorvision.debug_overlay import FrameTimings, ascii_text, draw_debug
from colorvision.detector import ColorRule, Detection, color_confidence, detect_rule
from colorvision.events import (
    EVENT_APPEARED, EVENT_DISAPPEARED, EVENT_MOVING, EVENT_STATE_OFF, EVENT_STATE_ON, EVENT_STOPPED, Event,
    event_from_state_change,
)
from colorvision.history import DetectionHistory
from colorvision.plugin import FrameResult, Pipeline, Plugin, PluginHost, process_frame
from colorvision.profile import Profile, ProfileError, load_profile, save_profile
from colorvision.states import Condition, StateChange, StateRule
from colorvision.tracking import MAX_OBJECTS_PER_RULE, ObjectTracker

GREEN = (0, 255, 0)
GREEN_RULE = dict(lower=(50, 200, 150), upper=(70, 255, 255), merge_px=0)


def blob(x, y, rule="r", size=10, score=0.9):
    return Detection(rule, x, y, size, size, size * size, (x + size // 2, y + size // 2), score=score)


def kinds(events):
    return [e.kind for e in events]


def run(tracker, positions, start=0.0, step=0.05, rule="r", distance=60.0, lost=1.0):
    """Feed one object along positions; None means it was not seen in that frame."""
    tracks, events, now = [], [], start
    for position in positions:
        found = [] if position is None else [blob(*position, rule=rule)]
        tracks, new = tracker.update({rule: found}, now, distance, lost)
        events.extend(new)
        now += step
    return tracks, events, now


# ------------------------------------------------------------------- confidence

def test_pixel_in_the_middle_of_the_range_scores_full():
    frame = np.zeros((20, 40, 3), np.uint8)
    frame[5:15, 5:30] = GREEN  # hue 60, exactly the middle of 50-70
    found = detect_rule(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), ColorRule("g", min_area=1, **GREEN_RULE))
    assert found[0].score == pytest.approx(1.0)


def test_edge_of_the_range_scores_lower_but_not_zero():
    rule = ColorRule("g", (60, 0, 0), (80, 255, 255))
    assert color_confidence(np.full(10, 60), 10, 10, rule) == pytest.approx(0.5)
    assert color_confidence(np.full(10, 70), 10, 10, rule) == pytest.approx(1.0)


def test_confidence_handles_red_wrapping_through_zero():
    rule = ColorRule("r", (170, 0, 0), (10, 255, 255))  # middle is hue 0
    assert color_confidence(np.array([0, 0]), 2, 2, rule) == pytest.approx(1.0)
    assert color_confidence(np.array([175, 5]), 2, 2, rule) == pytest.approx(0.75)


def test_gaps_filled_by_merging_lower_confidence():
    frame = np.zeros((20, 60, 3), np.uint8)
    frame[5:15, 5:15] = GREEN
    frame[5:15, 18:28] = GREEN  # 3 px gap closed by merging
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    merged = detect_rule(hsv, ColorRule("g", (50, 200, 150), (70, 255, 255), merge_px=5, min_area=1))
    assert len(merged) == 1
    assert merged[0].score == pytest.approx(200 / merged[0].area)
    assert merged[0].score < 1.0


def test_min_confidence_filters_and_is_validated():
    frame = np.zeros((20, 40, 3), np.uint8)
    frame[5:15, 5:30] = cv2.cvtColor(np.uint8([[[52, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    assert len(detect_rule(hsv, ColorRule("g", min_area=1, min_confidence=0.5, **GREEN_RULE))) == 1
    assert detect_rule(hsv, ColorRule("g", min_area=1, min_confidence=0.9, **GREEN_RULE)) == []
    with pytest.raises(ValueError):
        ColorRule("g", (0, 0, 0), (1, 1, 1), min_confidence=1.5)


def test_new_settings_survive_save_and_load(tmp_path):
    path = tmp_path / "p.json"
    save_profile(
        Profile(rules=[ColorRule("g", (1, 2, 3), (4, 5, 6), min_confidence=0.4)], track_max_distance=33, track_lost_ms=250),
        path,
    )
    loaded = load_profile(path)
    assert loaded.rules[0].min_confidence == 0.4
    assert (loaded.track_max_distance, loaded.track_lost_ms) == (33, 250)
    assert loaded.copy().track_max_distance == 33


def test_invalid_tracking_settings_are_reported(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"track_max_distance": 0}', encoding="utf-8")
    with pytest.raises(ProfileError):
        load_profile(path)


# ------------------------------------------------------------------- tracking

def test_object_keeps_its_id_while_moving():
    tracker = ObjectTracker()
    tracks, _, _ = run(tracker, [(i * 5, 50) for i in range(6)])
    assert [t.id for t in tracks] == [1]
    assert tracks[0].center == (30, 55)
    assert tracks[0].age == pytest.approx(0.25)


def test_single_frame_flicker_never_becomes_an_object():
    tracker = ObjectTracker()
    tracks, events = tracker.update({"r": [blob(10, 10)]}, 0.0, 60, 1.0)
    assert tracks == [] and events == []
    tracks, events, _ = run(tracker, [None, None], start=2.0)
    assert tracks == [] and events == []


def test_two_objects_are_kept_apart_and_nearest_pairs_win():
    tracker = ObjectTracker()
    tracker.update({"r": [blob(0, 0), blob(100, 0)]}, 0.0, 60, 1.0)
    tracks, _ = tracker.update({"r": [blob(105, 0), blob(5, 0)]}, 0.05, 60, 1.0)
    by_id = {t.id: t.x for t in tracks}
    assert by_id == {1: 5, 2: 105}


def test_jump_beyond_max_distance_is_a_new_object():
    tracker = ObjectTracker()
    run(tracker, [(0, 0), (1, 0)])
    tracks, _ = tracker.update({"r": [blob(500, 500)]}, 0.1, 60, 1.0)
    assert [t.id for t in tracks] == [1], "the new one is not confirmed yet"
    tracks, _ = tracker.update({"r": [blob(500, 500)]}, 0.15, 60, 1.0)
    assert [t.id for t in tracks] == [1, 2]


def test_rules_never_share_objects():
    tracker = ObjectTracker()
    tracker.update({"a": [blob(0, 0, "a")], "b": [blob(0, 0, "b")]}, 0.0, 60, 1.0)
    tracks, _ = tracker.update({"a": [blob(0, 0, "a")], "b": [blob(0, 0, "b")]}, 0.05, 60, 1.0)
    assert sorted((t.rule, t.id) for t in tracks) == [("a", 1), ("b", 2)]


def test_brief_gap_keeps_the_object_and_marks_it_unseen():
    tracker = ObjectTracker()
    run(tracker, [(10, 10), (10, 10)])
    tracks, events = tracker.update({"r": []}, 0.5, 60, 1.0)
    assert [(t.id, t.visible) for t in tracks] == [(1, False)]
    assert events == []
    tracks, _ = tracker.update({"r": [blob(10, 10)]}, 0.6, 60, 1.0)
    assert [(t.id, t.visible) for t in tracks] == [(1, True)]


def test_disappearance_is_reported_once_after_the_limit():
    tracker = ObjectTracker()
    run(tracker, [(10, 10), (10, 10)])
    _, early = tracker.update({"r": []}, 0.9, 60, 1.0)
    tracks, late = tracker.update({"r": []}, 1.2, 60, 1.0)
    assert early == []
    assert kinds(late) == [EVENT_DISAPPEARED]
    assert late[0].track_id == 1
    assert tracks == []
    assert tracker.update({"r": []}, 5.0, 60, 1.0)[1] == []


def test_disabled_rule_does_not_report_a_disappearance():
    tracker = ObjectTracker()
    run(tracker, [(10, 10), (10, 10)])
    tracks, events = tracker.update({}, 5.0, 60, 1.0)
    assert tracks == [] and events == []


def test_moving_and_stopping_are_detected_once_each():
    tracker = ObjectTracker()
    moving = [(i * 5, 0) for i in range(20)]  # 100 px/s
    still = [(95, 0)] * 20
    tracks, events, _ = run(tracker, moving + still)
    assert kinds(events) == [EVENT_APPEARED, EVENT_MOVING, EVENT_STOPPED]
    assert not tracks[0].moving


def test_centroid_jitter_does_not_count_as_movement():
    tracker = ObjectTracker()
    tracks, events, _ = run(tracker, [(10 + (i % 2), 10) for i in range(60)])
    assert kinds(events) == [EVENT_APPEARED]
    assert tracks[0].speed < 5


def test_speed_is_measured_in_pixels_per_second():
    tracker = ObjectTracker()
    tracks, _, _ = run(tracker, [(i * 5, 0) for i in range(20)])
    assert tracks[0].speed == pytest.approx(100, abs=2)


def test_trail_is_recorded_and_limited():
    tracker = ObjectTracker()
    tracks, _, _ = run(tracker, [(i, 0) for i in range(200)])
    assert len(tracks[0].trail) == 64
    assert tracks[0].trail[-1] == tracks[0].center


def test_noisy_rule_is_capped():
    tracker = ObjectTracker()
    many = [blob(i * 20, 0) for i in range(MAX_OBJECTS_PER_RULE + 50)]
    tracker.update({"r": many}, 0.0, 5, 1.0)
    tracks, _ = tracker.update({"r": many}, 0.05, 5, 1.0)
    assert len(tracks) == MAX_OBJECTS_PER_RULE


# ------------------------------------------------------------------- events and pipeline

def test_state_changes_become_events():
    on = event_from_state_change(StateChange("S", True, 3.0, 100.0))
    off = event_from_state_change(StateChange("S", False, 65.0, 200.0))
    assert (on.kind, on.message) == (EVENT_STATE_ON, "Stan S: start")
    assert (off.kind, off.message) == (EVENT_STATE_OFF, "Stan S: koniec po 1:05")


def test_pipeline_runs_plugins_states_and_tracking_in_order():
    seen = []

    class Probe(Plugin):
        name = "probe"

        def on_frame(self, result):
            result.detections["model"] = [blob(0, 0, "model")]

        def on_event(self, event):
            seen.append(event.kind)

    profile = Profile(states=[StateRule("S", (Condition("model"),), enter_ms=0)])
    pipeline = Pipeline(profile, PluginHost([Probe()]))
    frame = np.zeros((10, 765, 3), np.uint8)
    first = FrameResult(frame, {}, 0.0)
    changes, events = pipeline.step(first, 0.0)
    assert [c.name for c in changes] == ["S"]
    second = FrameResult(frame, {}, 0.0)
    pipeline.step(second, 0.05)
    assert [t.rule for t in second.tracks] == ["model"], "plugin results are tracked"
    assert seen == [EVENT_STATE_ON, EVENT_APPEARED]


def test_tracking_distance_scales_with_the_window():
    profile = Profile(reference_size=(100, 100), track_max_distance=10)
    pipeline = Pipeline(profile, PluginHost([]))
    frame = np.zeros((10, 400, 3), np.uint8)  # four times wider: 40 px is allowed
    for x, now in ((0, 0.0), (35, 0.05)):
        result = FrameResult(frame, {"r": [blob(x, 0)]}, 0.0)
        pipeline.step(result, now)
    assert [t.id for t in result.tracks] == [1]


def test_crashing_event_plugin_is_disabled(capsys):
    class Crash(Plugin):
        name = "crash"

        def on_event(self, event):
            raise RuntimeError("boom")

    host = PluginHost([Crash()])
    pipeline = Pipeline(Profile(states=[StateRule("S", (Condition("a"),), enter_ms=0)]), host)
    pipeline.step(FrameResult(np.zeros((5, 5, 3), np.uint8), {"a": [blob(0, 0, "a")]}, 0.0), 0.0)
    assert host.plugins == []


# ------------------------------------------------------------------- history

def make_history():
    history = DetectionHistory()
    for index, (count, score) in enumerate([(1, 0.5), (3, 0.9), (2, None)]):
        history.add_sample(100.0 + index, {"r": (count, score)})
    return history


def test_statistics_cover_only_the_window():
    stats = make_history().stats("r", 60, 102.0)
    assert (stats.current, stats.maximum) == (2, 3)
    assert stats.average == pytest.approx(2.0)
    assert stats.mean_score == pytest.approx(0.7)
    recent = make_history().stats("r", 1.5, 102.0)
    assert (recent.maximum, recent.average) == (3, 2.5)


def test_statistics_for_unknown_rule_are_empty():
    stats = DetectionHistory().stats("nope", 60, 0.0)
    assert (stats.current, stats.average, stats.maximum, stats.mean_score, stats.total_objects) == (0, 0.0, 0, None, 0)


def test_series_reports_seconds_ago():
    assert make_history().series("r", 60, 102.0) == [(2.0, 1), (1.0, 3), (0.0, 2)]


def test_total_objects_counts_appearances_only():
    history = DetectionHistory()
    history.add_event(Event(EVENT_APPEARED, "r", "m", 0.0, 1))
    history.add_event(Event(EVENT_APPEARED, "r", "m", 0.0, 2))
    history.add_event(Event(EVENT_DISAPPEARED, "r", "m", 0.0, 1))
    assert history.stats("r", 60, 0.0).total_objects == 2
    history.clear()
    assert history.stats("r", 60, 0.0).total_objects == 0
    assert len(history.events) == 0


def test_csv_exports_open_in_polish_excel_format(tmp_path):
    history = make_history()
    history.add_event(Event(EVENT_APPEARED, "żółty", "żółty #1 pojawił się", 100.0, 1))
    history.add_event(Event(EVENT_STATE_ON, "S", "Stan S: start", 101.0))

    samples = tmp_path / "s.csv"
    assert history.export_samples_csv(samples) == 3
    assert samples.read_bytes().startswith(b"\xef\xbb\xbf")
    with open(samples, encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file, delimiter=";"))
    assert rows[0] == ["czas", "regula", "liczba", "pewnosc_proc"]
    assert rows[1][1:] == ["r", "1", "50"]
    assert rows[3][1:] == ["r", "2", ""]

    events = tmp_path / "e.csv"
    assert history.export_events_csv(events) == 2
    with open(events, encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file, delimiter=";"))
    assert rows[1][1:] == [EVENT_APPEARED, "żółty", "1", "żółty #1 pojawił się"]
    assert rows[2][3] == ""


# ------------------------------------------------------------------- debug overlay

def overlay_result(frame, profile):
    pipeline = Pipeline(profile, PluginHost([]))
    for now in (0.0, 0.05, 0.1):
        result = process_frame(frame, profile)
        pipeline.step(result, now)
    return result


def overlay_scene():
    frame = np.full((503, 765, 3), (40, 62, 74), np.uint8)
    frame[100:105, 200:225] = GREEN
    profile = Profile(
        regions={"gra": (4, 4, 512, 334)},
        rules=[ColorRule("zielona łąka", (50, 200, 150), (70, 255, 255), region="gra")],
        states=[StateRule("Pełny", (Condition("zielona łąka"),), enter_ms=0)],
    )
    return frame, profile


def test_ascii_text_removes_polish_letters():
    assert ascii_text("Żółć łąka Ł") == "Zolc laka L"


def test_overlay_draws_panel_track_label_and_region():
    frame, profile = overlay_scene()
    result = overlay_result(frame, profile)
    assert [t.id for t in result.tracks] == [1]
    canvas = frame.copy()
    draw_debug(canvas, profile, result, 20.0, FrameTimings(1.0, 2.0, 3.0))
    changed = np.any(canvas != frame, axis=2)
    assert changed[450:, :60].any(), "panel in the bottom left corner"
    assert changed[106:118, 200:300].any(), "label under the object"
    assert changed[4, 100:400].all(), "region outline"


def test_labels_can_be_left_to_the_overlay():
    from colorvision.detector import annotate

    frame, profile = overlay_scene()
    result = process_frame(frame, profile)
    with_labels = annotate(frame, profile.items, result.detections)
    without = annotate(frame, profile.items, result.detections, labels=False)
    assert np.any(with_labels[80:100] != frame[80:100]), "label above the object"
    assert np.array_equal(without[80:99], frame[80:99])


def test_overlay_copes_with_a_tiny_frame_and_no_data():
    canvas = np.zeros((20, 30, 3), np.uint8)
    draw_debug(canvas, Profile(), FrameResult(canvas.copy(), {}, 0.0), 0.0, FrameTimings())


# ------------------------------------------------------------------- interface

pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, QPointF, Qt, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from colorvision.capture import write_png  # noqa: E402
from colorvision.gui import main_window as main_window_module  # noqa: E402
from colorvision.gui import theme  # noqa: E402
from colorvision.gui.main_window import MainWindow  # noqa: E402
from colorvision.gui.widgets import HistoryChart  # noqa: E402


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    application.setStyleSheet(theme.STYLESHEET)
    return application


def spin(condition, timeout_ms=6000):
    loop = QEventLoop()
    timer = QTimer()
    timer.timeout.connect(lambda: condition() and loop.quit())
    timer.start(10)
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    timer.stop()
    return condition()


class MovingSource:
    """Frames of a green bar sliding right, after which it vanishes."""

    def __init__(self, _path):
        self.reads = 0

    def read(self):
        self.reads += 1
        frame = np.full((503, 765, 3), (40, 70, 90), np.uint8)
        if self.reads <= 30:
            x = self.reads * 8
            frame[100:105, x:x + 25] = GREEN
        return frame

    def close(self):
        pass


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    monkeypatch.setattr(main_window_module, "ImageSource", MovingSource)
    path = tmp_path / "profile.json"
    save_profile(
        Profile(
            rules=[ColorRule("green", (50, 200, 150), (70, 255, 255))],
            states=[StateRule("Widac", (Condition("green"),), enter_ms=0, exit_ms=0)],
            track_lost_ms=100,
        ),
        path,
    )
    win = MainWindow(settings_path=tmp_path / "s.ini", plugins_dir=tmp_path / "none", profile_path=path)
    win.warnings = []
    win._warn = lambda title, text: win.warnings.append(title)
    win.show()
    app.processEvents()
    yield win
    if win.worker is not None:
        win.worker.requestInterruption()
        win.worker.wait(3000)
    win.dirty = False
    win.close()


def event_kinds(window):
    table = window.events_table
    return [table.item(r, 1).data(Qt.UserRole) for r in range(table.rowCount())]


def test_moving_object_produces_events_table_and_history(window, app):
    window.tabs.setCurrentWidget(window.objects_tab)
    window.fps_spin.setValue(30)
    window.use_image("nieuzywane.png")

    table = window.objects_table
    assert spin(lambda: table.rowCount() == 1)
    assert table.item(0, 0).text() == "#1"
    assert table.item(0, 1).text() == "green"
    assert spin(lambda: table.rowCount() == 1 and table.item(0, 7).text() == "W ruchu")

    assert spin(lambda: EVENT_DISAPPEARED in event_kinds(window))
    kinds_seen = set(event_kinds(window))
    assert {EVENT_APPEARED, EVENT_MOVING, EVENT_STATE_ON, EVENT_STATE_OFF} <= kinds_seen
    assert len(window.history.events) == window.events_table.rowCount()

    window.tabs.setCurrentWidget(window.history_tab)
    window._refresh_history()
    history = window.history_table
    assert history.rowCount() == 1
    assert history.item(0, 0).text() == "green"
    assert history.item(0, 5).text() == "1"
    assert window._history_rule == "green"
    assert window.history_chart._points


def test_event_filter_hides_and_shows_groups(window):
    window._on_event(Event(EVENT_MOVING, "r", "ruch", 0.0, 1))
    window._on_event(Event(EVENT_STATE_ON, "S", "stan", 0.0))
    table = window.events_table
    row_of = {table.item(r, 1).data(Qt.UserRole): r for r in range(2)}
    window.event_filters["Ruch"].setChecked(False)
    assert table.isRowHidden(row_of[EVENT_MOVING])
    assert not table.isRowHidden(row_of[EVENT_STATE_ON])
    window._on_event(Event(EVENT_STOPPED, "r", "stop", 0.0, 1))
    assert table.isRowHidden(0), "new events respect the filter"
    window.event_filters["Ruch"].setChecked(True)
    assert not any(table.isRowHidden(r) for r in range(table.rowCount()))


def test_event_table_is_limited_but_history_keeps_more(window):
    for index in range(main_window_module.MAX_EVENT_ROWS + 20):
        window._on_event(Event(EVENT_MOVING, "r", str(index), 0.0, 1))
    assert window.events_table.rowCount() == main_window_module.MAX_EVENT_ROWS
    assert len(window.history.events) == main_window_module.MAX_EVENT_ROWS + 20
    assert window.events_table.item(0, 3).text() == str(main_window_module.MAX_EVENT_ROWS + 19)


def test_export_and_clear_from_the_interface(window, tmp_path):
    window._on_event(Event(EVENT_MOVING, "r", "ruch", 100.0, 1))
    window.history.add_sample(100.0, {"r": (2, 0.5)})
    target = tmp_path / "wynik.csv"
    window._ask_save_path = lambda title, name: str(target)
    window._export_events()
    assert "ruch" in target.read_text(encoding="utf-8-sig")
    window._export_samples()
    assert ";r;2;50" in target.read_text(encoding="utf-8-sig")

    window._ask_save_path = lambda title, name: ""
    target.unlink()
    window._export_events()
    assert not target.exists(), "cancelled dialog writes nothing"

    window._ask_save_path = lambda title, name: str(tmp_path / "nie_ma" / "x.csv")
    window._export_events()
    assert window.warnings == ["Nie można zapisać pliku"]

    window._clear_history()
    assert window.events_table.rowCount() == 0
    assert window.history.rules() == []
    assert window.history_table.rowCount() == 0


def test_history_refresh_skipped_unless_the_tab_is_open(window):
    window.history.add_sample(100.0, {"r": (1, None)})
    window.tabs.setCurrentWidget(window.detections_table)
    window._refresh_history()
    assert window.history_table.rowCount() == 0
    window.tabs.setCurrentWidget(window.history_tab)
    window._refresh_history()
    assert window.history_table.rowCount() == 1


def test_debug_overlay_changes_the_preview(window, app):
    window.use_image("nieuzywane.png")
    assert spin(lambda: window.last_packet is not None)

    def lit_panel():
        image = window.last_packet.image.convertToFormat(window.last_packet.image.Format.Format_RGB888)
        return sum(image.pixelColor(x, 495).lightness() > 150 for x in range(4, 150))

    window.debug_action.setChecked(True)
    assert spin(lambda: lit_panel() > 0)
    window.debug_action.setChecked(False)
    assert spin(lambda: lit_panel() == 0)


def test_min_confidence_and_tracking_settings_are_editable(window, tmp_path):
    window.rules_tree.topLevelItem(0).setSelected(True)
    window.min_confidence_spin.setValue(40)
    assert window.profile.rules[0].min_confidence == 0.4
    window.track_distance_spin.setValue(123)
    window.track_lost_spin.setValue(450)
    assert window._save()
    loaded = load_profile(window.profile_path)
    assert (loaded.rules[0].min_confidence, loaded.track_max_distance, loaded.track_lost_ms) == (0.4, 123, 450)


def test_loading_a_profile_does_not_mark_it_changed(window):
    assert window.track_lost_spin.value() == 100
    assert not window.dirty


def test_chart_finds_nearest_sample_and_draws_without_data(app):
    chart = HistoryChart(60)
    chart.resize(300, 150)
    assert chart.point_at(100) is None
    chart.grab()
    chart.set_series("r", [(60.0, 1), (30.0, 5), (0.0, 2)])
    plot = chart._plot()
    assert chart.point_at(plot.right()) == (0.0, 2)
    assert chart.point_at(plot.left()) == (60.0, 1)
    chart._hover_x = plot.center().x()
    assert not chart.grab().isNull()
