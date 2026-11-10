from pathlib import Path

import cv2
import numpy as np
import pytest

from colorvision.__main__ import main
from colorvision.calibrate import range_from_sample, upsert_rule
from colorvision.capture import write_png
from colorvision.detector import ColorRule, detect_rule
from colorvision.plugin import Plugin, PluginHost, load_plugins, process_frame
from colorvision.profile import Profile, ProfileError, load_profile, save_profile

DEFAULT_PROFILE = Path(__file__).resolve().parent.parent / "profiles" / "default.json"
GREEN, RED, YELLOW = (0, 255, 0), (0, 0, 255), (0, 255, 255)


def synthetic_frame() -> np.ndarray:
    """A fake 765x503 client: HP bar in the game view, two yellow minimap dots."""
    frame = np.full((503, 765, 3), (40, 70, 90), np.uint8)
    frame[100:105, 200:225] = GREEN
    frame[100:105, 225:235] = RED
    frame[50:53, 600:603] = YELLOW
    frame[90:93, 650:653] = YELLOW
    frame[450:455, 20:60] = GREEN  # outside game_view: must be ignored
    return frame


def to_hsv(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


def test_default_profile_finds_synthetic_objects():
    result = process_frame(synthetic_frame(), load_profile(DEFAULT_PROFILE))
    green = result.detections["hp_bar_green"]
    assert len(green) == 1
    assert (green[0].x, green[0].y, green[0].width, green[0].height) == (200, 100, 25, 5)
    assert len(result.detections["hp_bar_red"]) == 1
    assert len(result.detections["minimap_yellow"]) == 2


def test_red_matches_on_both_sides_of_hue_zero():
    frame = np.zeros((20, 40, 3), np.uint8)
    frame[5:15, 2:12] = cv2.cvtColor(np.uint8([[[178, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    frame[5:15, 25:35] = cv2.cvtColor(np.uint8([[[2, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    rule = ColorRule("red", (175, 200, 200), (5, 255, 255), merge_px=0)
    assert len(detect_rule(to_hsv(frame), rule)) == 2


def test_area_limits_and_sorting():
    frame = np.zeros((60, 60, 3), np.uint8)
    frame[0:2, 0:2] = GREEN  # area 4
    frame[10:15, 10:15] = GREEN  # area 25
    frame[30:50, 30:50] = GREEN  # area 400
    rule = ColorRule("g", (55, 200, 200), (65, 255, 255), min_area=5, max_area=500, merge_px=0)
    assert [d.area for d in detect_rule(to_hsv(frame), rule)] == [400, 25]


def test_merge_joins_a_split_object():
    frame = np.zeros((20, 40, 3), np.uint8)
    frame[5:10, 5:15] = GREEN
    frame[5:10, 17:27] = GREEN  # 2 px gap
    separate = ColorRule("g", (55, 200, 200), (65, 255, 255), merge_px=0)
    merged = ColorRule("g", (55, 200, 200), (65, 255, 255), merge_px=3)
    assert len(detect_rule(to_hsv(frame), separate)) == 2
    assert len(detect_rule(to_hsv(frame), merged)) == 1


def test_regions_scale_with_window_and_stay_clipped():
    profile = Profile(reference_size=(100, 100), regions={"a": (10, 10, 20, 20), "edge": (90, 90, 50, 50)})
    assert profile.region_rect("a", 200, 300) == (20, 30, 40, 60)
    assert profile.region_rect("edge", 100, 100) == (90, 90, 10, 10)
    assert profile.region_rect(None, 50, 60) == (0, 0, 50, 60)


def test_sampling_red_produces_a_wrapping_range():
    frame = np.zeros((10, 10, 3), np.uint8)
    frame[:] = RED
    lower, upper = range_from_sample(to_hsv(frame), 5, 5)
    assert lower[0] > upper[0]  # wraps through 0
    rule = ColorRule("red", lower, upper, merge_px=0)
    assert detect_rule(to_hsv(frame), rule)[0].area == 100


def test_profile_round_trip_and_upsert(tmp_path):
    path = tmp_path / "p.json"
    profile = load_profile(DEFAULT_PROFILE)
    upsert_rule(profile, "new_rule", (1, 2, 3), (4, 5, 6), "chat")
    upsert_rule(profile, "hp_bar_green", (50, 100, 100), (70, 255, 255), None)
    save_profile(profile, path)
    loaded = load_profile(path)
    assert loaded.rules == profile.rules
    green = next(r for r in loaded.rules if r.name == "hp_bar_green")
    assert green.region == "game_view" and green.max_area == 1500  # settings kept
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "content",
    [
        "{broken",
        '{"rules": [{"name": "x"}]}',
        '{"rules": [{"name": "x", "lower": [200, 0, 0], "upper": [10, 255, 255]}]}',
        '{"rules": [{"name": "x", "lower": [0, 0, 0], "upper": [1, 1, 1], "region": "nope"}]}',
    ],
)
def test_invalid_profiles_are_rejected(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ProfileError):
        load_profile(path)


def test_plugin_loader_skips_broken_files(tmp_path, capsys):
    (tmp_path / "good.py").write_text(
        "from colorvision import Plugin\nclass Good(Plugin):\n    name = 'good'\n", encoding="utf-8"
    )
    (tmp_path / "bad.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    (tmp_path / "_ignored.py").write_text("raise RuntimeError('skip')\n", encoding="utf-8")
    assert [p.name for p in load_plugins(tmp_path)] == ["good"]
    assert "bad.py" in capsys.readouterr().err


def test_host_disables_only_the_crashing_plugin(capsys):
    calls = []

    class Crash(Plugin):
        name = "crash"

        def on_frame(self, result):
            raise RuntimeError("boom")

    class Fine(Plugin):
        name = "fine"

        def on_frame(self, result):
            calls.append(result)

    host = PluginHost([Crash(), Fine()])
    host.frame("a")
    host.frame("b")
    assert [p.name for p in host.plugins] == ["fine"]
    assert calls == ["a", "b"]
    assert "crash" in capsys.readouterr().err


def test_cli_runs_end_to_end_on_an_image(tmp_path, capsys):
    image = tmp_path / "klatka_ąę.png"  # non-ASCII path on purpose
    write_png(image, synthetic_frame())
    code = main(["run", "--image", str(image), "--no-preview", "--max-frames", "3", "--fps", "200"])
    output = capsys.readouterr().out
    assert code == 0
    assert "Przetworzono klatek: 3" in output
    assert "target_watch" in output
    assert output.count("hp_bar_green: widoczny") == 1


def test_cli_reports_missing_profile(capsys):
    assert main(["run", "--profile", "nie_ma.json", "--no-preview", "--max-frames", "1"]) == 1
    assert "Błąd" in capsys.readouterr().err
