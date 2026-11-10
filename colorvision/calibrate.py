"""Interactive HSV calibration: click a colour, tune the sliders, save it as a rule."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .detector import ColorRule, detect_rule, hsv_mask
from .profile import Profile, load_profile, save_profile

WINDOW = "Kalibracja (klik=probka, S=zapisz, Q=wyjdz)"
SLIDERS = ("H min", "H max", "S min", "S max", "V min", "V max")
LIMITS = (179, 179, 255, 255, 255, 255)
HUE_MARGIN, SV_MARGIN = 6, 40


def range_from_sample(hsv: np.ndarray, x: int, y: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Build a range around the median of a 5x5 patch; hue may wrap through 0."""
    height, width = hsv.shape[:2]
    patch = hsv[max(0, y - 2):min(height, y + 3), max(0, x - 2):min(width, x + 3)].reshape(-1, 3)

    # Hue is circular: take the median relative to the clicked pixel, so a red
    # patch mixing 178 and 1 does not average out to a false cyan around 90.
    clicked_hue = int(hsv[y, x, 0])
    offsets = (patch[:, 0].astype(int) - clicked_hue + 90) % 180 - 90
    hue = (clicked_hue + int(np.median(offsets))) % 180
    sat, val = int(np.median(patch[:, 1])), int(np.median(patch[:, 2]))

    lower = ((hue - HUE_MARGIN) % 180, max(0, sat - SV_MARGIN), max(0, val - SV_MARGIN))
    upper = ((hue + HUE_MARGIN) % 180, min(255, sat + SV_MARGIN), min(255, val + SV_MARGIN))
    return lower, upper


def upsert_rule(profile: Profile, rule_name: str, lower, upper, region: str | None) -> None:
    """Update the range of an existing rule, keeping its settings, or add a new one."""
    for index, rule in enumerate(profile.rules):
        if rule.name == rule_name:
            updated = replace(rule, lower=lower, upper=upper)
            if region is not None:
                updated = replace(updated, region=region)
            profile.rules[index] = updated
            return
    profile.rules.append(ColorRule(name=rule_name, lower=lower, upper=upper, region=region))


def run_calibration(source, profile_path: str | Path, rule_name: str, region: str | None = None) -> None:
    profile = load_profile(profile_path)
    if region is not None and region not in profile.regions:
        raise ValueError(f"Nieznany region: {region!r}")
    existing = next((r for r in profile.rules if r.name == rule_name), None)
    lower = existing.lower if existing else (0, 0, 0)
    upper = existing.upper if existing else (179, 255, 255)

    state = {"sample": None}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["sample"] = (x, y)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    initial = (lower[0], upper[0], lower[1], upper[1], lower[2], upper[2])
    for label, limit, value in zip(SLIDERS, LIMITS, initial):
        cv2.createTrackbar(label, WINDOW, value, limit, lambda _v: None)

    frame = None
    while True:
        latest = source.read()
        if latest is not None:
            frame = latest
        if frame is None:
            cv2.waitKey(200)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            print("Czekam na okno klienta...", end="\r")
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if state["sample"] is not None:
            x, y = state["sample"]
            state["sample"] = None
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                sampled_lower, sampled_upper = range_from_sample(hsv, x, y)
                values = (*zip(sampled_lower, sampled_upper),)
                flat = (values[0][0], values[0][1], values[1][0], values[1][1], values[2][0], values[2][1])
                for label, value in zip(SLIDERS, flat):
                    cv2.setTrackbarPos(label, WINDOW, value)

        h_lo, h_hi, s_lo, s_hi, v_lo, v_hi = (cv2.getTrackbarPos(s, WINDOW) for s in SLIDERS)
        # Dragging S/V past each other is harmless: keep the range valid.
        s_lo, s_hi = min(s_lo, s_hi), max(s_lo, s_hi)
        v_lo, v_hi = min(v_lo, v_hi), max(v_lo, v_hi)
        lower, upper = (h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi)

        # Matching pixels keep their colour, everything else is dimmed.
        mask = hsv_mask(hsv, lower, upper)
        canvas = (frame * 0.25).astype(np.uint8)
        canvas[mask > 0] = frame[mask > 0]
        height, width = frame.shape[:2]
        rule_region = region or (existing.region if existing else None)
        rect = profile.region_rect(rule_region, width, height)
        rule = ColorRule(rule_name, lower, upper, min_area=existing.min_area if existing else 10)
        for d in detect_rule(hsv, rule, rect):
            cv2.rectangle(canvas, (d.x, d.y), (d.x + d.width - 1, d.y + d.height - 1), (255, 255, 255), 1)
        cv2.imshow(WINDOW, canvas)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (ord("s"), ord("S")):
            upsert_rule(profile, rule_name, lower, upper, region)
            save_profile(profile, profile_path)
            print(f"Zapisano regułę {rule_name}: lower={lower} upper={upper}")
    cv2.destroyAllWindows()
