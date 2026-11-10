"""Colour detection: pure image processing, independent of the game window."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

Rect = tuple[int, int, int, int]  # x, y, width, height
Triple = tuple[int, int, int]

SHAPE_ANY = "any"
SHAPE_BAR = "bar"
SHAPE_RECTANGLE = "rectangle"
SHAPE_CIRCLE = "circle"
SHAPE_DOT = "dot"
SHAPE_OTHER = "other"
SHAPE_TEMPLATE = "template"  # reported for matches of a TemplateRule, not selectable
SHAPES = (SHAPE_ANY, SHAPE_BAR, SHAPE_RECTANGLE, SHAPE_CIRCLE, SHAPE_DOT, SHAPE_OTHER)

DOT_MAX_SIDE = 6
BAR_MIN_RATIO = 2.5
BAR_MIN_FILL = 0.7
RECTANGLE_MIN_FILL = 0.9
CIRCLE_MAX_RATIO = 1.3
CIRCLE_FILL_RANGE = (0.65, 0.9)  # an ideal circle fills about 78.5% of its bounding box
TEMPLATE_MIN_SIDE = 4
TEMPLATE_MIN_CONTRAST = 3.0


def classify_shape(width: int, height: int, area: int) -> str:
    """Name a blob's shape from its bounding box and how much of it the blob fills."""
    long_side, short_side = max(width, height), min(width, height)
    if long_side <= DOT_MAX_SIDE:
        return SHAPE_DOT
    ratio = long_side / short_side
    fill = area / (width * height)
    if ratio >= BAR_MIN_RATIO and fill >= BAR_MIN_FILL:
        return SHAPE_BAR
    if ratio < BAR_MIN_RATIO and fill >= RECTANGLE_MIN_FILL:
        return SHAPE_RECTANGLE
    if ratio <= CIRCLE_MAX_RATIO and CIRCLE_FILL_RANGE[0] <= fill < CIRCLE_FILL_RANGE[1]:
        return SHAPE_CIRCLE
    return SHAPE_OTHER


@dataclass(frozen=True)
class ColorRule:
    """HSV range in OpenCV units: H 0-179, S 0-255, V 0-255.

    If the lower hue is greater than the upper hue, the range wraps through 0.
    That is what red needs, e.g. H 175..3.
    """

    name: str
    lower: Triple
    upper: Triple
    region: str | None = None
    min_area: int = 10
    max_area: int | None = None
    merge_px: int = 3  # morphological closing size; joins blobs split by small gaps
    draw_bgr: Triple = (0, 255, 255)
    enabled: bool = True
    shape: str = SHAPE_ANY
    min_confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError(f"{self.name}: unknown shape {self.shape!r}")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(f"{self.name}: min_confidence must be within 0-1")
        h_lo, s_lo, v_lo = self.lower
        h_hi, s_hi, v_hi = self.upper
        if not (0 <= h_lo <= 179 and 0 <= h_hi <= 179):
            raise ValueError(f"{self.name}: H must be within 0-179")
        if not (0 <= s_lo <= s_hi <= 255 and 0 <= v_lo <= v_hi <= 255):
            raise ValueError(f"{self.name}: S/V must be within 0-255 with lower <= upper")
        if self.min_area < 1 or self.merge_px < 0:
            raise ValueError(f"{self.name}: min_area must be >= 1 and merge_px >= 0")


@dataclass(frozen=True)
class TemplateRule:
    """Finds places in the frame that look like a stored picture.

    captured_size is the frame size the picture was cut from; in a differently sized
    window the picture is scaled by the same ratio before searching.
    """

    name: str
    image: np.ndarray = field(compare=False, repr=False)
    threshold: float = 0.9
    region: str | None = None
    max_results: int = 10
    draw_bgr: Triple = (255, 160, 60)
    enabled: bool = True
    captured_size: tuple[int, int] = (765, 503)

    def __post_init__(self) -> None:
        if self.image is None or self.image.ndim != 3 or self.image.shape[2] != 3 or self.image.size == 0:
            raise ValueError(f"{self.name}: template must be a colour image")
        if not 0.1 <= self.threshold <= 1.0:
            raise ValueError(f"{self.name}: threshold must be within 0.1-1.0")
        if self.max_results < 1:
            raise ValueError(f"{self.name}: max_results must be >= 1")


@dataclass(frozen=True)
class Detection:
    rule: str
    x: int
    y: int
    width: int
    height: int
    area: int
    center: tuple[int, int]
    shape: str = SHAPE_OTHER
    score: float | None = None  # confidence 0-1; see color_confidence and match_template


def _contrast(image: np.ndarray) -> float:
    """Largest per-channel spread; across all channels a flat colour would look contrasty."""
    return float(image.reshape(-1, 3).std(axis=0).max())


def template_is_usable(image: np.ndarray) -> bool:
    """A picture needs some size and contrast: a flat one would match anything or nothing."""
    height, width = image.shape[:2]
    return min(width, height) >= TEMPLATE_MIN_SIDE and _contrast(image) >= TEMPLATE_MIN_CONTRAST


def match_template(frame_bgr: np.ndarray, template: TemplateRule, rect: Rect | None = None) -> list[Detection]:
    """Find separate matches of a template, best first, in full-frame coordinates."""
    frame_h, frame_w = frame_bgr.shape[:2]
    x0, y0, w, h = rect if rect is not None else (0, 0, frame_w, frame_h)
    roi = frame_bgr[y0:y0 + h, x0:x0 + w]

    picture = template.image
    if (frame_w, frame_h) != template.captured_size:
        new_w = round(picture.shape[1] * frame_w / template.captured_size[0])
        new_h = round(picture.shape[0] * frame_h / template.captured_size[1])
        if new_w < 1 or new_h < 1:
            return []
        picture = cv2.resize(picture, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pic_h, pic_w = picture.shape[:2]
    if roi.shape[0] < pic_h or roi.shape[1] < pic_w or _contrast(picture) < 1.0:
        return []

    scores = np.nan_to_num(cv2.matchTemplate(roi, picture, cv2.TM_CCOEFF_NORMED), nan=0.0, posinf=0.0, neginf=0.0)
    found: list[Detection] = []
    while len(found) < template.max_results:
        _, best, _, (x, y) = cv2.minMaxLoc(scores)
        if best < template.threshold:
            break
        found.append(
            Detection(
                rule=template.name,
                x=x0 + x,
                y=y0 + y,
                width=pic_w,
                height=pic_h,
                area=pic_w * pic_h,
                center=(x0 + x + pic_w // 2, y0 + y + pic_h // 2),
                shape=SHAPE_TEMPLATE,
                score=float(best),
            )
        )
        # Blank the neighbourhood so one object is not reported many times.
        scores[max(0, y - pic_h // 2):y + pic_h // 2 + 1, max(0, x - pic_w // 2):x + pic_w // 2 + 1] = -1.0
    return found


def hsv_mask(hsv: np.ndarray, lower: Triple, upper: Triple) -> np.ndarray:
    """Return a 0/255 mask for an HSV range, handling hue wrap-around."""
    if lower[0] <= upper[0]:
        return cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    high_part = cv2.inRange(
        hsv, np.array(lower, np.uint8), np.array((179, upper[1], upper[2]), np.uint8)
    )
    low_part = cv2.inRange(
        hsv, np.array((0, lower[1], lower[2]), np.uint8), np.array(upper, np.uint8)
    )
    return cv2.bitwise_or(high_part, low_part)


def color_confidence(hue_values: np.ndarray, matching_pixels: int, area: int, rule: ColorRule) -> float:
    """How well a blob fits its rule, 0-1. A measure of fit, not a probability of being right.

    Multiplies two things: the share of the blob's pixels truly in the colour range (merging
    can fill gaps), and how close their hue is to the middle of the range. Pixels exactly at
    the edge of the range count half.
    """
    if matching_pixels == 0 or area == 0:
        return 0.0
    low, high = rule.lower[0], rule.upper[0]
    span = high - low if low <= high else high + 180 - low
    middle = (low + span / 2) % 180
    distance = np.abs(hue_values.astype(float) - middle)
    distance = np.minimum(distance, 180 - distance)
    off_center = min(1.0, float(distance.mean()) / max(span / 2, 1.0))
    return (matching_pixels / area) * (1.0 - 0.5 * off_center)


def detect_rule(hsv: np.ndarray, rule: ColorRule, rect: Rect | None = None) -> list[Detection]:
    """Find the blobs matching a rule, largest first, in full-frame coordinates."""
    frame_h, frame_w = hsv.shape[:2]
    x0, y0, w, h = rect if rect is not None else (0, 0, frame_w, frame_h)
    roi = hsv[y0:y0 + h, x0:x0 + w]
    if roi.size == 0:
        return []

    matching = hsv_mask(roi, rule.lower, rule.upper)
    mask = matching
    if rule.merge_px > 0:
        # An even-sized kernel would shift the result by one pixel, so force it odd.
        size = rule.merge_px | 1
        kernel = np.ones((size, size), np.uint8)
        mask = cv2.morphologyEx(matching, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    found: list[Detection] = []
    for i in range(1, count):  # label 0 is the background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < rule.min_area or (rule.max_area is not None and area > rule.max_area):
            continue
        width, height = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        shape = classify_shape(width, height, area)
        if rule.shape != SHAPE_ANY and shape != rule.shape:
            continue
        left, top = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        in_blob = labels[top:top + height, left:left + width] == i
        truly_matching = in_blob & (matching[top:top + height, left:left + width] > 0)
        hues = roi[top:top + height, left:left + width, 0][truly_matching]
        score = color_confidence(hues, int(truly_matching.sum()), area, rule)
        if score < rule.min_confidence:
            continue
        found.append(
            Detection(
                rule=rule.name,
                x=x0 + left,
                y=y0 + top,
                width=width,
                height=height,
                area=area,
                center=(x0 + round(centroids[i][0]), y0 + round(centroids[i][1])),
                shape=shape,
                score=score,
            )
        )
    found.sort(key=lambda d: d.area, reverse=True)
    return found


def annotate(frame_bgr: np.ndarray, rules: list, detections: dict[str, list[Detection]], labels: bool = True) -> np.ndarray:
    """Return a copy of the frame with boxes and labels; rules may be colour or template rules."""
    canvas = frame_bgr.copy()
    for rule in rules:
        for d in detections.get(rule.name, []):
            cv2.rectangle(canvas, (d.x, d.y), (d.x + d.width - 1, d.y + d.height - 1), rule.draw_bgr, 1)
            cv2.circle(canvas, d.center, 2, rule.draw_bgr, -1)
            if labels:
                cv2.putText(
                    canvas, rule.name, (d.x, max(10, d.y - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, rule.draw_bgr, 1, cv2.LINE_AA,
                )
    return canvas
