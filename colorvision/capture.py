"""Frame sources: the client window on screen, or a saved image for testing."""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SHELL_WINDOW_CLASSES = {"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"}
GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x80
DWMWA_CLOAKED = 14
WHOLE_SCREEN_SHARE = 0.9


def read_image(path: str | Path) -> np.ndarray:
    """cv2.imread that also handles non-ASCII paths on Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Nie można odczytać obrazu: {path}")
    return image


def write_png(path: str | Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Nie można zakodować PNG")
    encoded.tofile(str(path))


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str


def _user32():
    if sys.platform != "win32":
        raise RuntimeError("Przechwytywanie okna działa tylko w Windows")
    return ctypes.windll.user32


def enable_dpi_awareness() -> None:
    """Without this, Windows scaling (125%, 150%) would shift the capture area."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except OSError:
        pass  # already set, or an older Windows version


def window_title(hwnd: int) -> str:
    user32 = _user32()
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    length = user32.GetWindowTextLengthW(hwnd)
    if not length:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value


def is_capturable(hwnd: int) -> bool:
    """False for the desktop, the taskbar, tool windows and windows Windows keeps hidden.

    The desktop matters most: choosing it would capture the whole screen.
    """
    user32 = _user32()
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    if not user32.IsWindowVisible(hwnd):
        return False
    class_name = ctypes.create_unicode_buffer(128)
    user32.GetClassNameW(hwnd, class_name, 128)
    if class_name.value in SHELL_WINDOW_CLASSES:
        return False
    if user32.GetWindowLongW(hwnd, GWL_EXSTYLE) & WS_EX_TOOLWINDOW:
        return False
    cloaked = ctypes.c_int(0)
    result = ctypes.windll.dwmapi.DwmGetWindowAttribute(
        wintypes.HWND(hwnd), DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked)
    )
    return not (result == 0 and cloaked.value)


def list_windows() -> list[WindowInfo]:
    user32 = _user32()
    found: list[WindowInfo] = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd, _):
        if hwnd and is_capturable(hwnd):
            title = window_title(hwnd)
            if title:
                found.append(WindowInfo(int(hwnd), title))
        return True

    user32.EnumWindows(callback_type(callback), 0)
    return found


def foreground_window() -> WindowInfo | None:
    """The active window, unless it belongs to this program or cannot be captured."""
    user32 = _user32()
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None
    process_id = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
    if process_id.value == os.getpid() or not is_capturable(hwnd):
        return None
    title = window_title(hwnd)
    return WindowInfo(int(hwnd), title) if title else None


def intersect(rect: tuple[int, int, int, int], other: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    left, top = max(rect[0], other[0]), max(rect[1], other[1])
    right, bottom = min(rect[0] + rect[2], other[0] + other[2]), min(rect[1] + rect[3], other[1] + other[3])
    return (left, top, right - left, bottom - top) if right > left and bottom > top else None


def covers_a_monitor(rect: tuple[int, int, int, int], monitors: list[tuple[int, int, int, int]]) -> bool:
    """Whether the area fills nearly a whole monitor, as a maximised or fullscreen window does."""
    for monitor in monitors:
        common = intersect(rect, monitor)
        if common and common[2] * common[3] >= WHOLE_SCREEN_SHARE * monitor[2] * monitor[3]:
            return True
    return False


def find_window(title: str) -> WindowInfo | None:
    """Prefer an exact title match, then a case-insensitive substring match."""
    wanted = title.casefold()
    windows = list_windows()
    for window in windows:
        if window.title.casefold() == wanted:
            return window
    for window in windows:
        if wanted in window.title.casefold():
            return window
    return None


def client_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    """Screen rectangle of the window's client area, or None if unavailable."""
    user32 = _user32()
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    if user32.IsIconic(hwnd) or not user32.IsWindowVisible(hwnd):
        return None
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    origin = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        return None
    if rect.right <= 0 or rect.bottom <= 0:
        return None
    return (origin.x, origin.y, rect.right, rect.bottom)


class WindowSource:
    """Grabs the client area from the screen; it must be visible and uncovered."""

    def __init__(self, title: str, hwnd: int | None = None):
        import mss  # imported here so image-based tests do not need a display

        enable_dpi_awareness()
        self.title = title
        # Newer mss versions renamed the entry point; requirements allow both.
        self._sct = (getattr(mss, "MSS", None) or mss.mss)()
        self._hwnd: int | None = hwnd
        self._crop: tuple[int, int, int, int] | None = None

    def set_crop(self, crop: tuple[int, int, int, int] | None) -> None:
        """Capture only this part of the client area (x, y, width, height), or all of it."""
        self._crop = crop

    def _capture_rect(self) -> tuple[int, int, int, int] | None:
        """On-screen rectangle being captured. A crop that no longer fits is ignored."""
        rect = client_rect(self._hwnd) if self._hwnd is not None else None
        if rect is None or self._crop is None:
            return rect
        x, y, width, height = self._crop
        return intersect(rect, (rect[0] + x, rect[1] + y, width, height)) or rect

    def describe(self) -> dict | None:
        """What is being captured right now, for showing to the user."""
        rect = self._capture_rect()
        if rect is None:
            return None
        monitors = [(m["left"], m["top"], m["width"], m["height"]) for m in self._sct.monitors[1:]]
        return {
            "title": window_title(self._hwnd),
            "width": rect[2],
            "height": rect[3],
            "whole_screen": covers_a_monitor(rect, monitors),
        }

    def read(self) -> np.ndarray | None:
        """Return a BGR frame, or None while the window is missing or minimised."""
        rect = self._capture_rect()
        if rect is None:
            window = find_window(self.title)
            if window is None:
                self._hwnd = None
                return None
            self._hwnd = window.hwnd
            rect = self._capture_rect()
            if rect is None:
                return None
        left, top, width, height = rect
        shot = self._sct.grab({"left": left, "top": top, "width": width, "height": height})
        return np.asarray(shot)[:, :, :3].copy()  # BGRA -> BGR

    def is_covered(self) -> bool:
        """True if another window lies over the client's centre or corners."""
        rect = self._capture_rect()
        if rect is None:
            return False
        user32 = _user32()
        user32.WindowFromPoint.argtypes = [wintypes.POINT]
        user32.WindowFromPoint.restype = wintypes.HWND
        user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        user32.GetAncestor.restype = wintypes.HWND
        left, top, width, height = rect
        inset = min(8, width // 2, height // 2)
        right, bottom = left + width - 1 - inset, top + height - 1 - inset
        points = [
            (left + width // 2, top + height // 2),
            (left + inset, top + inset), (right, top + inset),
            (left + inset, bottom), (right, bottom),
        ]
        ga_root = 2
        for x, y in points:
            hit = user32.WindowFromPoint(wintypes.POINT(x, y))
            if hit is None or user32.GetAncestor(hit, ga_root) != self._hwnd:
                return True
        return False

    def input_target(self) -> tuple[int, int, int, int] | None:
        """The client rectangle, only if real input would land in the client right now."""
        if self._hwnd is None:
            return None
        user32 = _user32()
        user32.GetForegroundWindow.restype = wintypes.HWND
        if user32.GetForegroundWindow() != self._hwnd or self.is_covered():
            return None
        return self._capture_rect()

    def close(self) -> None:
        self._sct.close()


class ImageSource:
    """Serves one image repeatedly; for tuning and tests without the game."""

    def __init__(self, path: str | Path):
        self._image = read_image(path)
        self._crop: tuple[int, int, int, int] | None = None

    def set_crop(self, crop: tuple[int, int, int, int] | None) -> None:
        self._crop = crop

    def read(self) -> np.ndarray:
        if self._crop is None:
            return self._image.copy()
        x, y, width, height = self._crop
        part = self._image[y:y + height, x:x + width]
        return part.copy() if part.size else self._image.copy()

    def close(self) -> None:
        pass
