"""Real mouse and keyboard input on Windows, and the emergency-stop key."""
from __future__ import annotations

import ctypes
import sys

MOUSE_FLAGS = {"left": (0x0002, 0x0004), "right": (0x0008, 0x0010)}
KEY_UP = 0x0002
VK_F12 = 0x7B
SPECIAL_KEYS = {
    "space": 0x20, "enter": 0x0D, "esc": 0x1B, "tab": 0x09, "shift": 0x10, "ctrl": 0x11,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
}


def virtual_key(key: str) -> int:
    if key in SPECIAL_KEYS:
        return SPECIAL_KEYS[key]
    if len(key) == 1 and key.isalnum():
        return ord(key.upper())
    if key.startswith("f") and key[1:].isdigit():
        return 0x6F + int(key[1:])
    raise ValueError(f"Nieznany klawisz: {key}")


def emergency_key_pressed() -> bool:
    """True if F12 was pressed since the last check, whichever window has focus."""
    if sys.platform != "win32":
        return False
    return bool(ctypes.windll.user32.GetAsyncKeyState(VK_F12) & 0x8001)


class WindowsInput:
    def __init__(self):
        if sys.platform != "win32":
            raise RuntimeError("Sterowanie działa tylko w Windows")
        self._user32 = ctypes.windll.user32

    def click(self, x: int, y: int, button: str) -> None:
        down, up = MOUSE_FLAGS[button]
        self._user32.SetCursorPos(x, y)
        self._user32.mouse_event(down, 0, 0, 0, 0)
        self._user32.mouse_event(up, 0, 0, 0, 0)

    def press(self, key: str) -> None:
        code = virtual_key(key)
        self._user32.keybd_event(code, 0, 0, 0)
        self._user32.keybd_event(code, 0, KEY_UP, 0)
