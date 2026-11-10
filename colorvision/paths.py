"""Where the program keeps its files, when run from source and as a single .exe.

A single-file .exe unpacks itself to a temporary folder that disappears on exit, so
anything the user creates must live elsewhere. Replacing the .exe keeps all of it.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

APP_FOLDER = "OSRS Color Vision"
FROZEN = bool(getattr(sys, "frozen", False))


def data_root() -> Path:
    if not FROZEN:
        return Path(__file__).resolve().parent.parent
    base = os.environ.get("APPDATA")
    return (Path(base) if base else Path.home()) / APP_FOLDER


def bundled_files() -> Path | None:
    """Starter files inside the .exe; None when running from source."""
    return Path(sys._MEIPASS) / "bundled" if FROZEN else None


def prepare_user_files(root: Path, bundled: Path | None) -> None:
    """Create the folders and copy starter files, never overwriting what the user has."""
    for name in ("profiles", "plugins"):
        (root / name).mkdir(parents=True, exist_ok=True)
    if bundled is None:
        return
    for source in bundled.rglob("*"):
        target = root / source.relative_to(bundled)
        if source.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


DATA_ROOT = data_root()
