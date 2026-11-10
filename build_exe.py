"""Builds dist/OSRS Color Vision.exe. Run with build_exe.bat."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
NAME = "OSRS Color Vision"
# Parts of Qt the program never uses; leaving them out makes the file much smaller.
UNUSED_QT = [
    "QtNetwork", "QtQml", "QtQuick", "QtPdf", "QtPdfWidgets", "QtSql", "QtTest", "QtXml", "QtDBus", "QtOpenGL",
    "QtOpenGLWidgets", "QtHelp", "QtDesigner", "QtUiTools", "QtSvg", "QtSvgWidgets", "QtPrintSupport", "QtConcurrent",
]


def make_icon(path: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSize
    from PySide6.QtWidgets import QApplication

    from colorvision.gui.theme import app_icon

    app = QApplication.instance() or QApplication([])
    if not app_icon().pixmap(QSize(256, 256)).save(str(path), "ICO"):
        raise RuntimeError("Nie można zapisać ikony")


def main() -> None:
    import PyInstaller.__main__

    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    icon = build / "icon.ico"
    make_icon(icon)
    arguments = [
        str(ROOT / "launcher.py"),
        "--name", NAME,
        "--onefile",
        "--windowed",
        "--noconfirm",
        "--clean",
        "--icon", str(icon),
        "--distpath", str(ROOT / "dist"),
        "--workpath", str(build),
        "--specpath", str(build),
        "--paths", str(ROOT),
        "--add-data", f"{ROOT / 'profiles' / 'default.json'}{os.pathsep}bundled/profiles",
        "--add-data", f"{ROOT / 'plugins' / 'target_watch.py'}{os.pathsep}bundled/plugins",
    ]
    for module in UNUSED_QT:
        arguments += ["--exclude-module", f"PySide6.{module}"]
    arguments += ["--exclude-module", "pytest", "--exclude-module", "tkinter"]
    PyInstaller.__main__.run(arguments)
    exe = ROOT / "dist" / f"{NAME}.exe"
    print(f"Gotowe: {exe} ({exe.stat().st_size / 1_000_000:.0f} MB)")


if __name__ == "__main__":
    main()
