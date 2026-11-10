"""Command line: python -m colorvision <windows|snapshot|calibrate|run>."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from .calibrate import run_calibration
from .capture import ImageSource, WindowSource, enable_dpi_awareness, list_windows, write_png
from .detector import annotate
from .plugin import Pipeline, PluginHost, load_plugins, process_frame
from .profile import ProfileError, load_profile

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TITLE = "Old School RuneScape"
PREVIEW = "Color Vision (Q = wyjscie)"


def make_source(args):
    return ImageSource(args.image) if args.image else WindowSource(args.title)


def cmd_windows(_args) -> int:
    enable_dpi_awareness()
    for window in list_windows():
        print(window.title)
    return 0


def cmd_snapshot(args) -> int:
    source = make_source(args)
    try:
        frame = source.read()
    finally:
        source.close()
    if frame is None:
        print(f"Brak okna {args.title!r} albo jest zminimalizowane. Lista okien: windows", file=sys.stderr)
        return 1
    write_png(args.output, frame)
    print(f"Zapisano {args.output} ({frame.shape[1]}x{frame.shape[0]})")
    return 0


def cmd_calibrate(args) -> int:
    source = make_source(args)
    try:
        run_calibration(source, args.profile, args.rule, args.region)
    finally:
        source.close()
    return 0


def cmd_run(args) -> int:
    profile = load_profile(args.profile)
    host = PluginHost(load_plugins(args.plugins))
    print(f"Reguły: {len(profile.rules)}, pluginy: {[p.name for p in host.plugins]}")
    source = make_source(args)
    host.start(profile)
    pipeline = Pipeline(profile, host)

    frames, waiting, fps = 0, False, 0.0
    period = 1.0 / args.fps
    last = time.perf_counter()
    try:
        while args.max_frames is None or frames < args.max_frames:
            started = time.perf_counter()
            frame = source.read()
            if frame is None:
                if not waiting:
                    print(f"Czekam na okno {args.title!r} (nie może być zminimalizowane)...")
                    waiting = True
                pipeline.idle()
                if not args.no_preview:
                    cv2.waitKey(500)
                else:
                    time.sleep(0.5)
                continue
            waiting = False

            result = process_frame(frame, profile)
            _, events = pipeline.step(result, time.monotonic())
            for event in events:
                print(f"[{time.strftime('%H:%M:%S')}] {event.message}")
            frames += 1

            if args.no_preview:
                time.sleep(max(0.0, period - (time.perf_counter() - started)))
                continue

            canvas = annotate(frame, profile.items, result.detections)
            host.draw(canvas, result)
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            cv2.putText(canvas, f"{fps:.0f} FPS", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(PREVIEW, canvas)
            wait = max(1, int((period - (time.perf_counter() - started)) * 1000))
            key = cv2.waitKey(wait) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(PREVIEW, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        host.stop()
        source.close()
        cv2.destroyAllWindows()
    print(f"Przetworzono klatek: {frames}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="colorvision", description="Wykrywanie kolorów w oknie klienta OSRS.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_source(p):
        p.add_argument("--title", default=DEFAULT_TITLE, help="tytuł okna lub jego fragment")
        p.add_argument("--image", help="użyj zapisanego obrazu zamiast okna")

    def add_profile(p):
        p.add_argument("--profile", default=str(ROOT / "profiles" / "default.json"))

    sub.add_parser("windows", help="wypisz tytuły widocznych okien").set_defaults(func=cmd_windows)

    p = sub.add_parser("snapshot", help="zapisz klatkę z klienta do PNG")
    add_source(p)
    p.add_argument("-o", "--output", default="snapshot.png")
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("calibrate", help="dobierz kolor reguły myszką i suwakami")
    add_source(p)
    add_profile(p)
    p.add_argument("--rule", required=True, help="nazwa reguły do utworzenia lub zmiany")
    p.add_argument("--region", help="region z profilu, np. game_view")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("run", help="uruchom wykrywanie i pluginy")
    add_source(p)
    add_profile(p)
    p.add_argument("--plugins", default=str(ROOT / "plugins"))
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--no-preview", action="store_true", help="bez okna podglądu")
    p.add_argument("--max-frames", type=int, help="zakończ po tylu klatkach")
    p.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "fps", 1) <= 0:
        print("--fps musi być większe od 0", file=sys.stderr)
        return 2
    try:
        return args.func(args)
    except (ProfileError, ValueError, OSError) as exc:
        print(f"Błąd: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
