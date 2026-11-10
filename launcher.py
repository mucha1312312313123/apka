"""Entry point of the .exe. `--self-test FILE [--window TITLE]` checks the packaged program."""
import json
import sys


def self_test(output: str, window_title: str | None) -> int:
    """Exercise what packaging can break, and write the results, as the .exe has no console."""
    report = {}
    try:
        import numpy as np

        from colorvision.capture import WindowSource, write_png
        from colorvision.detector import TemplateRule
        from colorvision.gui.main_window import MainWindow  # noqa: F401  (imports all of Qt)
        from colorvision.paths import DATA_ROOT, bundled_files, prepare_user_files
        from colorvision.plugin import load_plugins, process_frame
        from colorvision.profile import load_profile

        prepare_user_files(DATA_ROOT, bundled_files())
        profile = load_profile(DATA_ROOT / "profiles" / "default.json")
        report["data_root"] = str(DATA_ROOT)
        report["rules"] = [rule.name for rule in profile.rules]
        report["states"] = [state.name for state in profile.states]
        report["plugins"] = [plugin.name for plugin in load_plugins(DATA_ROOT / "plugins")]

        frame = np.full((503, 765, 3), (40, 62, 74), np.uint8)
        frame[100:105, 200:225] = (0, 255, 0)
        icon = np.random.default_rng(1).integers(0, 256, (16, 16, 3), np.uint8)
        frame[300:316, 600:616] = icon
        profile.templates.append(TemplateRule("wzorzec", icon))
        found = process_frame(frame, profile).detections
        report["color"] = [(d.x, d.y) for d in found["hp_bar_green"]]
        report["template"] = [(d.x, d.y) for d in found["wzorzec"]]
        write_png(DATA_ROOT / "self_test.png", frame)
        (DATA_ROOT / "self_test.png").unlink()

        if window_title:
            source = WindowSource(window_title)
            grabbed = source.read()
            report["window"] = None if grabbed is None else list(grabbed.shape)
            source.close()
        report["ok"] = True
    except Exception as exc:  # the report is the whole point: it must say what broke
        report["ok"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
    with open(output, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return 0 if report["ok"] else 1


def main() -> int:
    if "--self-test" in sys.argv:
        arguments = sys.argv[sys.argv.index("--self-test") + 1:]
        title = arguments[arguments.index("--window") + 1] if "--window" in arguments else None
        return self_test(arguments[0], title)
    from colorvision.gui.main_window import run_app

    return run_app()


if __name__ == "__main__":
    sys.exit(main())
