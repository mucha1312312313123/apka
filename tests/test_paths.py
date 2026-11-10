from pathlib import Path

from colorvision.paths import bundled_files, data_root, prepare_user_files


def test_running_from_source_uses_the_project_folder():
    assert data_root() == Path(__file__).resolve().parent.parent
    assert bundled_files() is None


def starter(tmp_path):
    bundled = tmp_path / "bundled"
    (bundled / "profiles").mkdir(parents=True)
    (bundled / "plugins").mkdir()
    (bundled / "profiles" / "default.json").write_text("nowy", encoding="utf-8")
    (bundled / "plugins" / "p.py").write_text("x = 1", encoding="utf-8")
    return bundled


def test_first_start_copies_the_starter_files(tmp_path):
    root = tmp_path / "dane"
    prepare_user_files(root, starter(tmp_path))
    assert (root / "profiles" / "default.json").read_text(encoding="utf-8") == "nowy"
    assert (root / "plugins" / "p.py").exists()


def test_a_new_exe_never_overwrites_the_users_files(tmp_path):
    root = tmp_path / "dane"
    (root / "profiles").mkdir(parents=True)
    (root / "profiles" / "default.json").write_text("moje zmiany", encoding="utf-8")
    (root / "profiles" / "wlasny.json").write_text("moj profil", encoding="utf-8")
    prepare_user_files(root, starter(tmp_path))
    assert (root / "profiles" / "default.json").read_text(encoding="utf-8") == "moje zmiany"
    assert (root / "profiles" / "wlasny.json").read_text(encoding="utf-8") == "moj profil"


def test_running_from_source_only_creates_folders(tmp_path):
    prepare_user_files(tmp_path, None)
    assert (tmp_path / "profiles").is_dir() and (tmp_path / "plugins").is_dir()
