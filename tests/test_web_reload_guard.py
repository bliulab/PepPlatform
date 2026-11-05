"""The reload watcher must not cover the runtime root.

Starting a Web experiment stages ``*.py`` source snapshots under the runtime root. When the
runtime root sits inside a bare ``uvicorn --reload`` watch tree, those writes restart the
backend and the startup reconciliation cancels the run.
"""

from pepplatform.web.app import _reload_watches_runtime_root


def test_bare_reload_covers_project_runtime_root(tmp_path):
    runtime_root = tmp_path / ".pepplatform_web"

    assert _reload_watches_runtime_root(
        ["uvicorn", "pepplatform.web.app:app", "--reload"],
        runtime_root,
    )


def test_without_reload_the_runtime_root_is_safe(tmp_path):
    assert not _reload_watches_runtime_root(
        ["uvicorn", "pepplatform.web.app:app"],
        tmp_path / ".pepplatform_web",
    )


def test_reload_dirs_scoped_to_source_are_safe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    argv = [
        "uvicorn",
        "pepplatform.web.app:app",
        "--reload",
        "--reload-dir",
        "pepplatform",
        "--reload-dir=scripts",
    ]

    assert not _reload_watches_runtime_root(argv, tmp_path / ".pepplatform_web")
    assert _reload_watches_runtime_root(argv, tmp_path / "pepplatform")


def test_reload_dirs_including_the_runtime_root_are_flagged(tmp_path):
    runtime_root = tmp_path / ".pepplatform_web"

    assert _reload_watches_runtime_root(
        ["uvicorn", "--reload", f"--reload-dir={runtime_root}"],
        runtime_root,
    )


def test_absolute_reload_exclude_covers_the_runtime_root(tmp_path):
    runtime_root = tmp_path / ".pepplatform_web"

    assert not _reload_watches_runtime_root(
        ["uvicorn", "--reload", "--reload-exclude", str(runtime_root)],
        runtime_root,
    )


def test_no_reload_flag_value_is_treated_as_disabled(tmp_path):
    assert not _reload_watches_runtime_root(
        ["uvicorn", "--reload=false"],
        tmp_path / ".pepplatform_web",
    )
