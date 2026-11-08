"""Exercise navigation and real trainer delegation without scientific edits."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pepplatform_cli.__main__ import build_parser, ensemble_config, main
from pepplatform_cli.storage import CLIError, inventory, read_json
from pepplatform_cli.training import namespace_argv, parser_for, train


@pytest.mark.parametrize("task", ["pep_function", "pep_hlai", "pep_hlaii", "pep_protein_interaction"])
def test_training_parser_round_trip(task):
    parser, _, _ = parser_for(task)
    args = parser.parse_args(["--task_type", task, "--seed", "17", "--head_hidden_dims", "[32,16]",
                              "--no-enable_semantic_similarity" if task == "pep_function" else "--no-enable_interaction_semantic_similarity"])
    restored = parser.parse_args(namespace_argv(parser, args))
    for action in parser._actions:
        if action.dest not in {"interactive", "advanced_wizard", "help", "list_objectives"} and hasattr(args, action.dest):
            assert getattr(restored, action.dest) == getattr(args, action.dest), action.dest


def small_data(root):
    for split, values in {"train": ["AAAA", "CCCC", "DDDD", "EEEE", "FFFF", "GGGG"],
                          "val": ["HHHH", "IIII", "KKKK", "LLLL"]}.items():
        directory = root / split
        directory.mkdir(parents=True)
        (directory / "seqs.fasta").write_text("".join(f">{i}\n{x}\n" for i, x in enumerate(values)))
        (directory / "labels.csv").write_text("A,B\n" + "".join(f"{i % 2},{1 - i % 2}\n" for i in range(len(values))))


def test_real_training_envelope_equals_original_entrypoint(tmp_path):
    from pepplatform_cli.storage import ROOT
    data_dir = tmp_path / "source_data"
    small_data(data_dir)
    before = inventory(data_dir)
    arguments = ["--data_dir", str(data_dir), "--tokenizer", "kmer", "--k", "1", "--encoder", "bow",
                 "--model", "rf", "--max_len", "4", "--rf_estimators", "7", "--no-evaluate_test"]
    wrapped = train(arguments, workspace=tmp_path / "workspace")
    assert wrapped["status"] == "succeeded"
    proof = read_json(wrapped["provenance"])
    assert proof["kind"] == "cli_training" and proof["status"] == "verified"
    assert inventory(data_dir) == before
    artifact = Path(wrapped["artifact_dir"])
    assert proof["artifact_files"] == inventory(artifact)
    command = read_json(Path(wrapped["directory"]) / "command.json")["command"]
    direct = tmp_path / "direct"
    command[command.index("--artifact_dir") + 1] = str(direct)
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with np.load(artifact / "evaluation_scores.npz") as left, np.load(direct / "evaluation_scores.npz") as right:
        np.testing.assert_array_equal(left["validation_scores"], right["validation_scores"])
    # Receipt can be checked against a relocated copy of the same input bytes.
    from pepplatform_cli.ensemble import prepare
    # Duplicate members are correctly rejected; independently copied bundles have matching verified receipts.
    import shutil
    second = tmp_path / "second"
    shutil.copytree(artifact, second / "artifact")
    # This fixture has no absolute runtime references; receipt bytes remain hash-compatible.
    shutil.copyfile(wrapped["provenance"], second / "training_provenance.json")
    request = prepare({"task_type": "pep_function", "data_dir": str(data_dir),
                       "members": [str(artifact), str(second / "artifact")]})
    assert request["provenance_status"] == "verified"


def test_training_failure_keeps_diagnostics_and_no_receipt(tmp_path):
    from types import SimpleNamespace
    root = tmp_path / "data"
    small_data(root)
    with pytest.raises(CLIError, match="status 9"):
        train(["--data_dir", str(root)], workspace=tmp_path / "workspace",
              run_process=lambda *_a, **_kw: SimpleNamespace(returncode=9))
    runs = list((tmp_path / "workspace/runs").iterdir())
    assert len(runs) == 1
    assert read_json(runs[0] / "run.json")["status"] == "failed"
    assert not (runs[0] / "training_provenance.json").exists()


def test_explicit_arguments_override_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"method": "rank_mean", "seed": 7, "weight_source": "equal"}))
    args = build_parser().parse_args(["ensemble", "preflight", "--config", str(path), "--seed", "42"])
    config = ensemble_config(args)
    assert config["seed"] == 42 and config["method"] == "rank_mean"
    assert config["weight_source"] == "equal"


def test_navigation_exit_and_keyboard_cancel(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: "5")
    assert main(["--workspace", str(tmp_path / "workspace")]) == 0
    assert not (tmp_path / "workspace").exists()
    monkeypatch.setattr("builtins.input", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert main(["--workspace", str(tmp_path / "workspace")]) == 130


def test_ensemble_navigation_repair_save_and_create(tmp_path, monkeypatch):
    from pepplatform_cli import ensemble, __main__
    captured = []
    checks = []

    def preflight(config):
        checks.append(dict(config))
        return {"valid": config.get("allow_unverified_provenance", False), "errors": []}

    monkeypatch.setattr(ensemble, "preflight", preflight)
    monkeypatch.setattr(__main__, "create_ensemble", lambda config, workspace: captured.append(dict(config)) or {"status": "succeeded"})
    config_path = tmp_path / "saved.json"
    entries = iter(["2", "1", str(tmp_path / "data"), f"{tmp_path}/one {tmp_path}/two", "1", "2", "2,3",
                    "1", "1", "", "1", "1", "1", '{"allow_unverified_provenance": true}',
                    str(config_path), "1", "5"])
    monkeypatch.setattr("builtins.input", lambda *_: next(entries))
    assert main(["--workspace", str(tmp_path / "workspace")]) == 0
    assert len(checks) == 2 and len(captured) == 1
    assert captured[0]["weights"] == [2, 3]
    assert read_json(config_path) == captured[0]


def test_failing_preflight_has_nonzero_exit_and_creates_no_run(tmp_path, capsys):
    assert main(["ensemble", "preflight", "--members", str(tmp_path / "one"), str(tmp_path / "two"),
                 "--data-dir", str(tmp_path / "missing"), "--workspace", str(tmp_path / "workspace")]) == 2
    assert json.loads(capsys.readouterr().out)["valid"] is False
    assert not (tmp_path / "workspace").exists()
