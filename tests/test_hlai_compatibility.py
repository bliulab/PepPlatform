"""Public HLA-I interfaces and unchanged historical experiment behavior."""

from argparse import Namespace
import csv
import hashlib

import numpy as np
import pytest
from sqlalchemy import select, update

from pepplatform.hlai_input import HLAIInputError, hla_allele
from pepplatform.objectives import resolve_selection
from pepplatform.route_registry import resolve_effective_route
from pepplatform.semantic import semantic_relation_type
from pepplatform.task_compat import normalize_task_type, resolve_hlai_root
from pepplatform.web.capabilities import TASKS
from pepplatform.web.datasets import DatasetValidationError, _pair_partner
from pepplatform.web.store import MetadataStore, datasets
from scripts import train_pair_pipeline as pipeline


@pytest.mark.parametrize("old", ["pep_mhci", "pep_mhci_interaction", "peptide_mhci_interaction"])
def test_aliases_preserve_selection_route_semantics_and_input_schema(old):
    config = {"tokenizer": "kmer", "encoder": "onehot", "model": "cnn"}
    assert normalize_task_type(old) == "pep_hlai"
    assert (
        resolve_selection({"task_type": old}).to_dict()
        == resolve_selection({"task_type": "pep_hlai"}).to_dict()
    )
    assert resolve_effective_route(config, old) == resolve_effective_route(config, "pep_hlai")
    assert (
        semantic_relation_type(old)
        == semantic_relation_type("pep_hlai")
        == "interaction_profile_jaccard"
    )
    assert pipeline._pair_input_schema(Namespace(task_type=old)) == "peptide_hlai_pairs"


def test_capabilities_only_advertise_four_canonical_tasks():
    assert {t["id"] for t in TASKS} == {
        "pep_function",
        "pep_protein_interaction",
        "pep_hlai",
        "pep_hlaii",
    }
    assert (
        next(t["label"] for t in TASKS if t["id"] == "pep_hlai")
        == "Peptide–HLA-I ligand presentation prediction"
    )


@pytest.mark.parametrize("columns", ["hla", "mhc", "hla,mhc"])
def test_csv_aliases_preserve_resolved_rows_contexts_and_cache_identity(tmp_path, columns):
    allele, receptor = "HLA-A*02:01", "A" * 30
    path = tmp_path / "pairs.csv"
    fields = ["sample_id", *columns.split(","), "pep", "tgt"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "s1",
                "pep": "SIINFEKL",
                "tgt": 1,
                **{k: allele for k in columns.split(",")},
            }
        )
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    rows, contexts = pipeline.read_ppi_csv(
        path, hla_pseudo_map={allele: receptor}, return_contexts=True
    )
    legacy_rows, legacy_contexts = pipeline.read_ppi_csv(
        path, mhc_pseudo_map={allele: receptor}, return_contexts=True
    )
    assert rows == legacy_rows == [("s1", "SIINFEKL", receptor, 1)]
    assert contexts == legacy_contexts
    base = dict(tokenizer="kmer", encoder="onehot", max_length=30, _feature_train_sha256="a" * 64)
    assert pipeline._feature_cache_meta(
        Namespace(task_type="pep_hlai", **base), "train", "partner", [receptor]
    ) == pipeline._feature_cache_meta(
        Namespace(task_type="pep_mhci", **base), "train", "partner", [receptor]
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_conflicting_columns_fail_and_do_not_change_sequence_precedence():
    bad = {"hla": "HLA-A*02:01", "mhc": "HLA-B*07:02", "protein": "A" * 30}
    with pytest.raises(HLAIInputError, match="conflicting hla"):
        hla_allele(bad)
    with pytest.raises(DatasetValidationError, match="conflicting hla"):
        _pair_partner(bad)
    # The Web sequence view historically prefers an explicit pseudo-sequence.
    for field in ("hla", "mhc"):
        assert (
            _pair_partner(
                {field: "HLA-A*02:01", "pair_pseudo": "C" * 30}, {"HLA-A*02:01": "A" * 30}
            )
            == "C" * 30
        )


def test_existing_dataset_roots_are_read_without_migration(tmp_path):
    current = tmp_path / "datasets/task_peptide_HLAI_interaction"
    historical = tmp_path / "datasets/task_peptide_MHCI_interaction"
    assert resolve_hlai_root(tmp_path) == current
    historical.mkdir(parents=True)
    assert resolve_hlai_root(tmp_path) == historical
    assert not current.exists()
    current.mkdir()
    with pytest.raises(ValueError, match="specify task_root"):
        resolve_hlai_root(tmp_path)


def test_historical_database_rows_are_normalized_without_rewriting(tmp_path):
    store = MetadataStore(tmp_path / "metadata.sqlite3")
    data = store.upsert_dataset(
        "builtin-mhci", "historical", "pep_hlai", "builtin", "pair_split_dir", tmp_path
    )
    with store.connect() as conn:
        conn.execute(
            update(datasets).where(datasets.c.id == data["id"]).values(task_type="pep_mhci")
        )
    assert store.get_dataset("builtin-hlai")["id"] == "builtin-mhci"
    assert store.get_dataset("builtin-hlai")["task_type"] == "pep_hlai"
    with store.connect() as conn:
        assert conn.execute(select(datasets.c.task_type)).scalar_one() == "pep_mhci"
    fresh = store.create_dataset("new", "pep_mhci", "upload", "pair_split_dir", tmp_path)
    with store.connect() as conn:
        assert (
            conn.execute(
                select(datasets.c.task_type).where(datasets.c.id == fresh["id"])
            ).scalar_one()
            == "pep_hlai"
        )


def test_historical_recipe_and_sample_identity_are_preserved(tmp_path):
    from pepplatform.public_data.benchmarks import sample_key
    from pepplatform.public_data.common import canonical

    args = ["pep_mhci", "SIINFEKL", "HLA-A*02:01"]
    assert sample_key(*args) == hashlib.sha256(canonical(args)).hexdigest()


def test_column_resolution_does_not_consume_random_state():
    import random
    import torch

    before = random.getstate(), np.random.get_state(), torch.get_rng_state()
    assert hla_allele({"hla": "HLA-A*02:01"}) == "HLA-A*02:01"
    assert random.getstate() == before[0]
    after = np.random.get_state()
    assert np.array_equal(after[1], before[1][1]) and after[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])


@pytest.mark.parametrize(
    "command,column", [("decode_hlai_pseudoseq.py", "hla"), ("decode_mhci_pseudoseq.py", "mhc")]
)
@pytest.mark.parametrize("input_column", ["hla", "mhc"])
def test_decoder_commands_preserve_new_and_historical_output_formats(
    tmp_path, command, column, input_column
):
    from pathlib import Path
    import subprocess
    import sys

    source, output = tmp_path / "onehot.csv", tmp_path / "decoded.csv"
    source.write_text(f"{input_column},1_A,1_C,2_A,2_C\nHLA-A*02:01,1,0,0,1\n")
    tool = Path(__file__).resolve().parents[1] / "tools" / command
    subprocess.run(
        [sys.executable, str(tool), "--input_csv", str(source), "--output_csv", str(output)],
        check=True,
        capture_output=True,
    )
    with output.open() as handle:
        assert list(csv.DictReader(handle)) == [
            {column: "HLA-A*02:01", "pseudoseq": "AC", "status": "ok"}
        ]


def test_versioned_cli_accepts_historical_task_before_choice_validation():
    from pepplatform.versioning.__main__ import parser

    args = parser().parse_args(
        [
            "run",
            "--task",
            "pep_mhci",
            "--config",
            "config.json",
            "--data",
            "data",
            "--format",
            "pair_split_dir",
        ]
    )
    assert args.task == "pep_hlai"


def test_agent_filters_include_both_stored_task_spellings(tmp_path):
    from pepplatform.web.store import agent_campaigns, agent_execution_plans

    store = MetadataStore(tmp_path / "metadata.sqlite3")
    campaigns = [store.create_agent_campaign(None, "pep_hlai", "data", "aupr") for _ in range(2)]
    plans = [
        store.create_agent_execution_plan(None, "", "", "", "pep_hlai", "data", "aupr", [])
        for _ in range(2)
    ]
    with store.connect() as conn:
        conn.execute(
            update(agent_campaigns)
            .where(agent_campaigns.c.id == campaigns[0]["id"])
            .values(task_type="pep_mhci")
        )
        conn.execute(
            update(agent_execution_plans)
            .where(agent_execution_plans.c.id == plans[0]["id"])
            .values(task_type="pep_mhci")
        )
    for task in ("pep_hlai", "pep_mhci"):
        campaign_rows, count = store.list_agent_campaigns(task_type=task)
        assert count == 2 and {row["id"] for row in campaign_rows} == {
            row["id"] for row in campaigns
        }
        plan_rows, count = store.list_agent_execution_plans(task_type=task)
        assert count == 2 and {row["id"] for row in plan_rows} == {row["id"] for row in plans}


@pytest.mark.parametrize("column", ["hla", "mhc"])
def test_dataset_header_and_report_accept_either_column(tmp_path, column):
    from pepplatform.web.dataset_reports import _read_pair_csv
    from pepplatform.web.datasets import _read_pair_header

    path = tmp_path / "pairs.csv"
    path.write_text(f"{column},pep,tgt\nHLA-A*02:01,SIINFEKL,1\n")
    assert len(_read_pair_header(path)) == 1
    issues = []
    _read_pair_csv(path, tmp_path, "train", issues, 5)
    assert not any(issue.get("code") == "missing_column" for issue in issues)


def test_historical_builtin_name_is_only_changed_for_display(tmp_path):
    store = MetadataStore(tmp_path / "metadata.sqlite3")
    name = "Built-in peptide MHC-I BigMHC-EL 100k dataset"
    row = store.upsert_dataset(
        "builtin-mhci", name, "pep_mhci", "builtin", "pair_split_dir", tmp_path
    )
    assert row["name"] == "Built-in peptide HLA-I BigMHC-EL 100k dataset"
    with store.connect() as conn:
        assert conn.execute(select(datasets.c.name)).scalar_one() == name
    uploaded = store.create_dataset(name, "pep_hlai", "upload", "pair_split_dir", tmp_path)
    assert uploaded["name"] == name
