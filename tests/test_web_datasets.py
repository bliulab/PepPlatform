import csv
import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from pepplatform.web.capabilities import builtin_dataset_specs
from pepplatform.web.dataset_reports import build_dataset_report
from pepplatform.web.datasets import (
    DatasetService,
    DatasetValidationError,
    normalize_builtin_pair_root,
    seed_builtin_datasets,
    validate_pair_dir,
    validate_pair_split_dir,
    validate_sequence_dir,
)
from pepplatform.web.model_artifacts import delete_dataset_artifact, is_deletable_dataset_artifact
from pepplatform.web.paths import RuntimePaths
from pepplatform.web.store import MetadataStore


def _write_fasta(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(f">s{idx}\n{seq}\n" for idx, seq in enumerate(rows, start=1))


def _write_labels(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label"])
        writer.writerows([[row] for row in rows])


def _write_hlaii_pairs(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["peptide", "pair_pseudo", "label"])
        writer.writeheader()
        for peptide, pair_pseudo, label in rows:
            writer.writerow({"peptide": peptide, "pair_pseudo": pair_pseudo, "label": label})


def _write_pair_csv(path, rows, fieldnames=("sample_id", "label", "peptide", "protein")):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_pair_split(root, split):
    rows = [
        {"sample_id": f"{split}_p", "label": 1, "peptide": "ACD", "protein": "EFG"},
        {"sample_id": f"{split}_n", "label": 0, "peptide": "HIJ", "protein": "KLM"},
    ]
    _write_pair_csv(root / split / "pairs.csv", rows)
    _write_labels(root / split / "labels.csv", [1, 0])
    _write_fasta(root / split / "seqs.fasta", ["ACDGGGEFG", "HIJGGGKLM"])


def _write_hlaii_split(root, split):
    rows = [("ACDE", "QRSTVWY", 1), ("HIKL", "ACDFGH", 0)]
    _write_hlaii_pairs(root / split / "pairs.csv", rows)
    _write_labels(root / split / "labels.csv", [row[2] for row in rows])
    _write_fasta(root / split / "seqs.fasta", [f"{row[0]}GGG{row[1]}" for row in rows])


def _write_hlai_split(root, split):
    rows = [
        {"mhc": "HLA-A*01:01", "pep": "ACDEFGHIK", "pair_pseudo": "A" * 30, "tgt": 1},
        {"mhc": "HLA-B*07:02", "pep": "LMNPQRSTV", "pair_pseudo": "C" * 30, "tgt": 0},
    ]
    _write_pair_csv(
        root / split / "pairs.csv", rows, fieldnames=("mhc", "pep", "pair_pseudo", "tgt")
    )
    _write_labels(root / split / "labels.csv", [row["tgt"] for row in rows])
    _write_fasta(
        root / split / "seqs.fasta", [f"{row['pep']}GGG{row['pair_pseudo']}" for row in rows]
    )


def test_validate_sequence_dir(tmp_path):
    for split in ["train", "val"]:
        _write_fasta(tmp_path / split / "seqs.fasta", ["ACD", "EFG"])
        _write_labels(tmp_path / split / "labels.csv", [1, 0])
    metadata = validate_sequence_dir(tmp_path)
    assert metadata["splits"] == {"train": 2, "val": 2}
    assert metadata["length_stats"]["overall"]["sequence"]["max"] == 3


def test_dataset_report_summarizes_multilabel_sequence_dir(tmp_path):
    for split in ["train", "val"]:
        _write_fasta(tmp_path / split / "seqs.fasta", ["ACD", "EFGX"])
        (tmp_path / split / "labels.csv").write_text(
            "AMP,AVP\n1,0\n0,1\n",
            encoding="utf-8",
        )
    dataset = {
        "id": "dataset",
        "task_type": "pep_function",
        "format": "sequence_dir",
        "root_path": str(tmp_path),
        "metadata": {},
    }

    report = build_dataset_report(dataset, preview_limit=1)

    assert report["valid"] is True
    assert report["status"] == "warning"
    assert report["summary"]["splits"] == {"train": 2, "val": 2}
    assert report["summary"]["label_mode"] == "multi_label"
    assert report["summary"]["label_columns"] == ["AMP", "AVP"]
    assert report["splits"][0]["label_counts"]["positive"] == {"AMP": 1, "AVP": 1}
    assert report["splits"][0]["preview"][0]["labels"] == ["AMP"]
    assert report["splits"][0]["sequence_lengths"]["max"] == 4
    assert "missing_optional_test_split" in [issue["code"] for issue in report["issues"]]


def test_dataset_report_flags_pair_split_mismatches(tmp_path):
    _write_pair_csv(
        tmp_path / "train" / "pairs.csv",
        [
            {"sample_id": "p1", "label": 1, "peptide": "ACD", "protein": "EFG"},
            {"sample_id": "p2", "label": 0, "peptide": "HIJ", "protein": "KLM"},
        ],
    )
    _write_labels(tmp_path / "train" / "labels.csv", [1])
    _write_fasta(tmp_path / "train" / "seqs.fasta", ["ACDGGGEFG", "HIJGGGKLM"])
    _write_pair_split(tmp_path, "val")
    dataset = {
        "id": "dataset",
        "task_type": "pep_protein_interaction",
        "format": "pair_split_dir",
        "root_path": str(tmp_path),
        "metadata": {},
    }

    report = build_dataset_report(dataset)

    assert report["valid"] is False
    assert report["status"] == "error"
    assert any(
        issue["code"] == "row_count_mismatch" and issue.get("split") == "train"
        for issue in report["issues"]
    )
    assert report["splits"][0]["preview"][0]["peptide"] == "ACD"
    assert report["splits"][0]["label_counts"]["values"] == {"1": 1}


def test_validate_pair_dir(tmp_path):
    for split in ["train", "val"]:
        with open(tmp_path / f"{split}.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["peptide", "protein", "label"])
            writer.writeheader()
            writer.writerow({"peptide": "ACD", "protein": "EFG", "label": 1})
    metadata = validate_pair_dir(tmp_path)
    assert metadata["splits"] == {"train": 1, "val": 1}


def test_validate_sequence_dir_rejects_mismatched_counts(tmp_path):
    _write_fasta(tmp_path / "train" / "seqs.fasta", ["ACD"])
    _write_labels(tmp_path / "train" / "labels.csv", [1, 0])
    _write_fasta(tmp_path / "val" / "seqs.fasta", ["EFG"])
    _write_labels(tmp_path / "val" / "labels.csv", [1])
    with pytest.raises(DatasetValidationError):
        validate_sequence_dir(tmp_path)


def test_validate_hlaii_sequence_dir_requires_pair_files(tmp_path):
    for split in ["train", "val"]:
        _write_fasta(tmp_path / split / "seqs.fasta", ["ACD", "EFG"])
        _write_labels(tmp_path / split / "labels.csv", [1, 0])
    with pytest.raises(DatasetValidationError, match="pairs.csv"):
        validate_sequence_dir(tmp_path, require_pairs=True)


def test_validate_hlaii_sequence_dir_accepts_matching_pair_files(tmp_path):
    for split in ["train", "val"]:
        _write_fasta(tmp_path / split / "seqs.fasta", ["ACD", "EFG"])
        _write_labels(tmp_path / split / "labels.csv", [1, 0])
        _write_hlaii_pairs(
            tmp_path / split / "pairs.csv",
            [("ACD", "AAAA", 1), ("EFG", "BBBB", 0)],
        )
    metadata = validate_sequence_dir(tmp_path, require_pairs=True)
    assert metadata["splits"] == {"train": 2, "val": 2}


def test_validate_pair_split_dir(tmp_path):
    for split in ["train", "val"]:
        _write_pair_csv(
            tmp_path / split / "pairs.csv",
            [
                {"sample_id": f"{split}_1", "label": 1, "peptide": "ACD", "protein": "EFG"},
                {"sample_id": f"{split}_2", "label": 0, "peptide": "HIJ", "protein": "KLM"},
            ],
        )
        _write_labels(tmp_path / split / "labels.csv", [1, 0])
        _write_fasta(tmp_path / split / "seqs.fasta", ["ACDGGGEFG", "HIJGGGKLM"])

    metadata = validate_pair_split_dir(tmp_path)

    assert metadata["splits"] == {"train": 2, "val": 2}
    assert metadata["has_sequence_view"] is True
    assert metadata["length_stats"]["overall"]["peptide"]["max"] == 3
    assert metadata["length_stats"]["overall"]["partner"]["max"] == 3


def test_normalize_builtin_pair_root_splits_ppi_train_when_val_missing(tmp_path):
    project_root = tmp_path / "project"
    source_root = project_root / "datasets" / "task_peptide_protein_interaction"
    rows = []
    for index in range(10):
        rows.append({"sample_id": f"p{index}", "label": 1, "peptide": "ACD", "protein": "EFG"})
        rows.append({"sample_id": f"n{index}", "label": 0, "peptide": "HIJ", "protein": "KLM"})
    _write_pair_csv(source_root / "processed" / "train.csv", rows)
    _write_pair_csv(source_root / "processed" / "test_1440.csv", rows[:4])
    paths = RuntimePaths.from_project_root(project_root, tmp_path / "runtime").ensure()

    root, metadata = normalize_builtin_pair_root(
        {
            "id": "builtin-ppi",
            "task_type": "pep_protein_interaction",
            "root_path": source_root,
        },
        paths,
    )

    assert root == paths.datasets / "builtin-ppi"
    assert metadata["splits"] == {"train": 18, "val": 2, "test": 4}
    assert metadata["source_root"] == str(source_root)
    assert (root / "train" / "pairs.csv").exists()
    assert (root / "val" / "pairs.csv").exists()
    assert (root / "test" / "pairs.csv").exists()


def test_builtin_dataset_specs_include_full_and_quick_ppi(tmp_path):
    project_root = tmp_path / "project"
    ppi_root = project_root / "datasets" / "task_peptide_protein_interaction"

    specs = builtin_dataset_specs(project_root)
    by_id = {spec["id"]: spec for spec in specs}

    assert by_id["builtin-ppi"]["root_path"] == ppi_root
    assert by_id["builtin-ppi"]["format"] == "pair_root"
    assert by_id["builtin-ppi-quick"]["root_path"] == ppi_root / "data_quick_2000"
    assert by_id["builtin-ppi-quick"]["format"] == "pair_split_dir"
    assert by_id["builtin-ppi-quick"]["feature_dir"] == ppi_root / "data_quick_2000" / "features"


def test_builtin_dataset_specs_include_quick_and_full_pep_function(tmp_path):
    project_root = tmp_path / "project"
    task_root = project_root / "datasets" / "task_peptide_function"

    specs = builtin_dataset_specs(project_root)
    by_id = {spec["id"]: spec for spec in specs}

    assert by_id["builtin-pep-function-quick"]["name"] == (
        "Built-in peptide function quick dataset"
    )
    assert by_id["builtin-pep-function-quick"]["root_path"] == (task_root / "data_quick_384")
    assert by_id["builtin-pep-function-quick"]["format"] == "sequence_dir"
    assert by_id["builtin-pep-function"]["name"] == ("Built-in peptide function full dataset")
    assert by_id["builtin-pep-function"]["root_path"] == task_root / "data"


def test_seed_builtin_datasets_registers_pep_function_quick_and_full(tmp_path):
    project_root = tmp_path / "project"
    task_root = project_root / "datasets" / "task_peptide_function"
    for root in [task_root / "data_quick_384", task_root / "data"]:
        for split in ["train", "val", "test"]:
            _write_fasta(root / split / "seqs.fasta", ["ACD", "EFG"])
            _write_labels(root / split / "labels.csv", [1, 0])
    paths = RuntimePaths.from_project_root(project_root, tmp_path / "runtime").ensure()
    store = MetadataStore(tmp_path / "metadata.sqlite")

    seed_builtin_datasets(store, paths)
    datasets = {dataset["id"]: dataset for dataset in store.list_datasets()}

    assert datasets["builtin-pep-function-quick"]["root_path"] == str(task_root / "data_quick_384")
    assert datasets["builtin-pep-function-quick"]["metadata"]["splits"] == {
        "train": 2,
        "val": 2,
        "test": 2,
    }
    assert datasets["builtin-pep-function"]["root_path"] == str(task_root / "data")


def test_tppred_benchmark_is_optional_and_registers_without_changing_existing_data(tmp_path):
    project_root = tmp_path / "project"
    task_root = project_root / "datasets" / "task_peptide_function"
    original_root = task_root / "data"
    for split in ["train", "val", "test"]:
        _write_fasta(original_root / split / "seqs.fasta", ["ACD", "EFG"])
        _write_labels(original_root / split / "labels.csv", [1, 0])
    paths = RuntimePaths.from_project_root(project_root, tmp_path / "runtime").ensure()
    store = MetadataStore(tmp_path / "metadata.sqlite")
    seed_builtin_datasets(store, paths)
    assert store.get_dataset("builtin-tppred-le-benchmark") is None
    original = store.get_dataset("builtin-pep-function")

    benchmark_root = task_root / "data_tppred_benchmark_public"
    label_names = [
        "AMP",
        "TXP",
        "ABP",
        "AIP",
        "AVP",
        "ACP",
        "AFP",
        "DDV",
        "CPP",
        "CCC",
        "APP",
        "AAP",
        "AHTP",
        "PBP",
        "QSP",
    ]
    for split, sequences in {
        "train": ["ACD", "EFG", "HIK"],
        "val": ["LMN", "PQR"],
        "test": ["STV"],
    }.items():
        _write_fasta(benchmark_root / split / "seqs.fasta", sequences)
        with (benchmark_root / split / "labels.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(label_names)
            writer.writerows([[1, 0, 1] + [0] * 12 for _ in sequences])
    source_bytes = {path: path.read_bytes() for path in task_root.rglob("*") if path.is_file()}

    seed_builtin_datasets(store, paths)
    seed_builtin_datasets(store, paths)
    datasets = {dataset["id"]: dataset for dataset in store.list_datasets()}
    assert len(datasets) == 2
    benchmark = datasets["builtin-tppred-le-benchmark"]
    assert benchmark["name"] == "TPpred-LE benchmark"
    assert benchmark["task_type"] == "pep_function"
    assert benchmark["source"] == "builtin"
    assert benchmark["format"] == "sequence_dir"
    assert benchmark["root_path"] == str(benchmark_root)
    assert benchmark["metadata"]["splits"] == {"train": 3, "val": 2, "test": 1}
    assert not benchmark["metadata"]["managed"]
    assert datasets["builtin-pep-function"]["root_path"] == original["root_path"]
    assert datasets["builtin-pep-function"]["name"] == original["name"]
    assert all(path.read_bytes() == content for path, content in source_bytes.items())
    with (benchmark_root / "train/labels.csv").open(newline="") as handle:
        assert next(csv.reader(handle)) == label_names


def test_builtin_dataset_specs_include_hlaii_quick_and_full(tmp_path):
    project_root = tmp_path / "project"
    hlaii_root = project_root / "datasets" / "task_peptide_HLAII_interaction"

    specs = builtin_dataset_specs(project_root)
    by_id = {spec["id"]: spec for spec in specs}

    assert by_id["builtin-pep-hlaii"]["name"] == "Built-in peptide HLA-II quick dataset"
    assert by_id["builtin-pep-hlaii"]["root_path"] == hlaii_root / "data_quick_50k"
    assert by_id["builtin-pep-hlaii"]["format"] == "pair_split_dir"
    assert by_id["builtin-pep-hlaii-full"]["name"] == "Built-in peptide HLA-II full dataset"
    assert by_id["builtin-pep-hlaii-full"]["root_path"] == hlaii_root / "data"
    assert by_id["builtin-pep-hlaii-full"]["format"] == "pair_split_dir"


def test_builtin_dataset_specs_include_hlai_bigmhc_100k(tmp_path):
    project_root = tmp_path / "project"
    hlai_root = project_root / "datasets" / "task_peptide_HLAI_interaction"

    specs = builtin_dataset_specs(project_root)
    by_id = {spec["id"]: spec for spec in specs}

    assert by_id["builtin-hlai"]["name"] == "Built-in peptide HLA-I BigMHC-EL 100k dataset"
    assert by_id["builtin-hlai"]["root_path"] == hlai_root / "data_bigmhc_el_100k_original_split"
    assert by_id["builtin-hlai"]["format"] == "pair_split_dir"
    assert (
        by_id["builtin-hlai"]["feature_dir"]
        == hlai_root / "data_bigmhc_el_100k_original_split" / "features"
    )


def test_seed_builtin_datasets_registers_ppi_quick_with_isolated_feature_dir(tmp_path):
    project_root = tmp_path / "project"
    ppi_root = project_root / "datasets" / "task_peptide_protein_interaction"
    for root in [ppi_root / "data", ppi_root / "data_quick_2000"]:
        for split in ["train", "val", "test"]:
            _write_pair_split(root, split)
    paths = RuntimePaths.from_project_root(project_root, tmp_path / "runtime").ensure()
    store = MetadataStore(tmp_path / "metadata.sqlite")

    seed_builtin_datasets(store, paths)
    datasets = {dataset["id"]: dataset for dataset in store.list_datasets()}

    assert datasets["builtin-ppi"]["root_path"] == str(ppi_root / "data")
    assert datasets["builtin-ppi-quick"]["root_path"] == str(ppi_root / "data_quick_2000")
    assert datasets["builtin-ppi-quick"]["metadata"]["splits"] == {"train": 2, "val": 2, "test": 2}
    assert datasets["builtin-ppi-quick"]["metadata"]["feature_dir"] == str(
        ppi_root / "data_quick_2000" / "features"
    )


def test_seed_builtin_datasets_registers_hlaii_quick_and_full(tmp_path):
    project_root = tmp_path / "project"
    hlaii_root = project_root / "datasets" / "task_peptide_HLAII_interaction"
    for root in [hlaii_root / "data_quick_50k", hlaii_root / "data"]:
        for split in ["train", "val", "test"]:
            _write_hlaii_split(root, split)
    paths = RuntimePaths.from_project_root(project_root, tmp_path / "runtime").ensure()
    store = MetadataStore(tmp_path / "metadata.sqlite")

    seed_builtin_datasets(store, paths)
    datasets = {dataset["id"]: dataset for dataset in store.list_datasets()}

    assert datasets["builtin-pep-hlaii"]["name"] == "Built-in peptide HLA-II quick dataset"
    assert datasets["builtin-pep-hlaii"]["root_path"] == str(hlaii_root / "data_quick_50k")
    assert datasets["builtin-pep-hlaii"]["metadata"]["splits"] == {"train": 2, "val": 2, "test": 2}
    assert datasets["builtin-pep-hlaii"]["metadata"]["has_sequence_view"] is True
    assert datasets["builtin-pep-hlaii-full"]["name"] == "Built-in peptide HLA-II full dataset"
    assert datasets["builtin-pep-hlaii-full"]["root_path"] == str(hlaii_root / "data")
    assert datasets["builtin-pep-hlaii-full"]["metadata"]["splits"] == {
        "train": 2,
        "val": 2,
        "test": 2,
    }
    assert datasets["builtin-pep-hlaii-full"]["metadata"]["has_sequence_view"] is True


def test_seed_builtin_datasets_registers_hlai_bigmhc_100k(tmp_path):
    project_root = tmp_path / "project"
    hlai_root = project_root / "datasets" / "task_peptide_HLAI_interaction"
    data_root = hlai_root / "data_bigmhc_el_100k_original_split"
    for split in ["train", "val", "test"]:
        _write_hlai_split(data_root, split)
    paths = RuntimePaths.from_project_root(project_root, tmp_path / "runtime").ensure()
    store = MetadataStore(tmp_path / "metadata.sqlite")

    seed_builtin_datasets(store, paths)
    dataset = store.get_dataset("builtin-hlai")

    assert dataset["name"] == "Built-in peptide HLA-I BigMHC-EL 100k dataset"
    assert dataset["root_path"] == str(data_root)
    assert dataset["metadata"]["splits"] == {"train": 2, "val": 2, "test": 2}
    assert dataset["metadata"]["has_sequence_view"] is True
    assert dataset["metadata"]["source_root"] == str(hlai_root)
    assert dataset["metadata"]["sequence_view"] == "peptide_GGG_partner"
    assert dataset["metadata"]["feature_dir"] == str(data_root / "features")


def test_delete_dataset_artifact_only_removes_managed_upload_root(tmp_path):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    dataset_root = paths.datasets / "dataset-1"
    inner_root = dataset_root / "archive-root"
    inner_root.mkdir(parents=True)
    (inner_root / "train.csv").write_text("peptide,label\nACD,1\n", encoding="utf-8")
    dataset = {
        "id": "dataset-1",
        "source": "upload",
        "root_path": str(inner_root),
    }

    assert is_deletable_dataset_artifact(inner_root, "dataset-1", paths) is True
    assert delete_dataset_artifact(dataset, paths) is True
    assert not dataset_root.exists()


def test_delete_dataset_artifact_refuses_external_or_builtin_paths(tmp_path):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    external_root = tmp_path / "project" / "datasets" / "external"
    external_root.mkdir(parents=True)
    external_dataset = {
        "id": "dataset-1",
        "source": "upload",
        "root_path": str(external_root),
    }
    builtin_root = paths.datasets / "builtin-pep-function"
    builtin_root.mkdir(parents=True)
    builtin_dataset = {
        "id": "builtin-pep-function",
        "source": "builtin",
        "root_path": str(builtin_root),
    }

    assert is_deletable_dataset_artifact(external_root, "dataset-1", paths) is False
    assert delete_dataset_artifact(external_dataset, paths) is False
    assert external_root.exists()
    assert delete_dataset_artifact(builtin_dataset, paths) is False
    assert builtin_root.exists()


@pytest.fixture
def upload_service(tmp_path):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    return DatasetService(MetadataStore(tmp_path / "metadata.sqlite"), paths)


def _upload_files(service, task, files, wrapped=False):
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as archive:
        for name, data in files.items():
            archive.writestr(("bundle/" if wrapped else "") + name, data)
    return service.upload_archive(task, "Uploaded table", "dataset.zip", content.getvalue())


def _as_tsv(data):
    rows = csv.reader(io.StringIO(data.decode("utf-8"), newline=""))
    output = io.StringIO(newline="")
    csv.writer(output, delimiter="\t").writerows(rows)
    return output.getvalue().encode("utf-8")


def _input_files(root, task, flat, with_test):
    for split in ["train", "val", "test"] if with_test else ["train", "val"]:
        if task == "pep_function":
            _write_fasta(root / split / "seqs.fasta", ["ACD", "EFG"])
            (root / split / "labels.csv").write_bytes(b"AMP,AVP\r\n1,0\r\n0,1\r\n")
        elif task == "pep_hlaii":
            _write_hlaii_split(root, split)
        elif task == "pep_hlai":
            _write_hlai_split(root, split)
        else:
            _write_pair_split(root, split)
    files = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    if flat:
        return {
            name.split("/")[0] + ".csv": data
            for name, data in files.items()
            if name.endswith("pairs.csv")
        }
    return files


@pytest.mark.parametrize(
    "task,flat",
    [
        ("pep_function", False),
        ("pep_protein_interaction", False),
        ("pep_hlai", False),
        ("pep_hlaii", False),
        ("pep_protein_interaction", True),
        ("pep_hlai", True),
    ],
)
@pytest.mark.parametrize("with_test", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_tsv_upload_matches_csv_through_report(
    upload_service, tmp_path, task, flat, with_test, mixed
):
    originals = _input_files(tmp_path / "input", task, flat, with_test)
    adapted = {
        (
            name[:-4] + ".tsv"
            if name.endswith(".csv") and not (mixed and name.startswith("val"))
            else name
        ): (
            _as_tsv(data)
            if name.endswith(".csv") and not (mixed and name.startswith("val"))
            else data
        )
        for name, data in originals.items()
    }
    csv_dataset = _upload_files(upload_service, task, originals)
    tsv_dataset = _upload_files(upload_service, task, adapted, wrapped=True)
    csv_root, tsv_root = Path(csv_dataset["root_path"]), Path(tsv_dataset["root_path"])
    for canonical in csv_root.rglob("*"):
        if canonical.is_file():
            assert (
                tsv_root / canonical.relative_to(csv_root)
            ).read_bytes() == canonical.read_bytes()
    # The archive originals survive conversion, including for flat bundles.
    upload_root = tsv_root.parent if flat else tsv_root
    for name, data in adapted.items():
        assert (upload_root / name).read_bytes() == data
    assert tsv_dataset["format"] == csv_dataset["format"]
    assert tsv_dataset["metadata"]["splits"] == csv_dataset["metadata"]["splits"]
    assert tsv_dataset["metadata"]["length_stats"] == csv_dataset["metadata"]["length_stats"]
    reports = [build_dataset_report(dataset) for dataset in (csv_dataset, tsv_dataset)]
    for report in reports:
        assert report["valid"] is True
        report.pop("dataset_id")
        report.pop("generated_at")
    assert reports[0] == reports[1]


def test_upload_keeps_csv_bytes_and_ignores_conflicting_tsv(upload_service, tmp_path):
    files = _input_files(tmp_path / "input", "pep_protein_interaction", False, False)
    # Tabs in CSV headers/fields are legal and must not trigger TSV detection.
    files["train/pairs.csv"] = (
        b'peptide,protein,label,"note\tname"\nACD,EFG,1,"a\tb"\nHIJ,KLM,0,x\n'
    )
    files.update(
        {name[:-4] + ".tsv": b'"unterminated\t' for name in list(files) if name.endswith(".csv")}
    )
    dataset = _upload_files(upload_service, "pep_protein_interaction", files)
    for name, data in files.items():
        assert (Path(dataset["root_path"]) / name).read_bytes() == data


def test_tsv_conversion_preserves_cells_and_ignores_unrelated_tables(upload_service, tmp_path):
    files = _input_files(tmp_path / "input", "pep_protein_interaction", True, False)
    rows = [
        ["peptide", "protein", "label", "note\theader"],
        ["ACD", "EFG", "1", 'comma, tab\t and "quote"'],
        ["HIJ", "KLM", "0", "line one\nline two"],
    ]
    output = io.StringIO(newline="")
    csv.writer(output, delimiter="\t").writerows(rows)
    files.pop("train.csv")
    files["train.tsv"] = output.getvalue().encode()
    files["notes.tsv"] = b'"invalid'
    dataset = _upload_files(upload_service, "pep_protein_interaction", files)
    root = Path(dataset["root_path"])
    with (root / "train" / "pairs.csv").open(newline="") as handle:
        assert list(csv.reader(handle)) == rows
    assert not (root.parent / "notes.csv").exists()


def test_invalid_csv_does_not_fall_back_to_valid_tsv(upload_service, tmp_path):
    files = _input_files(tmp_path / "input", "pep_function", False, False)
    files["train/labels.tsv"] = _as_tsv(files["train/labels.csv"])
    files["train/labels.csv"] = b"AMP,AVP\n"
    with pytest.raises(DatasetValidationError, match="header and at least one row"):
        _upload_files(upload_service, "pep_function", files)


def test_flat_tsv_upload_requires_validation_split(upload_service):
    files = {"train.tsv": b"peptide\tprotein\tlabel\nACD\tEFG\t1\n"}
    with pytest.raises(DatasetValidationError, match="missing val.csv"):
        _upload_files(upload_service, "pep_protein_interaction", files)


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("missing_labels", "labels.csv"),
        ("missing_pairs", "pairs.csv"),
        ("missing_sequence", "seqs.fasta"),
        ("missing_columns", "must include"),
        ("count_mismatch", "pair count does not match label count"),
        ("empty_labels", "header and at least one row"),
        ("bad_quotes", "invalid TSV file"),
        ("bad_encoding", "invalid TSV file"),
    ],
)
def test_invalid_tsv_upload_uses_validation_and_cleans_up(
    upload_service, tmp_path, failure, expected
):
    originals = _input_files(tmp_path / "input", "pep_hlaii", False, False)
    files = {
        (name[:-4] + ".tsv" if name.endswith(".csv") else name): (
            _as_tsv(data) if name.endswith(".csv") else data
        )
        for name, data in originals.items()
    }
    if failure == "missing_labels":
        files.pop("train/labels.tsv")
    elif failure == "missing_pairs":
        files.pop("train/pairs.tsv")
    elif failure == "missing_sequence":
        files.pop("train/seqs.fasta")
    elif failure == "missing_columns":
        files["train/pairs.tsv"] = b"peptide\tlabel\nACD\t1\nEFG\t0\n"
    elif failure == "count_mismatch":
        files["train/labels.tsv"] = b"label\n1\n"
    elif failure == "empty_labels":
        files["train/labels.tsv"] = b"label\n"
    elif failure == "bad_quotes":
        files["train/pairs.tsv"] = b'peptide\tprotein\tlabel\n"unclosed'
    else:
        files["train/pairs.tsv"] = b"\xff"
    with pytest.raises(DatasetValidationError, match=expected):
        _upload_files(upload_service, "pep_hlaii", files)
    assert list(upload_service.paths.datasets.iterdir()) == []
    assert upload_service.store.list_datasets() == []


def _txt_files(originals, delimiter="\t", mixed=False):
    files = {}
    for name, data in originals.items():
        if mixed and name.startswith("val"):
            if name.endswith(".csv"):
                files[name[:-4] + ".tsv"] = _as_tsv(data)
            else:
                files[name] = data
            continue
        if name.endswith(".csv"):
            files[name[:-4] + ".txt"] = b"\xef\xbb\xbf" + (
                _as_tsv(data) if delimiter == "\t" else data
            )
        elif name.endswith(".fasta"):
            sequences = [line for line in data.decode().splitlines() if not line.startswith(">")]
            files[name[:-6] + ".txt"] = ("\ufeff" + "\r\n".join(sequences) + "\r\n").encode()
        else:
            files[name] = data
    return files


@pytest.mark.parametrize(
    "task,flat",
    [
        ("pep_function", False),
        ("pep_protein_interaction", False),
        ("pep_hlai", False),
        ("pep_hlaii", False),
        ("pep_protein_interaction", True),
        ("pep_hlai", True),
    ],
)
@pytest.mark.parametrize("with_test", [False, True])
@pytest.mark.parametrize("delimiter,mixed", [(",", False), ("\t", False), ("\t", True)])
def test_txt_upload_matches_canonical_files_and_report(
    upload_service, tmp_path, task, flat, with_test, delimiter, mixed
):
    originals = _input_files(tmp_path / "input", task, flat, with_test)
    files = _txt_files(originals, delimiter, mixed)
    canonical_dataset = _upload_files(upload_service, task, originals)
    txt_dataset = _upload_files(upload_service, task, files, wrapped=True)
    canonical_root, txt_root = Path(canonical_dataset["root_path"]), Path(txt_dataset["root_path"])
    for canonical in canonical_root.rglob("*"):
        if canonical.is_file():
            assert (
                txt_root / canonical.relative_to(canonical_root)
            ).read_bytes() == canonical.read_bytes()
    upload_root = txt_root.parent if flat else txt_root
    for name, data in files.items():
        assert (upload_root / name).read_bytes() == data
    provenance = txt_dataset["metadata"]["upload_conversion"]
    assert provenance["version"] == 1
    assert provenance["path_base"] == "archive_root"
    assert len(provenance["files"]) == sum(name.endswith(".txt") for name in files)
    for entry in provenance["files"]:
        assert entry["source"].startswith("bundle/")
        for key in ("source", "target"):
            path = upload_root.parent / entry[key]
            assert entry[key + "_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert txt_dataset["format"] == canonical_dataset["format"]
    assert txt_dataset["metadata"]["splits"] == canonical_dataset["metadata"]["splits"]
    assert txt_dataset["metadata"]["length_stats"] == canonical_dataset["metadata"]["length_stats"]
    reports = [build_dataset_report(dataset) for dataset in (canonical_dataset, txt_dataset)]
    for report in reports:
        assert report["valid"] is True
        report.pop("dataset_id")
        report.pop("generated_at")
    assert reports[0] == reports[1]


@pytest.mark.parametrize("delimiter", [",", "\t"])
def test_txt_table_preserves_quoted_cells_and_multiline_headers(
    upload_service, tmp_path, delimiter
):
    files = _input_files(tmp_path / "input", "pep_protein_interaction", True, False)
    rows = [
        ["peptide", "protein", "label", 'note,\t"header"\ncontinued'],
        ["ACD", "EFG", "1", 'comma, tab\t and "quote"'],
        ["HIJ", "KLM", "0", "line one\nline two"],
    ]
    output = io.StringIO(newline="")
    csv.writer(output, delimiter=delimiter).writerows(rows)
    files.pop("train.csv")
    files["train.txt"] = output.getvalue().encode()
    files["notes.txt"] = b'"invalid'
    dataset = _upload_files(upload_service, "pep_protein_interaction", files)
    root = Path(dataset["root_path"])
    with (root / "train" / "pairs.csv").open(newline="") as handle:
        assert list(csv.reader(handle)) == rows
    assert not (root.parent / "notes.csv").exists()
    assert (root.parent / "notes.txt").read_bytes() == files["notes.txt"]


@pytest.mark.parametrize("table_format", ["csv", "tsv"])
def test_existing_tables_and_fasta_win_over_invalid_txt(upload_service, tmp_path, table_format):
    originals = _input_files(tmp_path / "input", "pep_function", False, False)
    files = dict(originals)
    if table_format == "tsv":
        files = {
            (name[:-4] + ".tsv" if name.endswith(".csv") else name): (
                _as_tsv(data) if name.endswith(".csv") else data
            )
            for name, data in files.items()
        }
    files.update({name.rsplit(".", 1)[0] + ".txt": b"\xff" for name in originals})
    dataset = _upload_files(upload_service, "pep_function", files)
    assert "upload_conversion" not in dataset["metadata"]
    for name, data in originals.items():
        assert (Path(dataset["root_path"]) / name).read_bytes() == data


@pytest.mark.parametrize("extension", ["csv", "tsv", "fasta"])
def test_invalid_preferred_file_does_not_fall_back_to_txt(upload_service, tmp_path, extension):
    originals = _input_files(tmp_path / "input", "pep_function", False, False)
    files = _txt_files(originals)
    if extension == "fasta":
        files["train/seqs.fasta"] = b">only_one\nACD\n"
    else:
        files["train/labels." + extension] = b"AMP\n"
    with pytest.raises(DatasetValidationError):
        _upload_files(upload_service, "pep_function", files)
    assert list(upload_service.paths.datasets.iterdir()) == []
    assert upload_service.store.list_datasets() == []


def test_txt_sequences_preserve_case_order_and_duplicates(upload_service, tmp_path):
    files = _txt_files(_input_files(tmp_path / "input", "pep_function", False, False))
    files["train/seqs.txt"] = b"  aCd  \naCd\n"
    dataset = _upload_files(upload_service, "pep_function", files)
    root = Path(dataset["root_path"])
    assert (root / "train/seqs.fasta").read_bytes() == b">s1\naCd\n>s2\naCd\n"
    assert (root / "train/labels.csv").read_bytes() == b"AMP,AVP\r\n1,0\r\n0,1\r\n"


@pytest.mark.parametrize(
    "filename,content,expected",
    [
        ("train/labels.txt", None, "labels.csv"),
        ("train/pairs.txt", None, "pairs.csv"),
        ("train/seqs.txt", None, "seqs.fasta"),
        ("train/pairs.txt", b"peptide\tlabel\nACD\t1\nEFG\t0\n", "must include"),
        ("train/labels.txt", b"label\n1\n", "pair count does not match label count"),
        ("train/labels.txt", b"label\n", "header and at least one row"),
        ("train/labels.txt", b"", "nonempty header"),
        ("train/pairs.txt", b'peptide\tprotein\tlabel\n"unclosed', "invalid TXT file"),
        ("train/pairs.txt", b"\xff", "invalid TXT file"),
        ("train/pairs.txt", b"peptide,protein\tlabel\nACD,EFG\t1\n", "ambiguous header"),
        ("train/pairs.txt", b"peptide,protein,label\nACD,EFG\n", "line 2: cell count"),
        ("train/seqs.txt", b"ACD\n", "sequence count does not match pair count"),
        ("train/seqs.txt", b"ACD\n\nEFG\n", "line 2: expected"),
        ("train/seqs.txt", b"AC D\nEFG\n", "line 1: expected"),
        ("train/seqs.txt", b">s1\nACD\n", "use .fasta"),
        ("train/seqs.txt", b"ACD1\nEFG\n", "line 1: expected"),
        ("train/seqs.txt", "ACé\nEFG\n".encode(), "line 1: expected"),
        ("train/seqs.txt", b"\xff", "invalid TXT file"),
        ("train/seqs.txt", b"", "at least one sequence"),
    ],
)
def test_invalid_txt_upload_fails_and_cleans_up(
    upload_service, tmp_path, filename, content, expected
):
    files = _txt_files(_input_files(tmp_path / "input", "pep_hlaii", False, False))
    if content is None:
        files.pop(filename)
    else:
        files[filename] = content
    with pytest.raises(DatasetValidationError, match=expected):
        _upload_files(upload_service, "pep_hlaii", files)
    assert list(upload_service.paths.datasets.iterdir()) == []
    assert upload_service.store.list_datasets() == []


def test_flat_txt_requires_validation_split(upload_service):
    with pytest.raises(DatasetValidationError, match="missing val.csv"):
        _upload_files(
            upload_service,
            "pep_protein_interaction",
            {
                "train.txt": b"peptide\tprotein\tlabel\nACD\tEFG\t1\n",
            },
        )
