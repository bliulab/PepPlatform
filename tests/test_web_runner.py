import sys
import threading
import time
from pathlib import Path

import pytest

from pepplatform.web.paths import RuntimePaths
from pepplatform.web.runner import ExperimentRunner
from pepplatform.web.store import MetadataStore
from scripts import train_pair_pipeline, train_sequence_pipeline


def test_pair_split_dir_build_command_uses_original_task_root(tmp_path):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    source_root = tmp_path / "source-pair-root"
    split_root = tmp_path / "normalized-pair-root"
    source_root.mkdir()
    for split in ["train", "val", "test"]:
        split_dir = split_root / split
        split_dir.mkdir(parents=True)
        (split_dir / "pairs.csv").write_text("peptide,protein,label\nACD,EFG,1\n", encoding="utf-8")
    runner = ExperimentRunner(None, paths)

    command = runner.build_command(
        {
            "task_type": "pep_protein_interaction",
            "config": {"tokenizer": "esm2", "encoder": "ppi_fusion", "model": "mlp"},
            "artifact_dir": str(tmp_path / "artifact"),
        },
        {
            "format": "pair_split_dir",
            "root_path": str(split_root),
            "metadata": {"source_root": str(source_root)},
        },
    )

    assert command[command.index("--task_root") + 1] == str(source_root)
    assert command[command.index("--encoder") + 1] == "esm2"
    assert "ppi_fusion" not in command
    assert command[command.index("--train_csv") + 1] == str(split_root / "train" / "pairs.csv")
    assert command[command.index("--val_csv") + 1] == str(split_root / "val" / "pairs.csv")
    assert command[command.index("--test_csv") + 1] == str(split_root / "test" / "pairs.csv")


def test_pair_split_dir_build_command_uses_dataset_feature_dir(tmp_path):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    source_root = tmp_path / "source-pair-root"
    split_root = tmp_path / "quick-pair-root"
    feature_dir = split_root / "features"
    source_root.mkdir()
    for split in ["train", "val", "test"]:
        split_dir = split_root / split
        split_dir.mkdir(parents=True)
        (split_dir / "pairs.csv").write_text("peptide,protein,label\nACD,EFG,1\n", encoding="utf-8")
    runner = ExperimentRunner(None, paths)

    command = runner.build_command(
        {
            "task_type": "pep_protein_interaction",
            "config": {"tokenizer": "esm2", "encoder": "ppi_fusion", "model": "mlp"},
            "artifact_dir": str(tmp_path / "artifact"),
        },
        {
            "format": "pair_split_dir",
            "root_path": str(split_root),
            "metadata": {"source_root": str(source_root), "feature_dir": str(feature_dir)},
        },
    )

    assert command[command.index("--task_root") + 1] == str(source_root)
    assert command[command.index("--encoder") + 1] == "esm2"
    assert command[command.index("--feature_dir") + 1] == str(feature_dir)


def test_hlaii_pair_split_build_command_uses_pair_pipeline(tmp_path):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    split_root = tmp_path / "hla-root"
    for split in ["train", "val", "test"]:
        split_dir = split_root / split
        split_dir.mkdir(parents=True)
        (split_dir / "pairs.csv").write_text(
            "peptide,pair_pseudo,label\nACD,EFG,1\n", encoding="utf-8"
        )
    runner = ExperimentRunner(None, paths)

    command = runner.build_command(
        {
            "task_type": "pep_hlaii",
            "config": {
                "task_type": "pep_hlaii",
                "tokenizer": "kmer",
                "encoder": "index",
                "model": "cnn",
                "pair_strategy": "single_stream",
                "sequence_model": "cnn",
                "training_mode": "end_to_end",
                "classifier": "mlp",
            },
            "artifact_dir": str(tmp_path / "artifact"),
        },
        {
            "format": "pair_split_dir",
            "root_path": str(split_root),
            "metadata": {},
        },
    )

    assert command[1].endswith("scripts/train_pair_pipeline.py")
    assert command[command.index("--task_type") + 1] == "pep_hlaii"
    assert command[command.index("--task_root") + 1] == str(split_root)
    assert command[command.index("--train_csv") + 1] == str(split_root / "train" / "pairs.csv")
    assert command[command.index("--val_csv") + 1] == str(split_root / "val" / "pairs.csv")
    assert command[command.index("--test_csv") + 1] == str(split_root / "test" / "pairs.csv")


def test_hlaii_sequence_dir_build_command_uses_pairs_csv(tmp_path):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    data_root = tmp_path / "hla-sequence-root"
    for split in ["train", "val"]:
        split_dir = data_root / split
        split_dir.mkdir(parents=True)
        (split_dir / "pairs.csv").write_text(
            "peptide,pair_pseudo,label\nACD,EFG,1\n", encoding="utf-8"
        )
        (split_dir / "seqs.fasta").write_text(">x\nACDGGGEFG\n", encoding="utf-8")
        (split_dir / "labels.csv").write_text("label\n1\n", encoding="utf-8")
    runner = ExperimentRunner(None, paths)

    command = runner.build_command(
        {
            "task_type": "pep_hlaii",
            "config": {
                "task_type": "pep_hlaii",
                "tokenizer": "kmer",
                "encoder": "index",
                "model": "cnn",
            },
            "artifact_dir": str(tmp_path / "artifact"),
        },
        {
            "format": "sequence_dir",
            "root_path": str(data_root),
            "metadata": {},
        },
    )

    assert command[1].endswith("scripts/train_pair_pipeline.py")
    assert command[command.index("--train_csv") + 1] == str(data_root / "train" / "pairs.csv")
    assert command[command.index("--val_csv") + 1] == str(data_root / "val" / "pairs.csv")
    assert "--data_dir" not in command


def test_build_command_passes_pretrained_model_paths(tmp_path):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    data_root = tmp_path / "data"
    data_root.mkdir()
    runner = ExperimentRunner(None, paths)

    command = runner.build_command(
        {
            "task_type": "pep_function",
            "config": {
                "tokenizer": "esm2",
                "encoder": "index",
                "model": "mlp",
                "esm2_model": "models/esm2",
            },
            "artifact_dir": str(tmp_path / "artifact"),
        },
        {
            "format": "sequence_dir",
            "root_path": str(data_root),
            "metadata": {},
        },
    )

    assert command[1].endswith("scripts/train_sequence_pipeline.py")
    assert command[command.index("--esm2_model") + 1] == "models/esm2"


def test_sequence_build_command_isolates_generated_profile_cache_per_experiment(tmp_path):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    data_root = tmp_path / "data"
    data_root.mkdir()
    artifact_dir = tmp_path / "experiments" / "exp-1" / "artifact"
    runner = ExperimentRunner(None, paths)

    command = runner.build_command(
        {
            "task_type": "pep_function",
            "config": {
                "tokenizer": "kmer",
                "encoder": "pssm_blosum62_hybrid",
                "model": "cnn",
            },
            "artifact_dir": str(artifact_dir),
        },
        {
            "format": "sequence_dir",
            "root_path": str(data_root),
            "metadata": {},
        },
    )

    expected = artifact_dir.parent / "profile_cache"
    assert command[command.index("--profile_cache_dir") + 1] == str(expected)
    parsed = train_sequence_pipeline.build_argument_parser().parse_args(command[2:])
    assert parsed.profile_cache_dir == str(expected)


@pytest.mark.parametrize("encoder", ["ac", "cc", "acc"])
def test_web_covariance_sequence_command_is_accepted_by_real_parser(tmp_path, encoder):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    data_root = tmp_path / "data"
    data_root.mkdir()
    runner = ExperimentRunner(None, paths)
    lag_key = f"{encoder}_lag"

    command = runner.build_command(
        {
            "task_type": "pep_function",
            "config": {
                "tokenizer": "kmer",
                "encoder": encoder,
                "model": "mlp",
                "classifier": "mlp",
                "k": 1,
                lag_key: 3,
                "covariance_max_raw_length": 100000,
            },
            "artifact_dir": str(tmp_path / "artifact"),
        },
        {
            "format": "sequence_dir",
            "root_path": str(data_root),
            "metadata": {},
        },
    )

    parsed = train_sequence_pipeline.build_argument_parser().parse_args(command[2:])
    assert parsed.encoder == encoder
    assert getattr(parsed, lag_key) == 3
    assert parsed.covariance_max_raw_length == 100000


@pytest.mark.parametrize("encoder", ["ac", "cc", "acc"])
def test_web_covariance_pair_command_is_accepted_by_real_parser(tmp_path, encoder):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    data_root = tmp_path / "data"
    for split in ["train", "val"]:
        split_root = data_root / split
        split_root.mkdir(parents=True)
        (split_root / "pairs.csv").write_text(
            "peptide,protein,label\nACDE,FGHIK,1\n",
            encoding="utf-8",
        )
    runner = ExperimentRunner(None, paths)
    lag_key = f"{encoder}_lag"

    command = runner.build_command(
        {
            "task_type": "pep_protein_interaction",
            "config": {
                "tokenizer": "kmer",
                "encoder": encoder,
                "model": "mlp",
                "classifier": "mlp",
                "pair_strategy": "two_stream",
                "interaction_level": "vector",
                "k": 1,
                lag_key: 3,
                "covariance_max_raw_length": 100000,
            },
            "artifact_dir": str(tmp_path / "artifact"),
        },
        {
            "format": "pair_split_dir",
            "root_path": str(data_root),
            "metadata": {},
        },
    )

    parsed = train_pair_pipeline.build_argument_parser().parse_args(command[2:])
    assert parsed.encoder == encoder
    assert getattr(parsed, f"peptide_{lag_key}") == 3
    assert getattr(parsed, f"partner_{lag_key}") == 3
    assert parsed.peptide_covariance_max_raw_length == 100000
    assert parsed.partner_covariance_max_raw_length == 100000


@pytest.mark.parametrize(
    ("reduction", "topics_key", "memory_key"),
    [
        ("lda", "lda_topics", "lda_dense_memory_limit_gib"),
        ("plsa", "plsa_topics", "plsa_memory_limit_gib"),
    ],
)
def test_web_pair_topic_command_is_accepted_by_real_parser(
    tmp_path,
    reduction,
    topics_key,
    memory_key,
):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    data_root = tmp_path / "data"
    for split in ["train", "val"]:
        split_root = data_root / split
        split_root.mkdir(parents=True)
        (split_root / "pairs.csv").write_text(
            "peptide,protein,label\nACDE,FGHIK,1\n",
            encoding="utf-8",
        )
    runner = ExperimentRunner(None, paths)
    stream = {
        "feature_route": "classic",
        "tokenizer": "kmer",
        "encoder": "bow",
        "max_length": 512,
        "max_length_unit": "tokens",
        "params": {"k": 2, "frequency_max_raw_length": 50000},
    }

    command = runner.build_command(
        {
            "task_type": "pep_protein_interaction",
            "config": {
                "task_type": "pep_protein_interaction",
                "tokenizer": "kmer",
                "encoder": "bow",
                "model": "rf",
                "classifier": "rf",
                "pair_strategy": "two_stream",
                "interaction_level": "vector",
                "predictor_type": "classifier",
                "fusion_mode": "concat",
                "reduction": reduction,
                topics_key: 17,
                memory_key: 2.5,
                "streams": {"peptide": stream, "partner": stream},
            },
            "artifact_dir": str(tmp_path / "artifact"),
        },
        {
            "format": "pair_split_dir",
            "root_path": str(data_root),
            "metadata": {},
        },
    )

    parsed = train_pair_pipeline.build_argument_parser().parse_args(command[2:])
    assert parsed.reduction == reduction
    assert getattr(parsed, topics_key) == 17
    assert getattr(parsed, memory_key) == 2.5
    assert parsed.peptide_encoder == "bow"
    assert parsed.partner_encoder == "bow"


def test_build_command_passes_protbert_model_path(tmp_path):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    data_root = tmp_path / "data"
    data_root.mkdir()
    runner = ExperimentRunner(None, paths)

    command = runner.build_command(
        {
            "task_type": "pep_function",
            "config": {
                "tokenizer": "protbert",
                "encoder": "index",
                "model": "mlp",
                "protbert_model": "models/protbert",
            },
            "artifact_dir": str(tmp_path / "artifact"),
        },
        {
            "format": "sequence_dir",
            "root_path": str(data_root),
            "metadata": {},
        },
    )

    assert command[command.index("--protbert_model") + 1] == "models/protbert"


def test_build_command_accepts_legacy_probert_model_path(tmp_path):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    data_root = tmp_path / "data"
    data_root.mkdir()
    runner = ExperimentRunner(None, paths)

    command = runner.build_command(
        {
            "task_type": "pep_function",
            "config": {
                "tokenizer": "probert",
                "encoder": "index",
                "model": "mlp",
                "probert_model": "models/probert",
            },
            "artifact_dir": str(tmp_path / "artifact"),
        },
        {
            "format": "sequence_dir",
            "root_path": str(data_root),
            "metadata": {},
        },
    )

    assert command[command.index("--tokenizer") + 1] == "protbert"
    assert command[command.index("--encoder") + 1] == "protbert"
    assert command[command.index("--probert_model") + 1] == "models/probert"


def test_build_command_passes_custom_advanced_parameters(tmp_path):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    data_root = tmp_path / "data"
    data_root.mkdir()
    runner = ExperimentRunner(None, paths)

    command = runner.build_command(
        {
            "task_type": "pep_function",
            "config": {
                "tokenizer": "kmer",
                "encoder": "index",
                "model": "cnn",
                "k": 3,
                "seed": 17,
                "lr": 0.0005,
                "dropout": 0.25,
                "weight_decay": 0.02,
                "enable_semantic_similarity": True,
                "semantic_similarity_weight": 0.2,
                "ml_auto_tune": True,
                "ml_tuning_objective": "auc",
                "ml_tuning_mode": "cv",
                "ml_tuning_max_candidates": 6,
                "ml_cv_folds": 2,
                "svm_kernel": "linear",
            },
            "artifact_dir": str(tmp_path / "artifact"),
        },
        {
            "format": "sequence_dir",
            "root_path": str(data_root),
            "metadata": {},
        },
    )

    assert command[command.index("--k") + 1] == "3"
    assert command[command.index("--seed") + 1] == "17"
    assert command[command.index("--lr") + 1] == "0.0005"
    assert command[command.index("--dropout") + 1] == "0.25"
    assert command[command.index("--weight_decay") + 1] == "0.02"
    assert "--enable_semantic_similarity" in command
    assert command[command.index("--semantic_similarity_weight") + 1] == "0.2"
    assert "--semantic_loss_type" not in command
    assert "--ml_auto_tune" in command
    assert command[command.index("--ml_tuning_objective") + 1] == "auc"
    assert command[command.index("--ml_tuning_mode") + 1] == "cv"
    assert command[command.index("--ml_tuning_max_candidates") + 1] == "6"
    assert command[command.index("--ml_cv_folds") + 1] == "2"
    assert command[command.index("--svm_kernel") + 1] == "linear"


def test_build_command_passes_glove_training_parameters(tmp_path):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    data_root = tmp_path / "data"
    data_root.mkdir()
    runner = ExperimentRunner(None, paths)

    command = runner.build_command(
        {
            "task_type": "pep_function",
            "config": {
                "tokenizer": "kmer",
                "encoder": "glove",
                "model": "cnn",
                "embedding_vector_size": 64,
                "embedding_window": 7,
                "embedding_min_count": 2,
                "embedding_epochs": 30,
                "glove_x_max": 10.0,
                "glove_alpha": 0.5,
                "glove_learning_rate": 0.1,
                "glove_batch_size": 512,
            },
            "artifact_dir": str(tmp_path / "artifact"),
        },
        {
            "format": "sequence_dir",
            "root_path": str(data_root),
            "metadata": {},
        },
    )

    for key, value in {
        "embedding_vector_size": "64",
        "embedding_window": "7",
        "embedding_min_count": "2",
        "embedding_epochs": "30",
        "glove_x_max": "10.0",
        "glove_alpha": "0.5",
        "glove_learning_rate": "0.1",
        "glove_batch_size": "512",
    }.items():
        assert command[command.index(f"--{key}") + 1] == value


def test_build_command_passes_fasttext_contract_parameters(tmp_path):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    data_root = tmp_path / "data"
    data_root.mkdir()
    runner = ExperimentRunner(None, paths)

    command = runner.build_command(
        {
            "task_type": "pep_function",
            "config": {
                "tokenizer": "kmer",
                "encoder": "fasttext",
                "model": "cnn",
                "embedding_workers": 2,
                "fasttext_bucket": 500000,
                "static_embedding_memory_limit_gib": 3.5,
            },
            "artifact_dir": str(tmp_path / "artifact"),
        },
        {
            "format": "sequence_dir",
            "root_path": str(data_root),
            "metadata": {},
        },
    )

    for key, value in {
        "embedding_workers": "2",
        "fasttext_bucket": "500000",
        "static_embedding_memory_limit_gib": "3.5",
    }.items():
        assert command[command.index(f"--{key}") + 1] == value


def test_build_command_rejects_removed_cs_encoder(tmp_path):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    split_root = tmp_path / "pair-root"
    for split in ["train", "val", "test"]:
        split_dir = split_root / split
        split_dir.mkdir(parents=True)
        (split_dir / "pairs.csv").write_text("peptide,protein,label\nACD,EFG,1\n", encoding="utf-8")
    runner = ExperimentRunner(None, paths)

    with pytest.raises(ValueError, match="CS encoder has been removed"):
        runner.build_command(
            {
                "task_type": "pep_protein_interaction",
                "config": {
                    "tokenizer": "kmer",
                    "encoder": "cs",
                    "model": "rf",
                    "classifier": "rf",
                    "pair_strategy": "two_stream",
                    "interaction_level": "vector",
                    "predictor_type": "classifier",
                },
                "artifact_dir": str(tmp_path / "artifact"),
            },
            {
                "format": "pair_split_dir",
                "root_path": str(split_root),
                "metadata": {},
            },
        )


def test_sequence_build_command_filters_pair_only_fields(tmp_path):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    data_root = tmp_path / "data"
    data_root.mkdir()
    runner = ExperimentRunner(None, paths)

    command = runner.build_command(
        {
            "task_type": "pep_function",
            "config": {
                "task_type": "pep_function",
                "tokenizer": "kmer",
                "encoder": "index",
                "model": "cnn",
                "pair_strategy": "two_stream",
                "interaction_level": "vector",
                "enable_interaction_semantic_similarity": False,
                "cache_features": True,
            },
            "artifact_dir": str(tmp_path / "artifact"),
        },
        {
            "format": "sequence_dir",
            "root_path": str(data_root),
            "metadata": {},
        },
    )

    assert command[1].endswith("scripts/train_sequence_pipeline.py")
    assert "--pair_strategy" not in command
    assert "--interaction_level" not in command
    assert "--no-enable_interaction_semantic_similarity" not in command
    assert "--cache_features" not in command


def test_web_hybrid_command_is_accepted_by_real_training_parser(tmp_path):
    paths = RuntimePaths.from_project_root(
        tmp_path / "project",
        tmp_path / "runtime",
    ).ensure()
    data_root = tmp_path / "data"
    data_root.mkdir()
    runner = ExperimentRunner(None, paths)

    command = runner.build_command(
        {
            "task_type": "pep_function",
            "config": {
                "tokenizer": "kmer",
                "encoder": "pssm_blosum62_hybrid",
                "model": "cnn",
                "k": 1,
                "profile_on_missing": "blosum62_hybrid",
            },
            "artifact_dir": str(tmp_path / "artifact"),
        },
        {
            "format": "sequence_dir",
            "root_path": str(data_root),
            "metadata": {},
        },
    )

    parsed = train_sequence_pipeline.build_argument_parser().parse_args(command[2:])

    assert parsed.encoder == "pssm_blosum62_hybrid"
    assert parsed.profile_on_missing == "blosum62_hybrid"


def test_build_command_uses_flattened_sequence_features_for_classical_classifier(tmp_path):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    data_root = tmp_path / "data"
    data_root.mkdir()
    runner = ExperimentRunner(None, paths)

    command = runner.build_command(
        {
            "task_type": "pep_function",
            "config": {
                "tokenizer": "kmer",
                "encoder": "pssm",
                "model": "svm",
                "classifier": "svm",
            },
            "artifact_dir": str(tmp_path / "artifact"),
        },
        {
            "format": "sequence_dir",
            "root_path": str(data_root),
            "metadata": {},
        },
    )

    assert command[command.index("--tokenizer") + 1] == "kmer"
    assert command[command.index("--encoder") + 1] == "pssm"
    assert command[command.index("--model") + 1] == "svm"
    assert command[command.index("--classifier") + 1] == "svm"
    assert "--sequence_model" not in command
    assert "--training_mode" not in command
    assert "--epochs" not in command
    assert "--batch_size" not in command


def test_register_model_without_bundle_uses_task_specific_reason(tmp_path):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    store = MetadataStore(paths.database)
    dataset = store.create_dataset(
        "demo",
        "pep_hlaii",
        "upload",
        "pair_split_dir",
        tmp_path / "dataset",
    )
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    experiment = store.create_experiment(
        "hla run",
        dataset["id"],
        "pep_hlaii",
        {
            "task_type": "pep_hlaii",
            "tokenizer": "kmer",
            "encoder": "index",
            "model": "cnn",
            "pair_strategy": "single_stream",
            "sequence_model": "cnn",
            "training_mode": "end_to_end",
            "classifier": "mlp",
        },
        tmp_path / "train.log",
        artifact_dir,
    )
    runner = ExperimentRunner(store, paths)

    model = runner._register_model_if_present(experiment["id"], experiment)

    assert model["supports_inference"] is False
    assert model["metadata"]["reason_code"] == "legacy_bundle_missing"


def test_offline_mode_sets_huggingface_environment(monkeypatch):
    monkeypatch.setenv("PEPPLATFORM_OFFLINE_MODE", "1")

    env = ExperimentRunner.build_subprocess_env()

    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["HF_HUB_DISABLE_XET"] == "1"
    assert env["HF_HUB_DOWNLOAD_TIMEOUT"] == "120"
    assert env["HF_HUB_ETAG_TIMEOUT"] == "30"
    assert env["PYTHONUNBUFFERED"] == "1"


def test_default_huggingface_endpoint_uses_mirror(monkeypatch):
    monkeypatch.delenv("PEPPLATFORM_OFFLINE_MODE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("PEPPLATFORM_HF_ENDPOINT", raising=False)

    env = ExperimentRunner.build_subprocess_env()

    assert env["HF_ENDPOINT"] == "https://hf-mirror.com"
    assert env["HF_HUB_DISABLE_XET"] == "1"
    assert env["HF_HUB_DOWNLOAD_TIMEOUT"] == "120"
    assert env["HF_HUB_ETAG_TIMEOUT"] == "30"
    assert "HF_HUB_OFFLINE" not in env
    assert "TRANSFORMERS_OFFLINE" not in env


def test_custom_huggingface_endpoint_is_preserved(monkeypatch):
    monkeypatch.delenv("PEPPLATFORM_OFFLINE_MODE", raising=False)
    monkeypatch.setenv("HF_ENDPOINT", "https://hf.example.test")
    monkeypatch.setenv("PEPPLATFORM_HF_ENDPOINT", "https://hf-mirror.com")

    env = ExperimentRunner.build_subprocess_env()

    assert env["HF_ENDPOINT"] == "https://hf.example.test"


def test_custom_huggingface_download_environment_is_preserved(monkeypatch):
    monkeypatch.delenv("PEPPLATFORM_OFFLINE_MODE", raising=False)
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    monkeypatch.setenv("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    monkeypatch.setenv("HF_HUB_ETAG_TIMEOUT", "60")

    env = ExperimentRunner.build_subprocess_env()

    assert env["HF_HUB_DISABLE_XET"] == "0"
    assert env["HF_HUB_DOWNLOAD_TIMEOUT"] == "300"
    assert env["HF_HUB_ETAG_TIMEOUT"] == "60"


def test_pepplatform_huggingface_endpoint_is_used(monkeypatch):
    monkeypatch.delenv("PEPPLATFORM_OFFLINE_MODE", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setenv("PEPPLATFORM_HF_ENDPOINT", "https://hf.internal.test")

    env = ExperimentRunner.build_subprocess_env()

    assert env["HF_ENDPOINT"] == "https://hf.internal.test"


def test_cancel_running_experiment_terminates_active_process(tmp_path, monkeypatch):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    paths.project_root.mkdir(parents=True, exist_ok=True)
    store = MetadataStore(paths.database)
    data_root = tmp_path / "data"
    data_root.mkdir()
    dataset = store.create_dataset("demo", "pep_function", "upload", "sequence_dir", data_root)
    experiment_dir = paths.experiments / "cancel-active"
    artifact_dir = experiment_dir / "artifact"
    experiment = store.create_experiment(
        "cancel active",
        dataset["id"],
        "pep_function",
        {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
        experiment_dir / "train.log",
        artifact_dir,
        experiment_id="cancel-active",
    )
    runner = ExperimentRunner(store, paths)
    monkeypatch.setattr(
        runner,
        "build_command",
        lambda _experiment, _dataset: [sys.executable, "-c", "import time; time.sleep(60)"],
    )

    thread = threading.Thread(target=runner._run_experiment, args=(experiment["id"],), daemon=True)
    thread.start()
    deadline = time.time() + 5
    while time.time() < deadline:
        with runner._lock:
            has_process = experiment["id"] in runner._active_processes
        if has_process:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("training process did not start")

    stopped = runner.cancel(experiment["id"], reason="test stop")
    thread.join(timeout=5)

    assert stopped["status"] == "canceled"
    assert store.get_experiment(experiment["id"])["status"] == "canceled"
    assert not thread.is_alive()


def test_runner_persists_specific_structured_failure_from_subprocess(
    tmp_path,
    monkeypatch,
):
    paths = RuntimePaths.from_project_root(
        tmp_path / "project",
        tmp_path / "runtime",
    ).ensure()
    paths.project_root.mkdir(parents=True, exist_ok=True)
    store = MetadataStore(paths.database)
    data_root = tmp_path / "data"
    data_root.mkdir()
    dataset = store.create_dataset(
        "demo",
        "pep_function",
        "upload",
        "sequence_dir",
        data_root,
    )
    experiment_dir = paths.experiments / "structured-failure"
    failure = {
        "schema_version": 1,
        "code": "profile_no_nonself_hits",
        "category": "profile",
        "stage": "profile_generation",
        "message": "strict PSFM found no usable homologs",
        "context": {
            "feature": "psfm",
            "sample_id": "Train5",
            "sequence_length": 5,
            "nonself_hit_count": 0,
            "zero_nonself_hits": True,
        },
        "_specific": True,
    }
    experiment = store.create_experiment(
        "structured failure",
        dataset["id"],
        "pep_function",
        {"tokenizer": "kmer", "encoder": "psfm", "model": "cnn"},
        experiment_dir / "train.log",
        experiment_dir / "artifact",
        experiment_id="structured-failure",
    )
    runner = ExperimentRunner(store, paths)
    script = (
        "import json,sys;"
        f"print('[training-failure] '+json.dumps({failure!r}), flush=True);"
        "print('[training-failure] '+json.dumps({"
        "'schema_version':1,'code':'training_configuration_error',"
        "'category':'configuration','stage':'setup',"
        "'message':'outer profile cache build failed','context':{},"
        "'_specific':False}), flush=True);"
        "sys.exit(1)"
    )
    monkeypatch.setattr(
        runner,
        "build_command",
        lambda _experiment, _dataset: [sys.executable, "-c", script],
    )

    runner._run_experiment(experiment["id"])

    stored = store.get_experiment(experiment["id"])
    assert stored["status"] == "failed"
    assert stored["error"] == "strict PSFM found no usable homologs"
    assert stored["failure"]["code"] == "profile_no_nonself_hits"
    assert stored["failure"]["context"]["sample_id"] == "Train5"
    assert stored["failure"]["exit_code"] == 1


@pytest.mark.parametrize("failure_stage", ["resources", "command", "unexpected", "log_directory"])
def test_failed_preparation_does_not_stop_the_training_queue(tmp_path, monkeypatch, failure_stage):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    paths.project_root.mkdir()
    store = MetadataStore(paths.database)
    dataset = store.create_dataset(
        "demo", "pep_function", "upload", "sequence_dir", tmp_path / "data"
    )
    experiments = []
    for name in ("broken", "next"):
        directory = paths.experiments / name
        experiments.append(
            store.create_experiment(
                name,
                dataset["id"],
                "pep_function",
                {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
                directory / "train.log",
                directory / "artifact",
                experiment_id=name,
            )
        )
    runner = ExperimentRunner(store, paths)

    def build(experiment, _dataset):
        if failure_stage == "command" and experiment["id"] == "broken":
            raise ValueError("input preparation failed")
        return [sys.executable, "-c", "print('training completed', flush=True)"]

    monkeypatch.setattr(runner, "build_command", build)
    if failure_stage == "resources":
        calls = iter((False, True))

        def prepare():
            if not next(calls):
                raise RuntimeError("input preparation failed")

        runner.before_experiment = prepare
    elif failure_stage == "unexpected":
        original_run = runner._run_experiment

        def execute(experiment_id):
            if experiment_id == "broken":
                raise RuntimeError("input preparation failed")
            original_run(experiment_id)

        monkeypatch.setattr(runner, "_run_experiment", execute)
    elif failure_stage == "log_directory":
        Path(experiments[0]["log_path"]).mkdir(parents=True)

    runner.start()
    try:
        for experiment in experiments:
            runner.submit(experiment["id"])
        deadline = time.monotonic() + 10
        while (
            store.get_experiment("next")["status"] not in {"succeeded", "failed"}
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        first, second = (store.get_experiment(e["id"]) for e in experiments)
        assert first["status"] == "failed"
        assert first["failure"]["stage"] == "setup"
        assert "exit_code" not in first["failure"]
        assert second["status"] == "succeeded"
        assert runner.thread.is_alive()
        if failure_stage != "log_directory":
            assert "input preparation failed" in Path(first["log_path"]).read_text()
    finally:
        runner.stop()
        runner.thread.join(timeout=2)


def test_queue_preserves_completed_status_when_finalizer_raises(tmp_path, monkeypatch):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    store = MetadataStore(paths.database)
    dataset = store.create_dataset(
        "demo", "pep_function", "upload", "sequence_dir", tmp_path / "data"
    )
    for name in ("sealed", "next"):
        store.create_experiment(
            name,
            dataset["id"],
            "pep_function",
            {},
            paths.experiments / name / "train.log",
            paths.experiments / name / "artifact",
            experiment_id=name,
        )
    runner = ExperimentRunner(store, paths)

    def execute(experiment_id):
        store.update_experiment(experiment_id, status="succeeded")
        if experiment_id == "sealed":
            raise OSError("version export failed")

    monkeypatch.setattr(runner, "_run_experiment", execute)
    runner.start()
    try:
        runner.submit("sealed")
        runner.submit("next")
        deadline = time.monotonic() + 5
        while store.get_experiment("next")["status"] == "queued" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert store.get_experiment("sealed")["status"] == "succeeded"
        assert store.get_experiment("next")["status"] == "succeeded"
        assert runner.thread.is_alive()
    finally:
        runner.stop()
        runner.thread.join(timeout=2)


@pytest.mark.parametrize(
    "task", ["pep_function", "pep_protein_interaction", "pep_hlaii", "pep_hlai"]
)
@pytest.mark.parametrize("loss_type", [None, "legacy", "jaccard_supcon"])
@pytest.mark.parametrize("counts", [None, (1, 8), (4, 16), (4, 0)])
def test_build_command_round_trips_semantic_loss_selection(tmp_path, task, loss_type, counts):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    data_root = tmp_path / "data"
    data_root.mkdir()
    pair = task != "pep_function"
    runner = ExperimentRunner(None, paths)
    command = runner.build_command(
        {
            "task_type": task,
            "config": {
                "tokenizer": "kmer",
                "encoder": "onehot",
                "model": "cnn",
                "enable_semantic_similarity": True,
                **({"semantic_loss_type": loss_type} if loss_type else {}),
                **(
                    {"semantic_positive_partners": counts[0], "semantic_zero_partners": counts[1]}
                    if counts is not None
                    else {}
                ),
            },
            "artifact_dir": str(tmp_path / "artifact"),
        },
        {
            "format": "pair_split_dir" if pair else "sequence_dir",
            "root_path": str(data_root),
            "metadata": {},
        },
    )
    if loss_type is not None:
        assert command[command.index("--semantic_loss_type") + 1] == loss_type
    else:
        assert "--semantic_loss_type" not in command
    if counts is not None:
        assert command[command.index("--semantic_positive_partners") + 1] == str(counts[0])
        assert command[command.index("--semantic_zero_partners") + 1] == str(counts[1])
    else:
        assert "--semantic_positive_partners" not in command
        assert "--semantic_zero_partners" not in command
    module = train_pair_pipeline if pair else train_sequence_pipeline
    parsed = module.build_argument_parser().parse_args(command[2:])
    assert getattr(parsed, "semantic_loss_type", None) == loss_type
    assert parsed.semantic_positive_partners == (counts[0] if counts is not None else 1)
    assert parsed.semantic_zero_partners == (counts[1] if counts is not None else 8)
