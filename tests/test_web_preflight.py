import pytest

from pepplatform.web.preflight import preflight_experiment
from pepplatform.web.config import normalize_experiment_config_with_sources


def _dataset(task_type, fmt, splits=None):
    split_counts = splits or {"train": 10, "val": 2}
    by_split = {}
    for split, count in split_counts.items():
        by_split[split] = {
            "sequence": {"count": count, "min": 8, "max": 60, "mean": 20.0},
            "peptide": {"count": count, "min": 8, "max": 50, "mean": 15.0},
            "partner": {"count": count, "min": 10, "max": 512, "mean": 200.0},
        }
    return {
        "id": "dataset",
        "name": "demo",
        "task_type": task_type,
        "format": fmt,
        "metadata": {
            "splits": split_counts,
            "length_stats": {
                "version": 1,
                "by_split": by_split,
                "overall": {
                    "sequence": {
                        "count": sum(split_counts.values()),
                        "min": 8,
                        "max": 60,
                        "mean": 20.0,
                    },
                    "peptide": {
                        "count": sum(split_counts.values()),
                        "min": 8,
                        "max": 50,
                        "mean": 15.0,
                    },
                    "partner": {
                        "count": sum(split_counts.values()),
                        "min": 10,
                        "max": 512,
                        "mean": 200.0,
                    },
                },
            },
        },
    }


def test_preflight_accepts_sequence_function_route_and_reports_inference_support():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
    )
    assert report["valid"] is True
    assert report["execution"]["script"] == "scripts/train_sequence_pipeline.py"
    assert report["execution"]["supports_reusable_inference"] is True
    assert report["normalized_config"]["max_length_unit"] == "tokens"
    assert report["warnings"] == []


@pytest.mark.parametrize(
    ("encoder", "warning_code"),
    [
        ("pssm", "strict_profile_complete_coverage_required"),
        (
            "pssm_blosum62_hybrid",
            "pssm_blosum62_hybrid_fallback_enabled",
        ),
        (
            "psfm_blosum62_hybrid",
            "psfm_blosum62_hybrid_fallback_enabled",
        ),
    ],
)
def test_preflight_explains_profile_coverage_contract(encoder, warning_code):
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir"),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": encoder,
            "model": "cnn",
            "k": 1,
        },
    )

    assert report["valid"] is True
    assert warning_code in report["warning_codes"]


def test_preflight_defaults_hybrid_pssm_covariance_policy():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir"),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "ac_pssm_blosum62_hybrid",
            "model": "mlp",
            "ac_pssm_blosum62_hybrid_lag": 3,
        },
    )

    assert report["valid"] is True
    assert report["normalized_config"]["k"] == 1
    assert report["normalized_config"]["profile_on_missing"] == "blosum62_hybrid"
    assert report["parameter_sources"]["profile_on_missing"] == "default"
    assert "pssm_blosum62_hybrid_fallback_enabled" in report["warning_codes"]


def test_preflight_rejects_hybrid_pssm_covariance_policy_mismatch():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir"),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "cc_pssm_blosum62_hybrid",
            "model": "mlp",
            "cc_pssm_blosum62_hybrid_lag": 3,
            "profile_on_missing": "error",
        },
    )

    assert report["valid"] is False
    assert "hybrid_profile_policy_mismatch" in report["error_codes"]


def test_preflight_validates_hybrid_pssm_covariance_long_lag_key():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir"),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "acc_pssm_blosum62_hybrid",
            "model": "mlp",
            "acc_pssm_blosum62_hybrid_lag": 0,
        },
    )

    assert report["valid"] is False
    assert set(report["error_codes"]).intersection(
        {"normalization_error", "invalid_numeric_parameter"}
    )


def test_preflight_rejects_hybrid_pssm_covariance_short_sequences():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir"),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "ac_pssm_blosum62_hybrid",
            "model": "mlp",
            "ac_pssm_blosum62_hybrid_lag": 8,
        },
    )

    assert report["valid"] is False
    assert "pssm_covariance_lag_too_large" in report["error_codes"]


@pytest.mark.parametrize(
    "encoder",
    [
        "pssm",
        "pssm_blosum62_hybrid",
        "psfm_blosum62_hybrid",
        "psfm",
        "ac_pssm",
        "cc_pssm",
        "acc_pssm",
        "ac_pssm_blosum62_hybrid",
        "cc_pssm_blosum62_hybrid",
        "acc_pssm_blosum62_hybrid",
    ],
)
def test_preflight_derives_k1_for_raw_residue_profile_encoders(encoder):
    config = {
        "tokenizer": "kmer",
        "encoder": encoder,
        "model": "cnn",
    }
    if encoder in {
        "ac_pssm",
        "cc_pssm",
        "acc_pssm",
        "ac_pssm_blosum62_hybrid",
        "cc_pssm_blosum62_hybrid",
        "acc_pssm_blosum62_hybrid",
    }:
        config[f"{encoder}_lag"] = 7
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir"),
        "pep_function",
        config,
    )

    assert report["valid"] is True
    assert report["normalized_config"]["k"] == 1
    assert report["parameter_sources"]["k"] == "derived"


@pytest.mark.parametrize(
    "encoder",
    [
        "pssm",
        "pssm_blosum62_hybrid",
        "psfm_blosum62_hybrid",
        "psfm",
        "ac_pssm",
        "cc_pssm",
        "acc_pssm",
        "ac_pssm_blosum62_hybrid",
        "cc_pssm_blosum62_hybrid",
        "acc_pssm_blosum62_hybrid",
    ],
)
@pytest.mark.parametrize("k", [2, 3])
def test_preflight_rejects_explicit_non_residue_k_for_profile_encoder(encoder, k):
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir"),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": encoder,
            "model": "cnn",
            "k": k,
        },
    )

    assert report["valid"] is False
    assert "profile_encoder_requires_k1" in report["error_codes"]
    assert report["normalized_config"]["k"] == k


def test_preflight_validates_frequency_raw_length_limit():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir"),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "bow",
            "model": "rf",
            "frequency_max_raw_length": 100_001,
        },
    )

    assert report["valid"] is False
    assert "normalization_error" in report["error_codes"]
    assert any("frequency_max_raw_length" in message for message in report["errors"])


@pytest.mark.parametrize("encoder", ["ac", "cc", "acc"])
@pytest.mark.parametrize("classifier", ["mlp", "svm"])
def test_preflight_accepts_covariance_fixed_vector_classifiers(
    encoder,
    classifier,
):
    lag_key = f"{encoder}_lag"
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir"),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": encoder,
            "model": classifier,
            "classifier": classifier,
            lag_key: 7,
        },
    )

    assert report["valid"] is True
    assert report["normalized_config"]["k"] == 1
    assert report["normalized_config"][lag_key] == 7


@pytest.mark.parametrize("encoder", ["ac", "cc", "acc"])
def test_preflight_rejects_covariance_lag_at_dataset_minimum(encoder):
    lag_key = f"{encoder}_lag"
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir"),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": encoder,
            "model": "mlp",
            "classifier": "mlp",
            lag_key: 8,
        },
    )

    assert report["valid"] is False
    assert "covariance_lag_too_large" in report["error_codes"]
    assert any("minimum sequence length 8" in message for message in report["errors"])


def test_preflight_rejects_default_covariance_lag_for_short_quick_data():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir"),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "acc",
            "model": "mlp",
            "classifier": "mlp",
        },
    )

    assert report["normalized_config"]["acc_lag"] == 10
    assert report["valid"] is False
    assert "covariance_lag_too_large" in report["error_codes"]


def test_preflight_rejects_covariance_raw_length_below_dataset_maximum():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir"),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "ac",
            "model": "svm",
            "classifier": "svm",
            "ac_lag": 7,
            "covariance_max_raw_length": 59,
        },
    )

    assert report["valid"] is False
    assert "covariance_raw_length_exceeded" in report["error_codes"]
    assert any("maximum sequence length 60" in message for message in report["errors"])


def test_preflight_requires_covariance_dataset_length_statistics():
    dataset = _dataset("pep_function", "sequence_dir")
    del dataset["metadata"]["length_stats"]

    report = preflight_experiment(
        dataset,
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "cc",
            "model": "mlp",
            "classifier": "mlp",
            "cc_lag": 1,
        },
    )

    assert report["valid"] is False
    assert "covariance_length_stats_unavailable" in report["error_codes"]


def test_preflight_checks_covariance_homogeneous_streams_by_dataset_role():
    report = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir"),
        "pep_protein_interaction",
        {
            "tokenizer": "kmer",
            "encoder": "ac",
            "model": "mlp",
            "classifier": "mlp",
            "pair_strategy": "two_stream",
            "interaction_level": "vector",
            "fusion_mode": "concat",
            "ac_lag": 9,
        },
    )

    assert report["valid"] is False
    lag_errors = [
        message
        for code, message in zip(report["error_codes"], report["errors"])
        if code == "covariance_lag_too_large"
    ]
    assert len(lag_errors) == 1
    assert "peptide stream" in lag_errors[0]


def test_preflight_checks_heterogeneous_covariance_stream_limits_by_role():
    base = {
        "tokenizer": "kmer",
        "encoder": "ac",
        "model": "mlp",
        "classifier": "mlp",
        "pair_strategy": "two_stream",
        "interaction_level": "vector",
        "fusion_mode": "concat",
        "streams": {
            "peptide": {
                "feature_route": "classic",
                "tokenizer": "kmer",
                "encoder": "ac",
                "max_length": 60,
                "params": {
                    "k": 1,
                    "ac_lag": 7,
                    "covariance_max_raw_length": 50,
                },
            },
            "partner": {
                "feature_route": "classic",
                "tokenizer": "kmer",
                "encoder": "cc",
                "max_length": 512,
                "params": {
                    "k": 1,
                    "cc_lag": 9,
                    "covariance_max_raw_length": 512,
                },
            },
        },
    }

    valid = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir"),
        "pep_protein_interaction",
        base,
    )
    assert valid["valid"] is True

    too_short = {
        **base,
        "streams": {
            **base["streams"],
            "partner": {
                **base["streams"]["partner"],
                "params": {
                    **base["streams"]["partner"]["params"],
                    "covariance_max_raw_length": 511,
                },
            },
        },
    }
    invalid = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir"),
        "pep_protein_interaction",
        too_short,
    )

    assert invalid["valid"] is False
    raw_errors = [
        message
        for code, message in zip(invalid["error_codes"], invalid["errors"])
        if code == "covariance_raw_length_exceeded"
    ]
    assert len(raw_errors) == 1
    assert "CC partner stream" in raw_errors[0]


def test_preflight_rejects_raw_index_pair_vector_route():
    report = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir"),
        "pep_protein_interaction",
        {
            "tokenizer": "kmer",
            "encoder": "index",
            "model": "mlp",
            "pair_strategy": "two_stream",
            "interaction_level": "vector",
        },
    )

    assert report["valid"] is False
    assert "classical_index_unsupported" in report["error_codes"]


def test_preflight_accepts_index_pair_token_route():
    report = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir"),
        "pep_protein_interaction",
        {
            "tokenizer": "kmer",
            "encoder": "index",
            "model": "cross_attn",
            "pair_strategy": "two_stream",
            "interaction_level": "token",
            "interaction_model": "cross_attn",
        },
    )

    assert report["valid"] is True
    assert "classical_index_unsupported" not in report["error_codes"]


def test_preflight_validates_glove_training_parameters():
    dataset = _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2})
    valid = preflight_experiment(
        dataset,
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "glove",
            "model": "cnn",
            "embedding_vector_size": 64,
            "glove_x_max": 10.0,
            "glove_alpha": 0.5,
            "glove_learning_rate": 0.1,
            "glove_batch_size": 512,
        },
    )
    invalid = preflight_experiment(
        dataset,
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "glove",
            "model": "cnn",
            "glove_alpha": 1.5,
            "glove_batch_size": 0,
        },
    )

    assert valid["valid"] is True
    assert valid["normalized_config"]["glove_x_max"] == 10.0
    assert valid["normalized_config"]["glove_batch_size"] == 512
    assert invalid["valid"] is False
    assert "invalid_numeric_parameter" in invalid["error_codes"]


def test_preflight_validates_fasttext_resource_parameters():
    dataset = _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2})
    valid = preflight_experiment(
        dataset,
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "fasttext",
            "model": "cnn",
            "embedding_workers": 1,
            "fasttext_bucket": 2000000,
            "static_embedding_memory_limit_gib": 4.0,
        },
    )
    invalid = preflight_experiment(
        dataset,
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "fasttext",
            "model": "cnn",
            "embedding_workers": 0,
            "fasttext_bucket": -1,
            "static_embedding_memory_limit_gib": 0,
        },
    )

    assert valid["valid"] is True
    assert valid["normalized_config"]["fasttext_bucket"] == 2000000
    assert invalid["valid"] is False
    assert "invalid_numeric_parameter" in invalid["error_codes"]


def test_preflight_materializes_neural_and_profile_defaults_with_sources():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "pssm",
            "model": "cnn",
            "epochs": 12,
        },
    )

    config = report["normalized_config"]
    sources = report["parameter_sources"]
    assert report["valid"] is True
    assert config["sequence_model"] == "cnn"
    assert config["classifier"] == "mlp"
    assert config["batch_size"] == 32
    assert config["lr"] == 1e-3
    assert config["epochs"] == 12
    assert config["profile_iterations"] == 3
    assert config["profile_threads"] == 1
    assert sources["sequence_model"] == "derived"
    assert sources["batch_size"] == "default"
    assert sources["lr"] == "default"
    assert sources["epochs"] == "explicit"
    assert sources["profile_iterations"] == "default"


def test_preflight_materializes_rf_defaults_without_neural_training_parameters():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "pssm",
            "model": "rf",
            "classifier": "rf",
        },
    )

    config = report["normalized_config"]
    sources = report["parameter_sources"]
    assert report["valid"] is True
    assert config["sequence_model"] == ""
    assert "batch_size" not in config
    assert "lr" not in config
    assert "epochs" not in config
    assert config["rf_estimators"] == 300
    assert config["rf_max_depth"] == 0
    assert config["rf_min_samples_leaf"] == 1
    assert config["ml_tuning_objective"] == "f1"
    assert config["profile_iterations"] == 3
    assert sources["rf_estimators"] == "default"
    assert sources["ml_tuning_objective"] == "default"
    assert sources["profile_iterations"] == "default"


def test_classical_tuning_default_is_task_aware_and_explicit_values_win():
    pair_config, pair_sources = normalize_experiment_config_with_sources(
        "pep_protein_interaction",
        {"model": "rf", "classifier": "rf"},
    )
    function_config, _ = normalize_experiment_config_with_sources(
        "pep_function",
        {"model": "rf", "classifier": "rf"},
    )
    explicit_config, explicit_sources = normalize_experiment_config_with_sources(
        "pep_hlai",
        {"model": "rf", "classifier": "rf", "ml_tuning_objective": "f1"},
    )

    assert pair_config["ml_tuning_objective"] == "auc"
    assert pair_sources["ml_tuning_objective"] == "default"
    assert function_config["ml_tuning_objective"] == "f1"
    assert explicit_config["ml_tuning_objective"] == "f1"
    assert explicit_sources["ml_tuning_objective"] == "explicit"


@pytest.mark.parametrize("key", ["lgb_lr", "lgb_subsample", "lgb_colsample"])
def test_preflight_rejects_zero_lgb_positive_parameters(key):
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir"),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "bow",
            "model": "lgb",
            "classifier": "lgb",
            key: 0.0,
        },
    )

    assert report["valid"] is False
    assert "invalid_numeric_parameter" in report["error_codes"]


def test_preflight_rejects_negative_seed():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir"),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "bow",
            "model": "lgb",
            "classifier": "lgb",
            "seed": -1,
        },
    )

    assert report["valid"] is False
    assert "invalid_numeric_parameter" in report["error_codes"]


def test_tuning_budget_derives_bounded_cv_settings():
    config, sources = normalize_experiment_config_with_sources(
        "pep_function",
        {"model": "rf", "classifier": "rf", "ml_auto_tune": True, "ml_tuning_budget": "standard"},
    )

    assert config["ml_tuning_budget"] == "standard"
    assert config["ml_tuning_mode"] == "cv"
    assert config["ml_tuning_max_candidates"] == 12
    assert config["ml_cv_folds"] == 3
    assert sources["ml_tuning_max_candidates"] == "derived"


def test_preflight_rejects_conflicting_tuning_budget_values():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "bow",
            "model": "rf",
            "classifier": "rf",
            "ml_auto_tune": True,
            "ml_tuning_budget": "quick",
            "ml_tuning_max_candidates": 18,
            "ml_cv_folds": 5,
        },
    )

    assert "tuning_budget_conflict" in report["error_codes"]


def test_preflight_marks_scaled_sequence_function_route_reusable():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "onehot",
            "model": "cnn",
            "sequence_model": "cnn",
            "dl_scaler": "standard",
        },
    )
    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True
    assert "scaled_sequence_inference_unavailable" not in report["warning_codes"]


def test_preflight_marks_sequence_export_function_route_reusable():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "onehot",
            "model": "cnn",
            "sequence_model": "cnn",
            "training_mode": "export_embeddings",
            "classifier": "logreg",
        },
    )
    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True
    assert "sequence_export_inference_unavailable" not in report["warning_codes"]


def test_preflight_rejects_wrong_dataset_format():
    report = preflight_experiment(
        _dataset("pep_function", "pair_csv_bundle"),
        "pep_function",
        {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
    )
    assert report["valid"] is False
    assert "not supported" in report["errors"][0]
    assert report["error_codes"] == ["format_unsupported"]


def test_preflight_rejects_global_encoder_token_interaction():
    report = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir"),
        "pep_protein_interaction",
        {
            "tokenizer": "kmer",
            "encoder": "bow",
            "model": "cross_attn",
            "pair_strategy": "two_stream",
            "interaction_level": "token",
            "interaction_model": "cross_attn",
        },
    )
    assert report["valid"] is False
    assert "token-level interaction" in report["errors"][0]
    assert report["error_codes"] == ["token_interaction_requires_sequence_encoder"]


def test_preflight_explains_hlaii_pair_bundle_limitation():
    report = preflight_experiment(
        _dataset("pep_hlaii", "pair_csv_bundle"),
        "pep_hlaii",
        {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
    )
    assert report["valid"] is False
    assert "pairs.csv" in report["errors"][0]
    assert report["error_codes"] == ["hlaii_requires_pair_split_dir"]


def test_preflight_accepts_hlaii_pair_split_dir():
    report = preflight_experiment(
        _dataset("pep_hlaii", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_hlaii",
        {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
    )
    assert report["valid"] is True


def test_preflight_marks_ppi_graph_route_reusable():
    report = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_protein_interaction",
        {
            "tokenizer": "esm2",
            "encoder": "ppi_fusion",
            "model": "gcn",
            "pair_strategy": "two_stream",
            "interaction_level": "vector",
            "predictor_type": "sample_graph",
            "graph_predictor": "gcn",
        },
    )
    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True


def test_preflight_validates_graph_transformer_head_width():
    base = {
        "tokenizer": "esm2",
        "encoder": "ppi_fusion",
        "model": "graph_transformer",
        "pair_strategy": "two_stream",
        "interaction_level": "vector",
        "predictor_type": "sample_graph",
        "graph_predictor": "graph_transformer",
    }
    dataset = _dataset(
        "pep_protein_interaction",
        "pair_split_dir",
        {"train": 10, "val": 2},
    )

    valid = preflight_experiment(
        dataset,
        "pep_protein_interaction",
        {**base, "gnn_hidden_dim": 128},
    )
    invalid = preflight_experiment(
        dataset,
        "pep_protein_interaction",
        {**base, "gnn_hidden_dim": 130},
    )

    assert valid["valid"] is True
    assert invalid["valid"] is False
    assert "graph_transformer_hidden_dim_invalid" in invalid["error_codes"]


def test_preflight_accepts_typed_hgt_and_rejects_aligned_gated():
    base = {
        "tokenizer": "esm2",
        "encoder": "ppi_fusion",
        "model": "hgt",
        "pair_strategy": "two_stream",
        "interaction_level": "vector",
        "predictor_type": "sample_graph",
        "graph_predictor": "hgt",
    }
    valid = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir", {"train": 10, "val": 2}),
        "pep_protein_interaction",
        {**base, "fusion_mode": "concat_all"},
    )
    invalid = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir", {"train": 10, "val": 2}),
        "pep_protein_interaction",
        {**base, "fusion_mode": "aligned_gated"},
    )

    assert valid["valid"] is True
    assert invalid["valid"] is False
    assert "hgt_typed_graph_fusion_unsupported" in invalid["error_codes"]


def test_preflight_normalizes_audited_pair_fusion_aliases_and_rejects_unknown_mode():
    dataset = _dataset(
        "pep_protein_interaction",
        "pair_split_dir",
        {"train": 10, "val": 2, "test": 2},
    )
    base = {
        "tokenizer": "esm2",
        "encoder": "ppi_fusion",
        "model": "mlp",
        "classifier": "mlp",
        "pair_strategy": "two_stream",
        "interaction_level": "vector",
        "predictor_type": "classifier",
    }

    normalized = preflight_experiment(
        dataset,
        "pep_protein_interaction",
        {**base, "fusion_mode": "concat+absdiff"},
    )
    invalid = preflight_experiment(
        dataset,
        "pep_protein_interaction",
        {**base, "fusion_mode": "typo_mode"},
    )

    assert normalized["valid"] is True
    assert normalized["normalized_config"]["fusion_mode"] == "concat_absdiff"
    assert invalid["valid"] is False
    assert "pair_fusion_mode_unsupported" in invalid["error_codes"]


def test_preflight_hgt_uses_typed_roles_and_only_reads_legacy_fusion_markers():
    dataset = _dataset(
        "pep_protein_interaction",
        "pair_split_dir",
        {"train": 10, "val": 2, "test": 2},
    )
    base = {
        "tokenizer": "esm2",
        "encoder": "ppi_fusion",
        "model": "hgt",
        "pair_strategy": "two_stream",
        "interaction_level": "vector",
        "predictor_type": "sample_graph",
        "graph_predictor": "hgt",
    }

    current = preflight_experiment(dataset, "pep_protein_interaction", base)
    legacy = preflight_experiment(
        dataset,
        "pep_protein_interaction",
        {**base, "fusion_mode": "concat"},
    )
    invalid = preflight_experiment(
        dataset,
        "pep_protein_interaction",
        {**base, "fusion_mode": "concat_absdiff"},
    )

    assert current["valid"] is True
    assert "fusion_mode" not in current["normalized_config"]
    assert legacy["valid"] is True
    assert "fusion_mode" not in legacy["normalized_config"]
    assert "hgt_legacy_fusion_marker_ignored" in legacy["warning_codes"]
    assert invalid["valid"] is False
    assert "hgt_typed_graph_fusion_unsupported" in invalid["error_codes"]


def test_preflight_marks_pretrained_mlp_function_route_reusable():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {"tokenizer": "esm2", "encoder": "index", "model": "mlp"},
    )
    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True
    assert "inference_unavailable" not in report["warning_codes"]


def test_preflight_rejects_incompatible_official_esmc_runtime(monkeypatch):
    monkeypatch.setattr(
        "pepplatform.web.preflight.inspect_esmc_runtime",
        lambda: {
            "compatible": False,
            "code": "esmc_runtime_unsupported",
            "message": (
                "Official ESM-C requires esm==3.2.0, found 3.2.1.post1. "
                "Run: python -m pip install --no-cache-dir 'esm==3.2.0'"
            ),
        },
    )
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2}),
        "pep_function",
        {
            "tokenizer": "esmc",
            "encoder": "esmc",
            "model": "mlp",
            "esmc_model": "EvolutionaryScale/esmc-600m-2024-12",
        },
    )

    assert report["valid"] is False
    assert "esmc_runtime_unsupported" in report["error_codes"]
    assert any("pip install --no-cache-dir 'esm==3.2.0'" in message for message in report["errors"])


def test_preflight_rejects_incomplete_official_esmc_cache_offline(tmp_path, monkeypatch):
    incomplete = tmp_path / "models" / "EvolutionaryScale" / "esmc-300m-2024-12"
    incomplete.mkdir(parents=True)
    (incomplete / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("pepplatform.web.preflight.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "pepplatform.web.preflight.inspect_esmc_runtime",
        lambda: {"compatible": True, "code": "", "message": ""},
    )
    monkeypatch.setenv("PEPPLATFORM_OFFLINE_MODE", "1")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)

    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2}),
        "pep_function",
        {
            "tokenizer": "esmc",
            "encoder": "esmc",
            "model": "mlp",
            "esmc_model": "EvolutionaryScale/esmc-300m-2024-12",
        },
    )

    assert report["valid"] is False
    assert "pretrained_model_source_unavailable" in report["error_codes"]
    assert "pretrained_model_local_cache" not in report["warning_codes"]


@pytest.mark.parametrize("tokenizer", ["esm2", "protbert", "prott5", "esmc", "esm3"])
@pytest.mark.parametrize("architecture", ["cnn", "rnn", "gru", "lstm", "transformer", "gcn", "gat"])
@pytest.mark.parametrize(
    ("training_mode", "classifier"),
    [("end_to_end", "mlp"), ("export_embeddings", "rf")],
)
def test_preflight_accepts_pretrained_residue_matrix(
    tokenizer,
    architecture,
    training_mode,
    classifier,
):
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {
            "tokenizer": tokenizer,
            "encoder": tokenizer,
            "model": architecture,
            "sequence_model": architecture,
            "training_mode": training_mode,
            "classifier": classifier,
            "residue_adapter_dim": 128,
        },
    )

    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True
    expected_route = (
        "sequence.pretrained_residue_graph"
        if architecture in {"gcn", "gat"}
        else "sequence.pretrained_residue_sequence"
    )
    assert report["effective_route"]["route_id"] == expected_route


def test_preflight_accepts_esm2_residue_gat_function_route():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {"tokenizer": "esm2", "encoder": "index", "model": "gat"},
    )

    assert report["valid"] is True
    assert report["normalized_config"]["sequence_model"] == "gat"
    assert report["normalized_config"]["training_mode"] == "end_to_end"
    assert report["normalized_config"]["classifier"] == "mlp"
    assert report["execution"]["supports_reusable_inference"] is True


def test_preflight_warns_esm3_pooled_mlp_fp16_cache_rebuild():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {
            "tokenizer": "esm3",
            "encoder": "esm3",
            "model": "mlp",
            "dl_scaler": "none",
        },
    )

    assert report["valid"] is True
    assert "esm3_pooled_fp16_cache_rebuild" in report["warning_codes"]


def test_preflight_skips_esm3_fp16_cache_warning_for_scaled_mlp():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {
            "tokenizer": "esm3",
            "encoder": "esm3",
            "model": "mlp",
            "dl_scaler": "standard",
        },
    )

    assert report["valid"] is True
    assert "esm3_pooled_fp16_cache_rebuild" not in report["warning_codes"]


def test_preflight_skips_esm3_fp16_cache_warning_for_residue_route():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {
            "tokenizer": "esm3",
            "encoder": "esm3",
            "model": "cnn",
            "sequence_model": "cnn",
        },
    )

    assert report["valid"] is True
    assert "esm3_pooled_fp16_cache_rebuild" not in report["warning_codes"]


def test_preflight_rejects_classic_index_chain_graph_route():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {"tokenizer": "kmer", "encoder": "index", "model": "gcn"},
    )

    assert report["valid"] is False
    assert "chain_graph_encoder_unsupported" in report["error_codes"]


def test_preflight_accepts_classic_feature_chain_graph_route():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {"tokenizer": "kmer", "encoder": "onehot", "model": "gcn"},
    )

    assert report["valid"] is True
    assert report["normalized_config"]["sequence_model"] == "gcn"


def test_preflight_accepts_non_esm2_pretrained_residue_graph_route():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {"tokenizer": "prott5", "encoder": "index", "model": "gat"},
    )

    assert report["valid"] is True
    assert report["effective_route"]["route_id"] == "sequence.pretrained_residue_graph"


def test_preflight_accepts_esm2_residue_graph_export_mode():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {
            "tokenizer": "esm2",
            "encoder": "index",
            "model": "gcn",
            "sequence_model": "gcn",
            "training_mode": "export_embeddings",
            "classifier": "rf",
        },
    )

    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True


@pytest.mark.parametrize(
    ("task_type", "dataset_format", "config", "error_code"),
    [
        (
            "pep_function",
            "sequence_dir",
            {
                "tokenizer": "kmer",
                "encoder": "index",
                "model": "cnn",
                "sequence_model": "cnn",
                "training_mode": "end_to_end",
                "classifier": "rf",
            },
            "end_to_end_requires_mlp",
        ),
        (
            "pep_function",
            "sequence_dir",
            {
                "tokenizer": "kmer",
                "encoder": "index",
                "model": "cnn",
                "sequence_model": "cnn",
                "training_mode": "export_embeddings",
                "classifier": "mlp",
            },
            "export_embeddings_requires_classical_classifier",
        ),
        (
            "pep_protein_interaction",
            "pair_split_dir",
            {
                "tokenizer": "kmer",
                "encoder": "index",
                "model": "cnn",
                "pair_strategy": "single_stream",
                "sequence_model": "cnn",
                "training_mode": "end_to_end",
                "classifier": "rf",
            },
            "end_to_end_requires_mlp",
        ),
        (
            "pep_protein_interaction",
            "pair_split_dir",
            {
                "tokenizer": "esm2",
                "encoder": "ppi_fusion",
                "model": "cross_attn",
                "pair_strategy": "two_stream",
                "interaction_level": "token",
                "interaction_model": "cross_attn",
                "training_mode": "export_embeddings",
                "classifier": "mlp",
            },
            "export_embeddings_requires_classical_classifier",
        ),
    ],
)
def test_preflight_reports_training_mode_classifier_contract_error(
    task_type, dataset_format, config, error_code
):
    report = preflight_experiment(
        _dataset(task_type, dataset_format, {"train": 10, "val": 2, "test": 2}),
        task_type,
        config,
    )
    assert report["valid"] is False
    assert error_code in report["error_codes"]


def test_preflight_rejects_esm2_residue_graph_non_mlp_head():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {
            "tokenizer": "esm2",
            "encoder": "index",
            "model": "gat",
            "sequence_model": "gat",
            "training_mode": "end_to_end",
            "classifier": "rf",
        },
    )

    assert report["valid"] is False
    assert "pretrained_residue_end_to_end_requires_mlp" in report["error_codes"]


@pytest.mark.parametrize("adapter_dim", [15, 18, 1028, "wide"])
def test_preflight_rejects_invalid_pretrained_residue_adapter_dimension(adapter_dim):
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {
            "tokenizer": "esm2",
            "encoder": "esm2",
            "sequence_model": "lstm",
            "model": "lstm",
            "training_mode": "end_to_end",
            "classifier": "mlp",
            "residue_adapter_dim": adapter_dim,
        },
    )

    assert report["valid"] is False
    assert "normalization_error" in report["error_codes"]


def test_preflight_keeps_pair_pretrained_cross_attention_route_valid():
    report = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_protein_interaction",
        {
            "tokenizer": "esm2",
            "encoder": "ppi_fusion",
            "model": "cross_attn",
            "pair_strategy": "two_stream",
            "interaction_level": "token",
            "interaction_model": "cross_attn",
        },
    )

    assert report["valid"] is True
    assert report["normalized_config"]["interaction_model"] == "cross_attn"


def test_preflight_rejects_pair_attention_hidden_dim_not_divisible_by_heads():
    report = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_protein_interaction",
        {
            "tokenizer": "esm2",
            "encoder": "ppi_fusion",
            "model": "cross_attn",
            "pair_strategy": "two_stream",
            "interaction_level": "token",
            "interaction_model": "cross_attn",
            "interaction_hidden_dim": 7,
            "interaction_heads": 2,
        },
    )

    assert report["valid"] is False
    assert "pair_attention_dimension_invalid" in report["error_codes"]


def test_preflight_marks_pretrained_classical_function_route_reusable():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {"tokenizer": "esm2", "encoder": "index", "model": "logreg"},
    )
    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True


def test_preflight_warns_when_pretrained_model_will_download_remotely(tmp_path, monkeypatch):
    monkeypatch.delenv("PEPPLATFORM_OFFLINE_MODE", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PEPPLATFORM_HF_ENDPOINT", "https://hf-mirror.example")

    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {
            "tokenizer": "protbert",
            "encoder": "index",
            "model": "mlp",
            "protbert_model": "Rostlab/prot_bert",
        },
    )

    assert report["valid"] is True
    assert "pretrained_model_remote_download" in report["warning_codes"]
    assert "https://hf-mirror.example" in report["warnings"][0]


def test_preflight_normalizes_legacy_probert_tokenizer(tmp_path, monkeypatch):
    monkeypatch.delenv("PEPPLATFORM_OFFLINE_MODE", raising=False)
    local_model = tmp_path / "probert"
    local_model.mkdir()

    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {
            "tokenizer": "probert",
            "encoder": "index",
            "model": "mlp",
            "probert_model": str(local_model),
        },
    )

    assert report["valid"] is True
    assert report["normalized_config"]["tokenizer"] == "protbert"
    assert report["normalized_config"]["protbert_model"] == str(local_model)
    assert "pretrained_model_remote_download" not in report["warning_codes"]


def test_preflight_accepts_local_pretrained_model_without_remote_warning(tmp_path, monkeypatch):
    monkeypatch.delenv("PEPPLATFORM_OFFLINE_MODE", raising=False)
    local_model = tmp_path / "protbert"
    local_model.mkdir()

    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {
            "tokenizer": "protbert",
            "encoder": "index",
            "model": "mlp",
            "protbert_model": str(local_model),
        },
    )

    assert report["valid"] is True
    assert "pretrained_model_remote_download" not in report["warning_codes"]
    assert "pretrained_model_local_cache" in report["warning_codes"]


def test_preflight_accepts_cached_pretrained_model_without_remote_warning(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    repo_cache = cache / "models--Rostlab--prot_bert"
    snapshot = repo_cache / "snapshots" / "complete"
    snapshot.mkdir(parents=True)
    (repo_cache / "refs").mkdir()
    (repo_cache / "refs" / "main").write_text("complete", encoding="utf-8")
    for filename in ["config.json", "tokenizer_config.json", "vocab.txt", "pytorch_model.bin"]:
        (snapshot / filename).write_text("ok", encoding="utf-8")
    monkeypatch.delenv("PEPPLATFORM_OFFLINE_MODE", raising=False)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache))

    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {
            "tokenizer": "protbert",
            "encoder": "index",
            "model": "mlp",
            "protbert_model": "Rostlab/prot_bert",
        },
    )

    assert report["valid"] is True
    assert "pretrained_model_remote_download" not in report["warning_codes"]
    assert "pretrained_model_local_cache" in report["warning_codes"]


def test_preflight_rejects_remote_pretrained_model_in_offline_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("PEPPLATFORM_OFFLINE_MODE", "1")
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {
            "tokenizer": "protbert",
            "encoder": "index",
            "model": "mlp",
            "protbert_model": "Rostlab/prot_bert",
        },
    )

    assert report["valid"] is False
    assert "pretrained_model_source_unavailable" in report["error_codes"]


def test_preflight_marks_non_profile_flattened_sequence_features_reusable():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {"tokenizer": "kmer", "encoder": "onehot", "model": "rf", "classifier": "rf"},
    )

    assert report["valid"] is True
    assert report["normalized_config"]["model"] == "rf"
    assert report["normalized_config"]["sequence_model"] == ""
    assert report["execution"]["supports_reusable_inference"] is True


def test_preflight_marks_profile_flattened_sequence_features_reusable_with_blast_warning():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {"tokenizer": "kmer", "encoder": "pssm", "model": "rf", "classifier": "rf"},
    )

    assert report["valid"] is True
    assert report["normalized_config"]["model"] == "rf"
    assert report["normalized_config"]["sequence_model"] == ""
    assert report["execution"]["supports_reusable_inference"] is True
    assert "profile_inference_requires_blast" in report["warning_codes"]
    assert "profile_inference_unavailable" not in report["warning_codes"]


def test_preflight_marks_profile_sequence_route_reusable_with_blast_warning():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {"tokenizer": "kmer", "encoder": "pssm", "model": "cnn", "sequence_model": "cnn"},
    )

    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True
    assert "profile_inference_requires_blast" in report["warning_codes"]


def test_preflight_marks_pair_pretrained_vector_route_reusable():
    report = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_protein_interaction",
        {"tokenizer": "esm2", "encoder": "ppi_fusion", "model": "mlp"},
    )
    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True
    assert "inference_unavailable" not in report["warning_codes"]


def test_preflight_marks_hlai_pretrained_vector_route_reusable():
    report = preflight_experiment(
        _dataset("pep_hlai", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_hlai",
        {"tokenizer": "esm2", "encoder": "ppi_fusion", "model": "mlp"},
    )
    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True
    assert "hlai_route_inference_unavailable" not in report["warning_codes"]


def test_preflight_marks_hlaii_single_stream_route_reusable():
    report = preflight_experiment(
        _dataset("pep_hlaii", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_hlaii",
        {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
    )
    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True
    assert "hlaii_inference_unavailable" not in report["warning_codes"]


def test_preflight_marks_hlaii_pretrained_vector_route_reusable():
    report = preflight_experiment(
        _dataset("pep_hlaii", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_hlaii",
        {
            "tokenizer": "esm2",
            "encoder": "ppi_fusion",
            "model": "mlp",
            "pair_strategy": "two_stream",
            "interaction_level": "vector",
            "predictor_type": "classifier",
        },
    )
    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True
    assert "hlaii_inference_unavailable" not in report["warning_codes"]


def test_preflight_marks_hlaii_token_route_reusable():
    report = preflight_experiment(
        _dataset("pep_hlaii", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_hlaii",
        {
            "tokenizer": "esm2",
            "encoder": "ppi_fusion",
            "model": "cross_attn",
            "pair_strategy": "two_stream",
            "interaction_level": "token",
            "interaction_model": "cross_attn",
        },
    )
    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True
    assert "hlaii_inference_unavailable" not in report["warning_codes"]


def test_preflight_marks_pair_pretrained_classical_vector_route_reusable():
    report = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_protein_interaction",
        {"tokenizer": "esm2", "encoder": "ppi_fusion", "model": "logreg", "classifier": "logreg"},
    )
    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True


def test_preflight_marks_hlaii_pretrained_classical_vector_route_reusable():
    report = preflight_experiment(
        _dataset("pep_hlaii", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_hlaii",
        {
            "tokenizer": "esm2",
            "encoder": "ppi_fusion",
            "model": "logreg",
            "classifier": "logreg",
            "pair_strategy": "two_stream",
            "interaction_level": "vector",
            "predictor_type": "classifier",
        },
    )
    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True


def test_preflight_marks_pair_classic_glove_rf_vector_route_reusable():
    report = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_protein_interaction",
        {
            "tokenizer": "kmer",
            "encoder": "glove",
            "model": "rf",
            "classifier": "rf",
            "pair_strategy": "two_stream",
            "interaction_level": "vector",
            "predictor_type": "classifier",
            "fusion_mode": "concat_all",
        },
    )

    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True
    assert "inference_unavailable" not in report["warning_codes"]


def test_preflight_marks_pair_classic_tfidf_vector_route_reusable():
    report = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_protein_interaction",
        {
            "tokenizer": "kmer",
            "encoder": "tfidf",
            "model": "rf",
            "classifier": "rf",
            "pair_strategy": "two_stream",
            "interaction_level": "vector",
            "predictor_type": "classifier",
            "fusion_mode": "concat_all",
        },
    )

    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True
    assert "inference_unavailable" not in report["warning_codes"]


@pytest.mark.parametrize(
    ("encoder", "message"),
    [
        ("cs", "CS encoder has been removed"),
        ("property", "Property encoder has been removed"),
        ("physchem", "Property encoder has been removed"),
    ],
)
def test_preflight_rejects_removed_encoder(encoder, message):
    report = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_protein_interaction",
        {
            "tokenizer": "kmer",
            "encoder": encoder,
            "model": "rf",
            "classifier": "rf",
            "pair_strategy": "two_stream",
            "interaction_level": "vector",
            "predictor_type": "classifier",
            "fusion_mode": "concat_all",
        },
    )

    assert report["valid"] is False
    assert "normalization_error" in report["error_codes"]
    assert any(message in error for error in report["errors"])


def test_preflight_rejects_removed_labelled_lda():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2}),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "bow",
            "model": "rf",
            "classifier": "rf",
            "reduction": "labelled_lda",
        },
    )

    assert report["valid"] is False
    assert "normalization_error" in report["error_codes"]
    assert any(
        "Labelled LDA feature transformation has been removed" in error
        for error in report["errors"]
    )


def test_preflight_rejects_pair_lsa_with_specific_failure_code():
    report = preflight_experiment(
        _dataset(
            "pep_protein_interaction",
            "pair_split_dir",
            {"train": 10, "val": 2, "test": 2},
        ),
        "pep_protein_interaction",
        {
            "tokenizer": "kmer",
            "encoder": "tfidf",
            "model": "rf",
            "classifier": "rf",
            "pair_strategy": "two_stream",
            "interaction_level": "vector",
            "predictor_type": "classifier",
            "fusion_mode": "concat_all",
            "reduction": "lsa",
            "lsa_dim": 2,
        },
    )

    assert report["valid"] is False
    assert "lsa_pair_route_unsupported" in report["error_codes"]
    assert any("ml_reduction=tsvd" in error for error in report["errors"])


def test_preflight_rejects_invalid_lsa_memory_limit():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2}),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "bow",
            "model": "rf",
            "classifier": "rf",
            "reduction": "lsa",
            "lsa_dim": 2,
            "lsa_dense_memory_limit_gib": 0,
        },
    )

    assert report["valid"] is False
    assert "lsa_memory_limit_invalid" in report["error_codes"]


def test_preflight_rejects_invalid_tsvd_memory_limit():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2}),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "bow",
            "model": "rf",
            "classifier": "rf",
            "ml_reduction": "tsvd",
            "ml_reduction_dim": 2,
            "tsvd_dense_memory_limit_gib": 0,
        },
    )

    assert report["valid"] is False
    assert "tsvd_memory_limit_invalid" in report["error_codes"]


@pytest.mark.parametrize("kernel", ["rbf", "poly"])
def test_preflight_materializes_kpca_gamma_default(kernel):
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2}),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "bow",
            "model": "rf",
            "classifier": "rf",
            "ml_reduction": "kpca",
            "ml_reduction_dim": 2,
            "kpca_kernel": kernel,
        },
    )

    assert report["valid"] is True
    assert report["normalized_config"]["kpca_gamma"] == 0.1
    assert report["normalized_config"]["kpca_memory_limit_gib"] == 4.0


@pytest.mark.parametrize("kernel", ["linear", "cosine"])
def test_preflight_accepts_kpca_kernel_without_inapplicable_gamma(kernel):
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2}),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "bow",
            "model": "rf",
            "classifier": "rf",
            "ml_reduction": "kpca",
            "ml_reduction_dim": 2,
            "kpca_kernel": kernel,
        },
    )

    assert report["valid"] is True
    assert report["normalized_config"]["kpca_gamma"] is None


@pytest.mark.parametrize(
    ("updates", "code"),
    [
        ({"kpca_kernel": "rbf", "kpca_gamma": 0}, "kpca_gamma_invalid"),
        ({"kpca_kernel": "rbf", "kpca_gamma": -0.1}, "kpca_gamma_invalid"),
        ({"kpca_kernel": "linear", "kpca_gamma": 0.1}, "kpca_gamma_not_applicable"),
        ({"kpca_kernel": "sigmoid"}, "kpca_kernel_invalid"),
        ({"kpca_memory_limit_gib": 0}, "kpca_memory_limit_invalid"),
    ],
)
def test_preflight_rejects_invalid_kpca_configuration(updates, code):
    config = {
        "tokenizer": "kmer",
        "encoder": "bow",
        "model": "rf",
        "classifier": "rf",
        "ml_reduction": "kpca",
        "ml_reduction_dim": 2,
    }
    config.update(updates)
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2}),
        "pep_function",
        config,
    )

    assert report["valid"] is False
    assert code in report["error_codes"]


def test_preflight_rejects_invalid_pca_memory_limit():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2}),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "bow",
            "model": "rf",
            "classifier": "rf",
            "ml_reduction": "pca",
            "ml_reduction_dim": 2,
            "pca_dense_memory_limit_gib": 0,
        },
    )

    assert report["valid"] is False
    assert "pca_memory_limit_invalid" in report["error_codes"]


def test_preflight_rejects_pair_lda_with_non_strict_fusion():
    report = preflight_experiment(
        _dataset(
            "pep_protein_interaction",
            "pair_split_dir",
            {"train": 10, "val": 2, "test": 2},
        ),
        "pep_protein_interaction",
        {
            "tokenizer": "kmer",
            "encoder": "bow",
            "model": "rf",
            "classifier": "rf",
            "pair_strategy": "two_stream",
            "interaction_level": "vector",
            "predictor_type": "classifier",
            "fusion_mode": "concat_all",
            "reduction": "lda",
            "lda_topics": 2,
        },
    )

    assert report["valid"] is False
    assert "pair_bow_fusion_unsupported" in report["error_codes"]
    assert any("fusion_mode=concat" in error for error in report["errors"])


def test_preflight_rejects_invalid_lda_memory_limit():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2}),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "bow",
            "model": "rf",
            "classifier": "rf",
            "reduction": "lda",
            "lda_topics": 2,
            "lda_dense_memory_limit_gib": 0,
        },
    )

    assert report["valid"] is False
    assert "lda_memory_limit_invalid" in report["error_codes"]


@pytest.mark.parametrize(
    ("reduction", "control"),
    [("lda", {"lda_topics": 2}), ("plsa", {"plsa_topics": 2})],
)
def test_preflight_accepts_strict_pair_bow_topic_routes(reduction, control):
    report = preflight_experiment(
        _dataset(
            "pep_protein_interaction",
            "pair_split_dir",
            {"train": 10, "val": 2, "test": 2},
        ),
        "pep_protein_interaction",
        {
            "tokenizer": "kmer",
            "encoder": "bow",
            "model": "rf",
            "classifier": "rf",
            "pair_strategy": "two_stream",
            "interaction_level": "vector",
            "predictor_type": "classifier",
            "fusion_mode": "concat",
            "reduction": reduction,
            **control,
        },
    )

    assert report["valid"] is True
    assert report["normalized_config"]["reduction"] == reduction


def test_preflight_rejects_lda_on_weighted_encoder():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2}),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "tfidf",
            "model": "rf",
            "classifier": "rf",
            "reduction": "lda",
            "lda_topics": 2,
        },
    )

    assert report["valid"] is False
    assert "lda_encoder_unsupported" in report["error_codes"]


def test_preflight_rejects_pair_plsa_with_non_strict_fusion():
    report = preflight_experiment(
        _dataset(
            "pep_protein_interaction",
            "pair_split_dir",
            {"train": 10, "val": 2, "test": 2},
        ),
        "pep_protein_interaction",
        {
            "tokenizer": "kmer",
            "encoder": "bow",
            "model": "rf",
            "classifier": "rf",
            "pair_strategy": "two_stream",
            "interaction_level": "vector",
            "predictor_type": "classifier",
            "fusion_mode": "concat_all",
            "reduction": "plsa",
            "plsa_topics": 2,
        },
    )

    assert report["valid"] is False
    assert "pair_bow_fusion_unsupported" in report["error_codes"]


def test_preflight_rejects_invalid_plsa_memory_limit():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2}),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "bow",
            "model": "rf",
            "classifier": "rf",
            "reduction": "plsa",
            "plsa_topics": 2,
            "plsa_memory_limit_gib": 0,
        },
    )

    assert report["valid"] is False
    assert "plsa_memory_limit_invalid" in report["error_codes"]


def test_preflight_rejects_plsa_on_weighted_encoder():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2}),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "tfidf",
            "model": "rf",
            "classifier": "rf",
            "reduction": "plsa",
            "plsa_topics": 2,
        },
    )

    assert report["valid"] is False
    assert "plsa_encoder_unsupported" in report["error_codes"]


def test_preflight_rejects_removed_mismatch_tokenizer():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2}),
        "pep_function",
        {"tokenizer": "mismatch", "encoder": "tfidf", "model": "rf"},
    )

    assert report["valid"] is False
    assert "route_resolution_error" in report["error_codes"]


def test_preflight_marks_pair_top_ngram_vector_route_reusable_with_blast_warning():
    report = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_protein_interaction",
        {
            "tokenizer": "top-n-gram",
            "encoder": "bow",
            "model": "rf",
            "classifier": "rf",
            "pair_strategy": "two_stream",
            "interaction_level": "vector",
            "predictor_type": "classifier",
            "fusion_mode": "concat_all",
        },
    )

    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True
    assert "profile_inference_requires_blast" in report["warning_codes"]


def test_preflight_enforces_top_ngram_v2_range_and_no_fallback():
    dataset = _dataset(
        "pep_protein_interaction",
        "pair_split_dir",
        {"train": 10, "val": 2, "test": 2},
    )
    base = {
        "tokenizer": "top-n-gram",
        "encoder": "bow",
        "model": "rf",
        "classifier": "rf",
        "pair_strategy": "two_stream",
        "interaction_level": "vector",
        "predictor_type": "classifier",
        "fusion_mode": "concat_all",
    }

    invalid_n = preflight_experiment(
        dataset,
        "pep_protein_interaction",
        {**base, "top_ngram_n": 4},
    )
    assert invalid_n["valid"] is False
    assert set(invalid_n["error_codes"]).intersection(
        {"normalization_error", "invalid_numeric_parameter"}
    )

    invalid_fallback = preflight_experiment(
        dataset,
        "pep_protein_interaction",
        {**base, "profile_on_missing": "blosum62"},
    )
    assert invalid_fallback["valid"] is False
    assert "top_ngram_fallback_unsupported" in invalid_fallback["error_codes"]

    implicit_fallback = preflight_experiment(
        dataset,
        "pep_protein_interaction",
        base,
    )
    assert implicit_fallback["valid"] is True
    assert (
        implicit_fallback["normalized_config"]["profile_on_missing"] == "blosum62_target_frequency"
    )
    assert "top_ngram_profile_fallback_enabled" in implicit_fallback["warning_codes"]

    explicit_strict = preflight_experiment(
        dataset,
        "pep_protein_interaction",
        {**base, "profile_on_missing": "error"},
    )
    assert explicit_strict["valid"] is True
    assert "top_ngram_profile_fallback_enabled" not in explicit_strict["warning_codes"]


def test_preflight_enforces_dt_profile_fallback_contract_and_removed_hash():
    dataset = _dataset(
        "pep_function",
        "sequence_dir",
        {"train": 10, "val": 2, "test": 2},
    )
    base = {
        "tokenizer": "dt",
        "encoder": "bow",
        "model": "mlp",
        "classifier": "mlp",
        "max_length": 60,
    }

    implicit_fallback = preflight_experiment(dataset, "pep_function", base)
    assert implicit_fallback["valid"] is True
    assert (
        implicit_fallback["normalized_config"]["profile_on_missing"] == "blosum62_target_frequency"
    )
    assert implicit_fallback["parameter_sources"]["profile_on_missing"] == "default"
    assert "dt_profile_fallback_enabled" in implicit_fallback["warning_codes"]

    explicit_strict = preflight_experiment(
        dataset,
        "pep_function",
        {**base, "profile_on_missing": "error"},
    )
    assert explicit_strict["valid"] is True
    assert explicit_strict["normalized_config"]["profile_on_missing"] == "error"
    assert explicit_strict["parameter_sources"]["profile_on_missing"] == "explicit"
    assert "dt_profile_fallback_enabled" not in explicit_strict["warning_codes"]

    legacy_fallback = preflight_experiment(
        dataset,
        "pep_function",
        {**base, "profile_on_missing": "blosum62"},
    )
    assert legacy_fallback["valid"] is False
    assert "dt_fallback_unsupported" in legacy_fallback["error_codes"]

    legacy_hash = preflight_experiment(
        dataset,
        "pep_function",
        {**base, "dt_hash_size": 200_000},
    )
    assert legacy_hash["valid"] is False
    assert "dt_hash_size_removed" in legacy_hash["error_codes"]
    hash_error = legacy_hash["errors"][legacy_hash["error_codes"].index("dt_hash_size_removed")]
    assert "remove it and retrain with the DT v1 contract" in hash_error


def test_preflight_resolves_dt_dataset_lengths_and_reports_missing_stats():
    dataset = _dataset(
        "pep_function",
        "sequence_dir",
        {"train": 10, "val": 2, "test": 2},
    )
    dataset["metadata"]["length_stats"]["overall"]["sequence"]["max"] = 207
    config = {
        "tokenizer": "dt",
        "encoder": "bow",
        "model": "mlp",
        "classifier": "mlp",
        "max_length": 60,
    }

    adjusted = preflight_experiment(dataset, "pep_function", config)
    assert adjusted["valid"] is True
    assert adjusted["normalized_config"]["max_length"] == 207
    assert adjusted["parameter_sources"]["max_length"] == "dataset_derived"
    assert "dt_max_length_auto_adjusted" in adjusted["warning_codes"]

    missing_stats_dataset = _dataset("pep_function", "sequence_dir")
    del missing_stats_dataset["metadata"]["length_stats"]
    missing_stats = preflight_experiment(
        missing_stats_dataset,
        "pep_function",
        config,
    )
    assert missing_stats["valid"] is False
    assert "dt_length_stats_unavailable" in missing_stats["error_codes"]


def test_preflight_accepts_homogeneous_and_heterogeneous_pair_dt_streams():
    dataset = _dataset(
        "pep_protein_interaction",
        "pair_split_dir",
        {"train": 10, "val": 2, "test": 2},
    )
    pair_route = {
        "model": "mlp",
        "classifier": "mlp",
        "pair_strategy": "two_stream",
        "interaction_level": "vector",
        "predictor_type": "classifier",
        "fusion_mode": "concat",
    }

    homogeneous = preflight_experiment(
        dataset,
        "pep_protein_interaction",
        {
            **pair_route,
            "tokenizer": "dt",
            "encoder": "bow",
        },
    )
    assert homogeneous["valid"] is True
    assert homogeneous["normalized_config"]["stream_mode"] == "homogeneous"
    assert homogeneous["normalized_config"]["profile_on_missing"] == "blosum62_target_frequency"
    assert "dt_profile_fallback_enabled" in homogeneous["warning_codes"]

    heterogeneous = preflight_experiment(
        dataset,
        "pep_protein_interaction",
        {
            **pair_route,
            "tokenizer": "kmer",
            "encoder": "bow",
            "streams": {
                "peptide": {
                    "feature_route": "classic",
                    "tokenizer": "kmer",
                    "encoder": "bow",
                    "max_length": 60,
                    "params": {"k": 2},
                },
                "partner": {
                    "feature_route": "classic",
                    "tokenizer": "dt",
                    "encoder": "bow",
                    "max_length": 60,
                    "params": {"dt_max_distance": 5},
                },
            },
        },
    )
    assert heterogeneous["valid"] is True
    assert heterogeneous["normalized_config"]["stream_mode"] == "heterogeneous"
    assert heterogeneous["normalized_config"]["profile_on_missing"] == "blosum62_target_frequency"
    assert "dt_profile_fallback_enabled" in heterogeneous["warning_codes"]
    assert "profile_inference_requires_blast" in heterogeneous["warning_codes"]


def test_preflight_auto_adjusts_sequence_top_ngram_max_length():
    dataset = _dataset(
        "pep_function",
        "sequence_dir",
        {"train": 10, "val": 2, "test": 2},
    )
    dataset["metadata"]["length_stats"]["overall"]["sequence"]["max"] = 207

    report = preflight_experiment(
        dataset,
        "pep_function",
        {
            "tokenizer": "top-n-gram",
            "encoder": "bow",
            "model": "mlp",
            "classifier": "mlp",
            "max_length": 60,
        },
    )

    assert report["valid"] is True
    assert report["normalized_config"]["max_length"] == 207
    assert report["parameter_sources"]["max_length"] == "dataset_derived"
    assert "top_ngram_max_length_auto_adjusted" in report["warning_codes"]


def test_preflight_rejects_only_top_ngram_when_dataset_exceeds_platform_limit():
    dataset = _dataset("pep_function", "sequence_dir")
    dataset["metadata"]["length_stats"]["overall"]["sequence"]["max"] = 10_001

    top_ngram = preflight_experiment(
        dataset,
        "pep_function",
        {
            "tokenizer": "top-n-gram",
            "encoder": "bow",
            "model": "mlp",
            "classifier": "mlp",
        },
    )
    kmer = preflight_experiment(
        dataset,
        "pep_function",
        {"tokenizer": "kmer", "encoder": "bow", "model": "mlp"},
    )

    assert top_ngram["valid"] is False
    assert "top_ngram_sequence_limit_exceeded" in top_ngram["error_codes"]
    assert "top_ngram_sequence_limit_exceeded" not in kmer["error_codes"]


def test_preflight_marks_pair_aligned_gated_mlp_route_reusable():
    report = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_protein_interaction",
        {
            "tokenizer": "esm2",
            "encoder": "ppi_fusion",
            "model": "mlp",
            "classifier": "mlp",
            "pair_strategy": "two_stream",
            "interaction_level": "vector",
            "predictor_type": "classifier",
            "fusion_mode": "aligned_gated",
        },
    )

    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True
    assert "inference_unavailable" not in report["warning_codes"]


def test_preflight_marks_pair_classic_pssm_xgb_vector_route_reusable_with_blast_warning():
    report = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_protein_interaction",
        {
            "tokenizer": "kmer",
            "encoder": "pssm",
            "model": "xgb",
            "classifier": "xgb",
            "pair_strategy": "two_stream",
            "interaction_level": "vector",
            "predictor_type": "classifier",
            "fusion_mode": "concat_all",
        },
    )

    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True
    assert "profile_inference_requires_blast" in report["warning_codes"]
    assert "profile_inference_unavailable" not in report["warning_codes"]


def test_preflight_marks_pair_single_stream_end_to_end_route_reusable():
    report = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_protein_interaction",
        {
            "tokenizer": "kmer",
            "encoder": "index",
            "model": "cnn",
            "pair_strategy": "single_stream",
            "sequence_model": "cnn",
            "training_mode": "end_to_end",
            "classifier": "mlp",
        },
    )

    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True
    assert "inference_unavailable" not in report["warning_codes"]


def test_preflight_marks_pair_single_stream_export_route_reusable():
    report = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_protein_interaction",
        {
            "tokenizer": "kmer",
            "encoder": "index",
            "model": "cnn",
            "sequence_model": "cnn",
            "pair_strategy": "single_stream",
            "training_mode": "export_embeddings",
            "classifier": "logreg",
        },
    )

    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True
    assert "route_inference_unavailable" not in report["warning_codes"]


def test_preflight_marks_hlaii_single_stream_export_route_reusable():
    report = preflight_experiment(
        _dataset("pep_hlaii", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_hlaii",
        {
            "tokenizer": "kmer",
            "encoder": "index",
            "model": "cnn",
            "sequence_model": "cnn",
            "pair_strategy": "single_stream",
            "training_mode": "export_embeddings",
            "classifier": "logreg",
        },
    )

    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True
    assert "route_inference_unavailable" not in report["warning_codes"]


def test_preflight_marks_hlai_single_stream_export_route_reusable():
    report = preflight_experiment(
        _dataset("pep_hlai", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_hlai",
        {
            "tokenizer": "kmer",
            "encoder": "index",
            "model": "cnn",
            "sequence_model": "cnn",
            "pair_strategy": "single_stream",
            "training_mode": "export_embeddings",
            "classifier": "logreg",
        },
    )

    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True
    assert "hlai_route_inference_unavailable" not in report["warning_codes"]


def test_preflight_marks_pair_token_cross_attention_route_reusable():
    report = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_protein_interaction",
        {
            "tokenizer": "esm2",
            "encoder": "ppi_fusion",
            "model": "cross_attn",
            "pair_strategy": "two_stream",
            "interaction_level": "token",
            "interaction_model": "cross_attn",
        },
    )
    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True
    assert "inference_unavailable" not in report["warning_codes"]


def test_preflight_marks_pair_token_bi_cross_attention_route_reusable():
    report = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_protein_interaction",
        {
            "tokenizer": "esm2",
            "encoder": "ppi_fusion",
            "model": "bi_cross_attn",
            "pair_strategy": "two_stream",
            "interaction_level": "token",
            "interaction_model": "bi_cross_attn",
        },
    )
    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True
    assert "inference_unavailable" not in report["warning_codes"]


def test_preflight_marks_pair_token_classic_pssm_route_reusable_with_blast_warning():
    report = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_protein_interaction",
        {
            "tokenizer": "kmer",
            "encoder": "pssm",
            "model": "bi_cross_attn",
            "pair_strategy": "two_stream",
            "interaction_level": "token",
            "interaction_model": "bi_cross_attn",
        },
    )
    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True
    assert "profile_inference_requires_blast" in report["warning_codes"]
    assert "profile_inference_unavailable" not in report["warning_codes"]


def test_preflight_marks_pair_token_export_route_reusable():
    report = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_protein_interaction",
        {
            "tokenizer": "esm2",
            "encoder": "ppi_fusion",
            "model": "cross_attn",
            "pair_strategy": "two_stream",
            "interaction_level": "token",
            "interaction_model": "cross_attn",
            "training_mode": "export_embeddings",
            "classifier": "logreg",
        },
    )
    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True
    assert "inference_unavailable" not in report["warning_codes"]


def test_preflight_marks_heterogeneous_pair_token_export_route_reusable():
    report = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_protein_interaction",
        {
            "tokenizer": "kmer",
            "encoder": "index",
            "model": "cross_attn",
            "pair_strategy": "two_stream",
            "interaction_level": "token",
            "interaction_model": "cross_attn",
            "training_mode": "export_embeddings",
            "classifier": "rf",
            "streams": {
                "peptide": {
                    "feature_route": "classic",
                    "tokenizer": "kmer",
                    "encoder": "index",
                    "max_length": 60,
                    "params": {"k": 2},
                },
                "partner": {
                    "feature_route": "classic",
                    "tokenizer": "dr",
                    "encoder": "index",
                    "max_length": 60,
                    "params": {},
                },
            },
        },
    )
    assert report["valid"] is True
    assert report["normalized_config"]["stream_mode"] == "heterogeneous"
    assert report["execution"]["supports_reusable_inference"] is True


def test_preflight_marks_hlaii_pair_token_export_route_reusable():
    report = preflight_experiment(
        _dataset("pep_hlaii", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_hlaii",
        {
            "tokenizer": "esm2",
            "encoder": "ppi_fusion",
            "model": "cross_attn",
            "pair_strategy": "two_stream",
            "interaction_level": "token",
            "interaction_model": "cross_attn",
            "training_mode": "export_embeddings",
            "classifier": "logreg",
        },
    )
    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True
    assert "inference_unavailable" not in report["warning_codes"]


def test_preflight_marks_hlai_pair_token_export_route_reusable():
    report = preflight_experiment(
        _dataset("pep_hlai", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_hlai",
        {
            "tokenizer": "esm2",
            "encoder": "ppi_fusion",
            "model": "cross_attn",
            "pair_strategy": "two_stream",
            "interaction_level": "token",
            "interaction_model": "cross_attn",
            "training_mode": "export_embeddings",
            "classifier": "logreg",
        },
    )
    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True
    assert "inference_unavailable" not in report["warning_codes"]


def test_preflight_rejects_invalid_custom_numeric_parameters():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "index",
            "model": "cnn",
            "k": 0,
            "dropout": 1.0,
            "semantic_similarity_weight": -0.1,
        },
    )

    assert report["valid"] is False
    assert report["error_codes"] == [
        "invalid_numeric_parameter",
        "invalid_numeric_parameter",
        "invalid_numeric_parameter",
    ]
    assert any(
        "K-mer tokenizer k must be an integer in [1, 3]" in message for message in report["errors"]
    )
    assert any("dropout must be < 1.0" in message for message in report["errors"])
    assert any(
        "semantic_similarity_weight must be >= 0.0" in message for message in report["errors"]
    )


@pytest.mark.parametrize("key", ["xgb_subsample", "xgb_colsample"])
def test_preflight_rejects_zero_xgb_sampling_fraction(key):
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2}),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "bow",
            "model": "xgb",
            "classifier": "xgb",
            key: 0.0,
        },
    )

    assert report["valid"] is False
    assert "invalid_numeric_parameter" in report["error_codes"]
    assert any(f"{key} must be > 0.0" in message for message in report["errors"])


def test_preflight_rejects_zero_logreg_c_and_invalid_convergence_policy():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2}),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "bow",
            "model": "logreg",
            "classifier": "logreg",
            "logreg_c": 0.0,
            "logreg_convergence_policy": "unbounded",
        },
    )

    assert report["valid"] is False
    assert "invalid_numeric_parameter" in report["error_codes"]
    assert "invalid_choice_parameter" in report["error_codes"]


def test_preflight_defaults_web_logreg_to_bounded_retry():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2}),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "bow",
            "model": "logreg",
            "classifier": "logreg",
        },
    )

    assert report["valid"] is True
    assert report["normalized_config"]["logreg_convergence_policy"] == "bounded_retry"


def test_preflight_enforces_standard_kmer_upper_bound_only_for_kmer():
    dataset = _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2})
    invalid = preflight_experiment(
        dataset,
        "pep_function",
        {"tokenizer": "kmer", "encoder": "index", "model": "cnn", "k": 4},
    )
    valid = preflight_experiment(
        dataset,
        "pep_function",
        {"tokenizer": "kmer", "encoder": "index", "model": "cnn", "k": 3},
    )
    other = preflight_experiment(
        dataset,
        "pep_function",
        {"tokenizer": "dr", "encoder": "index", "model": "cnn", "k": 4},
    )

    assert invalid["valid"] is False
    assert "invalid_numeric_parameter" in invalid["error_codes"]
    assert any(
        "K-mer tokenizer k must be an integer in [1, 3]" in message for message in invalid["errors"]
    )
    assert valid["valid"] is True
    assert other["valid"] is True


def test_preflight_requires_two_profile_iterations():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "pssm",
            "model": "cnn",
            "profile_iterations": 1,
        },
    )

    assert report["valid"] is False
    assert "profile_iterations_too_low" in report["error_codes"]
    assert any("profile_iterations must be >= 2" in message for message in report["errors"])


def test_preflight_rejects_invalid_tuning_parameters():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "onehot",
            "model": "svm",
            "ml_auto_tune": True,
            "ml_cv_folds": 6,
            "ml_tuning_objective": "unknown_metric",
            "svm_kernel": "rbf",
            "svm_gamma": "0",
        },
    )

    assert report["valid"] is False
    assert "invalid_choice_parameter" in report["error_codes"]
    assert "invalid_numeric_parameter" in report["error_codes"]


def test_preflight_accepts_custom_semantic_similarity_parameters():
    report = preflight_experiment(
        _dataset("pep_hlai", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_hlai",
        {
            "tokenizer": "esm2",
            "encoder": "ppi_fusion",
            "model": "mlp",
            "enable_semantic_similarity": True,
            "semantic_similarity_weight": 0.2,
            "semantic_zero_weight": 0.15,
        },
    )

    assert report["valid"] is True
    assert report["normalized_config"]["enable_semantic_similarity"] is True
    assert report["normalized_config"]["semantic_similarity_weight"] == 0.2
    assert report["normalized_config"]["semantic_zero_weight"] == 0.15
    assert report["normalized_config"]["semantic_pair_mode"] == "role_fused"
    assert report["normalized_config"]["semantic_projection_dim"] == 64
    assert report["normalized_config"]["semantic_contrastive_temperature"] == 0.1
    assert report["normalized_config"]["semantic_positive_partners"] == 1
    assert report["normalized_config"]["semantic_zero_partners"] == 8
    assert "semantic_loss_type" not in report["normalized_config"]


@pytest.mark.parametrize(
    "task", ["pep_function", "pep_protein_interaction", "pep_hlaii", "pep_hlai"]
)
@pytest.mark.parametrize("loss_type", ["legacy", "jaccard_supcon", "invalid"])
def test_preflight_semantic_loss_selection(task, loss_type):
    pair = task != "pep_function"
    report = preflight_experiment(
        _dataset(
            task, "pair_split_dir" if pair else "sequence_dir", {"train": 10, "val": 2, "test": 2}
        ),
        task,
        {
            "tokenizer": "esm2" if pair else "kmer",
            "encoder": "ppi_fusion" if pair else "onehot",
            "model": "mlp" if pair else "cnn",
            "enable_semantic_similarity": True,
            "semantic_loss_type": loss_type,
        },
    )
    if loss_type == "invalid":
        assert "semantic_loss_type_invalid" in report["error_codes"]
    else:
        assert report["valid"], report["errors"]
        assert report["normalized_config"]["semantic_loss_type"] == loss_type


@pytest.mark.parametrize(
    "task", ["pep_function", "pep_protein_interaction", "pep_hlaii", "pep_hlai"]
)
@pytest.mark.parametrize(
    "counts,expected",
    [
        ({}, (1, 8)),
        ({"semantic_positive_partners": None, "semantic_zero_partners": None}, (1, 8)),
        ({"semantic_positive_partners": 4, "semantic_zero_partners": 16}, (4, 16)),
        ({"semantic_positive_partners": 4, "semantic_zero_partners": 0}, (4, 0)),
        ({"semantic_positive_partners": "4", "semantic_zero_partners": "0"}, (4, 0)),
        ({"semantic_positive_partners": 4.0, "semantic_zero_partners": 0.0}, (4, 0)),
    ],
)
def test_preflight_round_trips_semantic_partner_counts(task, counts, expected):
    report = preflight_experiment(
        _dataset(task, "sequence_dir" if task == "pep_function" else "pair_split_dir"),
        task,
        {
            "tokenizer": "kmer",
            "encoder": "onehot",
            "model": "cnn",
            "pair_strategy": "single_stream",
            "enable_semantic_similarity": True,
            **counts,
        },
    )
    assert report["valid"], report["errors"]
    config = report["normalized_config"]
    assert config["semantic_positive_partners"] == expected[0]
    assert config["semantic_zero_partners"] == expected[1]
    assert type(config["semantic_positive_partners"]) is int
    assert type(config["semantic_zero_partners"]) is int
    assert "semantic_loss_type" not in config


@pytest.mark.parametrize("key", ["semantic_positive_partners", "semantic_zero_partners"])
@pytest.mark.parametrize(
    "value",
    [
        1.5,
        -1,
        True,
        False,
        "1.5",
        "bad",
        "",
        "nan",
        "inf",
        float("nan"),
        float("inf"),
        -float("inf"),
        [],
        {},
    ],
)
def test_preflight_rejects_noninteger_semantic_partner_counts(key, value):
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir"),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "onehot",
            "model": "cnn",
            "enable_semantic_similarity": True,
            key: value,
        },
    )
    assert not report["valid"]
    assert f"{key}_invalid" in report["error_codes"]


def test_preflight_rejects_invalid_semantic_contrastive_parameters():
    report = preflight_experiment(
        _dataset("pep_hlai", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_hlai",
        {
            "tokenizer": "esm2",
            "encoder": "ppi_fusion",
            "model": "mlp",
            "enable_semantic_similarity": True,
            "semantic_pair_mode": "invalid",
            "semantic_projection_dim": 0,
            "semantic_contrastive_temperature": 0.0,
            "semantic_positive_partners": 0,
            "semantic_zero_partners": -1,
        },
    )

    assert report["valid"] is False
    assert "semantic_pair_mode_invalid" in report["error_codes"]
    assert "semantic_projection_dim_invalid" in report["error_codes"]
    assert "semantic_temperature_invalid" in report["error_codes"]
    assert "semantic_positive_partners_invalid" in report["error_codes"]
    assert "semantic_zero_partners_invalid" in report["error_codes"]


def test_preflight_rejects_removed_hlaii_context_semantic_mode():
    report = preflight_experiment(
        _dataset("pep_hlaii", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_hlaii",
        {
            "tokenizer": "esm2",
            "encoder": "ppi_fusion",
            "model": "mlp",
            "enable_interaction_semantic_similarity": True,
            "interaction_semantic_mode": "hla_context",
            "hla_context_semantic_margin": 0.2,
        },
    )

    assert report["valid"] is False
    assert "semantic_mode_removed" in report["error_codes"]
    assert "semantic_parameter_removed" in report["error_codes"]


def test_preflight_accepts_ppi_profile_jaccard_semantic_mode():
    report = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_protein_interaction",
        {
            "tokenizer": "esm2",
            "encoder": "ppi_fusion",
            "model": "mlp",
            "enable_interaction_semantic_similarity": True,
            "interaction_semantic_mode": "profile_jaccard",
        },
    )

    assert report["valid"] is True
    assert report["normalized_config"]["enable_semantic_similarity"] is True
    assert "interaction_semantic_mode" not in report["normalized_config"]


def test_preflight_rejects_invalid_interaction_semantic_mode():
    report = preflight_experiment(
        _dataset("pep_hlai", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_hlai",
        {
            "tokenizer": "esm2",
            "encoder": "ppi_fusion",
            "model": "mlp",
            "enable_interaction_semantic_similarity": True,
            "interaction_semantic_mode": "unknown_mode",
        },
    )

    assert report["valid"] is False
    assert "semantic_mode_invalid" in report["error_codes"]
    assert any(
        "only accepts the legacy value profile_jaccard" in message for message in report["errors"]
    )


def test_preflight_rejects_classic_classifier_with_unified_semantics():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "onehot",
            "model": "svm",
            "enable_semantic_similarity": True,
        },
    )

    assert report["valid"] is False
    assert "semantic_model_unsupported" in report["error_codes"]


def test_preflight_rejects_pair_sample_graph_with_unified_semantics():
    report = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_protein_interaction",
        {
            "tokenizer": "esm2",
            "encoder": "ppi_fusion",
            "model": "gcn",
            "predictor_type": "sample_graph",
            "enable_semantic_similarity": True,
        },
    )

    assert report["valid"] is False
    assert "semantic_model_unsupported" in report["error_codes"]


def test_preflight_rejects_export_then_classic_classifier_with_unified_semantics():
    report = preflight_experiment(
        _dataset("pep_protein_interaction", "pair_split_dir", {"train": 10, "val": 2, "test": 2}),
        "pep_protein_interaction",
        {
            "tokenizer": "esm2",
            "encoder": "ppi_fusion",
            "model": "cross_attn+mlp",
            "training_mode": "export_embeddings",
            "classifier": "rf",
            "enable_semantic_similarity": True,
        },
    )

    assert report["valid"] is False
    assert "semantic_model_unsupported" in report["error_codes"]


def test_preflight_accepts_canonical_calibrated_svm_config():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 12, "val": 4, "test": 4}),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "bow",
            "model": "svm",
            "classifier": "svm",
            "svm_calibration": "sigmoid",
            "svm_calibration_cv_folds": 3,
        },
    )

    assert report["valid"] is True
    assert report["normalized_config"]["model"] == "svm"
    assert report["normalized_config"]["classifier"] == "svm"
    assert report["normalized_config"]["svm_calibration"] == "sigmoid"
    assert report["normalized_config"]["svm_calibration_cv_folds"] == 3


def test_preflight_normalizes_legacy_calibrated_svm_alias():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 12, "val": 4, "test": 4}),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "bow",
            "model": "calibrated_svm",
            "classifier": "calibrated_svm",
        },
    )

    assert report["valid"] is True
    assert report["normalized_config"]["model"] == "svm"
    assert report["normalized_config"]["classifier"] == "svm"
    assert report["normalized_config"]["svm_calibration"] == "sigmoid"


def test_preflight_rejects_non_fixed_svm_calibration_folds():
    report = preflight_experiment(
        _dataset("pep_function", "sequence_dir", {"train": 12, "val": 4, "test": 4}),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "bow",
            "model": "svm",
            "classifier": "svm",
            "svm_calibration": "sigmoid",
            "svm_calibration_cv_folds": 5,
        },
    )

    assert report["valid"] is False
    assert "normalization_error" in report["error_codes"]
