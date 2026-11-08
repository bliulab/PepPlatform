import json
import shutil
from argparse import Namespace
from pathlib import Path

import joblib
import numpy as np
import pytest
import sklearn
import torch
from sklearn.preprocessing import StandardScaler

from pepplatform.models import PretrainedResiduePredictor, ResidueModelInput
from pepplatform.profile_generation import ProfileGenerationConfig
from pepplatform.sequence_profiles import (
    BLOSUM62_BACKGROUND_SHA256,
    BLOSUM62_MATRIX_SHA256,
    PROFILE_SCIENTIFIC_REFERENCES,
    PROFILE_HIT_SUMMARY_FIELDS,
    PROFILE_QUERY_COVERAGE_FORMULA,
    FEATURE_PSSM,
    FEATURE_TARGET_FREQUENCY,
    PROFILE_CONTRACT_VERSION,
    SOURCE_PSI_BLAST_PSSM,
    atomic_save_profile_matrix,
    atomic_write_json,
    new_profile_manifest,
    profile_audit_summary,
    profile_sample_metadata,
    sha256_file,
)
from pepplatform.top_ngram import (
    TOP_N_GRAM_FALLBACK_POLICY,
    TOP_N_GRAM_FALLBACK_POLICY_VERSION,
)
from pepplatform.web.inference import InferenceUnsupportedError, ModelService


class _DecisionEstimator:
    def decision_function(self, features):
        return features.sum(axis=1)


class _ProbabilityEstimator:
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        positive = np.clip(features.sum(axis=1), 0.0, 1.0)
        return np.stack([1.0 - positive, positive], axis=1)


class _EmbeddingEncoder(torch.nn.Module):
    def forward_with_embedding(self, inputs):
        embeddings = inputs.float()[:, :2]
        return embeddings.sum(dim=1, keepdim=True), embeddings


class _AlignedGatedPairModel(torch.nn.Module):
    def forward(self, peptide, protein):
        return (peptide.sum(dim=1, keepdim=True) - protein.sum(dim=1, keepdim=True)) * 0.0


class _ModelStore:
    def __init__(self, bundle_path):
        self.bundle_path = bundle_path

    def get_model(self, model_id):
        return {
            "id": model_id,
            "bundle_path": str(self.bundle_path),
            "supports_inference": True,
        }


def _rf_contract(version=None):
    return {
        "contract_version": 1,
        "algorithm": "random_forest",
        "implementation": "sklearn.ensemble.RandomForestClassifier",
        "implementation_version": version or sklearn.__version__,
        "status": "verified",
        "score_kind": "probability",
        "calibrated": False,
    }


def _svm_contract(version=None, implementation="sklearn.svm.LinearSVC", kernel="linear"):
    return {
        "contract_version": 1,
        "algorithm": "support_vector_machine",
        "implementation": implementation,
        "implementation_version": version or sklearn.__version__,
        "status": "verified_platform_variant",
        "input_contract": "fixed_width_global_vector",
        "kernel": kernel,
        "score_kind": "margin",
        "output": "decision_function_margin",
        "calibrated": False,
        "convergence": {"required": True, "status": "converged"},
    }


def test_legacy_rf_predictor_contract_warns_but_remains_loadable():
    metadata = {"args": {"model": "rf", "classifier": "rf"}}

    with pytest.warns(RuntimeWarning, match="legacy_only"):
        ModelService._validate_rf_predictor_contract(metadata)


def test_current_rf_predictor_contract_accepts_matching_sklearn_version():
    metadata = {
        "args": {"model": "rf", "classifier": "rf"},
        "predictor_contract": _rf_contract(),
    }

    ModelService._validate_rf_predictor_contract(metadata)


def test_current_rf_predictor_contract_rejects_sklearn_version_mismatch():
    metadata = {
        "args": {"model": "rf", "classifier": "rf"},
        "predictor_contract": _rf_contract("0.0.invalid"),
    }

    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._validate_rf_predictor_contract(metadata)

    assert exc_info.value.code == "rf_runtime_version_mismatch"
    assert exc_info.value.stage == "bundle"
    assert "0.0.invalid" in exc_info.value.technical_detail


def test_legacy_svm_predictor_contract_warns_but_remains_loadable():
    metadata = {"args": {"model": "svm", "classifier": "svm"}}

    with pytest.warns(RuntimeWarning, match="legacy_only"):
        ModelService._validate_svm_predictor_contract(metadata)


@pytest.mark.parametrize(
    ("implementation", "kernel"),
    [("sklearn.svm.LinearSVC", "linear"), ("sklearn.svm.SVC", "rbf")],
)
def test_current_svm_predictor_contract_accepts_supported_primitives(implementation, kernel):
    metadata = {
        "args": {"model": "svm", "classifier": "svm"},
        "predictor_contract": _svm_contract(implementation=implementation, kernel=kernel),
    }

    ModelService._validate_svm_predictor_contract(metadata)


def test_current_svm_predictor_contract_accepts_per_output_iteration_budgets():
    contract = _svm_contract()
    contract["num_outputs"] = 3
    contract["estimator_params"] = {
        "max_iter": 2000,
        "requested_max_iter": 2000,
        "effective_max_iter_by_output": [2000, 10_000, 2000],
    }
    contract["convergence"]["effective_max_iter_by_output"] = [2000, 10_000, 2000]

    ModelService._validate_svm_predictor_contract(
        {
            "args": {"model": "svm", "classifier": "svm"},
            "predictor_contract": contract,
        }
    )


def test_current_svm_predictor_contract_rejects_misaligned_iteration_budgets():
    contract = _svm_contract()
    contract["num_outputs"] = 2
    contract["estimator_params"] = {
        "effective_max_iter_by_output": [2000, 10_000],
    }
    contract["convergence"]["effective_max_iter_by_output"] = [2000]

    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._validate_svm_predictor_contract(
            {
                "args": {"model": "svm", "classifier": "svm"},
                "predictor_contract": contract,
            }
        )

    assert exc_info.value.code == "svm_predictor_contract_unsupported"


def test_current_svm_predictor_contract_rejects_incomplete_contract():
    contract = _svm_contract()
    contract["convergence"] = {"required": True, "status": "unknown"}
    metadata = {
        "args": {"model": "svm", "classifier": "svm"},
        "predictor_contract": contract,
    }

    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._validate_svm_predictor_contract(metadata)

    assert exc_info.value.code == "svm_predictor_contract_unsupported"
    assert exc_info.value.stage == "bundle"


def test_current_svm_predictor_contract_rejects_sklearn_version_mismatch():
    metadata = {
        "args": {"model": "svm", "classifier": "svm"},
        "predictor_contract": _svm_contract("0.0.invalid"),
    }

    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._validate_svm_predictor_contract(metadata)

    assert exc_info.value.code == "svm_runtime_version_mismatch"
    assert exc_info.value.stage == "bundle"
    assert "0.0.invalid" in exc_info.value.technical_detail


def test_legacy_calibrated_svm_claims_svm_and_warns_as_legacy_only():
    metadata = {"args": {"model": "calibrated_svm", "classifier": "calibrated_svm"}}

    assert ModelService._metadata_claims_svm(metadata) is True
    with pytest.warns(RuntimeWarning, match="legacy_only"):
        ModelService._validate_svm_predictor_contract(metadata)


def test_single_label_sequence_svm_uses_margin_threshold_and_reports_score_kind(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        ModelService,
        "_extract_pretrained_embeddings",
        staticmethod(
            lambda metadata, sequences, device: torch.tensor([[-1.0, 0.0]], dtype=torch.float)
        ),
    )
    joblib.dump(
        {
            "estimator_kind": "svm",
            "estimators": [_DecisionEstimator()],
            "task_reducer": None,
            "ml_scaler": None,
            "ml_reducer": None,
        },
        tmp_path / "model.joblib",
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "bundle_type": "pretrained_embedding_classical",
                "supports_inference": True,
                "label_names": ["active"],
                "threshold": -2.0,
                "operating_point": {
                    "threshold": -2.0,
                    "values": [-2.0],
                    "score_kind": "margin",
                },
                "predictor_contract": _svm_contract(),
                "args": {
                    "task_type": "pep_function",
                    "tokenizer": "esm2",
                    "encoder": "esm2",
                    "model": "svm",
                    "classifier": "svm",
                },
            }
        ),
        encoding="utf-8",
    )

    response = ModelService(_ModelStore(tmp_path)).predict("model", ["ACDE"])

    assert response["score_kind"] == "margin"
    assert response["thresholds"] == [-2.0]
    assert response["predictions"][0]["scores"] == [-1.0]
    assert response["predictions"][0]["predicted_labels"] == ["active"]


def _profile_provenance(
    config,
    feature_type=FEATURE_PSSM,
    subject_fasta_sha256="a" * 64,
    fallback_policy="error",
):
    return {
        "profile_contract_version": PROFILE_CONTRACT_VERSION,
        "algorithm": "NCBI PSI-BLAST",
        "algorithm_lineage": "Altschul_et_al_1997",
        "implementation": "NCBI BLAST+",
        "blast_version": "psiblast: test",
        "makeblastdb_version": "makeblastdb: test",
        "scientific_parameters": config.scientific_parameters(),
        "runtime_parameters": config.runtime_parameters(),
        "database_scope": "train_only",
        "subject_fasta_sha256": subject_fasta_sha256,
        "subject_fasta_artifact": "subjects/test.fasta",
        "database": {"train_fasta_sha256": "b" * 64},
        "self_exclusion": ("all subject IDs with an identical normalized query sequence"),
        "sequence_normalization": {
            "uppercase": True,
            "strip_whitespace": True,
            "J_U_O_to_X": True,
            "B_Z_X_preserved": True,
            "non_letters": "error",
        },
        "blosum62_matrix_sha256": BLOSUM62_MATRIX_SHA256,
        "blosum62_background_sha256": BLOSUM62_BACKGROUND_SHA256,
        "scientific_references": [dict(reference) for reference in PROFILE_SCIENTIFIC_REFERENCES],
        "quality_summary": {
            "outfmt_fields": list(PROFILE_HIT_SUMMARY_FIELDS),
            "query_coverage": PROFILE_QUERY_COVERAGE_FORMULA,
            "iteration_scope": "final_completed_iteration",
            "hit_count_unit": "unique_subject",
        },
        "profile_semantics": {
            "method_name": feature_type,
            "matrix_layout": "query_position_by_amino_acid",
        },
        "audit_artifact": "profile_audit.json",
        "audit_sha256": "e" * 64,
        "audit_summary": {
            "sample_count": 1,
            "samples_with_zero_observations": 0,
            "zero_observation_position_count": 0,
            "zero_observation_fraction": 0.0,
            "final_iteration_unique_subject_count": 1,
            "fallback_sample_count": 0,
            "fallback_fraction": 0.0,
            "fallback_reason_counts": {},
        },
        "generation_fingerprint": "c" * 64,
        "feature_type": feature_type,
        "fallback_policy": fallback_policy,
    }


def test_target_frequency_training_manifest_accepts_versioned_fallback_policy():
    config = ProfileGenerationConfig()
    provenance = _profile_provenance(
        config,
        feature_type=FEATURE_TARGET_FREQUENCY,
        fallback_policy=TOP_N_GRAM_FALLBACK_POLICY_VERSION,
    )
    profile = {
        "db_scope": "train_only",
        "feature": FEATURE_TARGET_FREQUENCY,
        "manifest_sha256": "d" * 64,
        "on_missing": TOP_N_GRAM_FALLBACK_POLICY,
        "training_profile_provenance": provenance,
    }
    metadata = {
        "classic": {"tokenizer": "top-n-gram", "encoder": "bow"},
    }

    contract = ModelService._matching_profile_training_contract(
        metadata,
        profile,
        config.scientific_parameters(),
    )

    assert contract == {
        "manifest_sha256": "d" * 64,
        "provenance": provenance,
    }


def _profile_generation_metadata(subject_path, *, iterations=3, evalue=0.001):
    config = ProfileGenerationConfig(
        iterations=iterations,
        evalue=evalue,
        inclusion_evalue=evalue,
        threads=1,
    )
    subject_sha256 = sha256_file(subject_path)
    provenance = _profile_provenance(
        config,
        subject_fasta_sha256=subject_sha256,
    )
    return {
        "profile_contract_version": PROFILE_CONTRACT_VERSION,
        "strategy": "psiblast_subject_fasta",
        "feature": FEATURE_PSSM,
        "subject_fasta": "profile_generation/subject.fasta",
        "db_prefix": "",
        "db_scope": "train_only",
        "iterations": config.iterations,
        "evalue": config.evalue,
        "inclusion_evalue": config.inclusion_evalue,
        "threads": config.threads,
        "on_missing": "error",
        "scientific_parameters": config.scientific_parameters(),
        "manifest_sha256": "d" * 64,
        "training_profile_provenance": provenance,
        "subject_fasta_sha256": subject_sha256,
    }


def _write_current_pssm_index(root, sequence_by_key):
    profile_root = root / "profiles"
    subject = profile_root / "subjects" / "test.fasta"
    subject.parent.mkdir(parents=True, exist_ok=True)
    subject.write_text(">TRAIN_000000\nACDEFGHIKLMNPQRSTVWY\n", encoding="ascii")
    samples = {}
    for key, sequence in sequence_by_key.items():
        matrix = np.ones((len(sequence), 20), dtype=np.int32)
        relative_path = Path("matrices") / f"{key.replace(':', '_')}.npz"
        digest = atomic_save_profile_matrix(
            profile_root / relative_path,
            matrix,
        )
        samples[key] = profile_sample_metadata(
            path=relative_path,
            matrix=matrix,
            sequence=sequence,
            matrix_sha256=digest,
            profile_source=SOURCE_PSI_BLAST_PSSM,
            fallback=False,
            quality={
                "nonself_hit_count": 1,
                "max_query_coverage": 1.0,
                "max_identity": 100.0,
                "min_evalue": 0.0,
                "zero_nonself_hits": False,
                "final_iteration": 1,
                "final_iteration_hit_rows": 1,
                "blast_warnings": [],
            },
        )
    provenance = _profile_provenance(
        ProfileGenerationConfig(),
        subject_fasta_sha256=sha256_file(subject),
    )
    audit_summary = profile_audit_summary(samples)
    audit = {
        "feature_type": FEATURE_PSSM,
        "generation_fingerprint": provenance["generation_fingerprint"],
        "summary": audit_summary,
        "samples": [
            {
                "sample_id": key,
                "sequence_length": len(sequence_by_key[key]),
                "zero_observation_positions": [],
                "profile_source": SOURCE_PSI_BLAST_PSSM,
                "fallback": False,
                "fallback_reason": "",
                "cache_reused": False,
                **samples[key]["quality"],
            }
            for key in sequence_by_key
        ],
    }
    audit_path = profile_root / "profile_audit.json"
    atomic_write_json(audit_path, audit)
    provenance["audit_sha256"] = sha256_file(audit_path)
    provenance["audit_summary"] = audit["summary"]
    manifest = new_profile_manifest(
        feature_type=FEATURE_PSSM,
        samples=samples,
        provenance=provenance,
    )
    index_path = profile_root / "index.json"
    atomic_write_json(index_path, manifest)
    return index_path


@pytest.mark.parametrize(
    ("bundle_type", "metadata_extra"),
    [
        ("sequence_classical", {}),
        (
            "pair_vector_classical",
            {
                "streams": {
                    "peptide": {"tokenizer": "kmer", "encoder": "onehot"},
                    "partner": {"tokenizer": "wordpiece", "encoder": "index"},
                }
            },
        ),
        ("pair_graph_nn", {"classic": {"tokenizer": "kmer", "encoder": "index"}}),
    ],
)
def test_raw_index_vector_artifacts_are_blocked_before_model_loading(
    tmp_path,
    bundle_type,
    metadata_extra,
):
    metadata = {
        "bundle_type": bundle_type,
        "supports_inference": True,
        "args": {
            "task_type": "pep_function",
            "tokenizer": "kmer",
            "encoder": "index",
            "model": "svm",
        },
        **metadata_extra,
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    service = ModelService(_ModelStore(tmp_path))

    with pytest.raises(InferenceUnsupportedError) as exc_info:
        service.predict("model", sequences=["AC"], pairs=[{"peptide": "A", "protein": "C"}])

    assert exc_info.value.code == "classical_index_unsupported"
    assert exc_info.value.stage == "bundle"
    readiness = service.readiness("model")
    assert readiness["ready"] is False
    assert readiness["errors"][0]["code"] == "classical_index_unsupported"


def test_pretrained_residue_nn_bundle_predicts_with_masked_scaling(tmp_path, monkeypatch):
    model = PretrainedResiduePredictor(
        "lstm",
        input_dim=3,
        adapter_dim=16,
        max_length=4,
        num_classes=1,
        dropout=0.0,
    )
    torch.save(model, tmp_path / "model.pt")
    scaler = StandardScaler().fit(np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32))
    joblib.dump({"dl_scaler": scaler}, tmp_path / "preprocess.joblib")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "bundle_type": "pretrained_residue_nn",
                "supports_inference": True,
                "label_names": ["active"],
                "thresholds": [0.5],
                "preprocess_path": "preprocess.joblib",
                "pretrained": {
                    "tokenizer": "esm2",
                    "model_name": "mock-esm2",
                    "max_length": 4,
                },
                "args": {"batch_size": 1},
            }
        ),
        encoding="utf-8",
    )
    raw = ResidueModelInput(
        torch.tensor(
            [
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [9.0, 9.0, 9.0], [9.0, 9.0, 9.0]],
                [[3.0, 4.0, 5.0], [9.0, 9.0, 9.0], [9.0, 9.0, 9.0], [9.0, 9.0, 9.0]],
            ]
        ),
        torch.tensor([[True, True, False, False], [True, False, False, False]]),
    )
    monkeypatch.setattr(
        ModelService,
        "_extract_pretrained_residue_features",
        staticmethod(lambda *_args, **_kwargs: raw),
    )

    response = ModelService(_ModelStore(tmp_path)).predict("model", ["AC", "A"])

    assert len(response["predictions"]) == 2
    assert all(np.isfinite(row["scores"][0]) for row in response["predictions"])
    transformed = ModelService._transform_pretrained_residue_features(
        tmp_path,
        json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8")),
        raw,
    )
    assert torch.count_nonzero(transformed.features[~transformed.mask]) == 0


def test_pretrained_residue_embedding_classical_bundle_predicts(tmp_path, monkeypatch):
    joblib.dump(
        {
            "estimator_kind": "probability_ensemble",
            "estimators": [{"kind": "sklearn", "value": _ProbabilityEstimator()}],
            "task_reducer": None,
            "ml_scaler": None,
            "ml_reducer": None,
        },
        tmp_path / "model.joblib",
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "bundle_type": "pretrained_residue_embedding_classical",
                "supports_inference": True,
                "label_names": ["active"],
                "pretrained": {
                    "tokenizer": "prott5",
                    "model_name": "mock-prott5",
                    "max_length": 4,
                },
                "embedding_export": {"encoder_path": "embedding_encoder.pt"},
                "args": {"batch_size": 2},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        ModelService,
        "_extract_pretrained_residue_export_features",
        staticmethod(
            lambda *_args, **_kwargs: torch.tensor([[0.2, 0.3], [0.1, 0.4]], dtype=torch.float)
        ),
    )

    response = ModelService(_ModelStore(tmp_path)).predict("model", ["AC", "DE"])

    assert [row["scores"] for row in response["predictions"]] == [
        [pytest.approx(0.5)],
        [pytest.approx(0.5)],
    ]


def test_profile_sequence_classical_predict_returns_unsupported_before_encoding(tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "bundle_type": "sequence_classical",
                "supports_inference": True,
                "args": {"tokenizer": "kmer", "encoder": "pssm", "model": "rf"},
            }
        ),
        encoding="utf-8",
    )

    class _Store:
        def get_model(self, model_id):
            return {
                "id": model_id,
                "bundle_path": str(tmp_path),
                "supports_inference": True,
            }

    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService(_Store()).predict("model", ["FLPILGNLLSGL"])

    assert "PSSM/PSFM" in str(exc_info.value)


def test_profile_sequence_nn_rejects_legacy_implicit_fallback_bundle(
    tmp_path,
):
    profile_dir = tmp_path / "profile_generation"
    profile_dir.mkdir()
    (profile_dir / "subject.fasta").write_text(">SUBJ_0\nAC\n", encoding="utf-8")
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(40, 1))
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.zero_()
    torch.save(model, tmp_path / "model.pt")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "bundle_type": "sequence_nn",
                "supports_inference": True,
                "label_names": ["antimicrobial"],
                "evaluation_protocol_version": 2,
                "thresholds": [0.6],
                "profile_generation": {
                    "strategy": "psiblast_subject_fasta",
                    "feature": "pssm",
                    "subject_fasta": "profile_generation/subject.fasta",
                    "db_prefix": "",
                    "db_scope": "train_val",
                    "iterations": 2,
                    "evalue": 0.01,
                    "threads": 1,
                    "on_missing": "blosum62",
                },
                "args": {
                    "task_type": "pep_function",
                    "tokenizer": "kmer",
                    "encoder": "pssm",
                    "model": "cnn",
                    "sequence_model": "cnn",
                    "training_mode": "end_to_end",
                    "classifier": "mlp",
                    "max_length": 2,
                    "k": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    class _Store:
        def get_model(self, model_id):
            return {"id": model_id, "bundle_path": str(tmp_path), "supports_inference": True}

    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService(_Store()).predict("model", ["AC"])
    assert exc_info.value.code == "profile_contract_legacy"


@pytest.mark.skipif(
    shutil.which("makeblastdb") is None or shutil.which("psiblast") is None,
    reason="ncbi-blast+ is not installed",
)
def test_profile_sequence_predict_with_real_blast(tmp_path):
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    profile_dir = tmp_path / "profile_generation"
    profile_dir.mkdir()
    (profile_dir / "subject.fasta").write_text(
        f">SUBJ_0\n{sequence}{sequence}\n",
        encoding="utf-8",
    )
    subject_path = profile_dir / "subject.fasta"
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(400, 1))
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.zero_()
    torch.save(model, tmp_path / "model.pt")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "bundle_type": "sequence_nn",
                "supports_inference": True,
                "label_names": ["active"],
                "profile_generation": _profile_generation_metadata(
                    subject_path,
                    iterations=2,
                    evalue=0.001,
                ),
                "args": {
                    "task_type": "pep_function",
                    "tokenizer": "kmer",
                    "encoder": "pssm",
                    "model": "cnn",
                    "sequence_model": "cnn",
                    "training_mode": "end_to_end",
                    "classifier": "mlp",
                    "max_length": 20,
                    "k": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    class _Store:
        def get_model(self, model_id):
            return {"id": model_id, "bundle_path": str(tmp_path), "supports_inference": True}

    response = ModelService(_Store()).predict("model", [sequence])

    assert response["predictions"][0]["scores"] == [0.5]
    assert response["diagnostics"]["profile"]["status"] == "ready"
    assert response["diagnostics"]["profile"]["generated"] == 1
    assert response["diagnostics"]["profile"]["fallbacks"] == 0
    assert list((profile_dir / "inference_cache").glob("*/pssm/index.json"))


def test_profile_inference_rejects_tampered_subject_fasta(tmp_path):
    profile_dir = tmp_path / "profile_generation"
    profile_dir.mkdir()
    subject = profile_dir / "subject.fasta"
    subject.write_text(">TRAIN_000000\nACDE\n", encoding="utf-8")
    metadata = {
        "profile_generation": _profile_generation_metadata(subject),
        "args": {
            "tokenizer": "kmer",
            "encoder": "pssm",
            "max_length": 4,
            "k": 1,
        },
    }
    subject.write_text(">TRAIN_000000\nAAAA\n", encoding="utf-8")

    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._build_inference_profile_cache(
            tmp_path,
            metadata,
            {"sample": "ACDE"},
        )
    assert exc_info.value.code == "profile_subject_digest_mismatch"


def test_legacy_labelled_lda_feature_transform_and_prediction_helpers():
    payload = {
        "task_reducer": {"kind": "labelled_lda", "value": np.array([[1.0, -1.0]])},
        "ml_scaler": None,
        "ml_reducer": None,
        "estimator_kind": "probability_ensemble",
        "estimators": [
            {"kind": "constant", "value": 0.25},
            {"kind": "sklearn", "value": _ProbabilityEstimator()},
        ],
    }
    with pytest.warns(RuntimeWarning, match="legacy_only"):
        transformed = ModelService._transform_classical_features(
            payload,
            np.array([[2.0, 1.0], [0.5, 0.25]]),
        )
    assert transformed.tolist() == [[1.0], [0.25]]

    scores = ModelService._predict_classical_scores(payload, transformed)
    assert scores.tolist() == [[0.25, 1.0], [0.25, 0.25]]


def test_svm_prediction_helper():
    payload = {
        "estimator_kind": "svm",
        "estimators": [_DecisionEstimator()],
    }
    scores = ModelService._predict_classical_scores(payload, np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert scores.tolist() == [[3.0], [7.0]]


def test_pretrained_tensor_preprocess_helper(tmp_path):
    scaler = StandardScaler().fit(np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32))
    import joblib

    joblib.dump({"dl_scaler": scaler}, tmp_path / "preprocess.joblib")

    transformed = ModelService._transform_pretrained_tensor_features(
        tmp_path,
        {"preprocess_path": "preprocess.joblib"},
        torch.tensor([[1.0, 1.0]], dtype=torch.float),
    )

    assert transformed.tolist() == [[0.0, 0.0]]


def test_sequence_nn_predict_applies_saved_preprocess(tmp_path, monkeypatch):
    import scripts.train_sequence_pipeline as train_sequence_pipeline

    class _Featurizer:
        def encode(self, sequence):
            return torch.tensor([1.0, 1.0], dtype=torch.float)

    class _Pipeline:
        def __init__(self, args):
            self.device = torch.device("cpu")

        def build_featurizer(self):
            return _Featurizer()

    monkeypatch.setattr(train_sequence_pipeline, "Pipeline", _Pipeline)
    monkeypatch.delenv("PEPPLATFORM_INFERENCE_BATCH_SIZE", raising=False)
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.zero_()
    torch.save(model, tmp_path / "model.pt")
    scaler = StandardScaler().fit(np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32))
    joblib.dump({"dl_scaler": scaler}, tmp_path / "preprocess.joblib")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "bundle_type": "sequence_nn",
                "supports_inference": True,
                "label_names": ["antimicrobial"],
                "preprocess_path": "preprocess.joblib",
                "args": {
                    "task_type": "pep_function",
                    "tokenizer": "kmer",
                    "encoder": "onehot",
                    "model": "cnn",
                    "dl_scaler": "standard",
                    "batch_size": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    class _Store:
        def get_model(self, model_id):
            return {"id": model_id, "bundle_path": str(tmp_path), "supports_inference": True}

    batches = []
    forward = torch.nn.Linear.forward

    def recorded_forward(self, inputs):
        batches.append(int(inputs.shape[0]))
        return forward(self, inputs)

    monkeypatch.setattr(torch.nn.Linear, "forward", recorded_forward)
    response = ModelService(_Store()).predict("model", ["ACDE", "AAAA", "CCCC"])

    assert response["labels"] == ["antimicrobial"]
    assert response["predictions"][0]["scores"] == [0.5]
    assert batches == [1, 1, 1]
    assert all(row["scores"] == [0.5] for row in response["predictions"])


def test_sequence_embedding_classical_predict_uses_stage_one_encoder(tmp_path, monkeypatch):
    import scripts.train_sequence_pipeline as train_sequence_pipeline

    class _Featurizer:
        def encode(self, sequence):
            return torch.tensor([0.2, 0.3], dtype=torch.float)

    class _Pipeline:
        def __init__(self, args):
            self.device = torch.device("cpu")

        def build_featurizer(self):
            return _Featurizer()

    monkeypatch.setattr(train_sequence_pipeline, "Pipeline", _Pipeline)
    torch.save(_EmbeddingEncoder(), tmp_path / "embedding_encoder.pt")
    joblib.dump(
        {
            "estimator_kind": "probability_ensemble",
            "estimators": [{"kind": "sklearn", "value": _ProbabilityEstimator()}],
            "task_reducer": None,
            "ml_scaler": None,
            "ml_reducer": None,
        },
        tmp_path / "model.joblib",
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "bundle_type": "sequence_embedding_classical",
                "supports_inference": True,
                "label_names": ["antimicrobial"],
                "embedding_export": {
                    "encoder_path": "embedding_encoder.pt",
                    "embedding_dim": 2,
                    "downstream": "logreg",
                },
                "args": {
                    "task_type": "pep_function",
                    "tokenizer": "kmer",
                    "encoder": "onehot",
                    "model": "cnn",
                    "sequence_model": "cnn",
                    "training_mode": "export_embeddings",
                    "classifier": "logreg",
                },
            }
        ),
        encoding="utf-8",
    )

    class _Store:
        def get_model(self, model_id):
            return {"id": model_id, "bundle_path": str(tmp_path), "supports_inference": True}

    response = ModelService(_Store()).predict("model", ["ACDE"])

    assert response["labels"] == ["antimicrobial"]
    assert response["predictions"][0]["scores"] == [0.5]


def test_extract_pretrained_embeddings_uses_recorded_esm2_source(monkeypatch):
    class _FakeTokenizer:
        def __init__(self, model_name, max_length):
            self.model_name = model_name
            self.max_length = max_length

        def batch_tokenize(self, sequences, device=None):
            return {"sequences": sequences}

    class _FakeEncoder:
        def __init__(self, model_name, device=None):
            self.model_name = model_name
            self.device = device

        def encode(self, inputs):
            return torch.tensor([[float(len(sequence)), 1.0] for sequence in inputs["sequences"]])

    import pepplatform.encoders as encoders
    import pepplatform.tokenizers as tokenizers

    monkeypatch.setattr(tokenizers, "EsmTokenizer", _FakeTokenizer)
    monkeypatch.setattr(encoders, "EsmEncoder", _FakeEncoder)

    embeddings = ModelService._extract_pretrained_embeddings(
        {
            "pretrained": {
                "tokenizer": "esm2",
                "model_name": "models/esm2",
                "max_length": 60,
            },
            "args": {"batch_size": 1},
        },
        ["ACD", "GG"],
        torch.device("cpu"),
    )

    assert embeddings.tolist() == [[3.0, 1.0], [2.0, 1.0]]


def test_esmc_inference_falls_back_from_stale_path_to_recorded_model_id(monkeypatch):
    import pepplatform.encoders as encoders
    import pepplatform.tokenizers as tokenizers
    import pepplatform.web.inference as inference_module

    expected_identity = {
        "provider": "EvolutionaryScale",
        "model_id": "EvolutionaryScale/esmc-600m-2024-12",
        "weight_sha256": "a" * 64,
        "runtime_version": "3.2.0",
        "hidden_size": 1152,
        "pooling": "residue_mean",
    }
    observed = {}

    class _FakeTokenizer:
        def __init__(self, model_name, max_length):
            observed["tokenizer_source"] = model_name
            observed["max_length"] = max_length

    class _FakeEncoder:
        model_identity = expected_identity

        def __init__(self, model_name, device):
            observed["encoder_source"] = model_name
            observed["device"] = device

    def resolve_source(_root, raw_model, _candidates, _label, _cli_arg, tokenizer):
        observed["raw_model"] = raw_model
        observed["tokenizer"] = tokenizer
        return raw_model

    monkeypatch.setattr(tokenizers, "EsmcTokenizer", _FakeTokenizer)
    monkeypatch.setattr(encoders, "EsmcEncoder", _FakeEncoder)
    monkeypatch.setattr(inference_module, "resolve_pretrained_model_source", resolve_source)

    tokenizer, encoder = ModelService._load_esmc_for_inference(
        {
            "model_name": "/missing/old-machine/esmc",
            "model_identity": expected_identity,
        },
        max_length=60,
        device="cpu",
    )

    assert isinstance(tokenizer, _FakeTokenizer)
    assert isinstance(encoder, _FakeEncoder)
    assert observed["raw_model"] == expected_identity["model_id"]
    assert observed["tokenizer_source"] == expected_identity["model_id"]
    assert observed["encoder_source"] == expected_identity["model_id"]


def test_pair_vector_features_fuse_recorded_pretrained_embeddings(monkeypatch):
    class _FakeTokenizer:
        def __init__(self, model_name, max_length):
            self.model_name = model_name
            self.max_length = max_length

        def batch_tokenize(self, sequences, device=None):
            return {"sequences": sequences}

    class _FakeEncoder:
        def __init__(self, model_name, device=None):
            self.model_name = model_name
            self.device = device

        def encode(self, inputs):
            return torch.tensor([[float(len(sequence)), 1.0] for sequence in inputs["sequences"]])

    import pepplatform.encoders as encoders
    import pepplatform.tokenizers as tokenizers

    monkeypatch.setattr(tokenizers, "EsmTokenizer", _FakeTokenizer)
    monkeypatch.setattr(encoders, "EsmEncoder", _FakeEncoder)

    features = ModelService._extract_pair_vector_features(
        {
            "pretrained": {
                "tokenizer": "esm2",
                "model_name": "models/esm2",
                "max_length": 60,
            },
            "pair": {"fusion_mode": "concat_all"},
            "args": {"batch_size": 1},
        },
        [{"peptide": "AC", "protein": "GGG"}],
        torch.device("cpu"),
    )

    assert features.tolist() == [[2.0, 1.0, 3.0, 1.0, 1.0, 0.0, 6.0, 1.0]]


def test_pair_vector_features_rebuild_classic_glove_embeddings(tmp_path):
    glove_path = tmp_path / "glove.vec"
    glove_path.write_text("A 1 0\nC 2 0\nG 0 1\n<UNK> 0 0\n", encoding="utf-8")

    features = ModelService._extract_pair_vector_features(
        {
            "classic": {
                "tokenizer": "kmer",
                "encoder": "glove",
                "max_length": 2,
                "k": 1,
                "glove_path": str(glove_path),
            },
            "pair": {"fusion_mode": "concat_all"},
            "args": {},
        },
        [{"peptide": "AC", "protein": "GG"}],
        torch.device("cpu"),
    )

    assert features.shape == (1, 16)
    assert features.tolist() == [
        [
            1.0,
            0.0,
            2.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            2.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    ]


def test_classic_inference_accepts_legacy_kmer_unit_and_rejects_unsupported_k():
    legacy, encoder = ModelService._build_classic_featurizer(
        {
            "classic": {
                "tokenizer": "kmer",
                "encoder": "index",
                "max_length": 2,
                "k": 2,
            }
        }
    )

    assert encoder == "index"
    assert legacy.tokenizer.tokenize("ACD") == ["AC", "CD"]

    with pytest.raises(InferenceUnsupportedError, match="unsupported K-mer artifact") as exc:
        ModelService._build_classic_featurizer(
            {
                "classic": {
                    "tokenizer": "kmer",
                    "encoder": "index",
                    "max_length": 2,
                    "k": 4,
                }
            }
        )
    assert exc.value.code == "kmer_k_unsupported"


@pytest.mark.parametrize(
    "implementation",
    [
        "pepplatform_bidirectional_elman_rnn",
        "pepplatform_bidirectional_pytorch_gru",
        "pepplatform_bidirectional_pytorch_lstm",
        "pepplatform_pytorch_transformer_encoder",
        "pepplatform_masked_chain_gcn",
    ],
)
def test_mask_aware_v2_web_reconstruction_preserves_explicit_valid_zero_mask(
    tmp_path,
    implementation,
):
    glove_path = tmp_path / "glove.vec"
    glove_path.write_text("A 1 2 3\n", encoding="utf-8")
    metadata = {
        "classic": {
            "tokenizer": "kmer",
            "encoder": "glove",
            "max_length": 4,
            "k": 1,
            "glove_path": str(glove_path),
        },
        "predictor_contract": {
            "implementation": implementation,
            "contract_version": 2,
            "status": "verified_platform_variant",
        },
    }

    features = ModelService._extract_sequence_features(
        metadata,
        ["AX"],
        pipeline=None,
    )

    assert isinstance(features, ResidueModelInput)
    assert features.mask.tolist() == [[True, True, False, False]]
    assert torch.count_nonzero(features.features[0, 1]) == 0

    legacy, _ = ModelService._build_classic_featurizer({"classic": metadata["classic"]})
    assert isinstance(legacy.encode("AX"), torch.Tensor)


def test_pair_vector_features_restore_tfidf_idf_from_bundle(tmp_path):
    idf_path = tmp_path / "classic_tfidf_idf.pt"
    from pepplatform.tokenizers import KmerTokenizer

    idf = torch.ones(KmerTokenizer(max_length=2, k=1).vocab_size, dtype=torch.float)
    idf[2] = 2.0
    idf[3] = 3.0
    torch.save(idf, idf_path)

    features = ModelService._extract_pair_vector_features(
        {
            "classic": {
                "tokenizer": "kmer",
                "encoder": "tfidf",
                "max_length": 2,
                "k": 1,
                "tfidf_idf_path": "classic_tfidf_idf.pt",
            },
            "pair": {"fusion_mode": "concat"},
            "args": {},
        },
        [{"peptide": "AA", "protein": "CC"}],
        torch.device("cpu"),
        bundle=tmp_path,
    )

    assert features.shape == (1, idf.numel() * 2)
    assert features[0, 2].item() == 2.0
    assert features[0, idf.numel() + 3].item() == 3.0
    assert torch.count_nonzero(features).item() == 2


def test_legacy_homogeneous_tfidf_rejects_ambiguous_role_state(tmp_path):
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "sample_id,peptide,pair_pseudo,label\na,AA,CC,1\nb,AA,DD,0\nc,GG,DD,1\n",
        encoding="utf-8",
    )
    top_level_idf = torch.ones(28)
    torch.save(top_level_idf, tmp_path / "classic_tfidf_idf.pt")
    stream_spec = {
        "feature_route": "classic",
        "tokenizer": "kmer",
        "encoder": "tfidf",
        "max_length": 2,
        "params": {"k": 1},
    }
    metadata = {
        "schema_version": 2,
        "args": {"train_csv": str(train_csv), "task_type": "pep_hlaii"},
        "classic": {
            "tokenizer": "kmer",
            "encoder": "tfidf",
            "max_length": 2,
            "k": 1,
            "tfidf_idf_path": "classic_tfidf_idf.pt",
        },
        "pair": {"fusion_mode": "concat", "stream_mode": "homogeneous"},
        "streams": {
            "peptide": {"spec": stream_spec},
            "partner": {"spec": stream_spec},
        },
    }

    with pytest.raises(
        InferenceUnsupportedError,
        match="missing its train-only IDF",
    ):
        ModelService._extract_pair_vector_features(
            metadata,
            [{"peptide": "AA", "protein": "DD"}],
            torch.device("cpu"),
            bundle=tmp_path,
        )


def test_removed_mismatch_bundle_uses_unsupported_tokenizer_failure_path():
    metadata = {
        "classic": {
            "tokenizer": "mismatch",
            "encoder": "tfidf",
            "tfidf_fit_scope": "combined_pair_roles",
        },
        "pair": {"stream_mode": "homogeneous"},
        "streams": {
            "peptide": {"spec": {"tokenizer": "mismatch", "encoder": "tfidf"}},
            "partner": {"spec": {"tokenizer": "mismatch", "encoder": "tfidf"}},
        },
    }

    with pytest.raises(InferenceUnsupportedError, match="unsupported tokenizer"):
        ModelService._extract_pair_vector_features(
            metadata,
            [{"peptide": "AA", "protein": "DD"}],
            torch.device("cpu"),
        )


def test_legacy_homogeneous_tfidf_does_not_guess_idf_from_feature_cache(
    tmp_path,
):
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    content = "sample_id,peptide,pair_pseudo,label\na,AA,CC,1\nb,AG,CD,0\nc,GG,DD,1\n"
    train_csv.write_text(content, encoding="utf-8")
    val_csv.write_text(content, encoding="utf-8")
    feature_dir = tmp_path / "features"
    stream_spec = {
        "feature_route": "classic",
        "tokenizer": "kmer",
        "encoder": "tfidf",
        "max_length": 2,
        "params": {"k": 1},
    }
    metadata = {
        "schema_version": 2,
        "args": {
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "feature_dir": str(feature_dir),
            "task_type": "pep_hlaii",
        },
        "classic": {"tokenizer": "kmer", "encoder": "tfidf", "max_length": 2, "k": 1},
        "pair": {"fusion_mode": "concat", "stream_mode": "homogeneous"},
        "streams": {
            "peptide": {"spec": stream_spec},
            "partner": {"spec": stream_spec},
        },
    }

    with pytest.raises(
        InferenceUnsupportedError,
        match="missing its train-only IDF",
    ):
        ModelService._extract_pair_vector_features(
            metadata,
            [{"peptide": "AA", "protein": "DD"}],
            torch.device("cpu"),
            bundle=tmp_path,
        )


def test_pair_vector_features_reject_removed_cs_artifact(tmp_path):
    metadata = {
        "classic": {
            "tokenizer": "kmer",
            "encoder": "cs",
            "max_length": 1,
            "k": 1,
        },
        "pair": {"fusion_mode": "concat"},
        "args": {},
    }

    with pytest.raises(InferenceUnsupportedError, match="CS encoder has been removed"):
        ModelService._extract_pair_vector_features(
            metadata,
            [{"peptide": "A", "protein": "A"}],
            torch.device("cpu"),
            bundle=tmp_path,
        )


def test_pair_vector_nn_predict_returns_pair_rows(tmp_path, monkeypatch):
    class _FakeTokenizer:
        def __init__(self, model_name, max_length):
            self.model_name = model_name
            self.max_length = max_length

        def batch_tokenize(self, sequences, device=None):
            return {"sequences": sequences}

    class _FakeEncoder:
        def __init__(self, model_name, device=None):
            self.model_name = model_name
            self.device = device

        def encode(self, inputs):
            return torch.tensor([[float(len(sequence)), 1.0] for sequence in inputs["sequences"]])

    import pepplatform.encoders as encoders
    import pepplatform.tokenizers as tokenizers

    monkeypatch.setattr(tokenizers, "EsmTokenizer", _FakeTokenizer)
    monkeypatch.setattr(encoders, "EsmEncoder", _FakeEncoder)

    model = torch.nn.Linear(8, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    torch.save(model, tmp_path / "model.pt")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "bundle_type": "pair_vector_nn",
                "supports_inference": True,
                "input_schema": "peptide_protein_pairs",
                "label_names": ["interaction"],
                "threshold": 0.6,
                "pretrained": {
                    "tokenizer": "esm2",
                    "model_name": "models/esm2",
                    "max_length": 60,
                },
                "pair": {"fusion_mode": "concat_all"},
                "args": {
                    "task_type": "pep_protein_interaction",
                    "tokenizer": "esm2",
                    "encoder": "ppi_fusion",
                    "model": "mlp",
                    "classifier": "mlp",
                    "pair_strategy": "two_stream",
                    "interaction_level": "vector",
                    "predictor_type": "classifier",
                    "batch_size": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    class _Store:
        def get_model(self, model_id):
            return {
                "id": model_id,
                "bundle_path": str(tmp_path),
                "supports_inference": True,
            }

    response = ModelService(_Store()).predict(
        "model",
        pairs=[{"peptide": "AC", "protein": "GGG"}],
    )

    assert response["labels"] == ["interaction"]
    assert response["predictions"] == [
        {
            "peptide": "AC",
            "protein": "GGG",
            "score": 0.5,
            "score_kind": "probability",
            "score_range": [0.0, 1.0],
            "threshold": 0.6,
            "is_interaction": False,
            "prediction_label": "No interaction",
            "scores": [0.5],
        }
    ]


def test_hlaii_pair_vector_nn_predict_returns_pair_rows(tmp_path, monkeypatch):
    def fake_pair_embeddings(metadata, pairs, device, bundle=None):
        peptide = torch.tensor([[1.0, 0.0] for _ in pairs], dtype=torch.float)
        hla = torch.tensor([[0.0, 1.0] for _ in pairs], dtype=torch.float)
        return peptide, hla

    monkeypatch.setattr(
        ModelService,
        "_extract_pair_embeddings",
        staticmethod(fake_pair_embeddings),
    )

    model = torch.nn.Linear(8, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    torch.save(model, tmp_path / "model.pt")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "bundle_type": "pair_vector_nn",
                "supports_inference": True,
                "input_schema": "peptide_hla_pairs",
                "label_names": ["interaction"],
                "threshold": 0.4,
                "pretrained": {"tokenizer": "esm2", "model_name": "models/esm2", "max_length": 2},
                "pair": {"fusion_mode": "concat_all"},
                "args": {"task_type": "pep_hlaii"},
            }
        ),
        encoding="utf-8",
    )

    class _Store:
        def get_model(self, model_id):
            return {"id": model_id, "bundle_path": str(tmp_path), "supports_inference": True}

    response = ModelService(_Store()).predict("model", pairs=[{"peptide": "AC", "protein": "HLA"}])

    assert response["labels"] == ["interaction"]
    prediction = response["predictions"][0]
    assert prediction["peptide"] == "AC"
    assert prediction["protein"] == "HLA"
    assert prediction["score"] == 0.5
    assert prediction["score_kind"] == "probability"
    assert prediction["is_interaction"] is True


def test_pair_vector_aligned_gated_nn_predict_uses_pair_embeddings(tmp_path, monkeypatch):
    def fake_pair_embeddings(metadata, pairs, device, bundle=None):
        peptide = torch.tensor([[1.0, 0.0] for _ in pairs], dtype=torch.float)
        protein = torch.tensor([[0.0, 1.0] for _ in pairs], dtype=torch.float)
        return peptide, protein

    monkeypatch.setattr(
        ModelService,
        "_extract_pair_embeddings",
        staticmethod(fake_pair_embeddings),
    )
    torch.save(_AlignedGatedPairModel(), tmp_path / "model.pt")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "bundle_type": "pair_vector_nn",
                "supports_inference": True,
                "input_schema": "peptide_protein_pairs",
                "label_names": ["interaction"],
                "threshold": 0.4,
                "pretrained": {"tokenizer": "esm2", "model_name": "models/esm2", "max_length": 2},
                "pair": {
                    "fusion_mode": "aligned_gated",
                    "peptide_dim": 2,
                    "protein_dim": 2,
                    "fused_dim": 6,
                },
                "args": {"task_type": "pep_protein_interaction"},
            }
        ),
        encoding="utf-8",
    )

    class _Store:
        def get_model(self, model_id):
            return {"id": model_id, "bundle_path": str(tmp_path), "supports_inference": True}

    response = ModelService(_Store()).predict("model", pairs=[{"peptide": "AC", "protein": "GG"}])

    prediction = response["predictions"][0]
    assert prediction["score"] == 0.5
    assert prediction["score_kind"] == "probability"
    assert prediction["is_interaction"] is True


def test_pair_vector_rf_predict_uses_classic_glove_features(tmp_path):
    glove_path = tmp_path / "glove.vec"
    glove_path.write_text("A 1 0\nC 2 0\nG 0 1\n<UNK> 0 0\n", encoding="utf-8")
    joblib.dump(
        {
            "estimator_kind": "rf",
            "estimator": _ProbabilityEstimator(),
        },
        tmp_path / "model.joblib",
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "bundle_type": "pair_vector_classical",
                "supports_inference": True,
                "input_schema": "peptide_protein_pairs",
                "label_names": ["interaction"],
                "threshold": 0.5,
                "classic": {
                    "tokenizer": "kmer",
                    "encoder": "glove",
                    "max_length": 2,
                    "k": 1,
                    "glove_path": str(glove_path),
                },
                "pair": {"fusion_mode": "concat_all"},
                "args": {
                    "task_type": "pep_protein_interaction",
                    "tokenizer": "kmer",
                    "encoder": "glove",
                    "model": "rf",
                    "classifier": "rf",
                    "pair_strategy": "two_stream",
                    "interaction_level": "vector",
                    "predictor_type": "classifier",
                },
            }
        ),
        encoding="utf-8",
    )

    class _Store:
        def get_model(self, model_id):
            return {
                "id": model_id,
                "bundle_path": str(tmp_path),
                "supports_inference": True,
            }

    response = ModelService(_Store()).predict(
        "model",
        pairs=[{"peptide": "AC", "protein": "GG"}],
    )

    prediction = response["predictions"][0]
    assert prediction["score"] == 1.0
    assert prediction["score_kind"] == "probability"
    assert prediction["score_range"] == [0.0, 1.0]
    assert prediction["is_interaction"] is True


def test_pair_vector_rf_predict_generates_classic_pssm_features(tmp_path, monkeypatch):
    def fake_profile_cache(bundle, metadata, sequence_by_key):
        del bundle, metadata
        index_path = _write_current_pssm_index(tmp_path, sequence_by_key)
        return {"pssm_index_path": str(index_path)}

    monkeypatch.setattr(
        ModelService,
        "_build_inference_profile_cache",
        staticmethod(fake_profile_cache),
    )
    joblib.dump(
        {
            "estimator_kind": "rf",
            "estimator": _ProbabilityEstimator(),
        },
        tmp_path / "model.joblib",
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "bundle_type": "pair_vector_classical",
                "supports_inference": True,
                "input_schema": "peptide_protein_pairs",
                "label_names": ["interaction"],
                "threshold": 0.5,
                "classic": {
                    "tokenizer": "kmer",
                    "encoder": "pssm",
                    "max_length": 1,
                    "k": 1,
                },
                "profile_generation": {
                    "strategy": "psiblast_subject_fasta",
                    "feature": "pssm",
                    "subject_fasta": "profile_generation/subject.fasta",
                    "iterations": 3,
                    "evalue": 0.001,
                    "threads": 1,
                    "on_missing": "error",
                },
                "pair": {"fusion_mode": "concat_all"},
                "args": {
                    "task_type": "pep_protein_interaction",
                    "tokenizer": "kmer",
                    "encoder": "pssm",
                    "model": "rf",
                    "classifier": "rf",
                    "pair_strategy": "two_stream",
                    "interaction_level": "vector",
                    "predictor_type": "classifier",
                },
            }
        ),
        encoding="utf-8",
    )

    class _Store:
        def get_model(self, model_id):
            return {
                "id": model_id,
                "bundle_path": str(tmp_path),
                "supports_inference": True,
            }

    response = ModelService(_Store()).predict(
        "model",
        pairs=[{"peptide": "AC", "protein": "GG"}],
    )

    prediction = response["predictions"][0]
    assert prediction["score"] == 1.0
    assert prediction["score_kind"] == "probability"
    assert prediction["is_interaction"] is True


def test_pair_graph_nn_predict_returns_pair_rows(tmp_path, monkeypatch):
    def fake_pair_embeddings(metadata, pairs, device, bundle=None):
        peptide = torch.tensor([[1.0, 0.0] for _ in pairs], dtype=torch.float)
        protein = torch.tensor([[0.0, 1.0] for _ in pairs], dtype=torch.float)
        return peptide, protein

    monkeypatch.setattr(
        ModelService,
        "_extract_pair_embeddings",
        staticmethod(fake_pair_embeddings),
    )

    from scripts.train_pair_pipeline import PairGCNClassifier

    model = PairGCNClassifier(input_dim=8, hidden_dim=4, dropout=0.0)
    for parameter in model.parameters():
        parameter.data.zero_()
    torch.save(
        {
            "model_class": "PairGCNClassifier",
            "graph_predictor": "gcn",
            "state_dict": model.state_dict(),
        },
        tmp_path / "model.pt",
    )
    torch.save(
        {"reference_features": torch.zeros((2, 8), dtype=torch.float)},
        tmp_path / "graph_reference.pt",
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "bundle_type": "pair_graph_nn",
                "supports_inference": True,
                "input_schema": "peptide_protein_pairs",
                "label_names": ["interaction"],
                "threshold": 0.4,
                "pretrained": {
                    "tokenizer": "esm2",
                    "model_name": "models/esm2",
                    "max_length": 2,
                    "input_dim": 2,
                },
                "pair": {
                    "fusion_mode": "concat_all",
                    "peptide_dim": 2,
                    "protein_dim": 2,
                    "fused_dim": 8,
                },
                "graph": {
                    "predictor": "gcn",
                    "gnn_k": 1,
                    "hidden_dim": 4,
                    "dropout": 0.0,
                    "input_dim": 8,
                    "reference_path": "graph_reference.pt",
                    "reference_source": "train_val",
                    "aligned_gated": False,
                },
                "args": {
                    "task_type": "pep_protein_interaction",
                    "tokenizer": "esm2",
                    "encoder": "ppi_fusion",
                    "model": "gcn",
                    "pair_strategy": "two_stream",
                    "interaction_level": "vector",
                    "predictor_type": "sample_graph",
                },
            }
        ),
        encoding="utf-8",
    )

    class _Store:
        def get_model(self, model_id):
            return {
                "id": model_id,
                "bundle_path": str(tmp_path),
                "supports_inference": True,
            }

    response = ModelService(_Store()).predict(
        "model",
        pairs=[{"peptide": "AC", "protein": "GG"}, {"peptide": "AA", "protein": "CC"}],
    )

    assert response["labels"] == ["interaction"]
    assert [round(row["score"], 4) for row in response["predictions"]] == [0.5, 0.5]
    assert all(row["score_kind"] == "probability" for row in response["predictions"])
    assert all(row["is_interaction"] is True for row in response["predictions"])


def test_pair_graph_loader_restores_gat_edge_chunk_size(tmp_path):
    from scripts.train_pair_pipeline import PairGATClassifier

    saved_model = PairGATClassifier(
        input_dim=8,
        hidden_dim=4,
        dropout=0.0,
        edge_chunk_size=7,
    )
    torch.save(
        {"state_dict": saved_model.state_dict(), "graph_predictor": "gat"},
        tmp_path / "model.pt",
    )

    loaded_model, _payload, predictor = ModelService._load_pair_graph_model(
        tmp_path,
        {
            "predictor_contract": {"contract_version": 2},
            "graph": {
                "predictor": "gat",
                "hidden_dim": 4,
                "dropout": 0.0,
                "edge_chunk_size": 7,
            },
        },
        input_dim=8,
        device=torch.device("cpu"),
    )

    assert predictor == "gat"
    assert loaded_model.edge_chunk_size == 7
    assert loaded_model.gat_predictor_contract_version == 2


def test_pair_graph_loader_keeps_unversioned_gat_on_legacy_attention(tmp_path):
    from scripts.train_pair_pipeline import PairGATClassifier

    saved_model = PairGATClassifier(input_dim=8, hidden_dim=4, dropout=0.0)
    torch.save({"state_dict": saved_model.state_dict()}, tmp_path / "model.pt")

    loaded_model, _payload, predictor = ModelService._load_pair_graph_model(
        tmp_path,
        {"graph": {"predictor": "gat", "hidden_dim": 4, "dropout": 0.0}},
        input_dim=8,
        device=torch.device("cpu"),
    )

    assert predictor == "gat"
    assert loaded_model.gat_predictor_contract_version == 1


def test_pair_graph_loader_selects_sage_contract_and_legacy_topology(tmp_path):
    from scripts.train_pair_pipeline import PairSAGEClassifier

    saved_model = PairSAGEClassifier(input_dim=8, hidden_dim=4, dropout=0.0)
    torch.save({"state_dict": saved_model.state_dict()}, tmp_path / "model.pt")

    current, _payload, predictor = ModelService._load_pair_graph_model(
        tmp_path,
        {
            "predictor_contract": {"contract_version": 1},
            "graph": {"predictor": "sage", "hidden_dim": 4, "dropout": 0.0},
        },
        input_dim=8,
        device=torch.device("cpu"),
    )
    legacy, _payload, legacy_predictor = ModelService._load_pair_graph_model(
        tmp_path,
        {"graph": {"predictor": "sage", "hidden_dim": 4, "dropout": 0.0}},
        input_dim=8,
        device=torch.device("cpu"),
    )

    assert predictor == legacy_predictor == "sage"
    assert current.sage_predictor_contract_version == 1
    assert legacy.sage_predictor_contract_version == 0


@pytest.mark.parametrize(
    ("contract_version", "expected_graph_kind"),
    [(1, "sage"), (0, "message")],
)
def test_pair_sage_web_inference_selects_contract_topology(
    monkeypatch,
    contract_version,
    expected_graph_kind,
):
    import scripts.train_pair_pipeline as pair_pipeline

    observed = {}

    def fake_build_structure(args, reference_x, query_x, device, graph_kind):
        observed["graph_kind"] = graph_kind
        return torch.empty((2, 0), dtype=torch.long, device=device)

    class FakeSAGE:
        sage_predictor_contract_version = contract_version

        def __call__(self, features, structure):
            return torch.zeros((features.shape[0], 1), device=features.device)

    monkeypatch.setattr(
        pair_pipeline,
        "build_inductive_graph_structure",
        fake_build_structure,
    )
    features = torch.randn(3, 4)
    ModelService._pair_graph_logits(
        FakeSAGE(),
        "sage",
        features[:2],
        features[2:],
        features[:2],
        features[2:],
        Namespace(gnn_k=1),
        "cpu",
    )

    assert observed["graph_kind"] == expected_graph_kind


@pytest.mark.parametrize(
    ("contract_version", "expected_graph_kind"),
    [(2, "gat"), (1, "message")],
)
def test_pair_gat_web_inference_selects_contract_topology(
    monkeypatch,
    contract_version,
    expected_graph_kind,
):
    import scripts.train_pair_pipeline as pair_pipeline

    observed = {}

    def fake_build_structure(args, reference_x, query_x, device, graph_kind):
        observed["graph_kind"] = graph_kind
        return torch.empty((2, 0), dtype=torch.long, device=device)

    class FakeGAT:
        gat_predictor_contract_version = contract_version

        def __call__(self, features, structure):
            return torch.zeros((features.shape[0], 1), device=features.device)

    monkeypatch.setattr(
        pair_pipeline,
        "build_inductive_graph_structure",
        fake_build_structure,
    )
    features = torch.randn(3, 4)
    ModelService._pair_graph_logits(
        FakeGAT(),
        "gat",
        features[:2],
        features[2:],
        features[:2],
        features[2:],
        Namespace(gnn_k=1),
        "cpu",
    )

    assert observed["graph_kind"] == expected_graph_kind


def test_pair_graph_transformer_loader_separates_current_and_legacy(tmp_path):
    from scripts.train_pair_pipeline import (
        LegacyPairGraphTransformerClassifier,
        PairGraphTransformerClassifier,
    )

    current_dir = tmp_path / "current"
    legacy_dir = tmp_path / "legacy"
    current_dir.mkdir()
    legacy_dir.mkdir()
    current_saved = PairGraphTransformerClassifier(
        input_dim=8,
        hidden_dim=8,
        heads=4,
        dropout=0.0,
    )
    legacy_saved = LegacyPairGraphTransformerClassifier(
        input_dim=8,
        hidden_dim=8,
        heads=4,
        dropout=0.0,
    )
    torch.save({"state_dict": current_saved.state_dict()}, current_dir / "model.pt")
    torch.save({"state_dict": legacy_saved.state_dict()}, legacy_dir / "model.pt")

    current, _payload, predictor = ModelService._load_pair_graph_model(
        current_dir,
        {
            "predictor_contract": {
                "implementation": ("pepplatform_inductive_pair_sample_graph_graph_transformer"),
                "contract_version": 1,
                "heads": 4,
                "layers": 2,
            },
            "graph": {
                "predictor": "graph_transformer",
                "hidden_dim": 8,
                "heads": 4,
                "layers": 2,
                "dropout": 0.0,
            },
        },
        input_dim=8,
        device=torch.device("cpu"),
    )
    with pytest.warns(RuntimeWarning, match="legacy_only"):
        legacy, _payload, legacy_predictor = ModelService._load_pair_graph_model(
            legacy_dir,
            {
                "graph": {
                    "predictor": "graph_transformer",
                    "hidden_dim": 8,
                    "dropout": 0.0,
                }
            },
            input_dim=8,
            device=torch.device("cpu"),
        )

    assert predictor == legacy_predictor == "graph_transformer"
    assert isinstance(current, PairGraphTransformerClassifier)
    assert current.graph_transformer_predictor_contract_version == 1
    assert isinstance(legacy, LegacyPairGraphTransformerClassifier)


@pytest.mark.parametrize(
    ("contract_version", "expected_graph_kind"),
    [(1, "graph_transformer"), (0, "message")],
)
def test_pair_graph_transformer_web_inference_selects_contract_topology(
    monkeypatch,
    contract_version,
    expected_graph_kind,
):
    import scripts.train_pair_pipeline as pair_pipeline

    observed = {}

    def fake_build_structure(args, reference_x, query_x, device, graph_kind):
        observed["graph_kind"] = graph_kind
        return torch.empty((2, 0), dtype=torch.long, device=device)

    class FakeGraphTransformer:
        graph_transformer_predictor_contract_version = contract_version

        def __call__(self, features, structure):
            return torch.zeros((features.shape[0], 1), device=features.device)

    monkeypatch.setattr(
        pair_pipeline,
        "build_inductive_graph_structure",
        fake_build_structure,
    )
    features = torch.randn(3, 4)
    ModelService._pair_graph_logits(
        FakeGraphTransformer(),
        "graph_transformer",
        features[:2],
        features[2:],
        features[:2],
        features[2:],
        Namespace(gnn_k=1),
        "cpu",
    )

    assert observed["graph_kind"] == expected_graph_kind


def test_pair_token_features_concatenate_recorded_pretrained_tokens(monkeypatch):
    def fake_token_sequences(metadata, sequences, device):
        if sequences == ["AC"]:
            return torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])
        if sequences == ["GGG"]:
            return torch.tensor([[[0.0, 3.0], [0.0, 4.0]]])
        raise AssertionError(sequences)

    monkeypatch.setattr(
        ModelService,
        "_extract_pretrained_token_sequences",
        staticmethod(fake_token_sequences),
    )

    features = ModelService._extract_pair_token_features(
        {"pretrained": {"tokenizer": "esm2", "model_name": "models/esm2", "max_length": 2}},
        [{"peptide": "AC", "protein": "GGG"}],
        torch.device("cpu"),
    )

    assert features.shape == (1, 4, 2)
    assert features.tolist() == [[[1.0, 0.0], [2.0, 0.0], [0.0, 3.0], [0.0, 4.0]]]


def test_pair_token_features_rebuild_classic_pssm_tokens(tmp_path, monkeypatch):
    def fake_profile_cache(bundle, metadata, sequence_by_key):
        del bundle, metadata
        index_path = _write_current_pssm_index(tmp_path, sequence_by_key)
        return {"pssm_index_path": str(index_path)}

    monkeypatch.setattr(
        ModelService,
        "_build_inference_profile_cache",
        staticmethod(fake_profile_cache),
    )

    features = ModelService._extract_pair_token_features(
        {
            "classic": {
                "tokenizer": "kmer",
                "encoder": "pssm",
                "max_length": 1,
                "k": 1,
            },
            "profile_generation": {
                "strategy": "psiblast_subject_fasta",
                "feature": "pssm",
                "subject_fasta": "profile_generation/subject.fasta",
                "iterations": 3,
                "evalue": 0.001,
                "threads": 1,
                "on_missing": "error",
            },
        },
        [{"peptide": "AC", "protein": "GG"}],
        torch.device("cpu"),
        bundle=tmp_path,
    )

    assert features.shape == (1, 2, 20)
    assert torch.allclose(features, torch.ones((1, 2, 20), dtype=torch.float))


def test_pair_token_nn_predict_returns_pair_rows(tmp_path, monkeypatch):
    def fake_token_sequences(metadata, sequences, device):
        rows = []
        for sequence in sequences:
            base = float(len(sequence))
            rows.append([[base, 0.0], [0.0, base]])
        return torch.tensor(rows, dtype=torch.float)

    monkeypatch.setattr(
        ModelService,
        "_extract_pretrained_token_sequences",
        staticmethod(fake_token_sequences),
    )

    from scripts.train_pair_pipeline import PairCrossAttentionClassifier

    model = PairCrossAttentionClassifier(
        seq_len=2,
        encoder_name="ppi_fusion",
        in_dim=2,
        hidden_dim=4,
        dropout=0.0,
        n_heads=2,
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    torch.save(
        {
            "model_class": "PairCrossAttentionClassifier",
            "state_dict": model.state_dict(),
        },
        tmp_path / "model.pt",
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "bundle_type": "pair_token_nn",
                "supports_inference": True,
                "input_schema": "peptide_protein_pairs",
                "label_names": ["interaction"],
                "threshold": 0.4,
                "pretrained": {
                    "tokenizer": "esm2",
                    "model_name": "models/esm2",
                    "max_length": 2,
                    "token_dim": 2,
                },
                "pair": {
                    "interaction_level": "token",
                    "interaction_model": "cross_attn",
                    "seq_len": 2,
                    "token_dim": 2,
                    "hidden_dim": 4,
                    "n_heads": 2,
                    "dropout": 0.0,
                },
                "args": {
                    "task_type": "pep_protein_interaction",
                    "tokenizer": "esm2",
                    "encoder": "ppi_fusion",
                    "model": "cross_attn+mlp",
                    "classifier": "mlp",
                    "pair_strategy": "two_stream",
                    "interaction_level": "token",
                    "interaction_model": "cross_attn",
                    "training_mode": "end_to_end",
                    "batch_size": 1,
                    "dropout": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )

    class _Store:
        def get_model(self, model_id):
            return {
                "id": model_id,
                "bundle_path": str(tmp_path),
                "supports_inference": True,
            }

    response = ModelService(_Store()).predict(
        "model",
        pairs=[{"peptide": "AC", "protein": "GGG"}],
    )

    assert response["predictions"] == [
        {
            "peptide": "AC",
            "protein": "GGG",
            "score": 0.5,
            "score_kind": "probability",
            "score_range": [0.0, 1.0],
            "threshold": 0.4,
            "is_interaction": True,
            "prediction_label": "Interact",
            "scores": [0.5],
        }
    ]


def test_pair_sequence_nn_predict_returns_pair_rows(tmp_path, monkeypatch):
    def fake_sequence_features(metadata, pairs, device, bundle=None):
        return torch.tensor([[1.0, 2.0, 3.0, 4.0] for _ in pairs], dtype=torch.float)

    monkeypatch.setattr(
        ModelService,
        "_extract_pair_sequence_features",
        staticmethod(fake_sequence_features),
    )

    model = torch.nn.Linear(4, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    torch.save(model, tmp_path / "model.pt")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "bundle_type": "pair_sequence_nn",
                "supports_inference": True,
                "input_schema": "peptide_protein_pairs",
                "label_names": ["interaction"],
                "threshold": 0.6,
                "classic": {"tokenizer": "kmer", "encoder": "index", "max_length": 2, "k": 1},
                "pair": {"pair_strategy": "single_stream", "sequence_model": "cnn", "seq_len": 2},
                "args": {"task_type": "pep_protein_interaction"},
            }
        ),
        encoding="utf-8",
    )

    class _Store:
        def get_model(self, model_id):
            return {"id": model_id, "bundle_path": str(tmp_path), "supports_inference": True}

    response = ModelService(_Store()).predict("model", pairs=[{"peptide": "AC", "protein": "GG"}])

    prediction = response["predictions"][0]
    assert prediction["score"] == 0.5
    assert prediction["score_kind"] == "probability"
    assert prediction["is_interaction"] is False


@pytest.mark.parametrize(
    "implementation",
    [
        "pepplatform_bidirectional_elman_rnn",
        "pepplatform_bidirectional_pytorch_gru",
        "pepplatform_bidirectional_pytorch_lstm",
        "pepplatform_pytorch_transformer_encoder",
    ],
)
def test_pair_recurrent_v2_web_features_keep_internal_explicit_masks(
    tmp_path,
    implementation,
):
    glove_path = tmp_path / "glove.vec"
    glove_path.write_text("A 1 0\nC 0 1\n", encoding="utf-8")
    metadata = {
        "classic": {
            "tokenizer": "kmer",
            "encoder": "glove",
            "max_length": 3,
            "k": 1,
            "glove_path": str(glove_path),
        },
        "predictor_contract": {
            "implementation": implementation,
            "contract_version": 2,
            "status": "verified_platform_variant",
        },
    }

    features = ModelService._extract_pair_sequence_features(
        metadata,
        [{"peptide": "AX", "protein": "C"}],
        torch.device("cpu"),
    )

    assert isinstance(features, ResidueModelInput)
    assert features.mask.tolist() == [[True, True, False, True, False, False]]
    assert torch.count_nonzero(features.features[0, 1]) == 0


def test_pair_embedding_features_use_saved_stage_one_encoder(tmp_path, monkeypatch):
    def fake_sequence_features(metadata, pairs, device, bundle=None):
        return torch.tensor([[2.0, 3.0, 4.0, 5.0]], dtype=torch.float)

    monkeypatch.setattr(
        ModelService,
        "_extract_pair_sequence_features",
        staticmethod(fake_sequence_features),
    )
    torch.save(_EmbeddingEncoder(), tmp_path / "embedding_encoder.pt")

    embeddings = ModelService._extract_pair_embedding_features(
        {
            "classic": {"tokenizer": "kmer", "encoder": "index", "max_length": 2, "k": 1},
            "pair": {"pair_strategy": "single_stream"},
            "embedding_export": {"encoder_path": "embedding_encoder.pt"},
        },
        [{"peptide": "AC", "protein": "GG"}],
        torch.device("cpu"),
        bundle=tmp_path,
    )

    assert embeddings.tolist() == [[2.0, 3.0]]


def test_pair_embedding_features_run_stage_one_encoder_in_batches(tmp_path, monkeypatch):
    inputs = torch.arange(30, dtype=torch.float).reshape(5, 6)
    calls = []
    load_calls = []

    def fake_sequence_features(metadata, pairs, device, bundle=None):
        return inputs

    class RecordingEncoder(torch.nn.Module):
        def forward_with_embedding(self, batch):
            calls.append(int(batch.shape[0]))
            embeddings = batch[:, :2]
            return embeddings.sum(dim=1, keepdim=True), embeddings

    (tmp_path / "embedding_encoder.pt").write_bytes(b"placeholder")
    monkeypatch.setattr(
        ModelService,
        "_extract_pair_sequence_features",
        staticmethod(fake_sequence_features),
    )

    def fake_compatibility_loader(path, device):
        load_calls.append((Path(path), device))
        return RecordingEncoder()

    monkeypatch.setattr(
        ModelService,
        "_load_torch_model",
        staticmethod(fake_compatibility_loader),
    )
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail(
            "pair embedding extraction must use the compatibility loader"
        ),
    )

    embeddings = ModelService._extract_pair_embedding_features(
        {
            "classic": {"tokenizer": "kmer", "encoder": "index", "max_length": 2, "k": 1},
            "pair": {"pair_strategy": "single_stream"},
            "embedding_export": {"encoder_path": "embedding_encoder.pt"},
            "args": {"inference_batch_size": 2},
        },
        [{"peptide": "AC", "protein": "GG"} for _ in range(5)],
        torch.device("cpu"),
        bundle=tmp_path,
    )

    assert calls == [2, 2, 1]
    assert load_calls == [(tmp_path / "embedding_encoder.pt", torch.device("cpu"))]
    torch.testing.assert_close(embeddings, inputs[:, :2])


def test_aligned_gated_pair_model_loader_installs_main_module_class(monkeypatch):
    import __main__

    calls = []
    previous = getattr(__main__, "AlignedGatedClassifier", None)
    had_previous = hasattr(__main__, "AlignedGatedClassifier")

    def fake_load(*_args, **_kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise AttributeError("Can't get attribute 'AlignedGatedClassifier' on __main__")
        assert hasattr(__main__, "AlignedGatedClassifier")
        return "loaded"

    monkeypatch.setattr(torch, "load", fake_load)
    assert ModelService._load_torch_model("model.pt", torch.device("cpu")) == "loaded"
    assert len(calls) == 2
    if had_previous:
        assert getattr(__main__, "AlignedGatedClassifier") is previous
    else:
        assert not hasattr(__main__, "AlignedGatedClassifier")


def test_pair_bicross_embedding_loader_installs_and_restores_main_alias(monkeypatch):
    import __main__

    from scripts.train_pair_pipeline import PairBiCrossAttentionClassifier

    calls = []
    previous = getattr(__main__, "PairBiCrossAttentionClassifier", None)
    had_previous = hasattr(__main__, "PairBiCrossAttentionClassifier")

    def fake_load(*_args, **_kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise AttributeError("Can't get attribute 'PairBiCrossAttentionClassifier' on __main__")
        assert getattr(__main__, "PairBiCrossAttentionClassifier") is PairBiCrossAttentionClassifier
        return "loaded"

    monkeypatch.setattr(torch, "load", fake_load)
    assert ModelService._load_torch_model("encoder.pt", torch.device("cpu")) == "loaded"
    assert len(calls) == 2
    if had_previous:
        assert getattr(__main__, "PairBiCrossAttentionClassifier") is previous
    else:
        assert not hasattr(__main__, "PairBiCrossAttentionClassifier")


def _legacy_concatenated_two_stream_metadata(*, feature_dim=28):
    descriptor = {
        "batch_size": 15365,
        "dtype": "torch.float32",
        "explicit_mask": False,
        "feature_dim": feature_dim,
        "input_kind": "feature",
        "rank": 3,
        "token_length": 1024,
    }
    return {
        "bundle_type": "pair_token_nn",
        "towers": None,
        "classic": {
            "tokenizer": "kmer",
            "encoder": "onehot",
            "max_length": 512,
            "token_dim": 28,
        },
        "pair": {
            "pair_strategy": "dual_tower",
            "tower_mode": "homogeneous",
            "interaction_level": "token",
            "interaction_model": "cross_attn",
            "seq_len": 512,
            "partner_seq_len": 0,
            "token_dim": 28,
            "partner_token_dim": 0,
        },
        "pair_strategy_contract": {
            "contract_version": 1,
            "strategy": "dual_tower",
            "role_order": ["peptide", "partner"],
            "tower_mode": "homogeneous",
            "tower_processing": "independent_encoders",
            "interaction_level": "token",
            "predictor_type": "classifier",
            "input_adapter": "dual_tower_token_interaction",
            "interaction_model": "cross_attn",
            "towers": {
                "peptide": dict(descriptor),
                "partner": dict(descriptor),
            },
        },
    }


def test_legacy_concatenated_two_stream_length_is_narrowly_compatible():
    metadata = _legacy_concatenated_two_stream_metadata()
    peptide = torch.zeros((2, 512, 28), dtype=torch.float32)
    partner = torch.zeros((2, 512, 28), dtype=torch.float32)

    with pytest.warns(RuntimeWarning, match="concatenated training tensor length"):
        contract = ModelService._validate_pair_strategy_metadata(
            metadata,
            peptide,
            partner,
        )

    assert contract["compatibility_mode"] == "legacy_concatenated_role_length_v1"
    assert contract["streams"]["peptide"]["token_length"] == 512
    assert contract["streams"]["partner"]["token_length"] == 512


def test_legacy_concatenated_two_stream_does_not_relax_other_mismatches():
    metadata = _legacy_concatenated_two_stream_metadata(feature_dim=27)
    peptide = torch.zeros((1, 512, 28), dtype=torch.float32)
    partner = torch.zeros((1, 512, 28), dtype=torch.float32)

    with pytest.raises(
        InferenceUnsupportedError,
        match="does not match its artifact strategy contract",
    ):
        ModelService._validate_pair_strategy_metadata(metadata, peptide, partner)


def test_pair_embedding_classical_predict_returns_pair_rows(tmp_path, monkeypatch):
    def fake_embedding_features(metadata, pairs, device, bundle=None):
        return torch.tensor([[0.2, 0.4] for _ in pairs], dtype=torch.float)

    monkeypatch.setattr(
        ModelService,
        "_extract_pair_embedding_features",
        staticmethod(fake_embedding_features),
    )
    joblib.dump(
        {
            "estimator_kind": "rf",
            "estimator": _ProbabilityEstimator(),
        },
        tmp_path / "model.joblib",
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "bundle_type": "pair_embedding_classical",
                "supports_inference": True,
                "input_schema": "peptide_protein_pairs",
                "label_names": ["interaction"],
                "threshold": 0.5,
                "classic": {"tokenizer": "kmer", "encoder": "index", "max_length": 2, "k": 1},
                "pair": {"pair_strategy": "single_stream"},
                "embedding_export": {"encoder_path": "embedding_encoder.pt", "downstream": "rf"},
                "args": {"task_type": "pep_protein_interaction"},
            }
        ),
        encoding="utf-8",
    )

    class _Store:
        def get_model(self, model_id):
            return {"id": model_id, "bundle_path": str(tmp_path), "supports_inference": True}

    response = ModelService(_Store()).predict("model", pairs=[{"peptide": "AC", "protein": "GG"}])

    prediction = response["predictions"][0]
    assert prediction["score"] == pytest.approx(0.6)
    assert prediction["score_kind"] == "probability"
    assert prediction["is_interaction"] is True


def test_pair_embedding_nn_predict_returns_pair_rows(tmp_path, monkeypatch):
    def fake_embedding_features(metadata, pairs, device, bundle=None):
        return torch.tensor([[1.0, 1.0] for _ in pairs], dtype=torch.float)

    monkeypatch.setattr(
        ModelService,
        "_extract_pair_embedding_features",
        staticmethod(fake_embedding_features),
    )
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    torch.save(model, tmp_path / "model.pt")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "bundle_type": "pair_embedding_nn",
                "supports_inference": True,
                "input_schema": "peptide_protein_pairs",
                "label_names": ["interaction"],
                "threshold": 0.4,
                "pretrained": {"tokenizer": "esm2", "model_name": "models/esm2", "max_length": 2},
                "pair": {"interaction_level": "token", "interaction_model": "cross_attn"},
                "embedding_export": {"encoder_path": "embedding_encoder.pt", "downstream": "mlp"},
                "args": {"task_type": "pep_protein_interaction"},
            }
        ),
        encoding="utf-8",
    )

    class _Store:
        def get_model(self, model_id):
            return {"id": model_id, "bundle_path": str(tmp_path), "supports_inference": True}

    response = ModelService(_Store()).predict("model", pairs=[{"peptide": "AC", "protein": "GG"}])

    prediction = response["predictions"][0]
    assert prediction["score"] == 0.5
    assert prediction["score_kind"] == "probability"
    assert prediction["is_interaction"] is True


def test_pair_token_bi_cross_nn_predict_returns_pair_rows(tmp_path, monkeypatch):
    def fake_token_features(metadata, pairs, device, bundle=None):
        return torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]] for _ in pairs],
            dtype=torch.float,
        )

    monkeypatch.setattr(
        ModelService,
        "_extract_pair_token_features",
        staticmethod(fake_token_features),
    )

    from scripts.train_pair_pipeline import PairBiCrossAttentionClassifier

    model = PairBiCrossAttentionClassifier(
        seq_len=2,
        encoder_name="ppi_fusion",
        in_dim=2,
        hidden_dim=4,
        dropout=0.0,
        n_heads=2,
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    torch.save(
        {
            "model_class": "PairBiCrossAttentionClassifier",
            "state_dict": model.state_dict(),
        },
        tmp_path / "model.pt",
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "bundle_type": "pair_token_nn",
                "supports_inference": True,
                "input_schema": "peptide_protein_pairs",
                "label_names": ["interaction"],
                "threshold": 0.5,
                "pretrained": {
                    "tokenizer": "esm2",
                    "model_name": "models/esm2",
                    "max_length": 2,
                    "token_dim": 2,
                },
                "pair": {
                    "interaction_level": "token",
                    "interaction_model": "bi_cross_attn",
                    "encoder": "ppi_fusion",
                    "seq_len": 2,
                    "token_dim": 2,
                    "hidden_dim": 4,
                    "n_heads": 2,
                    "dropout": 0.0,
                },
                "args": {
                    "task_type": "pep_protein_interaction",
                    "tokenizer": "esm2",
                    "encoder": "ppi_fusion",
                    "model": "bi_cross_attn+mlp",
                    "classifier": "mlp",
                    "pair_strategy": "two_stream",
                    "interaction_level": "token",
                    "interaction_model": "bi_cross_attn",
                    "training_mode": "end_to_end",
                    "batch_size": 1,
                    "dropout": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )

    class _Store:
        def get_model(self, model_id):
            return {
                "id": model_id,
                "bundle_path": str(tmp_path),
                "supports_inference": True,
            }

    response = ModelService(_Store()).predict(
        "model",
        pairs=[{"peptide": "AC", "protein": "GGG"}],
    )

    prediction = response["predictions"][0]
    assert prediction["score"] == 0.5
    assert prediction["score_kind"] == "probability"
    assert prediction["is_interaction"] is True


def test_pair_vector_svm_predict_marks_margin_score(tmp_path, monkeypatch):
    class _FakeTokenizer:
        def __init__(self, model_name, max_length):
            self.model_name = model_name
            self.max_length = max_length

        def batch_tokenize(self, sequences, device=None):
            return {"sequences": sequences}

    class _FakeEncoder:
        def __init__(self, model_name, device=None):
            self.model_name = model_name
            self.device = device

        def encode(self, inputs):
            return torch.tensor([[float(len(sequence)), 1.0] for sequence in inputs["sequences"]])

    import pepplatform.encoders as encoders
    import pepplatform.tokenizers as tokenizers

    monkeypatch.setattr(tokenizers, "EsmTokenizer", _FakeTokenizer)
    monkeypatch.setattr(encoders, "EsmEncoder", _FakeEncoder)

    joblib.dump(
        {
            "estimator_kind": "svm",
            "estimators": [_DecisionEstimator()],
        },
        tmp_path / "model.joblib",
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "bundle_type": "pair_vector_classical",
                "supports_inference": True,
                "input_schema": "peptide_protein_pairs",
                "label_names": ["interaction"],
                "threshold": 20.0,
                "pretrained": {
                    "tokenizer": "esm2",
                    "model_name": "models/esm2",
                    "max_length": 60,
                },
                "pair": {"fusion_mode": "concat_all"},
                "args": {
                    "task_type": "pep_protein_interaction",
                    "tokenizer": "esm2",
                    "encoder": "ppi_fusion",
                    "model": "svm",
                    "classifier": "svm",
                    "pair_strategy": "two_stream",
                    "interaction_level": "vector",
                    "predictor_type": "classifier",
                    "batch_size": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    class _Store:
        def get_model(self, model_id):
            return {
                "id": model_id,
                "bundle_path": str(tmp_path),
                "supports_inference": True,
            }

    response = ModelService(_Store()).predict(
        "model",
        pairs=[{"peptide": "AC", "protein": "GGG"}],
    )

    prediction = response["predictions"][0]
    assert prediction["score"] == 15.0
    assert prediction["score_kind"] == "margin"
    assert "score_range" not in prediction
    assert prediction["threshold"] == 20.0
    assert prediction["is_interaction"] is False
