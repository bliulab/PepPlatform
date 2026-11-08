import hashlib
import json
import warnings
from argparse import Namespace

import joblib
import numpy as np
import pytest
import sklearn
import torch
from torch.utils.data import TensorDataset

import pepplatform.lda as lda_module
from pepplatform.lda import (
    LDA_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    LDA_MAX_ITER,
    LDA_VARIANT,
    LDAContractError,
    build_lda_contract,
    fit_lda_transform,
    transform_lda_held_out,
    validate_lda_contract,
    validate_lda_route,
)
from pepplatform.ml_trainers import ClassicalMLTrainer
from pepplatform.web.inference import InferenceUnsupportedError, ModelService


def _matrix():
    return np.asarray(
        [
            [2.0, 0.0, 1.0, 0.0],
            [0.0, 3.0, 0.0, 1.0],
            [1.0, 1.0, 2.0, 0.0],
            [0.0, 1.0, 1.0, 3.0],
            [3.0, 0.0, 0.0, 1.0],
            [1.0, 2.0, 0.0, 2.0],
        ],
        dtype=np.float64,
    )


def _upstream(feature_dimension=4):
    return {
        "tokenizer": "kmer",
        "encoder": "bow",
        "k": 1,
        "frequency_encoder_contract": {
            "contract_version": 2,
            "stream_policy": "full_token_stream",
            "bow_formula": "x_j = c_j",
            "tf_formula": "c_j / sum_k(c_k)",
            "idf_formula": "ln((N + 1) / (df_j + 1)) + 1",
            "post_normalization": "none",
            "frequency_max_raw_length": 50_000,
            "feature_dimension": feature_dimension,
            "vocabulary_sha256": hashlib.sha256(b"test-vocabulary").hexdigest(),
            "encoder": "bow",
            "training_document_count": 0,
            "training_corpus_sha256": hashlib.sha256(b"train-corpus").hexdigest(),
            "fit_scope": "sequence_train_split",
        },
    }


def _fit(validation=None, *, topics=2, limit=LDA_DEFAULT_DENSE_MEMORY_LIMIT_GIB):
    train = _matrix()
    validation = train[:2] if validation is None else validation
    return fit_lda_transform(
        train,
        validation,
        n_components=topics,
        seed=17,
        dense_memory_limit_gib=limit,
        upstream_feature_contract=_upstream(),
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_lda_is_deterministic_train_only_batch_variational_bayes():
    train_a, val_a, _, estimator_a, contract_a = _fit()
    train_b, val_b, _, estimator_b, contract_b = _fit(_matrix()[2:5] * 100.0)

    np.testing.assert_allclose(train_a, train_b)
    np.testing.assert_allclose(estimator_a.components_, estimator_b.components_)
    assert val_a.shape == (2, 2)
    assert val_b.shape == (3, 2)
    assert contract_a == contract_b
    assert contract_a["variant"] == LDA_VARIANT
    assert contract_a["feature_fit_scope"] == "train_only"
    assert contract_a["input_semantics"] == "raw_nonnegative_integer_bow_counts"
    assert contract_a["diagnostics"]["converged"] is True
    assert contract_a["diagnostics"]["n_iter"] < LDA_MAX_ITER
    assert contract_a["estimator_params"]["learning_method"] == "batch"
    assert contract_a["estimator_params"]["evaluate_every"] == 1
    assert contract_a["estimator_params"]["n_jobs"] == 1


def test_lda_held_out_transform_is_batch_invariant_reload_stable_and_normalized(tmp_path):
    _, combined, _, estimator, contract = _fit(_matrix()[1:5])
    individual = np.vstack(
        [
            transform_lda_held_out(
                estimator,
                row[None, :],
                split="validation",
                input_feature_count=4,
                output_dimension=2,
                dense_memory_limit_gib=4.0,
            )
            for row in _matrix()[1:5]
        ]
    )
    np.testing.assert_allclose(combined, individual)
    np.testing.assert_allclose(combined.sum(axis=1), 1.0)
    assert np.all(combined >= 0.0)

    zero = transform_lda_held_out(
        estimator,
        np.zeros((1, 4), dtype=np.float64),
        split="validation",
        input_feature_count=4,
        output_dimension=2,
        dense_memory_limit_gib=4.0,
    )
    np.testing.assert_allclose(zero, [[0.5, 0.5]])

    path = tmp_path / "lda.joblib"
    joblib.dump({"estimator": estimator, "contract": contract}, path)
    restored = joblib.load(path)
    validate_lda_contract(
        restored["contract"],
        estimator=restored["estimator"],
        upstream_feature_contract=_upstream(),
    )
    np.testing.assert_allclose(
        combined,
        restored["estimator"].transform(_matrix()[1:5]),
    )


def test_lda_topics_are_not_artificially_bounded_by_feature_width():
    train, validation, _, estimator, contract = _fit(topics=5)

    assert train.shape == (6, 5)
    assert validation.shape == (2, 5)
    assert estimator.n_components == 5
    assert contract["output_dimension"] == 5


@pytest.mark.parametrize("encoder", ["tfidf", "textrank", "onehot", ""])
def test_lda_rejects_non_bow_encoders(encoder):
    with pytest.raises(LDAContractError) as exc_info:
        validate_lda_route(
            Namespace(reduction="lda", encoder=encoder, lda_topics=2),
            pair=False,
        )
    assert exc_info.value.code == "lda_encoder_unsupported"


@pytest.mark.parametrize("task_type", ["pep_protein_interaction", "pep_hlaii", "pep_hlai"])
def test_lda_rejects_incomplete_pair_routes(task_type):
    with pytest.raises(LDAContractError) as exc_info:
        validate_lda_route(
            Namespace(
                task_type=task_type,
                reduction="lda",
                encoder="bow",
                lda_topics=2,
            )
        )
    assert exc_info.value.code == "lda_pair_route_unsupported"
    assert "fused vectors" in str(exc_info.value)


@pytest.mark.parametrize("topics", [0, -1, True, 1.5, "not-an-integer"])
def test_lda_rejects_invalid_topic_counts(topics):
    with pytest.raises(LDAContractError) as exc_info:
        _fit(topics=topics)
    assert exc_info.value.code == "lda_topics_invalid"


@pytest.mark.parametrize(
    ("matrix", "code"),
    [
        (np.asarray([[1.0, -1.0], [0.0, 1.0]]), "lda_matrix_negative"),
        (np.asarray([[1.0, 0.5], [0.0, 1.0]]), "lda_matrix_not_counts"),
        (np.asarray([[1.0, np.nan], [0.0, 1.0]]), "lda_matrix_nonfinite"),
        (np.asarray([[1.0, np.inf], [0.0, 1.0]]), "lda_matrix_nonfinite"),
        (np.zeros((2, 2), dtype=np.float64), "lda_training_counts_empty"),
    ],
)
def test_lda_rejects_invalid_training_count_matrices(matrix, code):
    with pytest.raises(LDAContractError) as exc_info:
        fit_lda_transform(
            matrix,
            matrix[:1],
            n_components=2,
            seed=42,
            upstream_feature_contract=_upstream(matrix.shape[1]),
        )
    assert exc_info.value.code == code


def test_lda_allows_zero_rows_and_columns_when_training_has_counts():
    train = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    transformed, _, _, _, contract = fit_lda_transform(
        train,
        train[:2],
        n_components=2,
        seed=42,
        upstream_feature_contract=_upstream(),
    )
    assert transformed.shape == (4, 2)
    np.testing.assert_allclose(transformed.sum(axis=1), 1.0)
    assert contract["zero_document_semantics"] == "symmetric_doc_topic_prior"


def test_lda_memory_guard_fails_before_fit():
    with pytest.raises(LDAContractError) as exc_info:
        _fit(limit=1e-12)
    assert exc_info.value.code == "lda_dense_memory_limit_exceeded"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda contract: contract.pop("frequency_encoder_contract"),
        lambda contract: contract["frequency_encoder_contract"].update(contract_version=1),
        lambda contract: contract["frequency_encoder_contract"].update(post_normalization="l1"),
        lambda contract: contract["frequency_encoder_contract"].update(fit_scope="unknown"),
        lambda contract: contract["frequency_encoder_contract"].update(vocabulary_sha256="bad"),
        lambda contract: contract["frequency_encoder_contract"].update(
            training_corpus_sha256="bad"
        ),
        lambda contract: contract.update(tokenizer="wordpiece"),
    ],
)
def test_lda_rejects_missing_or_incompatible_upstream_contract(mutation):
    upstream = _upstream()
    mutation(upstream)
    with pytest.raises(LDAContractError) as exc_info:
        fit_lda_transform(
            _matrix(),
            _matrix()[:2],
            n_components=2,
            seed=17,
            upstream_feature_contract=upstream,
        )
    assert exc_info.value.code.startswith("lda_upstream")


def test_lda_fit_warning_and_nonconvergence_fail_closed(monkeypatch):
    original_factory = lda_module._new_estimator
    estimator = lda_module._new_estimator(2, 17)
    original_fit = estimator.fit

    def fit_with_warning(matrix):
        warnings.warn("synthetic fit warning", RuntimeWarning)
        return original_fit(matrix)

    monkeypatch.setattr(estimator, "fit", fit_with_warning)
    monkeypatch.setattr(lda_module, "_new_estimator", lambda _topics, _seed: estimator)
    with pytest.raises(LDAContractError) as exc_info:
        _fit()
    assert exc_info.value.code == "lda_fit_warning"

    monkeypatch.setattr(lda_module, "_new_estimator", original_factory)
    _, _, _, fitted, contract = _fit()
    fitted.n_iter_ = LDA_MAX_ITER
    with pytest.raises(LDAContractError) as exc_info:
        validate_lda_contract(contract, estimator=fitted)
    assert exc_info.value.code == "lda_not_converged"


def test_lda_artifact_contract_and_model_digest_round_trip(tmp_path):
    args = Namespace(
        task_type="pep_function",
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="bow",
        reduction="lda",
        lda_topics=2,
        lda_dense_memory_limit_gib=4.0,
        ml_scaler="none",
        ml_reduction="none",
        logreg_c=1.0,
        logreg_max_iter=100,
        logreg_class_weight="none",
        logreg_convergence_policy="strict",
        report_aupr=False,
        seed=42,
        _artifact_info={
            "classic": _upstream(),
            "label_names": ["active"],
        },
    )
    trainer = ClassicalMLTrainer(args)
    features = torch.tensor(_matrix(), dtype=torch.float32)
    labels = torch.tensor([[0], [1], [0], [1], [0], [1]], dtype=torch.float32)
    train = TensorDataset(features, labels)
    train.label_names = ["active"]
    validation = TensorDataset(features[:2], labels[:2])

    trainer.run_logreg(train, validation)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    payload = joblib.load(tmp_path / "model.joblib")
    assert metadata["lda_contract"] == payload["lda_contract"]
    validate_lda_contract(
        metadata["lda_contract"],
        estimator=payload["task_reducer"]["value"],
        upstream_feature_contract=metadata["classic"],
    )
    assert metadata["model_artifact"] == {
        "path": "model.joblib",
        "publication": "atomic_replace",
        "sha256": _sha256(tmp_path / "model.joblib"),
        "size_bytes": (tmp_path / "model.joblib").stat().st_size,
    }
    assert metadata["lda_contract"]["implementation_version"] == sklearn.__version__
    ModelService._validate_lda_artifact(metadata, tmp_path)
    ModelService._load_classical_payload(tmp_path, metadata)


def test_web_lda_validation_rejects_tampering_upstream_and_version_mismatch(tmp_path):
    _, _, _, estimator, contract = _fit()
    payload = {
        "task_reducer": {"kind": "sklearn", "value": estimator},
        "lda_contract": contract,
    }
    model_path = tmp_path / "model.joblib"
    joblib.dump(payload, model_path)
    metadata = {
        "task_type": "pep_function",
        "args": {"reduction": "lda"},
        "classic": _upstream(),
        "lda_contract": contract,
        "model_artifact": {
            "path": "model.joblib",
            "publication": "atomic_replace",
            "sha256": _sha256(model_path),
            "size_bytes": model_path.stat().st_size,
        },
    }

    metadata["model_artifact"]["sha256"] = "0" * 64
    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._validate_lda_artifact(metadata, tmp_path)
    assert exc_info.value.code == "lda_model_integrity_failed"

    metadata["model_artifact"]["sha256"] = _sha256(model_path)
    metadata["classic"]["frequency_encoder_contract"]["vocabulary_sha256"] = "1" * 64
    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._validate_lda_artifact(metadata, tmp_path)
    assert exc_info.value.code == "lda_upstream_contract_mismatch"

    metadata["classic"] = _upstream()
    metadata["lda_contract"] = {**contract, "implementation_version": "0.invalid"}
    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._validate_lda_artifact(metadata, tmp_path)
    assert exc_info.value.code == "lda_runtime_version_mismatch"


def test_web_legacy_lda_warns_and_preserves_historical_transform():
    metadata = {
        "task_type": "pep_protein_interaction",
        "args": {"reduction": "lda"},
    }
    with pytest.warns(RuntimeWarning, match="legacy_only"):
        assert ModelService._validate_lda_artifact(metadata, ".") is False

    _, _, _, estimator, _ = _fit()
    features = _matrix()[:2]
    expected = estimator.transform(features)
    transformed = ModelService._transform_classical_features(
        {
            "task_reducer": {"kind": "sklearn", "value": estimator},
            "ml_scaler": None,
            "ml_reducer": None,
        },
        features,
    )
    np.testing.assert_allclose(transformed, expected)


def test_lda_tuning_preprocessing_restores_persisted_state():
    args = Namespace(
        encoder="bow",
        reduction="lda",
        lda_topics=2,
        lda_dense_memory_limit_gib=4.0,
        ml_scaler="none",
        ml_reduction="none",
        seed=17,
        _artifact_info={"classic": _upstream()},
    )
    trainer = ClassicalMLTrainer(args)
    original_reducer = object()
    original_contract = {"sentinel": True}
    trainer._task_reducer = original_reducer
    trainer._lda_contract = original_contract
    original_info = args._artifact_info

    train, validation = trainer._preprocess_tuning_split(
        _matrix(),
        np.asarray([0, 1, 0, 1, 0, 1]),
        _matrix()[:2],
    )

    assert train.shape == (6, 2)
    assert validation.shape == (2, 2)
    assert trainer._task_reducer is original_reducer
    assert trainer._lda_contract is original_contract
    assert args._artifact_info is original_info
