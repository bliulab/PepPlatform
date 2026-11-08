from argparse import Namespace
from copy import deepcopy
from types import SimpleNamespace
import hashlib
import json
import os
import sys

import pytest
import numpy as np
import torch
import torch.nn as nn

from pepplatform.evaluators import EVALUATION_PROTOCOL_VERSION, Evaluator, OperatingPoint
from pepplatform.glove import GLOVE_ALGORITHM, GLOVE_IMPLEMENTATION, glove_cache_signature
from pepplatform.static_embeddings import word2vec_cache_signature
from pepplatform.workflow import (
    PAIR_TRAINING_ENCODER_OPTIONS,
    TRAINING_ENCODER_OPTIONS,
)
from scripts import train_pair_pipeline, train_pipeline, train_sequence_pipeline
from tools import build_pssm_cache, train_static_embeddings


def _argument_choices(parser, destination):
    action = next(action for action in parser._actions if action.dest == destination)
    return list(action.choices or [])


def test_training_entrypoint_encoder_catalogs_match_shared_workflow():
    sequence_parser = train_sequence_pipeline.build_argument_parser()
    pair_parser = train_pair_pipeline.build_argument_parser()

    assert _argument_choices(sequence_parser, "encoder") == TRAINING_ENCODER_OPTIONS
    assert _argument_choices(pair_parser, "encoder") == PAIR_TRAINING_ENCODER_OPTIONS
    assert _argument_choices(pair_parser, "aux_encoder") == PAIR_TRAINING_ENCODER_OPTIONS

    for encoder in TRAINING_ENCODER_OPTIONS:
        parsed = sequence_parser.parse_args(["--encoder", encoder])
        assert parsed.encoder == encoder


def test_training_entrypoint_encoder_aliases_are_normalized_before_choices():
    sequence_parser = train_sequence_pipeline.build_argument_parser()

    assert sequence_parser.parse_args(["--encoder", "TF-IDF"]).encoder == "tfidf"
    with pytest.raises(SystemExit):
        sequence_parser.parse_args(["--encoder", "physchem"])


@pytest.mark.parametrize(
    "parser_builder",
    [
        train_sequence_pipeline.build_argument_parser,
        train_pair_pipeline.build_argument_parser,
    ],
)
def test_training_entrypoints_reject_removed_labelled_lda(parser_builder):
    with pytest.raises(SystemExit):
        parser_builder().parse_args(["--reduction", "labelled_lda"])


@pytest.mark.parametrize("reduction", ["lda", "plsa"])
def test_pair_training_entrypoint_advertises_strict_topic_transforms(reduction):
    parser = train_pair_pipeline.build_argument_parser()
    assert parser.parse_args(["--reduction", reduction]).reduction == reduction


@pytest.mark.parametrize(
    ("reduction", "topics_key", "memory_key"),
    [
        ("lda", "lda_topics", "lda_dense_memory_limit_gib"),
        ("plsa", "plsa_topics", "plsa_memory_limit_gib"),
    ],
)
def test_pair_training_entrypoint_exposes_topic_controls(
    reduction,
    topics_key,
    memory_key,
):
    parsed = train_pair_pipeline.build_argument_parser().parse_args(
        [
            "--reduction",
            reduction,
            f"--{topics_key}",
            "17",
            f"--{memory_key}",
            "2.5",
        ]
    )

    assert getattr(parsed, topics_key) == 17
    assert getattr(parsed, memory_key) == 2.5


def test_pair_training_entrypoint_keeps_lsa_unavailable():
    with pytest.raises(SystemExit):
        train_pair_pipeline.build_argument_parser().parse_args(["--reduction", "lsa"])


def test_pair_training_entrypoint_accepts_generic_tsvd_reduction():
    parsed = train_pair_pipeline.build_argument_parser().parse_args(
        [
            "--ml_reduction",
            "tsvd",
            "--ml_reduction_dim",
            "64",
            "--tsvd_dense_memory_limit_gib",
            "2.5",
        ]
    )
    assert parsed.ml_reduction == "tsvd"
    assert parsed.ml_reduction_dim == 64
    assert parsed.tsvd_dense_memory_limit_gib == 2.5


@pytest.mark.parametrize(
    "parser_builder",
    [
        train_sequence_pipeline.build_argument_parser,
        train_pair_pipeline.build_argument_parser,
    ],
)
def test_training_entrypoints_expose_contracted_pca_and_kpca_controls(parser_builder):
    parser = parser_builder()
    parsed = parser.parse_args(
        [
            "--pca_dense_memory_limit_gib",
            "2.5",
            "--kpca_memory_limit_gib",
            "3.5",
            "--kpca_kernel",
            "poly",
            "--kpca_gamma",
            "0.2",
        ]
    )

    assert parsed.pca_dense_memory_limit_gib == 2.5
    assert parsed.kpca_memory_limit_gib == 3.5
    assert parsed.kpca_kernel == "poly"
    assert parsed.kpca_gamma == 0.2
    assert "sigmoid" not in _argument_choices(parser, "kpca_kernel")


def test_sequence_training_entrypoint_rejects_removed_pca_dim():
    with pytest.raises(SystemExit) as exc_info:
        train_sequence_pipeline.build_argument_parser().parse_args(["--pca_dim", "32"])

    assert exc_info.value.code == 2


def test_sequence_training_entrypoint_exposes_lsa_memory_guard():
    parsed = train_sequence_pipeline.build_argument_parser().parse_args(
        ["--reduction", "lsa", "--lsa_dense_memory_limit_gib", "2.5"]
    )
    assert parsed.lsa_dense_memory_limit_gib == 2.5


def test_sequence_training_entrypoint_exposes_lda_memory_guard():
    parsed = train_sequence_pipeline.build_argument_parser().parse_args(
        ["--reduction", "lda", "--lda_dense_memory_limit_gib", "2.5"]
    )
    assert parsed.lda_dense_memory_limit_gib == 2.5


def test_sequence_training_entrypoint_exposes_plsa_memory_guard():
    parsed = train_sequence_pipeline.build_argument_parser().parse_args(
        ["--reduction", "plsa", "--plsa_memory_limit_gib", "2.5"]
    )
    assert parsed.plsa_memory_limit_gib == 2.5


@pytest.mark.parametrize(
    "parser_builder",
    [
        train_sequence_pipeline.build_argument_parser,
        train_pair_pipeline.build_argument_parser,
    ],
)
@pytest.mark.parametrize(("raw", "expected"), [("1", 1), ("100000", 100000)])
def test_training_entrypoints_parse_covariance_max_raw_length(
    parser_builder,
    raw,
    expected,
):
    parsed = parser_builder().parse_args(["--covariance_max_raw_length", raw])

    assert parsed.covariance_max_raw_length == expected


@pytest.mark.parametrize(
    "parser_builder",
    [
        train_sequence_pipeline.build_argument_parser,
        train_pair_pipeline.build_argument_parser,
    ],
)
@pytest.mark.parametrize("raw", ["0", "100001", "not-an-integer"])
def test_training_entrypoints_reject_invalid_covariance_max_raw_length(
    parser_builder,
    raw,
):
    with pytest.raises(SystemExit) as exc_info:
        parser_builder().parse_args(["--covariance_max_raw_length", raw])

    assert exc_info.value.code == 2


def test_dispatcher_routes_sequence_task(monkeypatch):
    calls = {}

    def fake_run_target(module_name, script_name, argv):
        calls["module_name"] = module_name
        calls["script_name"] = script_name
        calls["argv"] = argv
        return "ok"

    monkeypatch.setattr(train_pipeline, "_run_target", fake_run_target)

    result = train_pipeline.main(["--task_type", "pep_function", "--epochs", "1"])

    assert result == "ok"
    assert calls == {
        "module_name": "sequence",
        "script_name": "train_sequence_pipeline.py",
        "argv": ["--task_type", "pep_function", "--epochs", "1"],
    }


def test_dispatcher_routes_pair_task(monkeypatch):
    calls = {}

    def fake_run_target(module_name, script_name, argv):
        calls["module_name"] = module_name
        calls["script_name"] = script_name
        calls["argv"] = argv
        return "ok"

    monkeypatch.setattr(train_pipeline, "_run_target", fake_run_target)

    result = train_pipeline.main(["--task_type", "pep_hlaii", "--epochs", "1"])

    assert result == "ok"
    assert calls == {
        "module_name": "pair",
        "script_name": "train_pair_pipeline.py",
        "argv": ["--task_type", "pep_hlaii", "--epochs", "1"],
    }


def test_train_pipeline_exports_sequence_pipeline_alias():
    assert train_pipeline.Pipeline is train_sequence_pipeline.Pipeline
    assert train_pipeline.SequencePipeline is train_sequence_pipeline.SequencePipeline


def test_static_embedding_kmer_sentences_enforce_supported_window():
    assert train_static_embeddings.build_kmer_sentences(["ACDE"], 2) == [["AC", "CD", "DE"]]
    with pytest.raises(ValueError, match=r"integer in \[1, 3\]"):
        train_static_embeddings.build_kmer_sentences(["ACDE"], 4)


def test_sequence_pipeline_rejects_pair_tasks():
    with pytest.raises(SystemExit, match="pair-interaction task"):
        train_sequence_pipeline.SequencePipeline(Namespace(task_type="pep_hlai"))


@pytest.mark.parametrize(
    ("entrypoint", "argv"),
    [
        (train_sequence_pipeline.main, ["train_sequence_pipeline.py", "--tokenizer", "mismatch"]),
        (train_pair_pipeline.main, ["train_pair_pipeline.py", "--tokenizer", "mismatch"]),
        (train_sequence_pipeline.main, ["train_sequence_pipeline.py", "--tokenizer", "multi-kmer"]),
        (train_pair_pipeline.main, ["train_pair_pipeline.py", "--tokenizer", "multi-kmer"]),
        (train_sequence_pipeline.main, ["train_sequence_pipeline.py", "--tokenizer", "reduced-aa"]),
        (train_pair_pipeline.main, ["train_pair_pipeline.py", "--tokenizer", "reduced-aa"]),
        (train_sequence_pipeline.main, ["train_sequence_pipeline.py", "--mismatch_m", "1"]),
        (train_pair_pipeline.main, ["train_pair_pipeline.py", "--mismatch_m", "1"]),
    ],
)
def test_training_clis_reject_removed_tokenizer_options(entrypoint, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        entrypoint()

    assert exc_info.value.code == 2


def test_pair_pipeline_defaults_hlai_to_original_split(tmp_path):
    task_root = str(tmp_path / "datasets" / "task_peptide_HLAI_interaction")

    assert train_pair_pipeline.default_pair_csv_candidates(task_root, "pep_hlai", "train") == [
        os.path.join(task_root, "data_bigmhc_el_100k_original_split", "train", "pairs.csv")
    ]
    assert train_pair_pipeline.default_pair_csv_candidates(task_root, "pep_hlai", "val") == [
        os.path.join(task_root, "data_bigmhc_el_100k_original_split", "val", "pairs.csv")
    ]
    assert train_pair_pipeline.default_pair_csv_candidates(task_root, "pep_hlai", "test") == [
        os.path.join(task_root, "data_bigmhc_el_100k_original_split", "test", "pairs.csv")
    ]
    assert train_pair_pipeline.default_pair_feature_dir(task_root, "pep_hlai") == os.path.join(
        task_root,
        "data_bigmhc_el_100k_original_split",
        "features",
    )


def test_pair_feature_cache_requires_matching_metadata(tmp_path):
    cache_path = tmp_path / "train_peptide.pt"
    old_tensor = torch.ones((2, 3), dtype=torch.float)
    new_tensor = torch.zeros((2, 3), dtype=torch.float)
    torch.save(old_tensor, cache_path)

    meta = {"cache_version": 2, "row_count": 2, "input_sha256": "expected"}
    assert (
        train_pair_pipeline._load_feature_cache(
            str(cache_path),
            meta,
            lambda item: isinstance(item, torch.Tensor) and item.shape == old_tensor.shape,
        )
        is None
    )

    train_pair_pipeline._save_feature_cache(str(cache_path), new_tensor, meta)
    loaded = train_pair_pipeline._load_feature_cache(
        str(cache_path),
        meta,
        lambda item: isinstance(item, torch.Tensor) and item.shape == new_tensor.shape,
    )
    assert torch.equal(loaded, new_tensor)


def _pretrained_pooled_cache_meta(**overrides):
    fingerprint = "tree:" + "a" * 64
    values = {
        "cache_version": 5,
        "feature_fit_scope": "train_only",
        "split": "train",
        "role": "peptide",
        "tokenizer": "esm2",
        "encoder": "esm2",
        "max_length": 48,
        "row_count": 2,
        "input_sha256": "input-digest",
        "train_sha256": "train-digest",
        "model_source": "/models/esm2-a",
        "model_fingerprint": fingerprint,
        "model_identity": {
            "contract_version": 1,
            "hidden_size": 3,
            "model_fingerprint": fingerprint,
            "model_id": "facebook/esm2_t30_150M_UR50D",
            "pooling": "residue_mean",
            "representation_contract_version": 1,
            "requested_source": "/models/esm2-a",
            "resolved_source": "/models/esm2-a",
            "runtime_package": "transformers",
            "runtime_version": "4.48.1",
            "weight_path": "/models/esm2-a/model.safetensors",
            "weight_sha256": "weight-digest",
        },
        "output_kind": "sequence_embedding",
        "pooling": "residue_mean",
        "representation_contract_version": 1,
        "runtime_environment": {
            "device_name": "GPU A",
            "torch_version": "2.5.1",
        },
    }
    values.update(overrides)
    return values


def test_pretrained_pooled_cache_exact_metadata_hits(tmp_path):
    cache_path = tmp_path / "train_peptide.pt"
    tensor = torch.ones((2, 3), dtype=torch.float32)
    metadata = _pretrained_pooled_cache_meta()
    train_pair_pipeline._save_feature_cache(str(cache_path), tensor, metadata)

    loaded = train_pair_pipeline._load_feature_cache(
        str(cache_path),
        metadata,
        lambda value: train_pair_pipeline._valid_pretrained_pooled_cache_tensor(
            value,
            row_count=2,
            output_dim=3,
        ),
    )

    assert torch.equal(loaded, tensor)


def test_pretrained_pooled_cache_same_fingerprint_allows_path_and_runtime_changes(
    tmp_path,
):
    cache_path = tmp_path / "train_peptide.pt"
    tensor = torch.ones((2, 3), dtype=torch.float32)
    actual = _pretrained_pooled_cache_meta()
    expected = deepcopy(actual)
    expected["model_source"] = "/different/esm2"
    expected["runtime_environment"] = {
        "device_name": "GPU B",
        "torch_version": "2.6.0",
    }
    expected["model_identity"].update(
        {
            "requested_source": "/different/esm2",
            "resolved_source": "/different/esm2",
            "runtime_package": "transformers-next",
            "runtime_version": "5.0.0",
            "weight_path": "/different/esm2/model.safetensors",
        }
    )
    train_pair_pipeline._save_feature_cache(str(cache_path), tensor, actual)

    loaded = train_pair_pipeline._load_feature_cache(
        str(cache_path),
        expected,
        lambda value: train_pair_pipeline._valid_pretrained_pooled_cache_tensor(
            value,
            row_count=2,
            output_dim=3,
        ),
    )

    assert torch.equal(loaded, tensor)


def _prott5_v3_cache_meta():
    return {
        "cache_version": 3,
        "encoder": "prott5",
        "feature_fit_scope": "train_only",
        "input_sha256": "input-digest",
        "max_length": 48,
        "model_fingerprint": "tree:" + "b" * 64,
        "model_source": "/legacy/prott5",
        "pooling": "attention_mask_mean",
        "role": "peptide",
        "row_count": 2,
        "split": "train",
        "tokenizer": "prott5",
        "train_sha256": "train-digest",
    }


def _prott5_v5_cache_meta():
    fingerprint = "tree:" + "b" * 64
    return {
        "cache_version": 5,
        "encoder": "prott5",
        "feature_fit_scope": "train_only",
        "input_normalization": "strip_whitespace_uppercase_UZOB_to_X",
        "input_sha256": "input-digest",
        "max_length": 48,
        "model_fingerprint": fingerprint,
        "model_identity": {
            "d_model": 1024,
            "model_fingerprint": fingerprint,
            "model_id": "Rostlab/prot_t5_xl_uniref50",
            "pooling": "residue_mean",
            "representation_contract_version": 1,
            "resolved_source": "/current/prott5",
            "runtime_package": "transformers",
            "runtime_version": "4.48.1",
            "weight_path": "/current/prott5/model.safetensors.index.json",
        },
        "model_source": "/current/prott5",
        "output_kind": "sequence_embedding",
        "pooling": "residue_mean",
        "representation_contract_version": 1,
        "role": "peptide",
        "row_count": 2,
        "runtime_environment": {"device_type": "cuda"},
        "special_tokens": "excluded",
        "split": "train",
        "tokenizer": "prott5",
        "train_sha256": "train-digest",
    }


def test_controlled_prott5_v3_pooled_cache_hits_current_v5_identity(tmp_path):
    cache_path = tmp_path / "train_peptide.pt"
    tensor = torch.ones((2, 1024), dtype=torch.float32)
    train_pair_pipeline._save_feature_cache(
        str(cache_path),
        tensor,
        _prott5_v3_cache_meta(),
    )

    loaded = train_pair_pipeline._load_feature_cache(
        str(cache_path),
        _prott5_v5_cache_meta(),
        lambda value: train_pair_pipeline._valid_pretrained_pooled_cache_tensor(
            value,
            row_count=2,
            output_dim=1024,
        ),
    )

    assert torch.equal(loaded, tensor)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_fingerprint", "tree:" + "c" * 64),
        ("input_sha256", "different-input"),
        ("max_length", 47),
        ("role", "protein"),
        ("representation_contract_version", 2),
    ],
)
def test_pretrained_pooled_cache_rejects_scientific_identity_changes(
    tmp_path,
    field,
    value,
):
    cache_path = tmp_path / f"train_peptide_{field}.pt"
    tensor = torch.ones((2, 3), dtype=torch.float32)
    actual = _pretrained_pooled_cache_meta()
    expected = deepcopy(actual)
    expected[field] = value
    train_pair_pipeline._save_feature_cache(str(cache_path), tensor, actual)

    assert (
        train_pair_pipeline._load_feature_cache(
            str(cache_path),
            expected,
            lambda item: train_pair_pipeline._valid_pretrained_pooled_cache_tensor(
                item,
                row_count=2,
                output_dim=3,
            ),
        )
        is None
    )


@pytest.mark.parametrize(
    "tensor",
    [
        torch.ones((1, 3), dtype=torch.float32),
        torch.ones((2, 4), dtype=torch.float32),
        torch.tensor([[1.0, 2.0, float("nan")], [1.0, 2.0, 3.0]]),
    ],
)
def test_pretrained_pooled_cache_rejects_incompatible_tensor(tmp_path, tensor):
    cache_path = tmp_path / f"invalid_{tensor.shape[0]}_{tensor.shape[1]}.pt"
    metadata = _pretrained_pooled_cache_meta()
    train_pair_pipeline._save_feature_cache(str(cache_path), tensor, metadata)

    assert (
        train_pair_pipeline._load_feature_cache(
            str(cache_path),
            metadata,
            lambda item: train_pair_pipeline._valid_pretrained_pooled_cache_tensor(
                item,
                row_count=2,
                output_dim=3,
            ),
        )
        is None
    )


def _graph_args(**overrides):
    values = {
        "gnn_k": 2,
        "gnn_exact_knn_max_nodes": 100,
        "gnn_rp_projections": 2,
        "gnn_rp_window": 0,
        "seed": 42,
    }
    values.update(overrides)
    return Namespace(**values)


def test_inductive_query_edges_only_flow_from_train_reference_to_queries():
    reference = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    query = torch.tensor([[0.9, 0.1], [0.1, 0.9]])

    edges = train_pair_pipeline.build_query_reference_edge_index(reference, query, k=2, seed=42)

    assert edges.shape == (2, 4)
    assert torch.all(edges[0] < len(reference))
    assert torch.all(edges[1] >= len(reference))
    assert torch.all(edges[1] < len(reference) + len(query))


@pytest.mark.parametrize(
    ("model_factory", "graph_kind"),
    [
        (lambda: train_pair_pipeline.PairGCNClassifier(4, 8, 0.0), "gcn"),
        (lambda: train_pair_pipeline.PairGATClassifier(4, 8, 0.0), "gat"),
        (lambda: train_pair_pipeline.PairSAGEClassifier(4, 8, 0.0), "message"),
        (
            lambda: train_pair_pipeline.PairGraphTransformerClassifier(4, 8, 2, 0.0),
            "graph_transformer",
        ),
    ],
)
def test_inductive_query_prediction_is_invariant_to_other_query_nodes(model_factory, graph_kind):
    torch.manual_seed(42)
    args = _graph_args()
    reference = torch.tensor(
        [[1.0, 0.0, 0.2, 0.1], [0.0, 1.0, 0.1, 0.2], [0.7, 0.7, 0.3, 0.4]],
        dtype=torch.float,
    )
    first_query = torch.tensor([[0.8, 0.2, 0.1, 0.3]], dtype=torch.float)
    other_query = torch.tensor([[0.1, 0.9, 0.4, 0.2]], dtype=torch.float)
    model = model_factory().eval()

    single_structure = train_pair_pipeline.build_inductive_graph_structure(
        args, reference, first_query, "cpu", graph_kind
    )
    batch_query = torch.cat([first_query, other_query], dim=0)
    batch_structure = train_pair_pipeline.build_inductive_graph_structure(
        args, reference, batch_query, "cpu", graph_kind
    )
    single = train_pair_pipeline._inductive_graph_scores(
        model, reference, first_query, single_structure
    )[0]
    batched = train_pair_pipeline._inductive_graph_scores(
        model, reference, batch_query, batch_structure
    )[0]

    assert np.isclose(single, batched, rtol=1e-6, atol=1e-6)


def test_inductive_test_graph_is_built_only_after_validation_operating_point(monkeypatch):
    state = {"operating_point_frozen": False, "test_seen": False}
    original_fit = train_pair_pipeline.Evaluator.fit_operating_point

    def tracked_fit(*args, **kwargs):
        point = original_fit(*args, **kwargs)
        state["operating_point_frozen"] = True
        return point

    def tracked_structure(args, reference_x, query_x, device, graph_kind):
        if query_x is not None and float(query_x[0, 0]) == 9.0:
            assert state["operating_point_frozen"] is True
            state["test_seen"] = True
        return None

    class StructureAgnosticModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, features, structure):
            return self.linear(features)

    monkeypatch.setattr(train_pair_pipeline.Evaluator, "fit_operating_point", tracked_fit)
    monkeypatch.setattr(train_pair_pipeline, "build_inductive_graph_structure", tracked_structure)
    args = _graph_args(
        artifact_dir="",
        lr=0.01,
        weight_decay=0.0,
        epochs=1,
        early_stop_min_delta=0.0,
        early_stop_patience=0,
        report_aupr=False,
        task_type="pep_protein_interaction",
        tokenizer="kmer",
        encoder="index",
        model="gcn",
    )

    train_pair_pipeline._train_inductive_graph_model(
        args,
        StructureAgnosticModel(),
        torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        torch.tensor([[0.0], [1.0]]),
        torch.tensor([[0.2, 0.2], [0.8, 0.8]]),
        torch.tensor([[0.0], [1.0]]),
        torch.tensor([[9.0, 9.0], [10.0, 10.0]]),
        torch.tensor([[0.0], [1.0]]),
        device="cpu",
        graph_kind="message",
        graph_label="TEST",
    )

    assert state == {"operating_point_frozen": True, "test_seen": True}


def test_linear_svc_uses_configured_seed():
    from pepplatform.ml_trainers import ClassicalMLTrainer

    trainer = ClassicalMLTrainer(
        Namespace(
            seed=42,
            svm_class_weight="none",
            svm_kernel="linear",
            svm_c=1.0,
            svm_max_iter=100,
            svm_tol=1e-3,
        )
    )
    estimators = trainer._fit_svm_estimators(
        np.asarray([[0.0], [1.0], [2.0], [3.0]]),
        np.asarray([[0], [0], [1], [1]]),
    )

    assert estimators[0].random_state == 42


@pytest.mark.parametrize("multilabel", [False, True])
def test_classical_protocol_evaluation_preserves_score_precision(tmp_path, multilabel):
    from pepplatform.ml_trainers import ClassicalMLTrainer

    low = np.nextafter(0.5, 0.0)
    high = np.nextafter(0.5, 1.0)
    if multilabel:
        labels = np.asarray([[0, 1], [1, 0]], dtype=np.int32)
        scores = np.asarray([[low, high], [high, low]], dtype=np.float64)
        label_names = ["a", "b"]
        task_type = "pep_function"
    else:
        labels = np.asarray([[0], [1]], dtype=np.int32)
        scores = np.asarray([[low], [high]], dtype=np.float64)
        label_names = None
        task_type = "pep_protein_interaction"

    point = Evaluator.fit_operating_point(
        labels,
        scores,
        label_names=label_names,
    )
    original_f1 = Evaluator.score_metrics(labels, scores, point)["f1"]
    rounded_f1 = Evaluator.score_metrics(
        labels,
        scores.astype(np.float32),
        point,
    )["f1"]
    assert rounded_f1 != original_f1

    trainer = ClassicalMLTrainer(Namespace(artifact_dir=str(tmp_path), task_type=task_type))
    trainer._save_protocol_evaluation(
        labels,
        scores,
        point,
        labels,
        scores,
    )

    evaluation = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))
    with np.load(tmp_path / "evaluation_scores.npz", allow_pickle=False) as arrays:
        assert arrays["validation_labels"].dtype == np.int32
        assert arrays["validation_scores"].dtype == np.float64
        assert arrays["test_labels"].dtype == np.int32
        assert arrays["test_scores"].dtype == np.float64
        assert np.array_equal(arrays["validation_scores"], scores)
        assert np.array_equal(arrays["test_scores"], scores)
        expected_validation = Evaluator.score_metrics(
            arrays["validation_labels"],
            arrays["validation_scores"],
            point,
        )
        expected_test = Evaluator.score_metrics(
            arrays["test_labels"],
            arrays["test_scores"],
            point,
        )

    assert evaluation["validation"] == expected_validation
    assert evaluation["test"] == expected_test


def test_pair_subword_tokenizer_corpus_is_train_only(tmp_path, monkeypatch):
    captured = {}

    class FakeTrainer:
        def __init__(self, vocab_size):
            captured["vocab_size"] = vocab_size
            self.last_cache_reused = False

        def ensure_trained(self, sequences, output_path, *, sequence_role):
            captured["sequences"] = list(sequences)
            captured["sequence_role"] = sequence_role
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            open(output_path, "w", encoding="utf-8").write("{}")

    monkeypatch.setattr(train_pair_pipeline, "BPETokenizerTrainer", FakeTrainer)
    args = Namespace(tokenizer="bpe", bpe_path="", bpe_vocab_size=32)
    train_rows = [("train", "TRAINPEP", "TRAINPRO", 1)]
    val_rows = [("val", "VALPEP", "VALPRO", 0)]

    train_pair_pipeline.ensure_subword_tokenizer_ready(args, str(tmp_path), train_rows, val_rows)

    assert captured["sequences"] == ["TRAINPEP", "TRAINPRO"]
    assert captured["sequence_role"] == "both"


def test_sequence_bpe_tokenizer_corpus_is_train_only(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    for split, sequence in [("train", "TRAINSEQ"), ("val", "VALSEQ")]:
        split_dir = data_dir / split
        split_dir.mkdir(parents=True)
        (split_dir / "seqs.fasta").write_text(f">{split}\n{sequence}\n", encoding="utf-8")
    captured = {}

    class FakeTrainer:
        def __init__(self, vocab_size):
            self.last_cache_reused = False

        def ensure_trained(self, sequences, output_path, *, sequence_role):
            captured["sequences"] = list(sequences)
            captured["sequence_role"] = sequence_role
            open(output_path, "w", encoding="utf-8").write("{}")

    monkeypatch.setattr(train_sequence_pipeline, "BPETokenizerTrainer", FakeTrainer)
    pipeline = train_sequence_pipeline.SequencePipeline.__new__(
        train_sequence_pipeline.SequencePipeline
    )
    pipeline.args = Namespace(tokenizer="bpe", data_dir=str(data_dir), bpe_vocab_size=32)
    pipeline.bpe_path = str(tmp_path / "bpe.json")
    pipeline.feat_factory = SimpleNamespace(bpe_path="")
    pipeline.load_sequences = train_sequence_pipeline.SequencePipeline.load_sequences.__get__(
        pipeline
    )

    pipeline.ensure_bpe_tokenizer_ready()

    assert captured["sequences"] == ["TRAINSEQ"]
    assert captured["sequence_role"] == "sequence"


def test_pair_wordpiece_tokenizer_corpus_is_train_only(
    tmp_path,
    monkeypatch,
):
    captured = {}

    class FakeTrainer:
        def __init__(self, vocab_size, max_raw_length):
            captured["vocab_size"] = vocab_size
            captured["max_raw_length"] = max_raw_length
            self.last_cache_reused = False

        def ensure_trained(
            self,
            sequences,
            output_path,
            *,
            sequence_role,
        ):
            captured["sequences"] = list(sequences)
            captured["sequence_role"] = sequence_role
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            open(output_path, "w", encoding="utf-8").write("{}")

    monkeypatch.setattr(
        train_pair_pipeline,
        "WordPieceTokenizerTrainer",
        FakeTrainer,
    )
    args = Namespace(
        tokenizer="wordpiece",
        wordpiece_path="",
        wordpiece_vocab_size=64,
        wordpiece_max_raw_length=50_000,
    )
    train_rows = [("train", "TRAINPEP", "TRAINPRO", 1)]
    val_rows = [("val", "VALPEP", "VALPRO", 0)]

    train_pair_pipeline.ensure_subword_tokenizer_ready(
        args,
        str(tmp_path),
        train_rows,
        val_rows,
    )

    assert captured["sequences"] == ["TRAINPEP", "TRAINPRO"]
    assert captured["sequence_role"] == "both"
    assert captured["max_raw_length"] == 50_000


def test_sequence_wordpiece_tokenizer_corpus_is_train_only(
    tmp_path,
    monkeypatch,
):
    data_dir = tmp_path / "data"
    for split, sequence in [("train", "TRAINSEQ"), ("val", "VALSEQ")]:
        split_dir = data_dir / split
        split_dir.mkdir(parents=True)
        (split_dir / "seqs.fasta").write_text(
            f">{split}\n{sequence}\n",
            encoding="utf-8",
        )
    captured = {}

    class FakeTrainer:
        def __init__(self, vocab_size, max_raw_length):
            self.last_cache_reused = False
            self.max_raw_length = max_raw_length

        def ensure_trained(
            self,
            sequences,
            output_path,
            *,
            sequence_role,
        ):
            captured["sequences"] = list(sequences)
            captured["sequence_role"] = sequence_role
            open(output_path, "w", encoding="utf-8").write("{}")

    monkeypatch.setattr(
        train_sequence_pipeline,
        "WordPieceTokenizerTrainer",
        FakeTrainer,
    )
    pipeline = train_sequence_pipeline.SequencePipeline.__new__(
        train_sequence_pipeline.SequencePipeline
    )
    pipeline.args = Namespace(
        tokenizer="wordpiece",
        data_dir=str(data_dir),
        wordpiece_vocab_size=64,
        wordpiece_max_raw_length=50_000,
    )
    pipeline.wordpiece_path = str(tmp_path / "wordpiece.json")
    pipeline.feat_factory = SimpleNamespace(
        wordpiece_path="",
        wordpiece_max_raw_length=0,
    )
    pipeline.load_sequences = train_sequence_pipeline.SequencePipeline.load_sequences.__get__(
        pipeline
    )

    pipeline.ensure_wordpiece_tokenizer_ready()

    assert captured["sequences"] == ["TRAINSEQ"]
    assert captured["sequence_role"] == "sequence"
    assert pipeline.feat_factory.wordpiece_max_raw_length == 50_000


def test_pssm_subject_database_contains_train_sequences_only(tmp_path):
    train_fasta = tmp_path / "train.fasta"
    subject_fasta = tmp_path / "subject.fasta"
    train_fasta.write_text(">train\nACDE\n", encoding="utf-8")

    build_pssm_cache.build_subject_fasta(str(train_fasta), str(subject_fasta))

    assert subject_fasta.read_text(encoding="utf-8") == ">TRAIN_000000\nACDE\n"


def test_static_embedding_training_reads_train_split_only(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    for split, sequence in [("train", "ACAC"), ("val", "ZZZZ")]:
        split_dir = data_dir / split
        split_dir.mkdir(parents=True)
        (split_dir / "seqs.fasta").write_text(f">{split}\n{sequence}\n", encoding="utf-8")
    save_root = tmp_path / "embeddings"
    monkeypatch.setattr(
        train_static_embeddings.sys,
        "argv",
        [
            "train_static_embeddings.py",
            "--data_dir",
            str(data_dir),
            "--save_root",
            str(save_root),
            "--tokenizer_mode",
            "kmer",
            "--k",
            "1",
            "--vector_size",
            "2",
            "--epochs",
            "5",
            "--train_glove",
            "--seed",
            "42",
        ],
    )

    train_static_embeddings.main()

    vectors = (save_root / "glove-kmer-1" / "glove.vec").read_text(encoding="utf-8")
    assert "A " in vectors and "C " in vectors
    assert "Z " not in vectors
    metadata = json.loads((save_root / "glove-kmer-1" / "meta.json").read_text(encoding="utf-8"))
    assert metadata["workers"] == 1
    assert metadata["algorithm"] == GLOVE_ALGORITHM
    assert metadata["implementation"] == GLOVE_IMPLEMENTATION
    assert metadata["vector_combination"] == "word_plus_context"
    assert metadata["final_loss"] < metadata["initial_loss"]
    assert all(len(line.split()) == 3 for line in vectors.strip().splitlines())


def test_pair_glove_explicit_path_is_not_overwritten_or_retrained(tmp_path, monkeypatch):
    explicit_path = tmp_path / "external.vec"
    explicit_path.write_text("AA 0.1 0.2\n", encoding="utf-8")
    args = Namespace(
        encoder="glove",
        tokenizer="kmer",
        k=2,
        dr_max_distance=5,
        glove_path=str(explicit_path),
    )

    def fail_run(*_args, **_kwargs):
        raise AssertionError("explicit GloVe vectors must bypass auto-training")

    monkeypatch.setattr(train_pair_pipeline.subprocess, "run", fail_run)
    train_pair_pipeline.ensure_ppi_static_embedding_ready(
        args,
        str(tmp_path / "task"),
        [("row_1", "ACD", "EFG", 1)],
        [],
    )

    assert args.glove_path == str(explicit_path)
    assert args._static_embedding_provenance["source"] == "external"
    assert args._static_embedding_provenance["algorithm_verified"] is False


def test_sequence_glove_explicit_path_is_not_overwritten_or_retrained(tmp_path, monkeypatch):
    explicit_path = tmp_path / "external.vec"
    explicit_path.write_text("AA 0.1 0.2\n", encoding="utf-8")
    pipeline = train_sequence_pipeline.SequencePipeline.__new__(
        train_sequence_pipeline.SequencePipeline
    )
    pipeline.args = Namespace(
        encoder="glove",
        tokenizer="kmer",
        k=2,
        dr_max_distance=5,
        glove_path=str(explicit_path),
    )
    pipeline.project_root = str(tmp_path)
    pipeline.task_root_rel = "task"
    pipeline._explicit_glove_path = str(explicit_path)
    pipeline.glove_path = str(explicit_path)
    pipeline.feat_factory = SimpleNamespace(glove_path="")

    def fail_run(*_args, **_kwargs):
        raise AssertionError("explicit GloVe vectors must bypass auto-training")

    monkeypatch.setattr(train_sequence_pipeline.subprocess, "run", fail_run)
    pipeline.ensure_static_embedding_ready()

    assert pipeline.glove_path == str(explicit_path)
    assert pipeline.feat_factory.glove_path == str(explicit_path)
    assert pipeline.args._static_embedding_provenance["source"] == "external"


@pytest.mark.parametrize(
    ("encoder", "suffix"),
    [("word2vec", ".vec"), ("fasttext", ".bin")],
)
def test_pair_predictive_embedding_explicit_path_bypasses_training(
    tmp_path,
    monkeypatch,
    encoder,
    suffix,
):
    explicit_path = tmp_path / f"external{suffix}"
    explicit_path.write_bytes(b"external-artifact")
    args = Namespace(
        encoder=encoder,
        tokenizer="kmer",
        k=2,
        dr_max_distance=5,
        **{f"{encoder}_path": str(explicit_path)},
    )

    monkeypatch.setattr(
        train_pair_pipeline.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "explicit static embeddings must bypass auto-training"
        ),
    )
    train_pair_pipeline.ensure_ppi_static_embedding_ready(
        args,
        str(tmp_path / "task"),
        [("row_1", "ACD", "EFG", 1)],
        [],
    )

    assert getattr(args, f"{encoder}_path") == str(explicit_path)
    assert args._static_embedding_provenance["verification"] == "unverified"


@pytest.mark.parametrize(
    ("encoder", "suffix"),
    [("word2vec", ".vec"), ("fasttext", ".bin")],
)
def test_sequence_predictive_embedding_explicit_path_bypasses_training(
    tmp_path,
    monkeypatch,
    encoder,
    suffix,
):
    explicit_path = tmp_path / f"external{suffix}"
    explicit_path.write_bytes(b"external-artifact")
    pipeline = train_sequence_pipeline.SequencePipeline.__new__(
        train_sequence_pipeline.SequencePipeline
    )
    pipeline.args = Namespace(
        encoder=encoder,
        tokenizer="kmer",
        k=2,
        dr_max_distance=5,
        **{f"{encoder}_path": str(explicit_path)},
    )
    pipeline.project_root = str(tmp_path)
    pipeline.task_root_rel = "task"
    pipeline._explicit_static_embedding_paths = {encoder: str(explicit_path)}
    setattr(pipeline, f"{encoder}_path", str(explicit_path))
    pipeline.feat_factory = SimpleNamespace(**{f"{encoder}_path": ""})

    monkeypatch.setattr(
        train_sequence_pipeline.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "explicit static embeddings must bypass auto-training"
        ),
    )
    pipeline.ensure_static_embedding_ready()

    assert getattr(pipeline, f"{encoder}_path") == str(explicit_path)
    assert getattr(pipeline.feat_factory, f"{encoder}_path") == str(explicit_path)
    assert pipeline.args._static_embedding_provenance["verification"] == ("unverified")


@pytest.mark.parametrize("suffix", [".vec", ".txt"])
def test_pair_fasttext_rejects_incomplete_explicit_artifact(
    tmp_path,
    suffix,
):
    explicit_path = tmp_path / f"external{suffix}"
    explicit_path.write_text("1 2\nAA 0.1 0.2\n", encoding="utf-8")
    args = Namespace(
        encoder="fasttext",
        tokenizer="kmer",
        k=2,
        dr_max_distance=5,
        fasttext_path=str(explicit_path),
    )

    with pytest.raises(SystemExit, match="omit subword state"):
        train_pair_pipeline.ensure_ppi_static_embedding_ready(
            args,
            str(tmp_path / "task"),
            [("row_1", "ACD", "EFG", 1)],
            [],
        )


def test_pair_protocol_metadata_uses_inductive_mode_for_sample_graphs():
    point = OperatingPoint(values=(0.4,)).to_dict()

    graph_metadata = train_pair_pipeline._pair_protocol_metadata(
        Namespace(predictor_type="sample_graph", _operating_point=point)
    )
    classifier_metadata = train_pair_pipeline._pair_protocol_metadata(
        Namespace(predictor_type="classifier", _operating_point=point)
    )

    assert graph_metadata["evaluation_mode"] == "inductive"
    assert classifier_metadata["evaluation_mode"] == "independent_iid"


def test_ppi_static_embedding_cache_requires_single_worker_and_matching_parameters(
    tmp_path, monkeypatch
):
    args = Namespace(
        encoder="word2vec",
        tokenizer="kmer",
        k=2,
        dr_max_distance=5,
        embedding_vector_size=8,
        embedding_window=3,
        embedding_min_count=1,
        embedding_epochs=2,
        seed=42,
    )
    rows = [("row_1", "ACD", "EFG", 1)]
    calls = []

    def fake_run(command, check, cwd):
        calls.append(command)
        out_dir = tmp_path / "word2vec-kmer-2"
        out_dir.mkdir(exist_ok=True)
        (out_dir / "word2vec.vec").write_text("AA 0.1\n", encoding="utf-8")
        train_fasta = tmp_path / "_ppi_static_embed_data" / "train" / "seqs.fasta"
        metadata = {
            "method": "word2vec",
            "tokenizer_mode": "kmer",
            "k": 2,
            "dr_max_distance": 5,
            "vector_size": 8,
            "window": args.embedding_window,
            "min_count": 1,
            "epochs": 2,
            "workers": 1,
            "seed": 42,
            "feature_fit_scope": "train_only",
            "train_sha256": hashlib.sha256(train_fasta.read_bytes()).hexdigest(),
            "artifact_file": "word2vec.vec",
            "artifact_sha256": hashlib.sha256((out_dir / "word2vec.vec").read_bytes()).hexdigest(),
            **word2vec_cache_signature(workers=1),
        }
        (out_dir / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")

    monkeypatch.setattr(train_pair_pipeline.subprocess, "run", fake_run)
    train_pair_pipeline.ensure_ppi_static_embedding_ready(args, str(tmp_path), rows, [])
    assert len(calls) == 1

    train_pair_pipeline.ensure_ppi_static_embedding_ready(args, str(tmp_path), rows, [])
    assert len(calls) == 1

    meta_path = tmp_path / "word2vec-kmer-2" / "meta.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["workers"] = 4
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    train_pair_pipeline.ensure_ppi_static_embedding_ready(args, str(tmp_path), rows, [])
    assert len(calls) == 2

    args.embedding_window = 7
    train_pair_pipeline.ensure_ppi_static_embedding_ready(args, str(tmp_path), rows, [])
    assert len(calls) == 3


def test_ppi_glove_cache_requires_standard_algorithm_signature(tmp_path, monkeypatch):
    args = Namespace(
        encoder="glove",
        tokenizer="kmer",
        k=2,
        dr_max_distance=5,
        glove_path="",
        embedding_vector_size=8,
        embedding_window=3,
        embedding_min_count=1,
        embedding_epochs=2,
        glove_x_max=100.0,
        glove_alpha=0.75,
        glove_learning_rate=0.05,
        glove_batch_size=16,
        seed=42,
    )
    rows = [("row_1", "ACD", "EFG", 1)]
    calls = []

    def fake_run(command, check, cwd):
        calls.append(command)
        out_dir = tmp_path / "glove-kmer-2"
        out_dir.mkdir(exist_ok=True)
        (out_dir / "glove.vec").write_text("AC 0.1 0.2\n", encoding="utf-8")
        train_fasta = tmp_path / "_ppi_static_embed_data" / "train" / "seqs.fasta"
        metadata = {
            "method": "glove",
            "tokenizer_mode": "kmer",
            "k": 2,
            "dr_max_distance": 5,
            "vector_size": 8,
            "window": 3,
            "min_count": 1,
            "epochs": 2,
            "workers": 1,
            "seed": 42,
            "feature_fit_scope": "train_only",
            "train_sha256": hashlib.sha256(train_fasta.read_bytes()).hexdigest(),
            "artifact_file": "glove.vec",
            "artifact_sha256": hashlib.sha256((out_dir / "glove.vec").read_bytes()).hexdigest(),
            **glove_cache_signature(
                x_max=args.glove_x_max,
                alpha=args.glove_alpha,
                learning_rate=args.glove_learning_rate,
                batch_size=args.glove_batch_size,
            ),
        }
        (out_dir / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")

    monkeypatch.setattr(train_pair_pipeline.subprocess, "run", fake_run)

    train_pair_pipeline.ensure_ppi_static_embedding_ready(args, str(tmp_path), rows, [])
    train_pair_pipeline.ensure_ppi_static_embedding_ready(args, str(tmp_path), rows, [])
    assert len(calls) == 1
    assert "--glove_x_max" in calls[0]

    args.glove_x_max = 10.0
    train_pair_pipeline.ensure_ppi_static_embedding_ready(args, str(tmp_path), rows, [])
    assert len(calls) == 2

    meta_path = tmp_path / "glove-kmer-2" / "meta.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata.pop("implementation_version")
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    train_pair_pipeline.ensure_ppi_static_embedding_ready(args, str(tmp_path), rows, [])
    assert len(calls) == 3
