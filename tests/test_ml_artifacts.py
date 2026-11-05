from argparse import Namespace
import hashlib
import json

import joblib
import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

from pepplatform.ml_trainers import ClassicalMLTrainer
from pepplatform.models import (
    FeatureGRU,
    FeatureLSTM,
    FeatureRNN,
    FeatureTextCNN,
    FeatureTransformer,
    IndexGRU,
    IndexLSTM,
    IndexRNN,
    IndexTextCNN,
    IndexTransformer,
    PretrainedResiduePredictor,
    ResidueGraphClassifier,
    ResidueModelInput,
)
from pepplatform.evaluators import EVALUATION_PROTOCOL_VERSION, OperatingPoint
from pepplatform.profile_generation import ProfileGenerationConfig
from pepplatform.sequence_profiles import (
    BLOSUM62_BACKGROUND_SHA256,
    BLOSUM62_MATRIX_SHA256,
    PROFILE_SCIENTIFIC_REFERENCES,
    PROFILE_HIT_SUMMARY_FIELDS,
    PROFILE_QUERY_COVERAGE_FORMULA,
    PROFILE_CONTRACT_VERSION,
)
from scripts.train_pipeline import Pipeline
from scripts.train_pair_pipeline import (
    _classic_extra_artifact_info,
    _write_masked_score_analysis,
    PairBiCrossAttentionClassifier,
    PairCrossAttentionClassifier,
    PairGATClassifier,
    PairGraphTransformerClassifier,
    PairHGTClassifier,
    configure_pair_graph_artifacts,
    configure_pair_vector_artifacts,
    save_pair_graph_nn_artifact,
    save_pair_sequence_nn_artifact,
    save_pair_token_nn_artifact,
    save_pair_vector_nn_artifact,
    train_cross_attention_model,
)


def _set_binary_operating_point(args, threshold):
    args._operating_point = OperatingPoint(values=(float(threshold),)).to_dict()


def _profile_provenance(
    *,
    iterations=2,
    evalue=0.01,
    subject_fasta_sha256="a" * 64,
):
    config = ProfileGenerationConfig(
        iterations=iterations,
        evalue=evalue,
        inclusion_evalue=0.001,
        threads=1,
    )
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
            "method_name": "pssm",
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
        },
        "generation_fingerprint": "c" * 64,
        "feature_type": "pssm",
        "fallback_policy": "error",
    }


def test_blosum62_artifact_marks_kmer_mean_as_derived():
    info = _classic_extra_artifact_info(Namespace(tokenizer="kmer", encoder="blosum62", k=2))
    contract = info["blosum62_contract"]
    assert contract["method_name"] == "BLOSUM62-derived k-mer mean"
    assert contract["canonical_blosum62_per_residue"] is False
    assert contract["kmer_aggregation"] == ("arithmetic_mean_of_residue_score_rows")


def test_memory_bounded_gat_matches_dense_edge_formula_and_gradients():
    torch.manual_seed(11)
    model = PairGATClassifier(
        input_dim=3,
        hidden_dim=4,
        dropout=0.0,
        edge_chunk_size=3,
    ).double()
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 0, 2, 1, 3, 0, 1],
            [0, 0, 0, 0, 1, 1, 2, 2, 3, 3],
        ],
        dtype=torch.long,
    )
    x = torch.randn(4, 4, dtype=torch.double, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    reference_lin_weight = model.lin1.weight.detach().clone().requires_grad_(True)
    reference_lin_bias = model.lin1.bias.detach().clone().requires_grad_(True)
    reference_src_weight = model.a_src1.weight.detach().clone().requires_grad_(True)
    reference_dst_weight = model.a_dst1.weight.detach().clone().requires_grad_(True)

    actual = model._gat_layer(
        x,
        edge_index,
        model.lin1,
        model.a_src1,
        model.a_dst1,
    )
    reference_h = torch.nn.functional.linear(
        reference_x,
        reference_lin_weight,
        reference_lin_bias,
    )
    src, dst = edge_index
    logits = torch.nn.functional.leaky_relu(
        torch.nn.functional.linear(reference_h[src], reference_src_weight)
        + torch.nn.functional.linear(reference_h[dst], reference_dst_weight),
        negative_slope=0.2,
    ).squeeze(-1)
    exp_logits = logits.exp()
    denom = torch.zeros(reference_h.shape[0], dtype=reference_h.dtype)
    denom.index_add_(0, dst, exp_logits)
    alpha = exp_logits / denom[dst].clamp(min=1e-9)
    expected = torch.zeros_like(reference_h)
    expected.index_add_(0, dst, alpha.unsqueeze(1) * reference_h[src])

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(
        model.lin1.weight.grad,
        reference_lin_weight.grad,
        rtol=1e-9,
        atol=1e-11,
    )
    torch.testing.assert_close(
        model.lin1.bias.grad,
        reference_lin_bias.grad,
        rtol=1e-9,
        atol=1e-11,
    )
    torch.testing.assert_close(
        model.a_src1.weight.grad,
        reference_src_weight.grad,
        rtol=1e-9,
        atol=1e-11,
    )
    torch.testing.assert_close(
        model.a_dst1.weight.grad,
        reference_dst_weight.grad,
        rtol=1e-9,
        atol=1e-11,
    )


@pytest.mark.parametrize(
    "model_cls", [PairCrossAttentionClassifier, PairBiCrossAttentionClassifier]
)
def test_homogeneous_pair_attention_separate_streams_match_legacy_input(model_cls):
    torch.manual_seed(7)
    legacy_model = model_cls(
        seq_len=2,
        encoder_name="ppi_fusion",
        in_dim=3,
        hidden_dim=4,
        dropout=0.0,
        n_heads=2,
    )
    stream_model = model_cls(
        seq_len=2,
        encoder_name="ppi_fusion",
        in_dim=3,
        hidden_dim=4,
        dropout=0.0,
        n_heads=2,
    )
    stream_model.load_state_dict(legacy_model.state_dict())
    peptide = torch.randn(3, 2, 3)
    partner = torch.randn(3, 2, 3)
    combined = torch.cat([peptide, partner], dim=1).requires_grad_(True)
    separate_peptide = peptide.clone().requires_grad_(True)
    separate_partner = partner.clone().requires_grad_(True)

    legacy_logits, legacy_embedding = legacy_model.forward_with_embedding(combined)
    stream_logits, stream_embedding = stream_model.forward_with_embedding(
        separate_peptide,
        separate_partner,
    )
    torch.testing.assert_close(stream_logits, legacy_logits)
    torch.testing.assert_close(stream_embedding, legacy_embedding)

    legacy_logits.sum().backward()
    stream_logits.sum().backward()
    torch.testing.assert_close(separate_peptide.grad, combined.grad[:, :2])
    torch.testing.assert_close(separate_partner.grad, combined.grad[:, 2:])
    for (legacy_name, legacy_param), (stream_name, stream_param) in zip(
        legacy_model.named_parameters(),
        stream_model.named_parameters(),
    ):
        assert legacy_name == stream_name
        torch.testing.assert_close(stream_param.grad, legacy_param.grad)


def test_logreg_saves_reusable_classical_bundle(tmp_path):
    args = Namespace(
        task_type="pep_function",
        artifact_dir=str(tmp_path),
        encoder="bow",
        reduction="none",
        ml_scaler="none",
        ml_reduction="none",
        logreg_c=1.0,
        logreg_max_iter=100,
        logreg_class_weight="none",
        report_aupr=False,
    )
    trainer = ClassicalMLTrainer(args)
    train = TensorDataset(
        torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float),
        torch.tensor([[0], [0], [1], [1]], dtype=torch.float),
    )
    val = TensorDataset(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float),
        torch.tensor([[0], [1]], dtype=torch.float),
    )

    trainer.run_logreg(train, val)

    payload = joblib.load(tmp_path / "model.joblib")
    assert payload["estimator_kind"] == "probability_ensemble"
    assert (tmp_path / "metadata.json").exists()


def test_profile_classical_bundle_is_not_raw_sequence_inference_ready(tmp_path):
    args = Namespace(
        task_type="pep_function",
        artifact_dir=str(tmp_path),
        encoder="pssm",
        tokenizer="kmer",
        reduction="none",
        ml_scaler="none",
        ml_reduction="none",
        logreg_c=1.0,
        logreg_max_iter=100,
        logreg_class_weight="none",
        report_aupr=False,
    )
    trainer = ClassicalMLTrainer(args)
    train = TensorDataset(
        torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float),
        torch.tensor([[0], [0], [1], [1]], dtype=torch.float),
    )
    val = TensorDataset(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float),
        torch.tensor([[0], [1]], dtype=torch.float),
    )

    trainer.run_logreg(train, val)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bundle_type"] == "sequence_classical"
    assert metadata["supports_inference"] is False
    assert metadata["reason_code"] == "profile_features_require_sample_id"


def test_masked_score_analysis_writes_validation_and_test_curves(tmp_path):
    args = Namespace(artifact_dir=str(tmp_path))
    y_all = torch.tensor([[0], [1], [0], [1]], dtype=torch.float)
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8], dtype=torch.float)
    val_mask = torch.tensor([False, False, True, True])
    test_mask = torch.tensor([True, True, False, False])

    val_y, val_scores, test_y, test_scores = _write_masked_score_analysis(
        args,
        y_all,
        val_mask,
        test_mask,
        lambda mask: scores[mask],
    )

    assert val_y.tolist() == [0, 1]
    assert [round(float(value), 4) for value in val_scores] == [0.2, 0.8]
    assert test_y.tolist() == [0, 1]
    assert [round(float(value), 4) for value in test_scores] == [0.1, 0.9]
    payload = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
    assert payload["curves"]["validation"]["roc"]["available"] is True
    assert payload["curves"]["validation"]["pr"]["available"] is True
    assert payload["curves"]["test"]["roc"]["available"] is True
    assert payload["curves"]["test"]["pr"]["available"] is True


def test_pretrained_embedding_logreg_saves_reusable_bundle(tmp_path):
    args = Namespace(
        task_type="pep_function",
        artifact_dir=str(tmp_path),
        encoder="index",
        reduction="none",
        ml_scaler="none",
        ml_reduction="none",
        logreg_c=1.0,
        logreg_max_iter=100,
        logreg_class_weight="none",
        report_aupr=False,
        _artifact_source="pretrained_embeddings",
        _artifact_info={
            "pretrained": {
                "tokenizer": "esm2",
                "model_name": "models/esm2",
                "max_length": 60,
                "pooling": "attention_mask_mean",
                "input_dim": 2,
            },
            "label_names": ["antimicrobial"],
        },
    )
    trainer = ClassicalMLTrainer(args)
    train = TensorDataset(
        torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float),
        torch.tensor([[0], [0], [1], [1]], dtype=torch.float),
    )
    val = TensorDataset(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float),
        torch.tensor([[0], [1]], dtype=torch.float),
    )

    trainer.run_logreg(train, val)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bundle_type"] == "pretrained_embedding_classical"
    assert metadata["pretrained"]["tokenizer"] == "esm2"
    assert metadata["label_names"] == ["antimicrobial"]
    assert "_artifact_source" not in metadata["args"]


def test_sequence_embedding_logreg_saves_reusable_bundle(tmp_path):
    scaler = StandardScaler().fit(np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32))
    args = Namespace(
        task_type="pep_function",
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="onehot",
        reduction="none",
        ml_scaler="none",
        ml_reduction="none",
        logreg_c=1.0,
        logreg_max_iter=100,
        logreg_class_weight="none",
        report_aupr=False,
        _artifact_source="sequence_embedding_features",
        _artifact_info={
            "input_schema": "peptide_sequences",
            "label_names": ["antimicrobial"],
            "embedding_export": {
                "encoder_path": "embedding_encoder.pt",
                "embedding_dim": 2,
                "downstream": "logreg",
            },
            "dl_scaler": scaler,
        },
    )
    trainer = ClassicalMLTrainer(args)
    train = TensorDataset(
        torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float),
        torch.tensor([[0], [0], [1], [1]], dtype=torch.float),
    )
    val = TensorDataset(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float),
        torch.tensor([[0], [1]], dtype=torch.float),
    )

    trainer.run_logreg(train, val)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bundle_type"] == "sequence_embedding_classical"
    assert metadata["supports_inference"] is True
    assert metadata["input_schema"] == "peptide_sequences"
    assert metadata["embedding_export"]["encoder_path"] == "embedding_encoder.pt"
    assert metadata["preprocess_path"] == "preprocess.joblib"
    assert metadata["label_names"] == ["antimicrobial"]
    assert "_artifact_source" not in metadata["args"]
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "preprocess.joblib").exists()


def test_pretrained_embedding_nn_artifact_saves_metadata_and_preprocess(tmp_path):
    pipeline = object.__new__(Pipeline)
    pipeline.args = Namespace(
        task_type="pep_function",
        artifact_dir=str(tmp_path),
        tokenizer="esm2",
        model="mlp",
        classifier="mlp",
        head_hidden_dims=[],
        max_length=60,
    )
    pipeline.device = torch.device("cpu")
    scaler = {"kind": "unit"}

    pipeline._save_pretrained_embedding_nn_artifact(
        torch.nn.Linear(2, 1),
        {
            "pretrained": {
                "tokenizer": "esm2",
                "model_name": "models/esm2",
                "max_length": 60,
                "pooling": "attention_mask_mean",
                "input_dim": 2,
            },
            "label_names": ["antimicrobial"],
        },
        dl_scaler=scaler,
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bundle_type"] == "pretrained_embedding_nn"
    assert metadata["supports_inference"] is True
    assert metadata["preprocess_path"] == "preprocess.joblib"
    assert metadata["pretrained"]["model_name"] == "models/esm2"
    assert metadata["prediction_head"] == {
        "schema_version": 1,
        "implementation": "standard_linear_mlp",
        "kind": "linear",
        "hidden_dims": [],
    }
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "preprocess.joblib").exists()


def test_profile_sequence_nn_export_requires_validated_profile_cache(tmp_path):
    pipeline = object.__new__(Pipeline)
    pipeline.args = Namespace(
        task_type="pep_function",
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="pssm",
        model="cnn",
        sequence_model="cnn",
        training_mode="end_to_end",
        classifier="mlp",
        dl_scaler="none",
    )
    pipeline.device = torch.device("cpu")
    train = TensorDataset(
        torch.tensor([[0.0, 1.0]], dtype=torch.float),
        torch.tensor([[1.0]], dtype=torch.float),
    )
    train.label_names = ["antimicrobial"]

    with pytest.raises(
        RuntimeError,
        match=rf"validated v{PROFILE_CONTRACT_VERSION} profile manifest",
    ):
        pipeline._save_sequence_nn_artifact(torch.nn.Linear(2, 1), train)
    assert not (tmp_path / "metadata.json").exists()


def test_profile_sequence_nn_artifact_saves_profile_generation(tmp_path):
    data_dir = tmp_path / "data"
    for split, seq in [("train", "ACDE"), ("val", "FGHI")]:
        split_dir = data_dir / split
        split_dir.mkdir(parents=True)
        content = f">{split}_1\n{seq}\n"
        if split == "train":
            content += f">{split}_2\n{seq}\n"
        (split_dir / "seqs.fasta").write_text(content, encoding="utf-8")
    pipeline = object.__new__(Pipeline)
    pipeline.args = Namespace(
        task_type="pep_function",
        data_dir=str(data_dir),
        artifact_dir=str(tmp_path / "artifact"),
        tokenizer="kmer",
        encoder="pssm",
        model="cnn",
        sequence_model="cnn",
        training_mode="end_to_end",
        classifier="mlp",
        dl_scaler="none",
        max_length=4,
        profile_iterations=2,
        profile_evalue=0.01,
        profile_inclusion_evalue=0.001,
        profile_threads=1,
        profile_on_missing="error",
    )
    pipeline.device = torch.device("cpu")
    pipeline._build_profile_blast_db = lambda subject_fasta, db_prefix: False
    subject_fasta_sha256 = hashlib.sha256(b">TRAIN_000000\nACDE\n>TRAIN_000001\nACDE\n").hexdigest()
    subject_source = tmp_path / "profile_cache" / "subjects" / "test.fasta"
    subject_source.parent.mkdir(parents=True)
    subject_source.write_bytes(b">TRAIN_000000\nACDE\n>TRAIN_000001\nACDE\n")
    pipeline._profile_manifest_for_current_route = lambda: (
        {"provenance": _profile_provenance(subject_fasta_sha256=subject_fasta_sha256)},
        "d" * 64,
        subject_source,
    )
    train = TensorDataset(
        torch.tensor([[0.0, 1.0]], dtype=torch.float),
        torch.tensor([[1.0]], dtype=torch.float),
    )
    train.label_names = ["antimicrobial"]

    pipeline._save_sequence_nn_artifact(torch.nn.Linear(2, 1), train)

    metadata = json.loads((tmp_path / "artifact" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bundle_type"] == "sequence_nn"
    assert metadata["supports_inference"] is True
    assert metadata["profile_generation"]["feature"] == "pssm"
    assert metadata["profile_generation"]["subject_fasta"] == "profile_generation/subject.fasta"
    subject = (tmp_path / "artifact" / "profile_generation" / "subject.fasta").read_text(
        encoding="utf-8"
    )
    assert "ACDE" in subject
    assert subject.count("ACDE") == 2
    assert "FGHI" not in subject
    assert (
        tmp_path / "artifact" / "profile_generation" / "subject.fasta"
    ).read_bytes() == subject_source.read_bytes()
    assert metadata["profile_generation"]["db_scope"] == "train_only"
    assert "reason_code" not in metadata


def test_profile_sequence_classical_artifact_saves_profile_generation(tmp_path):
    args = Namespace(
        task_type="pep_function",
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="pssm",
        reduction="none",
        ml_scaler="none",
        ml_reduction="none",
        logreg_c=1.0,
        logreg_max_iter=100,
        logreg_class_weight="none",
        report_aupr=False,
        _artifact_source="sequence_features",
        _artifact_info={
            "input_schema": "peptide_sequences",
            "label_names": ["antimicrobial"],
            "profile_generation": {
                "profile_contract_version": PROFILE_CONTRACT_VERSION,
                "strategy": "psiblast_subject_fasta",
                "feature": "pssm",
                "subject_fasta": "profile_generation/subject.fasta",
                "db_prefix": "",
                "db_scope": "train_only",
                "iterations": 2,
                "evalue": 0.01,
                "inclusion_evalue": 0.001,
                "threads": 1,
                "on_missing": "error",
                "scientific_parameters": ProfileGenerationConfig(
                    iterations=2,
                    evalue=0.01,
                    inclusion_evalue=0.001,
                    threads=1,
                ).scientific_parameters(),
                "manifest_sha256": "d" * 64,
                "training_profile_provenance": _profile_provenance(),
            },
        },
    )
    trainer = ClassicalMLTrainer(args)
    train = TensorDataset(
        torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float),
        torch.tensor([[0], [0], [1], [1]], dtype=torch.float),
    )
    val = TensorDataset(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float),
        torch.tensor([[0], [1]], dtype=torch.float),
    )

    trainer.run_logreg(train, val)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bundle_type"] == "sequence_classical"
    assert metadata["supports_inference"] is True
    assert metadata["profile_generation"]["subject_fasta"] == "profile_generation/subject.fasta"
    assert "reason_code" not in metadata


def test_sequence_nn_artifact_saves_dl_scaler_preprocess(tmp_path):
    pipeline = object.__new__(Pipeline)
    pipeline.args = Namespace(
        task_type="pep_function",
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="onehot",
        model="cnn",
        sequence_model="cnn",
        training_mode="end_to_end",
        classifier="mlp",
        dl_scaler="standard",
    )
    pipeline.device = torch.device("cpu")
    train = TensorDataset(
        torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float),
        torch.tensor([[0.0], [1.0]], dtype=torch.float),
    )
    train.label_names = ["antimicrobial"]
    scaler = StandardScaler().fit(train.tensors[0].numpy())

    pipeline._save_sequence_nn_artifact(
        FeatureTextCNN(in_dim=2, embed_dim=8, num_classes=1),
        train,
        dl_scaler=scaler,
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bundle_type"] == "sequence_nn"
    assert metadata["supports_inference"] is True
    assert metadata["preprocess_path"] == "preprocess.joblib"
    assert metadata["predictor_contract"]["contract_version"] == 2
    assert metadata["predictor_contract"]["readout"] == "masked_global_max"
    assert (tmp_path / "preprocess.joblib").exists()


def test_sequence_nn_artifact_saves_rnn_predictor_contract(tmp_path):
    pipeline = object.__new__(Pipeline)
    pipeline.args = Namespace(
        task_type="pep_function",
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="onehot",
        model="rnn",
        sequence_model="rnn",
        training_mode="end_to_end",
        classifier="mlp",
        dl_scaler="none",
    )
    pipeline.device = torch.device("cpu")
    train = TensorDataset(
        torch.randn(2, 4, 5),
        torch.tensor([[0.0], [1.0]], dtype=torch.float),
    )
    train.label_names = ["antimicrobial"]

    pipeline._save_sequence_nn_artifact(
        FeatureRNN(in_dim=5, hidden_dim=6, num_classes=1),
        train,
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["predictor_contract"]["implementation"] == "pepplatform_bidirectional_elman_rnn"
    assert metadata["predictor_contract"]["contract_version"] == 2
    assert metadata["predictor_contract"]["status"] == "verified_platform_variant"
    assert metadata["predictor_contract"]["padding_mask"] == "explicit_position_mask"


def test_sequence_nn_artifact_saves_gru_predictor_contract(tmp_path):
    pipeline = object.__new__(Pipeline)
    pipeline.args = Namespace(
        task_type="pep_function",
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="onehot",
        model="gru",
        sequence_model="gru",
        training_mode="end_to_end",
        classifier="mlp",
        dl_scaler="none",
    )
    pipeline.device = torch.device("cpu")
    train = TensorDataset(
        torch.randn(2, 4, 5),
        torch.tensor([[0.0], [1.0]], dtype=torch.float),
    )
    train.label_names = ["antimicrobial"]

    pipeline._save_sequence_nn_artifact(
        FeatureGRU(in_dim=5, hidden_dim=6, num_classes=1),
        train,
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert (
        metadata["predictor_contract"]["implementation"] == "pepplatform_bidirectional_pytorch_gru"
    )
    assert metadata["predictor_contract"]["contract_version"] == 2
    assert metadata["predictor_contract"]["status"] == "verified_platform_variant"
    assert metadata["predictor_contract"]["padding_mask"] == "explicit_position_mask"


def test_sequence_nn_artifact_saves_lstm_predictor_contract(tmp_path):
    pipeline = object.__new__(Pipeline)
    pipeline.args = Namespace(
        task_type="pep_function",
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="onehot",
        model="lstm",
        sequence_model="lstm",
        training_mode="end_to_end",
        classifier="mlp",
        dl_scaler="none",
    )
    pipeline.device = torch.device("cpu")
    train = TensorDataset(
        torch.randn(2, 4, 5),
        torch.tensor([[0.0], [1.0]], dtype=torch.float),
    )
    train.label_names = ["antimicrobial"]

    pipeline._save_sequence_nn_artifact(
        FeatureLSTM(in_dim=5, hidden_dim=6, num_classes=1),
        train,
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert (
        metadata["predictor_contract"]["implementation"] == "pepplatform_bidirectional_pytorch_lstm"
    )
    assert metadata["predictor_contract"]["contract_version"] == 2
    assert metadata["predictor_contract"]["status"] == "verified_platform_variant"
    assert metadata["predictor_contract"]["padding_mask"] == "explicit_position_mask"
    assert metadata["predictor_contract"]["gate_order"] == ["input", "forget", "cell", "output"]


def test_sequence_nn_artifact_saves_transformer_predictor_contract(tmp_path):
    pipeline = object.__new__(Pipeline)
    pipeline.args = Namespace(
        task_type="pep_function",
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="onehot",
        model="transformer",
        sequence_model="transformer",
        training_mode="end_to_end",
        classifier="mlp",
        dl_scaler="none",
    )
    pipeline.device = torch.device("cpu")
    train = TensorDataset(
        torch.randn(2, 4, 5),
        torch.tensor([[0.0], [1.0]], dtype=torch.float),
    )
    train.label_names = ["antimicrobial"]

    pipeline._save_sequence_nn_artifact(
        FeatureTransformer(
            in_dim=5,
            max_length=4,
            embed_dim=8,
            num_heads=2,
            ff_dim=16,
            num_classes=1,
        ),
        train,
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    contract = metadata["predictor_contract"]
    assert contract["implementation"] == "pepplatform_pytorch_transformer_encoder"
    assert contract["contract_version"] == 2
    assert contract["status"] == "verified_platform_variant"
    assert contract["padding_mask"] == "explicit_position_mask"
    assert contract["position_indexing"] == "valid_token_rank"
    assert contract["readout"] == "masked_mean"


def test_pretrained_residue_artifact_saves_transformer_predictor_contract(tmp_path):
    pipeline = object.__new__(Pipeline)
    pipeline.args = Namespace(
        task_type="pep_function",
        artifact_dir=str(tmp_path),
        tokenizer="esm2",
        encoder="esm2",
        model="transformer",
        sequence_model="transformer",
        training_mode="end_to_end",
        classifier="mlp",
        dl_scaler="none",
        _artifact_source="pretrained_residue_embeddings",
    )
    pipeline.device = torch.device("cpu")
    train = TensorDataset(
        torch.randn(2, 4, 5),
        torch.tensor([[0.0], [1.0]], dtype=torch.float),
    )
    train.label_names = ["antimicrobial"]

    pipeline._save_sequence_nn_artifact(
        PretrainedResiduePredictor(
            "transformer",
            input_dim=5,
            adapter_dim=8,
            max_length=4,
            heads=2,
            num_classes=1,
        ),
        train,
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bundle_type"] == "pretrained_residue_nn"
    contract = metadata["predictor_contract"]
    assert contract["implementation"] == "pepplatform_pytorch_transformer_encoder"
    assert contract["contract_version"] == 2
    assert contract["input_adapter"] == "layer_norm_linear_projection"
    assert contract["padding_mask"] == "explicit_residue_mask"


def _pair_vector_args(tmp_path, model="mlp", task_type="pep_protein_interaction"):
    return Namespace(
        task_type=task_type,
        artifact_dir=str(tmp_path),
        tokenizer="esm2",
        encoder="ppi_fusion",
        model=model,
        classifier=model,
        pair_strategy="two_stream",
        interaction_level="vector",
        predictor_type="classifier",
        fusion_mode="concat_all",
        max_length=512,
        batch_size=2,
        esm2_model="models/esm2",
        _resolved_pretrained_model_source="models/esm2",
    )


def _binary_train_val(width):
    train = TensorDataset(
        torch.tensor(
            [
                [0.0] * width,
                [0.25] * width,
                [0.75] * width,
                [1.0] * width,
            ],
            dtype=torch.float,
        ),
        torch.tensor([[0], [0], [1], [1]], dtype=torch.float),
    )
    val = TensorDataset(
        torch.tensor([[0.0] * width, [1.0] * width], dtype=torch.float),
        torch.tensor([[0], [1]], dtype=torch.float),
    )
    return train, val


def _pair_embedding_classical_args(tmp_path, task_type, input_schema):
    return Namespace(
        task_type=task_type,
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="index",
        model="logreg",
        classifier="logreg",
        reduction="none",
        ml_scaler="none",
        ml_reduction="none",
        logreg_c=1.0,
        logreg_max_iter=100,
        logreg_class_weight="none",
        report_aupr=False,
        _artifact_source="pair_embedding_features",
        _artifact_info={
            "input_schema": input_schema,
            "label_names": ["interaction"],
            "classic": {"tokenizer": "kmer", "encoder": "index", "max_length": 2, "k": 1},
            "pair": {"pair_strategy": "single_stream"},
            "embedding_export": {
                "encoder_path": "embedding_encoder.pt",
                "embedding_dim": 2,
                "downstream": "logreg",
            },
        },
    )


def test_pair_vector_nn_artifact_saves_metadata_and_preprocess(tmp_path):
    args = _pair_vector_args(tmp_path, "mlp")
    args.head_hidden_dims = [128]
    args.head_activation = "relu"
    args.head_dropout = 0.2
    peptide_embeddings = torch.ones(2, 2)
    protein_embeddings = torch.ones(2, 2)
    fused_features = torch.ones(2, 8)
    configure_pair_vector_artifacts(args, peptide_embeddings, protein_embeddings, fused_features)
    scaler = StandardScaler().fit(fused_features.numpy())
    _set_binary_operating_point(args, 0.35)

    save_pair_vector_nn_artifact(args, torch.nn.Linear(8, 1), dl_scaler=scaler, threshold=0.35)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bundle_type"] == "pair_vector_nn"
    assert metadata["input_schema"] == "peptide_protein_pairs"
    assert metadata["pair"]["fusion_mode"] == "concat_all"
    assert metadata["pair"]["fused_dim"] == 8
    assert metadata["pretrained"]["model_name"] == "models/esm2"
    assert metadata["label_names"] == ["interaction"]
    assert metadata["threshold"] == 0.35
    assert metadata["threshold_source"] == "validation_f1"
    assert metadata["evaluation_protocol_version"] == EVALUATION_PROTOCOL_VERSION
    assert metadata["feature_fit_scope"] == "train_only"
    assert metadata["checkpoint_metric"] == "validation_roc_auc"
    assert metadata["prediction_head"] == {
        "schema_version": 1,
        "implementation": "standard_linear_mlp",
        "kind": "mlp",
        "hidden_dims": [128],
        "activation": "relu",
        "dropout": 0.2,
    }
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "preprocess.joblib").exists()


def test_hlai_pair_vector_nn_artifact_saves_hlai_schema(tmp_path):
    args = _pair_vector_args(tmp_path, "mlp", task_type="pep_hlai")
    peptide_embeddings = torch.ones(2, 2)
    protein_embeddings = torch.ones(2, 2)
    fused_features = torch.ones(2, 8)
    configure_pair_vector_artifacts(args, peptide_embeddings, protein_embeddings, fused_features)
    _set_binary_operating_point(args, 0.55)

    save_pair_vector_nn_artifact(args, torch.nn.Linear(8, 1), threshold=0.55)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bundle_type"] == "pair_vector_nn"
    assert metadata["task_type"] == "pep_hlai"
    assert metadata["input_schema"] == "peptide_hlai_pairs"
    assert metadata["supports_inference"] is True
    assert metadata["threshold"] == 0.55


def test_hlaii_pair_vector_nn_artifact_saves_hla_schema(tmp_path):
    args = _pair_vector_args(tmp_path, "mlp", task_type="pep_hlaii")
    peptide_embeddings = torch.ones(2, 2)
    protein_embeddings = torch.ones(2, 2)
    fused_features = torch.ones(2, 8)
    configure_pair_vector_artifacts(args, peptide_embeddings, protein_embeddings, fused_features)
    _set_binary_operating_point(args, 0.45)

    save_pair_vector_nn_artifact(args, torch.nn.Linear(8, 1), threshold=0.45)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bundle_type"] == "pair_vector_nn"
    assert metadata["task_type"] == "pep_hlaii"
    assert metadata["input_schema"] == "peptide_hla_pairs"
    assert metadata["supports_inference"] is True
    assert metadata["threshold"] == 0.45


def test_hlaii_pair_sequence_nn_artifact_saves_hla_schema(tmp_path):
    args = Namespace(
        task_type="pep_hlaii",
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="index",
        model="cnn",
        classifier="mlp",
        pair_strategy="single_stream",
        sequence_model="cnn",
        training_mode="end_to_end",
        max_length=2,
        k=1,
    )
    train_x = torch.ones((2, 4), dtype=torch.long)
    _set_binary_operating_point(args, 0.4)

    save_pair_sequence_nn_artifact(
        args,
        IndexTextCNN(vocab_size=24, embed_dim=8, num_classes=1),
        train_x,
        threshold=0.4,
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bundle_type"] == "pair_sequence_nn"
    assert metadata["task_type"] == "pep_hlaii"
    assert metadata["input_schema"] == "peptide_hla_pairs"
    assert metadata["supports_inference"] is True
    assert metadata["pair"]["pair_strategy"] == "single_stream"
    assert metadata["predictor_contract"]["contract_version"] == 2
    assert metadata["predictor_contract"]["padding_mask"] == "token_id_nonzero"


def test_pair_sequence_nn_artifact_saves_rnn_predictor_contract(tmp_path):
    args = Namespace(
        task_type="pep_protein_interaction",
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="index",
        model="rnn",
        classifier="mlp",
        pair_strategy="single_stream",
        sequence_model="rnn",
        training_mode="end_to_end",
        max_length=2,
        k=1,
    )
    train_x = torch.tensor([[1, 2, 0, 3], [4, 0, 0, 5]], dtype=torch.long)
    _set_binary_operating_point(args, 0.4)

    save_pair_sequence_nn_artifact(
        args,
        IndexRNN(vocab_size=24, embed_dim=8, hidden_dim=6, num_classes=1),
        train_x,
        threshold=0.4,
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["predictor_contract"]["implementation"] == "pepplatform_bidirectional_elman_rnn"
    assert metadata["predictor_contract"]["contract_version"] == 2
    assert (
        metadata["predictor_contract"]["sequence_handling"] == "mask_compact_pack_padded_sequence"
    )
    assert metadata["predictor_contract"]["padding_mask"] == "token_id_nonzero"


def test_pair_sequence_nn_artifact_saves_gru_predictor_contract(tmp_path):
    args = Namespace(
        task_type="pep_protein_interaction",
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="index",
        model="gru",
        classifier="mlp",
        pair_strategy="single_stream",
        sequence_model="gru",
        training_mode="end_to_end",
        max_length=2,
        k=1,
    )
    train_x = torch.tensor([[1, 2, 0, 3], [4, 0, 0, 5]], dtype=torch.long)
    _set_binary_operating_point(args, 0.4)

    save_pair_sequence_nn_artifact(
        args,
        IndexGRU(vocab_size=24, embed_dim=8, hidden_dim=6, num_classes=1),
        train_x,
        threshold=0.4,
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert (
        metadata["predictor_contract"]["implementation"] == "pepplatform_bidirectional_pytorch_gru"
    )
    assert metadata["predictor_contract"]["contract_version"] == 2
    assert (
        metadata["predictor_contract"]["sequence_handling"] == "mask_compact_pack_padded_sequence"
    )
    assert metadata["predictor_contract"]["padding_mask"] == "token_id_nonzero"


def test_pair_sequence_nn_artifact_saves_lstm_predictor_contract(tmp_path):
    args = Namespace(
        task_type="pep_protein_interaction",
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="index",
        model="lstm",
        classifier="mlp",
        pair_strategy="single_stream",
        sequence_model="lstm",
        training_mode="end_to_end",
        max_length=2,
        k=1,
    )
    train_x = torch.tensor([[1, 2, 0, 3], [4, 0, 0, 5]], dtype=torch.long)
    _set_binary_operating_point(args, 0.4)

    save_pair_sequence_nn_artifact(
        args,
        IndexLSTM(vocab_size=24, embed_dim=8, hidden_dim=6, num_classes=1),
        train_x,
        threshold=0.4,
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert (
        metadata["predictor_contract"]["implementation"] == "pepplatform_bidirectional_pytorch_lstm"
    )
    assert metadata["predictor_contract"]["contract_version"] == 2
    assert (
        metadata["predictor_contract"]["sequence_handling"] == "mask_compact_pack_padded_sequence"
    )
    assert metadata["predictor_contract"]["padding_mask"] == "token_id_nonzero"


def test_pair_sequence_nn_artifact_saves_transformer_predictor_contract(tmp_path):
    args = Namespace(
        task_type="pep_protein_interaction",
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="index",
        model="transformer",
        classifier="mlp",
        pair_strategy="single_stream",
        sequence_model="transformer",
        training_mode="end_to_end",
        max_length=2,
        k=1,
    )
    train_x = torch.tensor([[1, 2, 0, 3], [4, 0, 0, 5]], dtype=torch.long)
    _set_binary_operating_point(args, 0.4)

    save_pair_sequence_nn_artifact(
        args,
        IndexTransformer(
            vocab_size=24,
            max_length=4,
            embed_dim=8,
            num_heads=2,
            ff_dim=16,
            num_classes=1,
        ),
        train_x,
        threshold=0.4,
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    contract = metadata["predictor_contract"]
    assert contract["implementation"] == "pepplatform_pytorch_transformer_encoder"
    assert contract["contract_version"] == 2
    assert contract["sequence_handling"] == "key_padding_mask_with_mask_compact_positions"
    assert contract["padding_mask"] == "token_id_nonzero"


def test_pair_sequence_nn_artifact_saves_gcn_predictor_contract(tmp_path):
    args = Namespace(
        task_type="pep_protein_interaction",
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="onehot",
        model="gcn",
        classifier="mlp",
        pair_strategy="single_stream",
        sequence_model="gcn",
        training_mode="end_to_end",
        max_length=2,
        k=1,
    )
    train_x = ResidueModelInput(
        torch.randn(2, 4, 5),
        torch.tensor([[True, True, True, False], [True, True, False, True]]),
    )
    _set_binary_operating_point(args, 0.4)

    save_pair_sequence_nn_artifact(
        args,
        ResidueGraphClassifier(in_dim=5, hidden_dim=8, num_classes=1, dropout=0.0),
        train_x,
        threshold=0.4,
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    contract = metadata["predictor_contract"]
    assert contract["implementation"] == "pepplatform_masked_chain_gcn"
    assert contract["contract_version"] == 2
    assert contract["padding_mask"] == "explicit_or_feature_compatibility_mask"
    assert contract["readout"] == "masked mean plus masked max"


def test_pair_graph_nn_artifact_saves_reference_and_preprocess(tmp_path):
    args = _pair_vector_args(tmp_path, "gcn")
    args.predictor_type = "sample_graph"
    args.graph_predictor = "gcn"
    args.gnn_k = 1
    args.gnn_hidden_dim = 4
    args.dropout = 0.0
    peptide_embeddings = torch.ones(3, 2)
    protein_embeddings = torch.ones(3, 2)
    graph_features = torch.ones(3, 8)
    configure_pair_graph_artifacts(args, peptide_embeddings, protein_embeddings, graph_features)
    _set_binary_operating_point(args, 0.4)
    scaler = StandardScaler().fit(peptide_embeddings.numpy())

    from scripts.train_pair_pipeline import PairGCNClassifier

    model = PairGCNClassifier(input_dim=8, hidden_dim=4, dropout=0.0)
    save_pair_graph_nn_artifact(
        args,
        model,
        reference_features=graph_features[:2],
        peptide_scaler=scaler,
        protein_scaler=scaler,
        threshold=0.4,
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    reference = torch.load(tmp_path / "graph_reference.pt", map_location="cpu", weights_only=False)
    payload = torch.load(tmp_path / "model.pt", map_location="cpu", weights_only=False)
    assert metadata["bundle_type"] == "pair_graph_nn"
    assert metadata["supports_inference"] is True
    assert metadata["graph"]["predictor"] == "gcn"
    assert metadata["graph"]["reference_source"] == "train_only"
    assert metadata["graph"]["inference_graph"] == "directed_train_reference_to_query"
    assert metadata["graph"]["input_dim"] == 8
    assert metadata["graph"]["aligned_gated"] is False
    assert (
        metadata["predictor_contract"]["implementation"]
        == "pepplatform_inductive_pair_sample_graph_gcn"
    )
    assert metadata["predictor_contract"]["status"] == "verified_platform_variant"
    assert (
        metadata["predictor_contract"]["node_definition"]
        == "one node per fused peptide-partner sample"
    )
    assert metadata["threshold"] == 0.4
    assert reference["reference_features"].shape == (2, 8)
    assert payload["model_class"] == "PairGCNClassifier"
    assert "state_dict" in payload
    assert (tmp_path / "preprocess.joblib").exists()


def test_pair_graph_artifact_records_aligned_gated_canonical_flag(tmp_path):
    args = _pair_vector_args(tmp_path, "gcn")
    args.predictor_type = "sample_graph"
    args.graph_predictor = "gcn"
    args.fusion_mode = "aligned_gated"
    peptide_embeddings = torch.ones(3, 2)
    protein_embeddings = torch.ones(3, 2)
    graph_features = torch.ones(3, 6)

    info = configure_pair_graph_artifacts(
        args,
        peptide_embeddings,
        protein_embeddings,
        graph_features,
    )

    assert info["pair"]["fusion_mode"] == "aligned_gated"
    assert info["graph"]["aligned_gated"] is True


def test_pair_gat_artifact_records_memory_bounded_aggregation(tmp_path):
    args = _pair_vector_args(tmp_path, "gat")
    args.predictor_type = "sample_graph"
    args.graph_predictor = "gat"
    args.gnn_k = 2
    args.gnn_hidden_dim = 4
    args.dropout = 0.0
    peptide_embeddings = torch.ones(3, 2)
    protein_embeddings = torch.ones(3, 2)
    graph_features = torch.ones(3, 8)

    info = configure_pair_graph_artifacts(
        args,
        peptide_embeddings,
        protein_embeddings,
        graph_features,
    )

    assert info["graph"]["aggregation"] == "memory_bounded_exact_softmax"
    assert info["graph"]["topology_contract"] == "gat_unique_incoming_knn_v2"
    assert info["graph"]["reference_neighbors"] == ("exactly_k_unique_incoming_plus_one_self_loop")
    assert info["graph"]["query_neighbors"] == ("reference_to_query_only_plus_one_self_loop")
    assert info["graph"]["edge_chunk_size"] == 100_000
    assert info["predictor_contract"]["implementation"] == (
        "pepplatform_inductive_pair_sample_graph_gat"
    )
    assert info["predictor_contract"]["status"] == "verified_platform_variant"
    assert info["predictor_contract"]["topology_contract"] == ("gat_unique_incoming_knn_v2")
    assert info["predictor_contract"]["attention"] == (
        "single-head additive neighbor attention with destination-wise softmax"
    )


def test_pair_hgt_artifact_records_typed_graph_contract_and_references(tmp_path):
    args = _pair_vector_args(tmp_path, "hgt")
    args.predictor_type = "sample_graph"
    args.graph_predictor = "hgt"
    args.gnn_k = 2
    args.gnn_hidden_dim = 8
    args.dropout = 0.0
    args.fusion_mode = None
    peptide_embeddings = torch.ones(3, 3)
    partner_embeddings = torch.ones(3, 4)

    info = configure_pair_graph_artifacts(
        args,
        peptide_embeddings,
        partner_embeddings,
        None,
    )
    _set_binary_operating_point(args, 0.4)
    save_pair_graph_nn_artifact(
        args,
        PairHGTClassifier(3, 4, hidden_dim=8, heads=4, dropout=0.0),
        reference_peptide_features=peptide_embeddings,
        reference_protein_features=partner_embeddings,
        threshold=0.4,
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    reference = torch.load(tmp_path / "graph_reference.pt", map_location="cpu", weights_only=False)
    assert info["graph"]["schema"] == "typed_pair_hgt_v1"
    assert info["pair"]["input_adapter"] == "typed_role_projection"
    assert "fusion_mode" not in info["pair"]
    assert "fused_dim" not in info["pair"]
    assert "input_dim" not in info["graph"]
    assert info["graph"]["node_types"] == ["peptide", "partner"]
    assert info["graph"]["label_edges_in_message_graph"] is False
    assert info["predictor_contract"]["implementation"] == ("pepplatform_inductive_typed_pair_hgt")
    assert metadata["predictor_contract"]["contract_version"] == 1
    assert "fusion_mode" not in metadata["args"]
    assert reference["reference_peptide_features"].shape == (3, 3)
    assert reference["reference_protein_features"].shape == (3, 4)


def test_pair_sage_artifact_records_mean_root_neighbor_contract(tmp_path):
    args = _pair_vector_args(tmp_path, "sage")
    args.predictor_type = "sample_graph"
    args.graph_predictor = "sage"
    args.gnn_k = 2
    args.gnn_hidden_dim = 4
    args.dropout = 0.0
    peptide_embeddings = torch.ones(3, 2)
    protein_embeddings = torch.ones(3, 2)
    graph_features = torch.ones(3, 8)

    info = configure_pair_graph_artifacts(
        args,
        peptide_embeddings,
        protein_embeddings,
        graph_features,
    )

    assert info["graph"]["topology_contract"] == ("sage_unique_incoming_nonself_knn_v1")
    assert info["graph"]["reference_neighbors"] == ("exactly_k_unique_incoming_nonself")
    assert info["graph"]["query_neighbors"] == "reference_to_query_only"
    assert info["graph"]["self_loop_policy"] == "no_edge_self_loops_root_update_only"
    assert info["predictor_contract"]["implementation"] == (
        "pepplatform_inductive_pair_sample_graph_sage_mean"
    )
    assert info["predictor_contract"]["contract_version"] == 1
    assert info["predictor_contract"]["root_update"] == ("independent trainable root projection")

    _set_binary_operating_point(args, 0.4)
    from scripts.train_pair_pipeline import PairSAGEClassifier

    save_pair_graph_nn_artifact(
        args,
        PairSAGEClassifier(input_dim=8, hidden_dim=4, dropout=0.0),
        reference_features=graph_features[:2],
        threshold=0.4,
    )
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["predictor_contract"]["status"] == "verified_platform_variant"
    assert metadata["predictor_contract"]["aggregation"] == (
        "mean neighbor aggregation followed by independent trainable neighbor projection"
    )


def test_pair_graph_transformer_artifact_records_attention_block_contract(tmp_path):
    args = _pair_vector_args(tmp_path, "graph_transformer")
    args.predictor_type = "sample_graph"
    args.graph_predictor = "graph_transformer"
    args.gnn_k = 2
    args.gnn_hidden_dim = 8
    args.dropout = 0.0
    peptide_embeddings = torch.ones(3, 2)
    protein_embeddings = torch.ones(3, 2)
    graph_features = torch.ones(3, 8)

    info = configure_pair_graph_artifacts(
        args,
        peptide_embeddings,
        protein_embeddings,
        graph_features,
    )

    assert info["graph"]["topology_contract"] == ("graph_transformer_unique_incoming_knn_v1")
    assert info["graph"]["aggregation"] == ("exact_destination_softmax_multi_head_attention")
    assert info["graph"]["reference_neighbors"] == ("exactly_k_unique_incoming_plus_one_self_loop")
    assert info["graph"]["query_neighbors"] == ("reference_to_query_only_plus_one_self_loop")
    assert info["predictor_contract"]["implementation"] == (
        "pepplatform_inductive_pair_sample_graph_graph_transformer"
    )
    assert info["predictor_contract"]["contract_version"] == 1
    assert "LayerNorm" in info["predictor_contract"]["transformer_block"]

    _set_binary_operating_point(args, 0.4)
    save_pair_graph_nn_artifact(
        args,
        PairGraphTransformerClassifier(
            input_dim=8,
            hidden_dim=8,
            heads=4,
            dropout=0.0,
        ),
        reference_features=graph_features[:2],
        threshold=0.4,
    )
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    payload = torch.load(tmp_path / "model.pt", map_location="cpu", weights_only=False)
    assert metadata["predictor_contract"]["status"] == "verified_platform_variant"
    assert metadata["graph"]["self_loop_policy"] == "exactly_one_per_node"
    assert payload["model_class"] == "PairGraphTransformerClassifier"


def test_pair_token_cross_attention_artifact_saves_metadata(tmp_path):
    args = Namespace(
        task_type="pep_protein_interaction",
        artifact_dir=str(tmp_path),
        tokenizer="esm2",
        encoder="ppi_fusion",
        model="cross_attn+mlp",
        classifier="mlp",
        pair_strategy="two_stream",
        interaction_level="token",
        interaction_model="cross_attn",
        training_mode="end_to_end",
        max_length=2,
        batch_size=2,
        epochs=1,
        lr=0.001,
        dropout=0.0,
        interaction_hidden_dim=16,
        interaction_heads=2,
        weight_decay=0.0,
        grad_clip_norm=0.0,
        early_stop_patience=0,
        early_stop_min_delta=0.0,
        report_aupr=False,
        esm2_model="models/esm2",
        _resolved_pretrained_model_source="models/esm2",
    )
    train_x = torch.tensor(
        [
            [[[1.0, 0.0], [0.5, 0.0], [0.0, 1.0], [0.0, 0.5]]],
            [[[1.0, 0.2], [0.5, 0.2], [0.0, 1.2], [0.0, 0.7]]],
            [[[0.0, 1.0], [0.0, 0.5], [1.0, 0.0], [0.5, 0.0]]],
            [[[0.2, 1.0], [0.2, 0.5], [1.2, 0.0], [0.7, 0.0]]],
        ],
        dtype=torch.float,
    ).squeeze(1)
    train_y = torch.tensor([[1], [1], [0], [0]], dtype=torch.float)
    val_x = train_x.clone()
    val_y = train_y.clone()

    train_cross_attention_model(args, None, train_x, train_y, val_x, val_y, device="cpu")

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    payload = torch.load(tmp_path / "model.pt", map_location="cpu", weights_only=False)
    assert metadata["bundle_type"] == "pair_token_nn"
    assert metadata["supports_inference"] is True
    assert metadata["input_schema"] == "peptide_protein_pairs"
    assert metadata["pretrained"]["representation"] == "residue_token_hidden"
    assert metadata["pretrained"]["representation_contract_version"] == 1
    assert metadata["pretrained"]["explicit_residue_mask"] is False
    assert metadata["pretrained"]["token_dim"] == 2
    assert metadata["pair"]["interaction_level"] == "token"
    assert metadata["pair"]["interaction_model"] == "cross_attn"
    assert metadata["pair"]["seq_len"] == 2
    assert metadata["pair"]["hidden_dim"] == 16
    assert metadata["pair"]["n_heads"] == 2
    assert metadata["predictor_contract"]["direction"] == ("peptide_query_partner_key_value")
    assert metadata["threshold_source"] == "validation_f1"
    assert isinstance(metadata["threshold"], float)
    assert metadata["evaluation_protocol_version"] == EVALUATION_PROTOCOL_VERSION
    assert payload["model_class"] == "PairCrossAttentionClassifier"
    assert "state_dict" in payload


def test_pair_token_training_accepts_separate_homogeneous_streams_without_dataset_concat(
    tmp_path, monkeypatch
):
    import scripts.train_pair_pipeline as pair_pipeline

    args = Namespace(
        task_type="pep_protein_interaction",
        artifact_dir=str(tmp_path),
        tokenizer="esmc",
        encoder="ppi_fusion",
        model="bi_cross_attn+mlp",
        classifier="mlp",
        pair_strategy="two_stream",
        interaction_level="token",
        interaction_model="bi_cross_attn",
        training_mode="end_to_end",
        max_length=2,
        batch_size=2,
        epochs=1,
        lr=0.001,
        dropout=0.0,
        weight_decay=0.0,
        grad_clip_norm=0.0,
        early_stop_patience=0,
        early_stop_min_delta=0.0,
        report_aupr=False,
        esmc_model="models/esmc",
        _resolved_pretrained_model_source="models/esmc",
        _resolved_pretrained_model_identity={
            "provider": "EvolutionaryScale",
            "model_id": "EvolutionaryScale/esmc-600m-2024-12",
            "weight_sha256": "a" * 64,
            "runtime_version": "3.2.0",
            "pooling": "residue_mean",
        },
        _resolved_pretrained_runtime_environment={
            "python_version": "3.10.0",
            "torch_version": "2.5.1+cu121",
            "esm_version": "3.2.0",
        },
    )
    peptide = torch.randn(4, 2, 3)
    partner = torch.randn(4, 2, 3)
    labels = torch.tensor([[1], [1], [0], [0]], dtype=torch.float)
    original_cat = pair_pipeline.torch.cat

    def reject_full_token_concat(tensors, dim=0, *cat_args, **cat_kwargs):
        items = list(tensors)
        if (
            dim == 1
            and len(items) == 2
            and all(item.ndim == 3 and item.shape[0] == 4 for item in items)
        ):
            raise AssertionError("full homogeneous token streams were concatenated")
        return original_cat(items, dim=dim, *cat_args, **cat_kwargs)

    monkeypatch.setattr(pair_pipeline.torch, "cat", reject_full_token_concat)
    train_cross_attention_model(
        args,
        None,
        peptide,
        labels,
        peptide.clone(),
        labels.clone(),
        device="cpu",
        train_partner_x=partner,
        val_partner_x=partner.clone(),
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["pair"]["token_dim"] == 3
    assert metadata["pair"]["seq_len"] == 2
    assert metadata["pretrained"]["tokenizer"] == "esmc"
    assert metadata["pretrained"]["model_identity"]["runtime_version"] == "3.2.0"
    assert metadata["runtime_environment"]["esm_version"] == "3.2.0"


def test_pair_token_bi_cross_attention_artifact_saves_metadata(tmp_path):
    from scripts.train_pair_pipeline import PairBiCrossAttentionClassifier

    args = Namespace(
        task_type="pep_protein_interaction",
        artifact_dir=str(tmp_path),
        tokenizer="esm2",
        encoder="ppi_fusion",
        model="bi_cross_attn+mlp",
        classifier="mlp",
        pair_strategy="two_stream",
        interaction_level="token",
        interaction_model="bi_cross_attn",
        training_mode="end_to_end",
        max_length=2,
        batch_size=2,
        dropout=0.0,
        esm2_model="models/esm2",
        _resolved_pretrained_model_source="models/esm2",
    )
    model = PairBiCrossAttentionClassifier(
        seq_len=2,
        encoder_name="ppi_fusion",
        in_dim=2,
        hidden_dim=4,
        dropout=0.0,
        n_heads=2,
    )
    _set_binary_operating_point(args, 0.45)
    save_pair_token_nn_artifact(
        args, model, torch.ones((2, 4, 2), dtype=torch.float), threshold=0.45
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    payload = torch.load(tmp_path / "model.pt", map_location="cpu", weights_only=False)
    assert metadata["bundle_type"] == "pair_token_nn"
    assert metadata["pretrained"]["tokenizer"] == "esm2"
    assert metadata["pair"]["interaction_model"] == "bi_cross_attn"
    assert metadata["pair"]["token_dim"] == 2
    assert metadata["predictor_contract"]["implementation"] == (
        "pepplatform_pair_token_bidirectional_cross_attention"
    )
    assert metadata["predictor_contract"]["attention_parameterization"] == (
        "independent parameters for each direction"
    )
    assert payload["model_class"] == "PairBiCrossAttentionClassifier"
    assert not any(key.startswith("attn.") for key in payload["state_dict"])


def test_pair_token_classic_pssm_artifact_records_profile_generation(tmp_path, monkeypatch):
    import scripts.train_pair_pipeline as pair_pipeline
    from scripts.train_pair_pipeline import PairBiCrossAttentionClassifier

    monkeypatch.setattr(
        pair_pipeline, "_build_pair_profile_blast_db", lambda subject_fasta, db_prefix: False
    )
    subject_fasta_sha256 = hashlib.sha256(
        b">TRAIN_000000\nAC\n>TRAIN_000001\nGG\n>TRAIN_000002\nAA\n>TRAIN_000003\nCC\n"
    ).hexdigest()
    subject_source = tmp_path / "ppi_pssm" / "subjects" / "test.fasta"
    subject_source.parent.mkdir(parents=True)
    subject_source.write_bytes(
        b">TRAIN_000000\nAC\n>TRAIN_000001\nGG\n>TRAIN_000002\nAA\n>TRAIN_000003\nCC\n"
    )
    args = Namespace(
        task_type="pep_protein_interaction",
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="pssm",
        model="bi_cross_attn+mlp",
        classifier="mlp",
        pair_strategy="two_stream",
        interaction_level="token",
        interaction_model="bi_cross_attn",
        training_mode="end_to_end",
        max_length=1,
        k=1,
        batch_size=2,
        dropout=0.0,
        pssm_index_path=str(tmp_path / "ppi_pssm" / "index.json"),
        psfm_index_path=str(tmp_path / "ppi_psfm" / "index.json"),
        profile_iterations=2,
        profile_evalue=0.01,
        profile_inclusion_evalue=0.001,
        profile_threads=1,
        profile_on_missing="error",
        _profile_manifest_sha256="d" * 64,
        _profile_manifest_provenance=_profile_provenance(subject_fasta_sha256=subject_fasta_sha256),
    )
    model = PairBiCrossAttentionClassifier(
        seq_len=1,
        encoder_name="pssm",
        in_dim=20,
        hidden_dim=4,
        dropout=0.0,
        n_heads=2,
    )
    _set_binary_operating_point(args, 0.4)
    save_pair_token_nn_artifact(
        args,
        model,
        torch.ones((2, 2, 20), dtype=torch.float),
        threshold=0.4,
        profile_rows=[
            ("row_1", "AC", "GG", 1),
            ("row_2", "AA", "CC", 0),
        ],
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bundle_type"] == "pair_token_nn"
    assert metadata["classic"]["tokenizer"] == "kmer"
    assert metadata["classic"]["encoder"] == "pssm"
    assert metadata["pair"]["interaction_model"] == "bi_cross_attn"
    assert metadata["profile_generation"]["strategy"] == "psiblast_subject_fasta"
    assert metadata["profile_generation"]["feature"] == "pssm"
    assert metadata["profile_generation"]["iterations"] == 2
    exported_subject = tmp_path / metadata["profile_generation"]["subject_fasta"]
    assert exported_subject.read_bytes() == subject_source.read_bytes()
    assert metadata["profile_generation"]["db_scope"] == "train_only"


def test_pair_vector_classical_artifact_saves_metadata(tmp_path):
    args = _pair_vector_args(tmp_path, "logreg")
    args.reduction = "none"
    args.ml_scaler = "none"
    args.ml_reduction = "none"
    args.logreg_c = 1.0
    args.logreg_max_iter = 100
    args.logreg_class_weight = "none"
    args.report_aupr = False
    configure_pair_vector_artifacts(args, torch.ones(4, 2), torch.ones(4, 2), torch.ones(4, 8))
    trainer = ClassicalMLTrainer(args)
    train = TensorDataset(
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            ],
            dtype=torch.float,
        ),
        torch.tensor([[0], [0], [1], [1]], dtype=torch.float),
    )
    val = TensorDataset(
        torch.tensor([[0.0] * 8, [1.0] * 8], dtype=torch.float),
        torch.tensor([[0], [1]], dtype=torch.float),
    )

    trainer.run_logreg(train, val)

    payload = joblib.load(tmp_path / "model.joblib")
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert payload["estimator_kind"] == "probability_ensemble"
    assert metadata["bundle_type"] == "pair_vector_classical"
    assert metadata["supports_inference"] is True
    assert metadata["input_schema"] == "peptide_protein_pairs"
    assert isinstance(metadata["threshold"], float)
    assert metadata["threshold_source"] == "validation_f1"


def test_hlaii_pair_vector_classical_artifact_saves_hla_schema(tmp_path):
    args = _pair_vector_args(tmp_path, "logreg", task_type="pep_hlaii")
    args.reduction = "none"
    args.ml_scaler = "none"
    args.ml_reduction = "none"
    args.logreg_c = 1.0
    args.logreg_max_iter = 100
    args.logreg_class_weight = "none"
    args.report_aupr = False
    configure_pair_vector_artifacts(args, torch.ones(4, 2), torch.ones(4, 2), torch.ones(4, 8))
    train, val = _binary_train_val(8)

    ClassicalMLTrainer(args).run_logreg(train, val)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bundle_type"] == "pair_vector_classical"
    assert metadata["task_type"] == "pep_hlaii"
    assert metadata["input_schema"] == "peptide_hla_pairs"
    assert metadata["supports_inference"] is True
    assert metadata["threshold_source"] == "validation_f1"


def test_hlaii_pair_embedding_classical_artifact_saves_hla_schema(tmp_path):
    args = _pair_embedding_classical_args(tmp_path, "pep_hlaii", "peptide_hla_pairs")
    train, val = _binary_train_val(2)

    ClassicalMLTrainer(args).run_logreg(train, val)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert (tmp_path / "model.joblib").exists()
    assert metadata["bundle_type"] == "pair_embedding_classical"
    assert metadata["task_type"] == "pep_hlaii"
    assert metadata["input_schema"] == "peptide_hla_pairs"
    assert metadata["supports_inference"] is True
    assert metadata["embedding_export"]["downstream"] == "logreg"
    assert metadata["threshold_source"] == "validation_f1"


def test_hlai_pair_embedding_classical_artifact_saves_hlai_schema(tmp_path):
    args = _pair_embedding_classical_args(tmp_path, "pep_hlai", "peptide_hlai_pairs")
    train, val = _binary_train_val(2)

    ClassicalMLTrainer(args).run_logreg(train, val)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bundle_type"] == "pair_embedding_classical"
    assert metadata["task_type"] == "pep_hlai"
    assert metadata["input_schema"] == "peptide_hlai_pairs"
    assert metadata["supports_inference"] is True
    assert metadata["embedding_export"]["downstream"] == "logreg"
    assert metadata["threshold_source"] == "validation_f1"


def test_pair_vector_classic_glove_rf_artifact_saves_metadata(tmp_path):
    args = _pair_vector_args(tmp_path, "rf")
    args.tokenizer = "kmer"
    args.encoder = "glove"
    args.k = 1
    args.max_length = 2
    args.glove_path = str(tmp_path / "glove.vec")
    args.reduction = "none"
    args.ml_scaler = "none"
    args.ml_reduction = "none"
    args.rf_estimators = 2
    args.rf_max_depth = 0
    args.rf_min_samples_leaf = 1
    args.report_aupr = False
    args.ml_auto_tune = False
    configure_pair_vector_artifacts(args, torch.ones(4, 2), torch.ones(4, 2), torch.ones(4, 8))
    trainer = ClassicalMLTrainer(args)
    train = TensorDataset(
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            ],
            dtype=torch.float,
        ),
        torch.tensor([[0], [0], [1], [1]], dtype=torch.float),
    )
    val = TensorDataset(
        torch.tensor([[0.0] * 8, [1.0] * 8], dtype=torch.float),
        torch.tensor([[0], [1]], dtype=torch.float),
    )

    trainer.run_rf(train, val)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bundle_type"] == "pair_vector_classical"
    assert metadata["supports_inference"] is True
    assert "pretrained" not in metadata
    assert metadata["classic"]["tokenizer"] == "kmer"
    assert metadata["classic"]["encoder"] == "glove"
    assert metadata["classic"]["glove_path"] == str(tmp_path / "glove.vec")
    assert metadata["pair"]["fusion_mode"] == "concat_all"
    assert metadata["predictor_contract"]["algorithm"] == "random_forest"
    assert metadata["predictor_contract"]["input_contract"] == "fixed_width_global_vector"
    assert metadata["predictor_contract"]["multioutput"] == "native_sklearn_multi_output"


def test_pair_embedding_rf_artifact_saves_downstream_predictor_contract(tmp_path):
    args = _pair_embedding_classical_args(
        tmp_path, "pep_protein_interaction", "peptide_protein_pairs"
    )
    args.model = "rf"
    args.classifier = "rf"
    args.rf_estimators = 3
    args.rf_max_depth = 0
    args.rf_min_samples_leaf = 1
    args.ml_auto_tune = False
    args._artifact_info["embedding_export"]["downstream"] = "rf"
    train, val = _binary_train_val(2)

    ClassicalMLTrainer(args).run_rf(train, val)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bundle_type"] == "pair_embedding_classical"
    assert metadata["embedding_export"]["downstream"] == "rf"
    assert metadata["predictor_contract"]["algorithm"] == "random_forest"
    assert metadata["predictor_contract"]["estimator_params"]["n_estimators"] == 3


def test_pair_vector_extratrees_artifact_saves_predictor_contract(tmp_path):
    args = _pair_vector_args(tmp_path, "extratrees")
    args.reduction = "none"
    args.ml_scaler = "none"
    args.ml_reduction = "none"
    args.ml_auto_tune = False
    args.et_estimators = 3
    args.et_max_depth = 0
    args.et_min_samples_leaf = 1
    args.report_aupr = False
    configure_pair_vector_artifacts(args, torch.ones(4, 2), torch.ones(4, 2), torch.ones(4, 8))
    train, val = _binary_train_val(8)

    ClassicalMLTrainer(args).run_extratrees(train, val)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bundle_type"] == "pair_vector_classical"
    assert metadata["predictor_contract"]["algorithm"] == "extra_trees"
    assert metadata["predictor_contract"]["input_feature_count"] == 8
    assert metadata["predictor_contract"]["estimator_params"]["bootstrap"] is False


def test_pair_embedding_extratrees_artifact_saves_predictor_contract(tmp_path):
    args = _pair_embedding_classical_args(
        tmp_path, "pep_protein_interaction", "peptide_protein_pairs"
    )
    args.model = "extratrees"
    args.classifier = "extratrees"
    args.ml_auto_tune = False
    args.et_estimators = 3
    args.et_max_depth = 0
    args.et_min_samples_leaf = 1
    args._artifact_info["embedding_export"]["downstream"] = "extratrees"
    train, val = _binary_train_val(2)

    ClassicalMLTrainer(args).run_extratrees(train, val)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bundle_type"] == "pair_embedding_classical"
    assert metadata["embedding_export"]["downstream"] == "extratrees"
    assert metadata["predictor_contract"]["algorithm"] == "extra_trees"
    assert metadata["predictor_contract"]["input_feature_count"] == 2


def test_sequence_embedding_rf_artifact_keeps_backbone_and_classifier_contracts(tmp_path):
    args = Namespace(
        task_type="pep_function",
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="onehot",
        model="cnn",
        sequence_model="cnn",
        classifier="rf",
        training_mode="export_embeddings",
        reduction="none",
        ml_scaler="none",
        ml_reduction="none",
        ml_auto_tune=False,
        rf_estimators=3,
        rf_max_depth=0,
        rf_min_samples_leaf=1,
        report_aupr=False,
        _artifact_source="sequence_embedding_features",
        _artifact_info={
            "input_schema": "peptide_sequences",
            "label_names": ["active"],
            "embedding_export": {
                "encoder_path": "embedding_encoder.pt",
                "embedding_dim": 2,
                "downstream": "rf",
                "architecture_contract": {
                    "implementation": "pepplatform_multiscale_residual_cnn",
                    "contract_version": 2,
                },
            },
        },
    )
    train, val = _binary_train_val(2)

    ClassicalMLTrainer(args).run_rf(train, val)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bundle_type"] == "sequence_embedding_classical"
    assert metadata["predictor_contract"]["algorithm"] == "random_forest"
    assert metadata["embedding_export"]["architecture_contract"] == {
        "implementation": "pepplatform_multiscale_residual_cnn",
        "contract_version": 2,
    }


def test_sequence_embedding_extratrees_keeps_backbone_and_classifier_contracts(
    tmp_path,
):
    args = Namespace(
        task_type="pep_function",
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="onehot",
        model="cnn",
        sequence_model="cnn",
        classifier="extratrees",
        training_mode="export_embeddings",
        reduction="none",
        ml_scaler="none",
        ml_reduction="none",
        ml_auto_tune=False,
        et_estimators=3,
        et_max_depth=0,
        et_min_samples_leaf=1,
        report_aupr=False,
        _artifact_source="sequence_embedding_features",
        _artifact_info={
            "input_schema": "peptide_sequences",
            "label_names": ["active"],
            "embedding_export": {
                "encoder_path": "embedding_encoder.pt",
                "embedding_dim": 2,
                "downstream": "extratrees",
                "architecture_contract": {
                    "implementation": "pepplatform_multiscale_residual_cnn",
                    "contract_version": 2,
                },
            },
        },
    )
    train, val = _binary_train_val(2)

    ClassicalMLTrainer(args).run_extratrees(train, val)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bundle_type"] == "sequence_embedding_classical"
    assert metadata["predictor_contract"]["algorithm"] == "extra_trees"
    assert metadata["predictor_contract"]["input_feature_count"] == 2
    assert metadata["embedding_export"]["architecture_contract"] == {
        "implementation": "pepplatform_multiscale_residual_cnn",
        "contract_version": 2,
    }


def test_sequence_embedding_svm_artifact_keeps_backbone_and_classifier_contracts(tmp_path):
    args = Namespace(
        task_type="pep_function",
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="onehot",
        model="cnn",
        sequence_model="cnn",
        classifier="svm",
        training_mode="export_embeddings",
        reduction="none",
        ml_scaler="standard",
        ml_reduction="none",
        ml_auto_tune=False,
        svm_c=1.0,
        svm_class_weight="none",
        svm_kernel="linear",
        svm_gamma="scale",
        svm_max_iter=5000,
        svm_tol=1e-4,
        seed=42,
        report_aupr=False,
        _artifact_source="sequence_embedding_features",
        _artifact_info={
            "input_schema": "peptide_sequences",
            "label_names": ["active"],
            "embedding_export": {
                "encoder_path": "embedding_encoder.pt",
                "embedding_dim": 2,
                "downstream": "svm",
                "architecture_contract": {
                    "implementation": "pepplatform_multiscale_residual_cnn",
                    "contract_version": 2,
                },
            },
        },
    )
    train, val = _binary_train_val(2)

    ClassicalMLTrainer(args).run_svm(train, val)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bundle_type"] == "sequence_embedding_classical"
    assert metadata["predictor_contract"]["algorithm"] == "support_vector_machine"
    assert metadata["predictor_contract"]["score_kind"] == "margin"
    assert metadata["embedding_export"]["architecture_contract"] == {
        "implementation": "pepplatform_multiscale_residual_cnn",
        "contract_version": 2,
    }


def test_pair_vector_classic_pssm_artifact_records_profile_generation(tmp_path, monkeypatch):
    import scripts.train_pair_pipeline as pair_pipeline

    monkeypatch.setattr(
        pair_pipeline, "_build_pair_profile_blast_db", lambda subject_fasta, db_prefix: False
    )
    subject_fasta_sha256 = hashlib.sha256(
        b">TRAIN_000000\nAC\n>TRAIN_000001\nGG\n>TRAIN_000002\nAA\n>TRAIN_000003\nCC\n"
    ).hexdigest()
    subject_source = tmp_path / "ppi_pssm" / "subjects" / "test.fasta"
    subject_source.parent.mkdir(parents=True)
    subject_source.write_bytes(
        b">TRAIN_000000\nAC\n>TRAIN_000001\nGG\n>TRAIN_000002\nAA\n>TRAIN_000003\nCC\n"
    )
    args = _pair_vector_args(tmp_path, "rf")
    args.tokenizer = "kmer"
    args.encoder = "pssm"
    args.k = 1
    args.max_length = 1
    args.pssm_index_path = str(tmp_path / "ppi_pssm" / "index.json")
    args.psfm_index_path = str(tmp_path / "ppi_psfm" / "index.json")
    args.profile_iterations = 2
    args.profile_evalue = 0.01
    args.profile_inclusion_evalue = 0.001
    args.profile_threads = 1
    args.profile_on_missing = "error"
    args._profile_manifest_sha256 = "d" * 64
    args._profile_manifest_provenance = _profile_provenance(
        subject_fasta_sha256=subject_fasta_sha256
    )
    args.reduction = "none"
    args.ml_scaler = "none"
    args.ml_reduction = "none"
    args.rf_estimators = 2
    args.rf_max_depth = 0
    args.rf_min_samples_leaf = 1
    args.report_aupr = False
    args.ml_auto_tune = False
    configure_pair_vector_artifacts(
        args,
        torch.ones(4, 20),
        torch.ones(4, 20),
        torch.ones(4, 80),
        profile_rows=[
            ("row_1", "AC", "GG", 1),
            ("row_2", "AA", "CC", 0),
        ],
    )
    trainer = ClassicalMLTrainer(args)
    train = TensorDataset(
        torch.tensor(
            [
                [0.0] * 80,
                [0.0] * 40 + [1.0] * 40,
                [1.0] * 40 + [0.0] * 40,
                [1.0] * 80,
            ],
            dtype=torch.float,
        ),
        torch.tensor([[0], [0], [1], [1]], dtype=torch.float),
    )
    val = TensorDataset(
        torch.tensor([[0.0] * 80, [1.0] * 80], dtype=torch.float),
        torch.tensor([[0], [1]], dtype=torch.float),
    )

    trainer.run_rf(train, val)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bundle_type"] == "pair_vector_classical"
    assert metadata["supports_inference"] is True
    assert metadata["classic"]["encoder"] == "pssm"
    assert metadata["classic"]["pssm_index_path"] == str(tmp_path / "ppi_pssm" / "index.json")
    assert metadata["profile_generation"]["strategy"] == "psiblast_subject_fasta"
    assert metadata["profile_generation"]["feature"] == "pssm"
    assert metadata["profile_generation"]["iterations"] == 2
    assert metadata["profile_generation"]["on_missing"] == "error"
    exported_subject = tmp_path / metadata["profile_generation"]["subject_fasta"]
    assert exported_subject.read_bytes() == subject_source.read_bytes()
    assert metadata["profile_generation"]["db_scope"] == "train_only"


def test_pair_vector_svm_artifact_saves_margin_threshold(tmp_path):
    args = _pair_vector_args(tmp_path, "svm")
    args.reduction = "none"
    args.ml_scaler = "none"
    args.ml_reduction = "none"
    args.svm_c = 1.0
    args.svm_class_weight = "none"
    args.svm_kernel = "linear"
    args.svm_gamma = "scale"
    args.svm_max_iter = 5000
    args.svm_tol = 1e-4
    args.report_aupr = False
    args.ml_auto_tune = False
    configure_pair_vector_artifacts(args, torch.ones(4, 2), torch.ones(4, 2), torch.ones(4, 8))
    trainer = ClassicalMLTrainer(args)
    train = TensorDataset(
        torch.tensor(
            [
                [-2.0, -2.0, -2.0, -2.0, 0.0, 0.0, 4.0, 4.0],
                [-1.5, -1.0, -1.5, -1.0, 0.0, 0.0, 2.25, 1.0],
                [1.0, 1.5, 1.0, 1.5, 0.0, 0.0, 1.0, 2.25],
                [2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 4.0, 4.0],
            ],
            dtype=torch.float,
        ),
        torch.tensor([[0], [0], [1], [1]], dtype=torch.float),
    )
    val = TensorDataset(
        torch.tensor(
            [[-1.8, -1.8, -1.8, -1.8, 0.0, 0.0, 3.24, 3.24], [1.8] * 8], dtype=torch.float
        ),
        torch.tensor([[0], [1]], dtype=torch.float),
    )

    trainer.run_svm(train, val)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bundle_type"] == "pair_vector_classical"
    assert metadata["threshold_source"] == "validation_f1"
    assert isinstance(metadata["threshold"], float)
    assert metadata["predictor_contract"]["algorithm"] == "support_vector_machine"
    assert metadata["predictor_contract"]["status"] == "verified_platform_variant"
    assert metadata["predictor_contract"]["score_kind"] == "margin"
    assert metadata["predictor_contract"]["calibrated"] is False
