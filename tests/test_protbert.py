from types import SimpleNamespace

import pytest
import torch

from pepplatform.encoders import ProtBertEncoder
from pepplatform.protbert import (
    PROTBERT_CONFIG_SIGNATURE,
    PROTBERT_POOLING,
    PROTBERT_REPRESENTATION_CONTRACT_VERSION,
    is_complete_protbert_snapshot,
    normalize_protbert_sequence,
    protbert_model_identity,
    validate_protbert_config,
    validate_protbert_max_residues,
)
from pepplatform.tokenizers import ProtBertTokenizer


def test_protbert_normalization_matches_prottrans_contract():
    assert normalize_protbert_sequence(" a c\nU Z O B x ") == "ACXXXXX"
    with pytest.raises(ValueError, match="at least one residue"):
        normalize_protbert_sequence("  \n")
    with pytest.raises(ValueError, match="unsupported residues"):
        normalize_protbert_sequence("AC-1")


def test_protbert_max_length_is_a_residue_budget():
    assert validate_protbert_max_residues(1) == 1
    assert validate_protbert_max_residues(39998) == 39998
    tokenizer = object.__new__(ProtBertTokenizer)
    assert tokenizer.residue_length(" A C U ", 2) == 2
    with pytest.raises(ValueError):
        validate_protbert_max_residues(39999)


def test_protbert_pooling_excludes_cls_sep_and_padding():
    hidden = torch.tensor([[[10.0], [1.0], [3.0], [9.0], [0.0]]])
    encoder = object.__new__(ProtBertEncoder)
    encoder.device = torch.device("cpu")
    encoder.model = lambda **_kwargs: SimpleNamespace(last_hidden_state=hidden)
    inputs = {
        "input_ids": torch.ones(1, 5, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
        "special_tokens_mask": torch.tensor([[1, 0, 0, 1, 1]]),
    }

    pooled = encoder.encode(inputs)
    legacy = encoder.encode_legacy_attention_mask_mean(inputs)

    assert torch.equal(pooled, torch.tensor([[2.0]]))
    assert torch.equal(legacy, torch.tensor([[23.0 / 4.0]]))


def test_protbert_identity_rejects_wrong_architecture(tmp_path):
    config = dict(PROTBERT_CONFIG_SIGNATURE)
    assert validate_protbert_config(config) == config
    legacy_hf_config = dict(config)
    legacy_hf_config.pop("model_type")
    legacy_hf_config["architectures"] = ["BertForMaskedLM"]
    assert validate_protbert_config(legacy_hf_config)["model_type"] == "bert"
    wrong = dict(config, hidden_size=768)
    with pytest.raises(ValueError, match="architecture signature mismatch"):
        validate_protbert_config(wrong)

    snapshot = tmp_path / "protbert"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    assert not is_complete_protbert_snapshot(snapshot)


def test_protbert_identity_records_external_artifact_provenance(tmp_path):
    snapshot = tmp_path / "protbert"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        '{"model_type":"bert","hidden_size":1024,"num_attention_heads":16,'
        '"num_hidden_layers":30,"intermediate_size":4096,"vocab_size":30,'
        '"max_position_embeddings":40000}',
        encoding="utf-8",
    )
    (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (snapshot / "vocab.txt").write_text("[PAD]\n", encoding="utf-8")
    weight = snapshot / "pytorch_model.bin"
    weight.write_bytes(b"external test weights")

    identity = protbert_model_identity(str(snapshot))

    assert identity["model_id"] == "Rostlab/prot_bert"
    assert identity["pooling"] == PROTBERT_POOLING
    assert identity["representation_contract_version"] == PROTBERT_REPRESENTATION_CONTRACT_VERSION
    assert identity["provenance_status"] == "external_artifact_unverified"
    assert identity["weight_sha256"]
