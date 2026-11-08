import json
from types import SimpleNamespace

import pytest
import torch

import pepplatform.encoders as encoders
import pepplatform.tokenizers as tokenizers
from pepplatform.encoders import EsmEncoder, ResidueEncoding
from pepplatform.esm2 import (
    ESM2_CONFIG_SIGNATURE,
    ESM2_OFFICIAL_MODEL_ID,
    ESM2_REPRESENTATION_CONTRACT_VERSION,
    esm2_model_identity,
    inspect_esm2_source,
    is_complete_esm2_snapshot,
    validate_esm2_config,
    validate_esm2_model_identity,
)
from pepplatform.models import ResidueModelInput
from pepplatform.web.inference import ModelService


class _FakeEsmTokenizer:
    def num_special_tokens_to_add(self, pair=False):
        assert pair is False
        return 2

    def __call__(self, sequences, **kwargs):
        self.last_call = (list(sequences), kwargs)
        rows = []
        masks = []
        for sequence in sequences:
            ids = [1] + [ord(char) for char in sequence] + [2]
            rows.append(ids)
            masks.append([1] + [0] * len(sequence) + [1])
        width = max(len(row) for row in rows)
        input_ids = torch.zeros((len(rows), width), dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        special = torch.ones_like(input_ids)
        for index, (ids, mask) in enumerate(zip(rows, masks)):
            input_ids[index, : len(ids)] = torch.tensor(ids)
            attention[index, : len(ids)] = 1
            special[index, : len(mask)] = torch.tensor(mask)
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "special_tokens_mask": special,
        }


def test_esm2_tokenizer_uses_residue_budget_and_normalizes(monkeypatch):
    class _Factory:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            assert model_name == "test-esm2"
            assert kwargs == {}
            return _FakeEsmTokenizer()

    monkeypatch.setattr(tokenizers, "AutoTokenizer", _Factory)
    tokenizer = tokenizers.EsmTokenizer(model_name="test-esm2", max_length=3)
    result = tokenizer.batch_tokenize(["a cde", " G "])

    assert result["input_ids"].shape == (2, 5)
    assert result["special_tokens_mask"].tolist() == [
        [1, 0, 0, 0, 1],
        [1, 0, 1, 1, 1],
    ]
    assert tokenizer.tokenizer.last_call[0] == ["ACD", "G"]
    assert tokenizer.tokenizer.last_call[1]["max_length"] == 5


def test_esm2_legacy_tokenizer_preserves_total_token_budget(monkeypatch):
    class _Factory:
        @staticmethod
        def from_pretrained(_model_name, **_kwargs):
            return _FakeEsmTokenizer()

    monkeypatch.setattr(tokenizers, "AutoTokenizer", _Factory)
    tokenizer = tokenizers.EsmTokenizer(
        model_name="test-esm2",
        max_length=5,
        legacy_total_token_length=True,
    )
    tokenizer.batch_tokenize(["a cde"])
    sequences, kwargs = tokenizer.tokenizer.last_call
    assert sequences == ["a cde"]
    assert kwargs["max_length"] == 5


def test_esm2_encoder_filters_special_mask_and_pools_residues_only():
    hidden = torch.arange(1 * 5 * 2, dtype=torch.float).reshape(1, 5, 2)

    class _Model:
        def __call__(self, **kwargs):
            assert "special_tokens_mask" not in kwargs
            return SimpleNamespace(last_hidden_state=hidden)

    encoder = object.__new__(EsmEncoder)
    encoder.device = torch.device("cpu")
    encoder.model = _Model()
    inputs = {
        "input_ids": torch.ones((1, 5), dtype=torch.long),
        "attention_mask": torch.ones((1, 5), dtype=torch.long),
        "special_tokens_mask": torch.tensor([[1, 0, 0, 0, 1]], dtype=torch.long),
    }

    pooled = encoder.encode(inputs)
    legacy = encoder.encode_legacy_attention_mask_mean(inputs)
    assert torch.equal(pooled, hidden[:, 1:4].mean(dim=1))
    assert torch.equal(legacy, hidden.mean(dim=1))


def test_esm2_identity_and_snapshot_contract(tmp_path):
    source = tmp_path / "snapshot"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(ESM2_CONFIG_SIGNATURE), encoding="utf-8")
    for filename in ("tokenizer_config.json", "vocab.txt", "tokenizer.json"):
        (source / filename).write_text("ok", encoding="utf-8")
    (source / "model.safetensors").write_bytes(b"weights")

    assert is_complete_esm2_snapshot(source)
    facts = inspect_esm2_source(source)
    assert facts["complete"] is True
    assert facts["provenance_status"] == "external_artifact_unverified"
    identity = esm2_model_identity(
        source, config=ESM2_CONFIG_SIGNATURE, weight_path=source / "model.safetensors"
    )
    assert identity["model_id"] == ESM2_OFFICIAL_MODEL_ID
    assert identity["representation_contract_version"] == ESM2_REPRESENTATION_CONTRACT_VERSION
    with pytest.raises(ValueError, match="architecture signature mismatch"):
        validate_esm2_config(dict(ESM2_CONFIG_SIGNATURE, hidden_size=1))
    with pytest.raises(ValueError, match="identity differs"):
        validate_esm2_model_identity(identity, dict(identity, weight_sha256="0" * 64))


def test_esm2_residue_inference_uses_normalized_lengths(monkeypatch):
    lengths = []

    class _Tokenizer:
        @staticmethod
        def normalize_sequence(sequence):
            return "".join(str(sequence).split()).upper()

        def batch_tokenize_residues(self, sequences, max_length, device=None):
            return {"sequences": list(sequences), "max_length": max_length}

    class _Encoder:
        def encode_residue_batch(self, inputs, sequence_lengths, max_length):
            lengths.extend(sequence_lengths)
            features = torch.zeros((len(sequence_lengths), max_length, 2))
            mask = torch.zeros((len(sequence_lengths), max_length), dtype=torch.bool)
            for row, length in enumerate(sequence_lengths):
                mask[row, :length] = True
            return ResidueEncoding(features, mask)

    monkeypatch.setattr(
        ModelService,
        "_load_esm2_for_inference",
        staticmethod(lambda *_args: (_Tokenizer(), _Encoder(), False)),
    )
    result = ModelService._extract_pretrained_residue_features(
        {
            "pretrained": {
                "tokenizer": "esm2",
                "model_name": "test-esm2",
                "max_length": 3,
                "model_identity": {
                    "pooling": "residue_mean",
                    "representation_contract_version": 1,
                },
            },
            "args": {"batch_size": 2},
        },
        ["a cde", "GG"],
        torch.device("cpu"),
    )
    assert isinstance(result, ResidueModelInput)
    assert lengths == [3, 2]
    assert result.mask.tolist() == [[True, True, True], [True, True, False]]
