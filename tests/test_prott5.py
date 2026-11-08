import json
import os
from types import SimpleNamespace

import pytest
import torch

from pepplatform.encoders import ProT5Encoder, ResidueEncoding
from pepplatform.prott5 import (
    PROTT5_CONFIG_SIGNATURE,
    PROTT5_HF_REVISION,
    PROTT5_MAX_RESIDUES,
    PROTT5_N_POSITIONS,
    PROTT5_OFFICIAL_MODEL_ID,
    PROTT5_POOLING,
    PROTT5_REPRESENTATION_CONTRACT_VERSION,
    PROTT5_D_MODEL,
    PROTT5_VOCAB_SIZE,
    inspect_prott5_source,
    is_complete_prott5_snapshot,
    normalize_prott5_sequence,
    prott5_model_identity,
    validate_prott5_config,
    validate_prott5_max_residues,
    weight_manifest_sha256,
)
from pepplatform.tokenizers import ProT5Tokenizer
from pepplatform.models import ResidueModelInput
from pepplatform.web.inference import ModelService


# ── Normalization ────────────────────────────────────────────────────────────


def test_prott5_normalization_strips_whitespace_and_uppercases():
    assert normalize_prott5_sequence(" a c\nd E f ") == "ACDEF"


def test_prott5_normalization_maps_UZOB_to_X():
    assert normalize_prott5_sequence("AUZOBX") == "AXXXXX"


def test_prott5_normalization_rejects_empty():
    with pytest.raises(ValueError, match="at least one residue"):
        normalize_prott5_sequence("  \n\t")


def test_prott5_normalization_rejects_unsupported_residues():
    with pytest.raises(ValueError, match="unsupported residues"):
        normalize_prott5_sequence("AC-1")
    with pytest.raises(ValueError, match="unsupported residues"):
        normalize_prott5_sequence("AC9")


# ── Max residues validation ──────────────────────────────────────────────────


def test_prott5_max_residues_accepts_valid_integers():
    assert validate_prott5_max_residues(1) == 1
    assert validate_prott5_max_residues(PROTT5_MAX_RESIDUES) == PROTT5_MAX_RESIDUES
    assert validate_prott5_max_residues(300) == 300


def test_prott5_max_residues_rejects_boolean():
    with pytest.raises(ValueError, match="must be an integer"):
        validate_prott5_max_residues(True)


def test_prott5_max_residues_rejects_out_of_range():
    with pytest.raises(ValueError, match="must be between"):
        validate_prott5_max_residues(0)
    with pytest.raises(ValueError, match="must be between"):
        validate_prott5_max_residues(PROTT5_MAX_RESIDUES + 1)


# ── Config signature ─────────────────────────────────────────────────────────


def test_prott5_config_signature_is_t5_xl():
    assert PROTT5_CONFIG_SIGNATURE["model_type"] == "t5"
    assert PROTT5_CONFIG_SIGNATURE["d_model"] == PROTT5_D_MODEL
    assert PROTT5_CONFIG_SIGNATURE["num_heads"] == 32
    assert PROTT5_CONFIG_SIGNATURE["num_layers"] == 24
    assert PROTT5_CONFIG_SIGNATURE["vocab_size"] == PROTT5_VOCAB_SIZE
    assert PROTT5_CONFIG_SIGNATURE["n_positions"] == PROTT5_N_POSITIONS


def test_prott5_validate_config_accepts_correct_signature():
    config = dict(PROTT5_CONFIG_SIGNATURE)
    validated = validate_prott5_config(config)
    assert validated == config


def test_prott5_validate_config_accepts_t5_for_conditional_generation():
    config = dict(PROTT5_CONFIG_SIGNATURE)
    config["architectures"] = ["T5ForConditionalGeneration"]
    validated = validate_prott5_config(config)
    assert validated["model_type"] == "t5"


def test_prott5_validate_config_accepts_t5_model():
    config = dict(PROTT5_CONFIG_SIGNATURE)
    config["architectures"] = ["T5Model"]
    validated = validate_prott5_config(config)
    assert validated["model_type"] == "t5"


def test_prott5_official_config_accepts_omitted_t5_distance_default(tmp_path):
    config = dict(PROTT5_CONFIG_SIGNATURE, architectures=["T5ForConditionalGeneration"])
    del config["relative_attention_max_distance"]
    validated = validate_prott5_config(config)
    assert validated["relative_attention_max_distance"] == 128
    assert "relative_attention_max_distance" not in config

    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "spiece.model").write_bytes(b"spiece")
    (tmp_path / "pytorch_model.bin").write_bytes(b"weights")
    assert is_complete_prott5_snapshot(tmp_path)


def test_prott5_validate_config_rejects_explicit_wrong_distance():
    wrong = dict(PROTT5_CONFIG_SIGNATURE, relative_attention_max_distance=256)
    with pytest.raises(ValueError, match="relative_attention_max_distance"):
        validate_prott5_config(wrong)


def test_prott5_omitted_default_does_not_fill_missing_xl_dimensions():
    wrong = dict(PROTT5_CONFIG_SIGNATURE)
    del wrong["relative_attention_max_distance"]
    del wrong["d_ff"]
    with pytest.raises(ValueError, match="d_ff"):
        validate_prott5_config(wrong)


def test_prott5_validate_config_rejects_wrong_hidden_size():
    wrong = dict(PROTT5_CONFIG_SIGNATURE, d_model=768)
    with pytest.raises(ValueError, match="architecture signature mismatch"):
        validate_prott5_config(wrong)


def test_prott5_validate_config_rejects_wrong_model_type():
    wrong = dict(PROTT5_CONFIG_SIGNATURE, model_type="bert")
    with pytest.raises(ValueError, match="architecture signature mismatch"):
        validate_prott5_config(wrong)


def test_prott5_validate_config_rejects_non_mapping():
    with pytest.raises(ValueError, match="must be a mapping"):
        validate_prott5_config(None)


# ── Snapshot and provenance ──────────────────────────────────────────────────


def test_prott5_complete_snapshot_is_detected(tmp_path):
    source = tmp_path / "snapshot"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(PROTT5_CONFIG_SIGNATURE), encoding="utf-8")
    for filename in ("tokenizer_config.json", "spiece.model", "tokenizer.json"):
        (source / filename).write_text("ok", encoding="utf-8")
    (source / "model.safetensors").write_bytes(b"weights")

    assert is_complete_prott5_snapshot(source)


def test_prott5_incomplete_snapshot_rejected(tmp_path):
    source = tmp_path / "bad"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(PROTT5_CONFIG_SIGNATURE), encoding="utf-8")
    # missing tokenizer and weight files
    assert not is_complete_prott5_snapshot(source)


def test_prott5_snapshot_requires_all_indexed_shards(tmp_path):
    source = tmp_path / "sharded"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(PROTT5_CONFIG_SIGNATURE), encoding="utf-8")
    (source / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (source / "spiece.model").write_bytes(b"spiece")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"shared.weight": "model-00001-of-00002.safetensors"}}),
        encoding="utf-8",
    )

    assert not is_complete_prott5_snapshot(source)
    (source / "model-00001-of-00002.safetensors").write_bytes(b"weights")
    assert is_complete_prott5_snapshot(source)


def test_prott5_sharded_weight_digest_changes_when_a_shard_changes(tmp_path):
    source = tmp_path / "sharded"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(PROTT5_CONFIG_SIGNATURE), encoding="utf-8")
    (source / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (source / "spiece.model").write_bytes(b"spiece")
    shards = [
        source / "model-00001-of-00002.safetensors",
        source / "model-00002-of-00002.safetensors",
    ]
    for index, shard in enumerate(shards):
        shard.write_bytes(f"weights-{index}".encode("ascii"))
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": shards[0].name, "b": shards[1].name}}),
        encoding="utf-8",
    )
    before = inspect_prott5_source(source)["weight_sha256"]
    shards[1].write_bytes(b"changed")
    after = inspect_prott5_source(source)["weight_sha256"]
    assert after == weight_manifest_sha256(source)
    assert before != after


def test_prott5_inspect_source_records_local_provenance(tmp_path):
    source = tmp_path / "snapshot"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(PROTT5_CONFIG_SIGNATURE), encoding="utf-8")
    for filename in ("tokenizer_config.json", "spiece.model", "tokenizer.json"):
        (source / filename).write_text("ok", encoding="utf-8")
    (source / "model.safetensors").write_bytes(b"weights")

    facts = inspect_prott5_source(source)
    assert facts["complete"] is True
    assert facts["is_local"] is True
    assert facts["source"] == str(source)
    assert facts["provenance_status"] == "external_artifact_unverified"


def test_prott5_identity_returns_external_provenance_for_local_snapshot(tmp_path):
    source = tmp_path / "snapshot"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(PROTT5_CONFIG_SIGNATURE), encoding="utf-8")
    for filename in ("tokenizer_config.json", "spiece.model", "tokenizer.json"):
        (source / filename).write_text("ok", encoding="utf-8")
    weight = source / "pytorch_model.bin"
    weight.write_bytes(b"external test weights")

    identity = prott5_model_identity(str(source))

    assert identity["model_id"] == PROTT5_OFFICIAL_MODEL_ID
    assert identity["pooling"] == PROTT5_POOLING
    assert identity["representation_contract_version"] == PROTT5_REPRESENTATION_CONTRACT_VERSION
    assert identity["provenance_status"] == "external_artifact_unverified"
    assert isinstance(identity["weight_sha256"], str)
    assert len(identity["weight_sha256"]) == 64
    assert identity["resolved_source"] == str(source)
    assert identity["d_model"] == PROTT5_D_MODEL


def test_prott5_identity_pins_official_revision_for_remote():
    identity = prott5_model_identity(
        PROTT5_OFFICIAL_MODEL_ID,
        config=PROTT5_CONFIG_SIGNATURE,
    )
    assert identity["snapshot_revision"] == PROTT5_HF_REVISION
    assert identity["model_id"] == PROTT5_OFFICIAL_MODEL_ID
    assert len(identity["weight_sha256"]) == 64


# ── Tokenizer ────────────────────────────────────────────────────────────────


class _FakeT5Tokenizer:
    def __init__(self):
        self.last_kwargs = None

    def num_special_tokens_to_add(self, pair=False):
        assert pair is False
        return 1  # EOS only for T5

    def __call__(self, sequences, **kwargs):
        self.last_call = (list(sequences), kwargs)
        rows = []
        masks = []
        special_rows = []
        for sequence in sequences:
            residues = sequence.split()
            ids = [ord(char) for char in residues] + [1]  # 1 = EOS
            rows.append(ids)
            masks.append([1] * len(ids))
            special_rows.append([0] * len(residues) + [1])
        width = max(len(row) for row in rows)
        input_ids = torch.zeros((len(rows), width), dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        special = torch.ones_like(input_ids)
        for index, (ids, mask, spec) in enumerate(zip(rows, masks, special_rows)):
            input_ids[index, : len(ids)] = torch.tensor(ids)
            attention[index, : len(ids)] = 1
            special[index, : len(spec)] = torch.tensor(spec)
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "special_tokens_mask": special,
        }


def test_prott5_tokenizer_returns_special_tokens_mask(monkeypatch):
    class _Factory:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            assert model_name == "test-prott5"
            return _FakeT5Tokenizer()

    monkeypatch.setattr("pepplatform.tokenizers.T5Tokenizer", _Factory)
    tokenizer = ProT5Tokenizer(model_name="test-prott5", max_length=4)
    result = tokenizer.batch_tokenize(["A C D E", "G G"])

    assert "special_tokens_mask" in result
    # First: 4 residues + 1 EOS = 5 tokens; Second: 2 residues + 1 EOS = 3 tokens
    # padded to max width (5)
    assert result["input_ids"].shape == (2, 5)
    assert result["special_tokens_mask"].tolist() == [
        [0, 0, 0, 0, 1],
        [0, 0, 1, 1, 1],
    ]


def test_prott5_tokenizer_validates_length_before_loading(monkeypatch):
    calls = []

    class _Factory:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            calls.append(True)
            return _FakeT5Tokenizer()

    monkeypatch.setattr("pepplatform.tokenizers.T5Tokenizer", _Factory)
    with pytest.raises(ValueError, match="must be between"):
        ProT5Tokenizer(model_name="test-prott5", max_length=0)
    assert calls == []


def test_prott5_tokenizer_rejects_empty_batch_item(monkeypatch):
    class _Factory:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            return _FakeT5Tokenizer()

    monkeypatch.setattr("pepplatform.tokenizers.T5Tokenizer", _Factory)
    tokenizer = ProT5Tokenizer(model_name="test-prott5", max_length=4)
    with pytest.raises(ValueError, match="at least one residue"):
        tokenizer.batch_tokenize(["AC", "  \n"])


def test_prott5_tokenizer_uses_residue_budget_plus_eos(monkeypatch):
    class _Factory:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            return _FakeT5Tokenizer()

    monkeypatch.setattr("pepplatform.tokenizers.T5Tokenizer", _Factory)
    tokenizer = ProT5Tokenizer(model_name="test-prott5", max_length=5)
    result = tokenizer.batch_tokenize(["A C D E F G"])

    # max_length=5 residues + 1 EOS = 6 token budget
    assert result["input_ids"].shape[1] == 6
    sequences, kwargs = tokenizer.tokenizer.last_call
    # normalized and truncated to 5 residues
    assert sequences == ["A C D E F"]
    assert kwargs["max_length"] == 6  # 5 residues + 1 EOS


def test_prott5_tokenizer_normalizes_before_truncation(monkeypatch):
    class _Factory:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            return _FakeT5Tokenizer()

    monkeypatch.setattr("pepplatform.tokenizers.T5Tokenizer", _Factory)
    tokenizer = ProT5Tokenizer(model_name="test-prott5", max_length=3)
    # "A U Z B C" → strip whitespace → "AUZBC" → upper → "AUZBC"
    # → map U/Z/O/B → X → "AXXXC" → truncate to 3 residues → "A X X"
    tokenizer.batch_tokenize(["A U Z B C"])

    sequences, kwargs = tokenizer.tokenizer.last_call
    assert sequences == ["A X X"]
    assert kwargs["return_special_tokens_mask"] is True


def test_prott5_batch_tokenize_residues_uses_budget(monkeypatch):
    class _Factory:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            return _FakeT5Tokenizer()

    monkeypatch.setattr("pepplatform.tokenizers.T5Tokenizer", _Factory)
    tokenizer = ProT5Tokenizer(model_name="test-prott5", max_length=10)
    result = tokenizer.batch_tokenize_residues(["A C D E F G H"], max_residues=4)

    sequences, kwargs = tokenizer.tokenizer.last_call
    # max_length = max_residues + 1 (EOS) = 5
    assert kwargs["max_length"] == 5
    assert kwargs["return_special_tokens_mask"] is True


def test_prott5_tokenizer_reports_normalized_residue_length(monkeypatch):
    class _Factory:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            return _FakeT5Tokenizer()

    monkeypatch.setattr("pepplatform.tokenizers.T5Tokenizer", _Factory)
    tokenizer = ProT5Tokenizer(model_name="test-prott5", max_length=5)
    assert tokenizer.residue_length("A C", 5) == 2
    assert tokenizer.residue_length("A U Z", 5) == 3


def test_prott5_tokenizer_legacy_mode_keeps_total_token_budget(monkeypatch):
    class _Factory:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            return _FakeT5Tokenizer()

    monkeypatch.setattr("pepplatform.tokenizers.T5Tokenizer", _Factory)
    tokenizer = ProT5Tokenizer(
        model_name="test-prott5",
        max_length=4,
        legacy_total_token_length=True,
    )
    tokenizer.batch_tokenize(["A C D E F"])
    _sequences, kwargs = tokenizer.tokenizer.last_call
    assert kwargs["max_length"] == 4


# ── Encoder ──────────────────────────────────────────────────────────────────


def test_prott5_encoder_sets_identity_and_environment(monkeypatch, tmp_path):
    """ProT5Encoder must set output_dim, model_identity, and runtime_environment."""
    source = tmp_path / "snapshot"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(PROTT5_CONFIG_SIGNATURE), encoding="utf-8")
    for filename in ("tokenizer_config.json", "spiece.model", "tokenizer.json"):
        (source / filename).write_text("ok", encoding="utf-8")
    (source / "pytorch_model.bin").write_bytes(b"mock weights" * 100)

    class _MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            from transformers import T5Config

            self.config = T5Config(**PROTT5_CONFIG_SIGNATURE)

    mock_model = _MockModel().eval()

    class _Loader:
        @staticmethod
        def from_pretrained(model_name, use_safetensors=False):
            return mock_model

    monkeypatch.setattr("pepplatform.encoders.T5EncoderModel", _Loader)

    encoder = ProT5Encoder(model_name=str(source))
    assert encoder.output_dim == PROTT5_D_MODEL
    assert isinstance(encoder.model_identity, dict)
    assert encoder.model_identity["d_model"] == PROTT5_D_MODEL
    assert encoder.model_identity["pooling"] == PROTT5_POOLING
    assert isinstance(encoder.runtime_environment, dict)
    assert "torch_version" in encoder.runtime_environment


def test_prott5_encoder_does_not_retry_after_declared_bin_failure(monkeypatch, tmp_path):
    """A declared pytorch_model.bin failure must not trigger another format."""
    source = tmp_path / "snapshot"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(PROTT5_CONFIG_SIGNATURE), encoding="utf-8")
    for filename in ("tokenizer_config.json", "spiece.model", "tokenizer.json"):
        (source / filename).write_text("ok", encoding="utf-8")
    (source / "pytorch_model.bin").write_bytes(b"mock weights" * 100)

    call_count = [0]

    class _Loader:
        @staticmethod
        def from_pretrained(model_name, use_safetensors=False):
            call_count[0] += 1
            assert use_safetensors is False
            raise RuntimeError("bin load failed")

    monkeypatch.setattr("pepplatform.encoders.T5EncoderModel", _Loader)

    with pytest.raises(RuntimeError, match="bin load failed"):
        ProT5Encoder(model_name=str(source))
    assert call_count == [1]


def test_prott5_encoder_explicitly_checks_weight_format(monkeypatch, tmp_path):
    """When the source has pytorch_model.bin but no safetensors, the encoder
    uses pytorch_model.bin directly without a safetensors attempt."""
    source = tmp_path / "snapshot"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(PROTT5_CONFIG_SIGNATURE), encoding="utf-8")
    for filename in ("tokenizer_config.json", "spiece.model", "tokenizer.json"):
        (source / filename).write_text("ok", encoding="utf-8")
    (source / "pytorch_model.bin").write_bytes(b"mock weights" * 100)

    formats_requested = []

    class _MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            from transformers import T5Config

            self.config = T5Config(**PROTT5_CONFIG_SIGNATURE)

    mock_model = _MockModel().eval()

    class _Loader:
        @staticmethod
        def from_pretrained(model_name, use_safetensors=False):
            formats_requested.append(use_safetensors)
            return mock_model

    monkeypatch.setattr("pepplatform.encoders.T5EncoderModel", _Loader)

    ProT5Encoder(model_name=str(source))
    # Should request safetensors=False exactly once since pytorch_model.bin exists
    # and no safetensors file exists
    assert formats_requested == [False]


def test_prott5_encoder_loads_indexed_safetensors(monkeypatch, tmp_path):
    source = tmp_path / "sharded"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(PROTT5_CONFIG_SIGNATURE), encoding="utf-8")
    (source / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (source / "spiece.model").write_bytes(b"spiece")
    shard = source / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"weights")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"shared.weight": shard.name}}),
        encoding="utf-8",
    )

    class _MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            from transformers import T5Config

            self.config = T5Config(**PROTT5_CONFIG_SIGNATURE)

    calls = []

    class _Loader:
        @staticmethod
        def from_pretrained(model_name, use_safetensors=False):
            calls.append(use_safetensors)
            return _MockModel()

    monkeypatch.setattr("pepplatform.encoders.T5EncoderModel", _Loader)
    ProT5Encoder(model_name=str(source))
    assert calls == [True]


def test_prott5_encoder_does_not_fallback_after_sharded_load_error(monkeypatch, tmp_path):
    source = tmp_path / "sharded"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(PROTT5_CONFIG_SIGNATURE), encoding="utf-8")
    (source / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (source / "spiece.model").write_bytes(b"spiece")
    (source / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"shared.weight": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )
    (source / "pytorch_model.bin").write_bytes(b"legacy weights")
    calls = []

    class _Loader:
        @staticmethod
        def from_pretrained(model_name, use_safetensors=False):
            calls.append(use_safetensors)
            raise RuntimeError("safetensors load failed")

    monkeypatch.setattr("pepplatform.encoders.T5EncoderModel", _Loader)
    with pytest.raises(RuntimeError, match="safetensors load failed"):
        ProT5Encoder(model_name=str(source))
    assert calls == [True]


def test_prott5_pooling_excludes_eos_and_padding():
    hidden = torch.tensor([[[10.0], [1.0], [3.0], [9.0], [0.0], [0.0]]])
    encoder = object.__new__(ProT5Encoder)
    encoder.device = torch.device("cpu")
    encoder.model = lambda **_kwargs: SimpleNamespace(last_hidden_state=hidden)
    inputs = {
        "input_ids": torch.ones(1, 6, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0]]),
        "special_tokens_mask": torch.tensor([[0, 0, 0, 1, 1, 1]]),
    }

    pooled = encoder.encode(inputs)
    # Residues at positions 0,1,2: [10, 1, 3] -> mean = 14/3 ≈ 4.6667
    # EOS at position 3 (9.0) should be excluded
    assert torch.allclose(pooled, torch.tensor([[14.0 / 3.0]]))
    assert torch.isfinite(pooled).all()


def test_prott5_pooling_requires_explicit_mask_for_current_contract(tmp_path):
    hidden = torch.tensor([[[10.0], [1.0], [3.0], [9.0], [0.0]]])
    encoder = object.__new__(ProT5Encoder)
    encoder.device = torch.device("cpu")
    encoder.model = lambda **_kwargs: SimpleNamespace(last_hidden_state=hidden)
    inputs = {
        "input_ids": torch.ones(1, 5, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
    }

    with pytest.raises(RuntimeError, match="special_tokens_mask"):
        encoder.encode(inputs)

    pooled = encoder.encode_legacy_attention_mask_mean(inputs)
    assert torch.allclose(pooled, torch.tensor([[23.0 / 4.0]]), atol=1e-5)
    assert torch.isfinite(pooled).all()


def test_prott5_encoder_rejects_no_residue():
    """Encoder should reject inputs where no non-special residue tokens exist."""
    hidden = torch.zeros(1, 3, 1024)
    encoder = object.__new__(ProT5Encoder)
    encoder.device = torch.device("cpu")
    encoder.model = lambda **_kwargs: SimpleNamespace(last_hidden_state=hidden)
    inputs = {
        "input_ids": torch.ones(1, 3, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0]]),
        "special_tokens_mask": torch.tensor([[1, 1, 1]]),  # all special
    }

    with pytest.raises(ValueError, match="no residue"):
        encoder.encode(inputs)


# ── Residue batch encoding ───────────────────────────────────────────────────


def test_prott5_residue_batch_preserves_alignment(monkeypatch):
    hidden = torch.arange(2 * 7 * 3, dtype=torch.float).reshape(2, 7, 3)
    encoder = object.__new__(ProT5Encoder)
    encoder.device = torch.device("cpu")
    encoder.model = lambda **_kwargs: SimpleNamespace(last_hidden_state=hidden)
    # ProtT5 (T5) tokenizer: residues + EOS at end, no prefix token.
    # Row 0: 4 residues → [r0, r1, r2, r3, EOS, PAD, PAD] (7 tokens)
    # Row 1: 5 residues → [r0, r1, r2, r3, r4, EOS, PAD] (7 tokens)
    inputs = {
        "input_ids": torch.ones(2, 7, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0]]),
        "special_tokens_mask": torch.tensor([[0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1]]),
    }

    result = encoder.encode_residue_batch(inputs, [4, 5], max_length=5)
    assert isinstance(result, ResidueEncoding)
    assert result.features.shape == (2, 5, 3)
    assert result.mask.tolist() == [
        [True, True, True, True, False],
        [True, True, True, True, True],
    ]


def test_prott5_residue_batch_rejects_mismatched_alignment():
    """Residue batch should reject when residue count doesn't match."""
    hidden = torch.arange(1 * 5 * 3, dtype=torch.float).reshape(1, 5, 3)
    encoder = object.__new__(ProT5Encoder)
    encoder.device = torch.device("cpu")
    encoder.model = lambda **_kwargs: SimpleNamespace(last_hidden_state=hidden)
    inputs = {
        "input_ids": torch.ones(1, 5, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        "special_tokens_mask": torch.tensor([[0, 0, 0, 0, 0]]),
    }

    # 5 residues but claim length=3 - mismatch
    with pytest.raises(RuntimeError, match="alignment failed"):
        encoder.encode_residue_batch(inputs, [3], max_length=5)


def test_prott5_residue_batch_rejects_nonfinite_features():
    hidden = torch.zeros(1, 3, 3)
    hidden[0, 1, 0] = float("nan")
    encoder = object.__new__(ProT5Encoder)
    encoder.device = torch.device("cpu")
    encoder.model = lambda **_kwargs: SimpleNamespace(last_hidden_state=hidden)
    inputs = {
        "input_ids": torch.ones(1, 3, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "special_tokens_mask": torch.tensor([[0, 0, 1]]),
    }

    with pytest.raises(RuntimeError, match="non-finite"):
        encoder.encode_residue_batch(inputs, [2], max_length=2)


def test_prott5_residue_batch_rejects_empty_result():
    hidden = torch.empty(0, 5, 3)
    encoder = object.__new__(ProT5Encoder)
    encoder.device = torch.device("cpu")
    encoder.model = lambda **_kwargs: SimpleNamespace(last_hidden_state=hidden)
    inputs = {
        "input_ids": torch.ones(0, 5, dtype=torch.long),
        "attention_mask": torch.ones(0, 5, dtype=torch.long),
        "special_tokens_mask": torch.ones(0, 5, dtype=torch.long),
    }

    result = encoder.encode_residue_batch(inputs, [], max_length=5)
    assert result.features.shape == (0, 5, 3)


# ── Integration with existing test infrastructure ────────────────────────────


def test_prott5_residue_encoding_shape_and_dtype():
    """ResidueEncoding has expected [B,L,D] shape, bool mask, float16 features."""
    features = torch.randn(3, 5, 128, dtype=torch.float16)
    mask = torch.ones(3, 5, dtype=torch.bool)
    encoding = ResidueEncoding(features, mask)
    assert encoding.features.shape == (3, 5, 128)
    assert encoding.features.dtype == torch.float16
    assert encoding.mask.dtype == torch.bool


def test_prott5_max_residues_is_n_positions_minus_eos():
    assert PROTT5_MAX_RESIDUES == PROTT5_N_POSITIONS - 1
    assert PROTT5_MAX_RESIDUES == 511


def test_prott5_identity_contract_constants():
    assert PROTT5_POOLING == "residue_mean"
    assert PROTT5_REPRESENTATION_CONTRACT_VERSION == 1
    assert PROTT5_OFFICIAL_MODEL_ID == "Rostlab/prot_t5_xl_uniref50"
    assert PROTT5_D_MODEL == 1024
