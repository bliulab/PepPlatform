import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from pepplatform.cache_provenance import CACHE_PROVENANCE_VERSION
from pepplatform.encoders import EsmcEncoder
from pepplatform.esmc import (
    ESMC_BACKEND_ESMPLUSPLUS,
    ESMC_BACKEND_OFFICIAL,
    ESMC_OFFICIAL_300M_HIDDEN_SIZE,
    ESMC_OFFICIAL_300M_MODEL_ID,
    ESMC_OFFICIAL_300M_NUM_HEADS,
    ESMC_OFFICIAL_300M_NUM_LAYERS,
    ESMC_OFFICIAL_300M_WEIGHT_SHA256,
    ESMC_OFFICIAL_600M_HIDDEN_SIZE,
    ESMC_OFFICIAL_600M_MODEL_ID,
    ESMC_OFFICIAL_600M_WEIGHT_RELATIVE_PATH,
    ESMC_OFFICIAL_600M_WEIGHT_SHA256,
    ESMPLUSPLUS_SMALL_MODEL_ID,
    esmc_backend_for_source,
    esmc_model_identity,
    esmc_variant_for_source,
    inspect_esmc_runtime,
    official_esmc_model_options,
    require_official_esmc_runtime,
    validate_esmc_model_identity,
)
from pepplatform.tokenizers import EsmcTokenizer
from pepplatform.workflow import (
    PRETRAINED_MODEL_DEFAULTS,
    pretrained_model_local_candidates,
    resolve_pretrained_model_source,
)


class _FakeSequenceTokenizer:
    pad_token_id = 0


@pytest.fixture
def fake_esm_tokenization(monkeypatch):
    esm_module = ModuleType("esm")
    esm_module.__path__ = []
    tokenization_module = ModuleType("esm.tokenization")
    utils_module = ModuleType("esm.utils")
    utils_module.__path__ = []
    encoding_module = ModuleType("esm.utils.encoding")
    misc_module = ModuleType("esm.utils.misc")
    tokenizer = _FakeSequenceTokenizer()

    tokenization_module.get_esmc_model_tokenizers = lambda: tokenizer

    def tokenize_sequence(sequence, _tokenizer, add_special_tokens=True):
        assert add_special_tokens is True
        residue_ids = [3 + index for index, _residue in enumerate(sequence)]
        return torch.tensor([1, *residue_ids, 2], dtype=torch.long)

    def stack_variable_length_tensors(rows, constant_value=0):
        width = max(row.shape[0] for row in rows)
        result = torch.full((len(rows), width), constant_value, dtype=rows[0].dtype)
        for index, row in enumerate(rows):
            result[index, : row.shape[0]] = row
        return result

    encoding_module.tokenize_sequence = tokenize_sequence
    misc_module.stack_variable_length_tensors = stack_variable_length_tensors
    monkeypatch.setitem(sys.modules, "esm", esm_module)
    monkeypatch.setitem(sys.modules, "esm.tokenization", tokenization_module)
    monkeypatch.setitem(sys.modules, "esm.utils", utils_module)
    monkeypatch.setitem(sys.modules, "esm.utils.encoding", encoding_module)
    monkeypatch.setitem(sys.modules, "esm.utils.misc", misc_module)


def test_esmc_default_and_backend_identity(tmp_path):
    assert PRETRAINED_MODEL_DEFAULTS["esmc"] == ESMC_OFFICIAL_600M_MODEL_ID
    assert CACHE_PROVENANCE_VERSION == 5
    assert esmc_backend_for_source(ESMC_OFFICIAL_600M_MODEL_ID) == ESMC_BACKEND_OFFICIAL
    assert esmc_backend_for_source(ESMPLUSPLUS_SMALL_MODEL_ID) == ESMC_BACKEND_ESMPLUSPLUS

    official_root = tmp_path / "official"
    weight_path = official_root / ESMC_OFFICIAL_600M_WEIGHT_RELATIVE_PATH
    weight_path.parent.mkdir(parents=True)
    weight_path.write_bytes(b"checkpoint-shape-marker")
    assert esmc_backend_for_source(official_root) == ESMC_BACKEND_OFFICIAL

    esmplusplus_root = tmp_path / "esmplusplus"
    esmplusplus_root.mkdir()
    (esmplusplus_root / "config.json").write_text(
        json.dumps({"model_type": "esmplusplus"}), encoding="utf-8"
    )
    assert esmc_backend_for_source(esmplusplus_root) == ESMC_BACKEND_ESMPLUSPLUS

    official_identity = esmc_model_identity(ESMC_OFFICIAL_600M_MODEL_ID)
    assert official_identity["provider"] == "EvolutionaryScale"
    assert official_identity["hidden_size"] == ESMC_OFFICIAL_600M_HIDDEN_SIZE
    assert official_identity["pooling"] == "residue_mean"
    assert official_identity["weight_sha256"] == ESMC_OFFICIAL_600M_WEIGHT_SHA256

    compatibility_identity = esmc_model_identity(ESMPLUSPLUS_SMALL_MODEL_ID)
    assert compatibility_identity["provider"] == "Synthyra"
    assert compatibility_identity["model_id"] == ESMPLUSPLUS_SMALL_MODEL_ID
    assert compatibility_identity["pooling"] == "attention_mask_mean"
    assert compatibility_identity["provenance_status"] == "external_artifact_unverified"


def test_official_esmc_variants_are_distinct_and_json_safe():
    options = official_esmc_model_options()
    assert [option["id"] for option in options] == [
        ESMC_OFFICIAL_300M_MODEL_ID,
        ESMC_OFFICIAL_600M_MODEL_ID,
    ]
    assert options[0]["parameter_count"] == 300_000_000
    assert options[1]["parameter_count"] == 600_000_000
    assert esmc_variant_for_source(ESMC_OFFICIAL_300M_MODEL_ID)[0] == "300m"
    assert esmc_variant_for_source(ESMC_OFFICIAL_600M_MODEL_ID)[0] == "600m"


def test_official_esmc_weight_file_source_resolves_by_filename(tmp_path):
    weight = tmp_path / ESMC_OFFICIAL_600M_WEIGHT_RELATIVE_PATH.name
    weight.write_bytes(b"600m-marker")
    assert esmc_variant_for_source(weight)[0] == "600m"


def test_official_esmc_identity_records_variant_architecture():
    identity = esmc_model_identity(ESMC_OFFICIAL_300M_MODEL_ID)
    assert identity["variant"] == "300m"
    assert identity["parameter_count"] == 300_000_000
    assert identity["hidden_size"] == ESMC_OFFICIAL_300M_HIDDEN_SIZE
    assert identity["num_attention_heads"] == ESMC_OFFICIAL_300M_NUM_HEADS
    assert identity["num_hidden_layers"] == ESMC_OFFICIAL_300M_NUM_LAYERS
    assert identity["weight_sha256"] == ESMC_OFFICIAL_300M_WEIGHT_SHA256
    assert identity["provenance_status"] == "external_artifact_unverified"


def test_esmc_runtime_requires_exact_pinned_version(monkeypatch):
    import pepplatform.esmc as esmc_module

    inspect_esmc_runtime.cache_clear()
    monkeypatch.setattr(
        esmc_module.importlib.metadata,
        "version",
        lambda _name: "3.2.1.post1",
    )

    status = inspect_esmc_runtime()
    assert status["compatible"] is False
    assert status["code"] == "esmc_runtime_unsupported"
    assert "requires esm==3.2.0, found 3.2.1.post1" in status["message"]
    assert "pip install --no-cache-dir 'esm==3.2.0'" in status["message"]
    with pytest.raises(RuntimeError, match="found 3.2.1.post1"):
        require_official_esmc_runtime()
    inspect_esmc_runtime.cache_clear()


def test_esmc_model_identity_rejects_runtime_or_weight_drift():
    expected = esmc_model_identity(ESMC_OFFICIAL_600M_MODEL_ID)
    with pytest.raises(ValueError, match="runtime_version"):
        validate_esmc_model_identity(
            expected,
            dict(expected, runtime_version="3.2.1.post1"),
        )
    with pytest.raises(ValueError, match="weight_sha256"):
        validate_esmc_model_identity(
            expected,
            dict(expected, weight_sha256="0" * 64),
        )


def test_esmc_model_identity_rejects_variant_drift():
    expected = esmc_model_identity(
        ESMC_OFFICIAL_300M_MODEL_ID,
        weight_sha256="a" * 64,
        runtime_version="3.2.0",
    )
    observed = esmc_model_identity(
        ESMC_OFFICIAL_600M_MODEL_ID,
        weight_sha256="a" * 64,
        runtime_version="3.2.0",
    )
    with pytest.raises(ValueError, match="model_id"):
        validate_esmc_model_identity(expected, observed)


def test_esmc_model_identity_uses_resolved_weight_snapshot(tmp_path):
    weight_path = (
        tmp_path
        / "models--EvolutionaryScale--esmc-600m-2024-12"
        / "snapshots"
        / "revision-123"
        / ESMC_OFFICIAL_600M_WEIGHT_RELATIVE_PATH
    )
    weight_path.parent.mkdir(parents=True)
    weight_path.write_bytes(b"weight")

    identity = esmc_model_identity(
        ESMC_OFFICIAL_600M_MODEL_ID,
        weight_path=weight_path,
        weight_sha256=ESMC_OFFICIAL_600M_WEIGHT_SHA256,
        runtime_version="3.2.0",
    )

    assert identity["snapshot_revision"] == "revision-123"
    assert identity["weight_size_bytes"] == 6
    assert identity["provenance_status"] == "official_artifact_verified"


def test_esmc_local_candidates_are_checkpoint_specific(tmp_path):
    official = pretrained_model_local_candidates(tmp_path, "esmc", ESMC_OFFICIAL_600M_MODEL_ID)
    compatibility = pretrained_model_local_candidates(tmp_path, "esmc", ESMPLUSPLUS_SMALL_MODEL_ID)
    assert any("esmc-600m-2024-12" in value for value in official)
    assert all("ESMplusplus_small" not in value for value in official)
    assert any("ESMplusplus_small" in value for value in compatibility)
    assert all("esmc-600m-2024-12" not in value for value in compatibility)

    legacy_root = tmp_path / "models" / "Synthyra" / "ESMplusplus_small"
    legacy_root.mkdir(parents=True)
    (legacy_root / "config.json").write_text(
        json.dumps({"model_type": "esmplusplus"}), encoding="utf-8"
    )
    official_root = tmp_path / "models" / "esmc"
    official_weight = official_root / ESMC_OFFICIAL_600M_WEIGHT_RELATIVE_PATH
    official_weight.parent.mkdir(parents=True)
    official_weight.write_bytes(b"official-layout-marker")
    resolved = resolve_pretrained_model_source(
        tmp_path,
        ESMC_OFFICIAL_600M_MODEL_ID,
        [str(legacy_root)],
        "ESM-C",
        "esmc_model",
        tokenizer="esmc",
    )
    assert resolved == str(official_root)
    assert resolved != str(legacy_root)
    assert esmc_backend_for_source(resolved) == ESMC_BACKEND_OFFICIAL

    small_official = pretrained_model_local_candidates(
        tmp_path, "esmc", ESMC_OFFICIAL_300M_MODEL_ID
    )
    assert any("esmc-300m-2024-12" in value for value in small_official)
    assert all("esmc-600m-2024-12" not in value for value in small_official)


def test_offline_variant_resolution_does_not_fallback_to_600m(tmp_path, monkeypatch):
    six_hundred_root = tmp_path / "models" / "esmc"
    six_hundred_weight = six_hundred_root / ESMC_OFFICIAL_600M_WEIGHT_RELATIVE_PATH
    six_hundred_weight.parent.mkdir(parents=True)
    six_hundred_weight.write_bytes(b"600m-marker")
    three_hundred_root = tmp_path / "models" / "esmc-300m-2024-12"
    three_hundred_root.mkdir(parents=True)

    monkeypatch.setenv("PEPPLATFORM_OFFLINE_MODE", "1")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)

    with pytest.raises(SystemExit, match="ESM-C local model not found"):
        resolve_pretrained_model_source(
            tmp_path,
            ESMC_OFFICIAL_300M_MODEL_ID,
            [str(six_hundred_root)],
            "ESM-C",
            "esmc_model",
            tokenizer="esmc",
        )


def test_official_tokenizer_truncates_residues_and_marks_special_tokens(
    fake_esm_tokenization,
):
    tokenizer = EsmcTokenizer(max_length=3)
    batch = tokenizer.batch_tokenize(["acde", "G"])

    assert batch["sequence_tokens"].shape == (2, 5)
    assert batch["attention_mask"].tolist() == [
        [True, True, True, True, True],
        [True, True, True, False, False],
    ]
    assert batch["special_tokens_mask"].tolist() == [
        [True, False, False, False, True],
        [True, False, True, True, True],
    ]
    with pytest.raises(ValueError, match="unsupported residues: J"):
        tokenizer.tokenize("AJ")


def test_explicit_esmplusplus_keeps_transformers_compatibility(monkeypatch):
    import pepplatform.encoders as encoders_module
    import pepplatform.tokenizers as tokenizers_module

    class _CompatibilityTokenizer:
        def __call__(self, sequences, **_kwargs):
            count = 1 if isinstance(sequences, str) else len(sequences)
            return {
                "input_ids": torch.ones((count, 2), dtype=torch.long),
                "attention_mask": torch.ones((count, 2), dtype=torch.long),
            }

    class _TokenizerFactory:
        @staticmethod
        def from_pretrained(model_name, **_kwargs):
            assert model_name == ESMPLUSPLUS_SMALL_MODEL_ID
            return _CompatibilityTokenizer()

    class _CompatibilityModel:
        config = SimpleNamespace(hidden_size=960)

        def to(self, _device):
            return self

        def eval(self):
            return self

    class _ModelFactory:
        @staticmethod
        def from_pretrained(model_name, **_kwargs):
            assert model_name == ESMPLUSPLUS_SMALL_MODEL_ID
            return _CompatibilityModel()

    monkeypatch.setattr(tokenizers_module, "AutoTokenizer", _TokenizerFactory)
    monkeypatch.setattr(encoders_module, "AutoModelForMaskedLM", _ModelFactory)

    tokenizer = EsmcTokenizer(model_name=ESMPLUSPLUS_SMALL_MODEL_ID, max_length=8)
    encoder = EsmcEncoder(model_name=ESMPLUSPLUS_SMALL_MODEL_ID)
    assert tokenizer.backend == ESMC_BACKEND_ESMPLUSPLUS
    assert encoder.backend == ESMC_BACKEND_ESMPLUSPLUS
    assert encoder.output_dim == 960
    assert encoder.model_identity["provider"] == "Synthyra"


def test_official_encoder_selects_300m_variant_without_loading_real_weights(monkeypatch, tmp_path):
    weight_path = tmp_path / "esmc_300m_2024_12_v0.pth"
    weight_path.write_bytes(b"300m-test-weight")

    class _FakeModel:
        def eval(self):
            return self

    def fake_load(self):
        assert self.variant_key == "300m"
        assert self.variant["hidden_size"] == ESMC_OFFICIAL_300M_HIDDEN_SIZE
        return _FakeModel(), weight_path, "a" * 64, "3.2.0"

    monkeypatch.setattr(EsmcEncoder, "_load_official_model", fake_load)
    encoder = EsmcEncoder(model_name=ESMC_OFFICIAL_300M_MODEL_ID)
    assert encoder.output_dim == ESMC_OFFICIAL_300M_HIDDEN_SIZE
    assert encoder.model_identity["model_id"] == ESMC_OFFICIAL_300M_MODEL_ID
    assert encoder.model_identity["variant"] == "300m"
    assert encoder.model_identity["weight_sha256"] == "a" * 64


class _FakeOfficialModel:
    def __call__(self, *, sequence_tokens):
        values = sequence_tokens.float().unsqueeze(-1)
        embeddings = values.repeat(1, 1, ESMC_OFFICIAL_600M_HIDDEN_SIZE)
        return SimpleNamespace(embeddings=embeddings)


def test_official_encoder_uses_residue_only_pooling_and_alignment(
    fake_esm_tokenization,
):
    tokenizer = EsmcTokenizer(max_length=3)
    inputs = tokenizer.batch_tokenize(["ACD", "G"])
    encoder = EsmcEncoder.__new__(EsmcEncoder)
    encoder.device = torch.device("cpu")
    encoder.backend = ESMC_BACKEND_OFFICIAL
    encoder.output_dim = ESMC_OFFICIAL_600M_HIDDEN_SIZE
    encoder.model = _FakeOfficialModel()

    pooled = encoder.encode(inputs)
    assert pooled.shape == (2, ESMC_OFFICIAL_600M_HIDDEN_SIZE)
    assert torch.allclose(pooled[0], torch.full_like(pooled[0], 4.0))
    assert torch.allclose(pooled[1], torch.full_like(pooled[1], 3.0))

    residue = encoder.encode_residue_batch(inputs, [3, 1], max_length=3)
    assert residue.features.shape == (2, 3, ESMC_OFFICIAL_600M_HIDDEN_SIZE)
    assert residue.mask.tolist() == [[True, True, True], [True, False, False]]
    assert residue.features.dtype == torch.float16
    assert residue.features[0, :, 0].tolist() == [3.0, 4.0, 5.0]
