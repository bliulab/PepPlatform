import json
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

import pepplatform.esm3 as esm3_module
from pepplatform.encoders import Esm3Encoder, ResidueEncoding
from pepplatform.esm3 import (
    ESM3_HF_DOWNLOAD_METADATA_RELATIVE_PATH,
    ESM3_HIDDEN_SIZE,
    ESM3_OFFICIAL_MODEL_ID,
    ESM3_OFFICIAL_WEIGHT_RELATIVE_PATH,
    ESM3_POOLING,
    ESM3_REPRESENTATION_CONTRACT_VERSION,
    ESM3_RUNTIME_KEY,
    esm3_model_identity,
    esm3_residue_count,
    inspect_esm3_source,
    is_complete_esm3_source,
    normalize_esm3_model_name,
    normalize_esm3_sequence,
    truncate_esm3_sequence,
    validate_esm3_max_residues,
    validate_esm3_model_identity,
)
from pepplatform.models import ResidueModelInput
from pepplatform.tokenizers import Esm3Tokenizer
from pepplatform.web.inference import InferenceUnsupportedError, ModelService, _Esm3RuntimePool
from pepplatform.workflow import resolve_pretrained_model_source
from scripts import train_pair_pipeline
from scripts import train_sequence_pipeline


def _install_fake_inference_api(monkeypatch, *, nonfinite=False):
    api_module = ModuleType("esm.sdk.api")
    constants_module = ModuleType("esm.utils.constants.esm3")

    class FakeProtein:
        def __init__(self, sequence):
            self.sequence = sequence

    class FakeLogitsConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeClient:
        def encode(self, protein):
            residue_id = 4
            ids = [0]
            for char in protein.sequence:
                if char == "|":
                    ids.append(31)
                else:
                    ids.append(residue_id)
                    residue_id += 1
            ids.append(2)
            return SimpleNamespace(sequence=torch.tensor(ids))

        def logits(self, protein_tensor, _config):
            values = torch.arange(protein_tensor.sequence.shape[0], dtype=torch.float).reshape(
                -1, 1
            )
            if nonfinite:
                values[1] = float("nan")
            return SimpleNamespace(embeddings=values)

    api_module.ESMProtein = FakeProtein
    api_module.LogitsConfig = FakeLogitsConfig
    constants_module.SEQUENCE_BOS_TOKEN = 0
    constants_module.SEQUENCE_PAD_TOKEN = 1
    constants_module.SEQUENCE_EOS_TOKEN = 2
    constants_module.SEQUENCE_CHAINBREAK_TOKEN = 31
    constants_module.SEQUENCE_MASK_TOKEN = 32
    monkeypatch.setitem(sys.modules, "esm.sdk.api", api_module)
    monkeypatch.setitem(sys.modules, "esm.utils.constants.esm3", constants_module)
    return FakeClient()


def test_esm3_identity_aliases_and_sequence_contract():
    assert normalize_esm3_model_name(ESM3_OFFICIAL_MODEL_ID) == ESM3_RUNTIME_KEY
    assert normalize_esm3_model_name("esm3-sm-open-v1") == ESM3_RUNTIME_KEY
    assert normalize_esm3_sequence(" a c|de ") == "AC|DE"
    assert esm3_residue_count("AC|DE") == 4
    assert truncate_esm3_sequence("AC|DE", 3) == "AC|D"
    with pytest.raises(ValueError, match="at least one residue"):
        normalize_esm3_sequence("  ")
    with pytest.raises(ValueError, match="unsupported"):
        normalize_esm3_sequence("AC*")
    with pytest.raises(ValueError, match="chain breaks"):
        normalize_esm3_sequence("AC||DE")
    with pytest.raises(ValueError, match="integer"):
        validate_esm3_max_residues(3.5)


def test_esm3_source_requires_official_sdk_layout(tmp_path):
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    assert not is_complete_esm3_source(incomplete)
    assert inspect_esm3_source(incomplete)["model_fingerprint"] == ""

    complete = tmp_path / "complete"
    weight = complete / ESM3_OFFICIAL_WEIGHT_RELATIVE_PATH
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"esm3-test-weight")

    assert is_complete_esm3_source(complete)
    facts = inspect_esm3_source(complete)
    assert facts["complete"] is True
    assert facts["weight_sha256"]
    assert facts["provenance_status"] == "external_artifact_unverified"
    identity = esm3_model_identity(complete)
    assert identity["hidden_size"] == ESM3_HIDDEN_SIZE
    assert identity["pooling"] == ESM3_POOLING
    assert identity["representation_contract_version"] == ESM3_REPRESENTATION_CONTRACT_VERSION
    with pytest.raises(ValueError, match="identity differs"):
        validate_esm3_model_identity(identity, dict(identity, weight_sha256="0" * 64))


def test_esm3_local_dir_metadata_binds_revision_and_official_digest(tmp_path, monkeypatch):
    source = tmp_path / "esm3"
    weight = source / ESM3_OFFICIAL_WEIGHT_RELATIVE_PATH
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"official-esm3-test-weight")
    digest = esm3_module.sha256_file(weight)
    revision = "a" * 40
    metadata = source / ESM3_HF_DOWNLOAD_METADATA_RELATIVE_PATH
    metadata.parent.mkdir(parents=True)
    metadata.write_text(f"{revision}\n{digest}\n0\n", encoding="utf-8")
    monkeypatch.setattr(esm3_module, "ESM3_OFFICIAL_WEIGHT_SHA256", digest)
    monkeypatch.setattr(esm3_module, "ESM3_OFFICIAL_REVISIONS", frozenset({revision}))

    facts = inspect_esm3_source(source)

    assert facts["snapshot_revision"] == revision
    assert facts["weight_sha256"] == digest
    assert facts["provenance_status"] == "official_artifact_verified"


def test_esm3_local_dir_rejects_mismatched_download_metadata(tmp_path, monkeypatch):
    source = tmp_path / "esm3"
    weight = source / ESM3_OFFICIAL_WEIGHT_RELATIVE_PATH
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"official-esm3-test-weight")
    digest = esm3_module.sha256_file(weight)
    revision = "b" * 40
    metadata = source / ESM3_HF_DOWNLOAD_METADATA_RELATIVE_PATH
    metadata.parent.mkdir(parents=True)
    metadata.write_text(f"{revision}\n{'0' * 64}\n0\n", encoding="utf-8")
    monkeypatch.setattr(esm3_module, "ESM3_OFFICIAL_WEIGHT_SHA256", digest)
    monkeypatch.setattr(esm3_module, "ESM3_OFFICIAL_REVISIONS", frozenset({revision}))

    facts = inspect_esm3_source(source)

    assert facts["snapshot_revision"] == revision
    assert facts["provenance_status"] == "external_artifact_unverified"


def test_esm3_workflow_rejects_existing_incomplete_source(tmp_path):
    incomplete = tmp_path / "esm3"
    incomplete.mkdir()
    with pytest.raises(SystemExit, match="local model is incomplete"):
        resolve_pretrained_model_source(
            tmp_path,
            str(incomplete),
            [],
            "ESM3",
            "esm3_model",
            tokenizer="esm3",
        )


def test_esm3_tokenizer_uses_sdk_residue_budget_and_normalization():
    tokenizer = Esm3Tokenizer(max_length=3)
    result = tokenizer.batch_tokenize(["a cde", "G"])

    assert result["input_ids"].shape == (2, 5)
    assert result["special_tokens_mask"].tolist() == [
        [1, 0, 0, 0, 1],
        [1, 0, 1, 1, 1],
    ]
    assert tokenizer.residue_length("a cde", 3) == 3


def test_esm3_encoder_normalizes_public_model_id_for_sdk(monkeypatch):
    from esm.models.esm3 import ESM3

    called = {}

    class FakeModel:
        encoder = SimpleNamespace(embed_dim=ESM3_HIDDEN_SIZE)

        def to(self, device):
            called["to"] = str(device)
            return self

        def eval(self):
            called["eval"] = True
            return self

    def fake_from_pretrained(_cls, model_name, device=None):
        called["model_name"] = model_name
        called["device"] = device
        return FakeModel()

    monkeypatch.setattr(ESM3, "from_pretrained", classmethod(fake_from_pretrained))
    encoder = Esm3Encoder(model_name=ESM3_OFFICIAL_MODEL_ID, device=torch.device("cpu"))

    assert called["model_name"] == ESM3_RUNTIME_KEY
    assert called["eval"] is True
    assert encoder.output_dim == ESM3_HIDDEN_SIZE
    assert encoder.model_identity["runtime_key"] == ESM3_RUNTIME_KEY


def test_esm3_sequence_pooling_excludes_all_sdk_special_tokens(monkeypatch):
    encoder = object.__new__(Esm3Encoder)
    encoder.device = torch.device("cpu")
    encoder.client = _install_fake_inference_api(monkeypatch)

    encoded = encoder.encode_residue_batch(["A|C"], max_length=4)
    pooled = encoder.encode_sequences(["A|C"], max_length=4)

    assert encoded.mask.tolist() == [[True, True, False, False]]
    assert encoded.features[0, :2, 0].float().tolist() == [1.0, 3.0]
    assert pooled.shape == (1, 1)
    assert pooled.item() == pytest.approx(2.0)


def test_esm3_pooled_embeddings_are_float32_for_mlp_training(monkeypatch):
    encoder = object.__new__(Esm3Encoder)
    encoder.device = torch.device("cpu")
    encoder.client = _install_fake_inference_api(monkeypatch)

    pooled = encoder.encode_sequences(["A|C", "DE"], max_length=4)

    assert pooled.dtype == torch.float32
    assert pooled.shape == (2, 1)


def test_esm3_pooled_embeddings_empty_input_is_float32(monkeypatch):
    encoder = object.__new__(Esm3Encoder)
    encoder.device = torch.device("cpu")
    encoder.output_dim = ESM3_HIDDEN_SIZE
    encoder.client = _install_fake_inference_api(monkeypatch)

    pooled = encoder.encode_sequences([], max_length=4)

    assert pooled.dtype == torch.float32
    assert pooled.shape == (0, ESM3_HIDDEN_SIZE)


def test_esm3_rejects_nonfinite_residue_embeddings(monkeypatch):
    encoder = object.__new__(Esm3Encoder)
    encoder.device = torch.device("cpu")
    encoder.client = _install_fake_inference_api(monkeypatch, nonfinite=True)
    with pytest.raises(RuntimeError, match="non-finite"):
        encoder.encode_residue_batch(["AC"], max_length=4)


def test_pair_esm3_token_extraction_preserves_explicit_mask(monkeypatch):
    monkeypatch.setattr(train_pair_pipeline, "_feature_cache_meta", lambda *_args, **_kwargs: {})

    class FakeEncoder:
        def encode_residue_batch(self, sequences, max_length):
            features = torch.zeros((len(sequences), max_length, 2))
            mask = torch.zeros((len(sequences), max_length), dtype=torch.bool)
            for index, sequence in enumerate(sequences):
                mask[index, : len(sequence)] = True
            return ResidueEncoding(features, mask)

    args = SimpleNamespace(
        tokenizer="esm3",
        max_length=4,
        cache_features=False,
        esm3_extract_batch_size=2,
    )
    result = train_pair_pipeline.extract_pretrained_token_sequences(
        args,
        tokenizer=None,
        encoder=FakeEncoder(),
        sequences=["AC", "G"],
        split_name="train",
        role_name="peptide",
        device=torch.device("cpu"),
    )
    assert isinstance(result, ResidueModelInput)
    assert result.mask.tolist() == [
        [True, True, False, False],
        [True, False, False, False],
    ]


def test_pair_esm3_pooled_extraction_deduplicates_and_reuses_cache(tmp_path):
    class FakeEncoder:
        output_dim = 2

        def __init__(self):
            self.calls = []

        def encode_sequences(self, sequences, max_length):
            self.calls.append(list(sequences))
            return torch.tensor(
                [[float(len(sequence)), float(max_length)] for sequence in sequences]
            )

    args = SimpleNamespace(
        tokenizer="esm3",
        encoder="esm3",
        max_length=4,
        batch_size=16,
        cache_features=True,
        feature_dir=str(tmp_path),
        esm3_sdpa_backend="math",
        _feature_train_sha256="a" * 64,
    )
    encoder = FakeEncoder()
    first = train_pair_pipeline.extract_embeddings(
        args,
        tokenizer=None,
        encoder=encoder,
        sequences=["AC", "AC", "G"],
        split_name="train",
        role_name="peptide",
        device=torch.device("cpu"),
    )
    assert encoder.calls == [["AC", "G"]]
    torch.testing.assert_close(first[0], first[1])

    class FailingEncoder(FakeEncoder):
        def encode_sequences(self, sequences, max_length):
            raise AssertionError("validated final cache should be reused")

    second = train_pair_pipeline.extract_embeddings(
        args,
        tokenizer=None,
        encoder=FailingEncoder(),
        sequences=["AC", "AC", "G"],
        split_name="train",
        role_name="peptide",
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(first, second)


def test_pair_esm3_pooled_extraction_resumes_validated_shards(tmp_path):
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    sequences = []
    for value in range(130):
        current = value
        chars = []
        for _ in range(3):
            chars.append(alphabet[current % len(alphabet)])
            current //= len(alphabet)
        sequences.append("".join(chars))

    class InterruptingEncoder:
        output_dim = 2

        def __init__(self, interrupt=False):
            self.calls = 0
            self.interrupt = interrupt

        def encode_sequences(self, batch, max_length):
            self.calls += 1
            if self.interrupt and self.calls == 2:
                raise RuntimeError("simulated native interruption")
            return torch.tensor([[float(len(sequence)), float(max_length)] for sequence in batch])

    args = SimpleNamespace(
        tokenizer="esm3",
        encoder="esm3",
        max_length=4,
        batch_size=16,
        cache_features=True,
        feature_dir=str(tmp_path),
        esm3_sdpa_backend="math",
        _feature_train_sha256="b" * 64,
    )
    interrupted = InterruptingEncoder(interrupt=True)
    with pytest.raises(RuntimeError, match="simulated native interruption"):
        train_pair_pipeline.extract_embeddings(
            args,
            tokenizer=None,
            encoder=interrupted,
            sequences=sequences,
            split_name="train",
            role_name="protein",
            device=torch.device("cpu"),
        )
    assert interrupted.calls == 2

    resumed = InterruptingEncoder()
    result = train_pair_pipeline.extract_embeddings(
        args,
        tokenizer=None,
        encoder=resumed,
        sequences=sequences,
        split_name="train",
        role_name="protein",
        device=torch.device("cpu"),
    )
    assert resumed.calls == 2
    assert result.shape == (130, 2)


def test_pair_semantic_sample_slice_preserves_residue_mask():
    value = ResidueModelInput(
        torch.arange(24, dtype=torch.float32).reshape(3, 4, 2),
        torch.tensor(
            [
                [True, True, False, False],
                [True, False, False, False],
                [True, True, True, False],
            ]
        ),
    )
    sampled = train_pair_pipeline._sample_batch_to_device(
        value,
        torch.device("cpu"),
    )
    assert isinstance(sampled, ResidueModelInput)
    assert sampled.shape == (2, 4, 2)
    torch.testing.assert_close(sampled.mask, value.mask[:2])


def test_web_rejects_legacy_esm3_artifact_before_loading_weights():
    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._load_esm3_for_inference(
            {"model_name": ESM3_OFFICIAL_MODEL_ID},
            60,
            torch.device("cpu"),
        )
    assert exc_info.value.code == "esm3_legacy_artifact"


def test_esm3_runtime_pool_reuses_encoder_and_rejects_suspended_loads(monkeypatch):
    monkeypatch.setenv("PEPPLATFORM_ESM3_IDLE_TTL_SECONDS", "60")
    pool = _Esm3RuntimePool()
    calls = []

    class FakeEncoder:
        device = torch.device("cpu")

    def loader():
        calls.append(True)
        return FakeEncoder()

    try:
        first, first_hit = pool.get_or_load(("source",), loader)
        second, second_hit = pool.get_or_load(("source",), loader)
        assert first_hit is False
        assert second_hit is True
        assert len(calls) == 1
        assert first._entry is second._entry

        entered = threading.Event()
        release = threading.Event()

        def active_request():
            with pool.request():
                entered.set()
                release.wait(timeout=2)

        request_thread = threading.Thread(target=active_request)
        request_thread.start()
        assert entered.wait(timeout=2)
        suspend_thread = threading.Thread(target=lambda: pool.suspend("training"))
        suspend_thread.start()
        time.sleep(0.02)
        assert suspend_thread.is_alive()
        release.set()
        request_thread.join(timeout=2)
        suspend_thread.join(timeout=2)
        assert not suspend_thread.is_alive()
        assert pool.status()["cache_status"] == "suspended"
        with pytest.raises(InferenceUnsupportedError) as exc_info:
            pool.get_or_load(("source",), loader)
        assert exc_info.value.code == "gpu_busy_training"
    finally:
        pool.close()


def _esm3_cache_pipeline(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "seqs.fasta").write_text(">a\nACDE\n", encoding="utf-8")
    pipeline = object.__new__(train_sequence_pipeline.SequencePipeline)
    pipeline.args = SimpleNamespace(
        data_dir=str(data_dir),
        max_length=60,
        _resolved_pretrained_model_identity=None,
        _resolved_pretrained_runtime_environment=None,
    )
    monkeypatch.setattr(
        pipeline,
        "resolve_esm3_model_source",
        lambda: ESM3_OFFICIAL_MODEL_ID,
    )
    return pipeline


def test_esm3_cache_meta_records_float32_dtype(monkeypatch, tmp_path):
    pipeline = _esm3_cache_pipeline(monkeypatch, tmp_path)

    meta = pipeline._embedding_cache_meta("esm3", "train", ["ACDE"])

    assert meta["dtype"] == "float32"


def test_esm3_cache_meta_float32_dtype_is_esm3_only(monkeypatch, tmp_path):
    pipeline = _esm3_cache_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "resolve_esmc_model_source", lambda: "some/esmc-model")

    meta = pipeline._embedding_cache_meta("esmc", "train", ["ACDE"])

    assert "dtype" not in meta


def test_esm3_fp16_cache_is_rejected_and_rebuilt_as_float32(monkeypatch, tmp_path):
    pipeline = _esm3_cache_pipeline(monkeypatch, tmp_path)
    save_path = str(tmp_path / "train_embeddings.pt")
    sequences = ["ACDE", "G"]
    expected = pipeline._embedding_cache_meta("esm3", "train", sequences)
    legacy_meta = {key: value for key, value in expected.items() if key != "dtype"}

    torch.save(torch.zeros((2, ESM3_HIDDEN_SIZE), dtype=torch.float16), save_path)
    with open(f"{save_path}.meta.json", "w", encoding="utf-8") as handle:
        json.dump(legacy_meta, handle)

    cached = pipeline._load_embedding_cache(save_path, "esm3", "train", sequences)
    assert cached is None

    pipeline._save_embeddings(
        save_path,
        "esm3",
        "train",
        sequences,
        torch.zeros((2, ESM3_HIDDEN_SIZE), dtype=torch.float32),
    )
    with open(f"{save_path}.meta.json", "r", encoding="utf-8") as handle:
        saved_meta = json.load(handle)
    assert saved_meta["dtype"] == "float32"

    cached = pipeline._load_embedding_cache(save_path, "esm3", "train", sequences)
    assert cached is not None
    assert cached.dtype == torch.float32
