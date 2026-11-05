import pytest

import pepplatform.encoders as encoders
from pepplatform.encoders import ProtBertEncoder
from pepplatform.esm3 import ESM3_OFFICIAL_WEIGHT_RELATIVE_PATH
from pepplatform.workflow import resolve_cached_pretrained_model_source
from scripts.train_pipeline import _resolve_pretrained_model_source
from scripts.train_pair_pipeline import _resolve_model_source as _resolve_pair_model_source


def test_pretrained_model_source_prefers_local_candidate(tmp_path):
    local_model = tmp_path / "models" / "esm2"
    local_model.mkdir(parents=True)

    resolved = _resolve_pretrained_model_source(
        tmp_path,
        "facebook/esm2_t30_150M_UR50D",
        [str(local_model)],
        "ESM2",
        "esm2_model",
    )

    assert resolved == str(local_model)


def test_pretrained_model_source_resolves_explicit_relative_path(tmp_path):
    local_model = tmp_path / "custom" / "esm2"
    local_model.mkdir(parents=True)

    resolved = _resolve_pretrained_model_source(
        tmp_path,
        "custom/esm2",
        [],
        "ESM2",
        "esm2_model",
    )

    assert resolved == str(local_model)


def test_pretrained_model_source_prefers_complete_huggingface_cache(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    repo_cache = cache / "models--Rostlab--prot_bert"
    complete = repo_cache / "snapshots" / "complete"
    incomplete = repo_cache / "snapshots" / "incomplete"
    complete.mkdir(parents=True)
    incomplete.mkdir(parents=True)
    (repo_cache / "refs").mkdir()
    (repo_cache / "refs" / "main").write_text("complete", encoding="utf-8")
    for filename in ["config.json", "tokenizer_config.json", "vocab.txt", "pytorch_model.bin"]:
        (complete / filename).write_text("ok", encoding="utf-8")
    (incomplete / "model.safetensors").write_text("partial", encoding="utf-8")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache))

    cached = resolve_cached_pretrained_model_source("Rostlab/prot_bert", "protbert")
    resolved = _resolve_pretrained_model_source(
        tmp_path,
        "Rostlab/prot_bert",
        [str(tmp_path / "models" / "protbert")],
        "ProtBERT",
        "protbert_model",
    )

    assert cached == str(complete)
    assert resolved == str(complete)


def test_cached_pretrained_model_source_accepts_legacy_probert_alias(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    repo_cache = cache / "models--Rostlab--prot_bert"
    complete = repo_cache / "snapshots" / "complete"
    complete.mkdir(parents=True)
    (repo_cache / "refs").mkdir()
    (repo_cache / "refs" / "main").write_text("complete", encoding="utf-8")
    for filename in ["config.json", "tokenizer_config.json", "vocab.txt", "pytorch_model.bin"]:
        (complete / filename).write_text("ok", encoding="utf-8")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache))

    assert resolve_cached_pretrained_model_source("Rostlab/prot_bert", "probert") == str(complete)


def test_pretrained_model_source_fails_fast_in_offline_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("PEPPLATFORM_OFFLINE_MODE", "1")
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    with pytest.raises(SystemExit, match="ESM2 local model not found"):
        _resolve_pretrained_model_source(
            tmp_path,
            "facebook/esm2_t30_150M_UR50D",
            [str(tmp_path / "models" / "esm2")],
            "ESM2",
            "esm2_model",
        )


def test_esm3_offline_resolution_selects_complete_huggingface_snapshot(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    repo_cache = cache / "models--EvolutionaryScale--esm3-sm-open-v1"
    revision = "retained-esm3-revision"
    snapshot = repo_cache / "snapshots" / revision
    snapshot_weight = snapshot / ESM3_OFFICIAL_WEIGHT_RELATIVE_PATH
    snapshot_weight.parent.mkdir(parents=True)
    snapshot_weight.write_bytes(b"retained-esm3-weight")
    (repo_cache / "refs").mkdir()
    (repo_cache / "refs" / "main").write_text(revision, encoding="utf-8")

    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache))
    monkeypatch.setenv("PEPPLATFORM_OFFLINE_MODE", "1")

    resolved = _resolve_pretrained_model_source(
        tmp_path,
        "EvolutionaryScale/esm3-sm-open-v1",
        [],
        "ESM3",
        "esm3_model",
    )

    assert resolved == str(snapshot)


def test_protbert_encoder_uses_pytorch_bin_for_local_snapshot(tmp_path, monkeypatch):
    local_model = tmp_path / "prot_bert"
    local_model.mkdir()
    (local_model / "config.json").write_text(
        '{"model_type":"bert","hidden_size":1024,"num_attention_heads":16,'
        '"num_hidden_layers":30,"intermediate_size":4096,"vocab_size":30,'
        '"max_position_embeddings":40000}',
        encoding="utf-8",
    )
    (local_model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (local_model / "vocab.txt").write_text("[PAD]\n[UNK]\n", encoding="utf-8")
    (local_model / "pytorch_model.bin").write_text("weights", encoding="utf-8")
    calls = []

    class _LoadedModel:
        class _Config:
            model_type = "bert"
            hidden_size = 1024
            num_attention_heads = 16
            num_hidden_layers = 30
            intermediate_size = 4096
            vocab_size = 30
            max_position_embeddings = 40000

            def to_dict(self):
                return {
                    "model_type": self.model_type,
                    "hidden_size": self.hidden_size,
                    "num_attention_heads": self.num_attention_heads,
                    "num_hidden_layers": self.num_hidden_layers,
                    "intermediate_size": self.intermediate_size,
                    "vocab_size": self.vocab_size,
                    "max_position_embeddings": self.max_position_embeddings,
                }

        config = _Config()

        def to(self, device):
            return self

        def eval(self):
            return None

    class _AutoModel:
        @staticmethod
        def from_pretrained(model_name, use_safetensors=True):
            calls.append((model_name, use_safetensors))
            return _LoadedModel()

    monkeypatch.setattr(encoders, "AutoModel", _AutoModel)

    ProtBertEncoder(model_name=str(local_model))

    assert calls == [(str(local_model), False)]


def test_pair_pretrained_model_source_fails_fast_in_offline_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("PEPPLATFORM_OFFLINE_MODE", "1")
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    with pytest.raises(SystemExit, match="ESM2 local model not found"):
        _resolve_pair_model_source(
            "facebook/esm2_t30_150M_UR50D",
            [str(tmp_path / "models" / "esm2")],
            "ESM2",
            "esm2_model",
        )
