import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models

from pepplatform.bpe import (
    BPE_ALPHABET,
    BPE_CONTRACT_VERSION,
    BPE_SPECIAL_TOKEN_IDS,
    BPE_TOKENIZERS_VERSION,
    BPE_VARIANT,
    BPEContractError,
    bpe_provenance_path,
    bundle_bpe_artifact,
    inspect_bpe_artifact,
    normalize_bpe_sequence,
)
from pepplatform.featurizers import FeaturizerFactory
from pepplatform.tokenizers import BPETokenizer, BPETokenizerTrainer
from pepplatform.web.inference import InferenceUnsupportedError, ModelService
from scripts import train_pair_pipeline, train_sequence_pipeline


TOY_CORPUS = ["ACACAC", "ACAC", "ACGT"]


def train_toy(path: Path, *, vocab_size=32):
    trainer = BPETokenizerTrainer(
        vocab_size=vocab_size,
        show_progress=False,
    )
    trainer.train_sequences(TOY_CORPUS, path)
    return trainer


def test_bpe_toy_corpus_learns_expected_first_merge_and_encoding(tmp_path):
    path = tmp_path / "bpe.json"
    train_toy(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["model"]["type"] == "BPE"
    assert payload["model"]["merges"] == [["A", "C"]]
    assert {
        token: payload["model"]["vocab"][token] for token in BPE_SPECIAL_TOKEN_IDS
    } == BPE_SPECIAL_TOKEN_IDS

    tokenizer = BPETokenizer(path, max_length=10)
    vocab = payload["model"]["vocab"]
    assert tokenizer.tokenize("ac ac gt") == [
        BPE_SPECIAL_TOKEN_IDS["<CLS>"],
        vocab["AC"],
        vocab["AC"],
        vocab["G"],
        vocab["T"],
        BPE_SPECIAL_TOKEN_IDS["<SEP>"],
        0,
        0,
        0,
        0,
    ]


def test_bpe_training_is_byte_identical_and_deterministic(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    train_toy(first)
    train_toy(second)

    assert first.read_bytes() == second.read_bytes()
    assert inspect_bpe_artifact(first)["sha256"] == inspect_bpe_artifact(second)["sha256"]


def test_bpe_normalization_extended_alphabet_and_invalid_input(tmp_path):
    path = tmp_path / "bpe.json"
    train_toy(path)
    tokenizer = BPETokenizer(path, max_length=len(BPE_ALPHABET) + 2)

    spaced_lower = " acd\n efg\t"
    assert normalize_bpe_sequence(spaced_lower) == "ACDEFG"
    assert tokenizer.tokenize(spaced_lower) == tokenizer.tokenize("ACDEFG")

    all_symbols = tokenizer.tokenize("".join(BPE_ALPHABET))
    assert BPE_SPECIAL_TOKEN_IDS["<UNK>"] not in all_symbols
    assert all_symbols[0] == BPE_SPECIAL_TOKEN_IDS["<CLS>"]
    non_padding = [token_id for token_id in all_symbols if token_id != 0]
    assert non_padding[-1] == BPE_SPECIAL_TOKEN_IDS["<SEP>"]

    with pytest.raises(BPEContractError, match="empty"):
        tokenizer.tokenize(" \n\t ")
    with pytest.raises(BPEContractError, match="unsupported character"):
        tokenizer.tokenize("ACD*")


def test_bpe_has_no_wordpiece_100_character_unknown_limit(tmp_path):
    path = tmp_path / "bpe.json"
    train_toy(path)
    sequence = "A" * 101
    tokenizer = BPETokenizer(path, max_length=103)

    ids = tokenizer.tokenize(sequence)

    assert len(ids) == 103
    assert ids[0] == BPE_SPECIAL_TOKEN_IDS["<CLS>"]
    assert ids[-1] == BPE_SPECIAL_TOKEN_IDS["<SEP>"]
    assert BPE_SPECIAL_TOKEN_IDS["<UNK>"] not in ids


def test_bpe_truncation_preserves_cls_and_sep_boundaries(tmp_path):
    path = tmp_path / "bpe.json"
    train_toy(path)
    ids = BPETokenizer(path, max_length=5).tokenize("ACACGT")

    assert len(ids) == 5
    assert ids[0] == BPE_SPECIAL_TOKEN_IDS["<CLS>"]
    assert ids[-1] == BPE_SPECIAL_TOKEN_IDS["<SEP>"]
    assert 0 not in ids


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["model"].update(type="WordPiece"),
        lambda payload: payload["model"].update(dropout=0.1),
        lambda payload: payload["pre_tokenizer"].update(add_prefix_space=True),
        lambda payload: payload["model"]["vocab"].update({"<PAD>": 9}),
        lambda payload: payload.update(post_processor=None),
    ],
)
def test_bpe_contract_rejects_semantically_incompatible_json(tmp_path, mutation):
    valid = tmp_path / "valid.json"
    invalid = tmp_path / "invalid.json"
    train_toy(valid)
    payload = json.loads(valid.read_text(encoding="utf-8"))
    mutation(payload)
    invalid.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BPEContractError):
        inspect_bpe_artifact(invalid)


def test_bpe_contract_rejects_real_wordpiece_and_corrupt_json(tmp_path):
    wordpiece = tmp_path / "wordpiece.json"
    Tokenizer(
        models.WordPiece(
            vocab={"<UNK>": 0, "A": 1},
            unk_token="<UNK>",
            max_input_chars_per_word=100,
        )
    ).save(str(wordpiece))
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not-json", encoding="utf-8")

    with pytest.raises(BPEContractError, match="model.type"):
        inspect_bpe_artifact(wordpiece)
    with pytest.raises(BPEContractError, match="cannot be read as JSON"):
        inspect_bpe_artifact(corrupt)


def test_bpe_cache_reuses_only_exact_contract_and_inputs(tmp_path):
    path = tmp_path / "bpe.json"
    trainer = BPETokenizerTrainer(vocab_size=32, show_progress=False)
    trainer.ensure_trained(TOY_CORPUS, path, sequence_role="sequence")
    original = path.read_bytes()

    trainer.ensure_trained(TOY_CORPUS, path, sequence_role="sequence")
    assert trainer.last_cache_reused is True
    assert path.read_bytes() == original

    trainer.ensure_trained(TOY_CORPUS + ["AAAA"], path, sequence_role="sequence")
    assert trainer.last_cache_reused is False
    trainer.ensure_trained(TOY_CORPUS + ["AAAA"], path, sequence_role="sequence")
    assert trainer.last_cache_reused is True

    provenance_path = bpe_provenance_path(path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["runtime"]["tokenizers_version"] = "0.0.0"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    trainer.ensure_trained(TOY_CORPUS + ["AAAA"], path, sequence_role="sequence")
    assert trainer.last_cache_reused is False

    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    trainer.ensure_trained(TOY_CORPUS + ["AAAA"], path, sequence_role="sequence")
    assert trainer.last_cache_reused is False

    larger = BPETokenizerTrainer(vocab_size=33, show_progress=False)
    larger.ensure_trained(TOY_CORPUS + ["AAAA"], path, sequence_role="sequence")
    assert larger.last_cache_reused is False
    assert larger.last_provenance["trainer"]["vocab_size"] == 33


def test_bpe_invalid_rebuild_input_preserves_existing_cache(tmp_path):
    path = tmp_path / "bpe.json"
    train_toy(path)
    provenance_path = bpe_provenance_path(path)
    before = (path.read_bytes(), provenance_path.read_bytes())
    trainer = BPETokenizerTrainer(vocab_size=32, show_progress=False)

    with pytest.raises(BPEContractError, match="unsupported character"):
        trainer.ensure_trained(["ACD*"], path)

    assert (path.read_bytes(), provenance_path.read_bytes()) == before


def test_bpe_two_file_cache_commit_rolls_back_on_replace_failure(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "bpe.json"
    provenance = bpe_provenance_path(destination)
    destination.write_bytes(b"old-tokenizer")
    provenance.write_bytes(b"old-provenance")
    staged_tokenizer = tmp_path / "staged.json"
    staged_provenance = tmp_path / "staged.provenance.json"
    staged_tokenizer.write_bytes(b"new-tokenizer")
    staged_provenance.write_bytes(b"new-provenance")
    real_replace = os.replace
    calls = 0

    def fail_second_replace(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated provenance replace failure")
        return real_replace(source, target)

    monkeypatch.setattr(os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="simulated"):
        BPETokenizerTrainer._commit_staged_cache(
            staged_tokenizer,
            staged_provenance,
            destination,
        )

    assert destination.read_bytes() == b"old-tokenizer"
    assert provenance.read_bytes() == b"old-provenance"


def test_bpe_bundle_is_portable_and_validated(tmp_path):
    cache_dir = tmp_path / "cache"
    bundle = tmp_path / "bundle"
    cache_dir.mkdir()
    source = cache_dir / "bpe.json"
    train_toy(source)
    classic = {
        "tokenizer": "bpe",
        "encoder": "index",
        "max_length": 10,
        **bundle_bpe_artifact(source, bundle),
    }
    metadata = {"classic": classic, "args": {}}

    source.unlink()
    bpe_provenance_path(source).unlink()

    resolved = ModelService._resolve_bpe_path(metadata, bundle=bundle)
    assert Path(resolved) == bundle / "tokenizers" / "bpe.json"
    featurizer, encoder = ModelService._build_classic_featurizer(
        metadata,
        bundle=bundle,
    )
    assert encoder == "index"
    assert featurizer.encode("ACGT").shape == (10,)

    bundled_path = Path(resolved)
    bundled_path.write_text(
        bundled_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        InferenceUnsupportedError,
        match="SHA-256 mismatch",
    ) as mismatch:
        ModelService._resolve_bpe_path(metadata, bundle=bundle)
    assert mismatch.value.code == "bpe_artifact_digest_mismatch"


def test_bpe_bundle_rejects_missing_and_traversal_paths(tmp_path):
    descriptor = {
        "contract_version": BPE_CONTRACT_VERSION,
        "path": "tokenizers/missing.json",
        "sha256": "0" * 64,
        "provenance_path": "tokenizers/missing.provenance.json",
        "model_type": "BPE",
        "variant": BPE_VARIANT,
        "vocab_size": 32,
        "tokenizers_version": BPE_TOKENIZERS_VERSION,
    }
    metadata = {
        "classic": {
            "tokenizer": "bpe",
            "bpe_artifact": descriptor,
        }
    }
    with pytest.raises(InferenceUnsupportedError) as missing:
        ModelService._resolve_bpe_path(metadata, bundle=tmp_path)
    assert missing.value.code == "bpe_artifact_missing"

    traversal = copy.deepcopy(metadata)
    traversal["classic"]["bpe_artifact"]["path"] = "../outside.json"
    with pytest.raises(InferenceUnsupportedError) as escaped:
        ModelService._resolve_bpe_path(traversal, bundle=tmp_path)
    assert escaped.value.code == "bpe_artifact_invalid"


def test_pair_artifacts_embed_homogeneous_and_heterogeneous_bpe_roles(tmp_path):
    peptide_source = tmp_path / "peptide.json"
    partner_source = tmp_path / "partner.json"
    train_toy(peptide_source)
    BPETokenizerTrainer(vocab_size=32, show_progress=False).train_sequences(
        ["GGGG", "GGAC", "GTGT"],
        partner_source,
        sequence_role="partner",
    )
    bundle = tmp_path / "bundle"

    homogeneous_args = SimpleNamespace(
        tokenizer="bpe",
        encoder="index",
        bpe_path=str(peptide_source),
        artifact_dir=str(bundle),
    )
    homogeneous = train_pair_pipeline._classic_extra_artifact_info(homogeneous_args)
    assert homogeneous["bpe_path"] == "tokenizers/bpe.json"
    assert homogeneous["bpe_artifact"]["variant"] == BPE_VARIANT

    heterogeneous_args = SimpleNamespace(
        artifact_dir=str(bundle),
        streams={
            "peptide": {
                "tokenizer": "bpe",
                "encoder": "index",
                "max_length": 10,
                "params": {},
            },
            "partner": {
                "tokenizer": "bpe",
                "encoder": "index",
                "max_length": 10,
                "params": {},
            },
        },
        _stream_runtime={
            "peptide": {"resolved_params": {"bpe_path": str(peptide_source)}},
            "partner": {"resolved_params": {"bpe_path": str(partner_source)}},
        },
    )
    peptide_metadata = train_pair_pipeline._stream_artifact_metadata(
        heterogeneous_args,
        "peptide",
        torch.ones((1, 10), dtype=torch.long),
        "token",
    )
    partner_metadata = train_pair_pipeline._stream_artifact_metadata(
        heterogeneous_args,
        "partner",
        torch.ones((1, 10), dtype=torch.long),
        "token",
    )

    assert peptide_metadata["spec"]["params"]["bpe_path"] == "tokenizers/peptide_bpe.json"
    assert partner_metadata["spec"]["params"]["bpe_path"] == "tokenizers/partner_bpe.json"
    assert (bundle / "tokenizers" / "peptide_bpe.json").is_file()
    assert (bundle / "tokenizers" / "partner_bpe.json").is_file()


def test_sequence_artifact_embeds_bpe_and_is_independent_of_source(tmp_path):
    source = tmp_path / "source" / "bpe.json"
    source.parent.mkdir()
    train_toy(source)
    bundle = tmp_path / "sequence_bundle"
    pipeline = train_sequence_pipeline.SequencePipeline.__new__(
        train_sequence_pipeline.SequencePipeline
    )
    pipeline.args = SimpleNamespace(
        tokenizer="bpe",
        encoder="index",
        max_length=10,
        artifact_dir=str(bundle),
    )
    pipeline.bpe_path = str(source)
    train_data = SimpleNamespace(label_names=["activity"])

    info = pipeline._sequence_feature_artifact_info(train_data)
    source.unlink()
    bpe_provenance_path(source).unlink()

    assert info["classic"]["bpe_path"] == "tokenizers/bpe.json"
    resolved = ModelService._resolve_bpe_path(
        {"classic": info["classic"], "args": {}},
        bundle=bundle,
    )
    assert Path(resolved).is_file()


def test_bpe_factory_route_matches_canonical_tokenizer(tmp_path):
    path = tmp_path / "bpe.json"
    train_toy(path)
    tokenizer = BPETokenizer(path, max_length=10)
    canonical = FeaturizerFactory(bpe_path=str(path)).create(
        "bpe",
        "index",
        10,
        2,
    )

    expected = torch.tensor(tokenizer.tokenize("ac ac gt"), dtype=torch.long)
    assert torch.equal(expected, canonical.encode("ac ac gt"))
