import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer

from pepplatform.featurizers import FeaturizerFactory
from pepplatform.tokenizers import (
    WordPieceTokenizer,
    WordPieceTokenizerTrainer,
)
from pepplatform.web.inference import (
    InferenceUnsupportedError,
    ModelService,
)
from pepplatform.wordpiece import (
    WORDPIECE_ABSOLUTE_MAX_RAW_LENGTH,
    WORDPIECE_ALPHABET,
    WORDPIECE_CONTRACT_VERSION,
    WORDPIECE_DEFAULT_MAX_RAW_LENGTH,
    WORDPIECE_SPECIAL_TOKEN_IDS,
    WORDPIECE_TOKENIZERS_VERSION,
    WORDPIECE_VARIANT,
    WordPieceContractError,
    bundle_wordpiece_artifact,
    inspect_wordpiece_artifact,
    normalize_wordpiece_sequence,
    wordpiece_provenance_path,
)
from scripts import train_pair_pipeline


TOY_CORPUS = [
    "ACACAC",
    "ACAC",
    "ACGT",
    "MNPQRSTVWY",
    "DEFGHIKL",
]


def train_toy(path: Path, *, vocab_size=64, max_raw_length=50_000):
    trainer = WordPieceTokenizerTrainer(
        vocab_size=vocab_size,
        max_raw_length=max_raw_length,
        show_progress=False,
    )
    trainer.train_sequences(TOY_CORPUS, path)
    return trainer


def test_wordpiece_contract_and_residue_fallbacks(tmp_path):
    path = tmp_path / "wordpiece.json"
    trainer = train_toy(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["model"]["type"] == "WordPiece"
    assert payload["model"]["continuing_subword_prefix"] == "##"
    assert payload["model"]["max_input_chars_per_word"] == WORDPIECE_DEFAULT_MAX_RAW_LENGTH
    assert payload["pre_tokenizer"]["type"] == "WhitespaceSplit"
    assert {
        token: payload["model"]["vocab"][token] for token in WORDPIECE_SPECIAL_TOKEN_IDS
    } == WORDPIECE_SPECIAL_TOKEN_IDS
    for residue in WORDPIECE_ALPHABET:
        assert residue in payload["model"]["vocab"]
        assert f"##{residue}" in payload["model"]["vocab"]

    inspection = inspect_wordpiece_artifact(path)
    assert inspection["model_type"] == "WordPiece"
    assert inspection["vocab_size"] <= trainer.vocab_size


def test_wordpiece_normalization_and_invalid_input(tmp_path):
    path = tmp_path / "wordpiece.json"
    train_toy(path)
    tokenizer = WordPieceTokenizer(path, max_length=16)

    assert normalize_wordpiece_sequence(" acd\n efg\t") == "ACDEFG"
    assert normalize_wordpiece_sequence("BZJUO") == "XXXXX"
    assert tokenizer.tokenize("ac bzjuo") == tokenizer.tokenize("ACXXXXX")
    assert WORDPIECE_SPECIAL_TOKEN_IDS["<UNK>"] not in tokenizer.tokenize(
        "".join(WORDPIECE_ALPHABET)
    )

    with pytest.raises(WordPieceContractError, match="empty"):
        tokenizer.tokenize(" \n\t ")
    with pytest.raises(WordPieceContractError, match="unsupported character"):
        tokenizer.tokenize("ACD*")


def test_wordpiece_long_sequence_limits_are_explicit(tmp_path):
    path = tmp_path / "wordpiece.json"
    train_toy(path)
    tokenizer = WordPieceTokenizer(path, max_length=32)

    ids = tokenizer.tokenize("A" * 101)
    assert len(ids) == 32
    assert ids[0] == WORDPIECE_SPECIAL_TOKEN_IDS["<CLS>"]
    assert ids[-1] == WORDPIECE_SPECIAL_TOKEN_IDS["<SEP>"]
    assert WORDPIECE_SPECIAL_TOKEN_IDS["<UNK>"] not in ids

    long_ids = tokenizer.tokenize("A" * WORDPIECE_DEFAULT_MAX_RAW_LENGTH)
    assert len(long_ids) == 32
    assert WORDPIECE_SPECIAL_TOKEN_IDS["<UNK>"] not in long_ids

    with pytest.raises(
        WordPieceContractError,
        match="wordpiece_max_raw_length=50000",
    ):
        tokenizer.tokenize("A" * (WORDPIECE_DEFAULT_MAX_RAW_LENGTH + 1))

    expanded = WordPieceTokenizer(
        path,
        max_length=32,
        max_raw_length=WORDPIECE_ABSOLUTE_MAX_RAW_LENGTH,
    )
    assert len(expanded.tokenize("A" * (WORDPIECE_DEFAULT_MAX_RAW_LENGTH + 1))) == 32

    with pytest.raises(WordPieceContractError, match=r"\[1, 100000\]"):
        WordPieceTokenizer(
            path,
            max_raw_length=WORDPIECE_ABSOLUTE_MAX_RAW_LENGTH + 1,
        )


@pytest.mark.parametrize("max_length", [2, 3, 4, 5, 10, 32])
def test_trie_runtime_matches_huggingface_wordpiece(tmp_path, max_length):
    path = tmp_path / "wordpiece.json"
    train_toy(path)
    platform_tokenizer = WordPieceTokenizer(path, max_length=max_length)
    native_tokenizer = Tokenizer.from_file(str(path))
    native_tokenizer.model.max_input_chars_per_word = 1_000
    native_tokenizer.enable_padding(
        length=max_length,
        pad_id=WORDPIECE_SPECIAL_TOKEN_IDS["<PAD>"],
        pad_token="<PAD>",
    )
    native_tokenizer.enable_truncation(max_length=max_length)

    for sequence in [
        "ACACGT",
        "MNPQRSTVWY",
        "A" * 101,
        "bzjuo",
        "ACDEFGHIKLMNPQRSTVWY",
    ]:
        normalized = normalize_wordpiece_sequence(sequence)
        assert platform_tokenizer.tokenize(sequence) == native_tokenizer.encode(normalized).ids


def test_wordpiece_training_provenance_and_cache_are_strict(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    train_toy(first)
    train_toy(second)
    first_provenance = json.loads(wordpiece_provenance_path(first).read_text(encoding="utf-8"))
    second_provenance = json.loads(wordpiece_provenance_path(second).read_text(encoding="utf-8"))
    assert first_provenance["training_digest"] == second_provenance["training_digest"]
    assert first_provenance["trainer"]["vocab_size"] == 64
    assert second_provenance["trainer"]["vocab_size"] == 64
    inspect_wordpiece_artifact(
        first,
        expected_sha256=first_provenance["artifact_sha256"],
    )
    inspect_wordpiece_artifact(
        second,
        expected_sha256=second_provenance["artifact_sha256"],
    )

    trainer = WordPieceTokenizerTrainer(
        vocab_size=64,
        show_progress=False,
    )
    trainer.ensure_trained(TOY_CORPUS, first, sequence_role="sequence")
    assert trainer.last_cache_reused is True
    trainer.ensure_trained(
        TOY_CORPUS + ["AAAA"],
        first,
        sequence_role="sequence",
    )
    assert trainer.last_cache_reused is False
    trainer.ensure_trained(
        TOY_CORPUS + ["AAAA"],
        first,
        sequence_role="sequence",
    )
    assert trainer.last_cache_reused is True

    provenance_path = wordpiece_provenance_path(first)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["runtime"]["tokenizers_version"] = "0.0.0"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    trainer.ensure_trained(
        TOY_CORPUS + ["AAAA"],
        first,
        sequence_role="sequence",
    )
    assert trainer.last_cache_reused is False


def test_wordpiece_invalid_rebuild_preserves_existing_cache(tmp_path):
    path = tmp_path / "wordpiece.json"
    train_toy(path)
    provenance_path = wordpiece_provenance_path(path)
    before = (path.read_bytes(), provenance_path.read_bytes())
    trainer = WordPieceTokenizerTrainer(
        vocab_size=64,
        show_progress=False,
    )

    with pytest.raises(WordPieceContractError, match="unsupported character"):
        trainer.ensure_trained(["ACD*"], path)

    assert (path.read_bytes(), provenance_path.read_bytes()) == before


def test_wordpiece_commit_rolls_back_both_cache_files(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "wordpiece.json"
    train_toy(path)
    provenance_path = wordpiece_provenance_path(path)
    before = (path.read_bytes(), provenance_path.read_bytes())
    real_replace = __import__("os").replace

    def fail_provenance_commit(source, destination):
        if Path(destination) == provenance_path:
            raise OSError("simulated provenance commit failure")
        return real_replace(source, destination)

    monkeypatch.setattr(
        "pepplatform.tokenizers.os.replace",
        fail_provenance_commit,
    )
    trainer = WordPieceTokenizerTrainer(
        vocab_size=64,
        show_progress=False,
    )
    with pytest.raises(
        OSError,
        match="simulated provenance commit failure",
    ):
        trainer.ensure_trained(TOY_CORPUS + ["AAAA"], path)

    assert (path.read_bytes(), provenance_path.read_bytes()) == before


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["model"].update(type="BPE"),
        lambda payload: payload["model"].update(continuing_subword_prefix="_"),
        lambda payload: payload["model"].update(max_input_chars_per_word=100_001),
        lambda payload: payload["pre_tokenizer"].update(type="ByteLevel"),
        lambda payload: payload["model"]["vocab"].update({"<PAD>": 9}),
        lambda payload: payload["model"]["vocab"].update(
            {"##X": len(payload["model"]["vocab"]) + 1}
        ),
        lambda payload: payload.update(post_processor=None),
        lambda payload: payload["post_processor"].update(single=[]),
    ],
)
def test_wordpiece_contract_rejects_incompatible_json(
    tmp_path,
    mutation,
):
    valid = tmp_path / "valid.json"
    invalid = tmp_path / "invalid.json"
    train_toy(valid)
    payload = json.loads(valid.read_text(encoding="utf-8"))
    mutation(payload)
    invalid.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(WordPieceContractError):
        inspect_wordpiece_artifact(invalid)


def test_wordpiece_bundle_is_portable_and_validated(tmp_path):
    source = tmp_path / "wordpiece.json"
    bundle = tmp_path / "bundle"
    train_toy(source)
    classic = {
        "tokenizer": "wordpiece",
        "encoder": "index",
        "max_length": 10,
        "wordpiece_max_raw_length": 50_000,
        **bundle_wordpiece_artifact(source, bundle),
    }
    metadata = {"classic": classic, "args": {}}

    source.unlink()
    wordpiece_provenance_path(source).unlink()

    resolved = ModelService._resolve_wordpiece_path(metadata, bundle=bundle)
    assert Path(resolved) == bundle / "tokenizers" / "wordpiece.json"
    featurizer, encoder = ModelService._build_classic_featurizer(
        metadata,
        bundle=bundle,
    )
    assert encoder == "index"
    assert featurizer.encode("ACGT").shape == (10,)

    bundled = Path(resolved)
    bundled.write_text(
        bundled.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        InferenceUnsupportedError,
        match="SHA-256 mismatch",
    ) as mismatch:
        ModelService._resolve_wordpiece_path(metadata, bundle=bundle)
    assert mismatch.value.code == "wordpiece_artifact_digest_mismatch"


def test_wordpiece_bundle_rejects_mutated_provenance(tmp_path):
    source = tmp_path / "wordpiece.json"
    bundle = tmp_path / "bundle"
    train_toy(source)
    classic = {
        "tokenizer": "wordpiece",
        "encoder": "index",
        "max_length": 10,
        "wordpiece_max_raw_length": 50_000,
        **bundle_wordpiece_artifact(source, bundle),
    }
    metadata = {"classic": classic, "args": {}}
    provenance_path = bundle / classic["wordpiece_artifact"]["provenance_path"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["training_algorithm"] = "not-wordpiece"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(
        InferenceUnsupportedError,
        match="incompatible with WordPiece contract",
    ):
        ModelService._resolve_wordpiece_path(metadata, bundle=bundle)


def test_wordpiece_bundle_rejects_traversal_path(tmp_path):
    descriptor = {
        "contract_version": WORDPIECE_CONTRACT_VERSION,
        "path": "../outside.json",
        "sha256": "0" * 64,
        "provenance_path": "tokenizers/wordpiece.provenance.json",
        "model_type": "WordPiece",
        "variant": WORDPIECE_VARIANT,
        "vocab_size": 64,
        "tokenizers_version": WORDPIECE_TOKENIZERS_VERSION,
    }
    metadata = {
        "classic": {
            "tokenizer": "wordpiece",
            "wordpiece_artifact": descriptor,
        }
    }
    with pytest.raises(InferenceUnsupportedError) as escaped:
        ModelService._resolve_wordpiece_path(metadata, bundle=tmp_path)
    assert escaped.value.code == "wordpiece_artifact_invalid"


def test_pair_artifacts_embed_heterogeneous_wordpiece_roles(tmp_path):
    peptide_source = tmp_path / "peptide.json"
    partner_source = tmp_path / "partner.json"
    train_toy(peptide_source)
    WordPieceTokenizerTrainer(
        vocab_size=64,
        show_progress=False,
    ).train_sequences(
        ["GGGG", "GGAC", "GTGT"],
        partner_source,
        sequence_role="partner",
    )
    bundle = tmp_path / "bundle"
    args = type(
        "Args",
        (),
        {
            "artifact_dir": str(bundle),
            "streams": {
                "peptide": {
                    "tokenizer": "wordpiece",
                    "encoder": "index",
                    "max_length": 10,
                    "params": {},
                },
                "partner": {
                    "tokenizer": "wordpiece",
                    "encoder": "index",
                    "max_length": 10,
                    "params": {},
                },
            },
            "_stream_runtime": {
                "peptide": {
                    "resolved_params": {
                        "wordpiece_path": str(peptide_source),
                        "wordpiece_max_raw_length": 50_000,
                    }
                },
                "partner": {
                    "resolved_params": {
                        "wordpiece_path": str(partner_source),
                        "wordpiece_max_raw_length": 50_000,
                    }
                },
            },
        },
    )()

    peptide_metadata = train_pair_pipeline._stream_artifact_metadata(
        args,
        "peptide",
        torch.ones((1, 10), dtype=torch.long),
        "token",
    )
    partner_metadata = train_pair_pipeline._stream_artifact_metadata(
        args,
        "partner",
        torch.ones((1, 10), dtype=torch.long),
        "token",
    )

    assert (
        peptide_metadata["spec"]["params"]["wordpiece_path"] == "tokenizers/peptide_wordpiece.json"
    )
    assert (
        partner_metadata["spec"]["params"]["wordpiece_path"] == "tokenizers/partner_wordpiece.json"
    )
    assert (bundle / "tokenizers" / "peptide_wordpiece.json").is_file()
    assert (bundle / "tokenizers" / "partner_wordpiece.json").is_file()


def test_wordpiece_factory_route_matches_direct_tokenizer(tmp_path):
    path = tmp_path / "wordpiece.json"
    train_toy(path)
    direct = WordPieceTokenizer(path, max_length=10)
    factory = FeaturizerFactory(
        bpe_path="",
        wordpiece_path=str(path),
    ).create("wordpiece", "index", 10, 2)

    assert torch.equal(
        torch.tensor(direct.tokenize("ACGT")),
        factory.encode("ACGT"),
    )
