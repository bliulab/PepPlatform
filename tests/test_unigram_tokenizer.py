import json
import math
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from pepplatform.dataset import PeptideDataset
from pepplatform.featurizers import FeaturizerFactory
from pepplatform.tokenizers import UnigramTokenizer, UnigramTokenizerTrainer
from pepplatform.unigram import (
    UNIGRAM_ALPHABET,
    UNIGRAM_CONTRACT_VERSION,
    UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
    UNIGRAM_SPECIAL_TOKEN_IDS,
    UNIGRAM_TOKENIZERS_VERSION,
    SeededUnigramSampler,
    UnigramContractError,
    bundle_unigram_artifact,
    inspect_unigram_artifact,
    load_unigram_provenance,
    normalize_unigram_sequence,
    unigram_provenance_path,
    validate_unigram_sampling,
)
from tools.train_unigram_tokenizer import load_fasta_sequences
from pepplatform.web.inference import InferenceUnsupportedError, ModelService
from scripts.train_pair_pipeline import (
    _EpochSampledPairDataset,
    configure_joined_unigram_sampling,
)


CORPUS = [
    "ACDEFGHIKLMNPQRSTVWY" * 4,
    "ACDEACDEACDEFGHIKLMN",
    "PQRSTVWYPQRSTVWYACDE",
    "XXBZJUOACDEXXBZJUO",
]


def train_unigram(tmp_path, name="unigram.json", vocab_size=48):
    path = tmp_path / name
    trainer = UnigramTokenizerTrainer(
        vocab_size=vocab_size,
        show_progress=False,
    )
    trainer.train_sequences(CORPUS, path)
    return path, trainer


def test_unigram_artifact_contract_and_viterbi(tmp_path):
    path, trainer = train_unigram(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    inspection = inspect_unigram_artifact(path)
    provenance = load_unigram_provenance(path)

    assert inspection["contract_version"] == UNIGRAM_CONTRACT_VERSION
    assert payload["model"]["type"] == "Unigram"
    assert payload["model"]["unk_id"] == 1
    assert payload["model"]["byte_fallback"] is False
    assert payload["normalizer"] is None
    assert payload["pre_tokenizer"] is None
    assert {
        token: payload["model"]["vocab"][token_id][0]
        for token, token_id in UNIGRAM_SPECIAL_TOKEN_IDS.items()
    } == {token: token for token in UNIGRAM_SPECIAL_TOKEN_IDS}
    vocab = {entry[0] for entry in payload["model"]["vocab"]}
    assert set(UNIGRAM_ALPHABET).issubset(vocab)
    assert provenance["feature_fit_scope"] == "train_only"
    assert provenance["runtime"]["tokenizers_version"] == UNIGRAM_TOKENIZERS_VERSION
    assert trainer.last_provenance == provenance

    tokenizer = UnigramTokenizer(path, max_length=20)
    ids = tokenizer.tokenize(" acde\nxbzjuo ")
    assert len(ids) == 20
    assert ids[0] == UNIGRAM_SPECIAL_TOKEN_IDS["<CLS>"]
    assert ids[next(index for index, value in enumerate(ids) if value == 3)] == 3
    assert UNIGRAM_SPECIAL_TOKEN_IDS["<UNK>"] not in ids


def test_unigram_training_is_byte_deterministic(tmp_path):
    hashes = []
    bytes_by_run = []
    for index in range(6):
        path, trainer = train_unigram(tmp_path, f"unigram-{index}.json")
        hashes.append(trainer.last_provenance["artifact_sha256"])
        bytes_by_run.append(path.read_bytes())
    assert len(set(hashes)) == 1
    assert all(value == bytes_by_run[0] for value in bytes_by_run)


def test_unigram_cache_is_strict_and_reuses_only_exact_contract(tmp_path):
    path, first = train_unigram(tmp_path)
    second = UnigramTokenizerTrainer(vocab_size=48, show_progress=False)
    second.ensure_trained(CORPUS, path)
    assert second.last_cache_reused is True

    changed = UnigramTokenizerTrainer(vocab_size=49, show_progress=False)
    changed.ensure_trained(CORPUS, path)
    assert changed.last_cache_reused is False

    provenance_path = unigram_provenance_path(path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["runtime"]["tokenizers_version"] = "0.0.0"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    rebuilt = UnigramTokenizerTrainer(vocab_size=49, show_progress=False)
    rebuilt.ensure_trained(CORPUS, path)
    assert rebuilt.last_cache_reused is False


def test_unigram_normalization_and_length_fail_closed(tmp_path):
    assert normalize_unigram_sequence(" acd\ne ") == "ACDE"
    with pytest.raises(UnigramContractError, match="empty"):
        normalize_unigram_sequence(" \n\t")
    with pytest.raises(UnigramContractError, match="unsupported"):
        normalize_unigram_sequence("ACD*")
    with pytest.raises(UnigramContractError, match="exceeding"):
        normalize_unigram_sequence("A" * (UNIGRAM_DEFAULT_MAX_RAW_LENGTH + 1))

    path, _ = train_unigram(tmp_path)
    with pytest.raises(UnigramContractError, match="max_length"):
        UnigramTokenizer(path, max_length=1)


def _brute_force_paths(vocab, sequence):
    pieces = [
        (index, token, float(score))
        for index, (token, score) in enumerate(vocab)
        if token not in UNIGRAM_SPECIAL_TOKEN_IDS
    ]

    def visit(offset):
        if offset == len(sequence):
            return [(0.0, ())]
        result = []
        for token_id, token, score in pieces:
            if sequence.startswith(token, offset):
                for suffix_score, suffix_ids in visit(offset + len(token)):
                    result.append((score + suffix_score, (token_id,) + suffix_ids))
        return result

    return sorted(visit(0), key=lambda item: (-item[0], item[1]))


def test_seeded_sampler_matches_exact_nbest_and_is_reproducible(tmp_path):
    path, _ = train_unigram(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    sampler = SeededUnigramSampler(path)
    sequence = "ACDEACDE"
    expected = _brute_force_paths(payload["model"]["vocab"], sequence)[:16]
    actual = sampler.nbest_paths(sequence, nbest_size=16)
    assert len(actual) == len(expected)
    for (actual_score, actual_ids), (expected_score, expected_ids) in zip(
        actual,
        expected,
    ):
        assert actual_ids == expected_ids
        assert math.isclose(actual_score, expected_score, abs_tol=1e-12)

    tokenizer = UnigramTokenizer(path, max_length=20)
    first = tokenizer.tokenize_sampled(
        sequence,
        seed=123,
        epoch=4,
        sample_id="row-7",
        role="peptide",
        nbest_size=16,
        alpha=0.1,
    )
    second = tokenizer.tokenize_sampled(
        sequence,
        seed=123,
        epoch=4,
        sample_id="row-7",
        role="peptide",
        nbest_size=16,
        alpha=0.1,
    )
    assert first == second
    assert (
        len(
            {
                tuple(
                    tokenizer.tokenize_sampled(
                        sequence,
                        seed=123,
                        epoch=epoch,
                        sample_id="row-7",
                        role="peptide",
                        nbest_size=16,
                        alpha=0.1,
                    )
                )
                for epoch in range(32)
            }
        )
        > 1
    )


def test_sampler_uses_bounded_backpointers_for_long_sequences(tmp_path):
    path, _ = train_unigram(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    vocab = payload["model"]["vocab"]
    sequence = "ACDE" * 1024
    paths = SeededUnigramSampler(path).nbest_paths(
        sequence,
        nbest_size=8,
    )
    assert len(paths) == 8
    for _, token_ids in paths:
        assert "".join(vocab[token_id][0] for token_id in token_ids) == sequence


def test_sampling_parameters_are_strictly_bounded():
    with pytest.raises(UnigramContractError, match=r"\[2, 256\]"):
        validate_unigram_sampling(1, 0.1)
    with pytest.raises(UnigramContractError, match=r"\[2, 256\]"):
        validate_unigram_sampling(257, 0.1)
    with pytest.raises(UnigramContractError, match="finite and > 0"):
        validate_unigram_sampling(64, math.inf)
    with pytest.raises(UnigramContractError, match="integer"):
        validate_unigram_sampling(True, 0.1)


def test_unigram_sampler_requires_stable_sample_id(tmp_path):
    path, _ = train_unigram(tmp_path)
    sampler = SeededUnigramSampler(path)
    with pytest.raises(UnigramContractError, match="sample_id"):
        sampler.sample_ids(
            "ACDE",
            seed=42,
            epoch=0,
            sample_id=None,
        )


def test_unigram_bundle_is_portable_and_validated(tmp_path):
    path, _ = train_unigram(tmp_path)
    bundle = tmp_path / "bundle"
    descriptor = bundle_unigram_artifact(path, bundle)
    artifact = descriptor["unigram_artifact"]
    bundled = bundle / artifact["path"]
    bundled_provenance = bundle / artifact["provenance_path"]
    assert bundled.exists()
    assert bundled_provenance.exists()
    assert (
        inspect_unigram_artifact(
            bundled,
            expected_sha256=artifact["sha256"],
        )["vocab_size"]
        == artifact["vocab_size"]
    )
    assert Path(unigram_provenance_path(bundled)) == bundled_provenance
    UnigramTokenizer(str(bundled), max_length=32)

    payload = json.loads(bundled.read_text(encoding="utf-8"))
    payload["model"]["unk_id"] = None
    bundled.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UnigramContractError, match="unk_id"):
        inspect_unigram_artifact(bundled)


def test_web_resolves_only_strict_embedded_unigram_bundle(tmp_path):
    path, _ = train_unigram(tmp_path)
    bundle = tmp_path / "bundle"
    bundled = bundle_unigram_artifact(path, bundle)
    metadata = {
        "classic": {
            "tokenizer": "unigram",
            "unigram_max_raw_length": UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
            **bundled,
        }
    }
    resolved = ModelService._resolve_unigram_path(metadata, bundle=bundle)
    assert Path(resolved).resolve() == (bundle / bundled["unigram_artifact"]["path"]).resolve()
    UnigramTokenizer(resolved, max_length=32)

    invalid = json.loads(json.dumps(metadata))
    invalid["classic"]["unigram_artifact"]["path"] = "../escape.json"
    with pytest.raises(InferenceUnsupportedError, match="escapes"):
        ModelService._resolve_unigram_path(invalid, bundle=bundle)

    non_adjacent = json.loads(json.dumps(metadata))
    non_adjacent["classic"]["unigram_artifact"]["provenance_path"] = (
        "tokenizers/renamed.provenance.json"
    )
    (bundle / "tokenizers" / "renamed.provenance.json").write_bytes(
        (bundle / bundled["unigram_artifact"]["provenance_path"]).read_bytes()
    )
    with pytest.raises(InferenceUnsupportedError, match="adjacent"):
        ModelService._resolve_unigram_path(non_adjacent, bundle=bundle)

    with pytest.raises(InferenceUnsupportedError, match="missing"):
        ModelService._resolve_unigram_path(
            {"classic": {"tokenizer": "unigram"}},
            bundle=bundle,
        )


def test_sequence_dataset_samples_train_only_by_epoch(tmp_path):
    path, _ = train_unigram(tmp_path)
    fasta = tmp_path / "seqs.fasta"
    labels = tmp_path / "labels.csv"
    fasta.write_text(
        ">row-a\nACDEACDEACDE\n>row-b\nPQRSTVWYPQRSTVWY\n",
        encoding="utf-8",
    )
    labels.write_text("label\n1\n0\n", encoding="utf-8")
    factory = FeaturizerFactory(
        bpe_path="",
        unigram_path=str(path),
        unigram_max_raw_length=UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
    )
    featurizer = factory.create(
        "unigram",
        "index",
        20,
        2,
    )
    train_data = PeptideDataset(
        fasta,
        labels,
        featurizer=featurizer,
    )
    eval_data = PeptideDataset(
        fasta,
        labels,
        featurizer=featurizer,
    )
    train_data.enable_unigram_sampling(
        seed=42,
        nbest_size=16,
        alpha=0.1,
    )

    eval_first = eval_data[0][0]
    eval_second = eval_data[0][0]
    assert torch.equal(eval_first, eval_second)
    sampled = []
    for epoch in range(24):
        train_data.set_epoch(epoch)
        sampled.append(tuple(train_data[0][0].tolist()))
    assert len(set(sampled)) > 1
    train_data.set_epoch(7)
    assert torch.equal(train_data[0][0], train_data[0][0])


def test_offline_fasta_reader_excludes_headers_and_validation(tmp_path):
    fasta = tmp_path / "train.fasta"
    fasta.write_text(
        ">header with text\nACDE\nFG\n>second\nXXBZ\n",
        encoding="utf-8",
    )
    assert load_fasta_sequences(fasta) == ["ACDEFG", "XXBZ"]


def test_pair_dataset_reencodes_joined_tokens_per_epoch(tmp_path):
    path, _ = train_unigram(tmp_path)
    factory = FeaturizerFactory(
        bpe_path="",
        unigram_path=str(path),
    )
    featurizer = factory.create("unigram", "index", 20, 2)
    rows = [
        ("pair-a", "ACDEACDEACDE", "PQRSTVWYPQRSTVWY", 1),
        ("pair-b", "PQRSTVWYACDE", "ACDEACDEACDE", 0),
    ]
    args = Namespace(
        unigram_sampling=True,
        unigram_nbest_size=16,
        unigram_alpha=0.1,
        seed=91,
    )
    configure_joined_unigram_sampling(
        args,
        rows,
        featurizer,
        featurizer,
    )
    deterministic = torch.stack(
        [
            torch.cat(
                (
                    featurizer.encode(peptide, sample_id=f"{sample_id}::pep"),
                    featurizer.encode(partner, sample_id=f"{sample_id}::pro"),
                )
            )
            for sample_id, peptide, partner, _ in rows
        ]
    )
    labels = torch.tensor([[1.0], [0.0]])
    dataset = _EpochSampledPairDataset(
        (deterministic, labels),
        args._unigram_sampling_provider,
    )
    outputs = []
    for epoch in range(24):
        dataset.set_epoch(epoch)
        outputs.append(tuple(dataset[0][0].tolist()))
    assert len(set(outputs)) > 1
    dataset.set_epoch(5)
    assert torch.equal(dataset[0][0], dataset[0][0])
    gathered = dataset.gather_inputs([0, 1], input_count=1)
    assert gathered.shape == deterministic.shape
