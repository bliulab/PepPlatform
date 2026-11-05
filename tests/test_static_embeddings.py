import json

import numpy as np
import pytest
import torch

from pepplatform.encoders import FastTextEncoder, GloveEncoder, Word2VecEncoder
from pepplatform.static_embeddings import (
    ensure_fasttext_memory_budget,
    fasttext_cache_signature,
    word2vec_cache_signature,
)
from tools.train_static_embeddings import (
    save_cache_meta,
    save_fasttext_model,
    train_fasttext,
    train_word2vec,
)


def _sentences():
    return [
        ["AC", "CD", "DE", "EF"],
        ["AC", "CD", "DF", "FG"],
        ["GH", "HI", "IK", "KL"],
    ] * 4


def test_word2vec_single_worker_training_is_reproducible():
    first = train_word2vec(_sentences(), vector_size=8, window=2, min_count=1, epochs=3, workers=1)
    second = train_word2vec(_sentences(), vector_size=8, window=2, min_count=1, epochs=3, workers=1)

    assert first.wv.index_to_key == second.wv.index_to_key
    np.testing.assert_array_equal(first.wv.vectors, second.wv.vectors)
    assert word2vec_cache_signature(workers=1)["deterministic"] is True
    assert word2vec_cache_signature(workers=2)["deterministic"] is False


@pytest.mark.parametrize(
    "sentences",
    [
        "AA BB",
        ["AA BB"],
        [["AA", "bad token"]],
        [["AA", ""]],
    ],
)
def test_predictive_training_rejects_ambiguous_token_corpora(sentences):
    with pytest.raises(ValueError):
        train_word2vec(
            sentences,
            vector_size=4,
            window=2,
            min_count=1,
            epochs=1,
            workers=1,
        )


def test_predictive_training_rejects_invalid_seed():
    with pytest.raises(ValueError, match="seed"):
        train_word2vec(
            _sentences(),
            vector_size=4,
            window=2,
            min_count=1,
            epochs=1,
            workers=1,
            seed=-1,
        )


def test_word2vec_loader_requires_strict_header_and_rows(tmp_path):
    missing_header = tmp_path / "missing-header.vec"
    missing_header.write_text("AA 0.1 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must begin"):
        Word2VecEncoder(max_length=2, embedding_path=missing_header)

    bad_count = tmp_path / "bad-count.vec"
    bad_count.write_text("2 2\nAA 0.1 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="declares 2 vectors"):
        Word2VecEncoder(max_length=2, embedding_path=bad_count)

    non_finite = tmp_path / "non-finite.vec"
    non_finite.write_text("1 2\nAA nan 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        Word2VecEncoder(max_length=2, embedding_path=non_finite)


def test_text_embedding_coverage_is_observable(tmp_path):
    path = tmp_path / "glove.vec"
    path.write_text("AA 1 2\n<UNK> 0.25 0.5\n", encoding="utf-8")
    encoder = GloveEncoder(max_length=2, embedding_path=path)

    encoded = encoder.encode(["AA", "ZZ"])

    assert torch.equal(encoded[1], torch.tensor([0.25, 0.5]))
    assert encoder.coverage_stats() == {
        "lookups": 2,
        "in_vocabulary": 1,
        "oov": 1,
        "coverage": 0.5,
        "oov_policy": "provided_unk_else_null_vector",
    }


def test_fasttext_binary_roundtrip_preserves_subword_oov(tmp_path):
    model = train_fasttext(
        _sentences(),
        vector_size=8,
        window=2,
        min_count=1,
        epochs=3,
        workers=1,
        bucket=101,
        memory_limit_gib=0.1,
    )
    path = tmp_path / "fasttext.bin"
    save_fasttext_model(model, path)

    encoder = FastTextEncoder(max_length=2, embedding_path=path)
    oov = encoder.lookup("ACD")

    assert "ACD" not in encoder.keyed_vectors.key_to_index
    assert oov.shape == (8,)
    assert torch.isfinite(oov).all()
    assert not torch.equal(oov, torch.zeros_like(oov))
    assert encoder.min_n == 3
    assert encoder.max_n == 6
    assert encoder.bucket == 101
    assert encoder.coverage_stats()["subword_synthesized"] == 1
    signature = fasttext_cache_signature(workers=1, bucket=101)
    assert signature["artifact_format"] == "facebook_fasttext_binary"
    assert signature["oov_policy"] == "character_ngram_synthesis"


def test_fasttext_rejects_legacy_text_vectors(tmp_path):
    path = tmp_path / "fasttext.vec"
    path.write_text("1 2\nAA 0.1 0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="omit character n-gram state"):
        FastTextEncoder(max_length=2, embedding_path=path)


def test_fasttext_memory_budget_is_enforced():
    with pytest.raises(ValueError, match="exceeding"):
        ensure_fasttext_memory_budget(
            bucket=2_000_000,
            vector_size=128,
            vocab_size=100,
            limit_gib=0.01,
        )


def test_cache_metadata_binds_artifact_digest(tmp_path):
    artifact = tmp_path / "word2vec.vec"
    artifact.write_text("1 2\nAA 0.1 0.2\n", encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "tokenizer_mode": "kmer",
            "k": 2,
            "dr_max_distance": 5,
            "vector_size": 2,
            "window": 2,
            "min_count": 1,
            "epochs": 1,
            "workers": 1,
            "seed": 42,
            "train_sha256": "abc",
            "cache_suffix": "",
        },
    )()

    save_cache_meta(
        str(tmp_path),
        "word2vec",
        args,
        str(artifact),
        extra=word2vec_cache_signature(workers=1),
    )

    metadata = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert metadata["artifact_file"] == "word2vec.vec"
    assert len(metadata["artifact_sha256"]) == 64
    assert metadata["artifact_size_bytes"] == artifact.stat().st_size
    assert metadata["algorithm_verified"] is True
    assert metadata["verification"] == "internal_contract"
