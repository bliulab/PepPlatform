from unittest.mock import patch

import pytest
import torch

from pepplatform.cli_wizard import InteractiveConfigWizard
from pepplatform.featurizers import FeaturizerFactory
from pepplatform.kmer import KMER_MAX_K, KMER_MIN_K
from pepplatform.tokenizers import KmerTokenizer
from tools.train_static_embeddings import build_kmer_sentences


def test_kmer_tokenizer_uses_overlapping_stride_one_tokens_and_token_limit():
    tokenizer = KmerTokenizer(max_length=2, k=2)

    assert tokenizer.tokenize("acde") == ["AC", "CD"]
    assert tokenizer.to_indices(tokenizer.tokenize("A")) == [0, 0]


def test_kmer_vocab_sizes_include_pad_and_unk():
    assert KmerTokenizer(k=KMER_MIN_K).vocab_size == 26 + 2
    assert KmerTokenizer(k=KMER_MAX_K).vocab_size == 26**3 + 2


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        ("", [0, 0, 0, 0, 0]),
        ("A", [0, 0, 0, 0, 0]),
        ("acdef", [3, 30, 57, 84, 0]),
        ("A*", [1, 0, 0, 0, 0]),
        ("ACDEFGH", [3, 30, 57, 84, 111]),
    ],
)
def test_kmer_index_factory_route_preserves_ids_padding_and_truncation(
    sequence,
    expected,
):
    featurizer = FeaturizerFactory(bpe_path="").create(
        "kmer",
        "index",
        max_length=5,
        k=2,
    )

    encoded = featurizer.encode(sequence)

    assert encoded.dtype == torch.long
    assert encoded.tolist() == expected


def test_kmer_factory_routes_share_the_canonical_token_stream():
    factory = FeaturizerFactory(bpe_path="")
    index = factory.create("kmer", "index", max_length=5, k=2)
    onehot = factory.create("kmer", "onehot", max_length=5, k=2)
    bow = factory.create("kmer", "bow", max_length=5, k=2)

    ids = index.encode("ACAC")
    onehot_features = onehot.encode("ACAC")
    bow_features = bow.encode("ACAC")

    assert ids.tolist() == [3, 28, 3, 0, 0]
    assert onehot_features.shape == (5, index.vocab_size)
    assert onehot_features.dtype == torch.float
    assert onehot_features[:3].argmax(dim=1).tolist() == ids[:3].tolist()
    assert torch.count_nonzero(onehot_features[3:]) == 0
    assert bow_features.dtype == torch.float
    assert torch.isclose(bow_features.sum(), torch.tensor(3.0))
    assert torch.isclose(bow_features[3], torch.tensor(2.0))
    assert torch.isclose(bow_features[28], torch.tensor(1.0))


@pytest.mark.parametrize("encoder", ["property", "physchem"])
def test_kmer_factory_rejects_removed_property_encoder(encoder):
    with pytest.raises(ValueError, match="Property encoder has been removed"):
        FeaturizerFactory(bpe_path="").create(
            "kmer",
            encoder,
            max_length=5,
            k=2,
        )


@pytest.mark.parametrize("k", [0, -1, 4, 1.5, "2", True])
def test_kmer_tokenizer_rejects_unsupported_k_before_vocab_enumeration(k):
    with patch("pepplatform.tokenizers.itertools.product") as product:
        with pytest.raises(
            ValueError,
            match=r"K-mer tokenizer k must be an integer in \[1, 3\]",
        ):
            KmerTokenizer(k=k)

    product.assert_not_called()


def test_kmer_property_and_embedding_helpers_share_the_same_bound():
    with pytest.raises(ValueError, match=r"integer in \[1, 3\]"):
        build_kmer_sentences(["ACDE"], 4)


def test_interactive_integer_prompt_honors_kmer_upper_bound(monkeypatch, capsys):
    responses = iter(["4", "3"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(responses))

    value = InteractiveConfigWizard().prompt_int("K-mer size", 2, 1, max_value=3)

    assert value == 3
    assert "Please enter an integer <= 3." in capsys.readouterr().out
