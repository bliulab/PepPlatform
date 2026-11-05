import hashlib
from types import SimpleNamespace

import pytest
import torch

from pepplatform.dr import (
    DR_ALPHABET,
    DR_CONTRACT_VERSION,
    DRContractError,
    dr_contract_metadata,
    dr_feature_dim,
    ensure_dr_dense_memory_budget,
    estimate_dr_dense_tensor_bytes,
    parse_dr_dense_memory_limit_gib,
    parse_dr_max_distance,
    parse_dr_max_raw_length,
    validate_dr_max_distance,
)
from pepplatform.featurizers import FeaturizerFactory
from pepplatform.tokenizers import DRTokenizer
from pepplatform.stream_spec import normalize_pair_streams
from pepplatform.web.inference import (
    InferenceUnsupportedError,
    ModelService,
)
from pepplatform.web.preflight import preflight_experiment
from tools.train_static_embeddings import (
    build_dr_sentences,
    resolve_embedding_dir,
)


def _factory(**kwargs):
    return FeaturizerFactory(bpe_path="", **kwargs)


def test_dr_v2_uses_paper_alphabet_dimension_and_position_major_stream():
    tokenizer = DRTokenizer(max_length=6, max_distance=2)

    assert "".join(DR_ALPHABET) == "ARDCQEHIGNLKMFPSTWYV"
    assert tokenizer.feature_dim == dr_feature_dim(2) == 820
    assert tokenizer.vocab_size == 822
    assert tokenizer.tokenize("aglp") == [
        "A",
        "A_1_G",
        "A_2_L",
        "G",
        "G_1_L",
        "G_2_P",
        "L",
        "L_1_P",
        "P",
    ]


def test_dr_distance_zero_is_a_valid_residue_only_representation():
    tokenizer = DRTokenizer(max_length=2, max_distance=0)

    assert tokenizer.feature_dim == 20
    assert tokenizer.tokenize("ACD") == ["A", "C", "D"]
    assert tokenizer.to_indices(tokenizer.tokenize("ACD")) == [
        tokenizer.dr_vocab["A"],
        tokenizer.dr_vocab["C"],
    ]


@pytest.mark.parametrize(
    "value",
    [-1, 201, 1.5, "5", True, None],
)
def test_dr_distance_validation_is_strict(value):
    with pytest.raises(DRContractError, match="integer"):
        validate_dr_max_distance(value)


def test_dr_cli_parsers_accept_text_without_weakening_core_validation():
    assert parse_dr_max_distance("5") == 5
    assert parse_dr_max_raw_length("50000") == 50_000
    assert parse_dr_dense_memory_limit_gib("0.5") == 0.5
    with pytest.raises(DRContractError):
        parse_dr_max_distance("5.0")


@pytest.mark.parametrize("sequence", ["AB", "AX", "A*", "", " \n\t"])
def test_dr_rejects_noncanonical_or_empty_sequences(sequence):
    with pytest.raises(DRContractError):
        DRTokenizer(max_length=10, max_distance=1).tokenize(sequence)


def test_dr_normalizes_case_and_whitespace_but_enforces_raw_limit():
    tokenizer = DRTokenizer(
        max_length=10,
        max_distance=1,
        max_raw_length=3,
    )

    assert tokenizer.tokenize(" a\n c\td ") == [
        "A",
        "A_1_C",
        "C",
        "C_1_D",
        "D",
    ]
    with pytest.raises(DRContractError, match="dr_max_raw_length=3"):
        tokenizer.tokenize("ACDE")


def test_dr_zero_policy_preserves_distance_and_inference_contract():
    tokenizer = DRTokenizer(
        max_length=10,
        max_distance=2,
        unknown_residue_policy="zero",
    )

    assert tokenizer.tokenize("AXC") == ["A", "A_2_C", "C"]
    contract = dr_contract_metadata(
        max_distance=2,
        max_raw_length=100,
        unknown_residue_policy="zero",
    )
    assert contract["unknown_residue_policy"] == "zero"
    classic = {
        "tokenizer": "dr",
        "encoder": "bow",
        "max_length": 10,
        "dr_max_distance": 2,
        "dr_max_raw_length": 100,
        "dr_unknown_residue_policy": "zero",
        "dr_contract": contract,
    }
    featurizer, _encoder = ModelService._build_classic_featurizer({"classic": classic, "args": {}})
    encoded = featurizer.encode("AXC")
    assert encoded.sum().item() == 3.0
    assert encoded[tokenizer.feature_to_index["A_2_C"]].item() == 1.0


def test_dr_global_bow_uses_complete_sequence_and_raw_counts():
    featurizer = _factory().create(
        "dr",
        "bow",
        max_length=1,
        k=2,
        dr_max_distance=2,
    )

    encoded = featurizer.encode("AGLP")
    tokenizer = featurizer.tokenizer

    assert encoded.shape == (820,)
    assert encoded.sum().item() == 9.0
    assert encoded[tokenizer.feature_to_index["A"]].item() == 1.0
    assert encoded[tokenizer.feature_to_index["G_2_P"]].item() == 1.0


def test_dr_tfidf_and_textrank_are_full_sequence_global_extensions():
    tfidf = _factory().create(
        "dr",
        "tfidf",
        max_length=1,
        k=2,
        sequences=["AGLP"],
        dr_max_distance=2,
    )
    textrank = _factory().create(
        "dr",
        "textrank",
        max_length=1,
        k=2,
        dr_max_distance=2,
    )

    tfidf_value = tfidf.encode("AGLP")
    textrank_value = textrank.encode("AGLP")

    assert tfidf_value.shape == (820,)
    assert tfidf_value.sum().item() == pytest.approx(1.0)
    assert torch.count_nonzero(tfidf_value).item() == 9
    assert textrank_value.shape == (820,)
    assert textrank_value.sum().item() == pytest.approx(1.0)
    assert torch.count_nonzero(textrank_value).item() == 9


def test_dr_ordered_encoders_apply_max_length_to_dr_tokens():
    index = _factory().create(
        "dr",
        "index",
        max_length=3,
        k=2,
        dr_max_distance=2,
    )
    onehot = _factory().create(
        "dr",
        "onehot",
        max_length=3,
        k=2,
        dr_max_distance=2,
    )

    assert index.encode("AGLP").tolist() == [
        index.tokenizer.dr_vocab["A"],
        index.tokenizer.dr_vocab["A_1_G"],
        index.tokenizer.dr_vocab["A_2_L"],
    ]
    assert onehot.encode("AGLP").shape == (3, 822)


def test_dr_static_embedding_sentences_share_the_v2_stream():
    sentences = build_dr_sentences(["AGLP"], 2)

    assert sentences == [DRTokenizer(max_distance=2).tokenize("AGLP")]
    assert resolve_embedding_dir(
        "/tmp/cache",
        "word2vec",
        "dr",
        2,
        2,
        "partner",
    ).endswith("word2vec-dr-v2-2-partner")


def test_dr_artifact_requires_exact_v2_contract():
    classic = {
        "tokenizer": "dr",
        "encoder": "bow",
        "max_length": 1,
        "dr_max_distance": 2,
        "dr_max_raw_length": 100,
        "dr_contract": dr_contract_metadata(
            max_distance=2,
            max_raw_length=100,
        ),
    }

    featurizer, encoder = ModelService._build_classic_featurizer({"classic": classic, "args": {}})
    assert encoder == "bow"
    assert featurizer.encode("AGLP").shape == (820,)
    assert classic["dr_contract"]["contract_version"] == DR_CONTRACT_VERSION

    stale = {"classic": dict(classic), "args": {}}
    stale["classic"].pop("dr_contract")
    with pytest.raises(
        InferenceUnsupportedError,
        match="v2 contract",
    ):
        ModelService._build_classic_featurizer(stale)


def test_dr_artifact_rejects_tfidf_width_mismatch(tmp_path):
    idf_path = tmp_path / "idf.pt"
    torch.save(torch.ones(3), idf_path)
    classic = {
        "tokenizer": "dr",
        "encoder": "tfidf",
        "max_length": 1,
        "dr_max_distance": 2,
        "dr_max_raw_length": 100,
        "dr_contract": dr_contract_metadata(
            max_distance=2,
            max_raw_length=100,
        ),
        "tfidf_idf_path": str(idf_path),
    }

    with pytest.raises(
        InferenceUnsupportedError,
        match="width does not match",
    ):
        ModelService._build_classic_featurizer({"classic": classic, "args": {}})


def test_dr_static_embedding_artifact_is_bundle_relative_and_hashed(
    tmp_path,
):
    vector_dir = tmp_path / "classic"
    vector_dir.mkdir()
    vector_path = vector_dir / "dr_word2vec.vec"
    vector_path.write_text(
        "2 2\nA 1.0 0.0\nA_1_R 0.0 1.0\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(vector_path.read_bytes()).hexdigest()
    classic = {
        "tokenizer": "dr",
        "encoder": "word2vec",
        "max_length": 2,
        "dr_max_distance": 1,
        "dr_max_raw_length": 100,
        "dr_contract": dr_contract_metadata(
            max_distance=1,
            max_raw_length=100,
        ),
        "word2vec_path": "classic/dr_word2vec.vec",
        "static_embedding": {
            "method": "word2vec",
            "path": "classic/dr_word2vec.vec",
            "sha256": digest,
            "bundled": True,
        },
    }

    featurizer, encoder = ModelService._build_classic_featurizer(
        {"classic": classic, "args": {}},
        bundle=tmp_path,
    )
    assert encoder == "word2vec"
    assert featurizer.encode("AR").shape == (2, 2)

    vector_path.write_text(
        "2 2\nA 0.0 1.0\nA_1_R 0.0 1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(
        InferenceUnsupportedError,
        match="digest mismatch",
    ):
        ModelService._build_classic_featurizer(
            {"classic": classic, "args": {}},
            bundle=tmp_path,
        )


def test_dr_stream_defaults_are_validated_and_describe_length_units():
    source = SimpleNamespace(
        task_type="pep_protein_interaction",
        tokenizer="dr",
        encoder="bow",
        max_length=60,
        dr_max_distance=5,
    )

    streams = normalize_pair_streams(source)

    for stream in streams.values():
        assert stream["params"]["dr_max_distance"] == 5
        assert stream["params"]["dr_max_raw_length"] == 50_000
        assert stream["max_length_unit"] == ("not_applicable_full_normalized_sequence")


def test_dr_dense_memory_guard_estimates_actual_layout_and_rejects_limit():
    estimated = estimate_dr_dense_tensor_bytes(
        2,
        encoder="onehot",
        max_distance=2,
        max_length=3,
    )
    assert estimated == 2 * 3 * 822 * 4
    assert (
        ensure_dr_dense_memory_budget(
            estimated,
            limit_gib=1,
        )
        == estimated
    )
    with pytest.raises(
        DRContractError,
        match="exceeding dr_dense_memory_limit_gib",
    ):
        ensure_dr_dense_memory_budget(
            estimated,
            limit_gib=1e-9,
        )


def test_web_preflight_accepts_d_zero_and_rejects_out_of_contract_values():
    dataset = {
        "id": "toy",
        "name": "toy",
        "task_type": "pep_function",
        "format": "sequence_dir",
        "metadata": {"splits": {"train": 2, "val": 1}},
    }
    valid = preflight_experiment(
        dataset,
        "pep_function",
        {
            "tokenizer": "dr",
            "encoder": "bow",
            "model": "logreg",
            "classifier": "logreg",
            "dr_max_distance": 0,
            "dr_max_raw_length": 100,
            "dr_dense_memory_limit_gib": 0.5,
        },
    )
    invalid = preflight_experiment(
        dataset,
        "pep_function",
        {
            "tokenizer": "dr",
            "encoder": "bow",
            "model": "logreg",
            "classifier": "logreg",
            "dr_max_distance": 201,
            "dr_max_raw_length": 100_001,
            "dr_dense_memory_limit_gib": 0,
        },
    )

    assert valid["valid"]
    assert not invalid["valid"]
    messages = " ".join(str(issue) for issue in invalid["errors"])
    assert "dr_max_distance" in messages or "max_distance" in messages
    assert "max_raw_length" in messages
    assert "memory limit" in messages
