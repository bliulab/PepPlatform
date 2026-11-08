import json
from argparse import Namespace

import pytest
import torch

from pepplatform.models import IndexRNN, ResidueModelInput
from pepplatform.pair_strategy import (
    PairStrategyContractError,
    build_two_stream_strategy_contract,
    build_single_stream_strategy_contract,
    join_single_stream_batches,
    join_single_stream_rows,
    validate_single_stream_strategy_contract,
    validate_two_stream_strategy_contract,
)
from pepplatform.stream_spec import apply_pair_streams
from pepplatform.web.inference import InferenceUnsupportedError, ModelService
from pepplatform.web.preflight import preflight_experiment
from scripts.train_pair_pipeline import (
    encode_pair_sequences_for_dl,
    save_pair_sequence_nn_artifact,
)


@pytest.mark.parametrize(
    "shape,transpose", [((9, 5), False), ((5, 4, 7), True), ((1, 2, 45), False)]
)
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -float("inf")])
def test_finite_contract_checks_are_bounded_and_cover_noncontiguous_inputs(
    monkeypatch, shape, transpose, bad
):
    from pepplatform import pair_strategy

    monkeypatch.setattr(pair_strategy, "_FINITE_CHECK_MAX_ELEMENTS", 17)
    values = torch.ones(shape)
    if transpose:
        values = values.transpose(1, 2)
        assert not values.is_contiguous()
    if bad is not None:
        values[tuple(size - 1 for size in values.shape)] = bad
    before = values.clone()
    rng = torch.get_rng_state().clone()
    original = torch.isfinite
    calls = []

    def bounded(tensor):
        calls.append(tensor.numel())
        assert tensor.numel() <= 17
        return original(tensor)

    expected = bool(original(values).all())
    monkeypatch.setattr(torch, "isfinite", bounded)
    assert pair_strategy._all_finite_chunked(values) == expected
    assert calls
    torch.testing.assert_close(values, before, equal_nan=True)
    assert torch.equal(rng, torch.get_rng_state())


def test_finite_contract_rejects_nonfinite_at_chunk_boundary(monkeypatch):
    from pepplatform import pair_strategy

    monkeypatch.setattr(pair_strategy, "_FINITE_CHECK_MAX_ELEMENTS", 12)
    args = Namespace(
        pair_strategy="two_stream",
        interaction_level="token",
        predictor_type="classifier",
        model="cross_attn",
    )
    features = torch.ones(7, 2, 3)
    contract = build_two_stream_strategy_contract(args, features, features)
    assert contract["streams"]["peptide"]["batch_size"] == 7
    features[2, 0, 0] = float("nan")
    with pytest.raises(PairStrategyContractError, match="finite"):
        build_two_stream_strategy_contract(args, features, features)


class _IndexFeaturizer:
    vocab_size = 8

    def encode(self, sequence, sample_id=None):
        del sample_id
        values = {"A": [1, 0, 0], "AB": [1, 2, 0], "BC": [2, 3, 0], "C": [3, 0, 0]}
        return torch.tensor(values[sequence], dtype=torch.long)


def _single_stream_args(**overrides):
    values = {
        "task_type": "pep_protein_interaction",
        "pair_strategy": "single_stream",
        "tokenizer": "kmer",
        "encoder": "index",
        "model": "rnn",
        "sequence_model": "rnn",
        "classifier": "mlp",
        "training_mode": "end_to_end",
        "max_length": 3,
        "k": 1,
        "artifact_dir": "",
    }
    values.update(overrides)
    return Namespace(**values)


def _dataset():
    return {
        "id": "pairs",
        "name": "pairs",
        "task_type": "pep_protein_interaction",
        "format": "pair_csv_bundle",
        "metadata": {"splits": {"train": 4, "val": 2, "test": 2}},
    }


def test_two_stream_contract_records_heterogeneous_vector_roles_without_fusion_for_hgt():
    args = Namespace(
        pair_strategy="two_stream",
        stream_mode="heterogeneous",
        interaction_level="vector",
        predictor_type="sample_graph",
        graph_predictor="hgt",
        model="hgt",
        streams={"peptide": {}, "partner": {}},
        fusion_mode="concat",
    )
    contract = build_two_stream_strategy_contract(
        args,
        torch.ones(3, 2),
        torch.ones(3, 5),
    )
    assert contract["input_adapter"] == "typed_role_projection"
    assert "fusion_mode" not in contract
    assert contract["streams"]["peptide"]["feature_dim"] == 2
    assert contract["streams"]["partner"]["feature_dim"] == 5
    validate_two_stream_strategy_contract(
        contract,
        peptide=torch.ones(1, 2),
        partner=torch.ones(1, 5),
    )


def test_two_stream_contract_rejects_missing_or_tampered_metadata():
    args = Namespace(
        pair_strategy="two_stream",
        stream_mode="heterogeneous",
        interaction_level="token",
        predictor_type="classifier",
        model="cross_attn",
        streams={"peptide": {}, "partner": {}},
        interaction_model="cross_attn",
    )
    contract = build_two_stream_strategy_contract(
        args,
        torch.ones(2, 7, 3),
        torch.ones(2, 11, 5),
    )
    with pytest.raises(PairStrategyContractError, match="missing"):
        validate_two_stream_strategy_contract(None)
    tampered = json.loads(json.dumps(contract))
    tampered["streams"]["partner"]["feature_dim"] = 6
    with pytest.raises(PairStrategyContractError, match="does not match"):
        validate_two_stream_strategy_contract(
            tampered,
            peptide=torch.ones(1, 7, 3),
            partner=torch.ones(1, 11, 5),
        )

    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._validate_pair_strategy_metadata({"pair": {"pair_strategy": "two_stream"}})
    assert exc_info.value.code == "two_stream_strategy_contract_missing"


def test_two_stream_contract_rejects_invalid_descriptor_metadata_without_inputs():
    args = Namespace(
        pair_strategy="two_stream",
        stream_mode="heterogeneous",
        interaction_level="vector",
        predictor_type="classifier",
        model="mlp",
        streams={"peptide": {}, "partner": {}},
    )
    contract = build_two_stream_strategy_contract(
        args,
        torch.ones(2, 3),
        torch.ones(2, 5),
    )
    contract["streams"]["peptide"]["dtype"] = "torch.int64"
    with pytest.raises(PairStrategyContractError, match="descriptor is invalid"):
        validate_two_stream_strategy_contract(contract)

    contract = build_two_stream_strategy_contract(
        args,
        torch.ones(2, 3),
        torch.ones(2, 5),
    )
    contract["streams"]["partner"]["feature_dim"] = "not-an-integer"
    with pytest.raises(PairStrategyContractError, match="descriptor is invalid"):
        validate_two_stream_strategy_contract(contract)


def test_single_stream_index_join_preserves_role_boundary_and_pair_identity():
    first = join_single_stream_rows(
        torch.tensor([1, 0, 0]),
        torch.tensor([2, 3, 0]),
        separator_id=8,
    )
    second = join_single_stream_rows(
        torch.tensor([1, 2, 0]),
        torch.tensor([3, 0, 0]),
        separator_id=8,
    )

    assert first.tolist() == [1, 8, 2, 3, 0, 0, 0]
    assert second.tolist() == [1, 2, 8, 3, 0, 0, 0]
    assert not torch.equal(first, second)


def test_single_stream_feature_join_adds_role_channels_and_boundary():
    peptide = ResidueModelInput(
        torch.tensor([[[1.0, 2.0], [9.0, 9.0], [0.0, 0.0]]]),
        torch.tensor([[True, False, False]]),
    )
    partner = ResidueModelInput(
        torch.tensor([[[3.0, 4.0], [5.0, 6.0], [0.0, 0.0]]]),
        torch.tensor([[True, True, False]]),
    )

    joined = join_single_stream_batches(peptide, partner)

    assert isinstance(joined, ResidueModelInput)
    assert joined.features.shape == (1, 7, 5)
    assert joined.mask.tolist() == [[True, True, True, True, False, False, False]]
    assert joined.features[0, 0].tolist() == [1.0, 2.0, 1.0, 0.0, 0.0]
    assert joined.features[0, 1].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert joined.features[0, 2].tolist() == [3.0, 4.0, 0.0, 1.0, 0.0]


def test_single_stream_join_rejects_empty_nonfinite_and_separator_collision():
    with pytest.raises(PairStrategyContractError, match="valid peptide"):
        join_single_stream_rows(
            torch.zeros(2, dtype=torch.long),
            torch.tensor([1, 0]),
            separator_id=8,
        )
    with pytest.raises(PairStrategyContractError, match="finite"):
        join_single_stream_rows(
            torch.tensor([[float("nan")]]),
            torch.tensor([[1.0]]),
        )
    with pytest.raises(PairStrategyContractError, match="outside"):
        join_single_stream_rows(
            torch.tensor([8, 0]),
            torch.tensor([1, 0]),
            separator_id=8,
        )


def test_single_stream_encoder_emits_current_strategy_contract():
    args = _single_stream_args()
    joined = encode_pair_sequences_for_dl(
        _IndexFeaturizer(),
        [("one", "A", "BC", 1), ("two", "AB", "C", 0)],
        split_name="train",
        args=args,
    )

    assert joined.tolist() == [
        [1, 8, 2, 3, 0, 0, 0],
        [1, 2, 8, 3, 0, 0, 0],
    ]
    contract = validate_single_stream_strategy_contract(args._single_stream_strategy_contract)
    assert contract["sequence_join"]["index_separator_id"] == 8
    assert contract["sequence_join"]["joined_length"] == 7


def test_single_stream_rejects_heterogeneous_stream_specs_in_cli_and_preflight():
    streams = {
        "peptide": {
            "feature_route": "classic",
            "tokenizer": "kmer",
            "encoder": "onehot",
            "max_length": 32,
            "params": {"k": 1},
        },
        "partner": {
            "feature_route": "classic",
            "tokenizer": "wordpiece",
            "encoder": "onehot",
            "max_length": 64,
            "params": {"wordpiece_vocab_size": 1000},
        },
    }
    args = _single_stream_args(encoder="onehot", streams=streams)
    with pytest.raises(
        PairStrategyContractError,
        match="requires identical peptide and partner",
    ) as exc_info:
        apply_pair_streams(args)
    assert exc_info.value.code == "single_stream_requires_homogeneous_streams"

    report = preflight_experiment(
        _dataset(),
        "pep_protein_interaction",
        {
            "tokenizer": "kmer",
            "encoder": "onehot",
            "model": "rnn",
            "sequence_model": "rnn",
            "classifier": "mlp",
            "training_mode": "end_to_end",
            "pair_strategy": "single_stream",
            "streams": streams,
        },
    )
    assert report["valid"] is False
    assert "single_stream_requires_homogeneous_streams" in report["error_codes"]


def test_single_stream_artifact_persists_strategy_contract(tmp_path):
    args = _single_stream_args(artifact_dir=str(tmp_path))
    train_x = encode_pair_sequences_for_dl(
        _IndexFeaturizer(),
        [("one", "A", "BC", 1), ("two", "AB", "C", 0)],
        split_name="train",
        args=args,
    )
    args._operating_point = {
        "values": [0.4],
        "source": "validation",
        "objective": "f1",
        "score_kind": "probability",
        "protocol_version": 3,
    }
    model = IndexRNN(
        vocab_size=9,
        embed_dim=4,
        hidden_dim=3,
        num_classes=1,
        dropout=0.0,
    )

    save_pair_sequence_nn_artifact(args, model, train_x, threshold=0.4)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["pair_strategy_contract"] == args._single_stream_strategy_contract
    assert metadata["classic"]["vocab_size"] == 8
    assert metadata["classic"]["model_vocab_size"] == 9


def test_single_stream_web_join_uses_contract_and_legacy_layout(monkeypatch):
    peptide = torch.tensor([[1, 0, 0]], dtype=torch.long)
    partner = torch.tensor([[2, 3, 0]], dtype=torch.long)

    monkeypatch.setattr(
        ModelService,
        "_prepare_classic_profile_metadata",
        staticmethod(lambda metadata, pairs, bundle: (metadata, [None], [None])),
    )
    monkeypatch.setattr(
        ModelService,
        "_extract_classic_pair_token_sequences",
        staticmethod(
            lambda metadata, sequences, device, sample_ids=None, bundle=None, role=None: (
                peptide if role == "peptide" else partner
            )
        ),
    )
    joined = join_single_stream_batches(peptide, partner, separator_id=8)
    contract = build_single_stream_strategy_contract(
        joined,
        separator_id=8,
        source_feature_dim=None,
    )
    metadata = {
        "classic": {"tokenizer": "kmer", "encoder": "index"},
        "pair_strategy_contract": contract,
    }

    current = ModelService._extract_pair_sequence_features(
        metadata,
        [{"peptide": "A", "protein": "BC"}],
        torch.device("cpu"),
    )
    assert current.tolist() == [[1, 8, 2, 3, 0, 0, 0]]

    with pytest.warns(RuntimeWarning, match="legacy_only"):
        legacy = ModelService._extract_pair_sequence_features(
            {"classic": {"tokenizer": "kmer", "encoder": "index"}},
            [{"peptide": "A", "protein": "BC"}],
            torch.device("cpu"),
        )
    assert legacy.tolist() == [[1, 0, 0, 2, 3, 0]]


def test_single_stream_contract_rejects_tampering():
    joined = join_single_stream_batches(
        torch.tensor([[1, 0]]),
        torch.tensor([[2, 0]]),
        separator_id=8,
    )
    contract = build_single_stream_strategy_contract(
        joined,
        separator_id=8,
        source_feature_dim=None,
    )
    contract["role_order"] = ["partner", "peptide"]

    with pytest.raises(PairStrategyContractError, match="unsupported or inconsistent"):
        validate_single_stream_strategy_contract(contract)
