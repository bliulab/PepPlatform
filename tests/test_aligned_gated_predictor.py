import copy
import __main__

import pytest
import torch

from pepplatform.models import AlignedGatedClassifier
from pepplatform.pair_fusion import (
    AlignedGatedFusion,
    PairFusionError,
    aligned_gated_contract,
    validate_aligned_gated_contract,
)
from pepplatform.web.inference import InferenceUnsupportedError, ModelService
from pepplatform.web.preflight import preflight_experiment
from scripts.train_pair_pipeline import build_argument_parser


def _fusion(peptide_dim=2, partner_dim=2, **kwargs):
    return AlignedGatedFusion(peptide_dim, partner_dim, **kwargs)


def test_aligned_gated_formula_and_output_blocks_are_exact():
    fusion = _fusion(dropout=0.0)
    with torch.no_grad():
        for parameter in fusion.gate.parameters():
            parameter.zero_()
    peptide = torch.tensor([[1.0, 2.0]])
    partner = torch.tensor([[5.0, 6.0]])

    output = fusion(peptide, partner)
    expected = torch.tensor([[3.0, 4.0, 4.0, 4.0, 5.0, 12.0]])
    torch.testing.assert_close(output, expected)
    assert output.shape == (1, 6)


def test_equal_width_uses_identity_and_unequal_width_uses_role_projections():
    homogeneous = _fusion(3, 3)
    assert isinstance(homogeneous.peptide_projection, torch.nn.Identity)
    assert isinstance(homogeneous.partner_projection, torch.nn.Identity)
    assert homogeneous.alignment_dim == 3

    heterogeneous = _fusion(3, 5, alignment_dim=4)
    assert isinstance(heterogeneous.peptide_projection, torch.nn.Linear)
    assert isinstance(heterogeneous.partner_projection, torch.nn.Linear)
    assert heterogeneous.output_dim == 12
    assert heterogeneous(torch.ones(2, 3), torch.ones(2, 5)).shape == (2, 12)


@pytest.mark.parametrize(
    ("constructor", "code"),
    [
        (lambda: _fusion(0, 2), "aligned_gated_dimension_invalid"),
        (lambda: _fusion(2, 3, alignment_dim=0), "aligned_gated_dimension_invalid"),
        (lambda: _fusion(2, 2, dropout=1.0), "aligned_gated_dropout_invalid"),
        (lambda: _fusion(2, 2, dropout=float("nan")), "aligned_gated_dropout_invalid"),
    ],
)
def test_constructor_rejects_invalid_dimensions_and_dropout(constructor, code):
    with pytest.raises(PairFusionError) as exc_info:
        constructor()
    assert exc_info.value.code == code


@pytest.mark.parametrize(
    ("peptide", "partner", "code"),
    [
        (torch.ones(2), torch.ones(2), "aligned_gated_input_rank_invalid"),
        (torch.empty(0, 2), torch.empty(0, 2), "aligned_gated_input_empty"),
        (
            torch.ones(1, 2, dtype=torch.int64),
            torch.ones(1, 2),
            "aligned_gated_input_dtype_invalid",
        ),
        (torch.ones(1, 2), torch.ones(2, 2), "aligned_gated_batch_mismatch"),
        (torch.tensor([[float("nan"), 1.0]]), torch.ones(1, 2), "aligned_gated_input_nonfinite"),
        (torch.ones(1, 3), torch.ones(1, 2), "aligned_gated_input_width_invalid"),
    ],
)
def test_inputs_fail_closed(peptide, partner, code):
    with pytest.raises(PairFusionError) as exc_info:
        _fusion()(peptide, partner)
    assert exc_info.value.code == code


def test_aligned_gated_gradients_are_finite_and_eval_is_batch_invariant():
    model = AlignedGatedClassifier(peptide_dim=3, partner_dim=5, align_dim=4, dropout=0.0)
    peptide = torch.randn(2, 3)
    partner = torch.randn(2, 5)
    logits = model(peptide, partner)
    loss = torch.nn.BCEWithLogitsLoss()(logits, torch.ones_like(logits))
    loss.backward()
    assert torch.isfinite(logits).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    model.eval()
    with torch.no_grad():
        one = model(peptide[:1], partner[:1])
        batch = model(
            torch.cat([peptide[:1], peptide[1:]], dim=0),
            torch.cat([partner[:1], partner[1:]], dim=0),
        )[:1]
    torch.testing.assert_close(one, batch)


def test_contract_round_trip_and_tamper_rejection():
    fusion = _fusion(3, 5, alignment_dim=4, gate_hidden_dim=7, dropout=0.1)
    contract = aligned_gated_contract(fusion)
    assert contract["implementation"] == "pepplatform_aligned_gated_fusion"
    assert contract["output_blocks"] == [
        "convex_elementwise_mix",
        "absolute_difference",
        "hadamard_product",
    ]
    assert validate_aligned_gated_contract(contract, fusion) == contract

    tampered = copy.deepcopy(contract)
    tampered["alignment_dimension"] = 5
    with pytest.raises(PairFusionError) as exc_info:
        validate_aligned_gated_contract(tampered, fusion)
    assert exc_info.value.code == "aligned_gated_contract_invalid"


def test_graph_gate_loader_reconstructs_versioned_module():
    fusion = _fusion(2, 2, gate_hidden_dim=2, dropout=0.0)
    payload = {
        "gate_state_dict": {
            key: value.detach().clone() for key, value in fusion.state_dict().items()
        }
    }
    loaded = ModelService._load_pair_graph_gate(
        payload,
        2,
        torch.device("cpu"),
        contract=aligned_gated_contract(fusion),
    )
    torch.testing.assert_close(
        fusion(torch.ones(2, 2), torch.ones(2, 2)),
        loaded(torch.ones(2, 2), torch.ones(2, 2)),
    )


def test_preflight_rejects_nonpositive_or_noninteger_alignment_dim():
    base = {
        "tokenizer": "esm2",
        "encoder": "esm2",
        "model": "mlp",
        "classifier": "mlp",
        "pair_strategy": "two_stream",
        "interaction_level": "vector",
        "predictor_type": "classifier",
        "fusion_mode": "aligned_gated",
    }
    split_counts = {"train": 10, "val": 2, "test": 2}
    dataset = {
        "id": "aligned-gated-test",
        "name": "aligned-gated-test",
        "task_type": "pep_protein_interaction",
        "format": "pair_table",
        "metadata": {
            "splits": split_counts,
            "length_stats": {
                "version": 1,
                "by_split": {
                    split: {
                        "peptide": {"count": count, "min": 8, "max": 50, "mean": 15.0},
                        "partner": {"count": count, "min": 10, "max": 512, "mean": 200.0},
                    }
                    for split, count in split_counts.items()
                },
            },
        },
    }
    for value in (0, -1, 1.5, "abc"):
        report = preflight_experiment(
            dataset, "pep_protein_interaction", {**base, "stream_alignment_dim": value}
        )
        assert report["valid"] is False
        assert any(
            "stream_alignment_dim_invalid" in str(error)
            or "stream_alignment_dim must be a positive integer" in str(error)
            for error in report["errors"]
        )


def test_route_rejects_current_model_with_invalid_contract():
    model = AlignedGatedClassifier(input_dim=2, dropout=0.0)
    metadata = {
        "args": {"model": "mlp", "classifier": "mlp"},
        "aligned_gated_contract": {"contract_version": 1},
    }
    with pytest.raises(InferenceUnsupportedError):
        ModelService._validate_aligned_gated_model(metadata, model)


def test_legacy_gated_trainable_pickle_name_is_restored(monkeypatch):
    calls = []
    previous = getattr(__main__, "GatedTrainableClassifier", None)
    had_previous = hasattr(__main__, "GatedTrainableClassifier")

    def fake_load(*_args, **_kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise AttributeError("Can't get attribute 'GatedTrainableClassifier' on __main__")
        assert hasattr(__main__, "GatedTrainableClassifier")
        return "legacy-loaded"

    monkeypatch.setattr(torch, "load", fake_load)
    assert ModelService._load_torch_model("model.pt", torch.device("cpu")) == "legacy-loaded"
    assert len(calls) == 2
    if had_previous:
        assert getattr(__main__, "GatedTrainableClassifier") is previous
    else:
        assert not hasattr(__main__, "GatedTrainableClassifier")


def test_cli_parser_accepts_serialized_integer_alignment_dim_for_all_routes():
    args = build_argument_parser().parse_args(
        ["--tokenizer", "kmer", "--model", "mlp", "--stream_alignment_dim", "256"]
    )
    assert args.stream_alignment_dim == 256


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "abc", ""])
def test_cli_parser_rejects_invalid_alignment_dim(value):
    with pytest.raises(SystemExit):
        build_argument_parser().parse_args(["--stream_alignment_dim", value])
