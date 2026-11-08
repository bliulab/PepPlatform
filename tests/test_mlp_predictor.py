import pytest
import torch
import torch.nn as nn

from pepplatform.models import BoWMLP, PredictionHeadClassifier, build_prediction_head
from pepplatform.web.inference import InferenceUnsupportedError, ModelService
from pepplatform.workflow import (
    prediction_head_artifact_config,
    validate_prediction_head_artifact_config,
)


def _metadata(hidden_dims):
    kind = "mlp" if hidden_dims else "linear"
    head = {
        "schema_version": 1,
        "implementation": "standard_linear_mlp",
        "kind": kind,
        "hidden_dims": list(hidden_dims),
    }
    if hidden_dims:
        head.update({"activation": "relu", "dropout": 0.0})
    return {
        "bundle_type": "pretrained_embedding_nn",
        "args": {"model": "mlp", "classifier": "mlp"},
        "effective_route": {"predictor": {"head": {"id": kind}}},
        "prediction_head": head,
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"input_dim": 0, "output_dim": 1}, "input_dim"),
        ({"input_dim": 2, "output_dim": 0}, "output_dim"),
        ({"input_dim": 2, "output_dim": 1, "hidden_dims": [0]}, "hidden_dims"),
        ({"input_dim": 2, "output_dim": 1, "hidden_dims": [4], "dropout": float("nan")}, "dropout"),
        ({"input_dim": 2, "output_dim": 1, "hidden_dims": [4], "dropout": 1.0}, "dropout"),
    ],
)
def test_prediction_head_rejects_invalid_structure(kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_prediction_head(**kwargs)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (torch.ones(2, 3, 4), "shape"),
        (torch.empty(0, 4), "at least one sample"),
        (torch.ones(2, 3), "feature dimension"),
        (torch.ones(2, 4, dtype=torch.long), "floating-point"),
        (torch.tensor([[float("nan"), 0.0, 1.0, 2.0]]), "finite"),
        (torch.tensor([[float("inf"), 0.0, 1.0, 2.0]]), "finite"),
    ],
)
def test_mlp_rejects_invalid_fixed_width_inputs(value, message):
    model = PredictionHeadClassifier(4, 1, head_hidden_dims=[3])
    with pytest.raises(ValueError, match=message):
        model(value)


def test_mlp_produces_finite_logits_gradients_and_batch_invariant_eval():
    torch.manual_seed(42)
    model = PredictionHeadClassifier(
        4,
        2,
        head_hidden_dims=[6, 3],
        head_activation="gelu",
        head_dropout=0.25,
    )
    inputs = torch.randn(5, 4)
    logits = model(inputs)
    logits.square().mean().backward()
    assert torch.isfinite(logits).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    model.eval()
    with torch.no_grad():
        single = model(inputs[:1])
        batched = model(inputs)[:1]
    torch.testing.assert_close(single, batched)


def test_bow_mlp_honors_legacy_hidden_dim_when_no_explicit_head_is_given():
    model = BoWMLP(in_dim=8, hidden_dim=10, num_classes=2)
    linear_layers = [layer for layer in model.net if isinstance(layer, nn.Linear)]
    assert [layer.out_features for layer in linear_layers] == [10, 5, 2]


@pytest.mark.parametrize("hidden_dims", [[], [8], [8, 4]])
def test_prediction_head_artifact_contract_accepts_canonical_kinds(hidden_dims):
    contract = _metadata(hidden_dims)["prediction_head"]
    assert validate_prediction_head_artifact_config(contract) == contract


def test_prediction_head_artifact_contract_rejects_kind_mismatch():
    contract = _metadata([8])["prediction_head"]
    contract["kind"] = "linear"
    with pytest.raises(ValueError, match="kind"):
        validate_prediction_head_artifact_config(contract)


def test_prediction_head_artifact_writer_emits_a_valid_contract():
    contract = prediction_head_artifact_config(
        {
            "head_hidden_dims": [16, 8],
            "head_activation": "silu",
            "head_dropout": 0.25,
        }
    )
    assert validate_prediction_head_artifact_config(contract) == contract


@pytest.mark.parametrize("hidden_dims", [[], [8]])
def test_model_service_validates_prediction_head_against_loaded_model(hidden_dims):
    model = PredictionHeadClassifier(4, 1, head_hidden_dims=hidden_dims)
    ModelService._validate_prediction_head_contract(_metadata(hidden_dims), model)


def test_model_service_rejects_prediction_head_model_mismatch():
    model = PredictionHeadClassifier(4, 1, head_hidden_dims=[])
    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._validate_prediction_head_contract(_metadata([8]), model)
    assert exc_info.value.code == "prediction_head_model_mismatch"


def test_model_service_rejects_nonstandard_sequential_order():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Dropout(0.0),
    )
    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._validate_prediction_head_contract(_metadata([8]), model)
    assert exc_info.value.code == "prediction_head_model_mismatch"


def test_model_service_keeps_unversioned_mlp_as_legacy_only():
    metadata = _metadata([8])
    metadata.pop("prediction_head")
    with pytest.warns(RuntimeWarning, match="legacy_only"):
        ModelService._validate_prediction_head_contract(
            metadata,
            PredictionHeadClassifier(4, 1, head_hidden_dims=[8]),
        )
