import pytest
import torch

from pepplatform.models import (
    CNN_PREDICTOR_CONTRACT_VERSION,
    FeatureTextCNN,
    IndexTextCNN,
    MaskedResidualBlock,
    PretrainedResiduePredictor,
    ResidualBlock,
    ResidueModelInput,
    cnn_predictor_contract,
)


def _replace_with_legacy_blocks(model, input_dim):
    model.res_block1 = ResidualBlock(input_dim, 64, kernel_size=3, padding=1)
    model.res_block2 = ResidualBlock(input_dim, 64, kernel_size=5, padding=2)
    model.res_block3 = ResidualBlock(input_dim, 64, kernel_size=7, padding=3)
    del model.cnn_predictor_contract_version
    return model


def test_index_cnn_uses_verified_masked_contract():
    model = IndexTextCNN(vocab_size=24, embed_dim=8, num_classes=3)

    assert model.cnn_predictor_contract_version == CNN_PREDICTOR_CONTRACT_VERSION
    assert all(
        isinstance(block, MaskedResidualBlock)
        for block in [
            model.res_block1,
            model.res_block2,
            model.res_block3,
        ]
    )
    assert cnn_predictor_contract(model) == {
        "schema_version": 1,
        "implementation": "pepplatform_multiscale_residual_conv1d",
        "contract_version": 2,
        "status": "verified_platform_variant",
        "input_adapter": "learned_embedding",
        "kernel_sizes": [3, 5, 7],
        "readout": "masked_global_max",
        "padding_mask": "token_id_nonzero",
        "prediction_output": "logits",
    }


def test_index_cnn_is_invariant_to_right_padding_and_batch_companions():
    torch.manual_seed(7)
    model = IndexTextCNN(vocab_size=24, embed_dim=8, num_classes=3, dropout=0.0).eval()
    short = torch.tensor([[1, 2, 3]], dtype=torch.long)
    padded = torch.tensor([[1, 2, 3, 0, 0, 0, 0]], dtype=torch.long)
    companion = torch.tensor([[4, 5, 6, 7, 8, 0, 0]], dtype=torch.long)

    short_logits, short_embedding = model.forward_with_embedding(short)
    padded_logits, padded_embedding = model.forward_with_embedding(padded)
    batch_logits, batch_embedding = model.forward_with_embedding(
        torch.cat([padded, companion], dim=0)
    )

    assert torch.allclose(short_embedding, padded_embedding, atol=1e-6)
    assert torch.allclose(short_logits, padded_logits, atol=1e-6)
    assert torch.allclose(padded_embedding[0], batch_embedding[0], atol=1e-6)
    assert torch.allclose(padded_logits[0], batch_logits[0], atol=1e-6)


def test_index_cnn_masks_padding_gradients_and_rejects_empty_samples():
    torch.manual_seed(11)
    model = IndexTextCNN(vocab_size=24, embed_dim=8, num_classes=2, dropout=0.0)
    inputs = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]], dtype=torch.long)

    model(inputs).sum().backward()

    assert torch.count_nonzero(model.embedding.weight.grad[0]) == 0
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    with pytest.raises(ValueError, match="at least one non-padding token"):
        model(torch.zeros((1, 4), dtype=torch.long))


@pytest.mark.parametrize("model_kind", ["feature", "pretrained"])
def test_continuous_cnn_routes_ignore_masked_values_and_extra_padding(model_kind):
    torch.manual_seed(17)
    if model_kind == "feature":
        model = FeatureTextCNN(in_dim=5, embed_dim=8, num_classes=2, dropout=0.0)
    else:
        model = PretrainedResiduePredictor(
            "cnn",
            input_dim=5,
            adapter_dim=8,
            heads=2,
            num_classes=2,
            dropout=0.0,
        )
    model.eval()
    features = torch.randn(1, 3, 5)
    short = ResidueModelInput(features, torch.ones((1, 3), dtype=torch.bool))
    extended = ResidueModelInput(
        torch.cat([features, torch.full((1, 4, 5), 123.0)], dim=1),
        torch.tensor([[True, True, True, False, False, False, False]]),
    )

    short_logits, short_embedding = model.forward_with_embedding(short)
    extended_logits, extended_embedding = model.forward_with_embedding(extended)

    assert torch.allclose(short_embedding, extended_embedding, atol=1e-6)
    assert torch.allclose(short_logits, extended_logits, atol=1e-6)


@pytest.mark.parametrize("model_kind", ["index", "feature"])
def test_legacy_cnn_full_object_remains_loadable_with_explicit_warning(tmp_path, model_kind):
    torch.manual_seed(23)
    if model_kind == "index":
        model = _replace_with_legacy_blocks(
            IndexTextCNN(vocab_size=24, embed_dim=8, num_classes=2),
            8,
        )
        inputs = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    else:
        model = _replace_with_legacy_blocks(
            FeatureTextCNN(in_dim=5, embed_dim=8, num_classes=2),
            8,
        )
        inputs = torch.randn(1, 4, 5)
    model.eval()
    path = tmp_path / f"legacy_{model_kind}_cnn.pt"
    torch.save(model, path)

    loaded = torch.load(path, map_location="cpu", weights_only=False).eval()
    with pytest.warns(RuntimeWarning, match="legacy unmasked CNN contract"):
        logits, embedding = loaded.forward_with_embedding(inputs)

    assert logits.shape == (1, 2)
    assert embedding.shape == (1, 192)
    contract = cnn_predictor_contract(loaded)
    assert contract["contract_version"] == 1
    assert contract["status"] == "legacy_only"
    assert contract["readout"] == "unmasked_global_max"
