from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from pepplatform.encoders import StaticEmbeddingEncoder
from pepplatform.featurizers import ComposedFeaturizer
from pepplatform.models import (
    TRANSFORMER_PREDICTOR_CONTRACT_VERSION,
    FeatureTransformer,
    IndexTransformer,
    ModelFactory,
    PretrainedResiduePredictor,
    ResidueModelInput,
    sequence_predictor_contract,
    transformer_predictor_contract,
)
from pepplatform.tokenizers import KmerTokenizer
from scripts.train_pair_pipeline import build_classic_featurizer
from scripts.train_pipeline import Pipeline


def _transformer_models():
    return [
        (
            IndexTransformer(
                vocab_size=24,
                max_length=8,
                embed_dim=8,
                num_layers=2,
                num_heads=2,
                ff_dim=16,
                num_classes=3,
            ),
            "learned_embedding",
            "token_id_nonzero",
        ),
        (
            FeatureTransformer(
                in_dim=5,
                max_length=8,
                embed_dim=8,
                num_layers=2,
                num_heads=2,
                ff_dim=16,
                num_classes=3,
            ),
            "linear_projection",
            "explicit_position_mask",
        ),
        (
            PretrainedResiduePredictor(
                "transformer",
                input_dim=5,
                adapter_dim=8,
                max_length=8,
                heads=2,
                num_classes=3,
            ),
            "layer_norm_linear_projection",
            "explicit_residue_mask",
        ),
    ]


@pytest.mark.parametrize(
    ("model", "adapter", "mask_source"),
    _transformer_models(),
)
def test_transformer_contract_v2_records_actual_encoder_assembly(
    model,
    adapter,
    mask_source,
):
    assert model.transformer_predictor_contract_version == TRANSFORMER_PREDICTOR_CONTRACT_VERSION
    encoder = model.transformer if isinstance(model, PretrainedResiduePredictor) else model.encoder
    assert type(encoder) is nn.TransformerEncoder
    assert len(encoder.layers) == 2
    assert all(type(layer) is nn.TransformerEncoderLayer for layer in encoder.layers)
    assert all(type(layer.self_attn) is nn.MultiheadAttention for layer in encoder.layers)
    assert all(layer.self_attn.batch_first for layer in encoder.layers)
    assert all(layer.self_attn.num_heads == 2 for layer in encoder.layers)
    assert all(layer.linear1.out_features == 16 for layer in encoder.layers)
    assert all(layer.activation.__name__ == "gelu" for layer in encoder.layers)
    assert all(layer.norm_first is False for layer in encoder.layers)

    contract = transformer_predictor_contract(model)
    assert contract == {
        "schema_version": 1,
        "implementation": "pepplatform_pytorch_transformer_encoder",
        "contract_version": 2,
        "status": "verified_platform_variant",
        "input_adapter": adapter,
        "actual_encoder": "TransformerEncoder",
        "encoder_layers": 2,
        "attention": "multi_head_scaled_dot_product_self_attention",
        "model_dim": 8,
        "num_heads": 2,
        "feedforward_dim": 16,
        "activation": "gelu",
        "norm_first": False,
        "position_encoding": "learned_absolute_embedding",
        "position_indexing": "valid_token_rank",
        "sequence_handling": "key_padding_mask_with_mask_compact_positions",
        "padding_mask": mask_source,
        "readout": "masked_mean",
        "prediction_output": "logits",
    }
    assert sequence_predictor_contract(model) == contract


def test_factory_builds_transformer_encoder_for_index_and_feature_routes():
    index_model = ModelFactory().create(
        "transformer",
        "index",
        SimpleNamespace(vocab_size=24, max_length=7),
        num_classes=3,
        dropout=0.0,
    )
    feature_model = ModelFactory().create(
        "transformer",
        "onehot",
        SimpleNamespace(vocab_size=5, max_length=7),
        num_classes=3,
        dropout=0.0,
    )

    assert type(index_model) is IndexTransformer
    assert type(feature_model) is FeatureTransformer
    assert index_model.position.num_embeddings == 7
    assert feature_model.position.num_embeddings == 7
    assert transformer_predictor_contract(index_model)["status"] == "verified_platform_variant"
    assert transformer_predictor_contract(feature_model)["status"] == "verified_platform_variant"


def test_index_transformer_is_invariant_to_right_and_internal_pair_padding():
    torch.manual_seed(17)
    model = IndexTransformer(
        vocab_size=24,
        max_length=8,
        embed_dim=8,
        num_layers=2,
        num_heads=2,
        ff_dim=16,
        num_classes=3,
        dropout=0.0,
    ).eval()
    compact = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    right_padded = torch.tensor([[1, 2, 3, 4, 0, 0, 0]], dtype=torch.long)
    pair_joined = torch.tensor([[1, 2, 0, 0, 0, 3, 4]], dtype=torch.long)
    companion = torch.tensor([[5, 6, 7, 8, 9, 0, 0]], dtype=torch.long)

    with torch.inference_mode():
        compact_logits, compact_embedding = model.forward_with_embedding(compact)
        right_logits, right_embedding = model.forward_with_embedding(right_padded)
        pair_logits, pair_embedding = model.forward_with_embedding(pair_joined)
        batch_logits, batch_embedding = model.forward_with_embedding(
            torch.cat([right_padded, companion], dim=0)
        )

    assert torch.allclose(compact_embedding, right_embedding, atol=1e-6)
    assert torch.allclose(compact_logits, right_logits, atol=1e-6)
    assert torch.allclose(compact_embedding, pair_embedding, atol=1e-6)
    assert torch.allclose(compact_logits, pair_logits, atol=1e-6)
    assert torch.allclose(right_embedding[0], batch_embedding[0], atol=1e-6)
    assert torch.allclose(right_logits[0], batch_logits[0], atol=1e-6)


@pytest.mark.parametrize("route", ["feature", "pretrained"])
def test_feature_transformers_ignore_mask_holes_values_and_gradients(route):
    torch.manual_seed(23)
    if route == "feature":
        model = FeatureTransformer(
            in_dim=5,
            max_length=8,
            embed_dim=8,
            num_layers=2,
            num_heads=2,
            ff_dim=16,
            num_classes=3,
            dropout=0.0,
        )
    else:
        model = PretrainedResiduePredictor(
            "transformer",
            input_dim=5,
            adapter_dim=8,
            max_length=8,
            heads=2,
            num_classes=3,
            dropout=0.0,
        )
    model.eval()
    valid = torch.randn(1, 4, 5)
    compact = ResidueModelInput(valid, torch.ones((1, 4), dtype=torch.bool))
    polluted_features = torch.cat(
        [valid[:, :2], torch.full((1, 3, 5), 97.0), valid[:, 2:]],
        dim=1,
    ).requires_grad_(True)
    polluted_mask = torch.tensor(
        [[True, True, False, False, False, True, True]],
        dtype=torch.bool,
    )
    polluted = ResidueModelInput(polluted_features, polluted_mask)

    compact_logits, compact_embedding = model.forward_with_embedding(compact)
    polluted_logits, polluted_embedding = model.forward_with_embedding(polluted)
    polluted_logits.sum().backward()

    assert torch.allclose(compact_embedding, polluted_embedding, atol=1e-6)
    assert torch.allclose(compact_logits, polluted_logits, atol=1e-6)
    assert torch.count_nonzero(polluted_features.grad[~polluted_mask]) == 0
    assert torch.isfinite(polluted_features.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_index_transformer_validates_inputs_gradients_and_round_trips(tmp_path):
    torch.manual_seed(29)
    model = IndexTransformer(
        vocab_size=12,
        max_length=4,
        embed_dim=8,
        num_layers=2,
        num_heads=2,
        ff_dim=16,
        num_classes=2,
        dropout=0.0,
    )
    inputs = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]], dtype=torch.long)
    logits, embedding = model.forward_with_embedding(inputs)
    logits.sum().backward()

    assert logits.shape == (2, 2)
    assert embedding.shape == (2, 8)
    assert torch.isfinite(logits).all()
    assert torch.count_nonzero(model.embedding.weight.grad[0]) == 0
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    with pytest.raises(ValueError, match=r"shape \[B,L\]"):
        model(torch.ones((1, 2, 3), dtype=torch.long))
    with pytest.raises(TypeError, match="int32 or int64"):
        model(torch.ones((1, 3), dtype=torch.float))
    with pytest.raises(ValueError, match="sequence length must be positive"):
        model(torch.empty((1, 0), dtype=torch.long))
    with torch.inference_mode():
        with pytest.raises(ValueError, match="at least one non-padding token"):
            model(torch.zeros((1, 4), dtype=torch.long))
    with pytest.raises(ValueError, match=r"must be in \[0, 12\)"):
        model(torch.tensor([[1, 12]], dtype=torch.long))
    with pytest.raises(ValueError, match="capacity 4"):
        model(torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long))

    model.eval()
    checkpoint = tmp_path / "transformer_v2.pt"
    torch.save(model, checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False).eval()
    with torch.inference_mode():
        expected = model.forward_with_embedding(inputs)
        actual = loaded.forward_with_embedding(inputs)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert transformer_predictor_contract(loaded)["status"] == "verified_platform_variant"


def test_feature_transformer_rejects_integer_features_and_position_overflow():
    model = FeatureTransformer(
        in_dim=3,
        max_length=3,
        embed_dim=6,
        num_layers=1,
        num_heads=2,
        ff_dim=12,
        num_classes=2,
    )
    with pytest.raises(TypeError, match="floating-point"):
        model(
            ResidueModelInput(
                torch.ones((1, 2, 3), dtype=torch.long),
                torch.ones((1, 2), dtype=torch.bool),
            )
        )
    double_logits = model(
        ResidueModelInput(
            torch.ones((1, 2, 3), dtype=torch.float64),
            torch.ones((1, 2), dtype=torch.bool),
        )
    )
    assert torch.isfinite(double_logits).all()
    with pytest.raises(ValueError, match="capacity 3"):
        model(
            ResidueModelInput(
                torch.ones((1, 4, 3)),
                torch.ones((1, 4), dtype=torch.bool),
            )
        )


@pytest.mark.parametrize(
    ("model", "kind"),
    [(row[0], row[0].__class__.__name__) for row in _transformer_models()],
)
def test_unversioned_transformers_preserve_historical_positions(model, kind, tmp_path):
    torch.manual_seed(31)
    model.eval()
    if isinstance(model, IndexTransformer):
        inputs = torch.tensor([[1, 2, 0, 0, 3, 4]], dtype=torch.long)
        expected = model._legacy_forward_with_embedding(inputs)
    elif isinstance(model, FeatureTransformer):
        inputs = ResidueModelInput(
            torch.randn(1, 6, 5),
            torch.tensor([[True, True, False, False, True, True]]),
        )
        expected = model._legacy_forward_with_embedding(inputs)
    else:
        inputs = ResidueModelInput(
            torch.randn(1, 6, 5),
            torch.tensor([[True, True, False, False, True, True]]),
        )
        features, mask = model._validate_input(inputs)
        adapted = model._adapt(features, mask)
        positions = torch.arange(adapted.shape[1]).unsqueeze(0)
        encoded = model.transformer(
            adapted + model.position(positions),
            src_key_padding_mask=~mask,
        )
        valid = mask.unsqueeze(-1).to(dtype=encoded.dtype)
        embedding = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        expected = (model.head(embedding), embedding)

    del model.transformer_predictor_contract_version
    contract = transformer_predictor_contract(model)
    assert contract["contract_version"] == 1
    assert contract["status"] == "legacy_only"
    assert contract["position_indexing"] == "raw_padded_slot"

    checkpoint = tmp_path / f"{kind}_legacy_transformer.pt"
    torch.save(model, checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False).eval()
    with pytest.warns(RuntimeWarning, match="does not carry"):
        actual = loaded.forward_with_embedding(inputs)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_transformer_explicit_position_mask_preserves_valid_zero_oov(tmp_path):
    embedding_path = tmp_path / "vectors.txt"
    embedding_path.write_text("1 3\nA 1.0 2.0 3.0\n", encoding="utf-8")
    tokenizer = KmerTokenizer(max_length=4, k=1)
    encoder = StaticEmbeddingEncoder(
        max_length=4,
        embedding_path=embedding_path,
        has_header=True,
    )
    featurizer = ComposedFeaturizer(
        tokenizer,
        encoder,
    ).require_explicit_position_mask()

    encoded = featurizer.encode("AX")

    assert isinstance(encoded, ResidueModelInput)
    assert encoded.mask.tolist() == [True, True, False, False]
    assert torch.count_nonzero(encoded.features[1]) == 0


def test_pair_transformer_builder_enables_explicit_position_masks(tmp_path):
    args = SimpleNamespace(
        tokenizer="kmer",
        encoder="onehot",
        model="transformer",
        sequence_model="transformer",
        max_length=4,
        k=1,
    )
    featurizer = build_classic_featurizer(
        args,
        str(tmp_path),
        [("sample", "AX", "C", 1)],
    )

    encoded = featurizer.encode("AX")

    assert isinstance(encoded, ResidueModelInput)
    assert encoded.mask.tolist() == [True, True, False, False]


def test_sequence_transformer_builder_enables_explicit_position_masks():
    pipeline = object.__new__(Pipeline)
    pipeline.args = SimpleNamespace(
        tokenizer="kmer",
        encoder="onehot",
        model="transformer",
        sequence_model="transformer",
        max_length=4,
        k=1,
        unigram_sampling=False,
    )
    pipeline.ensure_static_embedding_ready = lambda: None
    pipeline.ensure_pssm_cache_ready = lambda: None
    pipeline.ensure_bpe_tokenizer_ready = lambda: None
    pipeline.ensure_wordpiece_tokenizer_ready = lambda: None
    pipeline.ensure_unigram_tokenizer_ready = lambda: None

    class CapturingFeaturizer:
        explicit_position_mask = False

        def require_explicit_position_mask(self):
            self.explicit_position_mask = True
            return self

    featurizer = CapturingFeaturizer()

    class FakeFactory:
        def create(self, *args, **kwargs):
            return featurizer

    pipeline.feat_factory = FakeFactory()

    assert pipeline.build_featurizer().explicit_position_mask is True
