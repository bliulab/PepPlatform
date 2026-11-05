from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from pepplatform.encoders import StaticEmbeddingEncoder
from pepplatform.featurizers import ComposedFeaturizer
from pepplatform.models import (
    GRU_PREDICTOR_CONTRACT_VERSION,
    LSTM_PREDICTOR_CONTRACT_VERSION,
    RNN_PREDICTOR_CONTRACT_VERSION,
    FeatureGRU,
    FeatureLSTM,
    FeatureRNN,
    IndexGRU,
    IndexLSTM,
    IndexRNN,
    ModelFactory,
    PretrainedResiduePredictor,
    ResidueModelInput,
    gru_predictor_contract,
    lstm_predictor_contract,
    rnn_predictor_contract,
    sequence_predictor_contract,
)
from pepplatform.tokenizers import KmerTokenizer
from scripts.train_pair_pipeline import build_classic_featurizer
from scripts.train_pipeline import Pipeline


@pytest.mark.parametrize(
    "architecture",
    ["cnn", "rnn", "gru", "lstm", "transformer", "gcn", "gat"],
)
def test_hybrid_profile_encoder_builds_every_declared_sequence_model(architecture):
    model = ModelFactory().create(
        architecture,
        "pssm_blosum62_hybrid",
        SimpleNamespace(feature_dim=20, max_length=4),
        num_classes=3,
        dropout=0.0,
    )
    model.eval()

    with torch.no_grad():
        logits = model(torch.randn(2, 4, 20))

    assert logits.shape == (2, 3)


@pytest.mark.parametrize(
    ("encoder", "featurizer", "inputs", "rnn_class", "gru_class"),
    [
        (
            "index",
            SimpleNamespace(vocab_size=24),
            torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long),
            IndexRNN,
            IndexGRU,
        ),
        (
            "onehot",
            SimpleNamespace(vocab_size=5),
            torch.randn(2, 4, 5),
            FeatureRNN,
            FeatureGRU,
        ),
    ],
)
def test_factory_builds_distinct_rnn_and_gru_models(
    encoder, featurizer, inputs, rnn_class, gru_class
):
    factory = ModelFactory()

    rnn_model = factory.create("rnn", encoder, featurizer, num_classes=3, dropout=0.0)
    gru_model = factory.create("gru", encoder, featurizer, num_classes=3, dropout=0.0)

    assert type(rnn_model) is rnn_class
    assert type(gru_model) is gru_class
    assert type(rnn_model.rnn) is nn.RNN
    assert type(gru_model.rnn) is nn.GRU
    assert rnn_model.rnn.nonlinearity == "tanh"
    assert rnn_model.rnn.bidirectional is True
    assert gru_model.rnn.bidirectional is True
    assert rnn_model.rnn.weight_ih_l0.shape[0] == rnn_model.rnn.hidden_size
    assert gru_model.rnn.weight_ih_l0.shape[0] == 3 * gru_model.rnn.hidden_size

    for model in (rnn_model, gru_model):
        logits, embedding = model.forward_with_embedding(inputs)
        assert logits.shape == (2, 3)
        assert embedding.shape == (2, 2 * model.rnn.hidden_size)
        assert torch.equal(model(inputs), logits)


@pytest.mark.parametrize(
    ("encoder", "featurizer", "inputs", "expected_class"),
    [
        (
            "index",
            SimpleNamespace(vocab_size=24),
            torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long),
            IndexLSTM,
        ),
        (
            "onehot",
            SimpleNamespace(vocab_size=5),
            ResidueModelInput(
                torch.randn(2, 4, 5),
                torch.tensor([[True, True, True, False], [True, True, False, False]]),
            ),
            FeatureLSTM,
        ),
    ],
)
def test_factory_builds_pytorch_lstm(encoder, featurizer, inputs, expected_class):
    model = ModelFactory().create(
        "lstm",
        encoder,
        featurizer,
        num_classes=3,
        dropout=0.0,
    )

    assert type(model) is expected_class
    assert type(model.lstm) is nn.LSTM
    assert model.lstm.bidirectional is True
    assert model.lstm.weight_ih_l0.shape[0] == 4 * model.lstm.hidden_size

    logits, embedding = model.forward_with_embedding(inputs)
    assert logits.shape == (2, 3)
    assert embedding.shape == (2, 2 * model.lstm.hidden_size)
    assert torch.equal(model(inputs), logits)


@pytest.mark.parametrize(
    ("legacy_model", "legacy_cell", "replacement_class", "inputs"),
    [
        (
            IndexRNN(vocab_size=24, embed_dim=8, hidden_dim=6, num_classes=3),
            nn.GRU(8, 6, batch_first=True, bidirectional=True),
            IndexGRU,
            torch.tensor([[1, 2, 3, 0]], dtype=torch.long),
        ),
        (
            FeatureRNN(in_dim=5, hidden_dim=6, num_classes=3),
            nn.GRU(5, 6, batch_first=True, bidirectional=True),
            FeatureGRU,
            torch.randn(1, 4, 5),
        ),
    ],
)
def test_legacy_rnn_named_gru_checkpoint_remains_loadable(
    tmp_path,
    legacy_model,
    legacy_cell,
    replacement_class,
    inputs,
):
    legacy_model.rnn = legacy_cell
    checkpoint = tmp_path / f"{type(legacy_model).__name__}.pt"
    torch.save(legacy_model, checkpoint)

    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    replacement_kwargs = (
        {"vocab_size": 24, "embed_dim": 8} if replacement_class is IndexGRU else {"in_dim": 5}
    )
    replacement = replacement_class(hidden_dim=6, num_classes=3, **replacement_kwargs)
    replacement.load_state_dict(loaded.state_dict())

    assert type(loaded) is type(legacy_model)
    assert type(loaded.rnn) is nn.GRU
    assert loaded(inputs).shape == (1, 3)
    assert replacement(inputs).shape == (1, 3)


@pytest.mark.parametrize(
    ("model", "adapter", "mask_source"),
    [
        (
            IndexRNN(vocab_size=24, embed_dim=8, hidden_dim=6, num_classes=3),
            "learned_embedding",
            "token_id_nonzero",
        ),
        (FeatureRNN(in_dim=5, hidden_dim=6, num_classes=3), "identity", "explicit_position_mask"),
        (
            PretrainedResiduePredictor(
                "rnn",
                input_dim=5,
                adapter_dim=8,
                hidden_dim=6,
                heads=2,
                num_classes=3,
            ),
            "layer_norm_linear_projection",
            "explicit_residue_mask",
        ),
    ],
)
def test_rnn_contract_v2_records_actual_elman_assembly(model, adapter, mask_source):
    assert model.rnn_predictor_contract_version == RNN_PREDICTOR_CONTRACT_VERSION
    contract = rnn_predictor_contract(model)

    assert contract == {
        "schema_version": 1,
        "implementation": "pepplatform_bidirectional_elman_rnn",
        "contract_version": 2,
        "status": "verified_platform_variant",
        "input_adapter": adapter,
        "actual_cell": "RNN",
        "nonlinearity": "tanh",
        "bidirectional": True,
        "sequence_handling": "mask_compact_pack_padded_sequence",
        "padding_mask": mask_source,
        "readout": "concatenated_final_forward_backward_hidden",
        "prediction_output": "logits",
    }
    assert sequence_predictor_contract(model) == contract


@pytest.mark.parametrize(
    ("model", "adapter", "mask_source"),
    [
        (
            IndexGRU(vocab_size=24, embed_dim=8, hidden_dim=6, num_classes=3),
            "learned_embedding",
            "token_id_nonzero",
        ),
        (FeatureGRU(in_dim=5, hidden_dim=6, num_classes=3), "identity", "explicit_position_mask"),
        (
            PretrainedResiduePredictor(
                "gru",
                input_dim=5,
                adapter_dim=8,
                hidden_dim=6,
                heads=2,
                num_classes=3,
            ),
            "layer_norm_linear_projection",
            "explicit_residue_mask",
        ),
    ],
)
def test_gru_contract_v2_records_actual_pytorch_assembly(model, adapter, mask_source):
    assert model.gru_predictor_contract_version == GRU_PREDICTOR_CONTRACT_VERSION
    contract = gru_predictor_contract(model)

    assert contract == {
        "schema_version": 1,
        "implementation": "pepplatform_bidirectional_pytorch_gru",
        "contract_version": 2,
        "status": "verified_platform_variant",
        "input_adapter": adapter,
        "actual_cell": "GRU",
        "reset_gate_placement": "after_hidden_affine",
        "bidirectional": True,
        "sequence_handling": "mask_compact_pack_padded_sequence",
        "padding_mask": mask_source,
        "readout": "concatenated_final_forward_backward_hidden",
        "prediction_output": "logits",
    }
    assert sequence_predictor_contract(model) == contract


@pytest.mark.parametrize(
    ("model", "adapter", "mask_source"),
    [
        (
            IndexLSTM(vocab_size=24, embed_dim=8, hidden_dim=6, num_classes=3),
            "learned_embedding",
            "token_id_nonzero",
        ),
        (FeatureLSTM(in_dim=5, hidden_dim=6, num_classes=3), "identity", "explicit_position_mask"),
        (
            PretrainedResiduePredictor(
                "lstm",
                input_dim=5,
                adapter_dim=8,
                hidden_dim=6,
                heads=2,
                num_classes=3,
            ),
            "layer_norm_linear_projection",
            "explicit_residue_mask",
        ),
    ],
)
def test_lstm_contract_v2_records_actual_pytorch_assembly(model, adapter, mask_source):
    assert model.lstm_predictor_contract_version == LSTM_PREDICTOR_CONTRACT_VERSION
    contract = lstm_predictor_contract(model)

    assert contract == {
        "schema_version": 1,
        "implementation": "pepplatform_bidirectional_pytorch_lstm",
        "contract_version": 2,
        "status": "verified_platform_variant",
        "input_adapter": adapter,
        "actual_cell": "LSTM",
        "gate_order": ["input", "forget", "cell", "output"],
        "state_components": ["hidden", "cell"],
        "cell_state_update": "forget_retained_cell_plus_input_candidate",
        "bidirectional": True,
        "sequence_handling": "mask_compact_pack_padded_sequence",
        "padding_mask": mask_source,
        "readout": "concatenated_final_forward_backward_hidden",
        "prediction_output": "logits",
    }
    assert sequence_predictor_contract(model) == contract


def test_index_rnn_is_invariant_to_right_and_internal_pair_padding():
    torch.manual_seed(17)
    model = IndexRNN(vocab_size=24, embed_dim=8, hidden_dim=6, num_classes=3, dropout=0.0).eval()
    compact = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    right_padded = torch.tensor([[1, 2, 3, 4, 0, 0, 0]], dtype=torch.long)
    pair_joined = torch.tensor([[1, 2, 0, 0, 0, 3, 4]], dtype=torch.long)
    companion = torch.tensor([[5, 6, 7, 8, 9, 0, 0]], dtype=torch.long)

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


def test_index_rnn_validates_ids_masks_padding_gradients_and_round_trips(tmp_path):
    torch.manual_seed(19)
    model = IndexRNN(vocab_size=12, embed_dim=8, hidden_dim=6, num_classes=2, dropout=0.0)
    inputs = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]], dtype=torch.long)

    logits, embedding = model.forward_with_embedding(inputs)
    logits.sum().backward()

    assert logits.shape == (2, 2)
    assert embedding.shape == (2, 12)
    assert torch.count_nonzero(model.embedding.weight.grad[0]) == 0
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    with pytest.raises(ValueError, match="at least one non-padding token"):
        model(torch.zeros((1, 4), dtype=torch.long))
    with pytest.raises(TypeError, match="int32 or int64"):
        model(torch.ones((1, 4), dtype=torch.float))
    with pytest.raises(ValueError, match=r"\[0, 12\)"):
        model(torch.tensor([[1, 12]], dtype=torch.long))

    path = tmp_path / "rnn_v2.pt"
    torch.save(model.eval(), path)
    loaded = torch.load(path, map_location="cpu", weights_only=False).eval()
    assert torch.equal(model(inputs), loaded(inputs))
    assert rnn_predictor_contract(loaded)["status"] == "verified_platform_variant"


def test_index_gru_is_invariant_to_right_and_internal_pair_padding():
    torch.manual_seed(31)
    model = IndexGRU(vocab_size=24, embed_dim=8, hidden_dim=6, num_classes=3, dropout=0.0).eval()
    compact = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    right_padded = torch.tensor([[1, 2, 3, 4, 0, 0, 0]], dtype=torch.long)
    pair_joined = torch.tensor([[1, 2, 0, 0, 0, 3, 4]], dtype=torch.long)
    companion = torch.tensor([[5, 6, 7, 8, 9, 0, 0]], dtype=torch.long)

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


def test_index_gru_validates_ids_masks_padding_gradients_and_round_trips(tmp_path):
    torch.manual_seed(37)
    model = IndexGRU(vocab_size=12, embed_dim=8, hidden_dim=6, num_classes=2, dropout=0.0)
    inputs = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]], dtype=torch.long)

    logits, embedding = model.forward_with_embedding(inputs)
    logits.sum().backward()

    assert logits.shape == (2, 2)
    assert embedding.shape == (2, 12)
    assert torch.isfinite(logits).all()
    assert torch.count_nonzero(model.embedding.weight.grad[0]) == 0
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    with pytest.raises(ValueError, match="at least one non-padding token"):
        model(torch.zeros((1, 4), dtype=torch.long))
    with pytest.raises(TypeError, match="int32 or int64"):
        model(torch.ones((1, 4), dtype=torch.float))
    with pytest.raises(ValueError, match=r"\[0, 12\)"):
        model(torch.tensor([[1, 12]], dtype=torch.long))
    with pytest.raises(ValueError, match="sequence length must be positive"):
        model(torch.empty((1, 0), dtype=torch.long))

    path = tmp_path / "gru_v2.pt"
    torch.save(model.eval(), path)
    loaded = torch.load(path, map_location="cpu", weights_only=False).eval()
    assert torch.equal(model(inputs), loaded(inputs))
    assert gru_predictor_contract(loaded)["status"] == "verified_platform_variant"


def test_index_lstm_is_invariant_to_right_and_internal_pair_padding():
    torch.manual_seed(47)
    model = IndexLSTM(vocab_size=24, embed_dim=8, hidden_dim=6, num_classes=3, dropout=0.0).eval()
    compact = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    right_padded = torch.tensor([[1, 2, 3, 4, 0, 0, 0]], dtype=torch.long)
    pair_joined = torch.tensor([[1, 2, 0, 0, 0, 3, 4]], dtype=torch.long)
    companion = torch.tensor([[5, 6, 7, 8, 9, 0, 0]], dtype=torch.long)

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


def test_index_lstm_validates_ids_masks_padding_gradients_and_round_trips(tmp_path):
    torch.manual_seed(53)
    model = IndexLSTM(vocab_size=12, embed_dim=8, hidden_dim=6, num_classes=2, dropout=0.0)
    inputs = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]], dtype=torch.long)

    logits, embedding = model.forward_with_embedding(inputs)
    logits.sum().backward()

    assert logits.shape == (2, 2)
    assert embedding.shape == (2, 12)
    assert torch.isfinite(logits).all()
    assert torch.count_nonzero(model.embedding.weight.grad[0]) == 0
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    with pytest.raises(ValueError, match="at least one non-padding token"):
        model(torch.zeros((1, 4), dtype=torch.long))
    with pytest.raises(ValueError, match=r"shape \[B,L\]"):
        model(torch.ones((1, 4, 1), dtype=torch.long))
    with pytest.raises(TypeError, match="int32 or int64"):
        model(torch.ones((1, 4), dtype=torch.float))
    with pytest.raises(ValueError, match=r"\[0, 12\)"):
        model(torch.tensor([[1, 12]], dtype=torch.long))
    with pytest.raises(ValueError, match="sequence length must be positive"):
        model(torch.empty((1, 0), dtype=torch.long))

    path = tmp_path / "lstm_v2.pt"
    torch.save(model.eval(), path)
    loaded = torch.load(path, map_location="cpu", weights_only=False).eval()
    assert torch.equal(model(inputs), loaded(inputs))
    assert lstm_predictor_contract(loaded)["status"] == "verified_platform_variant"


@pytest.mark.parametrize("model_kind", ["feature", "pretrained"])
def test_continuous_rnn_routes_ignore_internal_masked_values(model_kind):
    torch.manual_seed(29)
    if model_kind == "feature":
        model = FeatureRNN(in_dim=5, hidden_dim=6, num_classes=2, dropout=0.0)
    else:
        model = PretrainedResiduePredictor(
            "rnn",
            input_dim=5,
            adapter_dim=8,
            hidden_dim=6,
            heads=2,
            num_classes=2,
            dropout=0.0,
        )
    model.eval()
    valid = torch.randn(1, 4, 5)
    compact = ResidueModelInput(valid, torch.ones((1, 4), dtype=torch.bool))
    polluted = ResidueModelInput(
        torch.cat([valid[:, :2], torch.full((1, 3, 5), 123.0), valid[:, 2:]], dim=1),
        torch.tensor([[True, True, False, False, False, True, True]]),
    )

    compact_logits, compact_embedding = model.forward_with_embedding(compact)
    polluted_logits, polluted_embedding = model.forward_with_embedding(polluted)

    assert torch.allclose(compact_embedding, polluted_embedding, atol=1e-6)
    assert torch.allclose(compact_logits, polluted_logits, atol=1e-6)


@pytest.mark.parametrize("model_kind", ["feature", "pretrained"])
def test_continuous_gru_routes_ignore_internal_masked_values_and_gradients(model_kind):
    torch.manual_seed(41)
    if model_kind == "feature":
        model = FeatureGRU(in_dim=5, hidden_dim=6, num_classes=2, dropout=0.0)
    else:
        model = PretrainedResiduePredictor(
            "gru",
            input_dim=5,
            adapter_dim=8,
            hidden_dim=6,
            heads=2,
            num_classes=2,
            dropout=0.0,
        )
    model.eval()
    valid = torch.randn(1, 4, 5)
    compact = ResidueModelInput(valid, torch.ones((1, 4), dtype=torch.bool))
    polluted_features = torch.cat(
        [valid[:, :2], torch.full((1, 3, 5), 123.0), valid[:, 2:]],
        dim=1,
    ).requires_grad_()
    polluted = ResidueModelInput(
        polluted_features,
        torch.tensor([[True, True, False, False, False, True, True]]),
    )

    compact_logits, compact_embedding = model.forward_with_embedding(compact)
    polluted_logits, polluted_embedding = model.forward_with_embedding(polluted)
    polluted_logits.sum().backward()

    assert torch.allclose(compact_embedding, polluted_embedding, atol=1e-6)
    assert torch.allclose(compact_logits, polluted_logits, atol=1e-6)
    assert torch.count_nonzero(polluted_features.grad[:, 2:5]) == 0
    assert torch.isfinite(polluted_features.grad).all()


@pytest.mark.parametrize("model_kind", ["feature", "pretrained"])
def test_continuous_lstm_routes_ignore_internal_masked_values_and_gradients(model_kind):
    torch.manual_seed(59)
    if model_kind == "feature":
        model = FeatureLSTM(in_dim=5, hidden_dim=6, num_classes=2, dropout=0.0)
    else:
        model = PretrainedResiduePredictor(
            "lstm",
            input_dim=5,
            adapter_dim=8,
            hidden_dim=6,
            heads=2,
            num_classes=2,
            dropout=0.0,
        )
    model.eval()
    valid = torch.randn(1, 4, 5)
    compact = ResidueModelInput(valid, torch.ones((1, 4), dtype=torch.bool))
    polluted_features = torch.cat(
        [valid[:, :2], torch.full((1, 3, 5), 123.0), valid[:, 2:]],
        dim=1,
    ).requires_grad_()
    polluted = ResidueModelInput(
        polluted_features,
        torch.tensor([[True, True, False, False, False, True, True]]),
    )

    compact_logits, compact_embedding = model.forward_with_embedding(compact)
    polluted_logits, polluted_embedding = model.forward_with_embedding(polluted)
    polluted_logits.sum().backward()

    assert torch.allclose(compact_embedding, polluted_embedding, atol=1e-6)
    assert torch.allclose(compact_logits, polluted_logits, atol=1e-6)
    assert torch.count_nonzero(polluted_features.grad[:, 2:5]) == 0
    assert torch.isfinite(polluted_features.grad).all()


def test_rnn_explicit_position_mask_preserves_valid_zero_oov(tmp_path):
    embedding_path = tmp_path / "vectors.txt"
    embedding_path.write_text("1 3\nA 1.0 2.0 3.0\n", encoding="utf-8")
    tokenizer = KmerTokenizer(max_length=4, k=1)
    encoder = StaticEmbeddingEncoder(
        max_length=4,
        embedding_path=embedding_path,
        has_header=True,
    )
    featurizer = ComposedFeaturizer(tokenizer, encoder).require_explicit_position_mask()

    encoded = featurizer.encode("AX")

    assert isinstance(encoded, ResidueModelInput)
    assert encoded.mask.tolist() == [True, True, False, False]
    assert torch.count_nonzero(encoded.features[1]) == 0


@pytest.mark.parametrize("architecture", ["rnn", "gru", "lstm", "gcn"])
def test_pair_recurrent_builder_enables_explicit_position_masks(tmp_path, architecture):
    args = SimpleNamespace(
        tokenizer="kmer",
        encoder="onehot",
        model=architecture,
        sequence_model=architecture,
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


@pytest.mark.parametrize("architecture", ["rnn", "gru", "lstm", "gcn"])
def test_sequence_recurrent_builder_enables_explicit_position_masks(architecture):
    pipeline = object.__new__(Pipeline)
    pipeline.args = SimpleNamespace(
        tokenizer="kmer",
        encoder="onehot",
        model=architecture,
        sequence_model=architecture,
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


def test_unversioned_and_mislabeled_rnn_contracts_are_legacy_only():
    unversioned = IndexRNN(vocab_size=12, embed_dim=8, hidden_dim=6, num_classes=2)
    del unversioned.rnn_predictor_contract_version
    with pytest.warns(RuntimeWarning, match="does not carry"):
        unversioned(torch.tensor([[1, 2, 0]], dtype=torch.long))
    assert rnn_predictor_contract(unversioned)["status"] == "legacy_only"

    mislabeled = IndexRNN(vocab_size=12, embed_dim=8, hidden_dim=6, num_classes=2)
    mislabeled.rnn = nn.GRU(8, 6, batch_first=True, bidirectional=True)
    with pytest.warns(RuntimeWarning, match="contains a GRU"):
        mislabeled(torch.tensor([[1, 2, 0]], dtype=torch.long))
    contract = rnn_predictor_contract(mislabeled)
    assert contract["status"] == "legacy_only"
    assert contract["implementation"] == "legacy_mislabeled_bidirectional_gru"
    assert contract["actual_cell"] == "GRU"


@pytest.mark.parametrize(
    "model",
    [
        IndexGRU(vocab_size=12, embed_dim=8, hidden_dim=6, num_classes=2),
        FeatureGRU(in_dim=5, hidden_dim=6, num_classes=2),
        PretrainedResiduePredictor(
            "gru",
            input_dim=5,
            adapter_dim=8,
            hidden_dim=6,
            heads=2,
            num_classes=2,
        ),
    ],
)
def test_unversioned_gru_contracts_are_legacy_only(model, tmp_path):
    del model.gru_predictor_contract_version
    contract = gru_predictor_contract(model)

    assert contract["contract_version"] == 1
    assert contract["status"] == "legacy_only"
    assert contract["implementation"] == "pepplatform_unversioned_bidirectional_pytorch_gru"

    path = tmp_path / f"{type(model).__name__}_legacy_gru.pt"
    torch.save(model.eval(), path)
    loaded = torch.load(path, map_location="cpu", weights_only=False).eval()
    if isinstance(loaded, IndexGRU):
        inputs = torch.tensor([[1, 2, 0, 0]], dtype=torch.long)
    elif isinstance(loaded, FeatureGRU):
        inputs = torch.randn(1, 4, 5)
    else:
        inputs = ResidueModelInput(
            torch.randn(1, 4, 5),
            torch.tensor([[True, True, False, False]]),
        )
    with pytest.warns(RuntimeWarning, match="does not carry"):
        logits = loaded(inputs)
    assert torch.isfinite(logits).all()


def test_unversioned_index_gru_preserves_raw_padded_numerics():
    torch.manual_seed(43)
    model = IndexGRU(vocab_size=12, embed_dim=8, hidden_dim=6, num_classes=2, dropout=0.0).eval()
    del model.gru_predictor_contract_version
    inputs = torch.tensor([[1, 2, 0, 0]], dtype=torch.long)

    expected_logits, expected_embedding = model._legacy_forward_with_embedding(inputs)
    with pytest.warns(RuntimeWarning, match="does not carry"):
        logits, embedding = model.forward_with_embedding(inputs)

    assert torch.equal(logits, expected_logits)
    assert torch.equal(embedding, expected_embedding)


@pytest.mark.parametrize(
    "model",
    [
        IndexLSTM(vocab_size=12, embed_dim=8, hidden_dim=6, num_classes=2),
        FeatureLSTM(in_dim=5, hidden_dim=6, num_classes=2),
        PretrainedResiduePredictor(
            "lstm",
            input_dim=5,
            adapter_dim=8,
            hidden_dim=6,
            heads=2,
            num_classes=2,
        ),
    ],
)
def test_unversioned_lstm_contracts_are_legacy_only(model, tmp_path):
    del model.lstm_predictor_contract_version
    contract = lstm_predictor_contract(model)

    assert contract["contract_version"] == 1
    assert contract["status"] == "legacy_only"
    assert contract["implementation"] == "pepplatform_unversioned_bidirectional_pytorch_lstm"

    path = tmp_path / f"{type(model).__name__}_legacy_lstm.pt"
    torch.save(model.eval(), path)
    loaded = torch.load(path, map_location="cpu", weights_only=False).eval()
    if isinstance(loaded, IndexLSTM):
        inputs = torch.tensor([[1, 2, 0, 0]], dtype=torch.long)
    elif isinstance(loaded, FeatureLSTM):
        inputs = torch.randn(1, 4, 5)
    else:
        inputs = ResidueModelInput(
            torch.randn(1, 4, 5),
            torch.tensor([[True, True, False, False]]),
        )
    with pytest.warns(RuntimeWarning, match="does not carry"):
        logits = loaded(inputs)
    assert torch.isfinite(logits).all()


def test_unversioned_index_lstm_preserves_raw_padded_numerics():
    torch.manual_seed(61)
    model = IndexLSTM(vocab_size=12, embed_dim=8, hidden_dim=6, num_classes=2, dropout=0.0).eval()
    del model.lstm_predictor_contract_version
    inputs = torch.tensor([[1, 2, 0, 0]], dtype=torch.long)

    expected_logits, expected_embedding = model._legacy_forward_with_embedding(inputs)
    with pytest.warns(RuntimeWarning, match="does not carry"):
        logits, embedding = model.forward_with_embedding(inputs)

    assert torch.equal(logits, expected_logits)
    assert torch.equal(embedding, expected_embedding)


def test_unversioned_pretrained_lstm_preserves_length_only_packed_numerics():
    torch.manual_seed(67)
    model = PretrainedResiduePredictor(
        "lstm",
        input_dim=5,
        adapter_dim=8,
        hidden_dim=6,
        heads=2,
        num_classes=2,
        dropout=0.0,
    ).eval()
    inputs = ResidueModelInput(
        torch.randn(1, 6, 5),
        torch.tensor([[True, True, False, False, True, True]]),
    )
    features, mask = model._validate_input(inputs)
    adapted = model._adapt(features, mask)
    packed = torch.nn.utils.rnn.pack_padded_sequence(
        adapted,
        mask.sum(dim=1).to(dtype=torch.long).cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    _, (hidden, _) = model.recurrent(packed)
    expected_embedding = torch.cat([hidden[-2], hidden[-1]], dim=1)
    expected_logits = model.head(expected_embedding)
    del model.lstm_predictor_contract_version

    with pytest.warns(RuntimeWarning, match="does not carry"):
        logits, embedding = model.forward_with_embedding(inputs)

    assert torch.equal(logits, expected_logits)
    assert torch.equal(embedding, expected_embedding)
