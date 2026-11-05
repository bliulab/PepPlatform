import warnings
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from pepplatform.workflow import (
    CHAIN_GRAPH_COMPATIBLE_ENCODERS,
    SEQUENCE_ENCODERS,
)
from pepplatform.pair_fusion import AlignedGatedFusion
from pepplatform.pssm_covariance import PSSM_COVARIANCE_ENCODERS


class ResidueModelInput(NamedTuple):
    features: torch.Tensor
    mask: torch.Tensor

    def to(self, device):
        return ResidueModelInput(self.features.to(device), self.mask.to(device))

    def cpu(self):
        return ResidueModelInput(self.features.cpu(), self.mask.cpu())

    def detach(self):
        return ResidueModelInput(self.features.detach(), self.mask.detach())

    def float(self):
        return ResidueModelInput(self.features.float(), self.mask)

    def index_select(self, dim, index):
        return ResidueModelInput(
            self.features.index_select(dim, index),
            self.mask.index_select(dim, index),
        )

    def batch_slice(self, start, end):
        return ResidueModelInput(self.features[start:end], self.mask[start:end])

    @property
    def shape(self):
        return self.features.shape

    @property
    def ndim(self):
        return self.features.ndim

    def numel(self):
        return self.features.numel()


def _residue_features_and_mask(inputs):
    """Unpack explicit residue masks, retaining tensor-only compatibility."""

    if isinstance(inputs, ResidueModelInput):
        features, mask = inputs.features, inputs.mask
    elif (
        isinstance(inputs, (tuple, list))
        and len(inputs) == 2
        and isinstance(inputs[0], torch.Tensor)
        and isinstance(inputs[1], torch.Tensor)
    ):
        features, mask = inputs
    else:
        features = inputs
        if not isinstance(features, torch.Tensor):
            raise TypeError("residue model input must be a Tensor or ResidueModelInput")
        mask = residue_mask(features)
    if features.ndim != 3 or mask.shape != features.shape[:2]:
        raise ValueError(
            "residue features/mask must have shapes [B,L,D] and [B,L], "
            f"got {tuple(features.shape)} and {tuple(mask.shape)}"
        )
    mask = mask.to(device=features.device, dtype=torch.bool)
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every residue sample must contain at least one valid position")
    return features, mask


def _masked_max_pool(values, mask):
    valid = mask.unsqueeze(-1)
    pooled = (
        values.masked_fill(
            ~valid,
            torch.finfo(values.dtype).min,
        )
        .max(dim=1)
        .values
    )
    return torch.where(
        mask.any(dim=1, keepdim=True),
        pooled,
        torch.zeros_like(pooled),
    )


def _run_feature_recurrent(module, inputs):
    features, mask = _residue_features_and_mask(inputs)
    lengths = mask.sum(dim=1).to(dtype=torch.long)
    compacted = features.new_zeros(features.shape)
    for row in range(features.shape[0]):
        count = int(lengths[row].item())
        compacted[row, :count] = features[row, mask[row]]
    packed = pack_padded_sequence(
        compacted,
        lengths.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    return module(packed)


RNN_PREDICTOR_CONTRACT_VERSION = 2
GRU_PREDICTOR_CONTRACT_VERSION = 2
LSTM_PREDICTOR_CONTRACT_VERSION = 2
TRANSFORMER_PREDICTOR_CONTRACT_VERSION = 2


def _rnn_cell(model):
    if isinstance(model, PretrainedResiduePredictor):
        return getattr(model, "recurrent", None)
    return getattr(model, "rnn", None)


def _rnn_contract_version(model):
    try:
        return int(getattr(model, "rnn_predictor_contract_version", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _uses_verified_rnn_contract(model):
    cell = _rnn_cell(model)
    version = _rnn_contract_version(model)
    return bool(
        version >= RNN_PREDICTOR_CONTRACT_VERSION
        and type(cell) is nn.RNN
        and getattr(cell, "nonlinearity", None) == "tanh"
        and bool(getattr(cell, "bidirectional", False))
    )


def _warn_legacy_rnn(model):
    cell = _rnn_cell(model)
    if type(cell) is nn.GRU:
        detail = "contains a GRU cell under the historical RNN name"
    else:
        detail = "does not carry the current mask-aware RNN contract"
    warnings.warn(
        f"{type(model).__name__} {detail}; it remains available for artifact "
        "compatibility but must not be used as a current RNN baseline without "
        "retraining",
        RuntimeWarning,
        stacklevel=3,
    )


def _gru_cell(model):
    if isinstance(model, PretrainedResiduePredictor):
        return getattr(model, "recurrent", None)
    return getattr(model, "rnn", None)


def _gru_contract_version(model):
    try:
        return int(getattr(model, "gru_predictor_contract_version", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _uses_verified_gru_contract(model):
    cell = _gru_cell(model)
    version = _gru_contract_version(model)
    return bool(
        version >= GRU_PREDICTOR_CONTRACT_VERSION
        and type(cell) is nn.GRU
        and bool(getattr(cell, "bidirectional", False))
    )


def _warn_legacy_gru(model):
    warnings.warn(
        f"{type(model).__name__} does not carry the current mask-aware GRU "
        "contract; it remains available for artifact compatibility but must "
        "not be used as a current GRU baseline without retraining",
        RuntimeWarning,
        stacklevel=3,
    )


def _lstm_cell(model):
    if isinstance(model, PretrainedResiduePredictor):
        return getattr(model, "recurrent", None)
    return getattr(model, "lstm", None)


def _lstm_contract_version(model):
    try:
        return int(getattr(model, "lstm_predictor_contract_version", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _uses_verified_lstm_contract(model):
    cell = _lstm_cell(model)
    version = _lstm_contract_version(model)
    return bool(
        version >= LSTM_PREDICTOR_CONTRACT_VERSION
        and type(cell) is nn.LSTM
        and bool(getattr(cell, "bidirectional", False))
    )


def _warn_legacy_lstm(model):
    warnings.warn(
        f"{type(model).__name__} does not carry the current mask-aware LSTM "
        "contract; it remains available for artifact compatibility but must "
        "not be used as a current LSTM baseline without retraining",
        RuntimeWarning,
        stacklevel=3,
    )


def _transformer_encoder(model):
    if isinstance(model, PretrainedResiduePredictor):
        return getattr(model, "transformer", None)
    return getattr(model, "encoder", None)


def _transformer_contract_version(model):
    try:
        return int(getattr(model, "transformer_predictor_contract_version", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _uses_verified_transformer_contract(model):
    encoder = _transformer_encoder(model)
    layers = list(getattr(encoder, "layers", []))
    return bool(
        _transformer_contract_version(model) >= TRANSFORMER_PREDICTOR_CONTRACT_VERSION
        and type(encoder) is nn.TransformerEncoder
        and layers
        and all(type(layer) is nn.TransformerEncoderLayer for layer in layers)
        and all(
            type(getattr(layer, "self_attn", None)) is nn.MultiheadAttention
            and bool(getattr(layer.self_attn, "batch_first", False))
            for layer in layers
        )
        and type(getattr(model, "position", None)) is nn.Embedding
    )


def _warn_legacy_transformer(model):
    warnings.warn(
        f"{type(model).__name__} does not carry the current mask-aware "
        "Transformer contract; it remains available for artifact compatibility "
        "but must not be used as a current Transformer baseline without "
        "retraining",
        RuntimeWarning,
        stacklevel=3,
    )


def _mask_compact_position_ids(mask, num_positions):
    """Assign contiguous position IDs to valid tokens while ignoring mask holes."""

    counts = mask.sum(dim=1, dtype=torch.long)
    if counts.numel() and int(counts.max().item()) > int(num_positions):
        raise ValueError(
            "Transformer valid sequence length exceeds positional embedding "
            f"capacity {int(num_positions)}"
        )
    ranked = mask.to(dtype=torch.long).cumsum(dim=1) - 1
    return torch.where(mask, ranked, torch.zeros_like(ranked))


def _validate_prediction_head_dimensions(input_dim, output_dim, hidden_dims):
    """Validate the fixed-width dimensions before constructing a head."""

    for name, value in (("input_dim", input_dim), ("output_dim", output_dim)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for index, value in enumerate(hidden_dims):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"hidden_dims[{index}] must be a positive integer")


def _validate_prediction_head_input(inputs, input_dim):
    if not isinstance(inputs, torch.Tensor) or inputs.ndim != 2:
        raise ValueError("prediction-head input must have shape [samples, features]")
    if int(inputs.shape[1]) != int(input_dim):
        raise ValueError(
            f"prediction-head input feature dimension must be {int(input_dim)}, "
            f"got {int(inputs.shape[1])}"
        )
    if int(inputs.shape[0]) <= 0:
        raise ValueError("prediction-head input must contain at least one sample")
    if not inputs.dtype.is_floating_point:
        raise ValueError("prediction-head input must use a floating-point dtype")
    if not torch.isfinite(inputs).all():
        raise ValueError("prediction-head input must contain only finite values")


def _validate_prediction_head_output(outputs):
    if not torch.isfinite(outputs).all():
        raise RuntimeError("prediction head produced non-finite logits")


def _prediction_head_input_dim(module):
    if isinstance(module, nn.Linear):
        return int(module.in_features)
    if isinstance(module, nn.Sequential):
        for layer in module:
            if isinstance(layer, nn.Linear):
                return int(layer.in_features)
    return None


class _ValidatedLinear(nn.Linear):
    """Linear prediction head that enforces the fixed-width tensor contract."""

    def forward(self, inputs):
        _validate_prediction_head_input(inputs, self.in_features)
        outputs = super().forward(inputs)
        _validate_prediction_head_output(outputs)
        return outputs


class _ValidatedSequential(nn.Sequential):
    """MLP prediction head with explicit input and finite-logit checks."""

    def __init__(self, input_dim, *modules):
        super().__init__(*modules)
        self.input_dim = int(input_dim)

    def forward(self, inputs):
        _validate_prediction_head_input(inputs, self.input_dim)
        outputs = super().forward(inputs)
        _validate_prediction_head_output(outputs)
        return outputs


def build_prediction_head(
    input_dim,
    output_dim,
    hidden_dims=None,
    activation="relu",
    dropout=0.0,
    default_hidden_dims=None,
):
    """Build a logits-producing linear or MLP prediction head."""
    widths = list(default_hidden_dims or []) if hidden_dims is None else list(hidden_dims)
    _validate_prediction_head_dimensions(input_dim, output_dim, widths)
    try:
        dropout_value = float(dropout)
    except (TypeError, ValueError) as exc:
        raise ValueError("prediction-head dropout must be numeric") from exc
    if not np.isfinite(dropout_value) or not 0.0 <= dropout_value < 1.0:
        raise ValueError("prediction-head dropout must be finite and in [0, 1)")
    if not widths:
        return _ValidatedLinear(input_dim, output_dim)
    activation_factories = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    activation_name = str(activation or "relu").strip().lower()
    if activation_name not in activation_factories:
        raise ValueError(f"unsupported prediction-head activation: {activation_name}")
    layers = []
    current_dim = int(input_dim)
    for width in widths:
        layers.extend(
            [
                nn.Linear(current_dim, int(width)),
                activation_factories[activation_name](),
                nn.Dropout(dropout_value),
            ]
        )
        current_dim = int(width)
    layers.append(nn.Linear(current_dim, int(output_dim)))
    return _ValidatedSequential(input_dim, *layers)


class PredictionHeadClassifier(nn.Module):
    """Classify an existing vector representation with a configurable head."""

    def __init__(
        self,
        input_dim,
        num_classes,
        dropout=0.0,
        head_hidden_dims=None,
        head_activation="relu",
        head_dropout=None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.head = build_prediction_head(
            input_dim,
            num_classes,
            hidden_dims=head_hidden_dims,
            activation=head_activation,
            dropout=dropout if head_dropout is None else head_dropout,
            default_hidden_dims=[128],
        )

    def forward_with_embedding(self, x):
        input_dim = getattr(self, "input_dim", None) or _prediction_head_input_dim(self.head)
        if input_dim is None:
            raise RuntimeError("prediction-head input dimension is missing")
        _validate_prediction_head_input(x, input_dim)
        if isinstance(self.head, nn.Linear):
            return self.head(x), x
        embedding = x
        for layer in list(self.head.children())[:-1]:
            embedding = layer(embedding)
        logits = self.head[-1](embedding)
        _validate_prediction_head_output(logits)
        return logits, embedding

    @property
    def semantic_embedding_dim(self) -> int:
        if isinstance(self.head, nn.Linear):
            return int(self.head.in_features)
        for layer in reversed(list(self.head.children())):
            if isinstance(layer, nn.Linear):
                return int(layer.in_features)
        raise RuntimeError("prediction head has no linear embedding layer")

    def forward(self, x):
        logits, _ = self.forward_with_embedding(x)
        return logits


class SemanticResidualAdapter(nn.Module):
    """Inference-visible semantic residual: classify h + w * proj(h).

    The wrapper keeps the base readout embedding ``h`` available through
    ``forward_with_embedding`` so the train-only semantic contrastive loss can
    still constrain the projection of the *unmodified* representation, while
    classification (training, validation and inference) consumes the residual
    representation ``h + w * proj(h)``.  The wrapper is intentionally a
    plain module so ``torch.save``/``torch.load`` round-trips through the same
    class and inference applies the same forward.
    """

    def __init__(self, base: nn.Module, weight: float):
        super().__init__()
        if not hasattr(base, "head") or not hasattr(base, "semantic_embedding_dim"):
            raise ValueError(
                "semantic residual adapter requires a base model with a head "
                "and a sample-level semantic embedding"
            )
        self.base = base
        self.weight = float(weight)
        projector = getattr(base, "semantic_projector", None)
        if projector is None:
            raise ValueError(
                "semantic residual adapter requires a semantic projector on the base model"
            )
        projection_dim = int(projector.output_dim)
        embedding_dim = int(base.semantic_embedding_dim)
        self.residual_projection = torch.nn.Linear(projection_dim, embedding_dim)
        device = next(base.parameters()).device
        self.residual_projection = self.residual_projection.to(device)

    def forward(self, inputs):
        return self.forward_with_embedding(inputs)[0]

    def forward_with_embedding(self, inputs):
        logits, embedding = self.base.forward_with_embedding(inputs)
        projector = getattr(self.base, "semantic_projector", None)
        if projector is None:
            raise RuntimeError(
                "semantic residual adapter requires prepare_semantic_projectors "
                "to be called on the base model"
            )
        residual = embedding + self.weight * self.residual_projection(projector(embedding))
        if isinstance(self.base.head, torch.nn.Linear):
            logits = self.base.head(residual)
        else:
            logits = self.base.head[-1](residual)
        return logits, embedding

    @property
    def semantic_embedding_dim(self) -> int:
        return int(self.base.semantic_embedding_dim)

    @property
    def head(self):
        return self.base.head

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = getattr(self, "base", None)
            if base is not None:
                return getattr(base, name)
            raise


class PairSemanticResidualAdapter(nn.Module):
    """Inference-visible semantic residual for two-input pair models.

    Classification consumes ``fused + w * residual_projection(projector(fused))``
    while ``forward_with_embedding`` and ``forward_with_role_embeddings`` still
    return the base fused/role embeddings, so the train-only contrastive loss
    keeps constraining the unmodified representation.

    With ``learned_gate=True`` the scalar ``w`` is replaced by a trainable
    parameter initialized at zero, so training starts exactly at the base
    model and the semantic path is turned on only when the gradient finds it
    useful.  ``forward_with_effective_embedding`` exposes the residual
    representation actually consumed by the head, which is the natural
    shared-space target for the semantic contrastive loss.
    """

    def __init__(
        self,
        base: nn.Module,
        weight: float,
        *,
        learned_gate: bool = False,
    ):
        super().__init__()
        if not hasattr(base, "head") or not hasattr(base, "semantic_embedding_dim"):
            raise ValueError(
                "pair semantic residual adapter requires a base model with a head "
                "and a sample-level fused embedding"
            )
        if not hasattr(base, "forward_with_role_embeddings"):
            raise ValueError(
                "pair semantic residual adapter requires a base model with "
                "forward_with_role_embeddings"
            )
        self.base = base
        self.weight = float(weight)
        self.learned_gate = bool(learned_gate)
        projector = getattr(base, "semantic_projector", None)
        if projector is None:
            raise ValueError(
                "pair semantic residual adapter requires a semantic projector on the base model"
            )
        projection_dim = int(projector.output_dim)
        embedding_dim = int(base.semantic_embedding_dim)
        self.residual_projection = torch.nn.Linear(projection_dim, embedding_dim)
        device = next(base.parameters()).device
        self.residual_projection = self.residual_projection.to(device)
        if self.learned_gate:
            self.gate = torch.nn.Parameter(torch.zeros((), device=device))

    def _residual_embedding(self, fused):
        projector = getattr(self.base, "semantic_projector", None)
        if projector is None:
            raise RuntimeError(
                "pair semantic residual adapter requires prepare_semantic_projectors "
                "to be called on the base model"
            )
        projected = projector(fused)
        if self.learned_gate:
            residual = fused + self.gate * self.residual_projection(projected)
        else:
            residual = fused + self.weight * self.residual_projection(projected)
        return residual

    def _residual_logits(self, fused):
        residual = self._residual_embedding(fused)
        return self.base.head(residual)

    def forward(self, peptide, partner):
        return self.forward_with_embedding(peptide, partner)[0]

    def forward_with_embedding(self, peptide, partner):
        logits, embedding = self.base.forward_with_embedding(peptide, partner)
        return self._residual_logits(embedding), embedding

    def forward_with_effective_embedding(self, peptide, partner):
        logits, embedding = self.base.forward_with_embedding(peptide, partner)
        effective = self._residual_embedding(embedding)
        return self._residual_logits(embedding), effective

    def forward_with_role_embeddings(self, peptide, partner):
        logits, fused, peptide_role, partner_role = self.base.forward_with_role_embeddings(
            peptide, partner
        )
        return (
            self._residual_logits(fused),
            fused,
            peptide_role,
            partner_role,
        )

    @property
    def semantic_embedding_dim(self) -> int:
        return int(self.base.semantic_embedding_dim)

    @property
    def semantic_role_embedding_dims(self) -> tuple[int, int] | None:
        return self.base.semantic_role_embedding_dims

    @property
    def head(self):
        return self.base.head

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = getattr(self, "base", None)
            if base is not None:
                return getattr(base, name)
            raise


class AlignedGatedClassifier(nn.Module):
    """Pair classifier backed by the versioned aligned-gated adapter."""

    def __init__(
        self,
        input_dim=None,
        hidden_dim=256,
        dropout=0.2,
        peptide_dim=None,
        partner_dim=None,
        align_dim=None,
        head_hidden_dims=None,
        head_activation="relu",
        head_dropout=None,
    ):
        super().__init__()
        if input_dim is not None:
            if peptide_dim is not None or partner_dim is not None:
                raise ValueError("input_dim cannot be combined with peptide_dim or partner_dim")
            peptide_dim = partner_dim = input_dim
            alignment_dim = None
        else:
            if peptide_dim is None or partner_dim is None:
                raise ValueError("provide input_dim or both peptide_dim and partner_dim")
            alignment_dim = align_dim
        self.fusion = AlignedGatedFusion(
            peptide_dim,
            partner_dim,
            alignment_dim=alignment_dim,
            gate_hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.head = build_prediction_head(
            self.fusion.output_dim,
            1,
            hidden_dims=head_hidden_dims,
            activation=head_activation,
            dropout=dropout if head_dropout is None else head_dropout,
            default_hidden_dims=[128],
        )

    def forward_with_embedding(self, peptide, partner):
        # Historical script pickles can resolve this class name while carrying
        # the pre-contract attribute layout. Keep their numerical path intact
        # and let Web inference mark them legacy_only.
        if not hasattr(self, "fusion"):
            peptide = self.peptide_projection(peptide.float())
            partner = self.partner_projection(partner.float())
            absolute = (peptide - partner).abs()
            product = peptide * partner
            weights = self.gate(torch.cat([peptide, partner, absolute, product], dim=1))
            representation = torch.cat(
                [weights * peptide + (1.0 - weights) * partner, absolute, product],
                dim=1,
            )
            return self.head(representation), representation
        representation = self.fusion(peptide, partner)
        return self.head(representation), representation

    @property
    def semantic_embedding_dim(self) -> int:
        if not hasattr(self, "fusion"):
            return self.align_dim * 3
        return self.fusion.output_dim

    @property
    def semantic_role_embedding_dims(self) -> tuple[int, int] | None:
        if not hasattr(self, "fusion"):
            return (self.align_dim, self.align_dim)
        return (self.fusion.alignment_dim, self.fusion.alignment_dim)

    def forward_with_role_embeddings(self, peptide, partner):
        if not hasattr(self, "fusion"):
            peptide = self.peptide_projection(peptide.float())
            partner = self.partner_projection(partner.float())
            absolute = (peptide - partner).abs()
            product = peptide * partner
            weights = self.gate(torch.cat([peptide, partner, absolute, product], dim=1))
            representation = torch.cat(
                [weights * peptide + (1.0 - weights) * partner, absolute, product],
                dim=1,
            )
            return self.head(representation), representation, peptide, partner
        peptide_projection, partner_projection = self.fusion.role_embeddings(peptide, partner)
        representation = self.fusion(peptide, partner)
        return (
            self.head(representation),
            representation,
            peptide_projection,
            partner_projection,
        )

    def forward(self, peptide, partner):
        logits, _ = self.forward_with_embedding(peptide, partner)
        return logits


class ResidualBlock(nn.Module):
    """Legacy unmasked CNN block retained for loading historical artifacts."""

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1), nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class MaskedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, inputs, mask):
        valid = mask.to(dtype=inputs.dtype)
        residual = self.shortcut(inputs) * valid
        hidden = F.relu(self.conv1(inputs)) * valid
        hidden = self.conv2(hidden) * valid
        return F.relu(hidden + residual) * valid


CNN_PREDICTOR_CONTRACT_VERSION = 2
GCN_PREDICTOR_CONTRACT_VERSION = 2
GAT_PREDICTOR_CONTRACT_VERSION = 2


def _cnn_blocks(model):
    if hasattr(model, "cnn_blocks"):
        return list(model.cnn_blocks)
    return [model.res_block1, model.res_block2, model.res_block3]


def _uses_masked_cnn_contract(model):
    version = getattr(model, "cnn_predictor_contract_version", None)
    if version is not None:
        return int(version) >= CNN_PREDICTOR_CONTRACT_VERSION
    return all(isinstance(block, MaskedResidualBlock) for block in _cnn_blocks(model))


def _warn_legacy_cnn(model):
    warnings.warn(
        f"{type(model).__name__} uses the legacy unmasked CNN contract; "
        "it remains available for artifact compatibility but must not be used "
        "as a padding-invariant CNN baseline without retraining",
        RuntimeWarning,
        stacklevel=3,
    )


class IndexTextCNN(nn.Module):
    def __init__(
        self,
        vocab_size=100,
        embed_dim=128,
        num_classes=15,
        dropout=0.0,
        head_hidden_dims=None,
        head_activation="relu",
        head_dropout=None,
    ):
        super().__init__()
        self.cnn_predictor_contract_version = CNN_PREDICTOR_CONTRACT_VERSION
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.res_block1 = MaskedResidualBlock(embed_dim, 64, kernel_size=3, padding=1)
        self.res_block2 = MaskedResidualBlock(embed_dim, 64, kernel_size=5, padding=2)
        self.res_block3 = MaskedResidualBlock(embed_dim, 64, kernel_size=7, padding=3)
        self.fc = build_prediction_head(
            64 * 3,
            num_classes,
            hidden_dims=head_hidden_dims,
            activation=head_activation,
            dropout=dropout if head_dropout is None else head_dropout,
            default_hidden_dims=[64],
        )

    def _legacy_forward_with_embedding(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.dropout(x)
        c1 = self.res_block1(x)
        c2 = self.res_block2(x)
        c3 = self.res_block3(x)
        p1 = torch.max(c1, dim=2)[0]
        p2 = torch.max(c2, dim=2)[0]
        p3 = torch.max(c3, dim=2)[0]
        cat = torch.cat([p1, p2, p3], dim=1)
        logits = self.fc(cat)
        return logits, cat

    def forward_with_embedding(self, x):
        if not _uses_masked_cnn_contract(self):
            _warn_legacy_cnn(self)
            return self._legacy_forward_with_embedding(x)
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError("index CNN input must have shape [B,L]")
        mask = x.ne(0)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every index CNN sample must contain at least one non-padding token")
        channel_mask = mask.unsqueeze(1)
        projected = self.dropout(self.embedding(x)).permute(0, 2, 1) * channel_mask
        c1 = self.res_block1(projected, channel_mask)
        c2 = self.res_block2(projected, channel_mask)
        c3 = self.res_block3(projected, channel_mask)
        p1 = _masked_max_pool(c1.permute(0, 2, 1), mask)
        p2 = _masked_max_pool(c2.permute(0, 2, 1), mask)
        p3 = _masked_max_pool(c3.permute(0, 2, 1), mask)
        embedding = torch.cat([p1, p2, p3], dim=1)
        return self.fc(embedding), embedding

    def forward(self, x):
        logits, _ = self.forward_with_embedding(x)
        return logits


class FeatureTextCNN(nn.Module):
    def __init__(
        self,
        in_dim,
        embed_dim=128,
        num_classes=15,
        dropout=0.0,
        head_hidden_dims=None,
        head_activation="relu",
        head_dropout=None,
    ):
        super().__init__()
        self.cnn_predictor_contract_version = CNN_PREDICTOR_CONTRACT_VERSION
        self.proj = nn.Linear(in_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.res_block1 = MaskedResidualBlock(embed_dim, 64, kernel_size=3, padding=1)
        self.res_block2 = MaskedResidualBlock(embed_dim, 64, kernel_size=5, padding=2)
        self.res_block3 = MaskedResidualBlock(embed_dim, 64, kernel_size=7, padding=3)
        self.fc = build_prediction_head(
            64 * 3,
            num_classes,
            hidden_dims=head_hidden_dims,
            activation=head_activation,
            dropout=dropout if head_dropout is None else head_dropout,
            default_hidden_dims=[64],
        )

    def _legacy_forward_with_embedding(self, x):
        features = x.features if isinstance(x, ResidueModelInput) else x
        projected = self.dropout(self.proj(features)).permute(0, 2, 1)
        c1 = self.res_block1(projected)
        c2 = self.res_block2(projected)
        c3 = self.res_block3(projected)
        embedding = torch.cat(
            [
                torch.max(c1, dim=2).values,
                torch.max(c2, dim=2).values,
                torch.max(c3, dim=2).values,
            ],
            dim=1,
        )
        return self.fc(embedding), embedding

    def forward_with_embedding(self, x):
        if not _uses_masked_cnn_contract(self):
            _warn_legacy_cnn(self)
            return self._legacy_forward_with_embedding(x)
        features, mask = _residue_features_and_mask(x)
        projected = self.dropout(self.proj(features))
        channel_mask = mask.unsqueeze(1)
        projected = projected.permute(0, 2, 1) * channel_mask
        c1 = self.res_block1(projected, channel_mask)
        c2 = self.res_block2(projected, channel_mask)
        c3 = self.res_block3(projected, channel_mask)
        p1 = _masked_max_pool(c1.permute(0, 2, 1), mask)
        p2 = _masked_max_pool(c2.permute(0, 2, 1), mask)
        p3 = _masked_max_pool(c3.permute(0, 2, 1), mask)
        cat = torch.cat([p1, p2, p3], dim=1)
        logits = self.fc(cat)
        return logits, cat

    def forward(self, x):
        logits, _ = self.forward_with_embedding(x)
        return logits


class BoWMLP(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim=256,
        num_classes=15,
        dropout=0.0,
        head_hidden_dims=None,
        head_activation="relu",
        head_dropout=None,
    ):
        super().__init__()
        self.input_dim = int(in_dim)
        default_hidden_dims = (
            [int(hidden_dim), max(1, int(hidden_dim) // 2)] if head_hidden_dims is None else None
        )
        self.net = build_prediction_head(
            in_dim,
            num_classes,
            hidden_dims=head_hidden_dims,
            activation=head_activation,
            dropout=dropout if head_dropout is None else head_dropout,
            default_hidden_dims=default_hidden_dims,
        )

    def forward_with_embedding(self, x):
        input_dim = getattr(self, "input_dim", None) or _prediction_head_input_dim(self.net)
        if input_dim is None:
            raise RuntimeError("prediction-head input dimension is missing")
        _validate_prediction_head_input(x, input_dim)
        if isinstance(self.net, nn.Linear):
            return self.net(x), x
        embedding = x
        layers = list(self.net.children())
        for layer in layers[:-1]:
            embedding = layer(embedding)
        logits = layers[-1](embedding)
        _validate_prediction_head_output(logits)
        return logits, embedding

    def forward(self, x):
        logits, _ = self.forward_with_embedding(x)
        return logits


class IndexRNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=128,
        num_classes=15,
        dropout=0.0,
        head_hidden_dims=None,
        head_activation="relu",
        head_dropout=None,
    ):
        super().__init__()
        self.rnn_predictor_contract_version = RNN_PREDICTOR_CONTRACT_VERSION
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.RNN(
            embed_dim,
            hidden_dim,
            nonlinearity="tanh",
            batch_first=True,
            bidirectional=True,
        )
        self.fc = build_prediction_head(
            hidden_dim * 2,
            num_classes,
            hidden_dims=head_hidden_dims,
            activation=head_activation,
            dropout=dropout if head_dropout is None else head_dropout,
            default_hidden_dims=[64],
        )

    def _legacy_forward_with_embedding(self, x):
        embedded = self.dropout(self.embedding(x))
        _, h = self.rnn(embedded)
        h = torch.cat([h[-2], h[-1]], dim=1)
        logits = self.fc(h)
        return logits, h

    def forward_with_embedding(self, x):
        if not _uses_verified_rnn_contract(self):
            _warn_legacy_rnn(self)
            return self._legacy_forward_with_embedding(x)
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError("index RNN input must have shape [B,L]")
        if x.dtype not in {torch.int32, torch.int64}:
            raise TypeError("index RNN input must use int32 or int64 token IDs")
        if x.shape[1] == 0:
            raise ValueError("index RNN input sequence length must be positive")
        if bool((x < 0).any()) or bool((x >= self.embedding.num_embeddings).any()):
            raise ValueError(f"index RNN token IDs must be in [0, {self.embedding.num_embeddings})")
        mask = x.ne(0)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every index RNN sample must contain at least one non-padding token")
        embedded = self.dropout(self.embedding(x))
        _, h = _run_feature_recurrent(
            self.rnn,
            ResidueModelInput(embedded, mask),
        )
        h = torch.cat([h[-2], h[-1]], dim=1)
        logits = self.fc(h)
        return logits, h

    def forward(self, x):
        logits, _ = self.forward_with_embedding(x)
        return logits


class FeatureRNN(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim=128,
        num_classes=15,
        dropout=0.0,
        head_hidden_dims=None,
        head_activation="relu",
        head_dropout=None,
    ):
        super().__init__()
        self.rnn_predictor_contract_version = RNN_PREDICTOR_CONTRACT_VERSION
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.RNN(
            in_dim,
            hidden_dim,
            nonlinearity="tanh",
            batch_first=True,
            bidirectional=True,
        )
        self.fc = build_prediction_head(
            hidden_dim * 2,
            num_classes,
            hidden_dims=head_hidden_dims,
            activation=head_activation,
            dropout=dropout if head_dropout is None else head_dropout,
            default_hidden_dims=[64],
        )

    def forward_with_embedding(self, x):
        if not _uses_verified_rnn_contract(self):
            _warn_legacy_rnn(self)
            if type(self.rnn) is nn.GRU:
                features = x.features if isinstance(x, ResidueModelInput) else x
                _, h = self.rnn(self.dropout(features))
                h = torch.cat([h[-2], h[-1]], dim=1)
                return self.fc(h), h
        features, mask = _residue_features_and_mask(x)
        _, h = _run_feature_recurrent(
            self.rnn,
            ResidueModelInput(self.dropout(features), mask),
        )
        h = torch.cat([h[-2], h[-1]], dim=1)
        logits = self.fc(h)
        return logits, h

    def forward(self, x):
        logits, _ = self.forward_with_embedding(x)
        return logits


class IndexGRU(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=128,
        num_classes=15,
        dropout=0.0,
        head_hidden_dims=None,
        head_activation="relu",
        head_dropout=None,
    ):
        super().__init__()
        self.gru_predictor_contract_version = GRU_PREDICTOR_CONTRACT_VERSION
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = build_prediction_head(
            hidden_dim * 2,
            num_classes,
            hidden_dims=head_hidden_dims,
            activation=head_activation,
            dropout=dropout if head_dropout is None else head_dropout,
            default_hidden_dims=[64],
        )

    def _legacy_forward_with_embedding(self, x):
        embedded = self.dropout(self.embedding(x))
        _, h = self.rnn(embedded)
        h = torch.cat([h[-2], h[-1]], dim=1)
        logits = self.fc(h)
        return logits, h

    def forward_with_embedding(self, x):
        if not _uses_verified_gru_contract(self):
            _warn_legacy_gru(self)
            return self._legacy_forward_with_embedding(x)
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError("index GRU input must have shape [B,L]")
        if x.dtype not in {torch.int32, torch.int64}:
            raise TypeError("index GRU input must use int32 or int64 token IDs")
        if x.shape[1] == 0:
            raise ValueError("index GRU input sequence length must be positive")
        if bool((x < 0).any()) or bool((x >= self.embedding.num_embeddings).any()):
            raise ValueError(f"index GRU token IDs must be in [0, {self.embedding.num_embeddings})")
        mask = x.ne(0)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every index GRU sample must contain at least one non-padding token")
        embedded = self.dropout(self.embedding(x))
        _, h = _run_feature_recurrent(
            self.rnn,
            ResidueModelInput(embedded, mask),
        )
        h = torch.cat([h[-2], h[-1]], dim=1)
        logits = self.fc(h)
        return logits, h

    def forward(self, x):
        logits, _ = self.forward_with_embedding(x)
        return logits


class FeatureGRU(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim=128,
        num_classes=15,
        dropout=0.0,
        head_hidden_dims=None,
        head_activation="relu",
        head_dropout=None,
    ):
        super().__init__()
        self.gru_predictor_contract_version = GRU_PREDICTOR_CONTRACT_VERSION
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(in_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = build_prediction_head(
            hidden_dim * 2,
            num_classes,
            hidden_dims=head_hidden_dims,
            activation=head_activation,
            dropout=dropout if head_dropout is None else head_dropout,
            default_hidden_dims=[64],
        )

    def forward_with_embedding(self, x):
        if not _uses_verified_gru_contract(self):
            _warn_legacy_gru(self)
            features = x.features if isinstance(x, ResidueModelInput) else x
            _, h = self.rnn(self.dropout(features))
            h = torch.cat([h[-2], h[-1]], dim=1)
            return self.fc(h), h
        features, mask = _residue_features_and_mask(x)
        _, h = _run_feature_recurrent(
            self.rnn,
            ResidueModelInput(self.dropout(features), mask),
        )
        h = torch.cat([h[-2], h[-1]], dim=1)
        logits = self.fc(h)
        return logits, h

    def forward(self, x):
        logits, _ = self.forward_with_embedding(x)
        return logits


class IndexLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=128,
        num_classes=15,
        dropout=0.0,
        head_hidden_dims=None,
        head_activation="relu",
        head_dropout=None,
    ):
        super().__init__()
        self.lstm_predictor_contract_version = LSTM_PREDICTOR_CONTRACT_VERSION
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = build_prediction_head(
            hidden_dim * 2,
            num_classes,
            hidden_dims=head_hidden_dims,
            activation=head_activation,
            dropout=dropout if head_dropout is None else head_dropout,
            default_hidden_dims=[64],
        )

    def _legacy_forward_with_embedding(self, x):
        embedded = self.dropout(self.embedding(x))
        _, (h, _) = self.lstm(embedded)
        h = torch.cat([h[-2], h[-1]], dim=1)
        logits = self.fc(h)
        return logits, h

    def forward_with_embedding(self, x):
        if not _uses_verified_lstm_contract(self):
            _warn_legacy_lstm(self)
            return self._legacy_forward_with_embedding(x)
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError("index LSTM input must have shape [B,L]")
        if x.dtype not in {torch.int32, torch.int64}:
            raise TypeError("index LSTM input must use int32 or int64 token IDs")
        if x.shape[1] == 0:
            raise ValueError("index LSTM input sequence length must be positive")
        if bool((x < 0).any()) or bool((x >= self.embedding.num_embeddings).any()):
            raise ValueError(
                f"index LSTM token IDs must be in [0, {self.embedding.num_embeddings})"
            )
        mask = x.ne(0)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every index LSTM sample must contain at least one non-padding token")
        embedded = self.dropout(self.embedding(x))
        _, (h, _) = _run_feature_recurrent(
            self.lstm,
            ResidueModelInput(embedded, mask),
        )
        h = torch.cat([h[-2], h[-1]], dim=1)
        logits = self.fc(h)
        return logits, h

    def forward(self, x):
        logits, _ = self.forward_with_embedding(x)
        return logits


class FeatureLSTM(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim=128,
        num_classes=15,
        dropout=0.0,
        head_hidden_dims=None,
        head_activation="relu",
        head_dropout=None,
    ):
        super().__init__()
        self.lstm_predictor_contract_version = LSTM_PREDICTOR_CONTRACT_VERSION
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(in_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = build_prediction_head(
            hidden_dim * 2,
            num_classes,
            hidden_dims=head_hidden_dims,
            activation=head_activation,
            dropout=dropout if head_dropout is None else head_dropout,
            default_hidden_dims=[64],
        )

    def forward_with_embedding(self, x):
        if not _uses_verified_lstm_contract(self):
            _warn_legacy_lstm(self)
        features, mask = _residue_features_and_mask(x)
        _, (h, _) = _run_feature_recurrent(
            self.lstm,
            ResidueModelInput(self.dropout(features), mask),
        )
        h = torch.cat([h[-2], h[-1]], dim=1)
        logits = self.fc(h)
        return logits, h

    def forward(self, x):
        logits, _ = self.forward_with_embedding(x)
        return logits


class IndexTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_length,
        embed_dim=128,
        num_layers=2,
        num_heads=4,
        ff_dim=256,
        num_classes=15,
        dropout=0.0,
        head_hidden_dims=None,
        head_activation="relu",
        head_dropout=None,
    ):
        super().__init__()
        self.transformer_predictor_contract_version = TRANSFORMER_PREDICTOR_CONTRACT_VERSION
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position = nn.Embedding(max_length, embed_dim)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = build_prediction_head(
            embed_dim,
            num_classes,
            hidden_dims=head_hidden_dims,
            activation=head_activation,
            dropout=dropout if head_dropout is None else head_dropout,
            default_hidden_dims=[64],
        )

    def _legacy_forward_with_embedding(self, x):
        mask = x == 0
        pos_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        x = self.embedding(x) + self.position(pos_ids)
        x = self.dropout(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        keep = (~mask).unsqueeze(-1).float()
        pooled = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        logits = self.fc(pooled)
        return logits, pooled

    def forward_with_embedding(self, x):
        if not _uses_verified_transformer_contract(self):
            _warn_legacy_transformer(self)
            return self._legacy_forward_with_embedding(x)
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError("index Transformer input must have shape [B,L]")
        if x.dtype not in {torch.int32, torch.int64}:
            raise TypeError("index Transformer input must use int32 or int64 token IDs")
        if x.shape[1] == 0:
            raise ValueError("index Transformer input sequence length must be positive")
        if bool((x < 0).any()) or bool((x >= self.embedding.num_embeddings).any()):
            raise ValueError(
                f"index Transformer token IDs must be in [0, {self.embedding.num_embeddings})"
            )
        valid_mask = x.ne(0)
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError(
                "every index Transformer sample must contain at least one non-padding token"
            )
        pos_ids = _mask_compact_position_ids(
            valid_mask,
            self.position.num_embeddings,
        )
        encoded = self.embedding(x) + self.position(pos_ids)
        encoded = self.dropout(encoded)
        encoded = self.encoder(encoded, src_key_padding_mask=~valid_mask)
        keep = valid_mask.unsqueeze(-1).to(dtype=encoded.dtype)
        pooled = (encoded * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        return self.fc(pooled), pooled

    def forward(self, x):
        logits, _ = self.forward_with_embedding(x)
        return logits


class FeatureTransformer(nn.Module):
    def __init__(
        self,
        in_dim,
        max_length,
        embed_dim=128,
        num_layers=2,
        num_heads=4,
        ff_dim=256,
        num_classes=15,
        dropout=0.0,
        head_hidden_dims=None,
        head_activation="relu",
        head_dropout=None,
    ):
        super().__init__()
        self.transformer_predictor_contract_version = TRANSFORMER_PREDICTOR_CONTRACT_VERSION
        self.proj = nn.Linear(in_dim, embed_dim)
        self.position = nn.Embedding(max_length, embed_dim)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = build_prediction_head(
            embed_dim,
            num_classes,
            hidden_dims=head_hidden_dims,
            activation=head_activation,
            dropout=dropout if head_dropout is None else head_dropout,
            default_hidden_dims=[64],
        )

    def _legacy_forward_with_embedding(self, x):
        features, valid_mask = _residue_features_and_mask(x)
        pos_ids = (
            torch.arange(features.size(1), device=features.device)
            .unsqueeze(0)
            .expand(features.size(0), features.size(1))
        )
        x = self.proj(features) + self.position(pos_ids)
        x = self.dropout(x)
        x = self.encoder(x, src_key_padding_mask=~valid_mask)
        keep = valid_mask.unsqueeze(-1).float()
        pooled = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        logits = self.fc(pooled)
        return logits, pooled

    def forward_with_embedding(self, x):
        if not _uses_verified_transformer_contract(self):
            _warn_legacy_transformer(self)
            return self._legacy_forward_with_embedding(x)
        features, valid_mask = _residue_features_and_mask(x)
        if not features.is_floating_point():
            raise TypeError("feature Transformer input must use floating-point values")
        features = features.to(dtype=self.proj.weight.dtype)
        pos_ids = _mask_compact_position_ids(
            valid_mask,
            self.position.num_embeddings,
        )
        encoded = self.proj(features) + self.position(pos_ids)
        encoded = self.dropout(encoded)
        encoded = self.encoder(encoded, src_key_padding_mask=~valid_mask)
        keep = valid_mask.unsqueeze(-1).to(dtype=encoded.dtype)
        pooled = (encoded * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        return self.fc(pooled), pooled

    def forward(self, x):
        logits, _ = self.forward_with_embedding(x)
        return logits


def residue_mask(x):
    """Return the valid-residue mask for padded index or feature tensors."""
    if x.ndim == 2:
        return x != 0
    if x.ndim == 3:
        return x.abs().sum(dim=-1) != 0
    raise ValueError(f"residue graph input must be [B,L] or [B,L,D], got {tuple(x.shape)}")


def _chain_neighbors(x, mask):
    """Collect previous, self, and next residue states with matching masks."""
    zeros = torch.zeros_like(x[:, :1])
    prev = torch.cat([zeros, x[:, :-1]], dim=1)
    nxt = torch.cat([x[:, 1:], zeros], dim=1)
    false = torch.zeros_like(mask[:, :1])
    prev_mask = torch.cat([false, mask[:, :-1]], dim=1)
    next_mask = torch.cat([mask[:, 1:], false], dim=1)
    return torch.stack([prev, x, nxt], dim=2), torch.stack([prev_mask, mask, next_mask], dim=2)


class ResidueGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x, mask):
        neighbors, neighbor_mask = _chain_neighbors(x, mask)
        weights = neighbor_mask.unsqueeze(-1).to(dtype=x.dtype)
        aggregated = (neighbors * weights).sum(dim=2) / weights.sum(dim=2).clamp(min=1.0)
        return self.proj(aggregated) * mask.unsqueeze(-1).to(dtype=x.dtype)


class ResidueGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        if int(heads) <= 0:
            raise ValueError("GAT heads must be positive")
        if out_dim % heads != 0:
            raise ValueError("GAT out_dim must be divisible by heads")
        self.heads = int(heads)
        self.head_dim = out_dim // heads
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(self.heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(self.heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x, mask):
        h = self.proj(x).view(x.shape[0], x.shape[1], self.heads, self.head_dim)
        neighbors, neighbor_mask = _chain_neighbors(h, mask)
        center = h.unsqueeze(2)
        scores = (center * self.attn_dst).sum(-1) + (neighbors * self.attn_src).sum(-1)
        scores = F.leaky_relu(scores, negative_slope=0.2)
        scores = scores.masked_fill(~neighbor_mask.unsqueeze(-1), torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=2).unsqueeze(-1)
        output = (attention * neighbors).sum(dim=2).reshape(x.shape[0], x.shape[1], -1)
        output = output + self.bias
        return output * mask.unsqueeze(-1).to(dtype=output.dtype)


class ResidueGraphClassifier(nn.Module):
    """Two-layer graph classifier over a padded residue chain."""

    def __init__(
        self,
        in_dim,
        num_classes=15,
        hidden_dim=128,
        graph_kind="gcn",
        heads=4,
        dropout=0.2,
        head_hidden_dims=None,
        head_activation="relu",
        head_dropout=None,
    ):
        super().__init__()
        graph_kind = str(graph_kind).strip().lower()
        if graph_kind not in {"gcn", "gat"}:
            raise ValueError(f"unsupported residue graph kind: {graph_kind}")
        layer = ResidueGCNLayer if graph_kind == "gcn" else ResidueGATLayer
        kwargs = {} if graph_kind == "gcn" else {"heads": heads}
        self.graph_kind = graph_kind
        if graph_kind == "gcn":
            self.gcn_predictor_contract_version = GCN_PREDICTOR_CONTRACT_VERSION
        else:
            self.gat_predictor_contract_version = GAT_PREDICTOR_CONTRACT_VERSION
        self.layer1 = layer(in_dim, hidden_dim, **kwargs)
        self.layer2 = layer(hidden_dim, hidden_dim, **kwargs)
        self.dropout = nn.Dropout(dropout)
        self.classifier = build_prediction_head(
            hidden_dim * 2,
            num_classes,
            hidden_dims=head_hidden_dims,
            activation=head_activation,
            dropout=dropout if head_dropout is None else head_dropout,
            default_hidden_dims=[hidden_dim],
        )

    def forward_with_embedding(self, x):
        features, mask = _residue_features_and_mask(x)
        contract_version = int(
            getattr(
                self,
                f"{self.graph_kind}_predictor_contract_version",
                0,
            )
            or 0
        )
        required_version = (
            GCN_PREDICTOR_CONTRACT_VERSION
            if self.graph_kind == "gcn"
            else GAT_PREDICTOR_CONTRACT_VERSION
        )
        if contract_version >= required_version and not features.is_floating_point():
            raise TypeError(
                f"{self.graph_kind.upper()} node features must use floating-point values"
            )
        h = features.float()
        h = self.dropout(F.relu(self.layer1(h, mask)))
        h = self.dropout(F.relu(self.layer2(h, mask)))
        valid = mask.unsqueeze(-1)
        mean_pool = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        max_pool = h.masked_fill(~valid, torch.finfo(h.dtype).min).max(dim=1).values
        max_pool = torch.where(mask.any(dim=1, keepdim=True), max_pool, torch.zeros_like(max_pool))
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        return self.classifier(pooled), pooled

    def forward(self, x):
        logits, _ = self.forward_with_embedding(x)
        return logits


class PretrainedResiduePredictor(nn.Module):
    """Mask-aware predictor over frozen pretrained residue embeddings."""

    def __init__(
        self,
        architecture,
        input_dim,
        adapter_dim=128,
        max_length=60,
        num_classes=15,
        dropout=0.0,
        hidden_dim=128,
        heads=4,
        head_hidden_dims=None,
        head_activation="relu",
        head_dropout=None,
    ):
        super().__init__()
        architecture = str(architecture).strip().lower()
        if architecture not in {"cnn", "rnn", "gru", "lstm", "transformer", "gcn", "gat"}:
            raise ValueError(f"unsupported pretrained residue architecture: {architecture}")
        if int(adapter_dim) <= 0 or int(adapter_dim) % int(heads) != 0:
            raise ValueError(f"adapter_dim must be positive and divisible by heads={heads}")
        self.architecture = architecture
        self.input_dim = int(input_dim)
        self.adapter_dim = int(adapter_dim)
        self.max_length = int(max_length)
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.adapter_norm = nn.LayerNorm(self.input_dim)
        self.adapter_projection = nn.Linear(self.input_dim, self.adapter_dim)
        self.dropout = nn.Dropout(float(dropout))

        if architecture == "cnn":
            self.cnn_predictor_contract_version = CNN_PREDICTOR_CONTRACT_VERSION
            self.cnn_blocks = nn.ModuleList(
                [
                    MaskedResidualBlock(self.adapter_dim, 64, kernel_size=3, padding=1),
                    MaskedResidualBlock(self.adapter_dim, 64, kernel_size=5, padding=2),
                    MaskedResidualBlock(self.adapter_dim, 64, kernel_size=7, padding=3),
                ]
            )
            readout_dim = 64 * 3
        elif architecture in {"rnn", "gru", "lstm"}:
            recurrent_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[architecture]
            recurrent_kwargs = {
                "input_size": self.adapter_dim,
                "hidden_size": self.hidden_dim,
                "batch_first": True,
                "bidirectional": True,
            }
            if architecture == "rnn":
                recurrent_kwargs["nonlinearity"] = "tanh"
                self.rnn_predictor_contract_version = RNN_PREDICTOR_CONTRACT_VERSION
            elif architecture == "gru":
                self.gru_predictor_contract_version = GRU_PREDICTOR_CONTRACT_VERSION
            elif architecture == "lstm":
                self.lstm_predictor_contract_version = LSTM_PREDICTOR_CONTRACT_VERSION
            self.recurrent = recurrent_cls(**recurrent_kwargs)
            readout_dim = self.hidden_dim * 2
        elif architecture == "transformer":
            self.transformer_predictor_contract_version = TRANSFORMER_PREDICTOR_CONTRACT_VERSION
            self.position = nn.Embedding(self.max_length, self.adapter_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.adapter_dim,
                nhead=self.heads,
                dim_feedforward=self.adapter_dim * 2,
                dropout=float(dropout),
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            readout_dim = self.adapter_dim
        else:
            if architecture == "gcn":
                self.gcn_predictor_contract_version = GCN_PREDICTOR_CONTRACT_VERSION
            else:
                self.gat_predictor_contract_version = GAT_PREDICTOR_CONTRACT_VERSION
            layer_cls = ResidueGCNLayer if architecture == "gcn" else ResidueGATLayer
            layer_kwargs = {} if architecture == "gcn" else {"heads": self.heads}
            self.graph_layer1 = layer_cls(self.adapter_dim, self.hidden_dim, **layer_kwargs)
            self.graph_layer2 = layer_cls(self.hidden_dim, self.hidden_dim, **layer_kwargs)
            readout_dim = self.hidden_dim * 2

        default_head = [self.hidden_dim] if architecture in {"gcn", "gat"} else [64]
        self.head = build_prediction_head(
            readout_dim,
            num_classes,
            hidden_dims=head_hidden_dims,
            activation=head_activation,
            dropout=float(dropout) if head_dropout is None else float(head_dropout),
            default_hidden_dims=default_head,
        )

    @staticmethod
    def _validate_input(inputs):
        if not isinstance(inputs, ResidueModelInput):
            if isinstance(inputs, (tuple, list)) and len(inputs) == 2:
                inputs = ResidueModelInput(inputs[0], inputs[1])
            else:
                raise TypeError(
                    "pretrained residue predictor requires ResidueModelInput(features, mask)"
                )
        features = inputs.features
        mask = inputs.mask.bool()
        if features.ndim != 3 or mask.shape != features.shape[:2]:
            raise ValueError(
                f"invalid residue input shapes: features={tuple(features.shape)}, mask={tuple(mask.shape)}"
            )
        if not bool(mask.any(dim=1).all()):
            raise ValueError("pretrained residue predictor does not accept empty residue sequences")
        return features.float(), mask

    def _adapt(self, features, mask):
        adapted = self.adapter_projection(self.adapter_norm(features))
        adapted = self.dropout(adapted)
        return adapted * mask.unsqueeze(-1).to(dtype=adapted.dtype)

    @staticmethod
    def _masked_max(values, mask, dim):
        masked = values.masked_fill(~mask, torch.finfo(values.dtype).min)
        return masked.max(dim=dim).values

    def _readout(self, adapted, mask):
        if self.architecture == "cnn":
            channel_first = adapted.permute(0, 2, 1)
            valid = mask.unsqueeze(1)
            pooled = [
                self._masked_max(block(channel_first, valid), valid, dim=2)
                for block in self.cnn_blocks
            ]
            return torch.cat(pooled, dim=1)
        if self.architecture in {"rnn", "gru", "lstm"}:
            if self.architecture == "rnn" and _uses_verified_rnn_contract(self):
                recurrent_output = _run_feature_recurrent(
                    self.recurrent,
                    ResidueModelInput(adapted, mask),
                )
                hidden = recurrent_output[1]
                return torch.cat([hidden[-2], hidden[-1]], dim=1)
            if self.architecture == "gru" and _uses_verified_gru_contract(self):
                recurrent_output = _run_feature_recurrent(
                    self.recurrent,
                    ResidueModelInput(adapted, mask),
                )
                hidden = recurrent_output[1]
                return torch.cat([hidden[-2], hidden[-1]], dim=1)
            if self.architecture == "lstm" and _uses_verified_lstm_contract(self):
                recurrent_output = _run_feature_recurrent(
                    self.recurrent,
                    ResidueModelInput(adapted, mask),
                )
                hidden = recurrent_output[1][0]
                return torch.cat([hidden[-2], hidden[-1]], dim=1)
            if self.architecture == "rnn":
                _warn_legacy_rnn(self)
            elif self.architecture == "gru":
                _warn_legacy_gru(self)
            elif self.architecture == "lstm":
                _warn_legacy_lstm(self)
            lengths = mask.sum(dim=1).to(dtype=torch.long).cpu()
            packed = pack_padded_sequence(adapted, lengths, batch_first=True, enforce_sorted=False)
            recurrent_output = self.recurrent(packed)
            state = recurrent_output[1]
            hidden = state[0] if self.architecture == "lstm" else state
            return torch.cat([hidden[-2], hidden[-1]], dim=1)
        if self.architecture == "transformer":
            if _uses_verified_transformer_contract(self):
                positions = _mask_compact_position_ids(
                    mask,
                    self.position.num_embeddings,
                )
            else:
                _warn_legacy_transformer(self)
                positions = torch.arange(
                    adapted.shape[1],
                    device=adapted.device,
                ).unsqueeze(0)
            encoded = self.transformer(
                adapted + self.position(positions),
                src_key_padding_mask=~mask,
            )
            valid = mask.unsqueeze(-1).to(dtype=encoded.dtype)
            return (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        hidden = self.dropout(F.relu(self.graph_layer1(adapted, mask)))
        hidden = self.dropout(F.relu(self.graph_layer2(hidden, mask)))
        valid = mask.unsqueeze(-1)
        mean_pool = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        max_pool = self._masked_max(hidden, valid, dim=1)
        return torch.cat([mean_pool, max_pool], dim=1)

    def forward_with_embedding(self, inputs):
        raw_features = None
        if isinstance(inputs, ResidueModelInput):
            raw_features = inputs.features
        elif isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            raw_features = inputs[0]
        if self.architecture in {"gcn", "gat"}:
            contract_version = int(
                getattr(
                    self,
                    f"{self.architecture}_predictor_contract_version",
                    0,
                )
                or 0
            )
            required_version = (
                GCN_PREDICTOR_CONTRACT_VERSION
                if self.architecture == "gcn"
                else GAT_PREDICTOR_CONTRACT_VERSION
            )
            if (
                contract_version >= required_version
                and isinstance(raw_features, torch.Tensor)
                and not raw_features.is_floating_point()
            ):
                raise TypeError(
                    f"{self.architecture.upper()} residue features must use floating-point values"
                )
        features, mask = self._validate_input(inputs)
        embedding = self._readout(self._adapt(features, mask), mask)
        return self.head(embedding), embedding

    def forward(self, inputs):
        logits, _ = self.forward_with_embedding(inputs)
        return logits


def cnn_predictor_contract(model):
    """Return auditable CNN assembly metadata, or None for non-CNN models."""

    if isinstance(model, PretrainedResiduePredictor):
        if model.architecture != "cnn":
            return None
        input_adapter = "layer_norm_linear_projection"
        mask_source = "explicit_residue_mask"
    elif isinstance(model, IndexTextCNN):
        input_adapter = "learned_embedding"
        mask_source = "token_id_nonzero"
    elif isinstance(model, FeatureTextCNN):
        input_adapter = "linear_projection"
        mask_source = "explicit_or_feature_compatibility_mask"
    else:
        return None
    masked = _uses_masked_cnn_contract(model)
    return {
        "schema_version": 1,
        "implementation": "pepplatform_multiscale_residual_conv1d",
        "contract_version": CNN_PREDICTOR_CONTRACT_VERSION if masked else 1,
        "status": "verified_platform_variant" if masked else "legacy_only",
        "input_adapter": input_adapter,
        "kernel_sizes": [3, 5, 7],
        "readout": "masked_global_max" if masked else "unmasked_global_max",
        "padding_mask": mask_source if masked else "none",
        "prediction_output": "logits",
    }


def rnn_predictor_contract(model):
    """Return auditable RNN assembly metadata, or None for non-RNN models."""

    if isinstance(model, PretrainedResiduePredictor):
        if model.architecture != "rnn":
            return None
        input_adapter = "layer_norm_linear_projection"
        mask_source = "explicit_residue_mask"
        legacy_sequence_handling = "pack_padded_sequence_unversioned"
    elif isinstance(model, IndexRNN):
        input_adapter = "learned_embedding"
        mask_source = "token_id_nonzero"
        legacy_sequence_handling = "raw_padded_sequence"
    elif isinstance(model, FeatureRNN):
        input_adapter = "identity"
        mask_source = "explicit_position_mask"
        legacy_sequence_handling = "runtime_mask_compact_pack_unversioned"
    else:
        return None

    cell = _rnn_cell(model)
    verified = _uses_verified_rnn_contract(model)
    mislabeled_gru = type(cell) is nn.GRU
    version = _rnn_contract_version(model)
    if verified:
        implementation = "pepplatform_bidirectional_elman_rnn"
        sequence_handling = "mask_compact_pack_padded_sequence"
        status = "verified_platform_variant"
        contract_version = RNN_PREDICTOR_CONTRACT_VERSION
    elif mislabeled_gru:
        implementation = "legacy_mislabeled_bidirectional_gru"
        sequence_handling = "raw_padded_sequence"
        status = "legacy_only"
        contract_version = 1
    else:
        implementation = "pepplatform_unversioned_bidirectional_elman_rnn"
        sequence_handling = legacy_sequence_handling
        status = "legacy_only"
        contract_version = version or 1
    return {
        "schema_version": 1,
        "implementation": implementation,
        "contract_version": contract_version,
        "status": status,
        "input_adapter": input_adapter,
        "actual_cell": type(cell).__name__ if cell is not None else "missing",
        "nonlinearity": getattr(cell, "nonlinearity", None),
        "bidirectional": bool(getattr(cell, "bidirectional", False)),
        "sequence_handling": sequence_handling,
        "padding_mask": mask_source if verified else "legacy_route_dependent",
        "readout": "concatenated_final_forward_backward_hidden",
        "prediction_output": "logits",
    }


def gru_predictor_contract(model):
    """Return auditable GRU assembly metadata, or None for non-GRU models."""

    if isinstance(model, PretrainedResiduePredictor):
        if model.architecture != "gru":
            return None
        input_adapter = "layer_norm_linear_projection"
        mask_source = "explicit_residue_mask"
        legacy_sequence_handling = "pack_padded_sequence_unversioned"
    elif isinstance(model, IndexGRU):
        input_adapter = "learned_embedding"
        mask_source = "token_id_nonzero"
        legacy_sequence_handling = "raw_padded_sequence"
    elif isinstance(model, FeatureGRU):
        input_adapter = "identity"
        mask_source = "explicit_position_mask"
        legacy_sequence_handling = "raw_padded_sequence"
    else:
        return None

    cell = _gru_cell(model)
    verified = _uses_verified_gru_contract(model)
    version = _gru_contract_version(model)
    if verified:
        implementation = "pepplatform_bidirectional_pytorch_gru"
        sequence_handling = "mask_compact_pack_padded_sequence"
        status = "verified_platform_variant"
        contract_version = GRU_PREDICTOR_CONTRACT_VERSION
    else:
        implementation = "pepplatform_unversioned_bidirectional_pytorch_gru"
        sequence_handling = legacy_sequence_handling
        status = "legacy_only"
        contract_version = version or 1
    return {
        "schema_version": 1,
        "implementation": implementation,
        "contract_version": contract_version,
        "status": status,
        "input_adapter": input_adapter,
        "actual_cell": type(cell).__name__ if cell is not None else "missing",
        "reset_gate_placement": ("after_hidden_affine" if type(cell) is nn.GRU else None),
        "bidirectional": bool(getattr(cell, "bidirectional", False)),
        "sequence_handling": sequence_handling,
        "padding_mask": mask_source if verified else "legacy_route_dependent",
        "readout": "concatenated_final_forward_backward_hidden",
        "prediction_output": "logits",
    }


def lstm_predictor_contract(model):
    """Return auditable LSTM assembly metadata, or None for non-LSTM models."""

    if isinstance(model, PretrainedResiduePredictor):
        if model.architecture != "lstm":
            return None
        input_adapter = "layer_norm_linear_projection"
        mask_source = "explicit_residue_mask"
        legacy_sequence_handling = "pack_padded_sequence_unversioned"
    elif isinstance(model, IndexLSTM):
        input_adapter = "learned_embedding"
        mask_source = "token_id_nonzero"
        legacy_sequence_handling = "raw_padded_sequence"
    elif isinstance(model, FeatureLSTM):
        input_adapter = "identity"
        mask_source = "explicit_position_mask"
        legacy_sequence_handling = "runtime_mask_compact_pack_unversioned"
    else:
        return None

    cell = _lstm_cell(model)
    verified = _uses_verified_lstm_contract(model)
    version = _lstm_contract_version(model)
    if verified:
        implementation = "pepplatform_bidirectional_pytorch_lstm"
        sequence_handling = "mask_compact_pack_padded_sequence"
        status = "verified_platform_variant"
        contract_version = LSTM_PREDICTOR_CONTRACT_VERSION
    else:
        implementation = "pepplatform_unversioned_bidirectional_pytorch_lstm"
        sequence_handling = legacy_sequence_handling
        status = "legacy_only"
        contract_version = version or 1
    is_lstm = type(cell) is nn.LSTM
    return {
        "schema_version": 1,
        "implementation": implementation,
        "contract_version": contract_version,
        "status": status,
        "input_adapter": input_adapter,
        "actual_cell": type(cell).__name__ if cell is not None else "missing",
        "gate_order": ["input", "forget", "cell", "output"] if is_lstm else None,
        "state_components": ["hidden", "cell"] if is_lstm else None,
        "cell_state_update": ("forget_retained_cell_plus_input_candidate" if is_lstm else None),
        "bidirectional": bool(getattr(cell, "bidirectional", False)),
        "sequence_handling": sequence_handling,
        "padding_mask": mask_source if verified else "legacy_route_dependent",
        "readout": "concatenated_final_forward_backward_hidden",
        "prediction_output": "logits",
    }


def transformer_predictor_contract(model):
    """Return auditable Transformer assembly metadata, or None otherwise."""

    if isinstance(model, PretrainedResiduePredictor):
        if model.architecture != "transformer":
            return None
        input_adapter = "layer_norm_linear_projection"
        mask_source = "explicit_residue_mask"
    elif isinstance(model, IndexTransformer):
        input_adapter = "learned_embedding"
        mask_source = "token_id_nonzero"
    elif isinstance(model, FeatureTransformer):
        input_adapter = "linear_projection"
        mask_source = "explicit_position_mask"
    else:
        return None

    encoder = _transformer_encoder(model)
    layers = list(getattr(encoder, "layers", []))
    first_layer = layers[0] if layers else None
    self_attn = getattr(first_layer, "self_attn", None)
    activation = getattr(first_layer, "activation", None)
    verified = _uses_verified_transformer_contract(model)
    version = _transformer_contract_version(model)
    return {
        "schema_version": 1,
        "implementation": (
            "pepplatform_pytorch_transformer_encoder"
            if verified
            else "pepplatform_unversioned_pytorch_transformer_encoder"
        ),
        "contract_version": (TRANSFORMER_PREDICTOR_CONTRACT_VERSION if verified else version or 1),
        "status": "verified_platform_variant" if verified else "legacy_only",
        "input_adapter": input_adapter,
        "actual_encoder": (type(encoder).__name__ if encoder is not None else "missing"),
        "encoder_layers": len(layers),
        "attention": "multi_head_scaled_dot_product_self_attention",
        "model_dim": getattr(self_attn, "embed_dim", None),
        "num_heads": getattr(self_attn, "num_heads", None),
        "feedforward_dim": getattr(
            getattr(first_layer, "linear1", None),
            "out_features",
            None,
        ),
        "activation": getattr(activation, "__name__", None),
        "norm_first": getattr(first_layer, "norm_first", None),
        "position_encoding": "learned_absolute_embedding",
        "position_indexing": ("valid_token_rank" if verified else "raw_padded_slot"),
        "sequence_handling": (
            "key_padding_mask_with_mask_compact_positions"
            if verified
            else "key_padding_mask_with_raw_positions"
        ),
        "padding_mask": mask_source if verified else "legacy_route_dependent",
        "readout": "masked_mean",
        "prediction_output": "logits",
    }


def gcn_predictor_contract(model):
    """Return auditable GCN assembly metadata, or None for non-GCN models."""

    if isinstance(model, PretrainedResiduePredictor):
        if model.architecture != "gcn":
            return None
        input_adapter = "layer_norm_linear_projection"
        mask_source = "explicit_residue_mask"
        layer = getattr(model, "graph_layer1", None)
        hidden_dim = getattr(getattr(layer, "proj", None), "out_features", None)
    elif isinstance(model, ResidueGraphClassifier):
        if model.graph_kind != "gcn":
            return None
        input_adapter = "identity"
        mask_source = "explicit_or_feature_compatibility_mask"
        layer = getattr(model, "layer1", None)
        hidden_dim = getattr(getattr(layer, "proj", None), "out_features", None)
    else:
        return None

    try:
        version = int(getattr(model, "gcn_predictor_contract_version", 0) or 0)
    except (TypeError, ValueError):
        version = 0
    current = version >= GCN_PREDICTOR_CONTRACT_VERSION
    return {
        "schema_version": 1,
        "implementation": "pepplatform_masked_chain_gcn",
        "contract_version": version,
        "status": "verified_platform_variant" if current else "legacy_only",
        "input_adapter": input_adapter,
        "node_definition": "one node per residue or sequence position",
        "edges": "bidirectional adjacent residues plus self-loops; masked neighbors excluded",
        "aggregation": "masked mean over previous, self, and next node followed by a trainable linear projection",
        "layers": 2,
        "hidden_dim": int(hidden_dim) if hidden_dim is not None else None,
        "readout": "masked mean plus masked max",
        "padding_mask": mask_source if current else "legacy_route_dependent",
        "prediction_output": "logits",
    }


def gat_predictor_contract(model):
    """Return auditable GAT assembly metadata, or None for non-GAT models."""

    if isinstance(model, PretrainedResiduePredictor):
        if model.architecture != "gat":
            return None
        input_adapter = "layer_norm_linear_projection"
        mask_source = "explicit_residue_mask"
        layer = getattr(model, "graph_layer1", None)
    elif isinstance(model, ResidueGraphClassifier):
        if model.graph_kind != "gat":
            return None
        input_adapter = "identity"
        mask_source = "explicit_or_feature_compatibility_mask"
        layer = getattr(model, "layer1", None)
    else:
        return None

    try:
        version = int(getattr(model, "gat_predictor_contract_version", 0) or 0)
    except (TypeError, ValueError):
        version = 0
    current = version >= GAT_PREDICTOR_CONTRACT_VERSION
    hidden_dim = getattr(getattr(layer, "proj", None), "out_features", None)
    return {
        "schema_version": 1,
        "implementation": "pepplatform_masked_chain_gat",
        "contract_version": version,
        "status": "verified_platform_variant" if current else "legacy_only",
        "input_adapter": input_adapter,
        "node_definition": "one node per residue or sequence position",
        "edges": "previous, self, and next valid residue positions",
        "attention": "multi-head additive neighbor attention with destination-wise softmax",
        "heads": int(getattr(layer, "heads", 0) or 0),
        "layers": 2,
        "hidden_dim": int(hidden_dim) if hidden_dim is not None else None,
        "readout": "masked mean plus masked max",
        "padding_mask": mask_source if current else "legacy_route_dependent",
        "prediction_output": "logits",
    }


def sequence_predictor_contract(model):
    """Return the architecture contract for a supported sequence Predictor."""

    return (
        cnn_predictor_contract(model)
        or gcn_predictor_contract(model)
        or gat_predictor_contract(model)
        or rnn_predictor_contract(model)
        or gru_predictor_contract(model)
        or lstm_predictor_contract(model)
        or transformer_predictor_contract(model)
    )


class GatedResidualClassifier(nn.Module):
    def __init__(self, input_dim=320, hidden_dim=256, num_classes=15, dropout=0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.input = nn.Linear(input_dim, hidden_dim)
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward_with_embedding(self, x):
        x = self.input(x)
        x = x + self.ffn1(x)
        x = x + self.ffn2(x)
        x = x * self.gate(x)
        logits = self.out(x)
        return logits, x

    @property
    def semantic_embedding_dim(self) -> int:
        return self.hidden_dim

    def forward(self, x):
        logits, _ = self.forward_with_embedding(x)
        return logits


# Old full-model artifacts resolve this module-level symbol during unpickling.
ESMClassifier = GatedResidualClassifier


class ModelFactory:
    POSITION_FEATURE_ENCODERS = frozenset(SEQUENCE_ENCODERS).difference({"index"})
    GRAPH_FEATURE_ENCODERS = frozenset(CHAIN_GRAPH_COMPATIBLE_ENCODERS)

    @staticmethod
    def _choices_text(encoders):
        return ",".join(sorted(encoders))

    def create(
        self,
        model,
        encoder,
        featurizer,
        num_classes,
        dropout=0.0,
        head_hidden_dims=None,
        head_activation="relu",
        head_dropout=None,
    ):
        head_kwargs = {
            "head_hidden_dims": head_hidden_dims,
            "head_activation": head_activation,
            "head_dropout": head_dropout,
        }
        if model in ["gcn", "gat"]:
            if encoder == "index":
                raise ValueError("residue gcn/gat requires a feature encoder, not raw indices")
            if encoder == "onehot":
                in_dim = featurizer.vocab_size
            elif encoder in self.GRAPH_FEATURE_ENCODERS:
                in_dim = featurizer.feature_dim
            else:
                raise ValueError(
                    "residue gcn/gat requires encoder in {"
                    + self._choices_text(self.GRAPH_FEATURE_ENCODERS)
                    + "}"
                )
            return ResidueGraphClassifier(
                in_dim=in_dim,
                num_classes=num_classes,
                hidden_dim=128,
                graph_kind=model,
                heads=4,
                dropout=dropout,
                **head_kwargs,
            )
        if model == "cnn":
            if encoder == "index":
                return IndexTextCNN(
                    vocab_size=featurizer.vocab_size,
                    num_classes=num_classes,
                    dropout=dropout,
                    **head_kwargs,
                )
            if encoder in self.POSITION_FEATURE_ENCODERS:
                in_dim = featurizer.vocab_size if encoder == "onehot" else featurizer.feature_dim
                return FeatureTextCNN(
                    in_dim=in_dim, num_classes=num_classes, dropout=dropout, **head_kwargs
                )
            raise ValueError(
                "cnn requires encoder in {index,"
                + self._choices_text(self.POSITION_FEATURE_ENCODERS)
                + "}"
            )
        if model == "rnn":
            if encoder == "index":
                return IndexRNN(
                    vocab_size=featurizer.vocab_size,
                    num_classes=num_classes,
                    dropout=dropout,
                    **head_kwargs,
                )
            if encoder in self.POSITION_FEATURE_ENCODERS:
                in_dim = featurizer.vocab_size if encoder == "onehot" else featurizer.feature_dim
                return FeatureRNN(
                    in_dim=in_dim, num_classes=num_classes, dropout=dropout, **head_kwargs
                )
            raise ValueError(
                "rnn requires encoder in {index,"
                + self._choices_text(self.POSITION_FEATURE_ENCODERS)
                + "}"
            )
        if model == "gru":
            if encoder == "index":
                return IndexGRU(
                    vocab_size=featurizer.vocab_size,
                    num_classes=num_classes,
                    dropout=dropout,
                    **head_kwargs,
                )
            if encoder in self.POSITION_FEATURE_ENCODERS:
                in_dim = featurizer.vocab_size if encoder == "onehot" else featurizer.feature_dim
                return FeatureGRU(
                    in_dim=in_dim, num_classes=num_classes, dropout=dropout, **head_kwargs
                )
            raise ValueError(
                "gru requires encoder in {index,"
                + self._choices_text(self.POSITION_FEATURE_ENCODERS)
                + "}"
            )
        if model == "lstm":
            if encoder == "index":
                return IndexLSTM(
                    vocab_size=featurizer.vocab_size,
                    num_classes=num_classes,
                    dropout=dropout,
                    **head_kwargs,
                )
            if encoder in self.POSITION_FEATURE_ENCODERS:
                in_dim = featurizer.vocab_size if encoder == "onehot" else featurizer.feature_dim
                return FeatureLSTM(
                    in_dim=in_dim, num_classes=num_classes, dropout=dropout, **head_kwargs
                )
            raise ValueError(
                "lstm requires encoder in {index,"
                + self._choices_text(self.POSITION_FEATURE_ENCODERS)
                + "}"
            )
        if model == "transformer":
            if encoder == "index":
                return IndexTransformer(
                    vocab_size=featurizer.vocab_size,
                    max_length=featurizer.max_length,
                    num_classes=num_classes,
                    dropout=dropout,
                    **head_kwargs,
                )
            if encoder in self.POSITION_FEATURE_ENCODERS:
                in_dim = featurizer.vocab_size if encoder == "onehot" else featurizer.feature_dim
                return FeatureTransformer(
                    in_dim=in_dim,
                    max_length=featurizer.max_length,
                    num_classes=num_classes,
                    dropout=dropout,
                    **head_kwargs,
                )
            raise ValueError(
                "transformer requires encoder in {index,"
                + self._choices_text(self.POSITION_FEATURE_ENCODERS)
                + "}"
            )
        if model == "mlp":
            if encoder in ["bow", "tfidf", "textrank"]:
                in_dim = featurizer.vocab_size
                return BoWMLP(
                    in_dim=in_dim,
                    num_classes=num_classes,
                    dropout=dropout,
                    **head_kwargs,
                )
            if encoder in {"ac", "cc", "acc", "pdt", *PSSM_COVARIANCE_ENCODERS}:
                in_dim = featurizer.feature_dim
                return BoWMLP(
                    in_dim=in_dim,
                    num_classes=num_classes,
                    dropout=dropout,
                    **head_kwargs,
                )
        if encoder == "index":
            return IndexTextCNN(
                vocab_size=featurizer.vocab_size,
                num_classes=num_classes,
                dropout=dropout,
                **head_kwargs,
            )
        if encoder in self.POSITION_FEATURE_ENCODERS:
            in_dim = featurizer.vocab_size if encoder == "onehot" else featurizer.feature_dim
            return FeatureTextCNN(
                in_dim=in_dim, num_classes=num_classes, dropout=dropout, **head_kwargs
            )
        if encoder in ["bow", "textrank"]:
            raise ValueError("bow/textrank requires model in {mlp,svm,rf,xgb,lgb,logreg}")
        if encoder == "ac":
            raise ValueError("ac requires model in {mlp,svm,rf,xgb,lgb,logreg}")
        if encoder == "cc":
            raise ValueError("cc requires model in {mlp,svm,rf,xgb,lgb,logreg}")
        if encoder == "acc":
            raise ValueError("acc requires model in {mlp,svm,rf,xgb,lgb,logreg}")
        if encoder == "pdt":
            raise ValueError("pdt requires model in {mlp,svm,rf,xgb,lgb,logreg}")
        if encoder in PSSM_COVARIANCE_ENCODERS:
            raise ValueError(f"{encoder} requires model in {{mlp,svm,rf,xgb,lgb,logreg}}")
        raise ValueError("unsupported encoder")
