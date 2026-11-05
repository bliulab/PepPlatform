"""Validated deterministic and learned vector fusion for ordered Pair streams."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import torch


PAIR_FUSION_CONTRACT_VERSION = 1
PAIR_FUSION_IMPLEMENTATION = "pepplatform_role_ordered_pair_vector_fusion"
ALIGNED_GATED_CONTRACT_VERSION = 1
ALIGNED_GATED_IMPLEMENTATION = "pepplatform_aligned_gated_fusion"
AUDITED_PAIR_FUSION_MODES = frozenset(
    {"concat", "concat_absdiff", "concat_hadamard", "concat_all", "gated"}
)
PAIR_FUSION_MODES = frozenset(
    {
        "concat",
        "concat_absdiff",
        "concat_hadamard",
        "concat_all",
        "gated",
        "aligned_gated",
    }
)
PAIR_FUSION_ALIASES = {
    "concat+absdiff+hadamard": "concat_all",
    "concat_absdiff_hadamard": "concat_all",
    "all": "concat_all",
    "concat+absdiff": "concat_absdiff",
    "concat+hadamard": "concat_hadamard",
    "gate": "gated",
    "gated_fusion": "gated",
    "gated(static)": "gated",
}
_REMOVED_SEMANTIC_MODES = frozenset({"concat_semantic", "semantic", "concat+semantic"})
_ELEMENTWISE_MODES = frozenset({"concat_absdiff", "concat_hadamard", "concat_all", "gated"})


class PairFusionError(ValueError):
    """Raised when a Pair fusion mode, input, or artifact violates its contract."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PairFusionError(
            "aligned_gated_dimension_invalid",
            f"{name} must be a positive integer",
        )
    return int(value)


def _dropout_probability(value: object) -> float:
    if isinstance(value, bool):
        valid = False
    else:
        try:
            value = float(value)
            valid = math.isfinite(value) and 0.0 <= value < 1.0
        except (TypeError, ValueError):
            valid = False
    if not valid:
        raise PairFusionError(
            "aligned_gated_dropout_invalid",
            "aligned_gated dropout must be finite and in [0, 1)",
        )
    return value


def _validate_aligned_input(
    role: str,
    value: object,
    *,
    expected_width: int,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise PairFusionError(
            "aligned_gated_input_type_invalid",
            f"{role} aligned_gated input must be a torch.Tensor",
        )
    if value.layout != torch.strided:
        raise PairFusionError(
            "aligned_gated_input_layout_invalid",
            f"{role} aligned_gated input must be a dense strided tensor",
        )
    if value.ndim != 2:
        raise PairFusionError(
            "aligned_gated_input_rank_invalid",
            f"{role} aligned_gated input must have shape [batch, features]",
        )
    if value.shape[0] < 1:
        raise PairFusionError(
            "aligned_gated_input_empty",
            "aligned_gated inputs must contain at least one sample",
        )
    if value.shape[1] != expected_width:
        raise PairFusionError(
            "aligned_gated_input_width_invalid",
            f"{role} aligned_gated input width must be {expected_width}, got {value.shape[1]}",
        )
    if not value.dtype.is_floating_point:
        raise PairFusionError(
            "aligned_gated_input_dtype_invalid",
            f"{role} aligned_gated input must use a floating dtype",
        )
    normalized = value.float()
    if not bool(torch.isfinite(normalized).all().item()):
        raise PairFusionError(
            "aligned_gated_input_nonfinite",
            f"{role} aligned_gated input must contain only finite values",
        )
    return normalized


class AlignedGatedFusion(torch.nn.Module):
    """Learned elementwise convex gating for ordered Pair stream vectors."""

    def __init__(
        self,
        peptide_dim: int,
        partner_dim: int,
        *,
        alignment_dim: int | None = None,
        gate_hidden_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.peptide_dim = _positive_integer("peptide_dim", peptide_dim)
        self.partner_dim = _positive_integer("partner_dim", partner_dim)
        self.gate_hidden_dim = _positive_integer("gate_hidden_dim", gate_hidden_dim)
        self.dropout_probability = _dropout_probability(dropout)

        if self.peptide_dim == self.partner_dim:
            if alignment_dim is not None and alignment_dim != self.peptide_dim:
                raise PairFusionError(
                    "aligned_gated_dimension_invalid",
                    "equal-width streams use identity projections, so alignment_dim "
                    "must be omitted or equal to the stream width",
                )
            self.alignment_dim = self.peptide_dim
            self.peptide_projection = torch.nn.Identity()
            self.partner_projection = torch.nn.Identity()
        else:
            self.alignment_dim = _positive_integer("alignment_dim", alignment_dim)
            self.peptide_projection = torch.nn.Linear(self.peptide_dim, self.alignment_dim)
            self.partner_projection = torch.nn.Linear(self.partner_dim, self.alignment_dim)

        self.gate = torch.nn.Sequential(
            torch.nn.Linear(self.alignment_dim * 4, self.gate_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(self.dropout_probability),
            torch.nn.Linear(self.gate_hidden_dim, self.alignment_dim),
            torch.nn.Sigmoid(),
        )

    @property
    def output_dim(self) -> int:
        return self.alignment_dim * 3

    def _aligned_inputs(
        self, peptide: object, partner: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        peptide = _validate_aligned_input("peptide", peptide, expected_width=self.peptide_dim)
        partner = _validate_aligned_input("partner", partner, expected_width=self.partner_dim)
        if peptide.device != partner.device:
            raise PairFusionError(
                "aligned_gated_device_mismatch",
                "peptide and partner aligned_gated inputs must use the same device",
            )
        if peptide.shape[0] != partner.shape[0]:
            raise PairFusionError(
                "aligned_gated_batch_mismatch",
                "peptide and partner aligned_gated inputs must have the same batch size",
            )
        peptide = self.peptide_projection(peptide)
        partner = self.partner_projection(partner)
        if not bool(torch.isfinite(peptide).all().item()) or not bool(
            torch.isfinite(partner).all().item()
        ):
            raise PairFusionError(
                "aligned_gated_projection_nonfinite",
                "aligned_gated role projection produced non-finite values",
            )
        return peptide, partner

    def role_embeddings(
        self, peptide: object, partner: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return aligned/projected role embeddings without fusing them."""

        return self._aligned_inputs(peptide, partner)

    @staticmethod
    def _basis(peptide: torch.Tensor, partner: torch.Tensor) -> torch.Tensor:
        return torch.cat([peptide, partner, (peptide - partner).abs(), peptide * partner], dim=1)

    def topology_basis(self, peptide: object, partner: object) -> torch.Tensor:
        peptide, partner = self._aligned_inputs(peptide, partner)
        basis = self._basis(peptide, partner)
        if not bool(torch.isfinite(basis).all().item()):
            raise PairFusionError(
                "aligned_gated_output_nonfinite",
                "aligned_gated topology basis contains non-finite values",
            )
        return basis

    def forward(self, peptide: object, partner: object) -> torch.Tensor:
        peptide, partner = self._aligned_inputs(peptide, partner)
        absolute = (peptide - partner).abs()
        product = peptide * partner
        weights = self.gate(torch.cat([peptide, partner, absolute, product], dim=1))
        fused = torch.cat(
            [weights * peptide + (1.0 - weights) * partner, absolute, product],
            dim=1,
        )
        if not bool(torch.isfinite(fused).all().item()):
            raise PairFusionError(
                "aligned_gated_output_nonfinite",
                "aligned_gated fusion produced non-finite values",
            )
        return fused


def aligned_gated_contract(module: AlignedGatedFusion) -> dict[str, Any]:
    if not isinstance(module, AlignedGatedFusion):
        raise PairFusionError(
            "aligned_gated_contract_invalid",
            "aligned_gated contract requires an AlignedGatedFusion module",
        )
    return {
        "contract_version": ALIGNED_GATED_CONTRACT_VERSION,
        "implementation": ALIGNED_GATED_IMPLEMENTATION,
        "status": "verified_platform_variant",
        "role_order": ["peptide", "partner"],
        "input_contract": "dense_finite_rank2_floating_tensor",
        "input_dimensions": {
            "peptide": module.peptide_dim,
            "partner": module.partner_dim,
        },
        "alignment_dimension": module.alignment_dim,
        "role_projections": {
            "peptide": "identity"
            if isinstance(module.peptide_projection, torch.nn.Identity)
            else "linear",
            "partner": "identity"
            if isinstance(module.partner_projection, torch.nn.Identity)
            else "linear",
        },
        "gate": {
            "input_blocks": [
                "peptide_aligned",
                "partner_aligned",
                "absolute_difference",
                "hadamard_product",
            ],
            "input_dimension": module.alignment_dim * 4,
            "hidden_dimension": module.gate_hidden_dim,
            "activation": "gelu",
            "dropout": module.dropout_probability,
            "output_dimension": module.alignment_dim,
            "output_activation": "sigmoid",
        },
        "mix": "gate * peptide_aligned + (1 - gate) * partner_aligned",
        "output_blocks": [
            "convex_elementwise_mix",
            "absolute_difference",
            "hadamard_product",
        ],
        "output_dimension": module.output_dim,
        "output_dtype": "float32",
        "torch_version": str(torch.__version__),
    }


def validate_aligned_gated_contract(contract: object, module: AlignedGatedFusion) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        raise PairFusionError(
            "aligned_gated_contract_missing",
            "artifact is missing aligned_gated_contract v1",
        )
    expected = aligned_gated_contract(module)
    comparable = dict(expected)
    comparable["torch_version"] = contract.get("torch_version")
    if not isinstance(contract.get("torch_version"), str) or not contract.get("torch_version"):
        raise PairFusionError(
            "aligned_gated_contract_invalid",
            "aligned_gated artifact must record its PyTorch version",
        )
    if dict(contract) != comparable:
        raise PairFusionError(
            "aligned_gated_contract_invalid",
            "aligned_gated artifact contract does not match the loaded module",
        )
    return dict(contract)


def normalize_pair_fusion_mode(raw_mode: object, *, default: str = "concat_all") -> str:
    mode = str(raw_mode or default).strip().lower()
    if mode in _REMOVED_SEMANTIC_MODES:
        raise PairFusionError(
            "pair_fusion_mode_removed",
            "fusion_mode=concat_semantic is disabled. Semantic similarity is "
            "auxiliary-loss-only; use enable_semantic_similarity instead.",
        )
    mode = PAIR_FUSION_ALIASES.get(mode, mode)
    if mode not in PAIR_FUSION_MODES:
        raise PairFusionError(
            "pair_fusion_mode_unsupported",
            f"unsupported pair fusion mode: {raw_mode}",
        )
    return mode


def _validate_tensor(role: str, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise PairFusionError(
            "pair_fusion_input_type_invalid",
            f"{role} fusion input must be a torch.Tensor",
        )
    if value.layout != torch.strided:
        raise PairFusionError(
            "pair_fusion_input_layout_invalid",
            f"{role} fusion input must be a dense strided tensor",
        )
    if value.ndim != 2:
        raise PairFusionError(
            "pair_fusion_input_rank_invalid",
            f"{role} fusion input must have shape [batch, features]",
        )
    if value.shape[1] < 1:
        raise PairFusionError(
            "pair_fusion_feature_width_invalid",
            f"{role} fusion input must have a positive feature width",
        )
    if value.dtype == torch.bool or value.dtype.is_complex:
        raise PairFusionError(
            "pair_fusion_input_dtype_invalid",
            f"{role} fusion input must use a real integer or floating dtype",
        )
    try:
        normalized = value.float()
    except (RuntimeError, TypeError) as exc:
        raise PairFusionError(
            "pair_fusion_input_dtype_invalid",
            f"{role} fusion input cannot be converted to float32",
        ) from exc
    if not bool(torch.isfinite(normalized).all().item()):
        raise PairFusionError(
            "pair_fusion_input_nonfinite",
            f"{role} fusion input must contain only finite values",
        )
    return normalized


def fuse_pair_embeddings(
    peptide_embeddings: object,
    partner_embeddings: object,
    mode: object,
) -> torch.Tensor:
    """Fuse two ordered Pair streams for one of the audited deterministic modes."""

    canonical = normalize_pair_fusion_mode(mode)
    if canonical not in AUDITED_PAIR_FUSION_MODES:
        raise PairFusionError(
            "pair_fusion_mode_unsupported",
            f"shared audited fusion does not implement mode={canonical}",
        )
    peptide = _validate_tensor("peptide", peptide_embeddings)
    partner = _validate_tensor("partner", partner_embeddings)
    if peptide.device != partner.device:
        raise PairFusionError(
            "pair_fusion_device_mismatch",
            "peptide and partner fusion inputs must use the same device",
        )
    if peptide.shape[0] != partner.shape[0]:
        raise PairFusionError(
            "pair_fusion_batch_mismatch",
            "peptide and partner fusion inputs must have the same batch size",
        )
    if canonical in _ELEMENTWISE_MODES and peptide.shape[1] != partner.shape[1]:
        raise PairFusionError(
            "pair_fusion_dimension_mismatch",
            f"fusion_mode={canonical} requires equal stream dimensions, got "
            f"peptide={peptide.shape[1]} and partner={partner.shape[1]}",
        )
    if canonical == "concat":
        fused = torch.cat([peptide, partner], dim=1)
    elif canonical == "concat_absdiff":
        fused = torch.cat([peptide, partner, (peptide - partner).abs()], dim=1)
    elif canonical == "concat_hadamard":
        fused = torch.cat([peptide, partner, peptide * partner], dim=1)
    elif canonical == "concat_all":
        fused = torch.cat(
            [
                peptide,
                partner,
                (peptide - partner).abs(),
                peptide * partner,
            ],
            dim=1,
        )
    else:
        gate = torch.sigmoid((peptide * partner).sum(dim=1, keepdim=True))
        gated = gate * peptide + (1.0 - gate) * partner
        fused = torch.cat(
            [gated, (peptide - partner).abs(), peptide * partner],
            dim=1,
        )
    if not bool(torch.isfinite(fused).all().item()):
        raise PairFusionError(
            "pair_fusion_output_nonfinite",
            f"fusion_mode={canonical} produced non-finite values",
        )
    return fused


def pair_fusion_contract(
    mode: object,
    peptide_embeddings: torch.Tensor,
    partner_embeddings: torch.Tensor,
    fused_embeddings: torch.Tensor,
) -> dict[str, Any]:
    canonical = normalize_pair_fusion_mode(mode)
    if canonical not in AUDITED_PAIR_FUSION_MODES:
        raise PairFusionError(
            "pair_fusion_contract_mode_invalid",
            f"pair fusion contract v1 does not cover mode={canonical}",
        )
    contract = {
        "contract_version": PAIR_FUSION_CONTRACT_VERSION,
        "implementation": PAIR_FUSION_IMPLEMENTATION,
        "mode": canonical,
        "role_order": ["peptide", "partner"],
        "input_contract": "dense_finite_rank2_real_tensor",
        "input_dimensions": {
            "peptide": int(peptide_embeddings.shape[1]),
            "partner": int(partner_embeddings.shape[1]),
        },
        "output_dimension": int(fused_embeddings.shape[1]),
        "output_dtype": "float32",
        "torch_version": str(torch.__version__),
    }
    if canonical == "concat_all":
        contract["output_blocks"] = [
            "peptide",
            "partner",
            "absolute_difference",
            "hadamard_product",
        ]
    elif canonical == "gated":
        contract["gate"] = {
            "kind": "static_scalar_similarity",
            "score": "sum(peptide * partner)",
            "activation": "sigmoid",
            "mix": "gate * peptide + (1 - gate) * partner",
            "output_blocks": [
                "gated_mix",
                "absolute_difference",
                "hadamard_product",
            ],
        }
    return contract


def validate_pair_fusion_contract(
    contract: object,
    *,
    mode: object,
    peptide_embeddings: torch.Tensor,
    partner_embeddings: torch.Tensor,
    fused_embeddings: torch.Tensor,
) -> dict[str, Any]:
    canonical = normalize_pair_fusion_mode(mode)
    if not isinstance(contract, Mapping):
        raise PairFusionError(
            "pair_fusion_contract_missing",
            f"{canonical} artifact is missing pair_fusion_contract v1",
        )
    expected = pair_fusion_contract(
        canonical,
        peptide_embeddings,
        partner_embeddings,
        fused_embeddings,
    )
    comparable = dict(expected)
    comparable["torch_version"] = contract.get("torch_version")
    if not isinstance(contract.get("torch_version"), str) or not contract.get("torch_version"):
        raise PairFusionError(
            "pair_fusion_contract_invalid",
            "pair fusion artifact must record its PyTorch version",
        )
    if dict(contract) != comparable:
        raise PairFusionError(
            "pair_fusion_contract_invalid",
            f"{canonical} artifact contract does not match reconstructed Pair features",
        )
    return dict(contract)
