"""Versioned Pair strategy contracts shared by training and inference."""

from __future__ import annotations

from typing import Any, Mapping

import torch


PAIR_STRATEGY_CONTRACT_VERSION = 1
SINGLE_TOWER_JOIN_CONTRACT_VERSION = 1
SINGLE_TOWER_JOIN_MODE = "role_boundary_sequence_join_v1"
SINGLE_TOWER_LAYOUT = "compact_valid_peptide_boundary_compact_valid_partner_right_pad"
PAIR_ROLE_ORDER = ("peptide", "partner")
FEATURE_ROLE_CHANNELS = ("peptide", "partner", "boundary")
DUAL_TOWER_PROCESSING = "independent_encoders"
DUAL_TOWER_VECTOR_INPUT_ADAPTER = "dual_tower_vector_fusion"
DUAL_TOWER_TOKEN_INPUT_ADAPTER = "dual_tower_token_interaction"
DUAL_TOWER_HGT_INPUT_ADAPTER = "typed_role_projection"
_DUAL_TOWER_INDEX_DTYPES = frozenset({"torch.int32", "torch.int64"})
_DUAL_TOWER_FLOAT_DTYPES = frozenset(
    {
        "torch.float16",
        "torch.float32",
        "torch.float64",
        "torch.bfloat16",
        "torch.float8_e4m3fn",
        "torch.float8_e4m3fnuz",
        "torch.float8_e5m2",
        "torch.float8_e5m2fnuz",
    }
)
_FINITE_CHECK_MAX_ELEMENTS = 8_388_608


class PairStrategyContractError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)


def _all_finite_chunked(tensor: torch.Tensor) -> bool:
    """Check every element without allocating dataset-sized boolean temporaries.

    Slice along dimensions rather than flattening: flattening a non-contiguous
    tensor can itself allocate a full copy. A single row larger than the budget
    is checked recursively along its next dimension.
    """
    if tensor.numel() <= _FINITE_CHECK_MAX_ELEMENTS:
        return bool(torch.isfinite(tensor).all())
    row_elements = tensor[0].numel()
    if row_elements > _FINITE_CHECK_MAX_ELEMENTS:
        return all(_all_finite_chunked(tensor[index]) for index in range(tensor.shape[0]))
    rows = max(1, _FINITE_CHECK_MAX_ELEMENTS // max(1, row_elements))
    return all(
        bool(torch.isfinite(tensor[start : start + rows]).all())
        for start in range(0, tensor.shape[0], rows)
    )


def require_single_tower_homogeneous(
    towers: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    if not towers:
        return
    peptide = towers.get("peptide")
    partner = towers.get("partner")
    if not isinstance(peptide, Mapping) or not isinstance(partner, Mapping):
        raise PairStrategyContractError(
            "single_tower_towers_invalid",
            "single_stream requires complete peptide and partner tower specifications",
        )
    if dict(peptide) != dict(partner):
        raise PairStrategyContractError(
            "single_tower_requires_homogeneous_towers",
            "The current Single-stream implementation requires identical "
            "peptide and partner tower specifications",
        )


def _strategy_tensor_descriptor(values, *, role: str, interaction_level: str) -> dict[str, Any]:
    ResidueModelInput = _residue_model_input_type()
    explicit_mask = isinstance(values, ResidueModelInput)
    tensor = values.features if explicit_mask else values
    if not isinstance(tensor, torch.Tensor):
        raise PairStrategyContractError(
            "dual_tower_input_type_invalid",
            f"{role} Two-stream input must be a torch Tensor or ResidueModelInput",
        )
    if tensor.ndim not in {2, 3}:
        raise PairStrategyContractError(
            "dual_tower_input_rank_invalid",
            f"{role} Two-stream input must have rank 2 or 3",
        )
    if tensor.shape[0] < 1 or tensor.shape[1] < 1:
        raise PairStrategyContractError(
            "dual_tower_input_empty",
            f"{role} Two-stream input must contain at least one row and feature/token position",
        )
    if interaction_level == "vector":
        if tensor.ndim != 2 or not tensor.is_floating_point():
            raise PairStrategyContractError(
                "dual_tower_feature_rank_invalid",
                f"{role} vector Two-stream inputs must be finite floating rank-2 tensors",
            )
        if not _all_finite_chunked(tensor):
            raise PairStrategyContractError(
                "dual_tower_input_nonfinite",
                f"{role} Two-stream features must contain only finite values",
            )
        input_kind = "feature"
        token_length = 0
        feature_dim = int(tensor.shape[1])
    elif tensor.is_floating_point():
        if tensor.ndim != 3:
            raise PairStrategyContractError(
                "dual_tower_feature_rank_invalid",
                f"{role} token Two-stream feature inputs must have rank 3",
            )
        if not _all_finite_chunked(tensor):
            raise PairStrategyContractError(
                "dual_tower_input_nonfinite",
                f"{role} Two-stream features must contain only finite values",
            )
        input_kind = "feature"
        token_length = int(tensor.shape[1])
        feature_dim = int(tensor.shape[2])
    elif tensor.ndim == 2 and tensor.dtype in {torch.int32, torch.int64}:
        if bool((tensor < 0).any()):
            raise PairStrategyContractError(
                "dual_tower_index_value_invalid",
                f"{role} Two-stream token IDs must be non-negative",
            )
        input_kind = "index"
        token_length = int(tensor.shape[1])
        feature_dim = 0
    else:
        raise PairStrategyContractError(
            "dual_tower_index_dtype_invalid",
            f"{role} Two-stream index inputs must be int32/int64 rank-2 tensors",
        )
    mask = values.mask if explicit_mask else None
    if mask is not None and (not isinstance(mask, torch.Tensor) or mask.shape != tensor.shape[:2]):
        raise PairStrategyContractError(
            "dual_tower_mask_shape_invalid",
            f"{role} Two-stream mask must match [batch, tokens]",
        )
    if mask is not None and not bool(mask.to(dtype=torch.bool).any(dim=1).all()):
        raise PairStrategyContractError(
            "dual_tower_empty_role",
            f"every Two-stream {role} row must contain a valid token",
        )
    return {
        "input_kind": input_kind,
        "rank": int(tensor.ndim),
        "dtype": str(tensor.dtype),
        "batch_size": int(tensor.shape[0]),
        "token_length": token_length,
        "feature_dim": feature_dim,
        "explicit_mask": bool(explicit_mask),
    }


def build_dual_tower_strategy_contract(
    args,
    peptide=None,
    partner=None,
    *,
    predictor_type: str | None = None,
    input_adapter: str | None = None,
) -> dict[str, Any]:
    """Build the v1 role contract used by every Two-stream artifact."""
    pair_strategy = (
        str(getattr(args, "pair_strategy", "dual_tower") or "dual_tower").strip().lower()
    )
    if pair_strategy != "dual_tower":
        raise PairStrategyContractError(
            "dual_tower_strategy_invalid",
            "two_stream strategy contract requires pair_strategy=two_stream",
        )
    level = str(getattr(args, "interaction_level", "vector") or "vector").strip().lower()
    if level not in {"vector", "token"}:
        raise PairStrategyContractError(
            "dual_tower_interaction_level_invalid",
            "two_stream interaction_level must be vector or token",
        )
    predictor = (
        str(predictor_type or getattr(args, "predictor_type", "classifier") or "classifier")
        .strip()
        .lower()
    )
    if predictor not in {"classifier", "sample_graph"}:
        raise PairStrategyContractError(
            "dual_tower_predictor_type_invalid",
            "two_stream predictor_type must be classifier or sample_graph",
        )
    model = (
        str(getattr(args, "graph_predictor", "") or getattr(args, "model", "") or "")
        .strip()
        .lower()
    )
    hgt = (
        model == "hgt"
        or str(getattr(args, "pair_input_adapter", "") or "").strip().lower()
        == DUAL_TOWER_HGT_INPUT_ADAPTER
    )
    if input_adapter is None:
        input_adapter = (
            DUAL_TOWER_HGT_INPUT_ADAPTER
            if hgt
            else DUAL_TOWER_TOKEN_INPUT_ADAPTER
            if level == "token"
            else DUAL_TOWER_VECTOR_INPUT_ADAPTER
        )
    input_adapter = str(input_adapter).strip().lower()
    allowed_adapters = {
        DUAL_TOWER_HGT_INPUT_ADAPTER,
        DUAL_TOWER_TOKEN_INPUT_ADAPTER,
        DUAL_TOWER_VECTOR_INPUT_ADAPTER,
    }
    if input_adapter not in allowed_adapters:
        raise PairStrategyContractError(
            "dual_tower_input_adapter_invalid",
            f"unsupported Two-stream input adapter: {input_adapter}",
        )
    if hgt and input_adapter != DUAL_TOWER_HGT_INPUT_ADAPTER:
        raise PairStrategyContractError(
            "dual_tower_hgt_adapter_invalid",
            "HGT two_stream routes require typed_role_projection",
        )
    if level == "token" and input_adapter != DUAL_TOWER_TOKEN_INPUT_ADAPTER:
        raise PairStrategyContractError(
            "dual_tower_token_adapter_invalid",
            "token two_stream routes require dual_tower_token_interaction",
        )
    if level == "vector" and input_adapter == DUAL_TOWER_TOKEN_INPUT_ADAPTER:
        raise PairStrategyContractError(
            "dual_tower_vector_adapter_invalid",
            "vector two_stream routes cannot use the token interaction adapter",
        )
    towers = getattr(args, "towers", None) or {}
    tower_mode = (
        str(
            getattr(args, "tower_mode", "")
            or (
                "heterogeneous"
                if towers and dict(towers.get("peptide", {})) != dict(towers.get("partner", {}))
                else "homogeneous"
            )
        )
        .strip()
        .lower()
    )
    if tower_mode not in {"homogeneous", "heterogeneous"}:
        raise PairStrategyContractError(
            "dual_tower_mode_invalid",
            "two_stream tower_mode must be homogeneous or heterogeneous",
        )
    if peptide is None or partner is None:
        raise PairStrategyContractError(
            "dual_tower_inputs_missing",
            "two_stream strategy contract requires both role inputs",
        )
    peptide_info = _strategy_tensor_descriptor(peptide, role="peptide", interaction_level=level)
    partner_info = _strategy_tensor_descriptor(partner, role="partner", interaction_level=level)
    if peptide_info["batch_size"] != partner_info["batch_size"]:
        raise PairStrategyContractError(
            "dual_tower_batch_mismatch",
            "two_stream peptide and partner batch sizes must match",
        )
    contract = {
        "contract_version": PAIR_STRATEGY_CONTRACT_VERSION,
        "strategy": "dual_tower",
        "role_order": list(PAIR_ROLE_ORDER),
        "tower_mode": tower_mode,
        "tower_processing": DUAL_TOWER_PROCESSING,
        "interaction_level": level,
        "predictor_type": predictor,
        "input_adapter": input_adapter,
        "towers": {
            "peptide": peptide_info,
            "partner": partner_info,
        },
    }
    if level == "token":
        contract["interaction_model"] = (
            str(getattr(args, "interaction_model", "cross_attn") or "cross_attn").strip().lower()
        )
    elif input_adapter != DUAL_TOWER_HGT_INPUT_ADAPTER:
        contract["fusion_mode"] = (
            str(getattr(args, "fusion_mode", "concat_all") or "concat_all").strip().lower()
        )
    if predictor == "sample_graph":
        contract["graph_predictor"] = (
            str(getattr(args, "graph_predictor", "") or getattr(args, "model", "") or "")
            .strip()
            .lower()
        )
    return contract


def validate_dual_tower_strategy_contract(
    contract: Mapping[str, Any] | None,
    *,
    peptide=None,
    partner=None,
) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        raise PairStrategyContractError(
            "dual_tower_strategy_contract_missing",
            "Two-stream artifact is missing pair_strategy_contract v1",
        )
    normalized = dict(contract)
    towers = normalized.get("towers")
    valid = (
        normalized.get("contract_version") == PAIR_STRATEGY_CONTRACT_VERSION
        and normalized.get("strategy") == "dual_tower"
        and normalized.get("role_order") == list(PAIR_ROLE_ORDER)
        and normalized.get("tower_mode") in {"homogeneous", "heterogeneous"}
        and normalized.get("tower_processing") == DUAL_TOWER_PROCESSING
        and normalized.get("interaction_level") in {"vector", "token"}
        and normalized.get("predictor_type") in {"classifier", "sample_graph"}
        and normalized.get("input_adapter")
        in {
            DUAL_TOWER_VECTOR_INPUT_ADAPTER,
            DUAL_TOWER_TOKEN_INPUT_ADAPTER,
            DUAL_TOWER_HGT_INPUT_ADAPTER,
        }
        and isinstance(towers, Mapping)
        and all(isinstance(towers.get(role), Mapping) for role in PAIR_ROLE_ORDER)
    )
    if not valid:
        raise PairStrategyContractError(
            "dual_tower_strategy_contract_invalid",
            "unsupported or inconsistent Two-stream strategy contract",
        )
    adapter = normalized["input_adapter"]
    level = normalized["interaction_level"]
    if level == "token":
        if normalized.get("interaction_model") not in {"cross_attn", "bi_cross_attn"}:
            raise PairStrategyContractError(
                "dual_tower_strategy_contract_invalid",
                "token Two-stream contract requires a supported interaction model",
            )
    elif adapter != DUAL_TOWER_HGT_INPUT_ADAPTER:
        fusion_mode = normalized.get("fusion_mode")
        if not isinstance(fusion_mode, str) or not fusion_mode.strip():
            raise PairStrategyContractError(
                "dual_tower_strategy_contract_invalid",
                "vector Two-stream contract requires an explicit fusion mode",
            )
    if level == "token" and adapter != DUAL_TOWER_TOKEN_INPUT_ADAPTER:
        raise PairStrategyContractError(
            "dual_tower_strategy_contract_invalid",
            "token Two-stream contract must use the token interaction adapter",
        )
    if level == "vector" and adapter == DUAL_TOWER_TOKEN_INPUT_ADAPTER:
        raise PairStrategyContractError(
            "dual_tower_strategy_contract_invalid",
            "vector Two-stream contract cannot use the token interaction adapter",
        )
    if adapter == DUAL_TOWER_HGT_INPUT_ADAPTER:
        if level != "vector" or normalized["predictor_type"] != "sample_graph":
            raise PairStrategyContractError(
                "dual_tower_strategy_contract_invalid",
                "typed_role_projection is reserved for vector sample-graph HGT",
            )
        if "fusion_mode" in normalized:
            raise PairStrategyContractError(
                "dual_tower_strategy_contract_invalid",
                "typed HGT strategy contracts must not contain fusion_mode",
            )
        if str(normalized.get("graph_predictor", "") or "").strip().lower() != "hgt":
            raise PairStrategyContractError(
                "dual_tower_strategy_contract_invalid",
                "typed HGT strategy contracts must record graph_predictor=hgt",
            )
    if (
        normalized["predictor_type"] == "sample_graph"
        and not str(normalized.get("graph_predictor", "") or "").strip()
    ):
        raise PairStrategyContractError(
            "dual_tower_strategy_contract_invalid",
            "sample-graph Two-stream contracts require graph_predictor",
        )
    for role in PAIR_ROLE_ORDER:
        info = towers[role]
        try:
            rank = int(info.get("rank"))
            batch_size = int(info.get("batch_size"))
            token_length = int(info.get("token_length"))
            feature_dim = int(info.get("feature_dim"))
        except (TypeError, ValueError):
            rank = batch_size = token_length = feature_dim = -1
        input_kind = info.get("input_kind")
        dtype = info.get("dtype")
        dtype_valid = (
            dtype in _DUAL_TOWER_INDEX_DTYPES
            if input_kind == "index"
            else dtype in _DUAL_TOWER_FLOAT_DTYPES
            if input_kind == "feature"
            else False
        )
        if (
            input_kind not in {"index", "feature"}
            or not dtype_valid
            or rank not in {2, 3}
            or batch_size <= 0
            or (level == "token" and token_length <= 0)
            or (level == "vector" and token_length != 0)
            or feature_dim < 0
            or not isinstance(info.get("explicit_mask"), bool)
        ):
            raise PairStrategyContractError(
                "dual_tower_strategy_contract_invalid",
                f"Two-stream {role} tensor descriptor is invalid",
            )
        if level == "vector" and (input_kind != "feature" or rank != 2 or feature_dim <= 0):
            raise PairStrategyContractError(
                "dual_tower_strategy_contract_invalid",
                f"Two-stream {role} vector descriptor must be floating rank-2 feature data",
            )
        if input_kind == "index" and rank != 2:
            raise PairStrategyContractError(
                "dual_tower_strategy_contract_invalid",
                f"Two-stream {role} index descriptor must have rank 2",
            )
        if level == "token" and input_kind == "feature" and (rank != 3 or feature_dim <= 0):
            raise PairStrategyContractError(
                "dual_tower_strategy_contract_invalid",
                f"Two-stream {role} feature descriptor must have positive width",
            )
        if level == "token" and input_kind == "index" and feature_dim != 0:
            raise PairStrategyContractError(
                "dual_tower_strategy_contract_invalid",
                f"Two-stream {role} index descriptor must have feature_dim=0",
            )
    if int(towers["peptide"]["batch_size"]) != int(towers["partner"]["batch_size"]):
        raise PairStrategyContractError(
            "dual_tower_strategy_contract_invalid",
            "Two-stream role descriptors must have matching batch sizes",
        )
    if peptide is not None and partner is not None:
        observed = build_dual_tower_strategy_contract(
            type(
                "_StrategyArgs",
                (),
                {
                    "pair_strategy": "dual_tower",
                    "tower_mode": normalized["tower_mode"],
                    "interaction_level": normalized["interaction_level"],
                    "predictor_type": normalized["predictor_type"],
                    "interaction_model": normalized.get("interaction_model", "cross_attn"),
                    "fusion_mode": normalized.get("fusion_mode", "concat_all"),
                    "pair_input_adapter": normalized["input_adapter"],
                    "model": normalized.get("graph_predictor", ""),
                    "graph_predictor": normalized.get("graph_predictor", ""),
                },
            )(),
            peptide,
            partner,
            input_adapter=normalized["input_adapter"],
        )
        for role in PAIR_ROLE_ORDER:
            for key in (
                "input_kind",
                "rank",
                "dtype",
                "token_length",
                "feature_dim",
                "explicit_mask",
            ):
                if observed["towers"][role][key] != towers[role].get(key):
                    raise PairStrategyContractError(
                        "dual_tower_strategy_contract_shape_mismatch",
                        f"Two-stream {role} input does not match its artifact strategy contract",
                    )
    return normalized


def _residue_model_input_type():
    from pepplatform.models import ResidueModelInput

    return ResidueModelInput


def _as_batched_sequence(values, *, role: str, batched: bool):
    ResidueModelInput = _residue_model_input_type()
    explicit_mask = isinstance(values, ResidueModelInput)
    if explicit_mask:
        features = values.features
        mask = values.mask
    else:
        features = values
        mask = None
    if not isinstance(features, torch.Tensor):
        raise PairStrategyContractError(
            "single_tower_input_type_invalid",
            f"{role} Single-stream input must be a torch Tensor or ResidueModelInput",
        )

    expected = (2, 3) if features.is_floating_point() else (1, 2)
    expected_rank = expected[1] if batched else expected[0]
    if features.ndim != expected_rank:
        kind = "floating feature" if features.is_floating_point() else "index"
        raise PairStrategyContractError(
            "single_tower_input_rank_invalid",
            f"{role} {kind} input has rank {features.ndim}; expected {expected_rank}",
        )
    if not batched:
        features = features.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    if features.is_floating_point():
        if not torch.isfinite(features).all():
            raise PairStrategyContractError(
                "single_tower_input_nonfinite",
                f"{role} Single-stream features must contain only finite values",
            )
        if mask is None:
            mask = features.abs().sum(dim=-1).ne(0)
    else:
        if features.dtype not in {torch.int32, torch.int64}:
            raise PairStrategyContractError(
                "single_tower_index_dtype_invalid",
                f"{role} Single-stream token IDs must use int32 or int64",
            )
        if bool((features < 0).any()):
            raise PairStrategyContractError(
                "single_tower_index_value_invalid",
                f"{role} Single-stream token IDs must be non-negative",
            )
        if mask is None:
            mask = features.ne(0)

    if not isinstance(mask, torch.Tensor) or mask.shape != features.shape[:2]:
        raise PairStrategyContractError(
            "single_tower_mask_shape_invalid",
            f"{role} Single-stream mask must match the sequence dimensions",
        )
    mask = mask.to(device=features.device, dtype=torch.bool)
    if not bool(mask.any(dim=1).all()):
        raise PairStrategyContractError(
            "single_tower_empty_role",
            f"every Single-stream sample must contain at least one valid {role} position",
        )
    return features, mask, explicit_mask


def _join_single_tower_sequences(
    peptide,
    partner,
    *,
    separator_id: int | None,
    batched: bool,
):
    peptide_values, peptide_mask, peptide_explicit = _as_batched_sequence(
        peptide,
        role="peptide",
        batched=batched,
    )
    partner_values, partner_mask, partner_explicit = _as_batched_sequence(
        partner,
        role="partner",
        batched=batched,
    )
    if peptide_values.is_floating_point() != partner_values.is_floating_point():
        raise PairStrategyContractError(
            "single_tower_input_kind_mismatch",
            "single_stream peptide and partner inputs must use the same index or feature kind",
        )
    if peptide_values.shape[0] != partner_values.shape[0]:
        raise PairStrategyContractError(
            "single_tower_batch_mismatch",
            "single_stream peptide and partner batch sizes must match",
        )

    batch_size = int(peptide_values.shape[0])
    joined_length = int(peptide_values.shape[1] + partner_values.shape[1] + 1)
    if peptide_values.is_floating_point():
        if peptide_values.shape[2] != partner_values.shape[2]:
            raise PairStrategyContractError(
                "single_tower_feature_width_mismatch",
                "single_stream peptide and partner feature widths must match",
            )
        source_dim = int(peptide_values.shape[2])
        joined = peptide_values.new_zeros(
            (batch_size, joined_length, source_dim + len(FEATURE_ROLE_CHANNELS))
        )
        joined_mask = torch.zeros(
            (batch_size, joined_length),
            dtype=torch.bool,
            device=peptide_values.device,
        )
        if partner_values.device != peptide_values.device:
            raise PairStrategyContractError(
                "single_tower_device_mismatch",
                "single_stream peptide and partner features must be on the same device",
            )
        for row in range(batch_size):
            peptide_compact = peptide_values[row, peptide_mask[row]]
            partner_compact = partner_values[row, partner_mask[row]]
            peptide_count = int(peptide_compact.shape[0])
            partner_count = int(partner_compact.shape[0])
            boundary = peptide_count
            partner_start = boundary + 1
            joined[row, :peptide_count, :source_dim] = peptide_compact
            joined[row, :peptide_count, source_dim] = 1
            joined[row, boundary, source_dim + 2] = 1
            joined[row, partner_start : partner_start + partner_count, :source_dim] = (
                partner_compact
            )
            joined[
                row,
                partner_start : partner_start + partner_count,
                source_dim + 1,
            ] = 1
            joined_mask[row, : partner_start + partner_count] = True
        ResidueModelInput = _residue_model_input_type()
        result = ResidueModelInput(joined, joined_mask)
    else:
        if separator_id is None or int(separator_id) <= 0:
            raise PairStrategyContractError(
                "single_tower_separator_missing",
                "index single_stream join requires a positive reserved separator ID",
            )
        separator_id = int(separator_id)
        if bool((peptide_values >= separator_id).any()) or bool(
            (partner_values >= separator_id).any()
        ):
            raise PairStrategyContractError(
                "single_tower_separator_collision",
                "single_stream separator ID must be outside both tower vocabularies",
            )
        if partner_values.device != peptide_values.device:
            raise PairStrategyContractError(
                "single_tower_device_mismatch",
                "single_stream peptide and partner token IDs must be on the same device",
            )
        joined = peptide_values.new_zeros((batch_size, joined_length))
        for row in range(batch_size):
            peptide_compact = peptide_values[row, peptide_mask[row]]
            partner_compact = partner_values[row, partner_mask[row]]
            peptide_count = int(peptide_compact.shape[0])
            partner_count = int(partner_compact.shape[0])
            boundary = peptide_count
            partner_start = boundary + 1
            joined[row, :peptide_count] = peptide_compact
            joined[row, boundary] = separator_id
            joined[row, partner_start : partner_start + partner_count] = partner_compact
        result = joined

    if batched:
        return result
    if isinstance(result, torch.Tensor):
        return result[0]
    ResidueModelInput = _residue_model_input_type()
    return ResidueModelInput(result.features[0], result.mask[0])


def join_single_tower_rows(peptide, partner, *, separator_id: int | None = None):
    return _join_single_tower_sequences(
        peptide,
        partner,
        separator_id=separator_id,
        batched=False,
    )


def join_single_tower_batches(peptide, partner, *, separator_id: int | None = None):
    return _join_single_tower_sequences(
        peptide,
        partner,
        separator_id=separator_id,
        batched=True,
    )


def build_single_tower_strategy_contract(
    joined,
    *,
    separator_id: int | None,
    source_feature_dim: int | None,
) -> dict[str, Any]:
    ResidueModelInput = _residue_model_input_type()
    values = joined.features if isinstance(joined, ResidueModelInput) else joined
    if not isinstance(values, torch.Tensor) or values.ndim not in {2, 3}:
        raise PairStrategyContractError(
            "single_tower_joined_input_invalid",
            "joined single_stream input must be a batched index or feature tensor",
        )
    feature_input = values.ndim == 3
    return {
        "contract_version": PAIR_STRATEGY_CONTRACT_VERSION,
        "strategy": "single_tower",
        "role_order": list(PAIR_ROLE_ORDER),
        "tower_mode": "homogeneous",
        "tower_processing": "shared_encoder",
        "interaction_level": "joint_sequence",
        "input_adapter": SINGLE_TOWER_JOIN_MODE,
        "sequence_join": {
            "contract_version": SINGLE_TOWER_JOIN_CONTRACT_VERSION,
            "layout": SINGLE_TOWER_LAYOUT,
            "input_kind": "feature" if feature_input else "index",
            "index_separator_id": None if feature_input else int(separator_id or 0),
            "feature_role_channels": list(FEATURE_ROLE_CHANNELS) if feature_input else [],
            "source_feature_dim": int(source_feature_dim or 0) if feature_input else 0,
            "joined_feature_dim": int(values.shape[2]) if feature_input else 0,
            "joined_length": int(values.shape[1]),
        },
    }


def validate_single_tower_strategy_contract(
    contract: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if contract is None:
        return None
    if not isinstance(contract, Mapping):
        raise PairStrategyContractError(
            "pair_strategy_contract_invalid",
            "pair_strategy_contract must be an object",
        )
    normalized = dict(contract)
    join = normalized.get("sequence_join")
    valid = (
        normalized.get("contract_version") == PAIR_STRATEGY_CONTRACT_VERSION
        and normalized.get("strategy") == "single_tower"
        and normalized.get("role_order") == list(PAIR_ROLE_ORDER)
        and normalized.get("tower_mode") == "homogeneous"
        and normalized.get("tower_processing") == "shared_encoder"
        and normalized.get("interaction_level") == "joint_sequence"
        and normalized.get("input_adapter") == SINGLE_TOWER_JOIN_MODE
        and isinstance(join, Mapping)
        and join.get("contract_version") == SINGLE_TOWER_JOIN_CONTRACT_VERSION
        and join.get("layout") == SINGLE_TOWER_LAYOUT
        and join.get("input_kind") in {"index", "feature"}
        and int(join.get("joined_length") or 0) > 0
    )
    if not valid:
        raise PairStrategyContractError(
            "pair_strategy_contract_invalid",
            "unsupported or inconsistent Single-stream strategy contract",
        )
    if join["input_kind"] == "index":
        if int(join.get("index_separator_id") or 0) <= 0:
            raise PairStrategyContractError(
                "pair_strategy_contract_invalid",
                "Single-stream index strategy contract requires a separator ID",
            )
    elif (
        join.get("feature_role_channels") != list(FEATURE_ROLE_CHANNELS)
        or int(join.get("source_feature_dim") or 0) <= 0
        or int(join.get("joined_feature_dim") or 0)
        != int(join.get("source_feature_dim") or 0) + len(FEATURE_ROLE_CHANNELS)
    ):
        raise PairStrategyContractError(
            "pair_strategy_contract_invalid",
            "Single-stream feature strategy contract has invalid role channels or widths",
        )
    return normalized
