"""Opt-in Web classification-loss fields; absent fields keep CLI defaults."""

import math

from pepplatform.task_compat import normalize_task_type
from .config import _is_neural_route


SEQUENCE_LOSS_FIELDS = frozenset({
    "multilabel_loss",
    "multilabel_pos_weight_cap",
    "asl_gamma_pos",
    "asl_gamma_neg",
    "asl_clip",
})
PAIR_LOSS_FIELDS = frozenset({"loss_type", "pos_weight_max"})
CLASSIFICATION_LOSS_FIELDS = SEQUENCE_LOSS_FIELDS | PAIR_LOSS_FIELDS


def validate_classification_loss_config(task_type, config, add_error):
    """Validate only supplied overrides, without migrating old configs."""
    supplied = CLASSIFICATION_LOSS_FIELDS.intersection(config)
    if not supplied:
        return
    is_pair = normalize_task_type(task_type) != "pep_function"
    allowed = PAIR_LOSS_FIELDS if is_pair else SEQUENCE_LOSS_FIELDS
    if supplied - allowed:
        add_error(
            "classification_loss_task_unsupported",
            "Classification-loss fields do not match this task: "
            + ", ".join(sorted(supplied - allowed)),
        )
    if not _is_neural_route(task_type, config) or config.get("ensemble_members"):
        add_error(
            "classification_loss_route_unsupported",
            "Classification-loss selection requires a neural training stage, not "
            "a pure classical classifier or a model ensemble",
        )
    for key, choices in (
        ("multilabel_loss", {"auto", "bce", "balanced_bce", "asymmetric_focal"}),
        ("loss_type", {"bce", "wbce"}),
    ):
        if key not in supplied:
            continue
        value = str(config[key]).strip().lower()
        if value not in choices:
            add_error(
                "classification_loss_invalid",
                f"{key} must be one of: {', '.join(sorted(choices))}",
            )
        else:
            config[key] = value
    for key, minimum, maximum in (
        ("multilabel_pos_weight_cap", 1.0, None),
        ("pos_weight_max", 1.0, None),
        ("asl_gamma_pos", 0.0, None),
        ("asl_gamma_neg", 0.0, None),
        ("asl_clip", 0.0, 1.0),
    ):
        if key not in supplied:
            continue
        raw = config[key]
        try:
            if isinstance(raw, bool):
                raise ValueError
            value = float(raw)
            if not math.isfinite(value) or value < minimum:
                raise ValueError
            if maximum is not None and value >= maximum:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            bound = f" and < {maximum:g}" if maximum is not None else ""
            add_error(
                "classification_loss_parameter_invalid",
                f"{key} must be finite, >= {minimum:g}{bound}",
            )
        else:
            config[key] = value
