"""Versioned, declarative selection objectives. No user code is executed."""

from __future__ import annotations
from pepplatform.task_compat import normalize_task_type
from pepplatform.pr_metrics import METRIC_NAMESPACE_VERSION, metric_definition

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

SELECTION_CONTRACT_VERSION = 1
PAIR_TASKS = frozenset({"pep_protein_interaction", "pep_hlai", "pep_hlaii"})
SEARCH_PROFILES = frozenset({"f1_exact_v1", "coordinate_quantile32_v1", "coordinate_all_v1"})


class ObjectiveError(ValueError):
    pass


REGISTRY = {
    "f1": {"requires_predictions": True, "tasks": ("binary",), "metric": "f1"},
    # Macro over positive-label columns; a one-column sequence target has one term.
    "macro_f1": {"requires_predictions": True, "tasks": ("binary", "multilabel"), "metric": "f1"},
    "sacc": {"requires_predictions": True, "tasks": ("multilabel",), "metric": "sacc"},
    "auc": {"requires_predictions": False, "tasks": ("binary", "multilabel"), "metric": "auc"},
    "aupr": {"requires_predictions": False, "tasks": ("binary", "multilabel"), "metric": "aupr"},
    "dual_f1_sacc": {
        "requires_predictions": True,
        "tasks": ("multilabel",),
        "metric": "dual_score",
    },
}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class ObjectiveSpec:
    id: str
    params: tuple = ()
    version: int = 1

    def __post_init__(self):
        if self.id not in REGISTRY or self.version != 1:
            raise ObjectiveError(f"unknown objective/version: {self.id}@{self.version}")
        params = dict(self.params)
        expected = {"reference_f1", "reference_sacc"} if self.id == "dual_f1_sacc" else set()
        if set(params) != expected:
            raise ObjectiveError(f"{self.id} requires exactly these parameters: {sorted(expected)}")
        for key, value in params.items():
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ObjectiveError(f"{key} must be a finite positive number")
        object.__setattr__(self, "params", tuple(sorted((k, float(v)) for k, v in params.items())))

    @classmethod
    def parse(cls, value, params=None, *, multilabel=None):
        if isinstance(value, cls):
            spec = value
        else:
            if isinstance(value, dict):
                data = value
                spec = cls(
                    str(data.get("id", "")),
                    tuple((data.get("params") or {}).items()),
                    int(data.get("version", 1)),
                )
                expected = spec.to_dict()
                if any(k in data and data[k] != expected[k] for k in expected):
                    raise ObjectiveError("objective definition digest or capability mismatch")
            else:
                name = str(value).strip().lower()
                name = {"roc_auc": "auc", "binary_f1": "f1"}.get(name, name)
                if name == "f1" and multilabel:
                    name = "macro_f1"
                spec = cls(name, tuple(parse_params(params).items()))
        if multilabel is not None:
            kind = "multilabel" if multilabel else "binary"
            if kind not in REGISTRY[spec.id]["tasks"]:
                raise ObjectiveError(f"{spec.id} does not support {kind} targets")
        return spec

    @property
    def requires_predictions(self):
        return REGISTRY[self.id]["requires_predictions"]

    @property
    def metric(self):
        return REGISTRY[self.id]["metric"]

    @property
    def checkpoint_metric(self):
        return {
            "macro_f1": "validation_macro_f1",
            "f1": "validation_f1",
            "auc": "validation_roc_auc",
            "aupr": "validation_aupr",
            "sacc": "validation_sacc",
            "dual_f1_sacc": "validation_dual_f1_sacc",
        }[self.id]

    def validate_threshold(self):
        if not self.requires_predictions:
            raise ObjectiveError(
                f"threshold objective {self.id} requires scores, not thresholded predictions"
            )
        return self

    def key(self, metrics):
        keys = ("f1", "sacc") if self.id == "dual_f1_sacc" else (self.metric,)
        values = []
        for name in keys:
            value = metrics.get(name)
            if value is None or not math.isfinite(float(value)):
                raise ObjectiveError(f"selection objective {self.id}: metric {name} is unavailable")
            values.append(float(value))
        if self.id in {"auc", "aupr"} and metrics.get("averaging") == "macro":
            by_label = metrics.get(f"{self.id}_by_label")
            if by_label is not None and any(v is None for v in by_label):
                raise ObjectiveError(
                    f"selection objective {self.id}: a validation label lacks both classes"
                )
        if self.id == "dual_f1_sacc":
            f1, sacc = values
            p = dict(self.params)
            a, b = f1 / p["reference_f1"], sacc / p["reference_sacc"]
            return (min(a, b), a + b, f1, sacc)
        return tuple(values)

    def to_dict(self):
        definition = {
            "id": self.id,
            "version": self.version,
            "params": dict(self.params),
            "direction": "max",
            "metric_namespace_version": METRIC_NAMESPACE_VERSION if self.id == "aupr" else 1,
            "requires_predictions": self.requires_predictions,
        }
        return {**definition, "digest": hashlib.sha256(_json(definition).encode()).hexdigest()}


def parse_params(value):
    if value in (None, ""):
        return {}
    try:
        data = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError) as exc:
        raise ObjectiveError("objective parameters must be a JSON object") from exc
    if not isinstance(data, dict):
        raise ObjectiveError("objective parameters must be a JSON object")
    return data


@dataclass(frozen=True)
class SelectionConfig:
    checkpoint: ObjectiveSpec
    threshold: ObjectiveSpec
    tuning: ObjectiveSpec
    search_profile: str
    checkpoint_threshold: ObjectiveSpec
    checkpoint_search_profile: str

    def to_dict(self):
        return {
            "version": SELECTION_CONTRACT_VERSION,
            "checkpoint": self.checkpoint.to_dict(),
            "threshold": self.threshold.to_dict(),
            "tuning": self.tuning.to_dict(),
            "search": search_definition(self.search_profile),
            "checkpoint_threshold": self.checkpoint_threshold.to_dict(),
            "checkpoint_search": search_definition(self.checkpoint_search_profile),
            "checkpoint_comparison": "primary_delta_then_lexicographic_v1",
        }


def search_definition(profile, max_sweeps=5):
    if profile not in SEARCH_PROFILES:
        raise ObjectiveError(f"unknown threshold search profile: {profile}")
    if int(max_sweeps) < 1:
        raise ObjectiveError("max_sweeps must be positive")
    return {
        "profile": profile,
        "max_candidates": 32 if profile == "coordinate_quantile32_v1" else None,
        "max_sweeps": int(max_sweeps),
        "initialization": "per_label_f1",
        "prediction_rule": ">=",
        "tie_break": "larger_threshold",
        "label_order": "stored",
    }


def resolve_selection(args):
    get = args.get if isinstance(args, dict) else lambda k, default=None: getattr(args, k, default)
    multi = normalize_task_type(get("task_type", "pep_function")) not in PAIR_TASKS
    default = "macro_f1" if multi else "auc"
    checkpoint = ObjectiveSpec.parse(
        get("checkpoint_objective") or default, get("checkpoint_objective_params"), multilabel=multi
    )
    threshold = ObjectiveSpec.parse(
        get("threshold_objective") or ("macro_f1" if multi else "f1"),
        get("threshold_objective_params"),
        multilabel=multi,
    ).validate_threshold()
    tuning_id = get("tuning_objective") or get("ml_tuning_objective") or default
    if get("tuning_objective") and get("ml_tuning_objective"):
        left = ObjectiveSpec.parse(
            get("tuning_objective"), get("tuning_objective_params"), multilabel=multi
        )
        right = ObjectiveSpec.parse(
            get("ml_tuning_objective"), get("tuning_objective_params"), multilabel=multi
        )
        if left != right:
            raise ObjectiveError("tuning_objective conflicts with ml_tuning_objective")
    tuning = ObjectiveSpec.parse(tuning_id, get("tuning_objective_params"), multilabel=multi)
    profile = get("threshold_search_profile") or (
        "f1_exact_v1" if threshold.id in {"f1", "macro_f1"} else "coordinate_quantile32_v1"
    )
    search_definition(profile)
    if threshold.id in {"sacc", "dual_f1_sacc"} and profile == "f1_exact_v1":
        raise ObjectiveError("f1_exact_v1 cannot optimize SAcc or dual")
    cp_threshold = (
        ObjectiveSpec.parse(
            get("checkpoint_threshold_objective"),
            get("checkpoint_threshold_objective_params"),
            multilabel=multi,
        ).validate_threshold()
        if get("checkpoint_threshold_objective")
        else threshold
    )
    cp_profile = get("checkpoint_threshold_search_profile") or (
        profile
        if cp_threshold == threshold
        else "f1_exact_v1"
        if cp_threshold.id in {"f1", "macro_f1"}
        else "coordinate_quantile32_v1"
    )
    search_definition(cp_profile)
    if cp_profile == "f1_exact_v1" and cp_threshold.id not in {"f1", "macro_f1"}:
        raise ObjectiveError("f1_exact_v1 cannot optimize SAcc or dual at checkpoint selection")
    return SelectionConfig(checkpoint, threshold, tuning, profile, cp_threshold, cp_profile)


def selection_fields(args):
    get = args.get if isinstance(args, dict) else lambda k, default=None: getattr(args, k, default)
    contract = resolve_selection(args).to_dict()
    classical = {"rf", "svm", "calibrated_svm", "extratrees", "logreg", "histgb", "xgb", "lgb"}
    scope = (
        "encoder"
        if get("training_mode") == "export_embeddings"
        else "not_applicable"
        if get("model") in classical
        else "final_model"
    )
    contract["checkpoint_scope"] = scope
    contract["tuning_enabled"] = bool(get("ml_auto_tune", False))
    fields = {
        **metric_definition(),
        "selection_contract_version": SELECTION_CONTRACT_VERSION,
        "selection_contract": contract,
    }
    if get("_selected_epoch") is not None:
        fields["selected_epoch"] = int(get("_selected_epoch"))
    return fields


def objective_capabilities():
    return [
        {
            "id": key,
            "version": 1,
            "direction": "max",
            **value,
            "requires_both_classes": key in {"auc", "aupr"},
            "parameters": ["reference_f1", "reference_sacc"] if key == "dual_f1_sacc" else [],
        }
        for key, value in REGISTRY.items()
    ]


def add_selection_arguments(parser, *, checkpoint=True):
    if checkpoint:
        parser.add_argument("--checkpoint_objective", choices=tuple(REGISTRY), default=None)
    for stage in ("checkpoint", "threshold", "tuning"):
        if stage != "checkpoint":
            parser.add_argument(f"--{stage}_objective", choices=tuple(REGISTRY), default=None)
        parser.add_argument(f"--{stage}_objective_params", default=None)
    parser.add_argument("--threshold_search_profile", choices=sorted(SEARCH_PROFILES), default=None)
    parser.add_argument("--checkpoint_threshold_objective", choices=tuple(REGISTRY), default=None)
    parser.add_argument("--checkpoint_threshold_objective_params", default=None)
    parser.add_argument(
        "--checkpoint_threshold_search_profile", choices=sorted(SEARCH_PROFILES), default=None
    )
    parser.add_argument("--list_objectives", action="store_true")


def checkpoint_improved(key, best_key, minimum_delta=0.0):
    if not math.isfinite(float(minimum_delta)) or minimum_delta < 0:
        raise ObjectiveError("early_stop_min_delta must be finite and non-negative")
    if best_key is None:
        return True
    return key[0] > best_key[0] + minimum_delta or (
        np.isclose(key[0], best_key[0], rtol=0.0, atol=minimum_delta) and key[1:] > best_key[1:]
    )
