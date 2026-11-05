from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from pepplatform.objectives import ObjectiveSpec, REGISTRY, search_definition
from pepplatform.pr_metrics import binary_aupr, has_current_aupr, metric_definition


EVALUATION_PROTOCOL_VERSION = 3
EVALUATION_CHECKPOINT_METRICS = frozenset(
    {
        "validation_macro_f1",
        "validation_roc_auc",
        "validation_dual_f1_sacc",
        "validation_f1",
        "validation_sacc",
        "validation_aupr",
    }
)
EVALUATION_MODES = frozenset({"independent_iid", "inductive"})
OPERATING_POINT_OBJECTIVES = frozenset(k for k, v in REGISTRY.items() if v["requires_predictions"])
# Per-column cap for the dual-threshold sweep candidate list; see
# Evaluator._dual_thresholds for the cost rationale.
_MAX_DUAL_THRESHOLD_CANDIDATES = 32


@dataclass(frozen=True)
class OperatingPoint:
    values: tuple[float, ...]
    source: str = "validation"
    objective: str = "f1"
    score_kind: str = "probability"
    label_names: tuple[str, ...] = ()
    fallback_labels: tuple[int, ...] = ()
    protocol_version: int = EVALUATION_PROTOCOL_VERSION
    objective_definition: dict | None = None
    search_config: dict | None = None

    @property
    def is_binary(self):
        return len(self.values) == 1

    @property
    def threshold(self):
        if not self.is_binary:
            raise ValueError("multi-label operating points do not have a scalar threshold")
        return float(self.values[0])

    def to_dict(self):
        payload = asdict(self)
        for optional in ("objective_definition", "search_config"):
            if payload[optional] is None:
                payload.pop(optional)
        payload["values"] = list(self.values)
        payload["label_names"] = list(self.label_names)
        payload["fallback_labels"] = list(self.fallback_labels)
        payload["threshold_source"] = f"{self.source}_{self.objective}"
        if self.is_binary:
            payload["threshold"] = self.threshold
        else:
            payload["thresholds"] = list(self.values)
            if self.label_names and len(self.label_names) == len(self.values):
                payload["threshold_by_label"] = dict(zip(self.label_names, self.values))
        return payload

    @classmethod
    def from_dict(cls, payload):
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, dict):
            raise TypeError("operating point must be a mapping")
        values = payload.get("values", payload.get("thresholds"))
        if values is None and payload.get("threshold") is not None:
            values = [payload["threshold"]]
        if values is None:
            raise ValueError("operating point does not contain thresholds")
        return cls(
            values=tuple(float(value) for value in values),
            source=str(payload.get("source", "validation")),
            objective=str(payload.get("objective", "f1")),
            score_kind=str(payload.get("score_kind", "probability")),
            label_names=tuple(str(value) for value in payload.get("label_names", [])),
            fallback_labels=tuple(int(value) for value in payload.get("fallback_labels", [])),
            protocol_version=int(payload.get("protocol_version", EVALUATION_PROTOCOL_VERSION)),
            objective_definition=payload.get("objective_definition"),
            search_config=payload.get("search_config"),
        )


def evaluation_protocol_errors(
    payload,
    *,
    allowed_checkpoint_metrics=None,
    allowed_evaluation_modes=None,
    require_binary_operating_point=False,
    allowed_threshold_objectives=None,
):
    """Return fail-closed validation errors for saved evaluation metadata."""
    if not isinstance(payload, dict):
        return ["protocol metadata is not a mapping"]

    checkpoint_metrics = frozenset(allowed_checkpoint_metrics or EVALUATION_CHECKPOINT_METRICS)
    evaluation_modes = frozenset(allowed_evaluation_modes or EVALUATION_MODES)
    errors = []
    if payload.get("evaluation_protocol_version") != EVALUATION_PROTOCOL_VERSION:
        errors.append(f"evaluation_protocol_version is not {EVALUATION_PROTOCOL_VERSION}")
    if payload.get("checkpoint_metric") not in checkpoint_metrics:
        errors.append("checkpoint_metric is missing or not approved")
    if payload.get("feature_fit_scope") != "train_only":
        errors.append("feature_fit_scope is not 'train_only'")
    if payload.get("evaluation_mode") not in evaluation_modes:
        errors.append("evaluation_mode is missing or not approved")

    point_payload = payload.get("operating_point")
    if not isinstance(point_payload, dict):
        return errors + ["operating_point is not a mapping"]
    if "source" not in point_payload:
        errors.append("operating_point.source is missing")
    if "objective" not in point_payload:
        errors.append("operating_point.objective is missing")
    try:
        operating_point = OperatingPoint.from_dict(point_payload)
    except (TypeError, ValueError, OverflowError) as exc:
        return errors + [f"invalid operating_point: {exc}"]

    if operating_point.source != "validation":
        errors.append("operating_point.source is not 'validation'")
    if operating_point.objective not in OPERATING_POINT_OBJECTIVES:
        errors.append("operating_point.objective is not approved")
    if (
        allowed_threshold_objectives is not None
        and operating_point.objective not in allowed_threshold_objectives
    ):
        errors.append("operating_point.objective is not approved for this experiment")
    if operating_point.objective_definition is not None:
        try:
            spec = ObjectiveSpec.parse(
                operating_point.objective_definition, multilabel=not operating_point.is_binary
            ).validate_threshold()
            if spec.id != operating_point.objective:
                errors.append("operating_point objective definition mismatch")
        except (ValueError, TypeError) as exc:
            errors.append(f"invalid operating_point objective definition: {exc}")
    if not operating_point.values:
        errors.append("operating_point does not contain thresholds")
    elif not np.all(np.isfinite(np.asarray(operating_point.values, dtype=np.float64))):
        errors.append("operating_point contains a non-finite threshold")
    if require_binary_operating_point and not operating_point.is_binary:
        errors.append("operating_point is not a single binary threshold")
    if operating_point.is_binary:
        try:
            serialized_threshold = float(point_payload["threshold"])
        except (KeyError, TypeError, ValueError, OverflowError):
            errors.append("operating_point.threshold is missing or invalid")
        else:
            if not np.isfinite(serialized_threshold):
                errors.append("operating_point.threshold is not finite")
            elif not np.isclose(
                serialized_threshold,
                operating_point.threshold,
                rtol=1e-12,
                atol=1e-12,
            ):
                errors.append("operating_point.threshold does not match values")
    else:
        serialized_thresholds = point_payload.get("thresholds")
        try:
            thresholds = tuple(float(value) for value in serialized_thresholds)
        except (TypeError, ValueError, OverflowError):
            errors.append("operating_point.thresholds is missing or invalid")
        else:
            if len(thresholds) != len(operating_point.values) or not np.allclose(
                thresholds,
                operating_point.values,
                rtol=1e-12,
                atol=1e-12,
            ):
                errors.append("operating_point.thresholds does not match values")
    if operating_point.protocol_version != EVALUATION_PROTOCOL_VERSION:
        errors.append(f"operating_point.protocol_version is not {EVALUATION_PROTOCOL_VERSION}")
    return errors


class Evaluator:
    EXTRA_METRIC_KEYS = ["balanced_accuracy", "precision", "recall", "specificity", "mcc"]

    @staticmethod
    def _as_label_matrix(values):
        labels = np.asarray(values).astype(np.int32)
        if labels.ndim == 1:
            labels = labels[:, None]
        return labels

    @staticmethod
    def _as_score_matrix(scores):
        matrix = np.asarray(scores, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        return matrix

    @staticmethod
    def _validate_shapes(y_true, scores):
        y_true = Evaluator._as_label_matrix(y_true)
        scores = Evaluator._as_score_matrix(scores)
        if y_true.shape != scores.shape:
            raise ValueError(f"label/score shape mismatch: {y_true.shape} != {scores.shape}")
        return y_true, scores

    @staticmethod
    def _default_threshold(score_kind):
        return 0.0 if str(score_kind).lower() in {"margin", "decision_function"} else 0.5

    @staticmethod
    def _best_f1_threshold(y_true, scores, default_threshold):
        y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        finite = np.isfinite(scores)
        y_true, scores = y_true[finite], scores[finite]
        if y_true.size == 0 or np.unique(y_true).size < 2:
            return float(default_threshold), True
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        if thresholds.size == 0:
            return float(default_threshold), True
        denominator = precision[:-1] + recall[:-1]
        f1_values = np.divide(
            2.0 * precision[:-1] * recall[:-1],
            denominator,
            out=np.zeros_like(denominator, dtype=np.float64),
            where=denominator > 0,
        )
        best = np.nanmax(f1_values)
        tied = np.flatnonzero(np.isclose(f1_values, best, rtol=1e-12, atol=1e-12))
        if tied.size == 0:
            return float(default_threshold), True
        return float(np.max(thresholds[tied])), False

    @staticmethod
    def dual_primary_key(f1, sacc, *, reference_f1, reference_sacc):
        """Return the deterministic maximin key for the two co-primary metrics."""
        spec = ObjectiveSpec.parse(
            "dual_f1_sacc", {"reference_f1": reference_f1, "reference_sacc": reference_sacc}
        )
        return spec.key({"f1": f1, "sacc": sacc})

    @staticmethod
    def dual_primary_score(f1, sacc, *, reference_f1, reference_sacc):
        return float(
            Evaluator.dual_primary_key(
                f1, sacc, reference_f1=reference_f1, reference_sacc=reference_sacc
            )[0]
        )

    @staticmethod
    def _macro_f1_vectorized(y_true, y_pred):
        """Macro-averaged F1 matching sklearn f1_score(average='macro', zero_division=0).

        The dual-threshold sweep calls this metric once per candidate threshold,
        so a vectorized implementation (instead of a per-call sklearn scoring
        round trip) keeps validation tractable for rich score columns.
        """
        y_true = np.asarray(y_true, dtype=np.int32)
        y_pred = np.asarray(y_pred, dtype=np.int32)
        tp = (y_true & y_pred).sum(axis=0).astype(np.float64)
        fp = (y_pred & ~y_true).sum(axis=0).astype(np.float64)
        fn = (y_true & ~y_pred).sum(axis=0).astype(np.float64)
        denominator = 2 * tp + fp + fn
        per_label = np.divide(
            2 * tp,
            denominator,
            out=np.zeros_like(tp),
            where=denominator > 0,
        )
        return float(per_label.mean())

    @staticmethod
    def _threshold_metric_pair(y_true, scores, thresholds):
        pred = (scores >= np.asarray(thresholds, dtype=np.float64)[None, :]).astype(np.int32)
        return (
            Evaluator._macro_f1_vectorized(y_true, pred),
            Evaluator._sample_accuracy(y_true, pred),
        )

    @staticmethod
    def _dual_thresholds(
        y_true,
        scores,
        initial,
        default_threshold,
        max_sweeps,
        *,
        objective,
        search_profile="coordinate_quantile32_v1",
    ):
        objective = ObjectiveSpec.parse(objective, multilabel=True).validate_threshold()
        thresholds = np.asarray(initial, dtype=np.float64).copy()
        candidates = []
        for column in range(y_true.shape[1]):
            finite = scores[np.isfinite(scores[:, column]), column]
            values = np.unique(
                np.concatenate([finite, np.asarray([default_threshold, thresholds[column]])])
            )
            # Cap the per-column candidate count.  A rich score column carries
            # O(val_rows) distinct values and the dual sweep evaluates every
            # candidate with a full macro-F1 pass, so columns x candidates x
            # sweeps can otherwise make validation prohibitively slow.  Columns
            # with at most _MAX_DUAL_THRESHOLD_CANDIDATES distinct values keep
            # the exact same search as before; denser columns fall back to
            # evenly spaced quantile candidates (plus the default and current
            # thresholds), bounding the sweep cost.
            if (
                search_profile != "coordinate_all_v1"
                and values.size > _MAX_DUAL_THRESHOLD_CANDIDATES
            ):
                quantiles = np.unique(
                    np.quantile(
                        finite,
                        np.linspace(
                            0.0,
                            1.0,
                            _MAX_DUAL_THRESHOLD_CANDIDATES - 2,
                        ),
                    )
                )
                values = np.unique(
                    np.concatenate(
                        [
                            quantiles,
                            np.asarray([default_threshold, thresholds[column]]),
                        ]
                    )
                )
            candidates.append(values)

        current_f1, current_sacc = Evaluator._threshold_metric_pair(y_true, scores, thresholds)
        current_key = objective.key({"f1": current_f1, "sacc": current_sacc})
        for _sweep in range(max(1, int(max_sweeps))):
            changed = False
            for column, values in enumerate(candidates):
                selected = float(thresholds[column])
                selected_key = (*current_key, selected)
                for value in values:
                    trial = thresholds.copy()
                    trial[column] = float(value)
                    f1, sacc = Evaluator._threshold_metric_pair(y_true, scores, trial)
                    key = (*objective.key({"f1": f1, "sacc": sacc}), float(value))
                    if key > selected_key:
                        selected = float(value)
                        selected_key = key
                if selected != thresholds[column]:
                    thresholds[column] = selected
                    current_key = selected_key[:-1]
                    changed = True
            if not changed:
                break
        return tuple(float(value) for value in thresholds)

    @staticmethod
    def fit_operating_point(
        y_true,
        scores,
        *,
        score_kind="probability",
        label_names=None,
        objective="f1",
        max_sweeps=5,
        objective_params=None,
        search_profile=None,
    ):
        y_true, scores = Evaluator._validate_shapes(y_true, scores)
        spec = ObjectiveSpec.parse(objective, objective_params, multilabel=y_true.shape[1] > 1)
        spec.validate_threshold()
        search_profile = search_profile or (
            "coordinate_quantile32_v1" if spec.id in {"dual_f1_sacc", "sacc"} else "f1_exact_v1"
        )
        search_config = search_definition(search_profile, max_sweeps)
        if spec.id in {"sacc", "dual_f1_sacc"} and search_profile == "f1_exact_v1":
            raise ValueError("f1_exact_v1 cannot optimize SAcc or dual")
        default_threshold = Evaluator._default_threshold(score_kind)
        thresholds = []
        fallback_labels = []
        for column in range(y_true.shape[1]):
            threshold, used_fallback = Evaluator._best_f1_threshold(
                y_true[:, column], scores[:, column], default_threshold
            )
            thresholds.append(threshold)
            if used_fallback:
                fallback_labels.append(column)
        if spec.id in {"dual_f1_sacc", "sacc"} and y_true.shape[1] > 1:
            thresholds = Evaluator._dual_thresholds(
                y_true,
                scores,
                thresholds,
                default_threshold,
                max_sweeps,
                objective=spec,
                search_profile=search_profile,
            )
        return OperatingPoint(
            values=tuple(thresholds),
            source="validation",
            objective=spec.id,
            score_kind=str(score_kind),
            label_names=tuple(str(value) for value in (label_names or [])),
            fallback_labels=tuple(fallback_labels),
            objective_definition=spec.to_dict(),
            search_config=search_config,
        )

    @staticmethod
    def _sample_accuracy(y_true, y_pred):
        inter = (y_true.astype(np.int32) & y_pred.astype(np.int32)).sum(axis=1)
        union = (y_true.astype(np.int32) | y_pred.astype(np.int32)).sum(axis=1)
        return float(np.mean(inter / np.maximum(union, 1)))

    @staticmethod
    def _binary_rates(y_true, y_pred):
        y_true = np.asarray(y_true).astype(np.int32).reshape(-1)
        y_pred = np.asarray(y_pred).astype(np.int32).reshape(-1)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        return {
            "recall": float(recall),
            "specificity": float(specificity),
            "balanced_accuracy": float((recall + specificity) / 2.0),
        }

    @staticmethod
    def _safe_metric(fn, *args, **kwargs):
        try:
            value = float(fn(*args, **kwargs))
            return value if np.isfinite(value) else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _ranking_metrics(y_true, scores):
        num_labels = y_true.shape[1]
        auc_by_label = []
        aupr_by_label = []
        for column in range(num_labels):
            target = y_true[:, column]
            if np.unique(target).size < 2:
                auc_by_label.append(None)
                aupr_by_label.append(None)
                continue
            auc_by_label.append(Evaluator._safe_metric(roc_auc_score, target, scores[:, column]))
            aupr_by_label.append(
                Evaluator._safe_metric(binary_aupr, target, scores[:, column])
            )
        valid_auc = [value for value in auc_by_label if value is not None]
        valid_aupr = [value for value in aupr_by_label if value is not None]
        flat_y, flat_scores = y_true.reshape(-1), scores.reshape(-1)
        auc_micro = None
        aupr_micro = None
        if np.unique(flat_y).size >= 2:
            auc_micro = Evaluator._safe_metric(roc_auc_score, flat_y, flat_scores)
            aupr_micro = Evaluator._safe_metric(binary_aupr, flat_y, flat_scores)
        auc_macro = float(np.mean(valid_auc)) if valid_auc else None
        aupr_macro = float(np.mean(valid_aupr)) if valid_aupr else None
        if num_labels == 1:
            primary_auc, primary_aupr, averaging = auc_by_label[0], aupr_by_label[0], "binary"
        else:
            primary_auc, primary_aupr, averaging = auc_macro, aupr_macro, "macro"
        return {
            **metric_definition(),
            "auc": primary_auc,
            "aupr": primary_aupr,
            "auc_macro": auc_macro,
            "auc_micro": auc_micro,
            "aupr_macro": aupr_macro,
            "aupr_micro": aupr_micro,
            "auc_by_label": auc_by_label,
            "aupr_by_label": aupr_by_label,
            "valid_auc_labels": len(valid_auc),
            "valid_aupr_labels": len(valid_aupr),
            "averaging": averaging,
        }

    @staticmethod
    def ranking_metrics(y_true, scores):
        """Compute threshold-independent ranking metrics only."""
        y_true, scores = Evaluator._validate_shapes(y_true, scores)
        return Evaluator._ranking_metrics(y_true, scores)

    @staticmethod
    def score_metrics(y_true, scores, operating_point):
        y_true, scores = Evaluator._validate_shapes(y_true, scores)
        operating_point = OperatingPoint.from_dict(operating_point)
        definition = operating_point.objective_definition
        objective = None
        if definition is not None:
            objective = ObjectiveSpec.parse(
                definition, multilabel=y_true.shape[1] > 1
            ).validate_threshold()
            if objective.id != operating_point.objective:
                raise ValueError("operating point objective definition mismatch")
        elif operating_point.objective == "dual_f1_sacc":
            raise ValueError(
                "dual_f1_sacc metrics require an explicit objective definition with reference parameters"
            )
        if len(operating_point.values) != y_true.shape[1]:
            raise ValueError(
                f"operating point has {len(operating_point.values)} thresholds for {y_true.shape[1]} labels"
            )
        thresholds = np.asarray(operating_point.values, dtype=np.float64)[None, :]
        pred = (scores >= thresholds).astype(np.int32)
        num_labels = y_true.shape[1]
        if y_true.shape[0] == 0:
            return {
                **Evaluator._empty_metrics(),
                "operating_point": operating_point.to_dict(),
            }
        ranking = Evaluator._ranking_metrics(y_true, scores)
        if num_labels == 1:
            yt, yp = y_true.reshape(-1), pred.reshape(-1)
            rates = Evaluator._binary_rates(yt, yp)
            metrics = {
                "acc": float(accuracy_score(yt, yp)),
                "sacc": float(accuracy_score(yt, yp)),
                "f1": float(f1_score(yt, yp, average="binary", zero_division=0)),
                "balanced_accuracy": rates["balanced_accuracy"],
                "precision": float(precision_score(yt, yp, average="binary", zero_division=0)),
                "recall": rates["recall"],
                "specificity": rates["specificity"],
                "mcc": float(matthews_corrcoef(yt, yp)),
            }
        else:
            rates = [Evaluator._binary_rates(y_true[:, c], pred[:, c]) for c in range(num_labels)]
            metrics = {
                "acc": float(accuracy_score(y_true, pred)),
                "sacc": Evaluator._sample_accuracy(y_true, pred),
                "f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
                "balanced_accuracy": float(np.mean([item["balanced_accuracy"] for item in rates])),
                "precision": float(precision_score(y_true, pred, average="macro", zero_division=0)),
                "recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
                "specificity": float(np.mean([item["specificity"] for item in rates])),
                "mcc": float(
                    np.mean(
                        [matthews_corrcoef(y_true[:, c], pred[:, c]) for c in range(num_labels)]
                    )
                ),
            }
        return {
            **metrics,
            **ranking,
            "dual_score": objective.key(metrics)[0]
            if objective is not None and objective.id == "dual_f1_sacc"
            else None,
            "operating_point": operating_point.to_dict(),
            "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        }

    @staticmethod
    def fit_and_score(
        y_true,
        scores,
        *,
        score_kind="probability",
        label_names=None,
        objective="f1",
        objective_params=None,
        search_profile=None,
    ):
        operating_point = Evaluator.fit_operating_point(
            y_true,
            scores,
            score_kind=score_kind,
            label_names=label_names,
            objective=objective,
            objective_params=objective_params,
            search_profile=search_profile,
        )
        return Evaluator.score_metrics(y_true, scores, operating_point), operating_point

    @staticmethod
    def _empty_metrics():
        return {
            **metric_definition(),
            "acc": 0.0,
            "sacc": 0.0,
            "f1": 0.0,
            "auc": None,
            "aupr": None,
            "auc_macro": None,
            "auc_micro": None,
            "aupr_macro": None,
            "aupr_micro": None,
            "balanced_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "mcc": 0.0,
            "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        }

    @staticmethod
    def nn_score_metrics(targets, scores, operating_point=None):
        if operating_point is None:
            metrics, _ = Evaluator.fit_and_score(targets, scores, score_kind="probability")
            return metrics
        return Evaluator.score_metrics(targets, scores, operating_point)

    @staticmethod
    def svm_score_metrics(scores, y, operating_point=None, score_kind="margin"):
        if operating_point is None:
            metrics, _ = Evaluator.fit_and_score(y, scores, score_kind=score_kind)
            return metrics
        return Evaluator.score_metrics(y, scores, operating_point)

    @staticmethod
    def binary_score_metrics(y_true, scores, threshold=0.5):
        point = OperatingPoint(values=(float(threshold),), source="provided", objective="f1")
        return Evaluator.score_metrics(y_true, scores, point)

    @staticmethod
    def format_extra_metrics(metrics, prefix=""):
        label_prefix = f"{prefix} " if prefix else ""
        message = (
            f" | {label_prefix}Balanced Accuracy: {metrics['balanced_accuracy']:.4f}"
            f" | {label_prefix}Precision: {metrics['precision']:.4f}"
            f" | {label_prefix}Recall: {metrics['recall']:.4f}"
            f" | {label_prefix}Specificity: {metrics['specificity']:.4f}"
            f" | {label_prefix}MCC: {metrics['mcc']:.4f}"
        )
        if metrics.get("averaging") == "macro":
            if metrics.get("auc_micro") is not None:
                message += f" | {label_prefix}Micro AUC: {metrics['auc_micro']:.4f}"
            if metrics.get("aupr_micro") is not None:
                message += f" | {label_prefix}Micro AUPR: {metrics['aupr_micro']:.4f}"
        if has_current_aupr(metrics):
            message += " | AUPRMethod: trapezoidal_pr_auc | MetricNamespace: 2"
        return message

    @staticmethod
    def collect_nn_scores(model, loader, device):
        model.eval()
        all_scores, all_targets = [], []
        with torch.no_grad():
            for inputs, targets in loader:
                outputs = torch.sigmoid(model(inputs.to(device)))
                all_scores.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        if not all_scores:
            return np.empty((0, 1)), np.empty((0, 1))
        return np.vstack(all_targets), np.vstack(all_scores)

    @staticmethod
    def eval_nn_metrics(model, loader, device, operating_point=None):
        targets, scores = Evaluator.collect_nn_scores(model, loader, device)
        if targets.shape[0] == 0:
            return Evaluator._empty_metrics()
        return Evaluator.nn_score_metrics(targets, scores, operating_point=operating_point)

    @staticmethod
    def eval_nn(model, loader, device, operating_point=None):
        metrics = Evaluator.eval_nn_metrics(model, loader, device, operating_point=operating_point)
        return metrics["acc"], metrics["sacc"], metrics["f1"], metrics["auc"]

    @staticmethod
    def eval_svm(scores, y, operating_point=None):
        metrics = Evaluator.svm_score_metrics(scores, y, operating_point=operating_point)
        return metrics["acc"], metrics["sacc"], metrics["f1"], metrics["auc"]
