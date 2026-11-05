from pepplatform.task_compat import normalize_task_type

import math
import re

from pepplatform.run_progress import parse_run_progress_line
from pepplatform.workflow import default_model_selection_objective
from pepplatform.pr_metrics import has_current_aupr, is_aupr_key, metric_definition, without_legacy_aupr


METRIC_NUMBER = r"-?\d+(?:\.\d+)?"
CLASSICAL_VALIDATION_MODELS = (
    "SVM",
    "CalibratedSVM",
    "RF",
    "ExtraTrees",
    "HistGB",
    "LogReg",
    "XGBoost",
    "LightGBM",
)
CLASSICAL_VALIDATION_MODEL_RE = "|".join(CLASSICAL_VALIDATION_MODELS)
EXTRA_VAL_METRICS_RE = (
    rf"(?:.*?(?:Val )?Balanced Accuracy: (?P<val_balanced_accuracy>{METRIC_NUMBER}))?"
    rf"(?:.*?(?:Val )?Precision: (?P<val_precision>{METRIC_NUMBER}))?"
    rf"(?:.*?(?:Val )?Recall(?: \(Sensitivity\))?: (?P<val_recall>{METRIC_NUMBER}))?"
    rf"(?:.*?(?:Val )?Specificity: (?P<val_specificity>{METRIC_NUMBER}))?"
    rf"(?:.*?(?:Val )?MCC: (?P<val_mcc>{METRIC_NUMBER}))?"
)
EXTRA_TEST_METRICS_RE = (
    rf"(?:.*?Balanced Accuracy: (?P<test_balanced_accuracy>{METRIC_NUMBER}))?"
    rf"(?:.*?Precision: (?P<test_precision>{METRIC_NUMBER}))?"
    rf"(?:.*?Recall(?: \(Sensitivity\))?: (?P<test_recall>{METRIC_NUMBER}))?"
    rf"(?:.*?Specificity: (?P<test_specificity>{METRIC_NUMBER}))?"
    rf"(?:.*?MCC: (?P<test_mcc>{METRIC_NUMBER}))?"
)

EPOCH_RE = re.compile(
    r"Epoch (?P<epoch>\d+)/(?P<total>\d+).*?"
    r"Train Loss: (?P<train_loss>-?\d+(?:\.\d+)?)"
    r"(?:.*?Val Loss: (?P<val_loss>-?\d+(?:\.\d+)?))?"
    r".*?Val Acc: (?P<val_acc>-?\d+(?:\.\d+)?)"
    r".*?Val SAcc: (?P<val_sacc>-?\d+(?:\.\d+)?)"
    r".*?Val F1: (?P<val_f1>-?\d+(?:\.\d+)?)"
    r".*?Val AUC: (?P<val_auc>-?\d+(?:\.\d+)?)"
    + EXTRA_VAL_METRICS_RE
    + r"(?:.*?Val AUPR: (?P<val_aupr>-?\d+(?:\.\d+)?))?"
)
BEST_F1_RE = re.compile(
    r"Best Val F1: (?P<best_val_f1>-?\d+(?:\.\d+)?) \(Epoch (?P<best_epoch>\d+)\)"
)
BEST_AUC_RE = re.compile(
    r"Best Val AUC: (?P<best_val_auc>-?\d+(?:\.\d+)?) \(Epoch (?P<best_epoch>\d+)\)"
)
VALIDATION_RE = re.compile(
    rf"(?P<validation_model>{CLASSICAL_VALIDATION_MODEL_RE}) Val "
    r"Acc: (?P<val_acc>-?\d+(?:\.\d+)?)"
    r".*?SAcc: (?P<val_sacc>-?\d+(?:\.\d+)?)"
    r".*?F1: (?P<val_f1>-?\d+(?:\.\d+)?)"
    r".*?AUC: (?P<val_auc>-?\d+(?:\.\d+)?)"
    + EXTRA_VAL_METRICS_RE
    + r"(?:.*?\|\s*(?:Val )?AUPR: (?P<val_aupr>-?\d+(?:\.\d+)?))?"
)
TEST_RE = re.compile(
    r"Test(?: \| TestSet: (?P<test_set>[^|]+))?.*?"
    r"(?:Loss: (?P<test_loss>-?\d+(?:\.\d+)?) \| )?"
    r"Acc: (?P<test_acc>-?\d+(?:\.\d+)?)"
    r".*?SAcc: (?P<test_sacc>-?\d+(?:\.\d+)?)"
    r".*?F1: (?P<test_f1>-?\d+(?:\.\d+)?)"
    r".*?AUC: (?P<test_auc>-?\d+(?:\.\d+)?)"
    + EXTRA_TEST_METRICS_RE
    + r"(?:.*?\|\s*AUPR: (?P<test_aupr>-?\d+(?:\.\d+)?))?"
)
PROFILE_PROGRESS_RE = re.compile(r"^\[profile-progress\]\s+(?P<fields>.+?)\s*$")
TUNING_PROGRESS_RE = re.compile(r"^\[tuning-progress\]\s+(?P<fields>.+?)\s*$")
SVM_PROGRESS_RE = re.compile(r"^\[svm-progress\]\s+(?P<fields>.+?)\s*$")


def _coerce(groups):
    result = {}
    for key, value in groups.items():
        if value in [None, ""]:
            continue
        if key in {"epoch", "total", "best_epoch"}:
            result[key] = int(value)
        elif key in {"test_set", "validation_model"}:
            result[key] = value.strip()
        else:
            result[key] = float(value)
    return result


def _parse_profile_progress_line(line):
    match = PROFILE_PROGRESS_RE.match(line)
    if not match:
        return None
    fields = dict(re.findall(r"(?P<key>[a-z_]+)=(?P<value>[^\s]+)", match.group("fields")))
    stage = fields.get("stage", "").strip()
    status = fields.get("status", "").strip()
    feature = fields.get("feature", "").strip()
    if not stage or status not in {"running", "complete"} or not feature:
        return None
    try:
        current = int(fields["current"])
        total = int(fields["total"])
    except (KeyError, TypeError, ValueError):
        return None
    if current < 0 or total < 0 or current > total:
        return None
    event = {
        "kind": "stage",
        "stage": stage,
        "status": status,
        "current": current,
        "total": total,
        "feature": feature,
    }
    elapsed = fields.get("elapsed_seconds")
    if elapsed is not None:
        try:
            elapsed_seconds = float(elapsed)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            return None
        event["elapsed_seconds"] = elapsed_seconds
    return event


def parse_metric_line(line):
    run_progress_event = parse_run_progress_line(line)
    if run_progress_event:
        return run_progress_event
    profile_event = _parse_profile_progress_line(line)
    if profile_event:
        return profile_event
    tuning_match = TUNING_PROGRESS_RE.match(line)
    if tuning_match:
        fields = dict(
            re.findall(r"(?P<key>[a-z_]+)=(?P<value>[^\s]+)", tuning_match.group("fields"))
        )
        try:
            current = int(fields["current"])
            total = int(fields["total"])
            candidate = int(fields["candidate"])
            candidates = int(fields["candidates"])
            folds = int(fields["folds"])
        except (KeyError, TypeError, ValueError):
            return None
        if total < 1 or current < 0 or current > total:
            return None
        return {
            "kind": "stage",
            "stage": "hyperparameter_tuning",
            "status": "complete" if fields.get("status") == "complete" else "running",
            "feature": fields.get("classifier", "classifier"),
            "current": current,
            "total": total,
            "candidate": candidate,
            "candidates": candidates,
            "folds": folds,
        }
    svm_match = SVM_PROGRESS_RE.match(line)
    if svm_match:
        fields = dict(
            re.findall(
                r"(?P<key>[a-z_]+)=(?P<value>[^\s]+)",
                svm_match.group("fields"),
            )
        )
        status = fields.get("status", "")
        if status not in {"running", "retrying", "complete", "failed"}:
            return None
        try:
            current = int(fields["current"])
            total = int(fields["total"])
            attempt = int(fields["attempt"])
            attempts = int(fields["attempts"])
            max_iter = int(fields["max_iter"])
        except (KeyError, TypeError, ValueError):
            return None
        if (
            current < 1
            or total < 1
            or current > total
            or attempt < 1
            or attempts < 1
            or attempt > attempts
            or max_iter < 1
        ):
            return None
        return {
            "kind": "stage",
            "stage": "svm_fit",
            "status": status,
            "feature": "svm",
            "current": current,
            "total": total,
            "attempt": attempt,
            "attempts": attempts,
            "max_iter": max_iter,
            "recovered": fields.get("recovered") == "true",
        }
    for kind, pattern in [
        ("epoch", EPOCH_RE),
        ("best", BEST_F1_RE),
        ("best", BEST_AUC_RE),
        ("validation", VALIDATION_RE),
        ("test", TEST_RE),
    ]:
        match = pattern.search(line)
        if match:
            event = {"kind": kind, **_coerce(match.groupdict())}
            prefix = "val" if kind in {"epoch", "validation"} else "test" if kind == "test" else ""
            if prefix:
                micro_auc = re.search(r"(?:Val )?Micro AUC: (?P<value>-?\d+(?:\.\d+)?)", line)
                micro_aupr = re.search(r"(?:Val )?Micro AUPR: (?P<value>-?\d+(?:\.\d+)?)", line)
                if micro_auc:
                    event[f"{prefix}_auc_micro"] = float(micro_auc.group("value"))
                if micro_aupr:
                    event[f"{prefix}_aupr_micro"] = float(micro_aupr.group("value"))
            if "AUPRMethod: trapezoidal_pr_auc" in line and "MetricNamespace: 2" in line:
                event.update(metric_definition())
            return event
    return None


def rebuild_metric_summary(events, existing_summary=None, task_type=None):
    """Build one coherent summary from parsed metric events.

    Neural validation metrics are taken from the epoch selected by the final
    ``best`` event.  When no best event has been emitted yet, the task's
    checkpoint objective is used (macro-F1 for peptide-function tasks and
    ROC-AUC for binary pair tasks).  This keeps all validation metrics tied to
    one checkpoint instead of maximizing them independently.
    """
    task_type = normalize_task_type(task_type)
    summary = without_legacy_aupr(existing_summary) if isinstance(existing_summary, dict) else {}
    metric_events = [event for event in events or [] if isinstance(event, dict)]
    # Historical log lines cannot be converted into PR areas without scores.
    # Preserve the other metrics, but never promote their old AP to AUPR.
    metric_events = [without_legacy_aupr(event) for event in metric_events]
    validation_events = [event for event in metric_events if event.get("kind") == "validation"]
    epoch_events = [event for event in metric_events if event.get("kind") == "epoch"]
    best_events = [event for event in metric_events if event.get("kind") == "best"]
    test_events = [event for event in metric_events if event.get("kind") == "test"]

    if validation_events or epoch_events:
        for key in list(summary):
            if key.startswith("val_"):
                summary.pop(key)

    if epoch_events or best_events:
        for key in ["best_epoch", "best_val_f1", "best_val_auc"]:
            summary.pop(key, None)

    for event in validation_events:
        summary.update(event)

    selection_objective = default_model_selection_objective(task_type)
    selection_key = f"val_{selection_objective}"
    selected_epoch = None
    best_event = best_events[-1] if best_events else None
    best_epoch = best_event.get("best_epoch") if best_event else None
    if best_epoch is not None:
        selected_epoch = next(
            (event for event in reversed(epoch_events) if event.get("epoch") == best_epoch),
            None,
        )
    if selected_epoch is None:
        candidates = [
            (index, event)
            for index, event in enumerate(epoch_events)
            if isinstance(event.get(selection_key), (int, float))
            and math.isfinite(float(event[selection_key]))
        ]
        if candidates:
            _, selected_epoch = min(
                candidates,
                key=lambda item: (
                    -float(item[1][selection_key]),
                    int(item[1].get("epoch", 0)),
                    item[0],
                ),
            )

    if selected_epoch is not None:
        summary.update(
            {key: value for key, value in selected_epoch.items() if key.startswith("val_")}
        )
        if best_event is None:
            summary["best_epoch"] = selected_epoch.get("epoch")
            summary[f"best_val_{selection_objective}"] = selected_epoch.get(selection_key)

    for event in best_events:
        summary.update(event)
    for event in test_events:
        summary.update(event)
    # A log replay must not erase areas already recomputed from the selected
    # model's saved scores. Historical AP events contain no replacement area.
    if has_current_aupr(existing_summary):
        validation_source = selected_epoch or (validation_events[-1] if validation_events else {})
        same_epoch = existing_summary.get("best_epoch") == summary.get("best_epoch")
        for key, value in existing_summary.items():
            if not is_aupr_key(key):
                continue
            if key.startswith("val_") and same_epoch and not has_current_aupr(validation_source):
                summary[key] = value
            elif key.startswith("test_") and not any(has_current_aupr(e) for e in test_events):
                summary[key] = value
        summary.update(metric_definition())
    if any(has_current_aupr(event) for event in metric_events):
        summary.update(metric_definition())
    if has_current_aupr(summary):
        # Historical events keep their own unavailable reason. It must not
        # describe the current summary after saved-score values are restored.
        summary.pop("aupr_unavailable_reason", None)
    return without_legacy_aupr(summary)
