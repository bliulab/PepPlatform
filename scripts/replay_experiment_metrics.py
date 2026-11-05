import argparse
import copy
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pepplatform.web.metrics import parse_metric_line, rebuild_metric_summary
from pepplatform.web.paths import RuntimePaths
from pepplatform.web.store import MetadataStore
from pepplatform.pr_metrics import preserve_legacy_summary


VALIDATION_SUMMARY_KEYS = {
    "val_acc",
    "val_sacc",
    "val_f1",
    "val_auc",
    "val_aupr",
    "val_balanced_accuracy",
    "val_precision",
    "val_recall",
    "val_specificity",
    "val_mcc",
}


def _default_database():
    return (
        os.environ.get("PEP_DATABASE_URL") or RuntimePaths.from_project_root(PROJECT_ROOT).database
    )


def _event_signature(event):
    return json.dumps(event, sort_keys=True, separators=(",", ":"))


def _read_metric_events(log_path):
    events = []
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            event = parse_metric_line(line)
            if event:
                events.append(event)
    return events


def replay_missing_validation_metrics(store, experiment_id, force=False, dry_run=False):
    experiment = store.get_experiment(experiment_id)
    if not experiment:
        raise SystemExit(f"experiment not found: {experiment_id}")

    log_path = Path(experiment["log_path"])
    if not log_path.exists():
        return {
            "experiment_id": experiment_id,
            "status": "experiment_log_not_found",
            "parsed_validation_events": 0,
            "parsed_metric_events": 0,
            "added_validation_events": 0,
            "added_metric_events": 0,
            "dry_run": dry_run,
        }

    parsed_events = _read_metric_events(log_path)
    parsed_validation_events = [
        event for event in parsed_events if event.get("kind") in {"epoch", "validation"}
    ]
    if not parsed_validation_events:
        return {
            "experiment_id": experiment_id,
            "status": "no_validation_metrics_in_log",
            "parsed_validation_events": 0,
            "parsed_metric_events": len(parsed_events),
            "added_validation_events": 0,
            "added_metric_events": 0,
            "dry_run": dry_run,
        }

    metrics = copy.deepcopy(store.raw_experiment_metrics(experiment_id))
    if not isinstance(metrics, dict):
        metrics = {}
    events = metrics.get("events") or []
    if not isinstance(events, list):
        events = []
    summary = metrics.get("summary") or {}
    if not isinstance(summary, dict):
        summary = {}
    existing = {_event_signature(event) for event in events}
    added_events = []
    for event in parsed_events:
        signature = _event_signature(event)
        if signature not in existing:
            added_events.append(event)
            existing.add(signature)

    rebuilt_summary = rebuild_metric_summary(
        events + added_events,
        summary,
        task_type=experiment.get("task_type"),
    )
    expected_validation = {
        key: value for key, value in rebuilt_summary.items() if key in VALIDATION_SUMMARY_KEYS
    }
    validation_is_current = bool(expected_validation) and all(
        summary.get(key) == value for key, value in expected_validation.items()
    )
    if validation_is_current and not added_events and not force:
        return {
            "experiment_id": experiment_id,
            "status": "validation_summary_already_present",
            "parsed_validation_events": len(parsed_validation_events),
            "parsed_metric_events": len(parsed_events),
            "added_validation_events": 0,
            "added_metric_events": 0,
            "dry_run": dry_run,
        }

    metrics["events"] = events + added_events
    preserve_legacy_summary(metrics)
    metrics["summary"] = rebuilt_summary

    if not dry_run:
        store.update_experiment(experiment_id, metrics_json=metrics)

    return {
        "experiment_id": experiment_id,
        "status": "updated" if added_events else "summary_refreshed",
        "parsed_validation_events": len(parsed_validation_events),
        "parsed_metric_events": len(parsed_events),
        "added_validation_events": len(
            [event for event in added_events if event.get("kind") in {"epoch", "validation"}]
        ),
        "added_metric_events": len(added_events),
        "dry_run": dry_run,
    }


def replay_all_missing_validation_metrics(store, force=False, dry_run=False):
    results = []
    for experiment in store.list_experiments():
        if experiment.get("status") in {"queued", "running"}:
            results.append(
                {
                    "experiment_id": experiment["id"],
                    "status": "active_experiment_skipped",
                    "parsed_validation_events": 0,
                    "parsed_metric_events": 0,
                    "added_validation_events": 0,
                    "added_metric_events": 0,
                    "dry_run": dry_run,
                }
            )
            continue
        try:
            result = replay_missing_validation_metrics(
                store,
                experiment["id"],
                force=force,
                dry_run=dry_run,
            )
        except Exception as exc:
            result = {
                "experiment_id": experiment["id"],
                "status": "error",
                "error": str(exc),
                "parsed_validation_events": 0,
                "parsed_metric_events": 0,
                "added_validation_events": 0,
                "added_metric_events": 0,
                "dry_run": dry_run,
            }
        results.append(result)

    status_counts = {}
    for result in results:
        status = result["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "scope": "all",
        "dry_run": dry_run,
        "experiment_count": len(results),
        "status_counts": status_counts,
        "added_validation_events": sum(result["added_validation_events"] for result in results),
        "added_metric_events": sum(result["added_metric_events"] for result in results),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Replay missing validation metrics from a Web experiment log."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--experiment-id")
    target.add_argument("--all", action="store_true", dest="all_experiments")
    parser.add_argument("--database", default=_default_database())
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refresh validation summary even when val_* keys exist.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Parse and report without updating metadata."
    )
    args = parser.parse_args()

    store = MetadataStore(args.database)
    if args.all_experiments:
        result = replay_all_missing_validation_metrics(
            store,
            force=args.force,
            dry_run=args.dry_run,
        )
    else:
        result = replay_missing_validation_metrics(
            store,
            args.experiment_id,
            force=args.force,
            dry_run=args.dry_run,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
