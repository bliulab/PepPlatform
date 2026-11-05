"""Migrate saved Web experiment PR areas; models and sealed files stay intact."""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pepplatform.metric_migration import (  # noqa: E402
    REFERENCE_NAME, load_metric_reference, recompute_artifact, summary_from_reference, write_json_atomic,
)
from pepplatform.pr_metrics import has_current_aupr, metric_definition, without_legacy_aupr  # noqa: E402


def migrate_experiment(store, experiment_id, *, dry_run=True):
    experiment = store.get_experiment(experiment_id)
    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_id}")
    if experiment["status"] in {"queued", "running"}:
        return {"experiment_id": experiment_id, "status": "active_experiment_skipped"}
    raw_root = str(experiment.get("artifact_dir") or "").strip()
    if not raw_root or not Path(raw_root).is_dir():
        return {"experiment_id": experiment_id, "status": "scores_unavailable", "dry_run": dry_run,
                "unavailable": {"all": "artifact_directory_missing"}, "splits": []}
    root = Path(raw_root)
    reference = recompute_artifact(root)
    result = {"experiment_id": experiment_id, "dry_run": dry_run,
              "splits": list(reference["splits"]), "unavailable": reference["unavailable"]}
    if dry_run:
        return {**result, "status": "ready" if reference["splits"] else "scores_unavailable"}
    # Access the original JSON rather than the sanitized public representation.
    raw = store.raw_experiment_metrics(experiment_id)
    original = copy.deepcopy(raw)
    try:
        existing = load_metric_reference(root)
    except (OSError, ValueError):
        existing = None  # Explicit migration can rebuild an invalid derived cache.
    if existing == reference and has_current_aupr(raw) and raw.get("summary") == {
        **without_legacy_aupr(raw.get("summary") or {}), **summary_from_reference(reference)
    }:
        return {**result, "status": "already_current"}
    backup = root / "metric_migration_v2_original.json"
    if not backup.exists():
        write_json_atomic(backup, {"experiment_id": experiment_id, "metrics": original})
    write_json_atomic(root / REFERENCE_NAME, reference)
    current = without_legacy_aupr(raw)
    current.update(metric_definition())
    current["summary"] = {**(current.get("summary") or {}), **summary_from_reference(reference)}
    if not reference["unavailable"]:
        current["summary"].pop("aupr_unavailable_reason", None)
    current["metric_reference"] = str(root / REFERENCE_NAME)
    store.update_experiment(experiment_id, metrics_json=current)
    return {**result, "status": "updated" if reference["splits"] else "scores_unavailable"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--experiment-id")
    target.add_argument("--all", action="store_true")
    parser.add_argument("--database")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    from pepplatform.web.paths import RuntimePaths
    from pepplatform.web.store import MetadataStore
    store = MetadataStore(
        args.database or os.environ.get("PEP_DATABASE_URL") or RuntimePaths.from_project_root(ROOT).database
    )
    ids = [args.experiment_id] if args.experiment_id else [e["id"] for e in store.list_experiments()]
    results = []
    for identity in ids:
        try:
            results.append(migrate_experiment(store, identity, dry_run=args.dry_run))
        except Exception as exc:
            results.append({"experiment_id": identity, "status": "error", "error": str(exc)})
    report = {**metric_definition(), "experiments": results, "dry_run": args.dry_run,
              "passed": all(r["status"] != "error" for r in results)}
    if args.report:
        write_json_atomic(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
