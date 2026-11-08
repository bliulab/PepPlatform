"""Explicit versioned entry point. Existing training entry points are untouched."""

from __future__ import annotations
from pepplatform.task_compat import normalize_task_type

import argparse
import json
from pathlib import Path
import sys
import uuid


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--runtime-root", type=Path, default=Path(".pepplatform_versioned"))
    sub = result.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument(
        "--task",
        type=normalize_task_type,
        choices=["pep_function", "pep_protein_interaction", "pep_hlai", "pep_hlaii"],
        required=True,
    )
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--data", type=Path, required=True)
    run.add_argument(
        "--format",
        choices=["sequence_dir", "pair_split_dir", "pair_csv_bundle", "builtin"],
        required=True,
    )
    run.add_argument("--source-root", type=Path)
    run.add_argument("--name", default="Versioned CLI run")
    for name in ("inspect", "verify", "export"):
        child = sub.add_parser(name)
        child.add_argument("run_id")
        if name == "export":
            child.add_argument("--output", type=Path, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    from sqlalchemy import create_engine
    from .core import VersionRepository

    runtime = args.runtime_root.resolve()
    database = runtime / "metadata.sqlite3"
    if args.action != "run" and not database.is_file():
        raise SystemExit("versioned runtime does not exist")
    runtime.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{database}")
    repository = VersionRepository(engine, runtime / "versions")
    if args.action in {"inspect", "export", "verify"}:
        run = repository.run(args.run_id)
        if run["status"] == "not_recorded":
            raise SystemExit("run is not recorded")
        if args.action == "verify":
            results = []
            for record in run["versions"]:
                try:
                    results.append(repository.verify(record["id"]))
                except (OSError, ValueError) as exc:
                    results.append({"id": record["id"], "status": "invalid", "error": str(exc)})
            from .core import now

            status = "verified" if all(x["status"] == "verified" for x in results) else "invalid"
            repository.update_run(
                args.run_id, last_verification={"status": status, "checked_at": now()}
            )
            print(json.dumps({"status": status, "versions": results}, indent=2))
            return 0 if status == "verified" else 1
        if args.action == "export":
            if args.output.exists():
                raise SystemExit("export output already exists")
            repository.export(args.run_id, args.output)
            print(args.output)
        else:
            print(json.dumps(run, indent=2, ensure_ascii=False))
        return 0
    from pepplatform.web.config import normalize_experiment_config
    from pepplatform.web.paths import RuntimePaths
    from pepplatform.web.store import MetadataStore
    from .web import VersionedExperimentRunner

    config = json.loads(args.config.read_text())
    if not isinstance(config, dict):
        raise SystemExit("config must be a JSON object")
    if config.get("web_ensemble") or config.get("ensemble_members"):
        raise SystemExit("Use Web ensemble creation to validate member access and provenance")
    config = normalize_experiment_config(args.task, config)
    root = Path(__file__).resolve().parents[2]
    paths = RuntimePaths.from_project_root(root, runtime).ensure()
    store = MetadataStore(database)
    metadata = {"source_root": str(args.source_root.resolve())} if args.source_root else {}
    dataset = store.create_dataset(
        args.data.name, args.task, "cli", args.format, args.data.resolve(), metadata
    )
    run_id = uuid.uuid4().hex
    directory = paths.experiments / run_id
    experiment = store.create_experiment(
        args.name,
        dataset["id"],
        args.task,
        config,
        directory / "train.log",
        directory / "artifact",
        experiment_id=run_id,
    )
    runner = VersionedExperimentRunner(store, paths, versions=repository)
    print(json.dumps({"run_id": run_id, "runtime_root": str(runtime)}), flush=True)
    runner._run_experiment(experiment["id"])  # Same lifecycle, synchronous; no background service.
    result = store.get_experiment(run_id)
    version_status = repository.run(run_id)["status"]
    print(
        json.dumps(
            {
                "run_id": run_id,
                "training_status": result["status"],
                "version_status": version_status,
                "error": result.get("error"),
            }
        )
    )
    return 0 if result["status"] == "succeeded" and version_status in {"sealed", "no_model"} else 1


if __name__ == "__main__":
    sys.exit(main())
