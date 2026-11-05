import argparse
import sqlite3
from pathlib import Path

from sqlalchemy import insert, select

from pepplatform.web.auth import hash_password
from pepplatform.web.store import (
    MetadataStore,
    datasets,
    experiments,
    inference_runs,
    models,
)


def _rows(conn, table):
    try:
        return list(conn.execute(f"SELECT * FROM {table}").fetchall())
    except sqlite3.OperationalError:
        return []


def _exists(target, table, record_id):
    with target.connect() as conn:
        return conn.execute(select(table.c.id).where(table.c.id == record_id)).first() is not None


def _insert_if_missing(target, table, payload):
    if _exists(target, table, payload["id"]):
        return False
    with target.connect() as conn:
        conn.execute(insert(table).values(**payload))
    return True


def import_legacy(source_path, target_url, admin_email, admin_password):
    source_path = Path(source_path)
    if not source_path.exists():
        raise SystemExit(f"legacy sqlite database not found: {source_path}")
    if not admin_email or not admin_password:
        raise SystemExit("admin email and password are required")

    target = MetadataStore(target_url)
    admin = target.ensure_initial_admin(admin_email, hash_password(admin_password))

    source = sqlite3.connect(source_path)
    source.row_factory = sqlite3.Row

    counts = {"datasets": 0, "experiments": 0, "models": 0, "inference_runs": 0}
    for row in _rows(source, "datasets"):
        counts["datasets"] += int(
            _insert_if_missing(
                target,
                datasets,
                {
                    "id": row["id"],
                    "name": row["name"],
                    "task_type": row["task_type"],
                    "source": row["source"],
                    "format": row["format"],
                    "root_path": row["root_path"],
                    "metadata_json": row["metadata_json"],
                    "owner_user_id": admin["id"],
                    "created_at": row["created_at"],
                },
            )
        )

    for row in _rows(source, "experiments"):
        counts["experiments"] += int(
            _insert_if_missing(
                target,
                experiments,
                {
                    "id": row["id"],
                    "dataset_id": row["dataset_id"],
                    "owner_user_id": admin["id"],
                    "name": row["name"],
                    "task_type": row["task_type"],
                    "config_json": row["config_json"],
                    "status": row["status"],
                    "log_path": row["log_path"],
                    "artifact_dir": row["artifact_dir"],
                    "metrics_json": row["metrics_json"],
                    "model_id": row["model_id"],
                    "error": row["error"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "started_at": row["started_at"],
                    "finished_at": row["finished_at"],
                },
            )
        )

    for row in _rows(source, "models"):
        counts["models"] += int(
            _insert_if_missing(
                target,
                models,
                {
                    "id": row["id"],
                    "experiment_id": row["experiment_id"],
                    "owner_user_id": admin["id"],
                    "name": row["name"],
                    "task_type": row["task_type"],
                    "bundle_path": row["bundle_path"],
                    "supports_inference": row["supports_inference"],
                    "metadata_json": row["metadata_json"],
                    "created_at": row["created_at"],
                },
            )
        )

    for row in _rows(source, "inference_runs"):
        counts["inference_runs"] += int(
            _insert_if_missing(
                target,
                inference_runs,
                {
                    "id": row["id"],
                    "model_id": row["model_id"],
                    "owner_user_id": admin["id"],
                    "input_kind": row["input_kind"],
                    "input_count": row["input_count"],
                    "prediction_count": row["prediction_count"],
                    "request_json": row["request_json"],
                    "response_json": row["response_json"],
                    "created_at": row["created_at"],
                },
            )
        )

    source.close()
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Import PepPlatform legacy SQLite metadata into PostgreSQL."
    )
    parser.add_argument("--source", default=".pepplatform_web/metadata.sqlite3")
    parser.add_argument(
        "--target", required=True, help="SQLAlchemy database URL, usually postgresql+psycopg://..."
    )
    parser.add_argument("--admin-email", required=True)
    parser.add_argument("--admin-password", required=True)
    args = parser.parse_args()
    counts = import_legacy(args.source, args.target, args.admin_email, args.admin_password)
    print(counts)


if __name__ == "__main__":
    main()
