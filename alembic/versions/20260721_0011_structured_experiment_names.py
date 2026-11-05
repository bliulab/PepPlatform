"""add structured experiment names

Revision ID: 20260721_0011
Revises: 20260720_0010
Create Date: 2026-07-21
"""

import json

from alembic import op
import sqlalchemy as sa


revision = "20260721_0011"
down_revision = "20260720_0010"
branch_labels = None
depends_on = None


GENERATED_NAME_QUALIFIERS = {"conservative", "balanced", "exploratory"}


def _generated_records(actions, execution):
    actions = actions if isinstance(actions, list) else []
    execution = execution if isinstance(execution, dict) else {}
    execution_ids = {}
    for entry in execution.get("created_experiments") or []:
        if not isinstance(entry, dict):
            continue
        action_id = str(entry.get("action_id") or "").strip()
        experiment = entry.get("experiment") if isinstance(entry.get("experiment"), dict) else {}
        experiment_id = str(entry.get("experiment_id") or experiment.get("id") or "").strip()
        if action_id and experiment_id:
            execution_ids[action_id] = experiment_id

    records = {}
    for action in actions:
        if not isinstance(action, dict) or str(action.get("type") or "") != "create_experiment":
            continue
        action_id = str(action.get("id") or "").strip()
        result = action.get("result") if isinstance(action.get("result"), dict) else {}
        experiment_id = str(
            result.get("experiment_id") or execution_ids.get(action_id) or ""
        ).strip()
        title = str(action.get("title") or "").strip()
        lane = str(action.get("lane") or "").strip()
        if not experiment_id or not title:
            continue
        names = {title}
        if lane:
            names.add(f"Agent {lane}: {title}"[:180])
        records[experiment_id] = {
            "names": names,
            "qualifier": lane if lane in GENERATED_NAME_QUALIFIERS else None,
        }
    return records


def upgrade():
    op.add_column("experiments", sa.Column("name_base_kind", sa.Text()))
    op.add_column("experiments", sa.Column("name_qualifier", sa.Text()))

    experiments_table = sa.table(
        "experiments",
        sa.column("id", sa.Text()),
        sa.column("name", sa.Text()),
        sa.column("name_kind", sa.Text()),
        sa.column("name_base", sa.Text()),
        sa.column("name_base_kind", sa.Text()),
        sa.column("name_qualifier", sa.Text()),
    )
    plans_table = sa.table(
        "agent_execution_plans",
        sa.column("actions_json", sa.Text()),
        sa.column("execution_json", sa.Text()),
    )
    connection = op.get_bind()
    connection.execute(
        experiments_table.update()
        .where(
            sa.and_(
                experiments_table.c.name_kind == "rerun",
                experiments_table.c.name_base.is_not(None),
            )
        )
        .values(name_base_kind="literal")
    )

    records = {}
    for row in connection.execute(
        sa.select(plans_table.c.actions_json, plans_table.c.execution_json)
    ).mappings():
        try:
            records.update(
                _generated_records(
                    json.loads(row["actions_json"] or "[]"),
                    json.loads(row["execution_json"] or "{}"),
                )
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

    if records:
        for row in connection.execute(
            sa.select(
                experiments_table.c.id,
                experiments_table.c.name,
                experiments_table.c.name_kind,
            ).where(experiments_table.c.id.in_(records))
        ).mappings():
            record = records[row["id"]]
            if row["name_kind"] is None and row["name"] in record["names"]:
                connection.execute(
                    experiments_table.update()
                    .where(experiments_table.c.id == row["id"])
                    .values(
                        name_kind="config",
                        name_qualifier=record["qualifier"],
                    )
                )


def downgrade():
    experiments_table = sa.table(
        "experiments",
        sa.column("name_kind", sa.Text()),
    )
    op.get_bind().execute(
        experiments_table.update()
        .where(experiments_table.c.name_kind == "config")
        .values(name_kind=None)
    )
    op.drop_column("experiments", "name_qualifier")
    op.drop_column("experiments", "name_base_kind")
