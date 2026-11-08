"""add locale preferences and generated-name metadata

Revision ID: 20260720_0009
Revises: 20260719_0008
Create Date: 2026-07-20
"""

from alembic import op
import sqlalchemy as sa


revision = "20260720_0009"
down_revision = "20260719_0008"
branch_labels = None
depends_on = None


RERUN_SUFFIXES = (" · rerun", " · 重跑")


def _generated_name_parts(name):
    base = str(name or "")
    count = 0
    while True:
        matched = next((suffix for suffix in RERUN_SUFFIXES if base.endswith(suffix)), None)
        if not matched:
            break
        base = base[: -len(matched)]
        count += 1
    base = base.strip()
    return (base, count) if base and count else ("", 0)


def upgrade():
    op.add_column(
        "users",
        sa.Column("preferred_locale", sa.Text(), nullable=False, server_default="en"),
    )
    op.add_column("experiments", sa.Column("name_kind", sa.Text()))
    op.add_column("experiments", sa.Column("name_base", sa.Text()))
    op.add_column(
        "experiments",
        sa.Column("rerun_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("experiments", sa.Column("rerun_source_experiment_id", sa.Text()))
    op.add_column(
        "models",
        sa.Column("name_is_custom", sa.Integer(), nullable=False, server_default="0"),
    )

    experiments_table = sa.table(
        "experiments",
        sa.column("id", sa.Text()),
        sa.column("name", sa.Text()),
        sa.column("name_kind", sa.Text()),
        sa.column("name_base", sa.Text()),
        sa.column("rerun_count", sa.Integer()),
    )
    models_table = sa.table(
        "models",
        sa.column("id", sa.Text()),
        sa.column("experiment_id", sa.Text()),
        sa.column("name", sa.Text()),
        sa.column("name_is_custom", sa.Integer()),
    )
    connection = op.get_bind()
    experiment_names = {}
    for row in connection.execute(
        sa.select(experiments_table.c.id, experiments_table.c.name)
    ).mappings():
        experiment_names[row["id"]] = row["name"]
        base, count = _generated_name_parts(row["name"])
        if count:
            connection.execute(
                experiments_table.update()
                .where(experiments_table.c.id == row["id"])
                .values(name_kind="rerun", name_base=base, rerun_count=count)
            )

    for row in connection.execute(
        sa.select(
            models_table.c.id,
            models_table.c.experiment_id,
            models_table.c.name,
        )
    ).mappings():
        inherited_name = experiment_names.get(row["experiment_id"])
        connection.execute(
            models_table.update()
            .where(models_table.c.id == row["id"])
            .values(name_is_custom=0 if row["name"] == inherited_name else 1)
        )


def downgrade():
    op.drop_column("models", "name_is_custom")
    op.drop_column("experiments", "rerun_source_experiment_id")
    op.drop_column("experiments", "rerun_count")
    op.drop_column("experiments", "name_base")
    op.drop_column("experiments", "name_kind")
    op.drop_column("users", "preferred_locale")
