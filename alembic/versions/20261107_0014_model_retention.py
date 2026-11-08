"""Add model lifetime controls without altering scientific records.

Revision ID: 20261107_0014
Revises: 20261107_0013
"""

import json

import sqlalchemy as sa

from alembic import op

revision = "20261107_0014"
down_revision = "20261107_0013"
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    if not sa.inspect(conn).has_table("model_retention_states"):
        op.create_table(
            "model_retention_states",
            sa.Column("model_id", sa.Text(), primary_key=True),
            sa.Column("experiment_id", sa.Text(), nullable=False),
            sa.Column("owner_user_id", sa.Text()),
            sa.Column("state", sa.Text(), nullable=False),
            sa.Column("notified_at", sa.Text()),
            sa.Column("delete_after", sa.Text()),
            sa.Column("removed_at", sa.Text()),
            sa.Column("reason", sa.Text()),
            sa.Column("manifest_json", sa.Text(), nullable=False),
            sa.Column("error", sa.Text()),
            sa.Column("updated_at", sa.Text(), nullable=False),
        )
        op.create_index(
            "ix_model_retention_owner_state", "model_retention_states", ["owner_user_id", "state"]
        )
    if not sa.inspect(conn).has_table("retired_model_versions"):
        op.create_table(
            "retired_model_versions",
            sa.Column("version_id", sa.Text(), primary_key=True),
            sa.Column("model_id", sa.Text(), nullable=False),
            sa.Column("removed_at", sa.Text(), nullable=False),
        )
    policies = sa.table(
        "platform_resource_policies",
        sa.column("id", sa.Text()),
        sa.column("config_json", sa.Text()),
    )
    settings = sa.table(
        "user_resource_settings",
        sa.column("user_id", sa.Text()),
        sa.column("overrides_json", sa.Text()),
    )
    row = conn.execute(sa.select(policies).where(policies.c.id == "resources")).mappings().first()
    if row:
        value = json.loads(row["config_json"])
        defaults = value.setdefault("defaults", {})
        defaults.setdefault("max_model_count", max(5, defaults.get("protected_model_limit", 3)))
        value.setdefault("retention", {}).setdefault("delete_unprotected_models", False)
        conn.execute(
            policies.update()
            .where(policies.c.id == "resources")
            .values(config_json=json.dumps(value))
        )
        for setting in conn.execute(sa.select(settings)).mappings().all():
            overrides = json.loads(setting["overrides_json"])
            count = overrides.get("max_model_count", defaults["max_model_count"])
            protected = overrides.get(
                "protected_model_limit", defaults.get("protected_model_limit", 3)
            )
            if protected > count:
                overrides["max_model_count"] = protected
                conn.execute(
                    settings.update()
                    .where(settings.c.user_id == setting["user_id"])
                    .values(overrides_json=json.dumps(overrides))
                )


def downgrade():
    op.drop_table("retired_model_versions")
    op.drop_index("ix_model_retention_owner_state", table_name="model_retention_states")
    op.drop_table("model_retention_states")
