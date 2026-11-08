"""Preserve daily entitlements and accepted work across quota changes.

Revision ID: 20261107_0015
Revises: 20261107_0014
"""

import sqlalchemy as sa

from alembic import op

revision = "20261107_0015"
down_revision = "20261107_0014"
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    if not sa.inspect(conn).has_table("resource_daily_quota_limits"):
        op.create_table(
            "resource_daily_quota_limits",
            sa.Column("user_id", sa.Text(), primary_key=True),
            sa.Column("usage_date", sa.Text(), primary_key=True),
            sa.Column("limits_json", sa.Text(), nullable=False),
            sa.Column("updated_at", sa.Text(), nullable=False),
        )
    if not sa.inspect(conn).has_table("resource_quota_changes"):
        op.create_table(
            "resource_quota_changes",
            sa.Column("id", sa.Text(), primary_key=True),
            sa.Column("user_id", sa.Text(), nullable=False),
            sa.Column("actor_user_id", sa.Text()),
            sa.Column("source", sa.Text(), nullable=False),
            sa.Column("changes_json", sa.Text(), nullable=False),
            sa.Column("created_at", sa.Text(), nullable=False),
        )
    indices = {index["name"] for index in sa.inspect(conn).get_indexes("resource_quota_changes")}
    if "ix_resource_quota_changes_user" not in indices:
        op.create_index(
            "ix_resource_quota_changes_user", "resource_quota_changes", ["user_id", "created_at"]
        )
    for table in ("resource_pending_jobs", "resource_leases"):
        if "policy_json" not in {column["name"] for column in sa.inspect(conn).get_columns(table)}:
            op.add_column(table, sa.Column("policy_json", sa.Text()))


def downgrade():
    for table in ("resource_pending_jobs", "resource_leases"):
        with op.batch_alter_table(table) as batch:
            batch.drop_column("policy_json")
    op.drop_table("resource_quota_changes")
    op.drop_table("resource_daily_quota_limits")
