"""Separate cleanup schedules from optional user announcements.

Revision ID: 20261107_0016
Revises: 20261107_0015
"""

import sqlalchemy as sa

from alembic import op

revision = "20261107_0016"
down_revision = "20261107_0015"
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    for name in ("resource_inventory", "model_retention_states"):
        if "scheduled_at" not in {column["name"] for column in sa.inspect(conn).get_columns(name)}:
            op.add_column(name, sa.Column("scheduled_at", sa.Text()))
        conn.execute(
            sa.text(
                f"UPDATE {name} SET scheduled_at = notified_at "
                "WHERE scheduled_at IS NULL AND notified_at IS NOT NULL"
            )
        )


def downgrade():
    for name in ("resource_inventory", "model_retention_states"):
        with op.batch_alter_table(name) as batch:
            batch.drop_column("scheduled_at")
