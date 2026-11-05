"""add user profile fields

Revision ID: 20260607_0002
Revises: 20260605_0001
Create Date: 2026-06-07
"""

from alembic import op
import sqlalchemy as sa


revision = "20260607_0002"
down_revision = "20260605_0001"
branch_labels = None
depends_on = None


def _default_display_name(email):
    prefix = str(email or "").split("@", 1)[0].strip()
    return prefix[:40] or "User"


def upgrade():
    op.add_column("users", sa.Column("display_name", sa.Text(), nullable=False, server_default=""))
    op.add_column("users", sa.Column("avatar_object_key", sa.Text()))
    op.add_column("users", sa.Column("avatar_content_type", sa.Text()))
    op.add_column("users", sa.Column("avatar_updated_at", sa.Text()))

    users_table = sa.table(
        "users",
        sa.column("id", sa.Text()),
        sa.column("email", sa.Text()),
        sa.column("display_name", sa.Text()),
    )
    connection = op.get_bind()
    rows = connection.execute(sa.select(users_table.c.id, users_table.c.email)).mappings().all()
    for row in rows:
        connection.execute(
            users_table.update()
            .where(users_table.c.id == row["id"])
            .values(display_name=_default_display_name(row["email"]))
        )


def downgrade():
    op.drop_column("users", "avatar_updated_at")
    op.drop_column("users", "avatar_content_type")
    op.drop_column("users", "avatar_object_key")
    op.drop_column("users", "display_name")
