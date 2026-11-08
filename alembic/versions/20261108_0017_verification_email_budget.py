"""Persist the shared daily account-code email attempt budget.

Revision ID: 20261108_0017
Revises: 20261107_0016
"""

import sqlalchemy as sa

from alembic import op

revision = "20261108_0017"
down_revision = "20261107_0016"
branch_labels = None
depends_on = None


def upgrade():
    if not sa.inspect(op.get_bind()).has_table("verification_email_daily_usage"):
        op.create_table(
            "verification_email_daily_usage",
            sa.Column("usage_date", sa.Text(), primary_key=True),
            sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("updated_at", sa.Text(), nullable=False),
        )


def downgrade():
    op.drop_table("verification_email_daily_usage")
