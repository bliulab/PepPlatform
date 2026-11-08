"""add AI provider configs

Revision ID: 20260609_0004
Revises: 20260608_0003
Create Date: 2026-06-09
"""

from alembic import op
import sqlalchemy as sa


revision = "20260609_0004"
down_revision = "20260608_0003"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "ai_provider_configs",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("owner_user_id", sa.Text(), sa.ForeignKey("users.id")),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("display_name", sa.Text(), nullable=False),
        sa.Column("base_url", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("api_key_ciphertext", sa.Text(), nullable=False),
        sa.Column("api_key_last4", sa.Text(), nullable=False),
        sa.Column("enabled", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
    )
    op.create_index(
        "ix_ai_provider_configs_owner_user_id", "ai_provider_configs", ["owner_user_id"]
    )
    op.create_index("ix_ai_provider_configs_provider", "ai_provider_configs", ["provider"])


def downgrade():
    op.drop_index("ix_ai_provider_configs_provider", table_name="ai_provider_configs")
    op.drop_index("ix_ai_provider_configs_owner_user_id", table_name="ai_provider_configs")
    op.drop_table("ai_provider_configs")
