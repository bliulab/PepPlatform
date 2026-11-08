"""add agent execution plans

Revision ID: 20260610_0005
Revises: 20260609_0004
Create Date: 2026-06-10
"""

from alembic import op
import sqlalchemy as sa


revision = "20260610_0005"
down_revision = "20260609_0004"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "agent_execution_plans",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("owner_user_id", sa.Text(), sa.ForeignKey("users.id")),
        sa.Column("provider_config_id", sa.Text()),
        sa.Column("provider", sa.Text(), nullable=False, server_default=""),
        sa.Column("model", sa.Text(), nullable=False, server_default=""),
        sa.Column("task_type", sa.Text(), nullable=False),
        sa.Column("dataset_id", sa.Text(), nullable=False),
        sa.Column("objective", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("actions_json", sa.Text(), nullable=False),
        sa.Column("execution_json", sa.Text(), nullable=False),
        sa.Column("review_json", sa.Text(), nullable=False),
        sa.Column("usage_json", sa.Text(), nullable=False),
        sa.Column("context_manifest_json", sa.Text(), nullable=False),
        sa.Column("warnings_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.Column("executed_at", sa.Text()),
        sa.Column("reviewed_at", sa.Text()),
    )
    op.create_index(
        "ix_agent_execution_plans_owner_user_id", "agent_execution_plans", ["owner_user_id"]
    )
    op.create_index("ix_agent_execution_plans_dataset_id", "agent_execution_plans", ["dataset_id"])
    op.create_index(
        "ix_agent_execution_plans_provider_config_id",
        "agent_execution_plans",
        ["provider_config_id"],
    )
    op.create_index("ix_agent_execution_plans_status", "agent_execution_plans", ["status"])


def downgrade():
    op.drop_index("ix_agent_execution_plans_status", table_name="agent_execution_plans")
    op.drop_index("ix_agent_execution_plans_provider_config_id", table_name="agent_execution_plans")
    op.drop_index("ix_agent_execution_plans_dataset_id", table_name="agent_execution_plans")
    op.drop_index("ix_agent_execution_plans_owner_user_id", table_name="agent_execution_plans")
    op.drop_table("agent_execution_plans")
