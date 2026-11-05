"""add agent campaigns

Revision ID: 20260611_0006
Revises: 20260610_0005
Create Date: 2026-06-11
"""

from alembic import op
import sqlalchemy as sa


revision = "20260611_0006"
down_revision = "20260610_0005"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "agent_campaigns",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("owner_user_id", sa.Text(), sa.ForeignKey("users.id")),
        sa.Column("task_type", sa.Text(), nullable=False),
        sa.Column("dataset_id", sa.Text(), nullable=False),
        sa.Column("objective", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("max_rounds", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("current_round_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
    )
    op.create_index("ix_agent_campaigns_owner_user_id", "agent_campaigns", ["owner_user_id"])
    op.create_index("ix_agent_campaigns_dataset_id", "agent_campaigns", ["dataset_id"])
    op.create_index("ix_agent_campaigns_status", "agent_campaigns", ["status"])

    op.add_column("agent_execution_plans", sa.Column("campaign_id", sa.Text()))
    op.add_column(
        "agent_execution_plans",
        sa.Column("round_index", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "agent_execution_plans",
        sa.Column("trace_json", sa.Text(), nullable=False, server_default="{}"),
    )
    op.create_index(
        "ix_agent_execution_plans_campaign_id", "agent_execution_plans", ["campaign_id"]
    )


def downgrade():
    op.drop_index("ix_agent_execution_plans_campaign_id", table_name="agent_execution_plans")
    op.drop_column("agent_execution_plans", "trace_json")
    op.drop_column("agent_execution_plans", "round_index")
    op.drop_column("agent_execution_plans", "campaign_id")

    op.drop_index("ix_agent_campaigns_status", table_name="agent_campaigns")
    op.drop_index("ix_agent_campaigns_dataset_id", table_name="agent_campaigns")
    op.drop_index("ix_agent_campaigns_owner_user_id", table_name="agent_campaigns")
    op.drop_table("agent_campaigns")
