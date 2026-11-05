"""add admin quotas and audit logs

Revision ID: 20260608_0003
Revises: 20260607_0002
Create Date: 2026-06-08
"""

from alembic import op
import sqlalchemy as sa


revision = "20260608_0003"
down_revision = "20260607_0002"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "user_quotas",
        sa.Column("user_id", sa.Text(), sa.ForeignKey("users.id"), primary_key=True),
        sa.Column("daily_experiment_limit", sa.Integer(), nullable=False, server_default="10"),
        sa.Column(
            "daily_concurrent_experiment_limit", sa.Integer(), nullable=False, server_default="1"
        ),
        sa.Column(
            "daily_inference_request_limit", sa.Integer(), nullable=False, server_default="200"
        ),
        sa.Column(
            "daily_inference_input_limit", sa.Integer(), nullable=False, server_default="10000"
        ),
        sa.Column("updated_at", sa.Text(), nullable=False),
    )
    op.create_table(
        "quota_usage",
        sa.Column("user_id", sa.Text(), sa.ForeignKey("users.id"), primary_key=True),
        sa.Column("usage_date", sa.Text(), primary_key=True),
        sa.Column("experiments_created", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("inference_requests", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("inference_inputs", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.Text(), nullable=False),
    )
    op.create_index("ix_quota_usage_date", "quota_usage", ["usage_date"])
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("actor_user_id", sa.Text(), sa.ForeignKey("users.id")),
        sa.Column("target_user_id", sa.Text(), sa.ForeignKey("users.id")),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("ip_address", sa.Text(), nullable=False, server_default=""),
        sa.Column("user_agent", sa.Text(), nullable=False, server_default=""),
        sa.Column("metadata_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
    )
    op.create_index("ix_audit_logs_actor_user_id", "audit_logs", ["actor_user_id"])
    op.create_index("ix_audit_logs_target_user_id", "audit_logs", ["target_user_id"])
    op.create_index("ix_audit_logs_event_type", "audit_logs", ["event_type"])
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"])


def downgrade():
    op.drop_index("ix_audit_logs_created_at", table_name="audit_logs")
    op.drop_index("ix_audit_logs_event_type", table_name="audit_logs")
    op.drop_index("ix_audit_logs_target_user_id", table_name="audit_logs")
    op.drop_index("ix_audit_logs_actor_user_id", table_name="audit_logs")
    op.drop_table("audit_logs")
    op.drop_index("ix_quota_usage_date", table_name="quota_usage")
    op.drop_table("quota_usage")
    op.drop_table("user_quotas")
