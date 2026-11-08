"""initial auth and metadata schema

Revision ID: 20260605_0001
Revises:
Create Date: 2026-06-05
"""

from alembic import op
import sqlalchemy as sa


revision = "20260605_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "users",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("email", sa.Text(), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("email_verified", sa.Integer(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.UniqueConstraint("email"),
    )
    op.create_table(
        "email_verification_codes",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("purpose", sa.Text(), nullable=False),
        sa.Column("code_hash", sa.Text(), nullable=False),
        sa.Column("expires_at", sa.Text(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("consumed_at", sa.Text()),
        sa.Column("created_at", sa.Text(), nullable=False),
    )
    op.create_index(
        "ix_email_codes_user_purpose", "email_verification_codes", ["user_id", "purpose"]
    )
    op.create_table(
        "sessions",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("token_hash", sa.Text(), nullable=False),
        sa.Column("expires_at", sa.Text(), nullable=False),
        sa.Column("user_agent", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("last_seen_at", sa.Text(), nullable=False),
        sa.Column("revoked_at", sa.Text()),
        sa.UniqueConstraint("token_hash"),
    )
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])
    op.create_table(
        "datasets",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("task_type", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("format", sa.Text(), nullable=False),
        sa.Column("root_path", sa.Text(), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=False),
        sa.Column("owner_user_id", sa.Text(), sa.ForeignKey("users.id")),
        sa.Column("created_at", sa.Text(), nullable=False),
    )
    op.create_index("ix_datasets_owner_user_id", "datasets", ["owner_user_id"])
    op.create_table(
        "experiments",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("dataset_id", sa.Text(), sa.ForeignKey("datasets.id"), nullable=False),
        sa.Column("owner_user_id", sa.Text(), sa.ForeignKey("users.id")),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("task_type", sa.Text(), nullable=False),
        sa.Column("config_json", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("log_path", sa.Text(), nullable=False),
        sa.Column("artifact_dir", sa.Text(), nullable=False),
        sa.Column("metrics_json", sa.Text(), nullable=False),
        sa.Column("model_id", sa.Text()),
        sa.Column("error", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.Column("started_at", sa.Text()),
        sa.Column("finished_at", sa.Text()),
    )
    op.create_index("ix_experiments_owner_user_id", "experiments", ["owner_user_id"])
    op.create_index("ix_experiments_dataset_id", "experiments", ["dataset_id"])
    op.create_table(
        "models",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("experiment_id", sa.Text(), sa.ForeignKey("experiments.id"), nullable=False),
        sa.Column("owner_user_id", sa.Text(), sa.ForeignKey("users.id")),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("task_type", sa.Text(), nullable=False),
        sa.Column("bundle_path", sa.Text(), nullable=False),
        sa.Column("supports_inference", sa.Integer(), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
    )
    op.create_index("ix_models_owner_user_id", "models", ["owner_user_id"])
    op.create_index("ix_models_experiment_id", "models", ["experiment_id"])
    op.create_table(
        "inference_runs",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("model_id", sa.Text(), sa.ForeignKey("models.id"), nullable=False),
        sa.Column("owner_user_id", sa.Text(), sa.ForeignKey("users.id")),
        sa.Column("input_kind", sa.Text(), nullable=False),
        sa.Column("input_count", sa.Integer(), nullable=False),
        sa.Column("prediction_count", sa.Integer(), nullable=False),
        sa.Column("request_json", sa.Text(), nullable=False),
        sa.Column("response_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
    )
    op.create_index("ix_inference_runs_owner_user_id", "inference_runs", ["owner_user_id"])
    op.create_index("ix_inference_runs_model_id", "inference_runs", ["model_id"])


def downgrade():
    op.drop_index("ix_inference_runs_model_id", table_name="inference_runs")
    op.drop_index("ix_inference_runs_owner_user_id", table_name="inference_runs")
    op.drop_table("inference_runs")
    op.drop_index("ix_models_experiment_id", table_name="models")
    op.drop_index("ix_models_owner_user_id", table_name="models")
    op.drop_table("models")
    op.drop_index("ix_experiments_dataset_id", table_name="experiments")
    op.drop_index("ix_experiments_owner_user_id", table_name="experiments")
    op.drop_table("experiments")
    op.drop_index("ix_datasets_owner_user_id", table_name="datasets")
    op.drop_table("datasets")
    op.drop_index("ix_sessions_user_id", table_name="sessions")
    op.drop_table("sessions")
    op.drop_index("ix_email_codes_user_purpose", table_name="email_verification_codes")
    op.drop_table("email_verification_codes")
    op.drop_table("users")
