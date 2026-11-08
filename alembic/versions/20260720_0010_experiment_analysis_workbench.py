"""add experiment analysis conversations

Revision ID: 20260720_0010
Revises: 20260720_0009
Create Date: 2026-07-20
"""

from alembic import op
import sqlalchemy as sa


revision = "20260720_0010"
down_revision = "20260720_0009"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "agent_chat_conversations",
        sa.Column("scope", sa.Text(), nullable=False, server_default="agent"),
    )
    sqlite = op.get_bind().dialect.name == "sqlite"
    experiment_column = sa.Column(
        "experiment_id",
        sa.Text(),
        sa.ForeignKey(
            "experiments.id",
            ondelete="CASCADE",
            name="fk_agent_chat_conversations_experiment_id" if sqlite else None,
        ),
    )
    if sqlite:
        # SQLite cannot add a foreign-key constraint with ALTER TABLE.
        with op.batch_alter_table("agent_chat_conversations") as batch:
            batch.add_column(experiment_column)
    else:
        op.add_column("agent_chat_conversations", experiment_column)
    op.create_index(
        "ix_agent_chat_conversations_owner_scope_experiment_updated",
        "agent_chat_conversations",
        ["owner_user_id", "scope", "experiment_id", "updated_at"],
    )
    op.add_column(
        "agent_chat_messages",
        sa.Column("kind", sa.Text(), nullable=False, server_default="chat"),
    )
    op.add_column(
        "agent_chat_messages",
        sa.Column("payload_json", sa.Text(), nullable=False, server_default="{}"),
    )
    op.add_column(
        "agent_chat_messages",
        sa.Column("context_manifest_json", sa.Text(), nullable=False, server_default="{}"),
    )


def downgrade():
    op.drop_column("agent_chat_messages", "context_manifest_json")
    op.drop_column("agent_chat_messages", "payload_json")
    op.drop_column("agent_chat_messages", "kind")
    op.drop_index(
        "ix_agent_chat_conversations_owner_scope_experiment_updated",
        table_name="agent_chat_conversations",
    )
    if op.get_bind().dialect.name == "sqlite":
        with op.batch_alter_table("agent_chat_conversations") as batch:
            batch.drop_column("experiment_id")
            batch.drop_column("scope")
    else:
        op.drop_column("agent_chat_conversations", "experiment_id")
        op.drop_column("agent_chat_conversations", "scope")
