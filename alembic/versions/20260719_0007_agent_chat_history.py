"""add agent chat history

Revision ID: 20260719_0007
Revises: 20260611_0006
Create Date: 2026-07-19
"""

from alembic import op
import sqlalchemy as sa


revision = "20260719_0007"
down_revision = "20260611_0006"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "agent_chat_conversations",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("owner_user_id", sa.Text(), sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("pinned_at", sa.Text()),
        sa.Column(
            "last_provider_config_id",
            sa.Text(),
            sa.ForeignKey("ai_provider_configs.id", ondelete="SET NULL"),
        ),
        sa.Column("last_provider", sa.Text(), nullable=False, server_default=""),
        sa.Column("last_model", sa.Text(), nullable=False, server_default=""),
        sa.Column("last_message_preview", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
    )
    op.create_index(
        "ix_agent_chat_conversations_owner_updated",
        "agent_chat_conversations",
        ["owner_user_id", "updated_at"],
    )
    op.create_index(
        "ix_agent_chat_conversations_owner_pinned",
        "agent_chat_conversations",
        ["owner_user_id", "pinned_at"],
    )

    op.create_table(
        "agent_chat_messages",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "conversation_id",
            sa.Text(),
            sa.ForeignKey("agent_chat_conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False, server_default=""),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column(
            "provider_config_id",
            sa.Text(),
            sa.ForeignKey("ai_provider_configs.id", ondelete="SET NULL"),
        ),
        sa.Column("provider", sa.Text(), nullable=False, server_default=""),
        sa.Column("model", sa.Text(), nullable=False, server_default=""),
        sa.Column("finish_reason", sa.Text(), nullable=False, server_default=""),
        sa.Column("error", sa.Text(), nullable=False, server_default=""),
        sa.Column("usage_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
    )
    op.create_index(
        "uq_agent_chat_messages_conversation_ordinal",
        "agent_chat_messages",
        ["conversation_id", "ordinal"],
        unique=True,
    )
    op.create_index(
        "ix_agent_chat_messages_conversation_ordinal",
        "agent_chat_messages",
        ["conversation_id", "ordinal"],
    )


def downgrade():
    op.drop_index("ix_agent_chat_messages_conversation_ordinal", table_name="agent_chat_messages")
    op.drop_index("uq_agent_chat_messages_conversation_ordinal", table_name="agent_chat_messages")
    op.drop_table("agent_chat_messages")
    op.drop_index("ix_agent_chat_conversations_owner_pinned", table_name="agent_chat_conversations")
    op.drop_index(
        "ix_agent_chat_conversations_owner_updated", table_name="agent_chat_conversations"
    )
    op.drop_table("agent_chat_conversations")
