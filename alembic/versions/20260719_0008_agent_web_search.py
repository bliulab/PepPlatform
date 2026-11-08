"""persist agent chat web search metadata

Revision ID: 20260719_0008
Revises: 20260719_0007
Create Date: 2026-07-19
"""

from alembic import op
import sqlalchemy as sa


revision = "20260719_0008"
down_revision = "20260719_0007"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "agent_chat_messages",
        sa.Column("web_search_json", sa.Text(), nullable=False, server_default="{}"),
    )


def downgrade():
    op.drop_column("agent_chat_messages", "web_search_json")
