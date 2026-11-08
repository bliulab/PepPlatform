"""Add independent content versions without rewriting historical experiments."""

from alembic import op
import sqlalchemy as sa

revision = "20260918_0012"
down_revision = "20260721_0011"
branch_labels = None
depends_on = None


def upgrade():
    # Inline schema keeps this migration independent of future service changes.
    tables = {
        "content_versions": [
            ("id", True, False),
            ("kind", False, False),
            ("identity_json", False, False),
            ("created_at", False, False),
        ],
        "versioning_runs": [
            ("run_id", True, False),
            ("owner_user_id", False, True),
            ("status", False, False),
            ("details_json", False, False),
            ("created_at", False, False),
            ("updated_at", False, False),
        ],
        "run_version_bindings": [
            ("run_id", True, False),
            ("role", True, False),
            ("version_id", False, False),
        ],
        "version_relations": [
            ("run_id", True, False),
            ("relation", True, False),
            ("target", True, False),
        ],
    }
    for name, columns in tables.items():
        if not sa.inspect(op.get_bind()).has_table(name):
            op.create_table(
                name,
                *(
                    sa.Column(key, sa.Text(), primary_key=pk, nullable=nullable)
                    for key, pk, nullable in columns
                ),
            )


def downgrade():
    for name in (
        "version_relations",
        "run_version_bindings",
        "versioning_runs",
        "content_versions",
    ):
        op.drop_table(name)
