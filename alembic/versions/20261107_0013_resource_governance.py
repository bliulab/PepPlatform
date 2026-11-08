"""Add administrative resource policies without rewriting scientific records."""

import sqlalchemy as sa

from alembic import op

revision = "20261107_0013"
down_revision = "20260918_0012"
branch_labels = None
depends_on = None

# Static, independent of future application defaults. Existing quota rows remain
# intact and become protected personal overrides when read by the new service.
SCHEMA = {
    "platform_resource_policies": "id! revision:i config_json updated_at",
    "user_resource_settings": "user_id! overrides_json allow_global_override:i updated_at",
    "resource_daily_usage": "user_id! usage_date! training_milliseconds:b registrations:i",
    "resource_leases": "id! user_id? kind resource_id? usage_date request_count:i input_count:b reserved_bytes:b expires_at created_at",
    "resource_executions": "experiment_id! user_id? status started_at accounted_at heartbeat_at policy_json pid:i? process_start?",
    "resource_pending_jobs": "experiment_id! enqueued_at",
    "resource_version_owners": "version_id! user_id!",
    "resource_inventory": "id! path kind owner_user_id? model_id? owners_json size_bytes:b protected:i reason state notified_at? delete_after? last_seen_at",
    "model_resource_protection": "model_id! pinned:i updated_at",
    "resource_cleanup_logs": "id! resource_id status released_bytes:b error? created_at",
    "platform_announcements": "id! title body audience_json pinned:i kind resource_id? publish_at expires_at? created_at updated_at",
    "announcement_reads": "announcement_id! user_id! read_at",
}


def upgrade():
    for name, spec in SCHEMA.items():
        if sa.inspect(op.get_bind()).has_table(name):
            continue
        columns = []
        for field in spec.split():
            nullable, primary = field.endswith("?"), field.endswith("!")
            parts = field.rstrip("?!").split(":")
            kind = {"i": sa.Integer, "b": sa.BigInteger}.get(
                parts[1] if len(parts) > 1 else "", sa.Text
            )
            columns.append(
                sa.Column(
                    parts[0],
                    kind(),
                    nullable=nullable,
                    primary_key=primary,
                    unique=name == "resource_inventory" and parts[0] == "path",
                )
            )
        op.create_table(name, *columns)
    indices = {index["name"] for index in sa.inspect(op.get_bind()).get_indexes("resource_leases")}
    if "ix_resource_leases_user_kind" not in indices:
        op.create_index("ix_resource_leases_user_kind", "resource_leases", ["user_id", "kind"])


def downgrade():
    for name in reversed(SCHEMA):
        op.drop_table(name)
