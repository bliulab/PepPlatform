"""Administrative policies and accounting, outside the scientific execution path.

All mutations share MetadataStore.atomic(): a PostgreSQL row lock or a SQLite
IMMEDIATE transaction. No policy is ever merged into an experiment configuration.
"""

# ruff: noqa: TRY004 -- ValueError is the public policy validation contract.

import hashlib
import json
import uuid
from copy import deepcopy
from datetime import datetime, timedelta, timezone

from sqlalchemy import (
    BigInteger,
    Column,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    delete,
    insert,
    inspect,
    or_,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError

GIB = 1024**3
DAILY_QUOTA_FIELDS = (
    "daily_experiment_limit",
    "daily_inference_request_limit",
    "daily_inference_input_limit",
    "daily_training_seconds_limit",
)
RESOURCE_METADATA = MetaData()
DEFAULT_QUOTAS = {
    "daily_experiment_limit": 10,
    "daily_concurrent_experiment_limit": 1,
    "daily_inference_request_limit": 200,
    "daily_inference_input_limit": 10000,
    "daily_training_seconds_limit": 7200,
    "max_experiment_seconds": 3600,
    "max_model_file_bytes": GIB,
    "max_model_bundle_bytes": 2 * GIB,
    "storage_limit_bytes": 10 * GIB,
    "max_model_count": 5,
    "model_retention_days": 14,
    "protected_model_limit": 3,
}
DEFAULT_POLICY = {
    "defaults": DEFAULT_QUOTAS,
    "limits": {
        "daily_registration_limit": 10,
        "daily_verification_email_limit": 20,
        "queue_capacity": 100,
        "disk_reserve_bytes": 10 * GIB,
        "disk_reserve_percent": 10,
        "disk_critical_bytes": 2 * GIB,
        "disk_critical_percent": 2,
        "inference_timeout_seconds": 300,
        "allowed_pretrained_models": [
            "facebook/esm2_t30_150M_UR50D",
            "Rostlab/prot_bert",
            "Rostlab/prot_t5_xl_uniref50",
            "EvolutionaryScale/esmc-300m-2024-12",
            "EvolutionaryScale/esmc-600m-2024-12",
            "EvolutionaryScale/esm3-sm-open-v1",
        ],
    },
    "retention": {
        "enabled": False,
        "delete_unprotected_models": False,
        "publish_announcements": False,
        "notice_hours": 72,
        "cache_idle_hours": 24,
    },
}

platform_policies = Table(
    "platform_resource_policies",
    RESOURCE_METADATA,
    Column("id", Text, primary_key=True),
    Column("revision", Integer, nullable=False),
    Column("config_json", Text, nullable=False),
    Column("updated_at", Text, nullable=False),
)
user_resource_settings = Table(
    "user_resource_settings",
    RESOURCE_METADATA,
    Column("user_id", Text, primary_key=True),
    Column("overrides_json", Text, nullable=False),
    Column("allow_global_override", Integer, nullable=False, default=0),
    Column("updated_at", Text, nullable=False),
)
resource_daily_usage = Table(
    "resource_daily_usage",
    RESOURCE_METADATA,
    Column("user_id", Text, primary_key=True),
    Column("usage_date", Text, primary_key=True),
    Column("training_milliseconds", BigInteger, nullable=False, default=0),
    Column("registrations", Integer, nullable=False, default=0),
)
verification_email_daily_usage = Table(
    "verification_email_daily_usage",
    RESOURCE_METADATA,
    Column("usage_date", Text, primary_key=True),
    Column("attempts", Integer, nullable=False, default=0),
    Column("updated_at", Text, nullable=False),
)
resource_daily_quota_limits = Table(
    "resource_daily_quota_limits",
    RESOURCE_METADATA,
    Column("user_id", Text, primary_key=True),
    Column("usage_date", Text, primary_key=True),
    Column("limits_json", Text, nullable=False),
    Column("updated_at", Text, nullable=False),
)
resource_quota_changes = Table(
    "resource_quota_changes",
    RESOURCE_METADATA,
    Column("id", Text, primary_key=True),
    Column("user_id", Text, nullable=False),
    Column("actor_user_id", Text),
    Column("source", Text, nullable=False),
    Column("changes_json", Text, nullable=False),
    Column("created_at", Text, nullable=False),
)
Index(
    "ix_resource_quota_changes_user",
    resource_quota_changes.c.user_id,
    resource_quota_changes.c.created_at,
)
resource_leases = Table(
    "resource_leases",
    RESOURCE_METADATA,
    Column("id", Text, primary_key=True),
    Column("user_id", Text),
    Column("kind", Text, nullable=False),
    Column("resource_id", Text),
    Column("usage_date", Text, nullable=False),
    Column("request_count", Integer, nullable=False, default=0),
    Column("input_count", BigInteger, nullable=False, default=0),
    Column("reserved_bytes", BigInteger, nullable=False, default=0),
    Column("expires_at", Text, nullable=False),
    Column("created_at", Text, nullable=False),
    Column("policy_json", Text),
)
Index("ix_resource_leases_user_kind", resource_leases.c.user_id, resource_leases.c.kind)
resource_executions = Table(
    "resource_executions",
    RESOURCE_METADATA,
    Column("experiment_id", Text, primary_key=True),
    Column("user_id", Text),
    Column("status", Text, nullable=False),
    Column("started_at", Text, nullable=False),
    Column("accounted_at", Text, nullable=False),
    Column("heartbeat_at", Text, nullable=False),
    Column("policy_json", Text, nullable=False),
    Column("pid", Integer),
    Column("process_start", Text),
)
resource_pending_jobs = Table(
    "resource_pending_jobs",
    RESOURCE_METADATA,
    Column("experiment_id", Text, primary_key=True),
    Column("enqueued_at", Text, nullable=False),
    Column("policy_json", Text),
)
resource_version_owners = Table(
    "resource_version_owners",
    RESOURCE_METADATA,
    Column("version_id", Text, primary_key=True),
    Column("user_id", Text, primary_key=True),
)
resource_inventory = Table(
    "resource_inventory",
    RESOURCE_METADATA,
    Column("id", Text, primary_key=True),
    Column("path", Text, nullable=False, unique=True),
    Column("kind", Text, nullable=False),
    Column("owner_user_id", Text),
    Column("model_id", Text),
    Column("owners_json", Text, nullable=False),
    Column("size_bytes", BigInteger, nullable=False),
    Column("protected", Integer, nullable=False),
    Column("reason", Text, nullable=False),
    Column("state", Text, nullable=False),
    Column("scheduled_at", Text),
    Column("notified_at", Text),
    Column("delete_after", Text),
    Column("last_seen_at", Text, nullable=False),
)
model_protection = Table(
    "model_resource_protection",
    RESOURCE_METADATA,
    Column("model_id", Text, primary_key=True),
    Column("pinned", Integer, nullable=False),
    Column("updated_at", Text, nullable=False),
)
model_retention = Table(
    "model_retention_states",
    RESOURCE_METADATA,
    Column("model_id", Text, primary_key=True),
    Column("experiment_id", Text, nullable=False),
    Column("owner_user_id", Text),
    Column("state", Text, nullable=False),
    Column("scheduled_at", Text),
    Column("notified_at", Text),
    Column("delete_after", Text),
    Column("removed_at", Text),
    Column("reason", Text),
    Column("manifest_json", Text, nullable=False),
    Column("error", Text),
    Column("updated_at", Text, nullable=False),
)
Index("ix_model_retention_owner_state", model_retention.c.owner_user_id, model_retention.c.state)
retired_model_versions = Table(
    "retired_model_versions",
    RESOURCE_METADATA,
    Column("version_id", Text, primary_key=True),
    Column("model_id", Text, nullable=False),
    Column("removed_at", Text, nullable=False),
)
resource_cleanup_logs = Table(
    "resource_cleanup_logs",
    RESOURCE_METADATA,
    Column("id", Text, primary_key=True),
    Column("resource_id", Text, nullable=False),
    Column("status", Text, nullable=False),
    Column("released_bytes", BigInteger, nullable=False),
    Column("error", Text),
    Column("created_at", Text, nullable=False),
)
platform_announcements = Table(
    "platform_announcements",
    RESOURCE_METADATA,
    Column("id", Text, primary_key=True),
    Column("title", Text, nullable=False),
    Column("body", Text, nullable=False),
    Column("audience_json", Text, nullable=False),
    Column("pinned", Integer, nullable=False),
    Column("kind", Text, nullable=False),
    Column("resource_id", Text),
    Column("publish_at", Text, nullable=False),
    Column("expires_at", Text),
    Column("created_at", Text, nullable=False),
    Column("updated_at", Text, nullable=False),
)
announcement_reads = Table(
    "announcement_reads",
    RESOURCE_METADATA,
    Column("announcement_id", Text, primary_key=True),
    Column("user_id", Text, primary_key=True),
    Column("read_at", Text, nullable=False),
)


def now_utc():
    return datetime.now(timezone.utc)


def dump(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def quota_exempt(quota):
    return quota.get("enforcement") == "exempt_admin"


def next_reset(current):
    return (current + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


def daily_reduction_mode(changes):
    mode = changes.get("daily_reduction_mode", "next_reset")
    if not isinstance(mode, str) or mode not in ("next_reset", "immediate"):
        raise ValueError("daily_reduction_mode must be next_reset or immediate")
    return mode


def announcement_requested(changes):
    value = changes.get("publish_announcements", False)
    if not isinstance(value, bool):
        raise ValueError("publish_announcements must be a boolean")
    return value


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= 2**63 - 1:
        raise ValueError(f"{name} must be an integer between {minimum} and {2**63 - 1}")
    return value


def iso_date(value, name):
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timezone required")
        return parsed.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a timestamp with a timezone") from exc


class PolicyConflict(ValueError):
    pass


class ResourceLimitError(RuntimeError):
    def __init__(self, code, *, status_code=429, **detail):
        self.status_code = status_code
        self.detail = {"code": code, **detail}
        super().__init__(detail.get("message") or code)


def check_quota(name, limit, used, requested=1):
    if limit == 0 or used + requested > limit:
        reset = {}
        if limit and name in {
            "daily_experiment_limit",
            "daily_inference_request_limit",
            "daily_inference_input_limit",
            "daily_training_seconds_limit",
            "daily_registration_limit",
        }:
            current = now_utc()
            midnight = current.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
                days=1
            )
            reset = {
                "reset_at": midnight.isoformat(),
                "retry_after": max(1, int((midnight - current).total_seconds())),
            }
        raise ResourceLimitError(
            "quota_exceeded", quota_type=name, limit=limit, used=used, requested=requested, **reset
        )


class ResourcePolicyStore:
    def __init__(self, store):
        self.store = store

    def current_time(self):
        # A request admitted across UTC midnight must check and charge one day.
        transaction = self.store._transaction.get()
        if transaction is None:
            return now_utc()
        return transaction.setdefault("resource_now", now_utc())

    def init_schema(self):
        RESOURCE_METADATA.create_all(self.store.engine)
        # MetadataStore also supports direct, non-Alembic initialization.
        with self.store.engine.begin() as conn:
            for table in (resource_pending_jobs, resource_leases):
                if "policy_json" not in {
                    column["name"] for column in inspect(conn).get_columns(table.name)
                }:
                    conn.exec_driver_sql(f"ALTER TABLE {table.name} ADD COLUMN policy_json TEXT")
            for table in (resource_inventory, model_retention):
                if "scheduled_at" not in {
                    column["name"] for column in inspect(conn).get_columns(table.name)
                }:
                    conn.exec_driver_sql(f"ALTER TABLE {table.name} ADD COLUMN scheduled_at TEXT")
                conn.execute(
                    update(table)
                    .where(table.c.scheduled_at.is_(None), table.c.notified_at.is_not(None))
                    .values(scheduled_at=table.c.notified_at)
                )
        try:
            with self.store.connect() as conn:
                if conn.execute(select(platform_policies.c.id)).first() is None:
                    conn.execute(
                        insert(platform_policies).values(
                            id="resources",
                            revision=1,
                            config_json=dump(DEFAULT_POLICY),
                            updated_at=self.current_time().isoformat(),
                        )
                    )
        except IntegrityError:
            pass  # Another application process initialized the singleton first.
        self._upgrade_model_quotas()
        self.backfill_admissions()

    def backfill_admissions(self):
        """Old queued jobs have no historical admission policy; preserve upgrade-time limits."""
        from .store import experiments

        with self.store.atomic(), self.store.connect() as conn:
            for row in (
                conn.execute(
                    select(experiments.c.id, experiments.c.owner_user_id).where(
                        experiments.c.status == "queued"
                    )
                )
                .mappings()
                .all()
            ):
                existing = (
                    conn.execute(
                        select(resource_pending_jobs).where(
                            resource_pending_jobs.c.experiment_id == row["id"]
                        )
                    )
                    .mappings()
                    .first()
                )
                if existing and existing["policy_json"]:
                    continue
                policy = {**self.quota(row["owner_user_id"]), "snapshot_source": "upgrade"}
                if existing:
                    conn.execute(
                        update(resource_pending_jobs)
                        .where(resource_pending_jobs.c.experiment_id == row["id"])
                        .values(policy_json=dump(policy))
                    )
                else:
                    conn.execute(
                        insert(resource_pending_jobs).values(
                            experiment_id=row["id"],
                            enqueued_at=self.current_time().isoformat(),
                            policy_json=dump(policy),
                        )
                    )

    def _upgrade_model_quotas(self):
        """Extend old JSON policies without lowering any existing protection allowance."""
        with self.store.atomic(), self.store.connect() as conn:
            row = (
                conn.execute(select(platform_policies).where(platform_policies.c.id == "resources"))
                .mappings()
                .one()
            )
            value = json.loads(row["config_json"])
            defaults = value.setdefault("defaults", {})
            changed = "max_model_count" not in defaults
            defaults.setdefault("max_model_count", max(5, defaults.get("protected_model_limit", 3)))
            value.setdefault("retention", {}).setdefault("delete_unprotected_models", False)
            if changed or "delete_unprotected_models" not in json.loads(row["config_json"]).get(
                "retention", {}
            ):
                conn.execute(
                    update(platform_policies)
                    .where(platform_policies.c.id == "resources")
                    .values(config_json=dump(value))
                )
            for setting in conn.execute(select(user_resource_settings)).mappings().all():
                overrides = json.loads(setting["overrides_json"])
                count = overrides.get("max_model_count", defaults["max_model_count"])
                protected = overrides.get(
                    "protected_model_limit", defaults.get("protected_model_limit", 3)
                )
                if protected > count:
                    overrides["max_model_count"] = protected
                    conn.execute(
                        update(user_resource_settings)
                        .where(user_resource_settings.c.user_id == setting["user_id"])
                        .values(overrides_json=dump(overrides))
                    )

    @staticmethod
    def validate_model_limits(quota):
        if quota["protected_model_limit"] > quota["max_model_count"]:
            raise ValueError("protected_model_limit must not exceed max_model_count")

    def policy(self):
        with self.store.connect() as conn:
            row = (
                conn.execute(select(platform_policies).where(platform_policies.c.id == "resources"))
                .mappings()
                .one()
            )
        value = deepcopy(DEFAULT_POLICY)
        stored = json.loads(row["config_json"])
        for group in value:
            value[group].update(stored.get(group, {}))
        return {**value, "revision": row["revision"], "updated_at": row["updated_at"]}

    def quota(self, user_id):
        return self.quotas([user_id])[user_id]

    def quotas(self, user_ids):
        from .store import user_quotas, users

        if not user_ids:
            return {}
        current = self.current_time()
        with self.store.connect() as conn:
            roles = dict(
                conn.execute(select(users.c.id, users.c.role).where(users.c.id.in_(user_ids))).all()
            )
            today_limits = {
                row["user_id"]: json.loads(row["limits_json"])
                for row in conn.execute(
                    select(resource_daily_quota_limits).where(
                        resource_daily_quota_limits.c.user_id.in_(user_ids),
                        resource_daily_quota_limits.c.usage_date == current.date().isoformat(),
                    )
                ).mappings()
            }
            settings = {
                row["user_id"]: row
                for row in conn.execute(
                    select(user_resource_settings).where(
                        user_resource_settings.c.user_id.in_(user_ids)
                    )
                ).mappings()
            }
            legacy_rows = {
                row["user_id"]: row
                for row in conn.execute(
                    select(user_quotas).where(user_quotas.c.user_id.in_(user_ids))
                ).mappings()
            }
        defaults = self.policy()["defaults"]
        result = {}
        for user_id in user_ids:
            row, legacy = settings.get(user_id), legacy_rows.get(user_id)
            overrides = (
                json.loads(row["overrides_json"])
                if row
                else {key: legacy[key] for key in DEFAULT_QUOTAS if legacy and key in legacy}
            )
            configured = {**defaults, **overrides}
            effective = dict(configured)
            for key, value in today_limits.get(user_id, {}).items():
                effective[key] = max(value, configured[key])
            exempt = roles.get(user_id) == "admin"
            result[user_id] = {
                "user_id": user_id,
                **effective,
                "configured": configured,
                "enforcement": "exempt_admin" if exempt else "limited",
                "pending": {}
                if exempt
                else {
                    key: {"value": configured[key], "effective_at": next_reset(current).isoformat()}
                    for key in DAILY_QUOTA_FIELDS
                    if effective[key] > configured[key]
                },
                "overrides": overrides,
                "sources": {key: "personal" if key in overrides else "global" for key in defaults},
                "allow_global_override": bool(row and row["allow_global_override"]),
                "updated_at": row["updated_at"]
                if row
                else (legacy["updated_at"] if legacy else None),
            }
        return result

    def _quota_transition(self, before, configured, selected, mode):
        """Resolve both previews and writes without changing an accepted job's policy."""
        current = self.current_time()
        limits = {key: max(before[key], configured[key]) for key in DAILY_QUOTA_FIELDS}
        changes = {}
        for key in sorted(selected):
            value = configured[key]
            daily = key in DAILY_QUOTA_FIELDS
            if daily and mode == "immediate":
                limits[key] = value
            if value == before["configured"][key] and not (daily and limits[key] != before[key]):
                continue
            deferred = daily and limits[key] > value
            changes[key] = {
                "from": before["configured"][key],
                "to": value,
                "effective_from": before[key],
                "effective_to": value,
                "effective_at": (next_reset(current) if deferred else current).isoformat(),
                **(
                    {"daily_reduction_mode": "next_reset" if deferred else "immediate"}
                    if daily
                    else {}
                ),
            }
        return limits, changes

    def _record_change(
        self,
        user_id,
        before,
        configured,
        *,
        selected,
        mode,
        actor_user_id=None,
        source="personal",
        publish_announcements=False,
    ):
        """Always persist entitlements and history; publish only when explicitly requested."""
        if quota_exempt(before):
            return
        limits, changes = self._quota_transition(before, configured, selected, mode)
        if not changes:
            return
        current = self.current_time()
        with self.store.connect() as conn:
            condition = (
                resource_daily_quota_limits.c.user_id == user_id,
                resource_daily_quota_limits.c.usage_date == current.date().isoformat(),
            )
            values = {"limits_json": dump(limits), "updated_at": current.isoformat()}
            if conn.execute(
                select(resource_daily_quota_limits.c.user_id).where(*condition)
            ).first():
                conn.execute(update(resource_daily_quota_limits).where(*condition).values(**values))
            else:
                conn.execute(
                    insert(resource_daily_quota_limits).values(
                        user_id=user_id, usage_date=current.date().isoformat(), **values
                    )
                )
            identifier = uuid.uuid4().hex
            conn.execute(
                insert(resource_quota_changes).values(
                    id=identifier,
                    user_id=user_id,
                    actor_user_id=actor_user_id,
                    source=source,
                    changes_json=dump(changes),
                    created_at=current.isoformat(),
                )
            )
            if publish_announcements:
                self.publish_announcement(
                    {"title": "Quota updated", "body": dump(changes), "audience": [user_id]},
                    identifier="quota-" + identifier,
                    kind="quota_change",
                    resource_id=identifier,
                )

    def set_quota(self, user_id, changes, *, actor_user_id=None, source="personal"):
        if not self.store.get_user(user_id):
            return None
        if not isinstance(changes, dict):
            raise ValueError("quota changes must be an object")
        mode = daily_reduction_mode(changes)
        publish = announcement_requested(changes)
        unknown = (
            set(changes)
            - set(DEFAULT_QUOTAS)
            - {"allow_global_override", "daily_reduction_mode", "publish_announcements"}
        )
        if unknown:
            raise ValueError(f"unknown quota fields: {', '.join(sorted(unknown))}")
        for key, value in changes.items():
            if key in {"daily_reduction_mode", "publish_announcements"}:
                continue
            if key == "allow_global_override":
                if not isinstance(value, bool):
                    raise ValueError("allow_global_override must be a boolean")
            elif value is not None:
                integer(value, key, minimum=1 if key == "model_retention_days" else 0)
        with self.store.atomic():
            before = self.quota(user_id)
            overrides = dict(before["overrides"])
            for key, value in changes.items():
                if key not in DEFAULT_QUOTAS:
                    continue
                if value is None:
                    overrides.pop(key, None)
                else:
                    overrides[key] = value
            self.validate_model_limits({**self.policy()["defaults"], **overrides})
            self._record_change(
                user_id,
                before,
                {**self.policy()["defaults"], **overrides},
                selected=set(changes) & set(DEFAULT_QUOTAS),
                mode=mode,
                actor_user_id=actor_user_id,
                source=source,
                publish_announcements=publish,
            )
            values = {
                "user_id": user_id,
                "overrides_json": dump(overrides),
                "allow_global_override": int(
                    changes.get("allow_global_override", before["allow_global_override"])
                ),
                "updated_at": self.current_time().isoformat(),
            }
            with self.store.connect() as conn:
                existing = conn.execute(
                    select(user_resource_settings.c.user_id).where(
                        user_resource_settings.c.user_id == user_id
                    )
                ).first()
                if existing:
                    conn.execute(
                        update(user_resource_settings)
                        .where(user_resource_settings.c.user_id == user_id)
                        .values(**values)
                    )
                else:
                    conn.execute(insert(user_resource_settings).values(**values))
            return self.quota(user_id)

    def validate_policy(self, changes):
        if not isinstance(changes, dict):
            raise ValueError("policy must be an object")
        unknown = set(changes) - {
            "defaults",
            "limits",
            "retention",
            "revision",
            "apply_to_users",
            "preview_token",
            "daily_reduction_mode",
            "publish_announcements",
        }
        if unknown:
            raise ValueError(f"unknown policy fields: {', '.join(sorted(unknown))}")
        daily_reduction_mode(changes)
        announcement_requested(changes)
        current = self.policy()
        integer(changes.get("revision"), "revision", minimum=1)
        if changes.get("revision") != current["revision"]:
            raise PolicyConflict("policy changed; reload and preview again")
        merged = {key: deepcopy(current[key]) for key in DEFAULT_POLICY}
        for group, existing_values in merged.items():
            updates = changes.get(group, {})
            if not isinstance(updates, dict) or set(updates) - set(existing_values):
                raise ValueError(f"invalid {group} fields")
            for key, value in updates.items():
                if key in {"enabled", "delete_unprotected_models", "publish_announcements"}:
                    if not isinstance(value, bool):
                        raise ValueError(f"{key} must be a boolean")
                elif key == "allowed_pretrained_models":
                    if (
                        not isinstance(value, list)
                        or len(value) > 100
                        or any(
                            not isinstance(v, str) or not v.strip() or len(v) > 300 for v in value
                        )
                    ):
                        raise ValueError(
                            "allowed_pretrained_models must contain at most 100 model identifiers"
                        )
                    value = list(dict.fromkeys(v.strip() for v in value))
                else:
                    integer(
                        value,
                        key,
                        minimum=1
                        if key
                        in {"model_retention_days", "inference_timeout_seconds", "cache_idle_hours"}
                        else 0,
                    )
                    if key.endswith("percent") and value > 100:
                        raise ValueError(f"{key} must not exceed 100")
                    if key == "notice_hours" and value < 72:
                        raise ValueError("notice_hours must be at least 72")
                merged[group][key] = value
        if "apply_to_users" in changes and not isinstance(changes["apply_to_users"], bool):
            raise ValueError("apply_to_users must be a boolean")
        self.validate_model_limits(merged["defaults"])
        limits = merged["limits"]
        if (
            limits["disk_critical_bytes"] > limits["disk_reserve_bytes"]
            or limits["disk_critical_percent"] > limits["disk_reserve_percent"]
        ):
            raise ValueError("critical disk thresholds must not exceed admission thresholds")
        return merged

    def preview(self, changes):
        with self.store.atomic():
            return self._preview(changes)

    def _preview(self, changes):
        from .store import users

        merged = self.validate_policy(changes)
        mode = daily_reduction_mode(changes)
        selected = set(changes.get("defaults", {}))
        affected, skipped, stamps, impacts = [], [], [], []
        exempt_count = 0
        with self.store.connect() as conn:
            user_rows = (
                conn.execute(select(users.c.id, users.c.email, users.c.role).order_by(users.c.id))
                .mappings()
                .all()
            )
        quotas = self.quotas([user["id"] for user in user_rows])
        for user in user_rows:
            quota = quotas[user["id"]]
            stamps.append(
                [
                    user["id"],
                    user["role"],
                    quota["overrides"],
                    quota["allow_global_override"],
                    {key: quota[key] for key in DAILY_QUOTA_FIELDS},
                ]
            )
            if quota_exempt(quota):
                exempt_count += 1
                continue
            fields = sorted(selected & set(quota["overrides"]))
            prospective = dict(quota["overrides"])
            if changes.get("apply_to_users") and quota["allow_global_override"]:
                for key in selected:
                    prospective.pop(key, None)
            try:
                self.validate_model_limits({**merged["defaults"], **prospective})
            except ValueError as exc:
                raise ValueError(f"Invalid model limits for {user['email']}: {exc}") from exc
            target = {**merged["defaults"], **prospective}
            _, field_changes = self._quota_transition(
                quota, target, selected - set(prospective), mode
            )
            if field_changes:
                impacts.append(
                    {
                        "user_id": user["id"],
                        "email": user["email"],
                        "changes": field_changes,
                        "blocks_new": self._blocks_new(user["id"], target, field_changes),
                    }
                )
            if not fields:
                continue
            entry = {"user_id": user["id"], "email": user["email"], "fields": fields}
            (affected if quota["allow_global_override"] else skipped).append(entry)
        token = hashlib.sha256(
            dump(
                [
                    changes.get("revision"),
                    self.current_time().date().isoformat(),
                    merged,
                    mode,
                    bool(changes.get("apply_to_users")),
                    announcement_requested(changes),
                    sorted(selected),
                    stamps,
                ]
            ).encode()
        ).hexdigest()
        return {
            "affected": affected,
            "skipped": skipped,
            "affected_count": len(affected),
            "skipped_count": len(skipped),
            "preview_token": token,
            "impacts": impacts,
            "announcement_count": len(impacts) if announcement_requested(changes) else 0,
            "immediate_count": sum(
                any(
                    item.get("daily_reduction_mode") != "next_reset"
                    for key, item in entry["changes"].items()
                )
                for entry in impacts
            ),
            "deferred_count": sum(
                any(
                    item.get("daily_reduction_mode") == "next_reset"
                    for key, item in entry["changes"].items()
                )
                for entry in impacts
            ),
            "blocked_count": sum(entry["blocks_new"] for entry in impacts),
            "exempt_count": exempt_count,
        }

    def _blocks_new(self, user_id, target, changes):
        """Report exhausted capacity, including today's reservations for immediate changes."""
        from sqlalchemy import func

        from .model_retention import active_model_condition
        from .store import experiments, models

        with self.store.connect() as conn:
            retained = conn.scalar(
                select(func.count())
                .select_from(models)
                .where(models.c.owner_user_id == user_id, active_model_condition())
            )
            reserved = conn.scalar(
                select(func.count())
                .select_from(experiments)
                .where(
                    experiments.c.owner_user_id == user_id,
                    experiments.c.status.in_(["queued", "running"]),
                    ~select(models.c.id).where(models.c.experiment_id == experiments.c.id).exists(),
                )
            )
            protected = conn.scalar(
                select(func.count())
                .select_from(
                    models.join(model_protection, models.c.id == model_protection.c.model_id)
                )
                .where(
                    models.c.owner_user_id == user_id,
                    active_model_condition(),
                    model_protection.c.pinned == 1,
                )
            )
            storage = sum(
                row["size_bytes"]
                for row in conn.execute(
                    select(resource_inventory).where(resource_inventory.c.state != "deleted")
                ).mappings()
                if user_id in json.loads(row["owners_json"])
            )
        used = {
            "max_model_count": retained + reserved,
            "protected_model_limit": protected,
            "daily_concurrent_experiment_limit": self.store.count_active_experiments_for_user(
                user_id
            ),
            "storage_limit_bytes": storage
            + sum(row["reserved_bytes"] for row in self.leases(user_id=user_id)),
        }
        usage = self.store.get_quota_usage(user_id)
        leases = [
            row
            for row in self.leases(user_id=user_id, kind="inference")
            if row["usage_date"] == usage["usage_date"]
        ]
        used.update(
            {
                "daily_experiment_limit": usage["experiments_created"],
                "daily_training_seconds_limit": usage["training_seconds"],
                "daily_inference_request_limit": usage["inference_requests"]
                + sum(row["request_count"] for row in leases),
                "daily_inference_input_limit": usage["inference_inputs"]
                + sum(row["input_count"] for row in leases),
            }
        )
        return any(
            key in used
            and target[key] <= used[key]
            and item.get("daily_reduction_mode") != "next_reset"
            for key, item in changes.items()
        )

    def save_policy(self, changes, *, actor_user_id=None):
        from .store import users

        with self.store.atomic():
            with self.store.connect() as conn:
                identifiers = list(conn.execute(select(users.c.id)).scalars())
            before = self.quotas(identifiers)
            merged = self.validate_policy(changes)
            mode = daily_reduction_mode(changes)
            selected = set(changes.get("defaults", {}))
            preview = self.preview(changes)
            if (
                changes.get("apply_to_users") or changes.get("preview_token") is not None
            ) and changes.get("preview_token") != preview["preview_token"]:
                raise PolicyConflict("users or policy changed; preview again before applying")
            with self.store.connect() as conn:
                conn.execute(
                    update(platform_policies)
                    .where(platform_policies.c.id == "resources")
                    .values(
                        revision=changes["revision"] + 1,
                        config_json=dump(merged),
                        updated_at=self.current_time().isoformat(),
                    )
                )
                if not (
                    merged["retention"]["enabled"]
                    and merged["retention"]["delete_unprotected_models"]
                ):
                    pending_ids = list(
                        conn.execute(
                            select(model_retention.c.model_id).where(
                                model_retention.c.state.in_(["scheduled", "notified"])
                            )
                        ).scalars()
                    )
                    conn.execute(
                        update(model_retention)
                        .where(model_retention.c.state.in_(["scheduled", "notified"]))
                        .values(
                            state="active",
                            scheduled_at=None,
                            notified_at=None,
                            delete_after=None,
                            updated_at=self.current_time().isoformat(),
                        )
                    )
                    for model_id in pending_ids:
                        conn.execute(
                            update(platform_announcements)
                            .where(platform_announcements.c.id == "model-cleanup-" + model_id)
                            .values(expires_at=self.current_time().isoformat())
                        )
            if changes.get("apply_to_users"):
                with self.store.connect() as conn:
                    for user in preview["affected"]:
                        overrides = {
                            key: value
                            for key, value in before[user["user_id"]]["overrides"].items()
                            if key not in selected
                        }
                        conn.execute(
                            update(user_resource_settings)
                            .where(user_resource_settings.c.user_id == user["user_id"])
                            .values(
                                overrides_json=dump(overrides),
                                updated_at=self.current_time().isoformat(),
                            )
                        )
            after = self.quotas(identifiers)
            for identifier in identifiers:
                self._record_change(
                    identifier,
                    before[identifier],
                    after[identifier]["configured"],
                    selected=selected - set(after[identifier]["overrides"]),
                    mode=mode,
                    actor_user_id=actor_user_id,
                    source="global",
                    publish_announcements=announcement_requested(changes),
                )
            return {
                "policy": self.policy(),
                "application": preview if changes.get("apply_to_users") else None,
                "impacts": preview["impacts"],
            }

    def daily(self, user_id, usage_date=None):
        day = usage_date or self.current_time().date().isoformat()
        with self.store.connect() as conn:
            row = (
                conn.execute(
                    select(resource_daily_usage).where(
                        resource_daily_usage.c.user_id == user_id,
                        resource_daily_usage.c.usage_date == day,
                    )
                )
                .mappings()
                .first()
            )
        return (
            dict(row)
            if row
            else {
                "user_id": user_id,
                "usage_date": day,
                "training_milliseconds": 0,
                "registrations": 0,
            }
        )

    def add_daily(self, user_id, day, *, milliseconds=0, registrations=0):
        with self.store.connect() as conn:
            row = conn.execute(
                select(resource_daily_usage).where(
                    resource_daily_usage.c.user_id == user_id,
                    resource_daily_usage.c.usage_date == day,
                )
            ).first()
            if row:
                conn.execute(
                    update(resource_daily_usage)
                    .where(
                        resource_daily_usage.c.user_id == user_id,
                        resource_daily_usage.c.usage_date == day,
                    )
                    .values(
                        training_milliseconds=resource_daily_usage.c.training_milliseconds
                        + milliseconds,
                        registrations=resource_daily_usage.c.registrations + registrations,
                    )
                )
            else:
                conn.execute(
                    insert(resource_daily_usage).values(
                        user_id=user_id,
                        usage_date=day,
                        training_milliseconds=milliseconds,
                        registrations=registrations,
                    )
                )

    def email_delivery_usage(self):
        """One shared UTC-day budget for all account-code SMTP attempts."""
        with self.store.atomic(), self.store.connect() as conn:
            current = self.current_time()
            day = current.date().isoformat()
            used = conn.execute(
                select(verification_email_daily_usage.c.attempts).where(
                    verification_email_daily_usage.c.usage_date == day
                )
            ).scalar_one_or_none() or 0
            limit = self.policy()["limits"]["daily_verification_email_limit"]
            return {
                "usage_date": day,
                "used": used,
                "limit": limit,
                "remaining": max(0, limit - used),
                "reset_at": next_reset(current).isoformat() if limit else None,
            }

    def reserve_verification_email(self):
        """Commit a conservative attempt before SMTP; never refund uncertain sends."""
        with self.store.atomic(), self.store.connect() as conn:
            usage = self.email_delivery_usage()
            if not usage["remaining"]:
                reset = {}
                if usage["reset_at"]:
                    reset = {
                        "reset_at": usage["reset_at"],
                        "retry_after": max(
                            1,
                            int((next_reset(self.current_time()) - self.current_time()).total_seconds()),
                        ),
                    }
                raise ResourceLimitError(
                    "quota_exceeded",
                    quota_type="daily_verification_email_limit",
                    limit=usage["limit"],
                    used=usage["used"],
                    requested=1,
                    **reset,
                )
            row = conn.execute(
                select(verification_email_daily_usage.c.usage_date).where(
                    verification_email_daily_usage.c.usage_date == usage["usage_date"]
                )
            ).first()
            values = {
                "attempts": usage["used"] + 1,
                "updated_at": self.current_time().isoformat(),
            }
            if row:
                conn.execute(
                    update(verification_email_daily_usage)
                    .where(verification_email_daily_usage.c.usage_date == usage["usage_date"])
                    .values(**values)
                )
            else:
                conn.execute(
                    insert(verification_email_daily_usage).values(
                        usage_date=usage["usage_date"], **values
                    )
                )

    def register_account(self):
        day = self.current_time().date().isoformat()
        used = self.daily("__public_registration__", day)["registrations"]
        check_quota(
            "daily_registration_limit", self.policy()["limits"]["daily_registration_limit"], used
        )
        self.add_daily("__public_registration__", day, registrations=1)

    def create_lease(
        self,
        kind,
        user_id=None,
        resource_id=None,
        *,
        reserved_bytes=0,
        request_count=0,
        input_count=0,
        seconds=600,
        policy=None,
    ):
        current = self.current_time()
        identifier = uuid.uuid4().hex
        with self.store.connect() as conn:
            conn.execute(
                insert(resource_leases).values(
                    id=identifier,
                    user_id=user_id,
                    kind=kind,
                    resource_id=resource_id,
                    usage_date=current.date().isoformat(),
                    reserved_bytes=reserved_bytes,
                    request_count=request_count,
                    input_count=input_count,
                    expires_at=(current + timedelta(seconds=seconds)).isoformat(),
                    created_at=current.isoformat(),
                    policy_json=dump(policy) if policy is not None else None,
                )
            )
        return identifier

    def release_lease(self, identifier):
        if identifier:
            with self.store.atomic(), self.store.connect() as conn:
                conn.execute(delete(resource_leases).where(resource_leases.c.id == identifier))

    def leases(self, *, user_id=None, kind=None):
        query = select(resource_leases).where(
            resource_leases.c.expires_at > self.current_time().isoformat()
        )
        if user_id is not None:
            query = query.where(resource_leases.c.user_id == user_id)
        if kind is not None:
            query = query.where(resource_leases.c.kind == kind)
        with self.store.connect() as conn:
            return [dict(row) for row in conn.execute(query).mappings()]

    def announcements(self, user=None, *, admin=False):
        from .store import users

        query = select(platform_announcements)
        current = self.current_time().isoformat()
        if not admin:
            audience = platform_announcements.c.audience_json == dump("all")
            if user:
                audience = or_(
                    audience,
                    platform_announcements.c.audience_json.contains(
                        dump(user["id"]), autoescape=True
                    ),
                )
            query = query.where(
                audience,
                platform_announcements.c.publish_at <= current,
                or_(
                    platform_announcements.c.expires_at.is_(None),
                    platform_announcements.c.expires_at > current,
                ),
            )
        with self.store.connect() as conn:
            rows = (
                conn.execute(
                    query.order_by(
                        platform_announcements.c.pinned.desc(),
                        platform_announcements.c.publish_at.desc(),
                    ).limit(500)
                )
                .mappings()
                .all()
            )
            read = set(
                conn.execute(
                    select(announcement_reads.c.announcement_id).where(
                        announcement_reads.c.user_id == (user or {}).get("id", ""),
                    )
                ).scalars()
            )
            labels = {}
            if admin:
                targets = sorted(
                    {
                        identifier
                        for row in rows
                        for identifier in (
                            json.loads(row["audience_json"])
                            if row["audience_json"] != dump("all")
                            else []
                        )
                    }
                )
                for offset in range(0, len(targets), 500):
                    labels.update(
                        dict(
                            conn.execute(
                                select(users.c.id, users.c.email).where(
                                    users.c.id.in_(targets[offset : offset + 500])
                                )
                            ).all()
                        )
                    )
        result = []
        for row in rows:
            audience = json.loads(row["audience_json"])
            if not admin and (
                row["publish_at"] > current
                or (row["expires_at"] and row["expires_at"] <= current)
                or (audience != "all" and (user or {}).get("id") not in audience)
            ):
                continue
            result.append(
                {
                    **{key: value for key, value in row.items() if key != "audience_json"},
                    "audience": audience,
                    "pinned": bool(row["pinned"]),
                    "read": row["id"] in read,
                    **(
                        {"quota_change": json.loads(row["body"])}
                        if row["kind"] == "quota_change"
                        else {}
                    ),
                    **(
                        {
                            "audience_labels": {
                                identifier: labels.get(identifier, identifier)
                                for identifier in audience
                            }
                        }
                        if admin and audience != "all"
                        else {}
                    ),
                }
            )
        return result

    def publish_announcement(self, payload, *, identifier=None, kind="manual", resource_id=None):
        from .store import users

        title = str(payload.get("title") or "").strip()
        body = str(payload.get("body") or "").strip()
        if not title or len(title) > 200 or not body or len(body) > 20000:
            raise ValueError(
                "announcement requires a title (1–200 characters) and body (1–20000 characters)"
            )
        audience = payload.get("audience", "all")
        if audience != "all":
            if (
                not isinstance(audience, list)
                or not audience
                or len(audience) > 1000
                or any(not isinstance(v, str) for v in audience)
            ):
                raise ValueError("audience must be 'all' or 1–1000 user IDs")
            audience = list(dict.fromkeys(audience))
            with self.store.connect() as conn:
                found = set(
                    conn.execute(select(users.c.id).where(users.c.id.in_(audience))).scalars()
                )
            if found != set(audience):
                raise ValueError("announcement audience contains unknown users")
        pinned = payload.get("pinned", False)
        if not isinstance(pinned, bool):
            raise ValueError("pinned must be a boolean")
        current = self.current_time().isoformat()
        publish_at = iso_date(payload.get("publish_at") or current, "publish_at")
        expires_at = (
            iso_date(payload["expires_at"], "expires_at") if payload.get("expires_at") else None
        )
        if expires_at and expires_at <= publish_at:
            raise ValueError("expires_at must follow publish_at")
        identifier = identifier or uuid.uuid4().hex
        with self.store.atomic(), self.store.connect() as conn:
            old = conn.execute(
                select(platform_announcements.c.id).where(platform_announcements.c.id == identifier)
            ).first()
            values = {
                "title": title,
                "body": body,
                "audience_json": dump(audience),
                "pinned": int(pinned),
                "kind": kind,
                "resource_id": resource_id,
                "publish_at": publish_at,
                "expires_at": expires_at,
                "updated_at": current,
            }
            if old:
                conn.execute(
                    update(platform_announcements)
                    .where(platform_announcements.c.id == identifier)
                    .values(**values)
                )
                conn.execute(
                    delete(announcement_reads).where(
                        announcement_reads.c.announcement_id == identifier
                    )
                )
            else:
                conn.execute(
                    insert(platform_announcements).values(
                        id=identifier, created_at=current, **values
                    )
                )
        return {
            "id": identifier,
            "title": title,
            "body": body,
            "audience": audience,
            "pinned": pinned,
            "kind": kind,
            "resource_id": resource_id,
            "publish_at": publish_at,
            "expires_at": expires_at,
            "read": False,
            "created_at": current,
            "updated_at": current,
        }

    def mark_read(self, announcement_id, user):
        if not any(row["id"] == announcement_id for row in self.announcements(user)):
            raise KeyError(announcement_id)
        with self.store.atomic(), self.store.connect() as conn:
            existing = conn.execute(
                select(announcement_reads).where(
                    announcement_reads.c.announcement_id == announcement_id,
                    announcement_reads.c.user_id == user["id"],
                )
            ).first()
            if not existing:
                conn.execute(
                    insert(announcement_reads).values(
                        announcement_id=announcement_id,
                        user_id=user["id"],
                        read_at=self.current_time().isoformat(),
                    )
                )

    def remove_user(self, user_id, conn):
        conn.execute(delete(model_retention).where(model_retention.c.owner_user_id == user_id))
        notice_ids = select(platform_announcements.c.id).where(
            platform_announcements.c.kind == "quota_change",
            platform_announcements.c.resource_id.in_(
                select(resource_quota_changes.c.id).where(
                    resource_quota_changes.c.user_id == user_id
                )
            ),
        )
        conn.execute(
            delete(announcement_reads).where(announcement_reads.c.announcement_id.in_(notice_ids))
        )
        conn.execute(
            delete(platform_announcements).where(platform_announcements.c.id.in_(notice_ids))
        )
        for table in (
            user_resource_settings,
            resource_daily_usage,
            resource_daily_quota_limits,
            resource_quota_changes,
            resource_leases,
            announcement_reads,
        ):
            conn.execute(delete(table).where(table.c.user_id == user_id))
