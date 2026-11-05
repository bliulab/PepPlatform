from pepplatform.stream_compat import normalize_stream_fields
from pepplatform.task_compat import (
    builtin_name_for_display,
    normalize_task_type,
    task_spellings,
    builtin_id_spellings,
)

import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import (
    Column,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    and_,
    case,
    create_engine,
    delete,
    event,
    func,
    insert,
    inspect,
    or_,
    select,
    text,
    update,
)
from sqlalchemy.exc import IntegrityError

from pepplatform.run_progress import RUN_PROGRESS_STAGES
from pepplatform.web.metrics import rebuild_metric_summary
from pepplatform.pr_metrics import preserve_legacy_summary, without_legacy_aupr
from pepplatform.workflow import PROFILE_DEPENDENT_INFERENCE_REASON, uses_profile_features


LEADERBOARD_METRICS = {
    "acc",
    "sacc",
    "f1",
    "auc",
    "aupr",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "mcc",
    "auc_micro",
    "aupr_micro",
}
LEADERBOARD_SPLITS = {"auto", "test", "validation", "val"}
LEADERBOARD_CONFIG_KEYS = [
    "feature_route",
    "tokenizer",
    "encoder",
    "feature_transformation",
    "model",
    "classifier",
    "sequence_model",
    "training_mode",
    "residue_adapter_dim",
    "pair_strategy",
    "interaction_level",
    "interaction_model",
    "predictor_type",
    "graph_predictor",
    "fusion",
    "epochs",
]
DEFAULT_USER_QUOTAS = {
    "daily_experiment_limit": 10,
    "daily_concurrent_experiment_limit": 1,
    "daily_inference_request_limit": 200,
    "daily_inference_input_limit": 10000,
}

SUPPORTED_LOCALES = {"en", "zh"}
RERUN_NAME_SUFFIXES = (" · rerun", " · 重跑")
GENERATED_NAME_QUALIFIERS = {"conservative", "balanced", "exploratory"}


def generated_rerun_name_parts(name):
    base = str(name or "")
    count = 0
    while True:
        matched = next((suffix for suffix in RERUN_NAME_SUFFIXES if base.endswith(suffix)), None)
        if not matched:
            break
        base = base[: -len(matched)]
        count += 1
    base = base.strip()
    return (base, count) if base and count else ("", 0)


STORE_METADATA = MetaData()

users = Table(
    "users",
    STORE_METADATA,
    Column("id", Text, primary_key=True),
    Column("email", Text, nullable=False, unique=True),
    Column("password_hash", Text, nullable=False),
    Column("display_name", Text, nullable=False, default=""),
    Column("avatar_object_key", Text),
    Column("avatar_content_type", Text),
    Column("avatar_updated_at", Text),
    Column("preferred_locale", Text, nullable=False, default="en"),
    Column("email_verified", Integer, nullable=False, default=0),
    Column("role", Text, nullable=False, default="user"),
    Column("status", Text, nullable=False, default="active"),
    Column("created_at", Text, nullable=False),
    Column("updated_at", Text, nullable=False),
)

email_verification_codes = Table(
    "email_verification_codes",
    STORE_METADATA,
    Column("id", Text, primary_key=True),
    Column("user_id", Text, ForeignKey("users.id"), nullable=False),
    Column("purpose", Text, nullable=False),
    Column("code_hash", Text, nullable=False),
    Column("expires_at", Text, nullable=False),
    Column("attempts", Integer, nullable=False, default=0),
    Column("consumed_at", Text),
    Column("created_at", Text, nullable=False),
)
Index(
    "ix_email_codes_user_purpose",
    email_verification_codes.c.user_id,
    email_verification_codes.c.purpose,
)

sessions = Table(
    "sessions",
    STORE_METADATA,
    Column("id", Text, primary_key=True),
    Column("user_id", Text, ForeignKey("users.id"), nullable=False),
    Column("token_hash", Text, nullable=False, unique=True),
    Column("expires_at", Text, nullable=False),
    Column("user_agent", Text, nullable=False, default=""),
    Column("created_at", Text, nullable=False),
    Column("last_seen_at", Text, nullable=False),
    Column("revoked_at", Text),
)
Index("ix_sessions_user_id", sessions.c.user_id)

user_quotas = Table(
    "user_quotas",
    STORE_METADATA,
    Column("user_id", Text, ForeignKey("users.id"), primary_key=True),
    Column(
        "daily_experiment_limit",
        Integer,
        nullable=False,
        default=DEFAULT_USER_QUOTAS["daily_experiment_limit"],
    ),
    Column(
        "daily_concurrent_experiment_limit",
        Integer,
        nullable=False,
        default=DEFAULT_USER_QUOTAS["daily_concurrent_experiment_limit"],
    ),
    Column(
        "daily_inference_request_limit",
        Integer,
        nullable=False,
        default=DEFAULT_USER_QUOTAS["daily_inference_request_limit"],
    ),
    Column(
        "daily_inference_input_limit",
        Integer,
        nullable=False,
        default=DEFAULT_USER_QUOTAS["daily_inference_input_limit"],
    ),
    Column("updated_at", Text, nullable=False),
)

quota_usage = Table(
    "quota_usage",
    STORE_METADATA,
    Column("user_id", Text, ForeignKey("users.id"), primary_key=True),
    Column("usage_date", Text, primary_key=True),
    Column("experiments_created", Integer, nullable=False, default=0),
    Column("inference_requests", Integer, nullable=False, default=0),
    Column("inference_inputs", Integer, nullable=False, default=0),
    Column("updated_at", Text, nullable=False),
)
Index("ix_quota_usage_date", quota_usage.c.usage_date)

audit_logs = Table(
    "audit_logs",
    STORE_METADATA,
    Column("id", Text, primary_key=True),
    Column("actor_user_id", Text, ForeignKey("users.id")),
    Column("target_user_id", Text, ForeignKey("users.id")),
    Column("event_type", Text, nullable=False),
    Column("ip_address", Text, nullable=False, default=""),
    Column("user_agent", Text, nullable=False, default=""),
    Column("metadata_json", Text, nullable=False),
    Column("created_at", Text, nullable=False),
)
Index("ix_audit_logs_actor_user_id", audit_logs.c.actor_user_id)
Index("ix_audit_logs_target_user_id", audit_logs.c.target_user_id)
Index("ix_audit_logs_event_type", audit_logs.c.event_type)
Index("ix_audit_logs_created_at", audit_logs.c.created_at)

datasets = Table(
    "datasets",
    STORE_METADATA,
    Column("id", Text, primary_key=True),
    Column("name", Text, nullable=False),
    Column("task_type", Text, nullable=False),
    Column("source", Text, nullable=False),
    Column("format", Text, nullable=False),
    Column("root_path", Text, nullable=False),
    Column("metadata_json", Text, nullable=False),
    Column("owner_user_id", Text, ForeignKey("users.id")),
    Column("created_at", Text, nullable=False),
)
Index("ix_datasets_owner_user_id", datasets.c.owner_user_id)

experiments = Table(
    "experiments",
    STORE_METADATA,
    Column("id", Text, primary_key=True),
    Column("dataset_id", Text, ForeignKey("datasets.id"), nullable=False),
    Column("owner_user_id", Text, ForeignKey("users.id")),
    Column("name", Text, nullable=False),
    Column("name_kind", Text),
    Column("name_base", Text),
    Column("name_base_kind", Text),
    Column("name_qualifier", Text),
    Column("rerun_count", Integer, nullable=False, default=0),
    Column("rerun_source_experiment_id", Text),
    Column("task_type", Text, nullable=False),
    Column("config_json", Text, nullable=False),
    Column("status", Text, nullable=False),
    Column("log_path", Text, nullable=False),
    Column("artifact_dir", Text, nullable=False),
    Column("metrics_json", Text, nullable=False),
    Column("model_id", Text),
    Column("error", Text, nullable=False),
    Column(
        "failure_json",
        Text,
        nullable=False,
        default="{}",
        server_default="{}",
    ),
    Column(
        "resume_json",
        Text,
        nullable=False,
        default="{}",
        server_default="{}",
    ),
    Column("created_at", Text, nullable=False),
    Column("updated_at", Text, nullable=False),
    Column("started_at", Text),
    Column("finished_at", Text),
)
Index("ix_experiments_owner_user_id", experiments.c.owner_user_id)
Index("ix_experiments_dataset_id", experiments.c.dataset_id)

models = Table(
    "models",
    STORE_METADATA,
    Column("id", Text, primary_key=True),
    Column("experiment_id", Text, ForeignKey("experiments.id"), nullable=False),
    Column("owner_user_id", Text, ForeignKey("users.id")),
    Column("name", Text, nullable=False),
    Column("name_is_custom", Integer, nullable=False, default=0),
    Column("task_type", Text, nullable=False),
    Column("bundle_path", Text, nullable=False),
    Column("supports_inference", Integer, nullable=False),
    Column("metadata_json", Text, nullable=False),
    Column("created_at", Text, nullable=False),
)
Index("ix_models_owner_user_id", models.c.owner_user_id)
Index("ix_models_experiment_id", models.c.experiment_id)

inference_runs = Table(
    "inference_runs",
    STORE_METADATA,
    Column("id", Text, primary_key=True),
    Column("model_id", Text, ForeignKey("models.id"), nullable=False),
    Column("owner_user_id", Text, ForeignKey("users.id")),
    Column("input_kind", Text, nullable=False),
    Column("input_count", Integer, nullable=False),
    Column("prediction_count", Integer, nullable=False),
    Column("request_json", Text, nullable=False),
    Column("response_json", Text, nullable=False),
    Column("created_at", Text, nullable=False),
)
Index("ix_inference_runs_owner_user_id", inference_runs.c.owner_user_id)
Index("ix_inference_runs_model_id", inference_runs.c.model_id)

ai_provider_configs = Table(
    "ai_provider_configs",
    STORE_METADATA,
    Column("id", Text, primary_key=True),
    Column("owner_user_id", Text, ForeignKey("users.id")),
    Column("provider", Text, nullable=False),
    Column("display_name", Text, nullable=False),
    Column("base_url", Text, nullable=False),
    Column("model", Text, nullable=False),
    Column("api_key_ciphertext", Text, nullable=False),
    Column("api_key_last4", Text, nullable=False),
    Column("capabilities_json", Text, nullable=False, default="{}"),
    Column("enabled", Integer, nullable=False, default=1),
    Column("created_at", Text, nullable=False),
    Column("updated_at", Text, nullable=False),
)
Index("ix_ai_provider_configs_owner_user_id", ai_provider_configs.c.owner_user_id)
Index("ix_ai_provider_configs_provider", ai_provider_configs.c.provider)

agent_chat_conversations = Table(
    "agent_chat_conversations",
    STORE_METADATA,
    Column("id", Text, primary_key=True),
    Column("owner_user_id", Text, ForeignKey("users.id", ondelete="CASCADE")),
    Column("scope", Text, nullable=False, default="agent"),
    Column("experiment_id", Text, ForeignKey("experiments.id", ondelete="CASCADE")),
    Column("title", Text, nullable=False),
    Column("pinned_at", Text),
    Column(
        "last_provider_config_id",
        Text,
        ForeignKey("ai_provider_configs.id", ondelete="SET NULL"),
    ),
    Column("last_provider", Text, nullable=False, default=""),
    Column("last_model", Text, nullable=False, default=""),
    Column("last_message_preview", Text, nullable=False, default=""),
    Column("created_at", Text, nullable=False),
    Column("updated_at", Text, nullable=False),
)
Index(
    "ix_agent_chat_conversations_owner_updated",
    agent_chat_conversations.c.owner_user_id,
    agent_chat_conversations.c.updated_at,
)
Index(
    "ix_agent_chat_conversations_owner_pinned",
    agent_chat_conversations.c.owner_user_id,
    agent_chat_conversations.c.pinned_at,
)
Index(
    "ix_agent_chat_conversations_owner_scope_experiment_updated",
    agent_chat_conversations.c.owner_user_id,
    agent_chat_conversations.c.scope,
    agent_chat_conversations.c.experiment_id,
    agent_chat_conversations.c.updated_at,
)

agent_chat_messages = Table(
    "agent_chat_messages",
    STORE_METADATA,
    Column("id", Text, primary_key=True),
    Column(
        "conversation_id",
        Text,
        ForeignKey("agent_chat_conversations.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("ordinal", Integer, nullable=False),
    Column("role", Text, nullable=False),
    Column("content", Text, nullable=False, default=""),
    Column("status", Text, nullable=False),
    Column("kind", Text, nullable=False, default="chat"),
    Column(
        "provider_config_id",
        Text,
        ForeignKey("ai_provider_configs.id", ondelete="SET NULL"),
    ),
    Column("provider", Text, nullable=False, default=""),
    Column("model", Text, nullable=False, default=""),
    Column("finish_reason", Text, nullable=False, default=""),
    Column("error", Text, nullable=False, default=""),
    Column("usage_json", Text, nullable=False, default="{}"),
    Column("web_search_json", Text, nullable=False, default="{}"),
    Column("payload_json", Text, nullable=False, default="{}"),
    Column("context_manifest_json", Text, nullable=False, default="{}"),
    Column("created_at", Text, nullable=False),
    Column("updated_at", Text, nullable=False),
)
Index(
    "uq_agent_chat_messages_conversation_ordinal",
    agent_chat_messages.c.conversation_id,
    agent_chat_messages.c.ordinal,
    unique=True,
)
Index(
    "ix_agent_chat_messages_conversation_ordinal",
    agent_chat_messages.c.conversation_id,
    agent_chat_messages.c.ordinal,
)

agent_campaigns = Table(
    "agent_campaigns",
    STORE_METADATA,
    Column("id", Text, primary_key=True),
    Column("owner_user_id", Text, ForeignKey("users.id")),
    Column("task_type", Text, nullable=False),
    Column("dataset_id", Text, nullable=False),
    Column("objective", Text, nullable=False),
    Column("status", Text, nullable=False),
    Column("max_rounds", Integer, nullable=False, default=3),
    Column("current_round_index", Integer, nullable=False, default=0),
    Column("created_at", Text, nullable=False),
    Column("updated_at", Text, nullable=False),
)
Index("ix_agent_campaigns_owner_user_id", agent_campaigns.c.owner_user_id)
Index("ix_agent_campaigns_dataset_id", agent_campaigns.c.dataset_id)
Index("ix_agent_campaigns_status", agent_campaigns.c.status)

agent_execution_plans = Table(
    "agent_execution_plans",
    STORE_METADATA,
    Column("id", Text, primary_key=True),
    Column("owner_user_id", Text, ForeignKey("users.id")),
    Column("campaign_id", Text, ForeignKey("agent_campaigns.id")),
    Column("round_index", Integer, nullable=False, default=0),
    Column("provider_config_id", Text),
    Column("provider", Text, nullable=False, default=""),
    Column("model", Text, nullable=False, default=""),
    Column("task_type", Text, nullable=False),
    Column("dataset_id", Text, nullable=False),
    Column("objective", Text, nullable=False),
    Column("status", Text, nullable=False),
    Column("actions_json", Text, nullable=False),
    Column("execution_json", Text, nullable=False),
    Column("review_json", Text, nullable=False),
    Column("usage_json", Text, nullable=False),
    Column("context_manifest_json", Text, nullable=False),
    Column("trace_json", Text, nullable=False, default="{}"),
    Column("warnings_json", Text, nullable=False),
    Column("created_at", Text, nullable=False),
    Column("updated_at", Text, nullable=False),
    Column("executed_at", Text),
    Column("reviewed_at", Text),
)
Index("ix_agent_execution_plans_owner_user_id", agent_execution_plans.c.owner_user_id)
Index("ix_agent_execution_plans_campaign_id", agent_execution_plans.c.campaign_id)
Index("ix_agent_execution_plans_dataset_id", agent_execution_plans.c.dataset_id)
Index("ix_agent_execution_plans_provider_config_id", agent_execution_plans.c.provider_config_id)
Index("ix_agent_execution_plans_status", agent_execution_plans.c.status)


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def quota_usage_date():
    return datetime.now(timezone.utc).date().isoformat()


def _dump(value):
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True)


def _load(value, default=None):
    if value in [None, ""]:
        return {} if default is None else default
    return json.loads(value)


def generated_agent_experiment_name_records(actions, execution):
    actions = actions if isinstance(actions, list) else []
    execution = execution if isinstance(execution, dict) else {}
    execution_ids = {}
    for entry in execution.get("created_experiments") or []:
        if not isinstance(entry, dict):
            continue
        action_id = str(entry.get("action_id") or "").strip()
        experiment = entry.get("experiment") if isinstance(entry.get("experiment"), dict) else {}
        experiment_id = str(entry.get("experiment_id") or experiment.get("id") or "").strip()
        if action_id and experiment_id:
            execution_ids[action_id] = experiment_id

    records = {}
    for action in actions:
        if not isinstance(action, dict) or str(action.get("type") or "") != "create_experiment":
            continue
        action_id = str(action.get("id") or "").strip()
        result = action.get("result") if isinstance(action.get("result"), dict) else {}
        experiment_id = str(
            result.get("experiment_id") or execution_ids.get(action_id) or ""
        ).strip()
        title = str(action.get("title") or "").strip()
        lane = str(action.get("lane") or "").strip()
        if not experiment_id or not title:
            continue
        names = {title}
        if lane:
            names.add(f"Agent {lane}: {title}"[:180])
        records[experiment_id] = {
            "names": names,
            "qualifier": lane if lane in GENERATED_NAME_QUALIFIERS else None,
        }
    return records


def _metadata_requires_sample_profile(metadata):
    bundle_type = str(metadata.get("bundle_type", "") or "")
    return (
        bundle_type in {"sequence_nn", "sequence_classical", "sequence_embedding_classical"}
        and uses_profile_features(metadata.get("args") or {})
        and not (metadata.get("profile_generation") or {}).get("subject_fasta")
    )


def _metadata_supports_raw_sequence_inference(row_supports_inference, metadata):
    supports = bool(row_supports_inference) and bool(
        metadata.get("supports_inference", row_supports_inference)
    )
    if supports and _metadata_requires_sample_profile(metadata):
        supports = False
    return supports


def _numeric_metric(value):
    if value in [None, ""]:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric == numeric else None


def _metric_for_split(summary, split, metric):
    prefix = "val" if split in {"validation", "val"} else "test"
    key = f"{prefix}_{metric}"
    value = _numeric_metric(summary.get(key))
    if value is not None:
        return value, key
    if prefix == "val" and metric == "f1":
        value = _numeric_metric(summary.get("best_val_f1"))
        if value is not None:
            return value, "best_val_f1"
    if prefix == "val" and metric == "auc":
        value = _numeric_metric(summary.get("best_val_auc"))
        if value is not None:
            return value, "best_val_auc"
    return None, ""


def _leaderboard_score(summary, split, metric):
    if split == "auto":
        value, source = _metric_for_split(summary, "test", metric)
        if value is not None:
            return value, source
        return _metric_for_split(summary, "validation", metric)
    return _metric_for_split(summary, split, metric)


def _split_metrics(summary, prefix):
    return {
        metric: _numeric_metric(summary.get(f"{prefix}_{metric}"))
        for metric in LEADERBOARD_METRICS
        if _numeric_metric(summary.get(f"{prefix}_{metric}")) is not None
    }


def _duration_seconds(started_at, finished_at):
    if not started_at or not finished_at:
        return None
    try:
        started = datetime.fromisoformat(started_at)
        finished = datetime.fromisoformat(finished_at)
    except ValueError:
        return None
    seconds = (finished - started).total_seconds()
    return seconds if seconds >= 0 else None


def _leaderboard_config_summary(config):
    return {
        key: config[key]
        for key in LEADERBOARD_CONFIG_KEYS
        if key in config and config[key] not in [None, ""]
    }


def normalize_email(email):
    return str(email or "").strip().lower()


def default_display_name(email):
    prefix = normalize_email(email).split("@", 1)[0].strip()
    return prefix[:40] or "User"


def _is_admin(user):
    return bool(user and user.get("role") == "admin")


def _owns(row, user):
    return bool(user and row.get("owner_user_id") and row.get("owner_user_id") == user.get("id"))


def _owner_scoped_user(user):
    if not _is_admin(user):
        return user
    scoped = dict(user)
    scoped["role"] = "user"
    return scoped


def _database_url(database):
    text_value = str(database)
    if "://" in text_value:
        return text_value
    return f"sqlite:///{Path(database)}"


class MetadataStore:
    def __init__(self, database):
        self.database = database
        self.database_url = _database_url(database)
        self.database_path = None
        if self.database_url.startswith("sqlite:///"):
            self.database_path = Path(self.database_url.replace("sqlite:///", "", 1))
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connect_args = (
            {"check_same_thread": False} if self.database_url.startswith("sqlite:///") else {}
        )
        self.engine = create_engine(self.database_url, future=True, connect_args=connect_args)
        if self.engine.dialect.name == "sqlite":
            event.listen(self.engine, "connect", self._enable_sqlite_foreign_keys)
        self.init_schema()

    @staticmethod
    def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.close()

    @contextmanager
    def connect(self):
        with self.engine.begin() as conn:
            yield conn

    def init_schema(self):
        STORE_METADATA.create_all(self.engine)
        self._ensure_user_profile_columns()
        self._ensure_owner_columns()
        self._ensure_agent_columns()
        self._ensure_localization_columns()
        self._ensure_experiment_failure_column()
        self._ensure_experiment_resume_column()
        self._recover_interrupted_agent_chats()

    def _recover_interrupted_agent_chats(self):
        """A process restart cannot resume an upstream HTTP stream."""
        now = utcnow()
        with self.engine.begin() as conn:
            conn.execute(
                update(agent_chat_messages)
                .where(agent_chat_messages.c.status == "streaming")
                .values(status="stopped", finish_reason="interrupted", updated_at=now)
            )

    def _ensure_user_profile_columns(self):
        inspector = inspect(self.engine)
        table_names = set(inspector.get_table_names())
        if "users" not in table_names:
            return
        columns = {column["name"] for column in inspector.get_columns("users")}
        profile_columns = {
            "display_name": "TEXT NOT NULL DEFAULT ''",
            "avatar_object_key": "TEXT",
            "avatar_content_type": "TEXT",
            "avatar_updated_at": "TEXT",
            "preferred_locale": "TEXT NOT NULL DEFAULT 'en'",
        }
        with self.engine.begin() as conn:
            for column_name, definition in profile_columns.items():
                if column_name not in columns:
                    conn.execute(text(f"ALTER TABLE users ADD COLUMN {column_name} {definition}"))
            if "display_name" not in columns:
                rows = conn.execute(select(users.c.id, users.c.email)).mappings().all()
                for row in rows:
                    conn.execute(
                        update(users)
                        .where(users.c.id == row["id"])
                        .values(display_name=default_display_name(row["email"]))
                    )

    def _ensure_experiment_failure_column(self):
        inspector = inspect(self.engine)
        if "experiments" not in set(inspector.get_table_names()):
            return
        columns = {column["name"] for column in inspector.get_columns("experiments")}
        if "failure_json" not in columns:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE experiments ADD COLUMN failure_json TEXT NOT NULL DEFAULT '{}'"
                    )
                )

    def _ensure_experiment_resume_column(self):
        inspector = inspect(self.engine)
        if "experiments" not in set(inspector.get_table_names()):
            return
        columns = {column["name"] for column in inspector.get_columns("experiments")}
        if "resume_json" not in columns:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE experiments ADD COLUMN resume_json TEXT NOT NULL DEFAULT '{}'"
                    )
                )

    def _ensure_localization_columns(self):
        inspector = inspect(self.engine)
        table_names = set(inspector.get_table_names())
        with self.engine.begin() as conn:
            if "experiments" in table_names:
                experiment_columns = {
                    column["name"] for column in inspector.get_columns("experiments")
                }
                additions = {
                    "name_kind": "TEXT",
                    "name_base": "TEXT",
                    "name_base_kind": "TEXT",
                    "name_qualifier": "TEXT",
                    "rerun_count": "INTEGER NOT NULL DEFAULT 0",
                    "rerun_source_experiment_id": "TEXT",
                }
                for column_name, definition in additions.items():
                    if column_name not in experiment_columns:
                        conn.execute(
                            text(f"ALTER TABLE experiments ADD COLUMN {column_name} {definition}")
                        )
                if "name_kind" not in experiment_columns:
                    rows = conn.execute(select(experiments.c.id, experiments.c.name)).mappings()
                    for row in rows:
                        base, count = generated_rerun_name_parts(row["name"])
                        if count:
                            conn.execute(
                                update(experiments)
                                .where(experiments.c.id == row["id"])
                                .values(
                                    name_kind="rerun",
                                    name_base=base,
                                    name_base_kind="literal",
                                    rerun_count=count,
                                )
                            )
                if "name_base_kind" not in experiment_columns:
                    conn.execute(
                        update(experiments)
                        .where(
                            and_(
                                experiments.c.name_kind == "rerun",
                                experiments.c.name_base.is_not(None),
                            )
                        )
                        .values(name_base_kind="literal")
                    )
                    if "agent_execution_plans" in table_names:
                        generated_records = {}
                        rows = conn.execute(
                            select(
                                agent_execution_plans.c.actions_json,
                                agent_execution_plans.c.execution_json,
                            )
                        ).mappings()
                        for row in rows:
                            try:
                                generated_records.update(
                                    generated_agent_experiment_name_records(
                                        _load(row["actions_json"], []),
                                        _load(row["execution_json"], {}),
                                    )
                                )
                            except (TypeError, ValueError, json.JSONDecodeError):
                                continue
                        if generated_records:
                            experiment_rows = conn.execute(
                                select(
                                    experiments.c.id,
                                    experiments.c.name,
                                    experiments.c.name_kind,
                                ).where(experiments.c.id.in_(generated_records))
                            ).mappings()
                            for row in experiment_rows:
                                record = generated_records[row["id"]]
                                if row["name_kind"] is None and row["name"] in record["names"]:
                                    conn.execute(
                                        update(experiments)
                                        .where(experiments.c.id == row["id"])
                                        .values(
                                            name_kind="config",
                                            name_qualifier=record["qualifier"],
                                        )
                                    )

            if "models" in table_names:
                model_columns = {column["name"] for column in inspector.get_columns("models")}
                if "name_is_custom" not in model_columns:
                    conn.execute(
                        text(
                            "ALTER TABLE models ADD COLUMN "
                            "name_is_custom INTEGER NOT NULL DEFAULT 0"
                        )
                    )
                    experiment_names = {
                        row["id"]: row["name"]
                        for row in conn.execute(
                            select(experiments.c.id, experiments.c.name)
                        ).mappings()
                    }
                    rows = conn.execute(
                        select(models.c.id, models.c.experiment_id, models.c.name)
                    ).mappings()
                    for row in rows:
                        inherited_name = experiment_names.get(row["experiment_id"])
                        if row["name"] != inherited_name:
                            conn.execute(
                                update(models)
                                .where(models.c.id == row["id"])
                                .values(name_is_custom=1)
                            )

    def _ensure_owner_columns(self):
        inspector = inspect(self.engine)
        table_names = set(inspector.get_table_names())
        owner_tables = {
            "datasets": "datasets",
            "experiments": "experiments",
            "models": "models",
            "inference_runs": "inference_runs",
        }
        with self.engine.begin() as conn:
            for table_name in owner_tables:
                if table_name not in table_names:
                    continue
                columns = {column["name"] for column in inspector.get_columns(table_name)}
                if "owner_user_id" not in columns:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN owner_user_id TEXT"))

    def _ensure_agent_columns(self):
        inspector = inspect(self.engine)
        table_names = set(inspector.get_table_names())
        with self.engine.begin() as conn:
            if "ai_provider_configs" in table_names:
                columns = {
                    column["name"] for column in inspector.get_columns("ai_provider_configs")
                }
                if "capabilities_json" not in columns:
                    conn.execute(
                        text(
                            "ALTER TABLE ai_provider_configs ADD COLUMN capabilities_json TEXT NOT NULL DEFAULT '{}'"
                        )
                    )
            if "agent_execution_plans" in table_names:
                columns = {
                    column["name"] for column in inspector.get_columns("agent_execution_plans")
                }
                if "campaign_id" not in columns:
                    conn.execute(
                        text("ALTER TABLE agent_execution_plans ADD COLUMN campaign_id TEXT")
                    )
                if "round_index" not in columns:
                    conn.execute(
                        text(
                            "ALTER TABLE agent_execution_plans ADD COLUMN round_index INTEGER NOT NULL DEFAULT 0"
                        )
                    )
                if "trace_json" not in columns:
                    conn.execute(
                        text(
                            "ALTER TABLE agent_execution_plans ADD COLUMN trace_json TEXT NOT NULL DEFAULT '{}'"
                        )
                    )
            if "agent_chat_conversations" in table_names:
                columns = {
                    column["name"] for column in inspector.get_columns("agent_chat_conversations")
                }
                additions = {
                    "scope": "TEXT NOT NULL DEFAULT 'agent'",
                    "experiment_id": "TEXT",
                }
                for column_name, definition in additions.items():
                    if column_name not in columns:
                        conn.execute(
                            text(
                                "ALTER TABLE agent_chat_conversations "
                                f"ADD COLUMN {column_name} {definition}"
                            )
                        )
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS "
                        "ix_agent_chat_conversations_owner_scope_experiment_updated "
                        "ON agent_chat_conversations "
                        "(owner_user_id, scope, experiment_id, updated_at)"
                    )
                )
            if "agent_chat_messages" in table_names:
                columns = {
                    column["name"] for column in inspector.get_columns("agent_chat_messages")
                }
                additions = {
                    "web_search_json": "TEXT NOT NULL DEFAULT '{}'",
                    "kind": "TEXT NOT NULL DEFAULT 'chat'",
                    "payload_json": "TEXT NOT NULL DEFAULT '{}'",
                    "context_manifest_json": "TEXT NOT NULL DEFAULT '{}'",
                }
                for column_name, definition in additions.items():
                    if column_name not in columns:
                        conn.execute(
                            text(
                                "ALTER TABLE agent_chat_messages "
                                f"ADD COLUMN {column_name} {definition}"
                            )
                        )

    def _fetch_one(self, stmt):
        with self.connect() as conn:
            return conn.execute(stmt).mappings().first()

    def _fetch_all(self, stmt):
        with self.connect() as conn:
            return list(conn.execute(stmt).mappings().all())

    def _visible_dataset(self, row, user):
        if row is None:
            return False
        if user is None or _is_admin(user):
            return True
        return row.get("source") == "builtin" or _owns(row, user)

    def _visible_owned_row(self, row, user):
        if row is None:
            return False
        if user is None or _is_admin(user):
            return True
        return _owns(row, user)

    def create_user(
        self,
        email,
        password_hash,
        role="user",
        email_verified=False,
        status="active",
        user_id=None,
        display_name=None,
        preferred_locale="en",
    ):
        email = normalize_email(email)
        if not email:
            raise ValueError("email is required")
        now = utcnow()
        user_id = user_id or uuid.uuid4().hex
        payload = {
            "id": user_id,
            "email": email,
            "password_hash": password_hash,
            "display_name": str(display_name or default_display_name(email)).strip()[:40],
            "preferred_locale": (
                str(preferred_locale or "en").strip().lower()
                if str(preferred_locale or "en").strip().lower() in SUPPORTED_LOCALES
                else "en"
            ),
            "email_verified": int(bool(email_verified)),
            "role": role,
            "status": status,
            "created_at": now,
            "updated_at": now,
        }
        try:
            with self.connect() as conn:
                conn.execute(insert(users).values(**payload))
        except IntegrityError as exc:
            raise ValueError("email already registered") from exc
        return self.get_user(user_id)

    def ensure_initial_admin(self, email, password_hash):
        email = normalize_email(email)
        if not email:
            return None
        user = self.get_user_by_email(email)
        if user:
            updates = {
                "role": "admin",
                "status": "active",
                "email_verified": 1,
                "updated_at": utcnow(),
            }
            with self.connect() as conn:
                conn.execute(update(users).where(users.c.id == user["id"]).values(**updates))
            return self.get_user(user["id"])
        return self.create_user(email, password_hash, role="admin", email_verified=True)

    def get_user(self, user_id):
        row = self._fetch_one(select(users).where(users.c.id == user_id))
        return self._user_row(row) if row else None

    def get_user_by_email(self, email):
        row = self._fetch_one(select(users).where(users.c.email == normalize_email(email)))
        return self._user_row(row, include_password=True) if row else None

    def verify_user_email(self, user_id):
        with self.connect() as conn:
            conn.execute(
                update(users)
                .where(users.c.id == user_id)
                .values(email_verified=1, status="active", updated_at=utcnow())
            )
        return self.get_user(user_id)

    def update_user_password(self, user_id, password_hash):
        with self.connect() as conn:
            conn.execute(
                update(users)
                .where(users.c.id == user_id)
                .values(password_hash=password_hash, updated_at=utcnow())
            )
        return self.get_user(user_id)

    def update_user_profile(self, user_id, display_name):
        with self.connect() as conn:
            conn.execute(
                update(users)
                .where(users.c.id == user_id)
                .values(display_name=display_name, updated_at=utcnow())
            )
        return self.get_user(user_id)

    def update_user_locale(self, user_id, preferred_locale):
        preferred_locale = str(preferred_locale or "").strip().lower()
        if preferred_locale not in SUPPORTED_LOCALES:
            raise ValueError("unsupported locale")
        with self.connect() as conn:
            conn.execute(
                update(users)
                .where(users.c.id == user_id)
                .values(preferred_locale=preferred_locale, updated_at=utcnow())
            )
        return self.get_user(user_id)

    def create_email_code(self, user_id, purpose, code_hash, expires_at):
        now = utcnow()
        code_id = uuid.uuid4().hex
        with self.connect() as conn:
            conn.execute(
                update(email_verification_codes)
                .where(
                    and_(
                        email_verification_codes.c.user_id == user_id,
                        email_verification_codes.c.purpose == purpose,
                        email_verification_codes.c.consumed_at.is_(None),
                    )
                )
                .values(consumed_at=now)
            )
            conn.execute(
                insert(email_verification_codes).values(
                    id=code_id,
                    user_id=user_id,
                    purpose=purpose,
                    code_hash=code_hash,
                    expires_at=expires_at,
                    attempts=0,
                    consumed_at=None,
                    created_at=now,
                )
            )
        return self.get_email_code(code_id)

    def get_email_code(self, code_id):
        row = self._fetch_one(
            select(email_verification_codes).where(email_verification_codes.c.id == code_id)
        )
        return self._email_code_row(row) if row else None

    def get_active_email_code(self, user_id, purpose):
        row = self._fetch_one(
            select(email_verification_codes)
            .where(
                and_(
                    email_verification_codes.c.user_id == user_id,
                    email_verification_codes.c.purpose == purpose,
                    email_verification_codes.c.consumed_at.is_(None),
                )
            )
            .order_by(email_verification_codes.c.created_at.desc())
        )
        return self._email_code_row(row) if row else None

    def increment_email_code_attempts(self, code_id):
        with self.connect() as conn:
            conn.execute(
                update(email_verification_codes)
                .where(email_verification_codes.c.id == code_id)
                .values(attempts=email_verification_codes.c.attempts + 1)
            )
        return self.get_email_code(code_id)

    def consume_email_code(self, code_id):
        with self.connect() as conn:
            conn.execute(
                update(email_verification_codes)
                .where(email_verification_codes.c.id == code_id)
                .values(consumed_at=utcnow())
            )
        return self.get_email_code(code_id)

    def create_session(self, user_id, token_hash, expires_at, user_agent=""):
        now = utcnow()
        session_id = uuid.uuid4().hex
        with self.connect() as conn:
            conn.execute(
                insert(sessions).values(
                    id=session_id,
                    user_id=user_id,
                    token_hash=token_hash,
                    expires_at=expires_at,
                    user_agent=user_agent or "",
                    created_at=now,
                    last_seen_at=now,
                    revoked_at=None,
                )
            )
        return self.get_session_by_hash(token_hash)

    def get_session_by_hash(self, token_hash):
        row = self._fetch_one(select(sessions).where(sessions.c.token_hash == token_hash))
        return self._session_row(row) if row else None

    def touch_session(self, session_id):
        with self.connect() as conn:
            conn.execute(
                update(sessions).where(sessions.c.id == session_id).values(last_seen_at=utcnow())
            )

    def revoke_session(self, token_hash):
        with self.connect() as conn:
            conn.execute(
                update(sessions)
                .where(and_(sessions.c.token_hash == token_hash, sessions.c.revoked_at.is_(None)))
                .values(revoked_at=utcnow())
            )

    def revoke_sessions_for_user(self, user_id):
        with self.connect() as conn:
            result = conn.execute(
                update(sessions)
                .where(and_(sessions.c.user_id == user_id, sessions.c.revoked_at.is_(None)))
                .values(revoked_at=utcnow())
            )
        return int(result.rowcount or 0)

    def list_user_sessions(self, user_id, limit=20):
        try:
            limit = max(1, min(int(limit), 100))
        except (TypeError, ValueError):
            limit = 20
        rows = self._fetch_all(
            select(sessions)
            .where(sessions.c.user_id == user_id)
            .order_by(sessions.c.created_at.desc())
            .limit(limit)
        )
        return [self._admin_session_row(row) for row in rows]

    def list_user_email_codes(self, user_id, limit=20):
        try:
            limit = max(1, min(int(limit), 100))
        except (TypeError, ValueError):
            limit = 20
        rows = self._fetch_all(
            select(email_verification_codes)
            .where(email_verification_codes.c.user_id == user_id)
            .order_by(email_verification_codes.c.created_at.desc())
            .limit(limit)
        )
        return [self._admin_email_code_row(row) for row in rows]

    def list_admin_users(
        self,
        search="",
        role="",
        status="",
        email_verified=None,
        sort_by="created_at",
        sort_order="desc",
        limit=50,
        offset=0,
    ):
        conditions = []
        search = str(search or "").strip().lower()
        if search:
            pattern = f"%{search}%"
            conditions.append(
                or_(
                    func.lower(users.c.email).like(pattern),
                    func.lower(users.c.display_name).like(pattern),
                )
            )
        if role:
            conditions.append(users.c.role == str(role).strip())
        if status:
            conditions.append(users.c.status == str(status).strip())
        if email_verified is not None:
            conditions.append(users.c.email_verified == int(bool(email_verified)))

        try:
            limit = max(1, min(int(limit), 200))
        except (TypeError, ValueError):
            limit = 50
        try:
            offset = max(0, int(offset))
        except (TypeError, ValueError):
            offset = 0

        sort_columns = {
            "created_at": users.c.created_at,
            "display_name": func.lower(users.c.display_name),
            "email": func.lower(users.c.email),
            "role": users.c.role,
            "status": users.c.status,
        }
        sort_column = sort_columns.get(str(sort_by or "").strip(), users.c.created_at)
        sort_direction = str(sort_order or "desc").strip().lower()
        primary_sort = sort_column.asc() if sort_direction == "asc" else sort_column.desc()

        where_clause = and_(*conditions) if conditions else None
        count_stmt = select(func.count().label("count")).select_from(users)
        list_stmt = (
            select(users).order_by(primary_sort, users.c.id.asc()).limit(limit).offset(offset)
        )
        if where_clause is not None:
            count_stmt = count_stmt.where(where_clause)
            list_stmt = list_stmt.where(where_clause)
        with self.connect() as conn:
            total_row = conn.execute(count_stmt).mappings().first()
            rows = list(conn.execute(list_stmt).mappings().all())
        items = self._admin_user_summaries([self._user_row(row) for row in rows])
        return {
            "items": items,
            "total": int(total_row["count"]) if total_row else 0,
            "limit": limit,
            "offset": offset,
        }

    def _admin_user_summaries(self, user_rows):
        if not user_rows:
            return []
        user_ids = [user["id"] for user in user_rows]
        usage_date = quota_usage_date()
        resource_keys = [
            "datasets",
            "experiments",
            "active_experiments",
            "models",
            "inference_runs",
            "agent_campaigns",
            "agent_execution_plans",
            "agent_chat_conversations",
        ]
        resources = {user_id: {key: 0 for key in resource_keys} for user_id in user_ids}

        def grouped_counts(conn, table, owner_column, key, condition=None):
            statement = (
                select(owner_column.label("owner_user_id"), func.count().label("count"))
                .select_from(table)
                .where(owner_column.in_(user_ids))
                .group_by(owner_column)
            )
            if condition is not None:
                statement = statement.where(condition)
            for row in conn.execute(statement).mappings().all():
                resources[row["owner_user_id"]][key] = int(row["count"] or 0)

        with self.connect() as conn:
            quota_rows = (
                conn.execute(select(user_quotas).where(user_quotas.c.user_id.in_(user_ids)))
                .mappings()
                .all()
            )
            usage_rows = (
                conn.execute(
                    select(quota_usage).where(
                        and_(
                            quota_usage.c.user_id.in_(user_ids),
                            quota_usage.c.usage_date == usage_date,
                        )
                    )
                )
                .mappings()
                .all()
            )
            grouped_counts(conn, datasets, datasets.c.owner_user_id, "datasets")
            grouped_counts(conn, experiments, experiments.c.owner_user_id, "experiments")
            grouped_counts(
                conn,
                experiments,
                experiments.c.owner_user_id,
                "active_experiments",
                experiments.c.status.in_(["queued", "running"]),
            )
            grouped_counts(conn, models, models.c.owner_user_id, "models")
            grouped_counts(conn, inference_runs, inference_runs.c.owner_user_id, "inference_runs")
            grouped_counts(
                conn, agent_campaigns, agent_campaigns.c.owner_user_id, "agent_campaigns"
            )
            grouped_counts(
                conn,
                agent_execution_plans,
                agent_execution_plans.c.owner_user_id,
                "agent_execution_plans",
            )
            grouped_counts(
                conn,
                agent_chat_conversations,
                agent_chat_conversations.c.owner_user_id,
                "agent_chat_conversations",
            )

        quotas_by_user = {row["user_id"]: self._quota_row(row) for row in quota_rows}
        usage_by_user = {row["user_id"]: self._quota_usage_row(row) for row in usage_rows}
        return [
            {
                **user,
                "quota": quotas_by_user.get(
                    user["id"],
                    {"user_id": user["id"], **DEFAULT_USER_QUOTAS, "updated_at": None},
                ),
                "usage": usage_by_user.get(
                    user["id"],
                    {
                        "user_id": user["id"],
                        "usage_date": usage_date,
                        "experiments_created": 0,
                        "inference_requests": 0,
                        "inference_inputs": 0,
                        "updated_at": None,
                    },
                ),
                "resource_counts": resources[user["id"]],
            }
            for user in user_rows
        ]

    def get_admin_overview(self):
        now = datetime.now(timezone.utc)
        generated_at = now.isoformat()
        new_user_cutoff = (now - timedelta(days=7)).isoformat()
        security_cutoff = (now - timedelta(hours=24)).isoformat()
        usage_date = quota_usage_date()

        with self.connect() as conn:
            user_counts = (
                conn.execute(
                    select(
                        func.count().label("total"),
                        func.sum(case((users.c.status == "active", 1), else_=0)).label("active"),
                        func.sum(case((users.c.status == "disabled", 1), else_=0)).label(
                            "disabled"
                        ),
                        func.sum(case((users.c.email_verified == 0, 1), else_=0)).label(
                            "unverified"
                        ),
                        func.sum(case((users.c.created_at >= new_user_cutoff, 1), else_=0)).label(
                            "new_last_7_days"
                        ),
                    ).select_from(users)
                )
                .mappings()
                .one()
            )
            usage_counts = (
                conn.execute(
                    select(
                        func.coalesce(func.sum(quota_usage.c.experiments_created), 0).label(
                            "experiments_created_today"
                        ),
                        func.coalesce(func.sum(quota_usage.c.inference_requests), 0).label(
                            "inference_requests_today"
                        ),
                        func.coalesce(func.sum(quota_usage.c.inference_inputs), 0).label(
                            "inference_inputs_today"
                        ),
                    )
                    .select_from(quota_usage)
                    .where(quota_usage.c.usage_date == usage_date)
                )
                .mappings()
                .one()
            )
            active_experiments = conn.execute(
                select(func.count())
                .select_from(experiments)
                .where(experiments.c.status.in_(["queued", "running"]))
            ).scalar_one()
            security_counts = (
                conn.execute(
                    select(
                        func.sum(
                            case((audit_logs.c.event_type == "auth.login_failed", 1), else_=0)
                        ).label("failed_logins"),
                        func.sum(case((audit_logs.c.event_type.like("admin.%"), 1), else_=0)).label(
                            "admin_actions"
                        ),
                        func.sum(
                            case((audit_logs.c.event_type == "admin.sessions_revoked", 1), else_=0)
                        ).label("session_revoke_events"),
                        func.sum(
                            case((audit_logs.c.event_type == "admin.user_deleted", 1), else_=0)
                        ).label("user_deletions"),
                    )
                    .select_from(audit_logs)
                    .where(audit_logs.c.created_at >= security_cutoff)
                )
                .mappings()
                .one()
            )

        def integer(value):
            return int(value or 0)

        return {
            "generated_at": generated_at,
            "usage_date": usage_date,
            "users": {key: integer(user_counts[key]) for key in user_counts},
            "workload": {
                **{key: integer(usage_counts[key]) for key in usage_counts},
                "active_experiments": integer(active_experiments),
            },
            "security_24h": {key: integer(security_counts[key]) for key in security_counts},
        }

    def get_admin_user_summary(self, user_id):
        user = self.get_user(user_id)
        return self._admin_user_summary(user) if user else None

    def get_admin_user_detail(self, user_id):
        user = self.get_admin_user_summary(user_id)
        if not user:
            return None
        delete_decision = self.can_delete_user(user_id)
        user["sessions"] = self.list_user_sessions(user_id)
        user["verification_codes"] = self.list_user_email_codes(user_id)
        user["can_delete"] = bool(delete_decision["allowed"])
        user["delete_block_reason"] = delete_decision.get("reason") or ""
        return user

    def can_delete_user(self, user_id):
        user = self.get_user(user_id)
        if not user:
            return {"allowed": False, "reason": "user_not_found", "resource_counts": None}
        resource_counts = self.count_user_resources(user_id)
        if user["role"] == "admin":
            return {"allowed": False, "reason": "admin_user", "resource_counts": resource_counts}
        blocking = [
            key
            for key in ["datasets", "experiments", "models", "inference_runs"]
            if int(resource_counts.get(key) or 0) > 0
        ]
        if blocking:
            return {
                "allowed": False,
                "reason": "user_has_resources",
                "blocking_resources": blocking,
                "resource_counts": resource_counts,
            }
        return {"allowed": True, "reason": "", "resource_counts": resource_counts}

    def delete_user_without_resources(self, user_id):
        user = self.get_user(user_id)
        decision = self.can_delete_user(user_id)
        if not user or not decision["allowed"]:
            return None, decision
        with self.connect() as conn:
            agent_chat_message_count = conn.execute(
                select(func.count())
                .select_from(agent_chat_messages)
                .join(
                    agent_chat_conversations,
                    agent_chat_conversations.c.id == agent_chat_messages.c.conversation_id,
                )
                .where(agent_chat_conversations.c.owner_user_id == user_id)
            ).scalar_one()
            agent_chat_conversation_count = (
                conn.execute(
                    delete(agent_chat_conversations).where(
                        agent_chat_conversations.c.owner_user_id == user_id
                    )
                ).rowcount
                or 0
            )
            session_count = (
                conn.execute(delete(sessions).where(sessions.c.user_id == user_id)).rowcount or 0
            )
            email_code_count = (
                conn.execute(
                    delete(email_verification_codes).where(
                        email_verification_codes.c.user_id == user_id
                    )
                ).rowcount
                or 0
            )
            quota_usage_count = (
                conn.execute(delete(quota_usage).where(quota_usage.c.user_id == user_id)).rowcount
                or 0
            )
            user_quota_count = (
                conn.execute(delete(user_quotas).where(user_quotas.c.user_id == user_id)).rowcount
                or 0
            )
            agent_execution_plan_count = (
                conn.execute(
                    delete(agent_execution_plans).where(
                        agent_execution_plans.c.owner_user_id == user_id
                    )
                ).rowcount
                or 0
            )
            agent_campaign_count = (
                conn.execute(
                    delete(agent_campaigns).where(agent_campaigns.c.owner_user_id == user_id)
                ).rowcount
                or 0
            )
            ai_provider_config_count = (
                conn.execute(
                    delete(ai_provider_configs).where(
                        ai_provider_configs.c.owner_user_id == user_id
                    )
                ).rowcount
                or 0
            )
            conn.execute(
                update(audit_logs)
                .where(audit_logs.c.actor_user_id == user_id)
                .values(actor_user_id=None)
            )
            conn.execute(
                update(audit_logs)
                .where(audit_logs.c.target_user_id == user_id)
                .values(target_user_id=None)
            )
            user_count = conn.execute(delete(users).where(users.c.id == user_id)).rowcount or 0
        return {
            "user": user,
            "deleted": bool(user_count),
            "deleted_counts": {
                "sessions": int(session_count),
                "email_verification_codes": int(email_code_count),
                "quota_usage": int(quota_usage_count),
                "user_quotas": int(user_quota_count),
                "agent_campaigns": int(agent_campaign_count),
                "agent_execution_plans": int(agent_execution_plan_count),
                "ai_provider_configs": int(ai_provider_config_count),
                "agent_chat_conversations": int(agent_chat_conversation_count),
                "agent_chat_messages": int(agent_chat_message_count),
                "users": int(user_count),
            },
        }, decision

    def update_user_admin_fields(self, user_id, role=None, status=None):
        values = {"updated_at": utcnow()}
        if role is not None:
            values["role"] = str(role).strip()
        if status is not None:
            values["status"] = str(status).strip()
        if len(values) == 1:
            return self.get_user(user_id)
        with self.connect() as conn:
            result = conn.execute(update(users).where(users.c.id == user_id).values(**values))
        if not result.rowcount:
            return None
        return self.get_user(user_id)

    def get_user_quota(self, user_id):
        row = self._fetch_one(select(user_quotas).where(user_quotas.c.user_id == user_id))
        if row:
            return self._quota_row(row)
        return {
            "user_id": user_id,
            **DEFAULT_USER_QUOTAS,
            "updated_at": None,
        }

    def set_user_quota(self, user_id, quota_updates):
        if not self.get_user(user_id):
            return None
        values = {}
        for key in DEFAULT_USER_QUOTAS:
            if key not in (quota_updates or {}):
                continue
            try:
                value = int(quota_updates[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be a non-negative integer") from exc
            if value < 0:
                raise ValueError(f"{key} must be a non-negative integer")
            values[key] = value
        if not values:
            return self.get_user_quota(user_id)
        now = utcnow()
        values["updated_at"] = now
        with self.connect() as conn:
            existing = conn.execute(
                select(user_quotas.c.user_id).where(user_quotas.c.user_id == user_id)
            ).first()
            if existing:
                conn.execute(
                    update(user_quotas).where(user_quotas.c.user_id == user_id).values(**values)
                )
            else:
                payload = {
                    "user_id": user_id,
                    **DEFAULT_USER_QUOTAS,
                    **values,
                }
                conn.execute(insert(user_quotas).values(**payload))
        return self.get_user_quota(user_id)

    def get_quota_usage(self, user_id, usage_date=None):
        usage_date = usage_date or quota_usage_date()
        row = self._fetch_one(
            select(quota_usage).where(
                and_(quota_usage.c.user_id == user_id, quota_usage.c.usage_date == usage_date)
            )
        )
        if row:
            return self._quota_usage_row(row)
        return {
            "user_id": user_id,
            "usage_date": usage_date,
            "experiments_created": 0,
            "inference_requests": 0,
            "inference_inputs": 0,
            "updated_at": None,
        }

    def increment_quota_usage(
        self,
        user_id,
        experiments_created=0,
        inference_requests=0,
        inference_inputs=0,
        usage_date=None,
    ):
        usage_date = usage_date or quota_usage_date()
        now = utcnow()
        increments = {
            "experiments_created": int(experiments_created or 0),
            "inference_requests": int(inference_requests or 0),
            "inference_inputs": int(inference_inputs or 0),
        }
        with self.connect() as conn:
            existing = conn.execute(
                select(quota_usage.c.user_id).where(
                    and_(quota_usage.c.user_id == user_id, quota_usage.c.usage_date == usage_date)
                )
            ).first()
            if existing:
                conn.execute(
                    update(quota_usage)
                    .where(
                        and_(
                            quota_usage.c.user_id == user_id, quota_usage.c.usage_date == usage_date
                        )
                    )
                    .values(
                        experiments_created=quota_usage.c.experiments_created
                        + increments["experiments_created"],
                        inference_requests=quota_usage.c.inference_requests
                        + increments["inference_requests"],
                        inference_inputs=quota_usage.c.inference_inputs
                        + increments["inference_inputs"],
                        updated_at=now,
                    )
                )
            else:
                conn.execute(
                    insert(quota_usage).values(
                        user_id=user_id,
                        usage_date=usage_date,
                        **increments,
                        updated_at=now,
                    )
                )
        return self.get_quota_usage(user_id, usage_date=usage_date)

    def count_active_experiments_for_user(self, user_id):
        row = self._fetch_one(
            select(func.count().label("count"))
            .select_from(experiments)
            .where(
                and_(
                    experiments.c.owner_user_id == user_id,
                    experiments.c.status.in_(["queued", "running"]),
                )
            )
        )
        return int(row["count"]) if row else 0

    def count_user_resources(self, user_id):
        with self.connect() as conn:
            dataset_count = conn.execute(
                select(func.count().label("count"))
                .select_from(datasets)
                .where(datasets.c.owner_user_id == user_id)
            ).scalar_one()
            experiment_count = conn.execute(
                select(func.count().label("count"))
                .select_from(experiments)
                .where(experiments.c.owner_user_id == user_id)
            ).scalar_one()
            active_experiment_count = conn.execute(
                select(func.count().label("count"))
                .select_from(experiments)
                .where(
                    and_(
                        experiments.c.owner_user_id == user_id,
                        experiments.c.status.in_(["queued", "running"]),
                    )
                )
            ).scalar_one()
            model_count = conn.execute(
                select(func.count().label("count"))
                .select_from(models)
                .where(models.c.owner_user_id == user_id)
            ).scalar_one()
            inference_run_count = conn.execute(
                select(func.count().label("count"))
                .select_from(inference_runs)
                .where(inference_runs.c.owner_user_id == user_id)
            ).scalar_one()
            agent_execution_plan_count = conn.execute(
                select(func.count().label("count"))
                .select_from(agent_execution_plans)
                .where(agent_execution_plans.c.owner_user_id == user_id)
            ).scalar_one()
            agent_campaign_count = conn.execute(
                select(func.count().label("count"))
                .select_from(agent_campaigns)
                .where(agent_campaigns.c.owner_user_id == user_id)
            ).scalar_one()
            agent_chat_conversation_count = conn.execute(
                select(func.count().label("count"))
                .select_from(agent_chat_conversations)
                .where(agent_chat_conversations.c.owner_user_id == user_id)
            ).scalar_one()
        return {
            "datasets": int(dataset_count),
            "experiments": int(experiment_count),
            "active_experiments": int(active_experiment_count),
            "models": int(model_count),
            "inference_runs": int(inference_run_count),
            "agent_campaigns": int(agent_campaign_count),
            "agent_execution_plans": int(agent_execution_plan_count),
            "agent_chat_conversations": int(agent_chat_conversation_count),
        }

    def create_audit_log(
        self,
        event_type,
        actor_user_id=None,
        target_user_id=None,
        metadata=None,
        ip_address="",
        user_agent="",
    ):
        log_id = uuid.uuid4().hex
        with self.connect() as conn:
            conn.execute(
                insert(audit_logs).values(
                    id=log_id,
                    actor_user_id=actor_user_id,
                    target_user_id=target_user_id,
                    event_type=str(event_type or "").strip(),
                    ip_address=str(ip_address or ""),
                    user_agent=str(user_agent or ""),
                    metadata_json=_dump(normalize_stream_fields(metadata)),
                    created_at=utcnow(),
                )
            )
        return self.get_audit_log(log_id)

    def get_audit_log(self, log_id):
        row = self._fetch_one(select(audit_logs).where(audit_logs.c.id == log_id))
        return self._audit_log_row(row) if row else None

    def list_audit_logs(
        self,
        event_type="",
        actor_user_id="",
        target_user_id="",
        user_id="",
        search="",
        created_from="",
        created_to="",
        limit=50,
        offset=0,
    ):
        conditions = []
        if event_type:
            conditions.append(audit_logs.c.event_type == str(event_type).strip())
        if actor_user_id:
            conditions.append(audit_logs.c.actor_user_id == str(actor_user_id).strip())
        if target_user_id:
            conditions.append(audit_logs.c.target_user_id == str(target_user_id).strip())
        if user_id:
            scoped_user_id = str(user_id).strip()
            conditions.append(
                or_(
                    audit_logs.c.actor_user_id == scoped_user_id,
                    audit_logs.c.target_user_id == scoped_user_id,
                )
            )
        search = str(search or "").strip().lower()
        if search:
            pattern = f"%{search}%"
            matching_users = select(users.c.id).where(
                or_(
                    func.lower(users.c.email).like(pattern),
                    func.lower(users.c.display_name).like(pattern),
                )
            )
            conditions.append(
                or_(
                    func.lower(audit_logs.c.event_type).like(pattern),
                    func.lower(audit_logs.c.ip_address).like(pattern),
                    func.lower(audit_logs.c.metadata_json).like(pattern),
                    audit_logs.c.actor_user_id.in_(matching_users),
                    audit_logs.c.target_user_id.in_(matching_users),
                )
            )
        if created_from:
            conditions.append(audit_logs.c.created_at >= str(created_from).strip())
        if created_to:
            conditions.append(audit_logs.c.created_at <= str(created_to).strip())
        try:
            limit = max(1, min(int(limit), 200))
        except (TypeError, ValueError):
            limit = 50
        try:
            offset = max(0, int(offset))
        except (TypeError, ValueError):
            offset = 0

        where_clause = and_(*conditions) if conditions else None
        count_stmt = select(func.count().label("count")).select_from(audit_logs)
        list_stmt = (
            select(audit_logs).order_by(audit_logs.c.created_at.desc()).limit(limit).offset(offset)
        )
        if where_clause is not None:
            count_stmt = count_stmt.where(where_clause)
            list_stmt = list_stmt.where(where_clause)
        with self.connect() as conn:
            total_row = conn.execute(count_stmt).mappings().first()
            rows = list(conn.execute(list_stmt).mappings().all())
        items = [self._audit_log_row(row) for row in rows]
        user_ids = {
            user_id
            for item in items
            for user_id in [item.get("actor_user_id"), item.get("target_user_id")]
            if user_id
        }
        user_rows = (
            self._fetch_all(select(users).where(users.c.id.in_(user_ids))) if user_ids else []
        )
        users_by_id = {row["id"]: self._user_row(row) for row in user_rows}
        for item in items:
            actor = users_by_id.get(item.get("actor_user_id"))
            target = users_by_id.get(item.get("target_user_id"))
            item["actor"] = self._audit_user_ref(actor) if actor else None
            item["target"] = self._audit_user_ref(target) if target else None
        return {
            "items": items,
            "total": int(total_row["count"]) if total_row else 0,
            "limit": limit,
            "offset": offset,
        }

    def list_audit_event_types(self):
        rows = self._fetch_all(
            select(audit_logs.c.event_type, func.count().label("count"))
            .group_by(audit_logs.c.event_type)
            .order_by(func.count().desc(), audit_logs.c.event_type.asc())
        )
        return {
            "items": [
                {"event_type": row["event_type"], "count": int(row["count"] or 0)} for row in rows
            ]
        }

    def upsert_dataset(
        self, dataset_id, name, task_type, source, fmt, root_path, metadata=None, owner_user_id=None
    ):
        task_type = normalize_task_type(task_type)
        now = utcnow()
        payload = {
            "id": dataset_id,
            "name": name,
            "task_type": task_type,
            "source": source,
            "format": fmt,
            "root_path": str(root_path),
            "metadata_json": _dump(metadata),
            "owner_user_id": owner_user_id,
            "created_at": now,
        }
        with self.connect() as conn:
            existing = conn.execute(
                select(datasets.c.id).where(datasets.c.id == dataset_id)
            ).first()
            if existing:
                values = dict(payload)
                values.pop("id")
                values.pop("created_at")
                if owner_user_id is None:
                    values.pop("owner_user_id")
                conn.execute(update(datasets).where(datasets.c.id == dataset_id).values(**values))
            else:
                conn.execute(insert(datasets).values(**payload))
        return self.get_dataset(dataset_id)

    def create_dataset(
        self, name, task_type, source, fmt, root_path, metadata=None, owner_user_id=None
    ):
        task_type = normalize_task_type(task_type)
        dataset_id = uuid.uuid4().hex
        return self.upsert_dataset(
            dataset_id, name, task_type, source, fmt, root_path, metadata, owner_user_id
        )

    def list_datasets(self, user=None):
        rows = self._fetch_all(select(datasets).order_by(datasets.c.created_at.desc()))
        return [self._dataset_row(row) for row in rows if self._visible_dataset(row, user)]

    def get_dataset(self, dataset_id, user=None):
        row = None
        for candidate_id in builtin_id_spellings(dataset_id):
            row = self._fetch_one(select(datasets).where(datasets.c.id == candidate_id))
            if row is not None:
                break
        if not row or not self._visible_dataset(row, user):
            return None
        return self._dataset_row(row)

    def count_experiments_for_dataset(self, dataset_id, user=None):
        conditions = [experiments.c.dataset_id == dataset_id]
        if user is not None and not _is_admin(user):
            conditions.append(experiments.c.owner_user_id == user["id"])
        row = self._fetch_one(
            select(func.count().label("count")).select_from(experiments).where(and_(*conditions))
        )
        return int(row["count"]) if row else 0

    def delete_dataset(self, dataset_id, user=None):
        dataset = self.get_dataset(dataset_id, user=user)
        if not dataset:
            return None
        with self.connect() as conn:
            conn.execute(delete(datasets).where(datasets.c.id == dataset_id))
        return dataset

    def create_experiment(
        self,
        name,
        dataset_id,
        task_type,
        config,
        log_path,
        artifact_dir,
        experiment_id=None,
        owner_user_id=None,
        name_kind=None,
        name_base=None,
        name_base_kind=None,
        name_qualifier=None,
        rerun_count=0,
        rerun_source_experiment_id=None,
        resume=None,
    ):
        task_type = normalize_task_type(task_type)
        experiment_id = experiment_id or uuid.uuid4().hex
        dataset = self.get_dataset(dataset_id)
        if owner_user_id is None and dataset:
            owner_user_id = dataset.get("owner_user_id")
        now = utcnow()
        with self.connect() as conn:
            conn.execute(
                insert(experiments).values(
                    id=experiment_id,
                    dataset_id=dataset_id,
                    owner_user_id=owner_user_id,
                    name=name,
                    name_kind=name_kind,
                    name_base=name_base,
                    name_base_kind=name_base_kind,
                    name_qualifier=(
                        name_qualifier if name_qualifier in GENERATED_NAME_QUALIFIERS else None
                    ),
                    rerun_count=max(0, int(rerun_count or 0)),
                    rerun_source_experiment_id=rerun_source_experiment_id,
                    task_type=task_type,
                    config_json=_dump(normalize_stream_fields(config)),
                    status="queued",
                    log_path=str(log_path),
                    artifact_dir=str(artifact_dir),
                    metrics_json=_dump({"events": [], "summary": {}}),
                    model_id=None,
                    error="",
                    failure_json="{}",
                    resume_json=_dump(resume if isinstance(resume, dict) else {}),
                    created_at=now,
                    updated_at=now,
                    started_at=None,
                    finished_at=None,
                )
            )
        return self.get_experiment(experiment_id)

    def list_experiments(self, user=None):
        rows = self._fetch_all(select(experiments).order_by(experiments.c.created_at.desc()))
        return [self._experiment_row(row) for row in rows if self._visible_owned_row(row, user)]

    def get_experiment(self, experiment_id, user=None):
        row = self._fetch_one(select(experiments).where(experiments.c.id == experiment_id))
        if not row or not self._visible_owned_row(row, user):
            return None
        return self._experiment_row(row)

    def update_experiment_name(self, experiment_id, name, user=None):
        if not self.get_experiment(experiment_id, user=user):
            return None
        with self.connect() as conn:
            conn.execute(
                update(experiments)
                .where(experiments.c.id == experiment_id)
                .values(
                    name=name,
                    name_kind=None,
                    name_base=None,
                    name_base_kind=None,
                    name_qualifier=None,
                    rerun_count=0,
                    rerun_source_experiment_id=None,
                )
            )
            conn.execute(
                update(models)
                .where(
                    and_(
                        models.c.experiment_id == experiment_id,
                        models.c.name_is_custom == 0,
                    )
                )
                .values(name=name)
            )
        return self.get_experiment(experiment_id, user=user)

    def refresh_metric_summaries(self):
        rows = self._fetch_all(
            select(experiments.c.id, experiments.c.task_type, experiments.c.metrics_json)
        )
        updates = []
        skipped = []
        for row in rows:
            try:
                metrics = _load(row["metrics_json"], {"events": [], "summary": {}})
                if not isinstance(metrics, dict):
                    metrics = {}
                events = metrics.get("events") or []
                if not isinstance(events, list):
                    events = []
                summary = metrics.get("summary") or {}
                if not isinstance(summary, dict):
                    summary = {}
                rebuilt_summary = rebuild_metric_summary(
                    events, summary, task_type=row["task_type"]
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                skipped.append(row["id"])
                continue
            if rebuilt_summary == summary:
                continue
            refreshed_metrics = dict(metrics)
            preserve_legacy_summary(refreshed_metrics)
            refreshed_metrics["events"] = events
            refreshed_metrics["summary"] = rebuilt_summary
            updates.append((row["id"], refreshed_metrics))

        if updates:
            now = utcnow()
            with self.connect() as conn:
                for experiment_id, metrics in updates:
                    conn.execute(
                        update(experiments)
                        .where(experiments.c.id == experiment_id)
                        .values(metrics_json=_dump(metrics), updated_at=now)
                    )

        return {
            "scanned": len(rows),
            "refreshed": len(updates),
            "unchanged": len(rows) - len(updates) - len(skipped),
            "skipped": skipped,
        }

    def update_experiment(self, experiment_id, **updates):
        allowed = {
            "status",
            "metrics_json",
            "model_id",
            "error",
            "failure",
            "started_at",
            "finished_at",
            "updated_at",
        }
        values = {}
        for key, value in updates.items():
            if key not in allowed:
                continue
            if key == "failure":
                values["failure_json"] = _dump(value)
            else:
                values[key] = _dump(value) if key == "metrics_json" else value
        if not values:
            return self.get_experiment(experiment_id)
        values.setdefault("updated_at", utcnow())
        with self.connect() as conn:
            conn.execute(
                update(experiments).where(experiments.c.id == experiment_id).values(**values)
            )
        return self.get_experiment(experiment_id)

    def raw_experiment_metrics(self, experiment_id):
        """Internal migration/event input; public reads hide unconverted AP."""
        row = self._fetch_one(select(experiments.c.metrics_json).where(experiments.c.id == experiment_id))
        if row is None:
            raise ValueError(f"Experiment not found: {experiment_id}")
        return _load(row["metrics_json"], {"events": [], "summary": {}})

    def cancel_active_experiments(self, reason):
        now = utcnow()
        with self.connect() as conn:
            rows = list(
                conn.execute(
                    select(experiments).where(experiments.c.status.in_(["queued", "running"]))
                )
                .mappings()
                .all()
            )
            conn.execute(
                update(experiments)
                .where(experiments.c.status.in_(["queued", "running"]))
                .values(status="canceled", error=reason, finished_at=now, updated_at=now)
            )
        return [self._experiment_row(row) for row in rows]

    def append_metric_event(self, experiment_id, event):
        experiment = self.get_experiment(experiment_id)
        metrics = self.raw_experiment_metrics(experiment_id)
        events = metrics.setdefault("events", [])
        if (
            isinstance(event, dict)
            and event.get("kind") == "stage"
            and event.get("stage") in RUN_PROGRESS_STAGES
        ):
            events[:] = [
                existing
                for existing in events
                if not (
                    isinstance(existing, dict)
                    and existing.get("kind") == "stage"
                    and existing.get("stage") == event.get("stage")
                )
            ]
        events.append(event)
        preserve_legacy_summary(metrics)
        metrics["summary"] = rebuild_metric_summary(
            events,
            metrics.get("summary"),
            task_type=experiment.get("task_type"),
        )
        self.update_experiment(experiment_id, metrics_json=metrics)
        return metrics

    def create_model(
        self,
        experiment_id,
        name,
        task_type,
        bundle_path,
        supports_inference,
        metadata=None,
        owner_user_id=None,
        name_is_custom=False,
    ):
        task_type = normalize_task_type(task_type)
        experiment = self.get_experiment(experiment_id)
        if owner_user_id is None and experiment:
            owner_user_id = experiment.get("owner_user_id")
        model_id = uuid.uuid4().hex
        with self.connect() as conn:
            conn.execute(
                insert(models).values(
                    id=model_id,
                    experiment_id=experiment_id,
                    owner_user_id=owner_user_id,
                    name=name,
                    name_is_custom=int(bool(name_is_custom)),
                    task_type=task_type,
                    bundle_path=str(bundle_path),
                    supports_inference=int(bool(supports_inference)),
                    metadata_json=_dump(normalize_stream_fields(metadata)),
                    created_at=utcnow(),
                )
            )
        return self.get_model(model_id)

    def list_models(self, user=None):
        rows = self._fetch_all(select(models).order_by(models.c.created_at.desc()))
        return [self._model_row(row) for row in rows if self._visible_owned_row(row, user)]

    def list_models_for_experiment(self, experiment_id, user=None):
        rows = self._fetch_all(
            select(models)
            .where(models.c.experiment_id == experiment_id)
            .order_by(models.c.created_at.desc())
        )
        return [self._model_row(row) for row in rows if self._visible_owned_row(row, user)]

    def get_model(self, model_id, user=None):
        row = self._fetch_one(select(models).where(models.c.id == model_id))
        if not row or not self._visible_owned_row(row, user):
            return None
        return self._model_row(row)

    def update_model_name(self, model_id, name, user=None):
        if not self.get_model(model_id, user=user):
            return None
        with self.connect() as conn:
            conn.execute(
                update(models).where(models.c.id == model_id).values(name=name, name_is_custom=1)
            )
        return self.get_model(model_id, user=user)

    def delete_model(self, model_id, user=None):
        model = self.get_model(model_id, user=user)
        if not model:
            return None
        with self.connect() as conn:
            conn.execute(
                update(experiments)
                .where(experiments.c.model_id == model_id)
                .values(model_id=None, updated_at=utcnow())
            )
            conn.execute(delete(inference_runs).where(inference_runs.c.model_id == model_id))
            conn.execute(delete(models).where(models.c.id == model_id))
        return model

    def delete_experiment(self, experiment_id, user=None):
        experiment = self.get_experiment(experiment_id, user=user)
        if not experiment:
            return None
        with self.connect() as conn:
            conversation_ids = select(agent_chat_conversations.c.id).where(
                agent_chat_conversations.c.experiment_id == experiment_id
            )
            conn.execute(
                delete(agent_chat_messages).where(
                    agent_chat_messages.c.conversation_id.in_(conversation_ids)
                )
            )
            conn.execute(
                delete(agent_chat_conversations).where(
                    agent_chat_conversations.c.experiment_id == experiment_id
                )
            )
            rows = list(
                conn.execute(select(models.c.id).where(models.c.experiment_id == experiment_id))
                .mappings()
                .all()
            )
            for row in rows:
                conn.execute(delete(inference_runs).where(inference_runs.c.model_id == row["id"]))
            conn.execute(delete(models).where(models.c.experiment_id == experiment_id))
            conn.execute(delete(experiments).where(experiments.c.id == experiment_id))
        return experiment

    def create_inference_run(self, model_id, request_payload, response_payload, owner_user_id=None):
        model = self.get_model(model_id)
        if not model:
            return None
        if owner_user_id is None:
            owner_user_id = model.get("owner_user_id")
        sequences = request_payload.get("sequences") if isinstance(request_payload, dict) else None
        pairs = request_payload.get("pairs") if isinstance(request_payload, dict) else None
        input_kind = "pairs" if pairs is not None else "sequences"
        input_count = len(pairs if pairs is not None else sequences or [])
        predictions = (
            response_payload.get("predictions") if isinstance(response_payload, dict) else []
        )
        run_id = uuid.uuid4().hex
        with self.connect() as conn:
            conn.execute(
                insert(inference_runs).values(
                    id=run_id,
                    model_id=model_id,
                    owner_user_id=owner_user_id,
                    input_kind=input_kind,
                    input_count=int(input_count),
                    prediction_count=int(len(predictions or [])),
                    request_json=_dump(request_payload),
                    response_json=_dump(response_payload),
                    created_at=utcnow(),
                )
            )
        return self.get_inference_run(run_id)

    def list_inference_runs(self, model_id, limit=20, user=None):
        rows = self._fetch_all(
            select(inference_runs)
            .where(inference_runs.c.model_id == model_id)
            .order_by(inference_runs.c.created_at.desc())
            .limit(max(1, min(int(limit), 200)))
        )
        return [self._inference_run_row(row) for row in rows if self._visible_owned_row(row, user)]

    def get_inference_run(self, run_id, user=None):
        row = self._fetch_one(select(inference_runs).where(inference_runs.c.id == run_id))
        if not row or not self._visible_owned_row(row, user):
            return None
        return self._inference_run_row(row)

    def delete_inference_run(self, run_id, user=None):
        run = self.get_inference_run(run_id, user=user)
        if not run:
            return None
        with self.connect() as conn:
            conn.execute(delete(inference_runs).where(inference_runs.c.id == run_id))
        return run

    def create_ai_provider_config(
        self,
        owner_user_id,
        provider,
        display_name,
        base_url,
        model,
        api_key_ciphertext,
        api_key_last4,
        enabled=True,
        capabilities=None,
        provider_id=None,
    ):
        now = utcnow()
        provider_id = provider_id or uuid.uuid4().hex
        with self.connect() as conn:
            conn.execute(
                insert(ai_provider_configs).values(
                    id=provider_id,
                    owner_user_id=owner_user_id,
                    provider=str(provider or "").strip(),
                    display_name=str(display_name or "").strip(),
                    base_url=str(base_url or "").strip(),
                    model=str(model or "").strip(),
                    api_key_ciphertext=str(api_key_ciphertext or ""),
                    api_key_last4=str(api_key_last4 or ""),
                    capabilities_json=_dump(capabilities),
                    enabled=int(bool(enabled)),
                    created_at=now,
                    updated_at=now,
                )
            )
        return self.get_ai_provider_config(provider_id, include_secret=True)

    def list_ai_provider_configs(self, user=None):
        rows = self._fetch_all(
            select(ai_provider_configs).order_by(ai_provider_configs.c.created_at.desc())
        )
        return [
            self._ai_provider_config_row(row) for row in rows if self._visible_owned_row(row, user)
        ]

    def get_ai_provider_config(self, provider_id, user=None, include_secret=False):
        row = self._fetch_one(
            select(ai_provider_configs).where(ai_provider_configs.c.id == provider_id)
        )
        if not row or not self._visible_owned_row(row, user):
            return None
        return self._ai_provider_config_row(row, include_secret=include_secret)

    def update_ai_provider_config(self, provider_id, updates, user=None):
        if not self.get_ai_provider_config(provider_id, user=user):
            return None
        allowed = {
            "provider",
            "display_name",
            "base_url",
            "model",
            "api_key_ciphertext",
            "api_key_last4",
            "capabilities",
            "enabled",
        }
        values = {}
        for key, value in (updates or {}).items():
            if key not in allowed:
                continue
            if key == "enabled":
                values[key] = int(bool(value))
            elif key == "capabilities":
                values["capabilities_json"] = _dump(value if isinstance(value, dict) else {})
            else:
                values[key] = value
        if not values:
            return self.get_ai_provider_config(provider_id, user=user, include_secret=True)
        values["updated_at"] = utcnow()
        with self.connect() as conn:
            conn.execute(
                update(ai_provider_configs)
                .where(ai_provider_configs.c.id == provider_id)
                .values(**values)
            )
        return self.get_ai_provider_config(provider_id, user=user, include_secret=True)

    def delete_ai_provider_config(self, provider_id, user=None):
        config = self.get_ai_provider_config(provider_id, user=user)
        if not config:
            return None
        with self.connect() as conn:
            conn.execute(delete(ai_provider_configs).where(ai_provider_configs.c.id == provider_id))
        return config

    @staticmethod
    def _agent_chat_owner_id(user):
        return user.get("id") if user else None

    @staticmethod
    def _agent_chat_owner_condition(user):
        owner_id = MetadataStore._agent_chat_owner_id(user)
        if owner_id is None:
            return agent_chat_conversations.c.owner_user_id.is_(None)
        return agent_chat_conversations.c.owner_user_id == owner_id

    @staticmethod
    def _agent_chat_scope_conditions(scope="agent", experiment_id=None):
        scope = str(scope or "agent").strip() or "agent"
        conditions = [agent_chat_conversations.c.scope == scope]
        if scope == "experiment_analysis":
            experiment_id = str(experiment_id or "").strip()
            if not experiment_id:
                raise ValueError("experiment_id is required for experiment analysis conversations")
            conditions.append(agent_chat_conversations.c.experiment_id == experiment_id)
        else:
            conditions.append(agent_chat_conversations.c.experiment_id.is_(None))
        return conditions

    def list_agent_chat_conversations(
        self,
        user=None,
        query="",
        limit=50,
        offset=0,
        scope="agent",
        experiment_id=None,
    ):
        try:
            limit = max(1, min(int(limit), 100))
        except (TypeError, ValueError):
            limit = 50
        try:
            offset = max(0, int(offset))
        except (TypeError, ValueError):
            offset = 0
        conditions = [self._agent_chat_owner_condition(user)]
        conditions.extend(self._agent_chat_scope_conditions(scope, experiment_id))
        query = str(query or "").strip()
        if query:
            pattern = f"%{query}%"
            matching_messages = select(agent_chat_messages.c.conversation_id).where(
                agent_chat_messages.c.content.ilike(pattern)
            )
            conditions.append(
                or_(
                    agent_chat_conversations.c.title.ilike(pattern),
                    agent_chat_conversations.c.id.in_(matching_messages),
                )
            )
        where_clause = and_(*conditions)
        with self.connect() as conn:
            rows = (
                conn.execute(
                    select(agent_chat_conversations)
                    .where(where_clause)
                    .order_by(
                        agent_chat_conversations.c.pinned_at.is_(None),
                        agent_chat_conversations.c.pinned_at.desc(),
                        agent_chat_conversations.c.updated_at.desc(),
                    )
                    .limit(limit)
                    .offset(offset)
                )
                .mappings()
                .all()
            )
            total = conn.execute(
                select(func.count()).select_from(agent_chat_conversations).where(where_clause)
            ).scalar_one()
        return {
            "items": [self._agent_chat_conversation_row(row) for row in rows],
            "total": int(total),
            "limit": limit,
            "offset": offset,
        }

    def get_agent_chat_conversation(
        self,
        conversation_id,
        user=None,
        scope="agent",
        experiment_id=None,
    ):
        conditions = [
            agent_chat_conversations.c.id == conversation_id,
            self._agent_chat_owner_condition(user),
        ]
        conditions.extend(self._agent_chat_scope_conditions(scope, experiment_id))
        row = self._fetch_one(select(agent_chat_conversations).where(and_(*conditions)))
        return self._agent_chat_conversation_row(row) if row else None

    def list_agent_chat_messages(
        self,
        conversation_id,
        user=None,
        limit=100,
        before_ordinal=None,
        scope="agent",
        experiment_id=None,
    ):
        conversation = self.get_agent_chat_conversation(
            conversation_id,
            user=user,
            scope=scope,
            experiment_id=experiment_id,
        )
        if not conversation:
            return None
        try:
            limit = max(1, min(int(limit), 200))
        except (TypeError, ValueError):
            limit = 100
        conditions = [agent_chat_messages.c.conversation_id == conversation_id]
        if before_ordinal not in [None, ""]:
            try:
                conditions.append(agent_chat_messages.c.ordinal < int(before_ordinal))
            except (TypeError, ValueError):
                pass
        rows = self._fetch_all(
            select(agent_chat_messages)
            .where(and_(*conditions))
            .order_by(agent_chat_messages.c.ordinal.desc())
            .limit(limit)
        )
        rows.reverse()
        return {
            "conversation": conversation,
            "messages": [self._agent_chat_message_row(row) for row in rows],
            "has_more": bool(rows and rows[0]["ordinal"] > 1),
            "next_before_ordinal": rows[0]["ordinal"] if rows else None,
        }

    def begin_agent_chat_turn(
        self,
        owner_user_id,
        provider_config_id,
        provider,
        model,
        content,
        conversation_id=None,
        user=None,
        scope="agent",
        experiment_id=None,
        assistant_kind="chat",
    ):
        content = str(content or "").strip()
        if not content:
            raise ValueError("content is required")
        expected_owner_id = self._agent_chat_owner_id(user)
        if owner_user_id != expected_owner_id:
            raise ValueError("chat owner mismatch")
        scope = str(scope or "agent").strip() or "agent"
        scope_conditions = self._agent_chat_scope_conditions(scope, experiment_id)
        experiment_id = str(experiment_id or "").strip() or None
        assistant_kind = str(assistant_kind or "chat").strip() or "chat"
        now = utcnow()
        conversation_id = str(conversation_id or "").strip() or uuid.uuid4().hex
        user_message_id = uuid.uuid4().hex
        assistant_message_id = uuid.uuid4().hex
        with self.connect() as conn:
            conversation_conditions = [
                agent_chat_conversations.c.id == conversation_id,
                self._agent_chat_owner_condition(user),
                *scope_conditions,
            ]
            conversation_row = (
                conn.execute(select(agent_chat_conversations).where(and_(*conversation_conditions)))
                .mappings()
                .first()
            )
            if not conversation_row:
                existing = conn.execute(
                    select(agent_chat_conversations.c.id).where(
                        agent_chat_conversations.c.id == conversation_id
                    )
                ).first()
                if existing:
                    raise ValueError("conversation not found")
                title = " ".join(content.split())[:60] or "New conversation"
                conn.execute(
                    insert(agent_chat_conversations).values(
                        id=conversation_id,
                        owner_user_id=owner_user_id,
                        scope=scope,
                        experiment_id=experiment_id,
                        title=title,
                        pinned_at=None,
                        last_provider_config_id=provider_config_id,
                        last_provider=str(provider or ""),
                        last_model=str(model or ""),
                        last_message_preview=content[:200],
                        created_at=now,
                        updated_at=now,
                    )
                )
                next_ordinal = 1
            else:
                maximum = conn.execute(
                    select(func.max(agent_chat_messages.c.ordinal)).where(
                        agent_chat_messages.c.conversation_id == conversation_id
                    )
                ).scalar_one()
                next_ordinal = int(maximum or 0) + 1
                conn.execute(
                    update(agent_chat_conversations)
                    .where(agent_chat_conversations.c.id == conversation_id)
                    .values(
                        last_provider_config_id=provider_config_id,
                        last_provider=str(provider or ""),
                        last_model=str(model or ""),
                        last_message_preview=content[:200],
                        updated_at=now,
                    )
                )
            conn.execute(
                insert(agent_chat_messages).values(
                    id=user_message_id,
                    conversation_id=conversation_id,
                    ordinal=next_ordinal,
                    role="user",
                    content=content,
                    status="complete",
                    kind="chat",
                    provider_config_id=None,
                    provider="",
                    model="",
                    finish_reason="",
                    error="",
                    usage_json="{}",
                    web_search_json="{}",
                    payload_json="{}",
                    context_manifest_json="{}",
                    created_at=now,
                    updated_at=now,
                )
            )
            conn.execute(
                insert(agent_chat_messages).values(
                    id=assistant_message_id,
                    conversation_id=conversation_id,
                    ordinal=next_ordinal + 1,
                    role="assistant",
                    content="",
                    status="streaming",
                    kind=assistant_kind,
                    provider_config_id=provider_config_id,
                    provider=str(provider or ""),
                    model=str(model or ""),
                    finish_reason="",
                    error="",
                    usage_json="{}",
                    web_search_json="{}",
                    payload_json="{}",
                    context_manifest_json="{}",
                    created_at=now,
                    updated_at=now,
                )
            )
        return {
            "conversation": self.get_agent_chat_conversation(
                conversation_id,
                user=user,
                scope=scope,
                experiment_id=experiment_id,
            ),
            "user_message": self.get_agent_chat_message(
                user_message_id,
                user=user,
                scope=scope,
                experiment_id=experiment_id,
            ),
            "assistant_message": self.get_agent_chat_message(
                assistant_message_id,
                user=user,
                scope=scope,
                experiment_id=experiment_id,
            ),
        }

    def get_agent_chat_message(
        self,
        message_id,
        user=None,
        scope="agent",
        experiment_id=None,
    ):
        conditions = [
            agent_chat_messages.c.id == message_id,
            self._agent_chat_owner_condition(user),
        ]
        conditions.extend(self._agent_chat_scope_conditions(scope, experiment_id))
        row = self._fetch_one(
            select(agent_chat_messages)
            .join(
                agent_chat_conversations,
                agent_chat_conversations.c.id == agent_chat_messages.c.conversation_id,
            )
            .where(and_(*conditions))
        )
        return self._agent_chat_message_row(row) if row else None

    def update_agent_chat_message(
        self,
        message_id,
        user=None,
        content=None,
        status=None,
        finish_reason=None,
        error=None,
        usage=None,
        web_search=None,
        provider=None,
        model=None,
        kind=None,
        payload=None,
        context_manifest=None,
        scope="agent",
        experiment_id=None,
    ):
        message = self.get_agent_chat_message(
            message_id,
            user=user,
            scope=scope,
            experiment_id=experiment_id,
        )
        if not message:
            return None
        values = {"updated_at": utcnow()}
        if content is not None:
            values["content"] = str(content)
        if status is not None:
            values["status"] = str(status)
        if finish_reason is not None:
            values["finish_reason"] = str(finish_reason)
        if error is not None:
            values["error"] = str(error)
        if usage is not None:
            values["usage_json"] = _dump(usage)
        if web_search is not None:
            values["web_search_json"] = _dump(web_search)
        if provider is not None:
            values["provider"] = str(provider)
        if model is not None:
            values["model"] = str(model)
        if kind is not None:
            values["kind"] = str(kind)
        if payload is not None:
            values["payload_json"] = _dump(payload)
        if context_manifest is not None:
            values["context_manifest_json"] = _dump(context_manifest)
        with self.connect() as conn:
            conn.execute(
                update(agent_chat_messages)
                .where(agent_chat_messages.c.id == message_id)
                .values(**values)
            )
            conversation_values = {}
            if content is not None:
                conversation_values["last_message_preview"] = str(content)[-200:]
            if provider is not None:
                conversation_values["last_provider"] = str(provider)
            if model is not None:
                conversation_values["last_model"] = str(model)
            if conversation_values:
                conversation_values["updated_at"] = values["updated_at"]
                conn.execute(
                    update(agent_chat_conversations)
                    .where(agent_chat_conversations.c.id == message["conversation_id"])
                    .values(**conversation_values)
                )
        return self.get_agent_chat_message(
            message_id,
            user=user,
            scope=scope,
            experiment_id=experiment_id,
        )

    def get_agent_chat_context(
        self,
        conversation_id,
        before_ordinal,
        user=None,
        message_limit=40,
        char_limit=60000,
        scope="agent",
        experiment_id=None,
    ):
        if not self.get_agent_chat_conversation(
            conversation_id,
            user=user,
            scope=scope,
            experiment_id=experiment_id,
        ):
            return None
        rows = self._fetch_all(
            select(agent_chat_messages)
            .where(
                and_(
                    agent_chat_messages.c.conversation_id == conversation_id,
                    agent_chat_messages.c.ordinal < int(before_ordinal),
                    agent_chat_messages.c.role.in_(["user", "assistant"]),
                    agent_chat_messages.c.content != "",
                )
            )
            .order_by(agent_chat_messages.c.ordinal.desc())
            .limit(max(1, min(int(message_limit), 100)))
        )
        remaining = max(1, int(char_limit))
        context = []
        for row in rows:
            content = str(row["content"] or "")
            if not content:
                continue
            if row["role"] == "assistant":
                web_search = _load(row.get("web_search_json"), {})
                source_lines = []
                for source in (web_search.get("sources") or [])[:5]:
                    if not isinstance(source, dict):
                        continue
                    index = source.get("index")
                    title = " ".join(str(source.get("title") or "").split())[:200]
                    url = str(source.get("url") or "").strip()[:1000]
                    snippet = " ".join(str(source.get("snippet") or "").split())[:400]
                    if index and url:
                        source_lines.append(f"[{index}] {title} — {url}\n{snippet}".strip())
                if source_lines:
                    content += "\n\nSources used for this answer:\n" + "\n".join(source_lines)
            if len(content) > remaining:
                content = content[-remaining:]
            context.append({"role": row["role"], "content": content})
            remaining -= len(content)
            if remaining <= 0:
                break
        context.reverse()
        return context

    def update_agent_chat_conversation(
        self,
        conversation_id,
        updates,
        user=None,
        scope="agent",
        experiment_id=None,
    ):
        conversation = self.get_agent_chat_conversation(
            conversation_id,
            user=user,
            scope=scope,
            experiment_id=experiment_id,
        )
        if not conversation:
            return None
        values = {"updated_at": utcnow()}
        if "title" in updates:
            title = " ".join(str(updates.get("title") or "").split())[:120]
            if not title:
                raise ValueError("title is required")
            values["title"] = title
        if "pinned" in updates:
            values["pinned_at"] = utcnow() if bool(updates.get("pinned")) else None
        with self.connect() as conn:
            conn.execute(
                update(agent_chat_conversations)
                .where(agent_chat_conversations.c.id == conversation_id)
                .values(**values)
            )
        return self.get_agent_chat_conversation(
            conversation_id,
            user=user,
            scope=scope,
            experiment_id=experiment_id,
        )

    def delete_agent_chat_conversation(
        self,
        conversation_id,
        user=None,
        scope="agent",
        experiment_id=None,
    ):
        conversation = self.get_agent_chat_conversation(
            conversation_id,
            user=user,
            scope=scope,
            experiment_id=experiment_id,
        )
        if not conversation:
            return None
        with self.connect() as conn:
            conn.execute(
                delete(agent_chat_conversations).where(
                    agent_chat_conversations.c.id == conversation_id
                )
            )
        return conversation

    def create_agent_campaign(
        self,
        owner_user_id,
        task_type,
        dataset_id,
        objective,
        max_rounds=3,
        status="active",
        campaign_id=None,
    ):
        task_type = normalize_task_type(task_type)
        campaign_id = campaign_id or uuid.uuid4().hex
        now = utcnow()
        try:
            max_rounds = max(1, min(int(max_rounds), 20))
        except (TypeError, ValueError):
            max_rounds = 3
        with self.connect() as conn:
            conn.execute(
                insert(agent_campaigns).values(
                    id=campaign_id,
                    owner_user_id=owner_user_id,
                    task_type=str(task_type or ""),
                    dataset_id=str(dataset_id or ""),
                    objective=str(objective or ""),
                    status=str(status or "active"),
                    max_rounds=max_rounds,
                    current_round_index=0,
                    created_at=now,
                    updated_at=now,
                )
            )
        return self.get_agent_campaign(campaign_id)

    def list_agent_campaigns(
        self, user=None, dataset_id=None, task_type=None, status=None, limit=20, offset=0
    ):
        task_type = normalize_task_type(task_type)
        try:
            limit = max(1, min(int(limit), 100))
        except (TypeError, ValueError):
            limit = 20
        try:
            offset = max(0, int(offset))
        except (TypeError, ValueError):
            offset = 0

        where_clauses = []
        if dataset_id:
            where_clauses.append(agent_campaigns.c.dataset_id == str(dataset_id).strip())
        if task_type:
            where_clauses.append(agent_campaigns.c.task_type.in_(task_spellings(task_type)))
        if status:
            where_clauses.append(agent_campaigns.c.status == str(status).strip())
        if user is not None and not _is_admin(user):
            where_clauses.append(agent_campaigns.c.owner_user_id == user.get("id"))

        base_stmt = select(agent_campaigns)
        if where_clauses:
            base_stmt = base_stmt.where(and_(*where_clauses))
        count_stmt = base_stmt.with_only_columns(func.count(agent_campaigns.c.id))
        rows_stmt = (
            base_stmt.order_by(agent_campaigns.c.created_at.desc()).limit(limit).offset(offset)
        )
        with self.connect() as conn:
            total = conn.execute(count_stmt).scalar() or 0
            rows = conn.execute(rows_stmt).mappings().all()
        return [self._agent_campaign_summary_row(row) for row in rows], total

    def get_agent_campaign(self, campaign_id, user=None, include_plans=False):
        row = self._fetch_one(select(agent_campaigns).where(agent_campaigns.c.id == campaign_id))
        if not row or not self._visible_owned_row(row, user):
            return None
        campaign = self._agent_campaign_row(row)
        if include_plans:
            plans, _total = self.list_agent_execution_plans(
                user=user,
                campaign_id=campaign["id"],
                limit=100,
                offset=0,
            )
            campaign["plans"] = sorted(
                plans, key=lambda item: (item.get("round_index") or 0, item.get("created_at") or "")
            )
        return campaign

    def update_agent_campaign(self, campaign_id, **updates):
        allowed_text = {"status", "objective"}
        allowed_int = {"max_rounds", "current_round_index"}
        values = {}
        for key, value in updates.items():
            if key in allowed_text:
                values[key] = str(value or "")
            elif key in allowed_int:
                try:
                    values[key] = int(value)
                except (TypeError, ValueError):
                    continue
        if not values:
            return self.get_agent_campaign(campaign_id)
        values["updated_at"] = utcnow()
        with self.connect() as conn:
            conn.execute(
                update(agent_campaigns).where(agent_campaigns.c.id == campaign_id).values(**values)
            )
        return self.get_agent_campaign(campaign_id)

    def create_agent_execution_plan(
        self,
        owner_user_id,
        provider_config_id,
        provider,
        model,
        task_type,
        dataset_id,
        objective,
        actions,
        usage=None,
        context_manifest=None,
        warnings=None,
        status="draft",
        campaign_id=None,
        round_index=0,
        trace=None,
        plan_id=None,
    ):
        task_type = normalize_task_type(task_type)
        plan_id = plan_id or uuid.uuid4().hex
        now = utcnow()
        with self.connect() as conn:
            conn.execute(
                insert(agent_execution_plans).values(
                    id=plan_id,
                    owner_user_id=owner_user_id,
                    campaign_id=str(campaign_id).strip() if campaign_id else None,
                    round_index=int(round_index or 0),
                    provider_config_id=str(provider_config_id or ""),
                    provider=str(provider or ""),
                    model=str(model or ""),
                    task_type=str(task_type or ""),
                    dataset_id=str(dataset_id or ""),
                    objective=str(objective or ""),
                    status=str(status or "draft"),
                    actions_json=_dump(actions if isinstance(actions, list) else []),
                    execution_json=_dump({}),
                    review_json=_dump({}),
                    usage_json=_dump(usage),
                    context_manifest_json=_dump(context_manifest),
                    trace_json=_dump(trace),
                    warnings_json=_dump(warnings if isinstance(warnings, list) else []),
                    created_at=now,
                    updated_at=now,
                    executed_at=None,
                    reviewed_at=None,
                )
            )
        return self.get_agent_execution_plan(plan_id)

    def list_agent_execution_plans(
        self,
        user=None,
        dataset_id=None,
        task_type=None,
        status=None,
        campaign_id=None,
        limit=20,
        offset=0,
    ):
        task_type = normalize_task_type(task_type)
        try:
            limit = max(1, min(int(limit), 100))
        except (TypeError, ValueError):
            limit = 20
        try:
            offset = max(0, int(offset))
        except (TypeError, ValueError):
            offset = 0

        where_clauses = []
        if dataset_id:
            where_clauses.append(agent_execution_plans.c.dataset_id == str(dataset_id).strip())
        if task_type:
            where_clauses.append(agent_execution_plans.c.task_type.in_(task_spellings(task_type)))
        if status:
            where_clauses.append(agent_execution_plans.c.status == str(status).strip())
        if campaign_id:
            where_clauses.append(agent_execution_plans.c.campaign_id == str(campaign_id).strip())
        if user is not None and not _is_admin(user):
            where_clauses.append(agent_execution_plans.c.owner_user_id == user.get("id"))

        base_stmt = select(agent_execution_plans)
        if where_clauses:
            base_stmt = base_stmt.where(and_(*where_clauses))

        count_stmt = base_stmt.with_only_columns(func.count(agent_execution_plans.c.id))
        rows_stmt = (
            base_stmt.order_by(agent_execution_plans.c.created_at.desc())
            .limit(limit)
            .offset(offset)
        )

        with self.connect() as conn:
            total = conn.execute(count_stmt).scalar() or 0
            rows = conn.execute(rows_stmt).mappings().all()

        plans = [self._agent_execution_plan_summary_row(row) for row in rows]
        return plans, total

    def get_agent_execution_plan(self, plan_id, user=None):
        row = self._fetch_one(
            select(agent_execution_plans).where(agent_execution_plans.c.id == plan_id)
        )
        if not row or not self._visible_owned_row(row, user):
            return None
        return self._agent_execution_plan_row(row)

    def delete_agent_execution_plan(self, plan_id, user=None):
        plan = self.get_agent_execution_plan(plan_id, user=user)
        if not plan:
            return None
        with self.connect() as conn:
            conn.execute(delete(agent_execution_plans).where(agent_execution_plans.c.id == plan_id))
        return plan

    def update_agent_execution_plan(self, plan_id, **updates):
        allowed_json = {
            "actions",
            "execution",
            "review",
            "usage",
            "context_manifest",
            "trace",
            "warnings",
        }
        allowed_text = {"status", "provider", "model", "objective"}
        allowed_time = {"executed_at", "reviewed_at"}
        allowed_int = {"round_index"}
        values = {}
        for key, value in updates.items():
            if key in allowed_json:
                values[f"{key}_json"] = _dump(value)
            elif key in allowed_text:
                values[key] = str(value or "")
            elif key in allowed_time:
                values[key] = value
            elif key in allowed_int:
                values[key] = int(value or 0)
        if not values:
            return self.get_agent_execution_plan(plan_id)
        values["updated_at"] = utcnow()
        with self.connect() as conn:
            conn.execute(
                update(agent_execution_plans)
                .where(agent_execution_plans.c.id == plan_id)
                .values(**values)
            )
        return self.get_agent_execution_plan(plan_id)

    def list_leaderboard(
        self,
        task_type=None,
        dataset_id=None,
        metric="f1",
        split="auto",
        supports_inference=None,
        status="succeeded",
        limit=200,
        user=None,
        owner_only=False,
    ):
        task_type = normalize_task_type(task_type)
        metric = str(metric or "f1").strip().lower()
        split = str(split or "auto").strip().lower()
        if split == "val":
            split = "validation"
        if metric not in LEADERBOARD_METRICS:
            raise ValueError(f"unsupported leaderboard metric: {metric}")
        if split not in LEADERBOARD_SPLITS:
            raise ValueError(f"unsupported leaderboard split: {split}")

        try:
            limit = max(1, min(int(limit), 1000))
        except (TypeError, ValueError):
            limit = 200

        scoped_user = _owner_scoped_user(user) if owner_only else user
        visible_datasets = self.list_datasets(user=scoped_user)
        datasets_by_id = {dataset["id"]: dataset for dataset in visible_datasets}
        model_rows = self.list_models(user=scoped_user)
        models_by_id = {model["id"]: model for model in model_rows}
        models_by_experiment = {}
        for model in model_rows:
            models_by_experiment.setdefault(model["experiment_id"], model)

        rows = []
        for experiment in self.list_experiments(user=scoped_user):
            if task_type and normalize_task_type(experiment["task_type"]) != normalize_task_type(
                task_type
            ):
                continue
            if dataset_id and experiment["dataset_id"] != dataset_id:
                continue
            if experiment["dataset_id"] not in datasets_by_id:
                continue
            if status and status != "all" and experiment["status"] != status:
                continue

            model = models_by_id.get(experiment.get("model_id")) or models_by_experiment.get(
                experiment["id"]
            )
            model_supports_inference = bool(model and model.get("supports_inference"))
            if supports_inference is True and not model_supports_inference:
                continue
            if supports_inference is False and model_supports_inference:
                continue

            dataset = datasets_by_id.get(experiment["dataset_id"])
            summary = (experiment.get("metrics") or {}).get("summary") or {}
            score, score_source = _leaderboard_score(summary, split, metric)
            rows.append(
                {
                    "rank": 0,
                    "task_type": experiment["task_type"],
                    "experiment": {
                        "id": experiment["id"],
                        "name": experiment["name"],
                        "generated_name": experiment.get("generated_name"),
                        "config": experiment.get("config") or {},
                        "task_type": experiment["task_type"],
                        "status": experiment["status"],
                        "created_at": experiment["created_at"],
                        "updated_at": experiment["updated_at"],
                        "started_at": experiment["started_at"],
                        "finished_at": experiment["finished_at"],
                    },
                    "dataset": {
                        "id": dataset["id"],
                        "name": dataset["name"],
                        "source": dataset["source"],
                        "format": dataset["format"],
                    }
                    if dataset
                    else None,
                    "model": {
                        "id": model["id"],
                        "name": model["name"],
                        "name_is_custom": model.get("name_is_custom", False),
                        "supports_inference": model["supports_inference"],
                        "evaluation_protocol_version": int(
                            (
                                (model.get("metadata") or {}).get("evaluation_protocol_version", 1)
                                or 1
                            )
                        ),
                        "created_at": model["created_at"],
                    }
                    if model
                    else None,
                    "metrics": {
                        "summary": summary,
                        "validation": _split_metrics(summary, "val"),
                        "test": _split_metrics(summary, "test"),
                    },
                    "config_summary": _leaderboard_config_summary(experiment.get("config") or {}),
                    "duration_seconds": _duration_seconds(
                        experiment.get("started_at"), experiment.get("finished_at")
                    ),
                    "score": score,
                    "score_source": score_source,
                }
            )

        rows.sort(
            key=lambda row: (
                row["score"] is not None,
                row["score"] if row["score"] is not None else float("-inf"),
                row["experiment"].get("finished_at") or row["experiment"].get("created_at") or "",
            ),
            reverse=True,
        )
        rows = rows[:limit]
        for index, row in enumerate(rows, start=1):
            row["rank"] = index
        return rows

    def _admin_user_summary(self, user):
        if not user:
            return None
        return {
            **user,
            "quota": self.get_user_quota(user["id"]),
            "usage": self.get_quota_usage(user["id"]),
            "resource_counts": self.count_user_resources(user["id"]),
        }

    @staticmethod
    def _audit_user_ref(user):
        return {
            "id": user["id"],
            "email": user["email"],
            "display_name": user.get("display_name") or default_display_name(user["email"]),
            "role": user["role"],
            "status": user["status"],
        }

    @staticmethod
    def _user_row(row, include_password=False):
        payload = {
            "id": row["id"],
            "email": row["email"],
            "display_name": row["display_name"] or default_display_name(row["email"]),
            "preferred_locale": row.get("preferred_locale") or "en",
            "email_verified": bool(row["email_verified"]),
            "role": row["role"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        if include_password:
            payload["password_hash"] = row["password_hash"]
        return payload

    @staticmethod
    def _email_code_row(row):
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "purpose": row["purpose"],
            "code_hash": row["code_hash"],
            "expires_at": row["expires_at"],
            "attempts": int(row["attempts"]),
            "consumed_at": row["consumed_at"],
            "created_at": row["created_at"],
        }

    @staticmethod
    def _quota_row(row):
        return {
            "user_id": row["user_id"],
            "daily_experiment_limit": int(row["daily_experiment_limit"]),
            "daily_concurrent_experiment_limit": int(row["daily_concurrent_experiment_limit"]),
            "daily_inference_request_limit": int(row["daily_inference_request_limit"]),
            "daily_inference_input_limit": int(row["daily_inference_input_limit"]),
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _quota_usage_row(row):
        return {
            "user_id": row["user_id"],
            "usage_date": row["usage_date"],
            "experiments_created": int(row["experiments_created"]),
            "inference_requests": int(row["inference_requests"]),
            "inference_inputs": int(row["inference_inputs"]),
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _audit_log_row(row):
        return {
            "id": row["id"],
            "actor_user_id": row["actor_user_id"],
            "target_user_id": row["target_user_id"],
            "event_type": row["event_type"],
            "ip_address": row["ip_address"],
            "user_agent": row["user_agent"],
            "metadata": _load(row["metadata_json"]),
            "created_at": row["created_at"],
        }

    @staticmethod
    def _admin_session_row(row):
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "expires_at": row["expires_at"],
            "user_agent": row["user_agent"],
            "created_at": row["created_at"],
            "last_seen_at": row["last_seen_at"],
            "revoked_at": row["revoked_at"],
            "active": row["revoked_at"] is None,
        }

    @staticmethod
    def _admin_email_code_row(row):
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "purpose": row["purpose"],
            "expires_at": row["expires_at"],
            "attempts": int(row["attempts"]),
            "consumed_at": row["consumed_at"],
            "created_at": row["created_at"],
            "active": row["consumed_at"] is None,
        }

    @staticmethod
    def _session_row(row):
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "token_hash": row["token_hash"],
            "expires_at": row["expires_at"],
            "user_agent": row["user_agent"],
            "created_at": row["created_at"],
            "last_seen_at": row["last_seen_at"],
            "revoked_at": row["revoked_at"],
        }

    @staticmethod
    def _dataset_row(row):
        return {
            "id": row["id"],
            "name": builtin_name_for_display(row["name"], row["source"], row["task_type"]),
            "task_type": normalize_task_type(row["task_type"]),
            "source": row["source"],
            "format": row["format"],
            "root_path": row["root_path"],
            "metadata": _load(row["metadata_json"]),
            "owner_user_id": row.get("owner_user_id"),
            "created_at": row["created_at"],
        }

    @staticmethod
    def _experiment_row(row):
        try:
            failure = _load(row.get("failure_json"), {})
        except (TypeError, ValueError, json.JSONDecodeError):
            failure = {}
        if not isinstance(failure, dict) or not failure:
            failure = None
        payload = {
            "id": row["id"],
            "dataset_id": row["dataset_id"],
            "owner_user_id": row.get("owner_user_id"),
            "name": row["name"],
            "task_type": normalize_task_type(row["task_type"]),
            "config": normalize_stream_fields(_load(row["config_json"])),
            "status": row["status"],
            "log_path": row["log_path"],
            "artifact_dir": row["artifact_dir"],
            "metrics": without_legacy_aupr(_load(row["metrics_json"], {"events": [], "summary": {}})),
            "model_id": row["model_id"],
            "error": row["error"],
            "failure": failure,
            "resume": _load(row.get("resume_json"), {}) or None,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
        }
        if row.get("name_kind") == "config":
            payload["generated_name"] = {"kind": "config"}
            if row.get("name_qualifier") in GENERATED_NAME_QUALIFIERS:
                payload["generated_name"]["qualifier"] = row["name_qualifier"]
        elif row.get("name_kind") == "rerun" and row.get("name_base"):
            payload["generated_name"] = {
                "kind": "rerun",
                "base_name": row["name_base"],
                "base_kind": ("config" if row.get("name_base_kind") == "config" else "literal"),
                "count": max(1, int(row.get("rerun_count") or 1)),
                "source_experiment_id": row.get("rerun_source_experiment_id"),
            }
            if row.get("name_qualifier") in GENERATED_NAME_QUALIFIERS:
                payload["generated_name"]["qualifier"] = row["name_qualifier"]
        else:
            payload["generated_name"] = None
        return payload

    @staticmethod
    def _model_row(row):
        metadata = normalize_stream_fields(_load(row["metadata_json"]))
        if _metadata_requires_sample_profile(metadata):
            metadata.setdefault("reason", PROFILE_DEPENDENT_INFERENCE_REASON)
            metadata.setdefault("reason_code", "profile_features_require_sample_id")
            metadata.setdefault("inference_unavailable_reason", PROFILE_DEPENDENT_INFERENCE_REASON)
            metadata.setdefault("inference_unavailable_code", "profile_features_require_sample_id")
        metadata = without_legacy_aupr(metadata)
        # A migration is a derived view, never a rewrite of sealed metadata.
        from pepplatform.metric_migration import load_metric_reference
        try:
            reference = load_metric_reference(row["bundle_path"])
        except (OSError, ValueError, TypeError):
            reference = None
        if reference is not None and isinstance(metadata.get("web_ensemble"), dict):
            splits = metadata["web_ensemble"].get("test_splits") or {}
            for split, entry in reference["splits"].items():
                if split in splits:
                    splits[split] = entry["metrics"]
            metadata["web_ensemble"]["test_splits"] = splits
        return {
            "id": row["id"],
            "experiment_id": row["experiment_id"],
            "owner_user_id": row.get("owner_user_id"),
            "name": row["name"],
            "name_is_custom": bool(row.get("name_is_custom")),
            "task_type": normalize_task_type(row["task_type"]),
            "bundle_path": row["bundle_path"],
            "supports_inference": _metadata_supports_raw_sequence_inference(
                row["supports_inference"],
                metadata,
            ),
            "metadata": metadata,
            "created_at": row["created_at"],
        }

    @staticmethod
    def _inference_run_row(row):
        return {
            "id": row["id"],
            "model_id": row["model_id"],
            "owner_user_id": row.get("owner_user_id"),
            "input_kind": row["input_kind"],
            "input_count": row["input_count"],
            "prediction_count": row["prediction_count"],
            "request": _load(row["request_json"]),
            "response": _load(row["response_json"]),
            "created_at": row["created_at"],
        }

    @staticmethod
    def _ai_provider_config_row(row, include_secret=False):
        payload = {
            "id": row["id"],
            "owner_user_id": row.get("owner_user_id"),
            "provider": row["provider"],
            "display_name": row["display_name"],
            "base_url": row["base_url"],
            "model": row["model"],
            "api_key_last4": row["api_key_last4"],
            "has_api_key": bool(row["api_key_last4"]),
            "capabilities": _load(row.get("capabilities_json"), {}),
            "enabled": bool(row["enabled"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        if include_secret:
            payload["api_key_ciphertext"] = row["api_key_ciphertext"]
        return payload

    @staticmethod
    def _agent_chat_conversation_row(row):
        return {
            "id": row["id"],
            "owner_user_id": row.get("owner_user_id"),
            "scope": row.get("scope") or "agent",
            "experiment_id": row.get("experiment_id"),
            "title": row["title"],
            "pinned": bool(row.get("pinned_at")),
            "pinned_at": row.get("pinned_at"),
            "last_provider_config_id": row.get("last_provider_config_id"),
            "last_provider": row.get("last_provider") or "",
            "last_model": row.get("last_model") or "",
            "last_message_preview": row.get("last_message_preview") or "",
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _agent_chat_message_row(row):
        return {
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "ordinal": int(row["ordinal"]),
            "role": row["role"],
            "content": row.get("content") or "",
            "status": row["status"],
            "kind": row.get("kind") or "chat",
            "provider_config_id": row.get("provider_config_id"),
            "provider": row.get("provider") or "",
            "model": row.get("model") or "",
            "finish_reason": row.get("finish_reason") or "",
            "error": row.get("error") or "",
            "usage": _load(row.get("usage_json"), {}),
            "web_search": _load(row.get("web_search_json"), {}),
            "payload": _load(row.get("payload_json"), {}),
            "context_manifest": _load(row.get("context_manifest_json"), {}),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _agent_campaign_row(row):
        return {
            "id": row["id"],
            "owner_user_id": row.get("owner_user_id"),
            "task_type": normalize_task_type(row["task_type"]),
            "dataset_id": row["dataset_id"],
            "objective": row["objective"],
            "status": row["status"],
            "max_rounds": row["max_rounds"],
            "current_round_index": row["current_round_index"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _agent_campaign_summary_row(row):
        return MetadataStore._agent_campaign_row(row)

    @staticmethod
    def _agent_execution_plan_row(row):
        return {
            "id": row["id"],
            "owner_user_id": row.get("owner_user_id"),
            "campaign_id": row.get("campaign_id") or "",
            "round_index": row.get("round_index") or 0,
            "provider_config_id": row["provider_config_id"],
            "provider": row["provider"],
            "model": row["model"],
            "task_type": normalize_task_type(row["task_type"]),
            "dataset_id": row["dataset_id"],
            "objective": row["objective"],
            "status": row["status"],
            "actions": _load(row["actions_json"], []),
            "execution": _load(row["execution_json"]),
            "review": _load(row["review_json"]),
            "usage": _load(row["usage_json"]),
            "context_manifest": _load(row["context_manifest_json"]),
            "trace": _load(row.get("trace_json"), {}),
            "warnings": _load(row["warnings_json"], []),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "executed_at": row["executed_at"],
            "reviewed_at": row["reviewed_at"],
        }

    @staticmethod
    def _agent_execution_plan_summary_row(row):
        actions = _load(row["actions_json"], [])
        execution = _load(row["execution_json"])
        review = _load(row["review_json"])
        created_entries = (
            execution.get("created_experiments") if isinstance(execution, dict) else []
        )
        canceled_entries = (
            execution.get("canceled_experiments") if isinstance(execution, dict) else []
        )
        review_summary = review.get("summary") if isinstance(review, dict) else ""
        return {
            "id": row["id"],
            "campaign_id": row.get("campaign_id") or "",
            "round_index": row.get("round_index") or 0,
            "status": row["status"],
            "task_type": normalize_task_type(row["task_type"]),
            "dataset_id": row["dataset_id"],
            "objective": row["objective"],
            "model": row["model"],
            "provider": row["provider"],
            "provider_config_id": row["provider_config_id"],
            "action_count": len(actions) if isinstance(actions, list) else 0,
            "valid_action_count": sum(
                1 for action in actions if isinstance(action, dict) and action.get("valid")
            ),
            "created_experiment_count": len(created_entries)
            if isinstance(created_entries, list)
            else 0,
            "canceled_experiment_count": len(canceled_entries)
            if isinstance(canceled_entries, list)
            else 0,
            "has_review": bool(review_summary),
            "trace": _load(row.get("trace_json"), {}),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "executed_at": row["executed_at"],
            "reviewed_at": row["reviewed_at"],
        }
