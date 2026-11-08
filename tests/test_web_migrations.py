import json
import logging
from logging.handlers import BufferingHandler
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.config import Config

from alembic import command


@pytest.mark.parametrize("direct_initialization", [False, True])
def test_email_budget_upgrade_preserves_old_policy_and_verification_codes(
    tmp_path, monkeypatch, direct_initialization
):
    from copy import deepcopy

    from pepplatform.web.resource_policy import DEFAULT_POLICY
    from pepplatform.web.store import MetadataStore

    root = Path(__file__).resolve().parents[1]
    database_url = f"sqlite:///{tmp_path / 'email-upgrade.sqlite3'}"
    monkeypatch.setenv("PEP_DATABASE_URL", database_url)
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    command.upgrade(config, "20261107_0016")
    old = MetadataStore(database_url, initialize=False)
    owner = old.create_user("preserved@example.test", "preserved-password-hash")
    code = old.create_email_code(owner["id"], "email_verify", "preserved-code-hash", "2099-01-01")
    policy = deepcopy(DEFAULT_POLICY)
    del policy["limits"]["daily_verification_email_limit"]
    policy["limits"]["daily_registration_limit"] = 7
    stored = json.dumps(policy)
    with old.connect() as conn:
        conn.execute(sa.text(
            "INSERT INTO platform_resource_policies (id,revision,config_json,updated_at) "
            "VALUES ('resources',4,:policy,'2026-11-07')"
        ), {"policy": stored})
    old.engine.dispose()
    if not direct_initialization:
        command.upgrade(config, "head")
        command.upgrade(config, "head")
    upgraded = MetadataStore(database_url)
    try:
        assert upgraded.resources.policy()["limits"]["daily_verification_email_limit"] == 20
        assert upgraded.resources.policy()["limits"]["daily_registration_limit"] == 7
        assert upgraded.get_email_code(code["id"]) == code
        assert upgraded.get_user_by_email(owner["email"])["password_hash"] == "preserved-password-hash"
        assert upgraded.resources.email_delivery_usage()["used"] == 0
        upgraded.resources.reserve_verification_email()
        assert upgraded.resources.email_delivery_usage()["used"] == 1
        with upgraded.connect() as conn:
            assert conn.scalar(sa.text("SELECT config_json FROM platform_resource_policies")) == stored
    finally:
        upgraded.engine.dispose()


def test_migrations_preserve_application_error_logging(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("PEP_DATABASE_URL", f"sqlite:///{tmp_path / 'metadata.sqlite3'}")
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    logger = logging.getLogger("pepplatform.web.auth")
    monkeypatch.setattr(logger, "disabled", False)

    command.upgrade(config, "head")

    handler = BufferingHandler(capacity=10)
    logger.addHandler(handler)
    try:
        logger.error("SMTP delivery failed after database migration")
        assert [record.getMessage() for record in handler.buffer] == [
            "SMTP delivery failed after database migration"
        ]
    finally:
        logger.removeHandler(handler)
        handler.close()


@pytest.mark.parametrize("existing_history", [False, True])
def test_sqlite_migrations_reach_head_and_preserve_chat_history(
    tmp_path, monkeypatch, existing_history
):
    root = Path(__file__).resolve().parents[1]
    database_url = f"sqlite:///{tmp_path / 'metadata.sqlite3'}"
    monkeypatch.setenv("PEP_DATABASE_URL", database_url)
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    engine = sa.create_engine(database_url)
    try:
        if existing_history:
            command.upgrade(config, "20260720_0009")
            with engine.begin() as connection:
                connection.execute(
                    sa.text(
                        "INSERT INTO agent_chat_conversations "
                        "(id, title, created_at, updated_at) "
                        "VALUES ('chat', 'Retained history', '2026-07-20', '2026-07-20')"
                    )
                )
                connection.execute(
                    sa.text(
                        "INSERT INTO agent_chat_messages "
                        "(id, conversation_id, ordinal, role, content, status, created_at, updated_at) "
                        "VALUES ('message', 'chat', 1, 'user', 'Retain this message', "
                        "'completed', '2026-07-20', '2026-07-20')"
                    )
                )

        command.upgrade(config, "head")
        command.upgrade(config, "head")
        assert any(
            foreign_key["referred_table"] == "experiments"
            and foreign_key["options"].get("ondelete") == "CASCADE"
            for foreign_key in sa.inspect(engine).get_foreign_keys("agent_chat_conversations")
        )
        with engine.connect() as connection:
            assert connection.scalar(sa.text("SELECT version_num FROM alembic_version")) == (
                "20261108_0017"
            )
        if existing_history:
            command.downgrade(config, "20260720_0009")
            command.upgrade(config, "head")
            with engine.connect() as connection:
                assert connection.execute(
                    sa.text("SELECT title, scope, experiment_id FROM agent_chat_conversations")
                ).one() == ("Retained history", "agent", None)
                assert connection.scalar(sa.text("SELECT content FROM agent_chat_messages")) == (
                    "Retain this message"
                )
    finally:
        engine.dispose()


@pytest.mark.parametrize("direct_initialization", [False, True])
def test_optional_announcements_upgrade_preserves_history_and_cleanup_deadlines(
    tmp_path, monkeypatch, direct_initialization
):
    from copy import deepcopy

    from pepplatform.versioning.core import VersionRepository
    from pepplatform.web.resource_policy import DEFAULT_POLICY
    from pepplatform.web.store import MetadataStore

    root = Path(__file__).resolve().parents[1]
    database_url = f"sqlite:///{tmp_path / 'announcement-upgrade.sqlite3'}"
    monkeypatch.setenv("PEP_DATABASE_URL", database_url)
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    command.upgrade(config, "20261107_0015")
    old = MetadataStore(database_url, initialize=False)
    old.init_schema()
    owner = old.create_user("retained@example.test", "unused")
    data = old.create_dataset("data", "pep_function", "upload", "sequence_dir", tmp_path / "data")
    experiment = old.create_experiment(
        "scientific",
        data["id"],
        "pep_function",
        {"seed": 42, "epochs": 23},
        tmp_path / "train.log",
        tmp_path / "artifact",
    )
    versions = VersionRepository(old.engine, tmp_path / "versions")
    source = tmp_path / "source"
    source.write_bytes(b"scientific content unchanged by announcement settings")
    version = versions.snapshot("dataset", {"input.txt": source})
    legacy_policy = deepcopy(DEFAULT_POLICY)
    legacy_policy["retention"].pop("publish_announcements")
    started = "2026-11-01T00:00:00+00:00"
    deadline = "2026-11-04T00:00:00+00:00"
    with old.engine.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO platform_resource_policies (id,revision,config_json,updated_at) "
                "VALUES ('resources',7,:policy,:started)"
            ),
            {"policy": json.dumps(legacy_policy), "started": started},
        )
        conn.execute(
            sa.text(
                "INSERT INTO resource_inventory "
                "(id,path,kind,owners_json,size_bytes,protected,reason,state,notified_at,delete_after,last_seen_at) "
                "VALUES ('cache',:path,'inference_cache','[]',10,0,'rebuildable','notified',:started,:deadline,:started)"
            ),
            {"path": str(tmp_path / "cache"), "started": started, "deadline": deadline},
        )
        conn.execute(
            sa.text(
                "INSERT INTO model_retention_states "
                "(model_id,experiment_id,state,notified_at,delete_after,manifest_json,updated_at) "
                "VALUES ('model',:experiment,'notified',:started,:deadline,'{}',:started)"
            ),
            {"experiment": experiment["id"], "started": started, "deadline": deadline},
        )
        conn.execute(
            sa.text(
                "INSERT INTO platform_announcements "
                "(id,title,body,audience_json,pinned,kind,publish_at,created_at,updated_at) "
                "VALUES ('old-notice','Cleanup','Retained announcement','\"all\"',0,'cleanup',:started,:started,:started)"
            ),
            {"started": started},
        )
        conn.execute(
            sa.text(
                "INSERT INTO announcement_reads (announcement_id,user_id,read_at) VALUES ('old-notice',:owner,:started)"
            ),
            {"owner": owner["id"], "started": started},
        )
        tables = (
            "resource_inventory",
            "model_retention_states",
            "platform_announcements",
            "announcement_reads",
        )
        before = {
            name: dict(conn.execute(sa.text(f"SELECT * FROM {name}")).mappings().one())
            for name in tables
        }
    old.engine.dispose()
    if not direct_initialization:
        command.upgrade(config, "head")
        command.upgrade(config, "head")
    upgraded = MetadataStore(database_url)
    try:
        assert upgraded.get_experiment(experiment["id"])["config"] == experiment["config"]
        assert (
            VersionRepository(upgraded.engine, tmp_path / "versions").verify(version["id"])[
                "status"
            ]
            == "verified"
        )
        assert upgraded.resources.policy()["revision"] == 7
        assert upgraded.resources.policy()["retention"]["publish_announcements"] is False
        with upgraded.connect() as conn:
            for name, previous in before.items():
                row = dict(conn.execute(sa.text(f"SELECT * FROM {name}")).mappings().one())
                if name in {"resource_inventory", "model_retention_states"}:
                    assert row.pop("scheduled_at") == started
                assert row == previous
    finally:
        upgraded.engine.dispose()


def test_quota_transition_upgrade_retains_records_and_backfills_queued_policy(
    tmp_path, monkeypatch
):
    from pepplatform.web.resource_policy import DEFAULT_POLICY, resource_pending_jobs
    from pepplatform.web.store import MetadataStore

    root = Path(__file__).resolve().parents[1]
    database_url = f"sqlite:///{tmp_path / 'quota-upgrade.sqlite3'}"
    monkeypatch.setenv("PEP_DATABASE_URL", database_url)
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    command.upgrade(config, "20261107_0014")
    old_store = MetadataStore(database_url, initialize=False)
    old_store.init_schema()
    owner = old_store.create_user("upgrade@example.test", "unused")
    data = old_store.create_dataset(
        "old data",
        "pep_function",
        "upload",
        "sequence_dir",
        tmp_path / "data",
        owner_user_id=owner["id"],
    )
    experiment = old_store.create_experiment(
        "accepted",
        data["id"],
        "pep_function",
        {"seed": 42, "epochs": 23},
        tmp_path / "train.log",
        tmp_path / "artifact",
        owner_user_id=owner["id"],
    )
    old_store.increment_quota_usage(owner["id"], experiments_created=8)
    with old_store.connect() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO platform_resource_policies (id,revision,config_json,updated_at) VALUES ('resources',3,:policy,'before')"
            ),
            {"policy": json.dumps(DEFAULT_POLICY)},
        )
        conn.execute(
            sa.text(
                "INSERT INTO user_resource_settings (user_id,overrides_json,allow_global_override,updated_at) VALUES (:user,:quota,0,'before')"
            ),
            {"user": owner["id"], "quota": json.dumps({"daily_experiment_limit": 17})},
        )
        conn.execute(
            sa.text(
                "INSERT INTO resource_pending_jobs (experiment_id,enqueued_at) VALUES (:id,'before')"
            ),
            {"id": experiment["id"]},
        )
    old_store.engine.dispose()
    command.upgrade(config, "head")
    command.upgrade(config, "head")
    upgraded = MetadataStore(database_url)
    try:
        quota = upgraded.get_user_quota(owner["id"])
        assert quota["daily_experiment_limit"] == 17 and quota["pending"] == {}
        assert upgraded.get_quota_usage(owner["id"])["experiments_created"] == 8
        assert upgraded.get_experiment(experiment["id"])["config"] == experiment["config"]
        with upgraded.connect() as conn:
            row = conn.execute(sa.select(resource_pending_jobs)).mappings().one()
            snapshot = json.loads(row["policy_json"])
            assert row["enqueued_at"] == "before"
            assert (
                snapshot["daily_experiment_limit"] == 17
                and snapshot["snapshot_source"] == "upgrade"
            )
            assert conn.scalar(sa.text("SELECT updated_at FROM user_resource_settings")) == "before"
    finally:
        upgraded.engine.dispose()


def test_model_retention_upgrade_preserves_existing_quotas_and_requires_new_opt_in(
    tmp_path, monkeypatch
):
    root = Path(__file__).resolve().parents[1]
    database_url = f"sqlite:///{tmp_path / 'metadata.sqlite3'}"
    monkeypatch.setenv("PEP_DATABASE_URL", database_url)
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    command.upgrade(config, "20261107_0013")
    engine = sa.create_engine(database_url)
    old = {
        "defaults": {"protected_model_limit": 8, "daily_experiment_limit": 17},
        "retention": {"enabled": True, "notice_hours": 96},
    }
    try:
        with engine.begin() as conn:
            conn.execute(
                sa.text(
                    "INSERT INTO platform_resource_policies (id,revision,config_json,updated_at) VALUES ('resources',9,:policy,'before')"
                ),
                {"policy": json.dumps(old)},
            )
            conn.execute(
                sa.text(
                    "INSERT INTO user_resource_settings (user_id,overrides_json,allow_global_override,updated_at) VALUES ('existing',:quota,0,'before')"
                ),
                {
                    "quota": json.dumps(
                        {"protected_model_limit": 10, "daily_inference_request_limit": 72}
                    )
                },
            )
        command.upgrade(config, "head")
        command.upgrade(config, "head")
        with engine.connect() as conn:
            row = conn.execute(
                sa.text("SELECT revision,config_json,updated_at FROM platform_resource_policies")
            ).one()
            policy = json.loads(row.config_json)
            assert row.revision == 9 and row.updated_at == "before"
            assert policy["defaults"] == {**old["defaults"], "max_model_count": 8}
            assert policy["retention"] == {**old["retention"], "delete_unprotected_models": False}
            quota = conn.execute(
                sa.text(
                    "SELECT overrides_json,allow_global_override,updated_at FROM user_resource_settings"
                )
            ).one()
            assert json.loads(quota.overrides_json) == {
                "protected_model_limit": 10,
                "max_model_count": 10,
                "daily_inference_request_limit": 72,
            }
            assert quota.allow_global_override == 0 and quota.updated_at == "before"
        assert {"model_retention_states", "retired_model_versions"} <= set(
            sa.inspect(engine).get_table_names()
        )
    finally:
        engine.dispose()
