import logging
from logging.handlers import BufferingHandler
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.config import Config

from alembic import command


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
def test_sqlite_migrations_reach_head_and_preserve_chat_history(tmp_path, monkeypatch, existing_history):
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
                "20260918_0012"
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
