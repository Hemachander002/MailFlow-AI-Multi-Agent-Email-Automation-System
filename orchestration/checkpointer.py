from pathlib import Path

import psycopg
from dotenv import dotenv_values
from langgraph.checkpoint.postgres import PostgresSaver


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG = dotenv_values(BASE_DIR / ".env")

DATABASE_URL = CONFIG.get("DATABASE_URL")

_connection = None
_checkpointer = None


def get_checkpointer():
    """
    Return a persistent PostgreSQL-backed LangGraph checkpointer.
    """

    global _connection, _checkpointer

    if _checkpointer is not None:
        return _checkpointer

    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is missing from the project .env file."
        )

    connection_url = DATABASE_URL.replace(
        "postgresql+psycopg://",
        "postgresql://",
        1,
    )

    _connection = psycopg.connect(
        connection_url,
        autocommit=True,
    )

    _checkpointer = PostgresSaver(_connection)

    # Create LangGraph checkpoint tables if they do not exist.
    _checkpointer.setup()

    print("[CHECKPOINTER] PostgreSQL checkpointing initialized.")

    return _checkpointer


def close_checkpointer():
    """
    Close the persistent PostgreSQL connection during shutdown.
    """

    global _connection, _checkpointer

    if _connection is not None:
        _connection.close()

    _connection = None
    _checkpointer = None