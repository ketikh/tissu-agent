"""Shared test fixtures for the Tissu Agent test suite."""

import os
import sqlite3
import tempfile
from typing import Generator

import pytest


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set test environment variables so real keys are never used."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


@pytest.fixture
def temp_db() -> Generator[str, None, None]:
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    os.unlink(db_path)


@pytest.fixture
def db_connection(temp_db: str) -> Generator[sqlite3.Connection, None, None]:
    """Create a database connection for testing."""
    conn = sqlite3.connect(temp_db)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()
