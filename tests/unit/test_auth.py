"""Unit tests for the APIKeyMiddleware.

These tests run the middleware against a tiny throwaway FastAPI app so we
don't need a real Postgres connection. We cover: happy path, missing key,
wrong key, exempt paths, and the case where ADMIN_API_KEY itself is
unset in the environment (fail closed).
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.auth import APIKeyMiddleware


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "secret-xyz")

    app = FastAPI()
    app.add_middleware(APIKeyMiddleware)

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/leads")
    def leads():
        return {"leads": []}

    @app.get("/api/owner-confirm/{token}")
    def owner_confirm(token: str):
        return {"token": token}

    @app.get("/api/meta/data-deletion")
    def data_deletion():
        return {"ok": True}

    @app.get("/")
    def home():
        return {"page": "home"}

    return TestClient(app)


def test_health_is_exempt(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200


def test_missing_key_rejected(client):
    resp = client.get("/api/leads")
    assert resp.status_code == 401
    assert resp.json() == {"error": "unauthorized"}


def test_wrong_key_rejected(client):
    resp = client.get("/api/leads", headers={"X-API-Key": "nope"})
    assert resp.status_code == 401
    assert resp.json() == {"error": "unauthorized"}


def test_valid_key_allowed(client):
    resp = client.get("/api/leads", headers={"X-API-Key": "secret-xyz"})
    assert resp.status_code == 200


def test_owner_confirm_exempt(client):
    resp = client.get("/api/owner-confirm/any-token")
    assert resp.status_code == 200


def test_meta_data_deletion_exempt(client):
    resp = client.get("/api/meta/data-deletion")
    assert resp.status_code == 200


def test_non_api_routes_bypass(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_fail_closed_when_env_unset(monkeypatch):
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    app = FastAPI()
    app.add_middleware(APIKeyMiddleware)

    @app.get("/api/leads")
    def leads():
        return {"leads": []}

    c = TestClient(app)
    resp = c.get("/api/leads", headers={"X-API-Key": "anything"})
    assert resp.status_code == 401
    assert resp.json() == {"error": "unauthorized"}
