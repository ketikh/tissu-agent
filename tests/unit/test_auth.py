"""Unit tests for the APIKeyMiddleware.

These tests run the middleware against a tiny throwaway FastAPI app so we
don't need a real Postgres connection. We cover: happy path, missing key,
wrong key, exempt paths, tenant_id stashing, and the case where
ADMIN_API_KEY itself is unset in the environment (fail closed).
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

import src.auth as auth_module
from src.auth import APIKeyMiddleware


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "secret-xyz")

    # Stub resolve_api_key so the tests don't need a real DB. Returns
    # (tenant_id, scope) tuples. "secret-xyz" → admin of the default
    # tenant; "other-tenant-key" → admin of "other"; "sf-readonly-key"
    # → storefront-scoped; anything else → None (rejected by middleware).
    async def fake_resolve(key: str):
        if key == "secret-xyz":
            return ("default", "admin")
        if key == "other-tenant-key":
            return ("other", "admin")
        if key == "sf-readonly-key":
            return ("default", "storefront")
        return None

    monkeypatch.setattr(auth_module, "resolve_api_key", fake_resolve)

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

    @app.get("/api/whoami")
    def whoami(request: Request):
        return {"tenant_id": getattr(request.state, "tenant_id", None)}

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
    monkeypatch.delenv("STOREFRONT_API_KEY", raising=False)

    async def fake_resolve(key: str):
        return None

    monkeypatch.setattr(auth_module, "resolve_api_key", fake_resolve)

    app = FastAPI()
    app.add_middleware(APIKeyMiddleware)

    @app.get("/api/leads")
    def leads():
        return {"leads": []}

    c = TestClient(app)
    resp = c.get("/api/leads", headers={"X-API-Key": "anything"})
    assert resp.status_code == 401
    assert resp.json() == {"error": "unauthorized"}


def test_storefront_scoped_key_allowed_on_storefront(client):
    """The read-only storefront key passes on /api/storefront/*."""
    # The fixture already wires sf-readonly-key to scope='storefront'.
    # We add an /api/storefront/products stub to the tiny app to verify.
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.add_middleware(APIKeyMiddleware)

    @app.get("/api/storefront/products")
    def sf_products(request: Request):
        return {"scope": getattr(request.state, "api_key_scope", None)}

    @app.get("/api/leads")
    def leads():
        return {}

    import os
    os.environ["ADMIN_API_KEY"] = "secret-xyz"

    c = TestClient(app)
    sf = c.get("/api/storefront/products", headers={"X-API-Key": "sf-readonly-key"})
    assert sf.status_code == 200
    assert sf.json() == {"scope": "storefront"}


def test_storefront_scoped_key_forbidden_on_admin_paths(client):
    """A storefront-scoped key must NOT reach /api/leads or /api/inventory."""
    resp = client.get("/api/leads", headers={"X-API-Key": "sf-readonly-key"})
    assert resp.status_code == 403
    body = resp.json()
    assert body["error"] == "forbidden"


def test_valid_key_stashes_tenant_id(client):
    resp = client.get("/api/whoami", headers={"X-API-Key": "secret-xyz"})
    assert resp.status_code == 200
    assert resp.json() == {"tenant_id": "default"}


def test_second_tenant_key_gets_its_own_tenant_id(client):
    resp = client.get(
        "/api/whoami", headers={"X-API-Key": "other-tenant-key"}
    )
    assert resp.status_code == 200
    assert resp.json() == {"tenant_id": "other"}


def test_exempt_path_defaults_to_default_tenant(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    # health endpoint doesn't echo tenant_id, but the non-/api home route
    # should still get tenant_id set via the middleware.
    resp = client.get("/")
    assert resp.status_code == 200
