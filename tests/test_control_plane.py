import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import Session, sessionmaker

from core.control_plane.app import app
from core.db.models import Base
from core.db.session import get_db, init_db


@pytest.fixture
def client_with_test_db():
    """Create a FastAPI TestClient wired to an isolated in-memory SQLite database."""
    # StaticPool ensures the in-memory database persists across all sessions and requests
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(target_engine=test_engine)
    test_session_factory = sessionmaker(test_engine, expire_on_commit=False)

    def override_get_db():
        with test_session_factory() as session:
            yield session

    # Swap production DB dependency with in-memory test DB
    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        yield client

    # Cleanup after test finishes
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=test_engine)


def test_register_new_node(client_with_test_db: TestClient):
    """Verify registering a new GPU node returns 200 and records the node."""
    payload = {
        "id": "node-rtx3050-01",
        "hostname": "local-workstation",
        "ip_address": "192.168.1.100",
        "gpu_name": "NVIDIA GeForce RTX 3050 Laptop GPU",
        "total_vram_mb": 4096,
        "free_vram_mb": 3962,
    }
    response = client_with_test_db.post("/api/v1/nodes/register", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "registered", "node_id": "node-rtx3050-01"}


def test_register_node_idempotent(client_with_test_db: TestClient):
    """Verify that registering the same node twice updates hardware without database errors."""
    payload_1 = {
        "id": "node-rtx3050-01",
        "hostname": "local-workstation",
        "ip_address": "192.168.1.100",
        "gpu_name": "RTX 3050",
        "total_vram_mb": 4096,
        "free_vram_mb": 3962,
    }
    # First registration
    res1 = client_with_test_db.post("/api/v1/nodes/register", json=payload_1)
    assert res1.status_code == 200

    # Second registration with new IP and updated free VRAM
    payload_2 = {
        "id": "node-rtx3050-01",
        "hostname": "local-workstation",
        "ip_address": "192.168.1.105",  # DHCP renewed IP
        "gpu_name": "RTX 3050",
        "total_vram_mb": 4096,
        "free_vram_mb": 2500,           # Lower free VRAM
    }
    res2 = client_with_test_db.post("/api/v1/nodes/register", json=payload_2)
    assert res2.status_code == 200
    assert res2.json() == {"status": "registered", "node_id": "node-rtx3050-01"}

    # Verify updated values via GET /api/v1/nodes
    list_res = client_with_test_db.get("/api/v1/nodes")
    assert list_res.status_code == 200
    nodes = list_res.json()
    assert len(nodes) == 1
    assert nodes[0]["ip_address"] == "192.168.1.105"
    assert nodes[0]["free_vram_mb"] == 2500


def test_node_heartbeat_success(client_with_test_db: TestClient):
    """Verify periodic heartbeat updates telemetry."""
    # Register first
    client_with_test_db.post(
        "/api/v1/nodes/register",
        json={"id": "node-01", "hostname": "host", "ip_address": "127.0.0.1"},
    )

    # Send heartbeat
    heartbeat_payload = {"free_vram_mb": 2048, "status": "online"}
    response = client_with_test_db.post("/api/v1/nodes/node-01/heartbeat", json=heartbeat_payload)
    assert response.status_code == 200
    assert response.json() == {"status": "acknowledged", "node_id": "node-01"}


def test_node_heartbeat_not_found(client_with_test_db: TestClient):
    """Verify heartbeat on an unregistered node returns 404."""
    heartbeat_payload = {"free_vram_mb": 2048, "status": "online"}
    response = client_with_test_db.post("/api/v1/nodes/ghost-node/heartbeat", json=heartbeat_payload)
    assert response.status_code == 404
    assert "not registered" in response.json()["detail"]


def test_list_nodes(client_with_test_db: TestClient):
    """Verify list_nodes returns all registered cluster nodes."""
    client_with_test_db.post(
        "/api/v1/nodes/register",
        json={"id": "node-a", "hostname": "host-a", "ip_address": "10.0.0.1"},
    )
    client_with_test_db.post(
        "/api/v1/nodes/register",
        json={"id": "node-b", "hostname": "host-b", "ip_address": "10.0.0.2"},
    )

    response = client_with_test_db.get("/api/v1/nodes")
    assert response.status_code == 200
    node_ids = [n["id"] for n in response.json()]
    assert "node-a" in node_ids
    assert "node-b" in node_ids


def test_create_api_key(client_with_test_db: TestClient):
    """Verify creating an API key returns 201 with the plain secret key and safe prefix."""
    payload = {"name": "analytics-service", "expires_in_days": 7}
    response = client_with_test_db.post("/api/v1/api-keys", json=payload)
    assert response.status_code == 201

    data = response.json()
    assert data["name"] == "analytics-service"
    assert data["api_key"].startswith("sovra_sk_live_")
    assert data["prefix"] == data["api_key"][:16]
    assert data["expires_at"] is not None


def test_list_api_keys_safe_metadata(client_with_test_db: TestClient):
    """Verify listing keys returns safe metadata and never leaks plain secret or raw hash."""
    client_with_test_db.post("/api/v1/api-keys", json={"name": "finance-bot"})

    response = client_with_test_db.get("/api/v1/api-keys")
    assert response.status_code == 200

    keys = response.json()
    assert len(keys) == 1
    key_entry = keys[0]

    assert key_entry["name"] == "finance-bot"
    assert key_entry["is_active"] is True
    assert "prefix" in key_entry

    # Security assertion: Neither the plaintext secret nor the raw SHA-256 hash must EVER be returned
    assert "api_key" not in key_entry
    assert "key_hash" not in key_entry


def test_revoke_api_key(client_with_test_db: TestClient):
    """Verify revoking an API key sets is_active=False immediately."""
    create_res = client_with_test_db.post("/api/v1/api-keys", json={"name": "to-be-revoked"})
    key_id = create_res.json()["id"]

    # Revoke
    del_res = client_with_test_db.delete(f"/api/v1/api-keys/{key_id}")
    assert del_res.status_code == 200
    assert del_res.json() == {"status": "revoked", "id": key_id}

    # Verify status in list
    list_res = client_with_test_db.get("/api/v1/api-keys")
    revoked_key = [k for k in list_res.json() if k["id"] == key_id][0]
    assert revoked_key["is_active"] is False


def test_revoke_api_key_not_found(client_with_test_db: TestClient):
    """Verify revoking a non-existent key returns 404."""
    del_res = client_with_test_db.delete("/api/v1/api-keys/fake-uuid-123")
    assert del_res.status_code == 404
    assert "not found" in del_res.json()["detail"]

