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
