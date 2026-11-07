import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import Session, sessionmaker

from core.control_plane.app import SOVRA_ADMIN_TOKEN, app
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


ADMIN_HEADERS = {"Authorization": f"Bearer {SOVRA_ADMIN_TOKEN}"}


def test_admin_routes_missing_token_returns_401(client_with_test_db: TestClient):
    """Verify admin endpoints reject requests with missing Authorization header."""
    res_post = client_with_test_db.post("/api/v1/api-keys", json={"name": "test"})
    assert res_post.status_code == 401
    assert "Missing Authorization header" in res_post.json()["detail"]

    res_get = client_with_test_db.get("/api/v1/api-keys")
    assert res_get.status_code == 401

    res_del = client_with_test_db.delete("/api/v1/api-keys/some-id")
    assert res_del.status_code == 401


def test_admin_routes_invalid_token_returns_403(client_with_test_db: TestClient):
    """Verify admin endpoints reject requests with invalid/wrong admin token."""
    bad_headers = {"Authorization": "Bearer totally-wrong-admin-token"}
    response = client_with_test_db.get("/api/v1/api-keys", headers=bad_headers)
    assert response.status_code == 403
    assert "Forbidden" in response.json()["detail"]


def test_create_api_key(client_with_test_db: TestClient):
    """Verify creating an API key returns 201 with the plain secret key and safe prefix."""
    payload = {"name": "analytics-service", "expires_in_days": 7}
    response = client_with_test_db.post("/api/v1/api-keys", json=payload, headers=ADMIN_HEADERS)
    assert response.status_code == 201

    data = response.json()
    assert data["name"] == "analytics-service"
    assert data["api_key"].startswith("sovra_sk_live_")
    assert data["prefix"] == data["api_key"][:16]
    assert data["expires_at"] is not None


def test_list_api_keys_safe_metadata(client_with_test_db: TestClient):
    """Verify listing keys returns safe metadata and never leaks plain secret or raw hash."""
    client_with_test_db.post("/api/v1/api-keys", json={"name": "finance-bot"}, headers=ADMIN_HEADERS)

    response = client_with_test_db.get("/api/v1/api-keys", headers=ADMIN_HEADERS)
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
    create_res = client_with_test_db.post("/api/v1/api-keys", json={"name": "to-be-revoked"}, headers=ADMIN_HEADERS)
    key_id = create_res.json()["id"]

    # Revoke
    del_res = client_with_test_db.delete(f"/api/v1/api-keys/{key_id}", headers=ADMIN_HEADERS)
    assert del_res.status_code == 200
    assert del_res.json() == {"status": "revoked", "id": key_id}

    # Verify status in list
    list_res = client_with_test_db.get("/api/v1/api-keys", headers=ADMIN_HEADERS)
    revoked_key = [k for k in list_res.json() if k["id"] == key_id][0]
    assert revoked_key["is_active"] is False


def test_revoke_api_key_not_found(client_with_test_db: TestClient):
    """Verify revoking a non-existent key returns 404."""
    del_res = client_with_test_db.delete("/api/v1/api-keys/fake-uuid-123", headers=ADMIN_HEADERS)
    assert del_res.status_code == 404
    assert "not found" in del_res.json()["detail"]


def test_create_project_success(client_with_test_db: TestClient):
    """Verify admin can create an isolated project workspace."""
    payload = {"name": "Risk Analysis", "description": "Banking fraud & risk models"}
    res = client_with_test_db.post("/api/v1/projects", json=payload, headers=ADMIN_HEADERS)
    assert res.status_code == 201

    data = res.json()
    assert data["name"] == "Risk Analysis"
    assert data["description"] == "Banking fraud & risk models"
    assert "id" in data
    assert "created_at" in data


def test_create_project_unauthorized(client_with_test_db: TestClient):
    """Verify project creation requires admin authorization."""
    payload = {"name": "Unauthorized Project"}
    res = client_with_test_db.post("/api/v1/projects", json=payload)
    assert res.status_code == 401


def test_list_projects(client_with_test_db: TestClient):
    """Verify admin can list all registered projects."""
    client_with_test_db.post("/api/v1/projects", json={"name": "Project Alpha"}, headers=ADMIN_HEADERS)
    client_with_test_db.post("/api/v1/projects", json={"name": "Project Beta"}, headers=ADMIN_HEADERS)

    res = client_with_test_db.get("/api/v1/projects", headers=ADMIN_HEADERS)
    assert res.status_code == 200
    projects = res.json()
    names = [p["name"] for p in projects]
    assert "Project Alpha" in names
    assert "Project Beta" in names


def test_create_api_key_with_project_and_role(client_with_test_db: TestClient):
    """Verify creating an API key scoped to a project and assigned a specific role."""
    proj_res = client_with_test_db.post("/api/v1/projects", json={"name": "Healthcare AI"}, headers=ADMIN_HEADERS)
    project_id = proj_res.json()["id"]

    payload = {
        "name": "clinic-bot-key",
        "project_id": project_id,
        "role": "operator",
        "expires_in_days": 30,
    }
    key_res = client_with_test_db.post("/api/v1/api-keys", json=payload, headers=ADMIN_HEADERS)
    assert key_res.status_code == 201

    key_data = key_res.json()
    assert key_data["project_id"] == project_id
    assert key_data["role"] == "operator"

    # Verify listing also includes project_id and role
    list_res = client_with_test_db.get("/api/v1/api-keys", headers=ADMIN_HEADERS)
    match = [k for k in list_res.json() if k["id"] == key_data["id"]][0]
    assert match["project_id"] == project_id
    assert match["role"] == "operator"


def test_create_api_key_invalid_project_fails(client_with_test_db: TestClient):
    """Verify creating a key referencing a non-existent project returns 404."""
    payload = {
        "name": "orphan-key",
        "project_id": "non-existent-project-id",
        "role": "inference",
    }
    res = client_with_test_db.post("/api/v1/api-keys", json=payload, headers=ADMIN_HEADERS)
    assert res.status_code == 404
    assert "not found" in res.json()["detail"]


def test_create_api_key_invalid_role_fails(client_with_test_db: TestClient):
    """Verify creating a key with an unknown role returns 400."""
    payload = {
        "name": "bad-role-key",
        "role": "super-admin-godmode",
    }
    res = client_with_test_db.post("/api/v1/api-keys", json=payload, headers=ADMIN_HEADERS)
    assert res.status_code == 400
    assert "Invalid role" in res.json()["detail"]


def test_serve_console_ui(client_with_test_db: TestClient):
    """Verify that the Control Plane serves the Sovra Console single-page UI."""
    res = client_with_test_db.get("/console")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "Sovra" in res.text


def test_metrics_overview_endpoint(client_with_test_db: TestClient):
    """Verify the cluster metrics overview returns accurate aggregate counts."""
    res = client_with_test_db.get("/api/v1/metrics/overview", headers=ADMIN_HEADERS)
    assert res.status_code == 200
    data = res.json()
    assert "total_requests" in data
    assert "total_tokens" in data
    assert "avg_latency_ms" in data
    assert data["air_gapped"] is True


def test_audit_logs_endpoint(client_with_test_db: TestClient):
    """Verify fetching audit logs returns list of trace events."""
    res = client_with_test_db.get("/api/v1/audit-logs", headers=ADMIN_HEADERS)
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_create_and_list_deployments(client_with_test_db: TestClient):
    """Verify deploying a model creates the model, node, and deployment records."""
    payload = {
        "model_id": "qwen2.5-coder:7b",
        "name": "Qwen 2.5 Coder 7B",
        "engine_type": "ollama",
        "port": 11434,
    }
    # 1. Create deployment
    res = client_with_test_db.post("/api/v1/deployments", json=payload, headers=ADMIN_HEADERS)
    assert res.status_code == 201
    data = res.json()
    assert data["model_id"] == "qwen2.5-coder:7b"
    assert data["status"] == "running"
    deploy_id = data["id"]

    # 2. List deployments
    res_list = client_with_test_db.get("/api/v1/deployments", headers=ADMIN_HEADERS)
    assert res_list.status_code == 200
    deployments = res_list.json()
    assert len(deployments) == 1
    assert deployments[0]["model_id"] == "qwen2.5-coder:7b"
    assert deployments[0]["engine_type"] == "ollama"

    # 3. Delete deployment
    res_del = client_with_test_db.delete(f"/api/v1/deployments/{deploy_id}", headers=ADMIN_HEADERS)
    assert res_del.status_code == 200
    assert res_del.json()["status"] == "deleted"

    # 4. List again should be empty
    res_empty = client_with_test_db.get("/api/v1/deployments", headers=ADMIN_HEADERS)
    assert len(res_empty.json()) == 0


def test_verify_deployment_endpoint_healthy(client_with_test_db: TestClient):
    """Verify live probing of a vLLM engine returns 200 and latency when engine is healthy."""
    from unittest.mock import AsyncMock, patch
    import httpx

    mock_resp = httpx.Response(
        200,
        json={"data": [{"id": "Qwen/Qwen2.5-Coder-32B-Instruct"}]},
        request=httpx.Request("GET", "http://10.240.0.15:8000/v1/models"),
    )

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp

        payload = {"host": "10.240.0.15", "port": 8000, "engine_type": "vllm"}
        res = client_with_test_db.post("/api/v1/deployments/verify", json=payload, headers=ADMIN_HEADERS)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert "Qwen/Qwen2.5-Coder-32B-Instruct" in data["models_found"]


def test_verify_deployment_endpoint_unreachable(client_with_test_db: TestClient):
    """Verify probing an unreachable remote engine returns 502 with diagnostic detail."""
    from unittest.mock import AsyncMock, patch
    import httpx

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = httpx.ConnectError("Connection refused")

        payload = {"host": "10.240.0.99", "port": 8000, "engine_type": "vllm"}
        res = client_with_test_db.post("/api/v1/deployments/verify", json=payload, headers=ADMIN_HEADERS)
        assert res.status_code == 502
        assert "Connection failed" in res.json()["detail"]


def test_create_remote_linux_deployment(client_with_test_db: TestClient):
    """Verify deploying to a remote Linux IP registers a remote NodeRecord and routes properly."""
    payload = {
        "model_id": "deepseek-ai/DeepSeek-V3",
        "name": "DeepSeek V3 (Customer Cluster)",
        "engine_type": "vllm",
        "host": "10.0.1.50",
        "port": 8000,
    }
    res = client_with_test_db.post("/api/v1/deployments", json=payload, headers=ADMIN_HEADERS)
    assert res.status_code == 201

    res_list = client_with_test_db.get("/api/v1/deployments", headers=ADMIN_HEADERS)
    assert res_list.status_code == 200
    deploys = res_list.json()
    assert any(d["model_id"] == "deepseek-ai/DeepSeek-V3" and d["port"] == 8000 for d in deploys)






