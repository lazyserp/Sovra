from unittest.mock import AsyncMock, patch
import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.db.models import ApiKeyRecord, Base, DeploymentRecord, ModelRecord, NodeRecord
from core.db.session import get_db, init_db
from core.gateway.app import app
from core.security.keys import hash_key

TEST_API_KEY = "sovra_sk_live_valid_test_key_12345"
AUTH_HEADERS = {"Authorization": f"Bearer {TEST_API_KEY}"}


@pytest.fixture
def client_with_gateway_db():
    """Create a FastAPI TestClient wired to an isolated in-memory DB with an active deployment and API key."""
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(target_engine=test_engine)
    test_session_factory = sessionmaker(test_engine, expire_on_commit=False)

    # Seed the database with a node, model, deployment, and valid API key
    with test_session_factory() as session:
        node = NodeRecord(
            id="node-test",
            hostname="test-host",
            ip_address="127.0.0.1",
            status="online",
        )
        model = ModelRecord(
            id="qwen-2.5-0.5b-instruct",
            name="Qwen 2.5 0.5B Instruct",
            artifact_path="./storage/models/qwen2.5-0.5b-instruct",
            min_vram_mb=1500,
        )
        deployment = DeploymentRecord(
            id="deploy-active-1",
            model_id="qwen-2.5-0.5b-instruct",
            node_id="node-test",
            port=8000,
            status="running",
        )
        api_key = ApiKeyRecord(
            name="test-key",
            key_hash=hash_key(TEST_API_KEY),
            prefix=TEST_API_KEY[:16],
            is_active=True,
        )
        session.add_all([node, model, deployment, api_key])
        session.commit()

    def override_get_db():
        with test_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=test_engine)


def test_gateway_health(client_with_gateway_db: TestClient):
    """Verify unauthenticated health check endpoint returns 200."""
    response = client_with_gateway_db.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "sovra-gateway"}


def test_gateway_list_models(client_with_gateway_db: TestClient):
    """Verify model listing dynamically queries active deployments."""
    response = client_with_gateway_db.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    model_ids = [m["id"] for m in data["data"]]
    assert "qwen-2.5-0.5b-instruct" in model_ids


def test_chat_completions_missing_auth(client_with_gateway_db: TestClient):
    """Verify request with no Authorization header returns 401."""
    payload = {
        "model": "qwen-2.5-0.5b-instruct",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = client_with_gateway_db.post("/v1/chat/completions", json=payload)
    assert response.status_code == 401
    assert "Missing or malformed Authorization header" in response.json()["detail"]


def test_chat_completions_invalid_auth(client_with_gateway_db: TestClient):
    """Verify request with invalid/fake API key returns 401."""
    payload = {
        "model": "qwen-2.5-0.5b-instruct",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    bad_headers = {"Authorization": "Bearer sovra_sk_live_completely_fake_token"}
    response = client_with_gateway_db.post(
        "/v1/chat/completions",
        json=payload,
        headers=bad_headers,
    )
    assert response.status_code == 401
    assert "Invalid, revoked, or expired API key" in response.json()["detail"]


def test_chat_completions_model_not_found(client_with_gateway_db: TestClient):
    """Verify authenticated request for un-deployed model returns 404."""
    payload = {
        "model": "non-existent-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = client_with_gateway_db.post(
        "/v1/chat/completions",
        json=payload,
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 404
    assert "not currently running" in response.json()["detail"]


def test_chat_completions_backend_unreachable(client_with_gateway_db: TestClient):
    """Verify authenticated request returns 503 error when inference engine port is offline."""
    payload = {
        "model": "qwen-2.5-0.5b-instruct",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = client_with_gateway_db.post(
        "/v1/chat/completions",
        json=payload,
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 503
    assert "unreachable" in response.json()["detail"]


def test_chat_completions_forward_success(client_with_gateway_db: TestClient):
    """Verify successful proxying and response handling when inference engine responds."""
    mock_engine_response = {
        "id": "chatcmpl-test-123",
        "object": "chat.completion",
        "model": "qwen-2.5-0.5b-instruct",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello from behind dynamic Sovra Gateway!",
                },
                "finish_reason": "stop",
            }
        ],
    }

    mock_response = httpx.Response(
        status_code=200,
        json=mock_engine_response,
        request=httpx.Request("POST", "http://localhost:8000/v1/chat/completions"),
    )

    payload = {
        "model": "qwen-2.5-0.5b-instruct",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.5,
    }

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        response = client_with_gateway_db.post(
            "/v1/chat/completions",
            json=payload,
            headers=AUTH_HEADERS,
        )
        assert response.status_code == 200
        assert response.json() == mock_engine_response
        mock_post.assert_called_once()
