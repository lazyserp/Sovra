from unittest.mock import AsyncMock, patch
import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.db.models import (
    ApiKeyRecord,
    AuditLogRecord,
    Base,
    DeploymentRecord,
    ModelRecord,
    NodeRecord,
    ProjectRecord,
)
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
        # Expose session_factory to test cases for database state verification
        client.session_factory = test_session_factory
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


def test_chat_completions_model_not_found_logs_audit(client_with_gateway_db: TestClient):
    """Verify authenticated request for un-deployed model returns 404 and writes an audit log."""
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

    # Verify audit log recorded the 404 failure
    with client_with_gateway_db.session_factory() as session:
        log = session.scalar(
            select(AuditLogRecord).where(
                AuditLogRecord.model_id == "non-existent-model",
                AuditLogRecord.status_code == 404,
            )
        )
        assert log is not None
        assert "not currently running" in log.error_message


def test_chat_completions_backend_unreachable_logs_audit(client_with_gateway_db: TestClient):
    """Verify authenticated request returns 503 and writes an audit log when inference engine is offline."""
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

    # Verify audit log recorded the 503 failure
    with client_with_gateway_db.session_factory() as session:
        log = session.scalar(
            select(AuditLogRecord).where(
                AuditLogRecord.model_id == "qwen-2.5-0.5b-instruct",
                AuditLogRecord.status_code == 503,
            )
        )
        assert log is not None
        assert "unreachable" in log.error_message


def test_chat_completions_forward_success_records_telemetry(client_with_gateway_db: TestClient):
    """Verify successful proxying returns 200 and records full token telemetry and latency in audit logs."""
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
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 25,
            "total_tokens": 40,
        },
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

        # Verify audit log recorded the successful transaction with token metrics
        with client_with_gateway_db.session_factory() as session:
            log = session.scalar(
                select(AuditLogRecord).where(
                    AuditLogRecord.model_id == "qwen-2.5-0.5b-instruct",
                    AuditLogRecord.status_code == 200,
                )
            )
            assert log is not None
            assert log.prompt_tokens == 15
            assert log.completion_tokens == 25
            assert log.total_tokens == 40
            assert log.latency_ms > 0
            assert log.error_message is None


def test_chat_completions_rate_limit_exceeded_returns_429(client_with_gateway_db: TestClient):
    """Verify that exceeding burst capacity triggers 429 Too Many Requests with Retry-After header."""
    from core.gateway.app import rate_limiter

    # Use a small burst limiter for this test
    rate_limiter.capacity = 2.0
    rate_limiter.rate = 0.1  # Very slow refill (1 token per 10s)
    rate_limiter._buckets.clear()

    payload = {
        "model": "qwen-2.5-0.5b-instruct",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # First 2 requests consume the burst capacity
    res1 = client_with_gateway_db.post("/v1/chat/completions", json=payload, headers=AUTH_HEADERS)
    res2 = client_with_gateway_db.post("/v1/chat/completions", json=payload, headers=AUTH_HEADERS)
    assert res1.status_code in [200, 503]
    assert res2.status_code in [200, 503]

    # 3rd request MUST be blocked by rate limiter
    res3 = client_with_gateway_db.post("/v1/chat/completions", json=payload, headers=AUTH_HEADERS)
    assert res3.status_code == 429
    assert "Rate limit exceeded" in res3.json()["detail"]
    assert "Retry-After" in res3.headers

    # Verify audit log recorded the 429 rate limit violation
    with client_with_gateway_db.session_factory() as session:
        log = session.scalar(
            select(AuditLogRecord).where(
                AuditLogRecord.status_code == 429,
            )
        )
        assert log is not None
        assert "Rate limit exceeded" in log.error_message

    # Reset limiter settings back to defaults
    rate_limiter.capacity = 10.0
    rate_limiter.rate = 1.0
    rate_limiter._buckets.clear()


def test_chat_completions_audit_log_stamps_project_id(client_with_gateway_db: TestClient):
    """Verify inference audit logs record the caller's project_id for tenant telemetry & billing."""
    project_key_plain = "sovra_sk_live_project_scoped_secret_9999"
    project_headers = {"Authorization": f"Bearer {project_key_plain}"}

    with client_with_gateway_db.session_factory() as session:
        project = ProjectRecord(id="proj-finance-101", name="Finance AI Team")
        key_record = ApiKeyRecord(
            name="finance-key",
            key_hash=hash_key(project_key_plain),
            prefix=project_key_plain[:16],
            project_id="proj-finance-101",
            role="inference",
            is_active=True,
        )
        session.add_all([project, key_record])
        session.commit()

    payload = {
        "model": "qwen-2.5-0.5b-instruct",
        "messages": [{"role": "user", "content": "Calculate Q3 risk metrics."}],
    }

    mock_engine_response = {
        "id": "chatcmpl-test-project",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "qwen-2.5-0.5b-instruct",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Done."}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    }

    mock_response = httpx.Response(
        status_code=200,
        json=mock_engine_response,
        request=httpx.Request("POST", "http://localhost:8000/v1/chat/completions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        res = client_with_gateway_db.post("/v1/chat/completions", json=payload, headers=project_headers)
        assert res.status_code == 200

    # Verify audit log recorded the exact project_id
    with client_with_gateway_db.session_factory() as session:
        log = session.scalar(
            select(AuditLogRecord).where(
                AuditLogRecord.project_id == "proj-finance-101",
            )
        )
        assert log is not None
        assert log.api_key_id == key_record.id
        assert log.model_id == "qwen-2.5-0.5b-instruct"
        assert log.prompt_tokens == 12
        assert log.completion_tokens == 8
        assert log.total_tokens == 20


