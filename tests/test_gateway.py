from unittest.mock import AsyncMock, patch
import httpx
import pytest
from fastapi.testclient import TestClient

from core.gateway.app import app

client = TestClient(app)


def test_gateway_health():
    """Verify gateway health check endpoint returns 200."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "sovra-gateway"}


def test_gateway_list_models():
    """Verify OpenAI-compatible model listing returns active models."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    model_ids = [m["id"] for m in data["data"]]
    assert "qwen-2.5-0.5b-instruct" in model_ids


def test_chat_completions_model_not_found():
    """Verify request for unknown model returns 404."""
    payload = {
        "model": "non-existent-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 404
    assert "not registered" in response.json()["detail"]


def test_chat_completions_backend_unreachable():
    """Verify 503 error when inference engine backend is offline."""
    payload = {
        "model": "qwen-2.5-0.5b-instruct",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    # When no mock is present and port 8000 has no vLLM server:
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 503
    assert "unreachable" in response.json()["detail"]


def test_chat_completions_forward_success():
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
                    "content": "Hello from behind Sovra Gateway!",
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

        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert response.json() == mock_engine_response
        mock_post.assert_called_once()

