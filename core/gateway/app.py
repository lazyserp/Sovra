import os
import httpx
import time
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.db.session import get_db, init_db
from core.gateway.schema import AnthropicMessageRequest, ChatCompletionRequest
from core.db.models import ApiKeyRecord, AuditLogRecord, DeploymentRecord, NodeRecord
from core.security.keys import verify_api_key
from core.gateway.limiter import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(requests_per_minute=60, burst=10)
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

app = FastAPI(title="Sovra AI Gateway", version="0.1.0")
init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def resolve_target_host(host: str) -> str:
    """Translate localhost/127.0.0.1 to host.docker.internal when executing inside Docker."""
    if host in ["127.0.0.1", "localhost"]:
        if os.path.exists("/.dockerenv") or os.environ.get("RUNNING_IN_DOCKER") == "1":
            return "host.docker.internal"
    return host


def get_current_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    x_api_key: str | None = Depends(api_key_header),
    db: Session = Depends(get_db),
) -> ApiKeyRecord:
    """Enforce API key authentication via Bearer token or x-api-key header."""
    raw_key = None
    if credentials and credentials.credentials:
        raw_key = credentials.credentials
    elif x_api_key:
        raw_key = x_api_key
    elif "x-api-key" in request.headers:
        raw_key = request.headers["x-api-key"]
    elif "authorization" in request.headers:
        auth = request.headers["authorization"]
        if auth.lower().startswith("bearer "):
            raw_key = auth[7:].strip()

    if not raw_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header. Expected 'Bearer <api_key>' or 'x-api-key: <api_key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    api_key_record = verify_api_key(raw_key, db)
    if not api_key_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid, revoked, or expired API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return api_key_record


@app.get("/health")
async def health():
    """Gateway health check."""
    return {"status": "healthy", "service": "sovra-gateway"}


@app.get("/v1/models")
def list_models(db: Session = Depends(get_db)):
    """List all models currently running in the database."""
    stmt = select(DeploymentRecord.model_id).where(DeploymentRecord.status == "running")
    active_models = db.scalars(stmt).all()
    return {
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "owned_by": "sovra"}
            for model_id in active_models
        ],
    }


# ==========================================
# 1. OpenAI-Compatible Chat Completions API
# ==========================================
@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    api_key: ApiKeyRecord = Depends(get_current_api_key),
    db: Session = Depends(get_db),
):
    """Route chat completions dynamically to the active engine port from the database."""
    start_time = time.perf_counter()

    stmt = select(DeploymentRecord).where(
        DeploymentRecord.model_id == request.model,
        DeploymentRecord.status == "running",
    )
    deployment = db.scalar(stmt)

    allowed, retry_after = rate_limiter.check(api_key.id)
    if not allowed:
        audit_log = AuditLogRecord(
            api_key_id=api_key.id,
            model_id=request.model,
            project_id=api_key.project_id,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_message=f"Rate limit exceeded. Retry after {retry_after}s.",
        )
        db.add(audit_log)
        db.commit()

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(int(retry_after) or 1)},
        )

    if not deployment:
        audit_log = AuditLogRecord(
            api_key_id=api_key.id,
            model_id=request.model,
            status_code=status.HTTP_404_NOT_FOUND,
            project_id=api_key.project_id,
            error_message=f"Model {request.model} is not currently running",
        )
        db.add(audit_log)
        db.commit()

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{request.model}' is not currently running in Sovra.",
        )

    node = db.get(NodeRecord, deployment.node_id)
    raw_host = node.ip_address if (node and node.ip_address) else "127.0.0.1"
    host = resolve_target_host(raw_host)
    backend_url = f"http://{host}:{deployment.port}/v1/chat/completions"
    payload = request.model_dump(exclude_none=True)

    # 1. Real-time SSE Streaming Mode
    if request.stream:
        client = httpx.AsyncClient(timeout=120.0)
        try:
            req = client.build_request(
                "POST",
                backend_url,
                json=payload,
                headers={"Accept": "text/event-stream"},
            )
            response = await client.send(req, stream=True)
            if response.status_code >= 400:
                status_c = response.status_code
                await response.aclose()
                await client.aclose()
                raise HTTPException(status_code=status_c, detail=f"Inference engine returned error status {status_c}")

            async def stream_generator():
                try:
                    async for chunk in response.aiter_raw():
                        yield chunk
                finally:
                    await response.aclose()
                    await client.aclose()
                    latency_ms = (time.perf_counter() - start_time) * 1000.0
                    audit_log = AuditLogRecord(
                        api_key_id=api_key.id,
                        model_id=request.model,
                        project_id=api_key.project_id,
                        latency_ms=round(latency_ms, 2),
                        status_code=200,
                    )
                    db.add(audit_log)
                    db.commit()

            return StreamingResponse(
                stream_generator(),
                status_code=response.status_code,
                media_type="text/event-stream",
            )
        except httpx.ConnectError:
            await client.aclose()
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            audit_log = AuditLogRecord(
                api_key_id=api_key.id,
                model_id=request.model,
                latency_ms=round(latency_ms, 2),
                project_id=api_key.project_id,
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                error_message=f"Inference engine for '{request.model}' on port {deployment.port} is unreachable.",
            )
            db.add(audit_log)
            db.commit()

            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Inference engine for '{request.model}' on port {deployment.port} is unreachable.",
            )

    # 2. Standard Buffered JSON Mode
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                backend_url,
                json=payload,
            )

            data = response.json()
            usage = data.get("usage", {})
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            audit_log = AuditLogRecord(
                api_key_id=api_key.id,
                model_id=request.model,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                latency_ms=round(latency_ms, 2),
                project_id=api_key.project_id,
                status_code=response.status_code,
            )
            db.add(audit_log)
            db.commit()
            return data

        except httpx.ConnectError:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            audit_log = AuditLogRecord(
                api_key_id=api_key.id,
                model_id=request.model,
                latency_ms=round(latency_ms, 2),
                project_id=api_key.project_id,
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                error_message=f"Inference engine for '{request.model}' on port {deployment.port} is unreachable.",
            )
            db.add(audit_log)
            db.commit()

            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Inference engine for '{request.model}' on port {deployment.port} is unreachable.",
            )


# ==========================================
# 2. Anthropic Messages API (Claude Code)
# ==========================================
@app.post("/v1/messages")
async def messages(
    request: AnthropicMessageRequest,
    api_key: ApiKeyRecord = Depends(get_current_api_key),
    db: Session = Depends(get_db),
):
    """Route Anthropic Messages API requests dynamically to the active engine port."""
    start_time = time.perf_counter()

    stmt = select(DeploymentRecord).where(
        DeploymentRecord.model_id == request.model,
        DeploymentRecord.status == "running",
    )
    deployment = db.scalar(stmt)

    allowed, retry_after = rate_limiter.check(api_key.id)
    if not allowed:
        audit_log = AuditLogRecord(
            api_key_id=api_key.id,
            model_id=request.model,
            project_id=api_key.project_id,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_message=f"Rate limit exceeded. Retry after {retry_after}s.",
        )
        db.add(audit_log)
        db.commit()

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(int(retry_after) or 1)},
        )

    if not deployment:
        audit_log = AuditLogRecord(
            api_key_id=api_key.id,
            model_id=request.model,
            status_code=status.HTTP_404_NOT_FOUND,
            project_id=api_key.project_id,
            error_message=f"Model {request.model} is not currently running",
        )
        db.add(audit_log)
        db.commit()

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{request.model}' is not currently running in Sovra.",
        )

    node = db.get(NodeRecord, deployment.node_id)
    raw_host = node.ip_address if (node and node.ip_address) else "127.0.0.1"
    host = resolve_target_host(raw_host)
    backend_url = f"http://{host}:{deployment.port}/v1/messages"
    payload = request.model_dump(exclude_none=True)

    # 1. Real-time SSE Streaming Mode
    if request.stream:
        client = httpx.AsyncClient(timeout=120.0)
        try:
            req = client.build_request(
                "POST",
                backend_url,
                json=payload,
                headers={"Accept": "text/event-stream"},
            )
            response = await client.send(req, stream=True)
            if response.status_code >= 400:
                status_c = response.status_code
                await response.aclose()
                await client.aclose()
                raise HTTPException(status_code=status_c, detail=f"Inference engine returned error status {status_c}")

            async def stream_generator():
                try:
                    async for chunk in response.aiter_raw():
                        yield chunk
                finally:
                    await response.aclose()
                    await client.aclose()
                    latency_ms = (time.perf_counter() - start_time) * 1000.0
                    audit_log = AuditLogRecord(
                        api_key_id=api_key.id,
                        model_id=request.model,
                        project_id=api_key.project_id,
                        latency_ms=round(latency_ms, 2),
                        status_code=200,
                    )
                    db.add(audit_log)
                    db.commit()

            return StreamingResponse(
                stream_generator(),
                status_code=response.status_code,
                media_type="text/event-stream",
            )
        except httpx.ConnectError:
            await client.aclose()
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            audit_log = AuditLogRecord(
                api_key_id=api_key.id,
                model_id=request.model,
                latency_ms=round(latency_ms, 2),
                project_id=api_key.project_id,
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                error_message=f"Inference engine for '{request.model}' on port {deployment.port} is unreachable.",
            )
            db.add(audit_log)
            db.commit()

            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Inference engine for '{request.model}' on port {deployment.port} is unreachable.",
            )

    # 2. Standard Buffered JSON Mode
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                backend_url,
                json=payload,
            )

            data = response.json()
            usage = data.get("usage", {})
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            prompt_toks = usage.get("input_tokens") or usage.get("prompt_tokens", 0)
            comp_toks = usage.get("output_tokens") or usage.get("completion_tokens", 0)
            total_toks = prompt_toks + comp_toks

            audit_log = AuditLogRecord(
                api_key_id=api_key.id,
                model_id=request.model,
                prompt_tokens=prompt_toks,
                completion_tokens=comp_toks,
                total_tokens=total_toks,
                latency_ms=round(latency_ms, 2),
                project_id=api_key.project_id,
                status_code=response.status_code,
            )
            db.add(audit_log)
            db.commit()
            return data

        except httpx.ConnectError:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            audit_log = AuditLogRecord(
                api_key_id=api_key.id,
                model_id=request.model,
                latency_ms=round(latency_ms, 2),
                project_id=api_key.project_id,
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                error_message=f"Inference engine for '{request.model}' on port {deployment.port} is unreachable.",
            )
            db.add(audit_log)
            db.commit()

            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Inference engine for '{request.model}' on port {deployment.port} is unreachable.",
            )
