import httpx
import time
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.db.session import get_db
from core.gateway.schema import ChatCompletionRequest
from core.db.models import ApiKeyRecord, AuditLogRecord, DeploymentRecord
from core.security.keys import verify_api_key
from core.gateway.limiter import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(requests_per_minute=60,burst=10)
bearer_scheme = HTTPBearer(auto_error=False)

app = FastAPI(title="Sovra AI Gateway", version="0.1.0")


def get_current_api_key(credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme) , db: Session = Depends(get_db)) -> ApiKeyRecord:
    """Enforce API key authentication via Bearer token."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header. Expected 'Bearer <api_key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    api_key_record = verify_api_key(credentials.credentials,db)
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


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    api_key: ApiKeyRecord = Depends(get_current_api_key),
    db: Session = Depends(get_db),
):
    """Route chat completions dynamically to the active engine port from the database."""
    #  Look up in the database: Is this model currently running?
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
        audit_log= AuditLogRecord(
            api_key_id=api_key.id,
            model_id=request.model,
            status_code=status.HTTP_404_NOT_FOUND,
            error_message=f"Model {request.model} is not currently running"
        )
        db.add(audit_log)
        db.commit()

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{request.model}' is not currently running in Sovra.",
        )

    backend_url = f"http://localhost:{deployment.port}/v1/chat/completions"

    #  Proxy forward to the engine port
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                backend_url,
                json=request.model_dump(exclude_none=True),
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
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                error_message=f"Inference engine for '{request.model}' on port {deployment.port} is unreachable.",
            )
            db.add(audit_log)
            db.commit()

            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Inference engine for '{request.model}' on port {deployment.port} is unreachable.",
            )
