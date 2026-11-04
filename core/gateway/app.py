import httpx
from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.db.models import DeploymentRecord
from core.db.session import get_db
from core.gateway.schema import ChatCompletionRequest

app = FastAPI(title="Sovra AI Gateway", version="0.1.0")


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
    db: Session = Depends(get_db),
):
    """Route chat completions dynamically to the active engine port from the database."""
    #  Look up in the database: Is this model currently running?
    stmt = select(DeploymentRecord).where(
        DeploymentRecord.model_id == request.model,
        DeploymentRecord.status == "running",
    )
    deployment = db.scalar(stmt)

    if not deployment:
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
            return response.json()
        except httpx.ConnectError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Inference engine for '{request.model}' on port {deployment.port} is unreachable.",
            )