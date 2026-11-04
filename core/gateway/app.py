import httpx
from fastapi import FastAPI,HTTPException,status
from core.gateway.schema import ChatCompletionRequest
from core.models.specs import QWEN_0_5B_SPEC

app = FastAPI(title="Sovra AI Gateway",version="0.1.0")

# Active deployment registry mapping model_id -> backend engine URL
ACTIVE_MODELS: dict[str, str] = {
    QWEN_0_5B_SPEC.id: "http://localhost:8000",
}

@app.get("/health")
async def health():
    """Gateway health check"""
    return {"status": "healthy",  "service": "sovra-gateway"}


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible model list endpoint"""
    return{
        "object": "list",
        "data": [ {"id": model_id, "object": "model", "owned_by": "sovra"} for model_id in ACTIVE_MODELS ],
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Route chat completion requests to the active inference engine."""
    #Model validation
    if request.model not in ACTIVE_MODELS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{request.model}' is not registered or deployed in Sovra.",
        )
        
    backend_url = f"{ACTIVE_MODELS[request.model]}/v1/chat/completions"

    #  Proxy forward to inference engine
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
                detail=f"Inference engine for '{request.model}' is unreachable.",
            )