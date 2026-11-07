import os
import secrets
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.control_plane.schemas import (
    ApiKeyCreateRequest,
    ApiKeyResponse,
    DeploymentCreateRequest,
    DeploymentResponse,
    DeploymentVerifyRequest,
    NodeHeartbeatRequest,
    NodeRegisterRequest,
    ProjectCreateRequest,
    ProjectResponse,
)
from core.db.models import (
    ApiKeyRecord,
    AuditLogRecord,
    DeploymentRecord,
    ModelRecord,
    NodeRecord,
    ProjectRecord,
)
from core.db.session import get_db, init_db
from core.security.keys import generate_api_key

app = FastAPI(title="Sovra Control Plane", version="0.1.0")
init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

SOVRA_ADMIN_TOKEN = os.environ.get("SOVRA_ADMIN_TOKEN") or "devkey"
admin_bearer_scheme = HTTPBearer(auto_error=False)

def verify_admin_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(admin_bearer_scheme),
):
    """Enforce constant-time master token authentication on admin routes."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header. Expected 'Bearer <admin_token>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not secrets.compare_digest(credentials.credentials, SOVRA_ADMIN_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Invalid admin token.",
        )
    return True


def resolve_target_host(host: str) -> str:
    """Translate localhost/127.0.0.1 to host.docker.internal when executing inside Docker."""
    if host in ["127.0.0.1", "localhost"]:
        if os.path.exists("/.dockerenv") or os.environ.get("RUNNING_IN_DOCKER") == "1":
            return "host.docker.internal"
    return host

@app.post("/api/v1/nodes/register")
def register_node(request: NodeRegisterRequest, db: Session = Depends(get_db)):
    """Idempotently register or update a GPU node."""
    node = db.get(NodeRecord, request.id)

    if not node:
        # insert
        node = NodeRecord(
            id=request.id,
            hostname=request.hostname,
            ip_address=request.ip_address,
            gpu_name=request.gpu_name,
            total_vram_mb=request.total_vram_mb,
            free_vram_mb=request.free_vram_mb,
            status="online",
        )
        db.add(node)
    else:
        # update
        node.hostname = request.hostname
        node.ip_address = request.ip_address
        node.gpu_name = request.gpu_name
        node.total_vram_mb = request.total_vram_mb
        node.free_vram_mb = request.free_vram_mb
        node.status = "online"
        node.last_heartbeat = datetime.now(timezone.utc)

    db.commit()
    db.refresh(node)
    return {"status": "registered", "node_id": node.id}


@app.post("/api/v1/nodes/{node_id}/heartbeat")
def node_heartbeat(node_id: str, request: NodeHeartbeatRequest, db: Session = Depends(get_db)):
    """Receive periodic health and VRAM telemetry from a node."""
    node = db.get(NodeRecord, node_id)
    if not node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Node {node_id} is not registered",
        )

    node.free_vram_mb = request.free_vram_mb
    node.status = request.status
    node.last_heartbeat = datetime.now(timezone.utc)

    db.commit()
    return {"status": "acknowledged", "node_id": node.id}


@app.get("/api/v1/nodes")
def list_nodes(db: Session = Depends(get_db)):
    """List all registered nodes and their current status."""
    nodes = db.scalars(select(NodeRecord)).all()
    return [
        {
            "id": n.id,
            "hostname": n.hostname,
            "ip_address": n.ip_address,
            "status": n.status,
            "gpu_name": n.gpu_name,
            "total_vram_mb": n.total_vram_mb,
            "free_vram_mb": n.free_vram_mb,
            "last_heartbeat": n.last_heartbeat.isoformat(),
        }
        for n in nodes
    ]


@app.post("/api/v1/api-keys", status_code=status.HTTP_201_CREATED, dependencies=[Depends(verify_admin_token)])
def create_api_key(request: ApiKeyCreateRequest, db: Session = Depends(get_db)):
    """Generate a new cryptographic API key and store its SHA-256 hash."""
    plain_key, key_hash, prefix = generate_api_key()

    expires_at = None
    if request.expires_in_days:
        expires_at = datetime.now(timezone.utc) + timedelta(days=request.expires_in_days)

    if not request.role in ["admin","operator","inference"]:
        raise HTTPException(status_code=400,detail=f"Invalid role '{request.role}'. Allowed roles: admin, operator, inference.")
    
    if request.project_id:
        db_project = db.get(ProjectRecord, request.project_id)
        if not db_project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project '{request.project_id}' not found.",
            )


    record = ApiKeyRecord(
        name=request.name,
        key_hash=key_hash,
        prefix=prefix,
        is_active=True,
        expires_at=expires_at,
        project_id=request.project_id,
        role=request.role,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    # Return the plain secret key ONCE to the user
    return {
        "id": record.id,
        "name": record.name,
        "api_key": plain_key,
        "project_id": record.project_id,
        "role": record.role,
        "prefix": record.prefix,
        "created_at": record.created_at.isoformat(),
        "expires_at": record.expires_at.isoformat() if record.expires_at else None,
    }


@app.get("/api/v1/api-keys", response_model=list[ApiKeyResponse], dependencies=[Depends(verify_admin_token)])
def list_api_keys(db: Session = Depends(get_db)):
    """List all API keys (without revealing secret hashes)."""
    keys = db.scalars(select(ApiKeyRecord)).all()
    return keys


@app.delete("/api/v1/api-keys/{key_id}", dependencies=[Depends(verify_admin_token)])
def revoke_api_key(key_id: str, db: Session = Depends(get_db)):
    """Revoke an API key immediately by setting is_active=False."""
    record = db.get(ApiKeyRecord, key_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key '{key_id}' not found.",
        )

    record.is_active = False
    db.commit()
    return {"status": "revoked", "id": record.id}

@app.post("/api/v1/projects", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED,dependencies=[Depends(verify_admin_token)])
def create_project(request: ProjectCreateRequest, db: Session = Depends(get_db)):
    project = ProjectRecord(name=request.name,description=request.description)
    db.add(project)
    db.commit()
    db.refresh(project)
    
    return project


@app.get("/api/v1/projects", status_code=status.HTTP_200_OK, dependencies=[Depends(verify_admin_token)], response_model=list[ProjectResponse])
def get_all_projects(db: Session = Depends(get_db)):
    return db.scalars(select(ProjectRecord)).all()


@app.get("/api/v1/audit-logs", dependencies=[Depends(verify_admin_token)])
def list_audit_logs(limit: int = 100, db: Session = Depends(get_db)):
    """Fetch recent inference trace & audit events for Langfuse-style observability."""
    stmt = select(AuditLogRecord).order_by(AuditLogRecord.created_at.desc()).limit(limit)
    records = db.scalars(stmt).all()
    return [
        {
            "id": r.id,
            "api_key_id": r.api_key_id,
            "project_id": r.project_id,
            "model_id": r.model_id,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "total_tokens": r.total_tokens,
            "latency_ms": r.latency_ms,
            "status_code": r.status_code,
            "error_message": r.error_message,
            "created_at": r.created_at.isoformat(),
        }
        for r in records
    ]


@app.get("/api/v1/metrics/overview", dependencies=[Depends(verify_admin_token)])
def get_metrics_overview(db: Session = Depends(get_db)):
    """Aggregate cluster metrics for dashboard analytics."""
    total_requests = db.scalar(select(func.count(AuditLogRecord.id))) or 0
    total_tokens = db.scalar(select(func.sum(AuditLogRecord.total_tokens))) or 0
    avg_latency = db.scalar(select(func.avg(AuditLogRecord.latency_ms))) or 0.0
    active_nodes = db.scalar(select(func.count(NodeRecord.id)).where(NodeRecord.status == "online")) or 0
    total_projects = db.scalar(select(func.count(ProjectRecord.id))) or 0
    total_keys = db.scalar(select(func.count(ApiKeyRecord.id)).where(ApiKeyRecord.is_active == True)) or 0
    error_count = db.scalar(select(func.count(AuditLogRecord.id)).where(AuditLogRecord.status_code >= 400)) or 0

    return {
        "total_requests": total_requests,
        "total_tokens": int(total_tokens),
        "avg_latency_ms": round(float(avg_latency), 2),
        "error_count": error_count,
        "active_nodes": active_nodes,
        "total_projects": total_projects,
        "total_keys": total_keys,
        "data_egress_bytes": 0,
        "air_gapped": True,
    }


# ==========================================
# Deployments & Model Management Routes
# ==========================================
@app.post("/api/v1/deployments/verify", dependencies=[Depends(verify_admin_token)])
async def verify_deployment_endpoint(request: DeploymentVerifyRequest):
    """Actively probe a remote (e.g. vLLM on Linux) or local inference engine to verify health."""
    import httpx
    import time
    target_host = resolve_target_host(request.host)
    target_base = f"http://{target_host}:{request.port}"
    probe_urls = [
        f"{target_base}/v1/models",
        f"{target_base}/health",
        f"{target_base}/api/tags",
        f"{target_base}/",
    ]

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=4.0) as client:
        for url in probe_urls:
            try:
                res = await client.get(url)
                if res.status_code in [200, 404]:
                    latency_ms = round((time.perf_counter() - start) * 1000, 2)
                    models_found = []
                    if res.status_code == 200:
                        try:
                            data = res.json()
                            if "data" in data and isinstance(data["data"], list):
                                models_found = [m.get("id") for m in data["data"] if "id" in m]
                            elif "models" in data and isinstance(data["models"], list):
                                models_found = [m.get("name") for m in data["models"] if "name" in m]
                        except Exception:
                            pass
                    return {
                        "status": "healthy",
                        "endpoint": target_base,
                        "latency_ms": latency_ms,
                        "probe_url": url,
                        "models_found": models_found,
                    }
            except Exception:
                continue

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=f"Connection failed: Could not reach {request.engine_type.upper()} at {target_base}. Check network access, IP address, and firewall rules.",
    )


@app.post("/api/v1/deployments", response_model=DeploymentResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(verify_admin_token)])
def create_deployment(request: DeploymentCreateRequest, db: Session = Depends(get_db)):
    """Deploy or register an AI model bound to an active GPU node (local or remote Linux)."""
    # 1. Resolve target Node
    node = None
    if request.node_id:
        node = db.get(NodeRecord, request.node_id)
        if not node:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Node '{request.node_id}' not found.",
            )
    else:
        # Match node by host/IP or auto-create a node record for this remote/local machine
        host = (request.host or "127.0.0.1").strip()
        node = db.scalar(select(NodeRecord).where(NodeRecord.ip_address == host))
        if not node:
            is_local = host in ["127.0.0.1", "localhost"]
            node_id = "node-local-pc" if is_local else f"node-{host.replace('.', '-')}"
            node = NodeRecord(
                id=node_id,
                hostname=f"gpu-node-{host}",
                ip_address=host,
                gpu_name="NVIDIA GPU Cluster Node" if not is_local else "NVIDIA Local GPU",
                total_vram_mb=24576 if not is_local else 4096,
                free_vram_mb=22000 if not is_local else 3500,
                status="online",
            )
            db.add(node)
            db.commit()
            db.refresh(node)

    # 2. Register or update Model Record
    model = db.get(ModelRecord, request.model_id)
    if not model:
        model = ModelRecord(
            id=request.model_id,
            name=request.name or request.model_id,
            artifact_path=f"{request.engine_type}://{request.model_id}",
            min_vram_mb=400,
            engine_type=request.engine_type,
        )
        db.add(model)
        db.commit()

    # 3. Create or update Deployment Record
    deploy_id = f"deploy-{request.model_id.replace(':', '-')}"
    deployment = db.get(DeploymentRecord, deploy_id)
    port = request.port or (11434 if request.engine_type == "ollama" else 8000)

    if not deployment:
        deployment = DeploymentRecord(
            id=deploy_id,
            model_id=request.model_id,
            node_id=node.id,
            port=port,
            status="running",
        )
        db.add(deployment)
    else:
        deployment.node_id = node.id
        deployment.port = port
        deployment.status = "running"

    db.commit()
    db.refresh(deployment)
    return deployment


@app.get("/api/v1/deployments", dependencies=[Depends(verify_admin_token)])
def list_deployments(db: Session = Depends(get_db)):
    """List all deployed models, ports, and statuses."""
    deployments = db.scalars(select(DeploymentRecord)).all()
    results = []
    for d in deployments:
        model = db.get(ModelRecord, d.model_id)
        node = db.get(NodeRecord, d.node_id)
        results.append({
            "id": d.id,
            "model_id": d.model_id,
            "model_name": model.name if model else d.model_id,
            "engine_type": model.engine_type if model else "vllm",
            "node_id": d.node_id,
            "node_hostname": node.hostname if node else d.node_id,
            "port": d.port,
            "status": d.status,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        })
    return results


@app.delete("/api/v1/deployments/{deployment_id}", dependencies=[Depends(verify_admin_token)])
def delete_deployment(deployment_id: str, db: Session = Depends(get_db)):
    """Undeploy or remove a model deployment."""
    deployment = db.get(DeploymentRecord, deployment_id)
    if not deployment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deployment '{deployment_id}' not found.",
        )
    db.delete(deployment)
    db.commit()
    return {"status": "deleted", "id": deployment_id}


@app.get("/api/v1/discover/ollama", dependencies=[Depends(verify_admin_token)])
def discover_ollama_models(port: int = 11434, host: str = "127.0.0.1"):
    """Scan local or host Ollama instance for pulled models ready for 1-click deployment."""
    import httpx
    target_host = resolve_target_host(host)
    try:
        r = httpx.get(f"http://{target_host}:{port}/api/tags", timeout=3.0)
        if r.status_code == 200:
            models = r.json().get("models", [])
            return [
                {
                    "name": m.get("name"),
                    "size_bytes": m.get("size", 0),
                    "modified_at": m.get("modified_at"),
                }
                for m in models
            ]
        return []
    except Exception:
        return []


CONSOLE_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "console", "index.html")
)


@app.get("/", response_class=HTMLResponse)
@app.get("/console", response_class=HTMLResponse)
def serve_console():
    """Serve the Sovra Enterprise Observability & Management Console UI."""
    if not os.path.exists(CONSOLE_FILE_PATH):
        return HTMLResponse(
            "<h2>Sovra Console is initializing... Please create core/console/index.html</h2>",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    with open(CONSOLE_FILE_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())



