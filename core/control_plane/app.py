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
    NodeHeartbeatRequest,
    NodeRegisterRequest,
    ProjectCreateRequest,ProjectResponse
)
from core.db.models import ApiKeyRecord, AuditLogRecord, NodeRecord, ProjectRecord
from core.db.session import get_db,init_db
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



