from datetime import datetime, timedelta, timezone
from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.control_plane.schemas import (
    ApiKeyCreateRequest,
    ApiKeyResponse,
    NodeHeartbeatRequest,
    NodeRegisterRequest,
)
from core.db.models import ApiKeyRecord, NodeRecord
from core.db.session import get_db
from core.security.keys import generate_api_key

app = FastAPI(title="Sovra Control Plane", version="0.1.0")


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


@app.post("/api/v1/api-keys", status_code=status.HTTP_201_CREATED)
def create_api_key(request: ApiKeyCreateRequest, db: Session = Depends(get_db)):
    """Generate a new cryptographic API key and store its SHA-256 hash."""
    plain_key, key_hash, prefix = generate_api_key()

    expires_at = None
    if request.expires_in_days:
        expires_at = datetime.now(timezone.utc) + timedelta(days=request.expires_in_days)

    record = ApiKeyRecord(
        name=request.name,
        key_hash=key_hash,
        prefix=prefix,
        is_active=True,
        expires_at=expires_at,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    # Return the plain secret key ONCE to the user
    return {
        "id": record.id,
        "name": record.name,
        "api_key": plain_key,
        "prefix": record.prefix,
        "created_at": record.created_at.isoformat(),
        "expires_at": record.expires_at.isoformat() if record.expires_at else None,
    }


@app.get("/api/v1/api-keys", response_model=list[ApiKeyResponse])
def list_api_keys(db: Session = Depends(get_db)):
    """List all API keys (without revealing secret hashes)."""
    keys = db.scalars(select(ApiKeyRecord)).all()
    return keys


@app.delete("/api/v1/api-keys/{key_id}")
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
