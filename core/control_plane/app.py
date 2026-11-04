from core.control_plane.schemas import NodeHeartbeatRequest
from core.db.models import NodeRecord
from core.db.session import get_db
from core.control_plane.schemas import NodeRegisterRequest
from datetime import datetime,timezone
from fastapi import Depends,FastAPI,HTTPException,status
from sqlalchemy import select
from sqlalchemy.orm import Session

app = FastAPI(title="Sovra Control Panel", version="0.1.0")


@app.post("/api/v1/nodes/register")
def regsiter_node(request: NodeRegisterRequest, db: Session = Depends(get_db)):
    """Idempotently register or update a GPU node."""
    node = db.get(NodeRecord,request.id)

    if not node:
        #insert
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
        #update
        node.hostname = request.hostname
        node.ip_address = request.ip_address
        node.gpu_name = request.gpu_name
        node.total_vram_mb = request.total_vram_mb
        node.free_vram_mb = request.free_vram_mb
        node.status = "online"
        node.last_heartbeat = datetime.now(timezone.utc)
    
    db.commit()
    db.refresh(node)
    return {"status": "registered", "node_id" : node.id}

@app.post("/api/v1/nodes/{node_id}/heartbeat")
def node_heartbeat(node_id : str, request: NodeHeartbeatRequest, db: Session = Depends(get_db)):
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