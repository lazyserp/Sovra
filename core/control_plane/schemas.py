from datetime import datetime
from pydantic import BaseModel, ConfigDict


class NodeRegisterRequest(BaseModel):
    id: str
    hostname: str
    ip_address: str
    gpu_name: str | None = None
    total_vram_mb: int = 0
    free_vram_mb: int = 0


class NodeHeartbeatRequest(BaseModel):
    free_vram_mb: int
    status: str = "online"


class DeploymentCreateRequest(BaseModel):
    model_id: str
    node_id: str
    port: int | None = None


class ApiKeyCreateRequest(BaseModel):
    name: str
    expires_in_days: int | None = None


class ApiKeyResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    prefix: str
    is_active: bool
    created_at: datetime
    expires_at: datetime | None = None
