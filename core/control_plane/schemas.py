from pydantic import BaseModel


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