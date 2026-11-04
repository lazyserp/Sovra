from pydantic import BaseModel

class ModelSpec(BaseModel):
    id: str
    name: str
    artifact_path: str
    min_vram_mb: int
    min_disk_gb: float = 2.0
    max_context_length: int = 4096
    engine_type: str = "vllm"
    quantization: str | None = None

QWEN_0_5B_SPEC = ModelSpec(
    id="qwen-2.5-0.5b-instruct",
    name="Qwen 2.5 0.5B Instruct",
    artifact_path="./storage/models/qwen2.5-0.5b-instruct",
    min_vram_mb=1500,  
    min_disk_gb=2.0,
    max_context_length=4096,
)