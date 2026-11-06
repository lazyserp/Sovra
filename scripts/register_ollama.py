"""Helper script to register a locally running Ollama model into Sovra's database.

Usage:
    python scripts/register_ollama.py [model_name] [port]
Example:
    python scripts/register_ollama.py qwen2.5:0.5b 11434
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.db.models import DeploymentRecord, ModelRecord, NodeRecord
from core.db.session import SessionLocal, init_db

def register_ollama(model_name: str = "qwen2.5:0.5b", port: int = 11434):
    init_db()
    with SessionLocal() as db:
        # 1. Register or update the local compute node
        node = db.get(NodeRecord, "node-local-pc")
        if not node:
            node = NodeRecord(
                id="node-local-pc",
                hostname="local-gpu-rig",
                ip_address="127.0.0.1",
                gpu_name="NVIDIA GeForce RTX 3050",
                total_vram_mb=4096,
                free_vram_mb=3500,
                status="online",
            )
            db.add(node)

        # 2. Register the model metadata
        model = db.get(ModelRecord, model_name)
        if not model:
            model = ModelRecord(
                id=model_name,
                name=f"Qwen 2.5 (Local Engine)",
                artifact_path=f"ollama://{model_name}",
                min_vram_mb=400,
                engine_type="ollama",
            )
            db.add(model)

        # 3. Register or activate the deployment
        deploy = db.get(DeploymentRecord, f"deploy-{model_name.replace(':', '-')}")
        if not deploy:
            deploy = DeploymentRecord(
                id=f"deploy-{model_name.replace(':', '-')}",
                model_id=model_name,
                node_id="node-local-pc",
                port=port,
                status="running",
            )
            db.add(deploy)
        else:
            deploy.status = "running"
            deploy.port = port

        db.commit()
        print(f"✅ Successfully registered deployment in Sovra:")
        print(f"   Model: {model_name}")
        print(f"   Port:  {port}")
        print(f"   Node:  node-local-pc (NVIDIA RTX 3050)")
        print(f"   Status: running")

if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:0.5b"
    p = int(sys.argv[2]) if len(sys.argv) > 2 else 11434
    register_ollama(m, p)
