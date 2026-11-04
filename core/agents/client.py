from core import preflight
import socket
import httpx

from core.control_plane.schemas import NodeHeartbeatRequest,NodeRegisterRequest
from core.preflight.checker import run_preflight

class NodeAgent:
    """Agent running on GPU hosts that registers with and reports to the Control Plane."""
    def __init__(self,control_plane_url: str = "http://localhost:8000", node_id :str | None = None):
        self.control_plane_url = control_plane_url
        self.hostname = socket.gethostname()
        self.node_id = node_id or f"node-{self.hostname.lower()}"
        self.ip_address = self._get_local_ip()


    def _get_local_ip(self) -> str:
        try:
            with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as s:
                s.connect(("10.255.255.255",1))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"


    def inspect_hardware(self) -> NodeRegisterRequest:
        """Run preflight checks and convert hardware diagnostics into a registration payload."""
        preflight = run_preflight()

        gpu_name = None
        total_vram = 0
        free_vram = 0

        for check in preflight.checks:
            if check.name == "nvidia_gpu" and check.passed:
                gpu_name = check.details.get("gpu_name")
                total_vram = check.details.get("total_vram_mb", 0)
                free_vram = check.details.get("free_vram_mb", 0)


        return NodeRegisterRequest(
        id=self.node_id,
        hostname=self.hostname,
        ip_address=self.ip_address,
        gpu_name=gpu_name,
        total_vram_mb=total_vram,
        free_vram_mb=free_vram,
    )

    def register(self) -> bool:
        """Register or update this node with the Control Plane."""
        payload = self.inspect_hardware()
        try:
            with httpx.Client(timeout=10.0) as client:
                res = client.post(
                    f"{self.control_plane_url}/api/v1/nodes/register",
                    json=payload.model_dump(),
                )
                return res.status_code == 200
        except httpx.HTTPError:
            return False

            
    def send_heartbeat(self, free_vram_mb: int = 0) -> bool:
        """Send current health telemetry to the Control Plane."""
        payload = NodeHeartbeatRequest(free_vram_mb=free_vram_mb, status="online")
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.post(
                    f"{self.control_plane_url}/api/v1/nodes/{self.node_id}/heartbeat",
                    json=payload.model_dump(),
                )
                return res.status_code == 200
        except httpx.HTTPError:
            return False