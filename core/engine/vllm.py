import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from core.models.specs import ModelSpec


class VLLMRunner:
    """Manages the lifecycle of a containerized vLLM inference engine instance."""

    def __init__(self,spec: ModelSpec,port: int = 8000,container_name: str | None = None,gpu_memory_utilization: float = 0.85,) -> None:
        self.spec = spec
        self.port = port
        self.container_name = container_name or f"sovra-engine-{spec.id}"
        self.gpu_memory_utilization = gpu_memory_utilization
        self.last_error: str | None = None

    def build_run_command(self) -> list[str]:
        """Generate the deterministic docker run command to launch vLLM.

        Returns:
            List of command arguments suitable for subprocess.run.
        """
        # Convert path to absolute POSIX format (safe for Windows Docker Desktop and Linux)
        mount_path = Path(self.spec.artifact_path).resolve().as_posix()

        cmd = [
            "docker", "run", "-d",
            "--name", self.container_name,
            "--gpus", "all",
            "-p", f"{self.port}:8000",
            "-v", f"{mount_path}:/model:ro",
            "vllm/vllm-openai:latest",
            "--model", "/model",
            "--max-model-len", str(self.spec.max_context_length),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
        ]

        if self.spec.quantization:
            cmd.extend(["--quantization", self.spec.quantization])

        return cmd

    def start(self) -> bool:
        """Start the vLLM container.

        Returns:
            True if the container started successfully, False otherwise.
        """
        cmd = self.build_run_command()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                self.last_error = None
                return True

            self.last_error = result.stderr.strip() or result.stdout.strip()
            return False

        except Exception as exc:
            self.last_error = str(exc)
            return False

    def stop(self) -> bool:
        """Stop and remove the vLLM container.

        Returns:
            True if stopped and removed without unhandled errors.
        """
        try:
            # Stop the running container
            subprocess.run(
                ["docker", "stop", self.container_name],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            # Force remove to ensure clean state
            subprocess.run(
                ["docker", "rm", "-f", self.container_name],
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
            return True
        except Exception as exc:
            self.last_error = str(exc)
            return False

    def is_running(self) -> bool:
        """Check if the container is currently running via docker ps.

        Returns:
            True if the container is active and in running state.
        """
        try:
            result = subprocess.run(
                [
                    "docker", "ps",
                    "--filter", f"name=^{self.container_name}$",
                    "--format", "{{.Names}}",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if result.returncode == 0:
                matching_containers = result.stdout.strip().splitlines()
                return self.container_name in matching_containers
            return False
        except Exception:
            return False

    def wait_until_ready(self,timeout_seconds: int = 60,poll_interval: float = 1.0,) -> bool:
        """Poll the vLLM health endpoint until the model finishes loading.

        Args:
            timeout_seconds: Maximum time to wait for the engine to initialize.
            poll_interval: Seconds to sleep between health checks.

        Returns:
            True if the engine responded HTTP 200 within timeout, False otherwise.
        """
        health_url = f"http://localhost:{self.port}/health"
        start_time = time.monotonic()

        while time.monotonic() - start_time < timeout_seconds:
            try:
                req = urllib.request.Request(health_url, headers={"User-Agent": "Sovra-Agent"})
                with urllib.request.urlopen(req, timeout=2) as response:
                    if response.status == 200:
                        return True
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
                # The engine is still initializing or port not yet open
                pass

            time.sleep(poll_interval)

        return False
