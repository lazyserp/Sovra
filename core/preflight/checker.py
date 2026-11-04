import subprocess
from core.preflight.models import CheckItem


def check_docker(timeout: int = 5) -> CheckItem:
    """Verify that Docker is installed and the Docker daemon is responding.

    Args:
        timeout: Maximum seconds to wait for Docker to respond.

    Returns:
        CheckItem with the status, human-readable message, and diagnostic details.
    """
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        if result.returncode == 0:
            return CheckItem(
                name="docker_daemon",
                passed=True,
                message="Docker daemon is running and responsive.",
            )

        error_output = result.stderr.strip() or result.stdout.strip()
        return CheckItem(
            name="docker_daemon",
            passed=False,
            message=f"Docker is installed, but daemon returned an error: {error_output}",
            details={"exit_code": result.returncode, "error": error_output},
        )

    except FileNotFoundError:
        return CheckItem(
            name="docker_daemon",
            passed=False,
            message="Docker executable not found on system PATH.",
            details={"error": "executable_not_found"},
        )
    except subprocess.TimeoutExpired:
        return CheckItem(
            name="docker_daemon",
            passed=False,
            message=f"Docker check timed out after {timeout} seconds (daemon may be unresponsive).",
            details={"error": "timeout", "timeout_seconds": timeout},
        )
    except Exception as exc:
        return CheckItem(
            name="docker_daemon",
            passed=False,
            message=f"Unexpected error while checking Docker: {exc}",
            details={"error": str(exc)},
        )


def check_nvidia_gpu(min_vram_mb: int = 0, timeout: int = 5) -> CheckItem:
    """Verify NVIDIA GPU presence, driver version, and available VRAM.

    Args:
        min_vram_mb: Minimum required VRAM in megabytes (default 0).
        timeout: Maximum seconds to wait for nvidia-smi.

    Returns:
        CheckItem with the status, hardware specs, and pass/fail verdict.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            return CheckItem(
                name="nvidia_gpu",
                passed=False,
                message=f"nvidia-smi returned an error: {error_msg}",
                details={"exit_code": result.returncode, "error": error_msg},
            )

    except FileNotFoundError:
        return CheckItem(
            name="nvidia_gpu",
            passed=False,
            message="NVIDIA driver or nvidia-smi executable not found on system PATH.",
            details={"error": "executable_not_found"},
        )
    except subprocess.TimeoutExpired:
        return CheckItem(
            name="nvidia_gpu",
            passed=False,
            message=f"nvidia-smi timed out after {timeout} seconds.",
            details={"error": "timeout", "timeout_seconds": timeout},
        )
    except Exception as exc:
        return CheckItem(
            name="nvidia_gpu",
            passed=False,
            message=f"Unexpected error querying nvidia-smi: {exc}",
            details={"error": str(exc)},
        )

    # Validate output lines safely
    lines = result.stdout.strip().splitlines()
    if not lines:
        return CheckItem(
            name="nvidia_gpu",
            passed=False,
            message="nvidia-smi produced no GPU output.",
            details={"error": "empty_output"},
        )

    try:
        parts = [p.strip() for p in lines[0].split(",")]
        gpu_name = parts[0]
        driver_version = parts[1]
        total_vram_mb = int(parts[2])
        free_vram_mb = int(parts[3])
        
    except (IndexError, ValueError) as parse_err:
        return CheckItem(
            name="nvidia_gpu",
            passed=False,
            message=f"Failed to parse nvidia-smi output: {parse_err}",
            details={"raw_output": lines[0], "error": str(parse_err)},
        )

    details = {
        "gpu_name": gpu_name,
        "driver_version": driver_version,
        "total_vram_mb": total_vram_mb,
        "free_vram_mb": free_vram_mb,
        "min_vram_required_mb": min_vram_mb,
    }

    if total_vram_mb < min_vram_mb:
        return CheckItem(
            name="nvidia_gpu",
            passed=False,
            message=(
                f"GPU '{gpu_name}' has insufficient VRAM: "
                f"{total_vram_mb} MB available, but {min_vram_mb} MB required."
            ),
            details=details,
        )

    return CheckItem(
        name="nvidia_gpu",
        passed=True,
        message=f"GPU '{gpu_name}' verified ({total_vram_mb} MB VRAM, driver {driver_version}).",
        details=details,
    )
