import shutil
import subprocess
from unittest.mock import MagicMock, patch

from core.preflight.checker import check_disk_space, check_docker, check_nvidia_gpu, run_preflight
from core.preflight.models import CheckItem


# ============================================================================
# Docker Checker Tests
# ============================================================================

def test_check_docker_success():
    """Verify check_docker returns passed=True when docker info succeeds."""
    mock_result = MagicMock(returncode=0, stdout="Server Version: 24.0.5", stderr="")

    with patch("subprocess.run", return_value=mock_result):
        item = check_docker()
        assert item.name == "docker_daemon"
        assert item.passed is True
        assert "running" in item.message.lower()


def test_check_docker_daemon_down():
    """Verify check_docker returns passed=False when docker daemon is unreachable."""
    mock_result = MagicMock(
        returncode=1,
        stdout="",
        stderr="Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?",
    )

    with patch("subprocess.run", return_value=mock_result):
        item = check_docker()
        assert item.name == "docker_daemon"
        assert item.passed is False
        assert "daemon returned an error" in item.message
        assert item.details["exit_code"] == 1


def test_check_docker_not_installed():
    """Verify check_docker returns passed=False when docker executable is not found in PATH."""
    with patch("subprocess.run", side_effect=FileNotFoundError("No such file")):
        item = check_docker()
        assert item.name == "docker_daemon"
        assert item.passed is False
        assert "not found" in item.message.lower()
        assert item.details["error"] == "executable_not_found"


def test_check_docker_timeout():
    """Verify check_docker returns passed=False when the call exceeds the timeout."""
    with patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["docker", "info"], timeout=5),
    ):
        item = check_docker(timeout=5)
        assert item.name == "docker_daemon"
        assert item.passed is False
        assert "timed out" in item.message.lower()
        assert item.details["error"] == "timeout"


def test_check_docker_live_contract():
    """Verify the live check returns a valid CheckItem instance regardless of host state."""
    item = check_docker()
    assert item.name == "docker_daemon"
    assert isinstance(item.passed, bool)
    assert isinstance(item.message, str)
    assert len(item.message) > 0


# ============================================================================
# NVIDIA GPU Checker Tests
# ============================================================================

def test_check_nvidia_gpu_success():
    """Verify GPU check passes when VRAM meets the minimum requirement."""
    mock_output = "NVIDIA GeForce RTX 3050 Laptop GPU, 610.88, 4096, 3962\n"
    mock_result = MagicMock(returncode=0, stdout=mock_output, stderr="")

    with patch("subprocess.run", return_value=mock_result):
        item = check_nvidia_gpu(min_vram_mb=2048)
        assert item.name == "nvidia_gpu"
        assert item.passed is True
        assert "verified" in item.message
        assert item.details["gpu_name"] == "NVIDIA GeForce RTX 3050 Laptop GPU"
        assert item.details["total_vram_mb"] == 4096
        assert item.details["free_vram_mb"] == 3962


def test_check_nvidia_gpu_insufficient_vram():
    """Verify GPU check fails when VRAM is lower than model requirements."""
    mock_output = "NVIDIA GeForce RTX 3050 Laptop GPU, 610.88, 4096, 3962\n"
    mock_result = MagicMock(returncode=0, stdout=mock_output, stderr="")

    with patch("subprocess.run", return_value=mock_result):
        item = check_nvidia_gpu(min_vram_mb=8192)  # Demanding 8 GB on a 4 GB card
        assert item.name == "nvidia_gpu"
        assert item.passed is False
        assert "insufficient VRAM" in item.message
        assert item.details["total_vram_mb"] == 4096
        assert item.details["min_vram_required_mb"] == 8192


def test_check_nvidia_gpu_not_installed():
    """Verify GPU check handles missing nvidia-smi / driver."""
    with patch("subprocess.run", side_effect=FileNotFoundError):
        item = check_nvidia_gpu()
        assert item.name == "nvidia_gpu"
        assert item.passed is False
        assert "not found" in item.message.lower()


def test_check_nvidia_gpu_timeout():
    """Verify GPU check handles hung nvidia-smi calls."""
    with patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["nvidia-smi"], timeout=5),
    ):
        item = check_nvidia_gpu(timeout=5)
        assert item.name == "nvidia_gpu"
        assert item.passed is False
        assert "timed out" in item.message.lower()


def test_check_nvidia_gpu_driver_error():
    """Verify GPU check handles non-zero exit codes from nvidia-smi."""
    mock_result = MagicMock(
        returncode=1,
        stdout="",
        stderr="NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.",
    )

    with patch("subprocess.run", return_value=mock_result):
        item = check_nvidia_gpu()
        assert item.name == "nvidia_gpu"
        assert item.passed is False
        assert "returned an error" in item.message


def test_check_nvidia_gpu_live():
    """Live verification against the host RTX 3050 hardware."""
    item = check_nvidia_gpu(min_vram_mb=2000)
    assert item.name == "nvidia_gpu"
    assert item.passed is True
    assert "RTX 3050" in item.details["gpu_name"]
    assert item.details["total_vram_mb"] == 4096


# ============================================================================
# Disk Space Checker Tests
# ============================================================================

def test_check_disk_space_success():
    """Verify disk space check passes when free space exceeds requirement."""
    total_bytes = 100 * (1024 ** 3)
    free_bytes = 50 * (1024 ** 3)
    used_bytes = 50 * (1024 ** 3)

    with patch("shutil.disk_usage", return_value=(total_bytes, used_bytes, free_bytes)):
        item = check_disk_space(path=".", min_free_gb=10.0)
        assert item.name == "disk_space"
        assert item.passed is True
        assert item.details["free_gb"] == 50.0
        assert "sufficient" in item.message.lower()


def test_check_disk_space_insufficient():
    """Verify disk space check fails when free space is below requirement."""
    total_bytes = 100 * (1024 ** 3)
    free_bytes = 2 * (1024 ** 3)
    used_bytes = 98 * (1024 ** 3)

    with patch("shutil.disk_usage", return_value=(total_bytes, used_bytes, free_bytes)):
        item = check_disk_space(path=".", min_free_gb=10.0)
        assert item.name == "disk_space"
        assert item.passed is False
        assert item.details["free_gb"] == 2.0
        assert "insufficient" in item.message.lower()


def test_check_disk_space_invalid_path():
    """Verify disk space check returns a failed CheckItem on invalid path without raising an exception."""
    with patch("shutil.disk_usage", side_effect=FileNotFoundError("Invalid path")):
        item = check_disk_space(path="/non/existent/path")
        assert item.name == "disk_space"
        assert item.passed is False
        assert "failed" in item.message.lower()
        assert "error" in item.details


def test_check_disk_space_live():
    """Live verification against current working drive."""
    item = check_disk_space(path=".", min_free_gb=1.0)
    assert item.name == "disk_space"
    assert isinstance(item.passed, bool)
    assert item.details["total_gb"] > 0
    assert item.details["free_gb"] > 0


# ============================================================================
# Preflight Orchestrator Tests
# ============================================================================

def test_run_preflight_all_pass():
    """Verify run_preflight passes when all checks pass."""
    pass_item = CheckItem(name="mock", passed=True, message="All good")

    with patch("core.preflight.checker.check_docker", return_value=pass_item), \
         patch("core.preflight.checker.check_nvidia_gpu", return_value=pass_item), \
         patch("core.preflight.checker.check_disk_space", return_value=pass_item):

        result = run_preflight(min_vram_mb=2048)
        assert result.passed is True
        assert len(result.checks) == 3


def test_run_preflight_one_fails():
    """Verify run_preflight fails if any single check fails."""
    pass_item = CheckItem(name="mock_pass", passed=True, message="OK")
    fail_item = CheckItem(name="mock_fail", passed=False, message="Hardware mismatch")

    with patch("core.preflight.checker.check_docker", return_value=pass_item), \
         patch("core.preflight.checker.check_nvidia_gpu", return_value=fail_item), \
         patch("core.preflight.checker.check_disk_space", return_value=pass_item):

        result = run_preflight(min_vram_mb=8192)
        assert result.passed is False
        assert len(result.checks) == 3
