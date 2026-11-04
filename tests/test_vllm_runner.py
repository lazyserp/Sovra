import urllib.error
from unittest.mock import MagicMock, patch

from core.engine.vllm import VLLMRunner
from core.models.specs import QWEN_0_5B_SPEC, ModelSpec


def test_build_run_command_basic():
    """Verify that build_run_command generates valid Docker CLI arguments."""
    runner = VLLMRunner(spec=QWEN_0_5B_SPEC, port=8000)
    cmd = runner.build_run_command()

    assert cmd[0:3] == ["docker", "run", "-d"]
    assert "--name" in cmd
    assert runner.container_name in cmd
    assert "--gpus" in cmd
    assert "all" in cmd
    assert "-p" in cmd
    assert "8000:8000" in cmd
    assert "--model" in cmd
    assert "/model" in cmd
    assert "--max-model-len" in cmd
    assert "4096" in cmd
    assert "--gpu-memory-utilization" in cmd
    assert "0.85" in cmd
    assert "vllm/vllm-openai:latest" in cmd


def test_build_run_command_with_quantization():
    """Verify that quantization flag is dynamically appended when set."""
    awq_spec = ModelSpec(
        id="qwen-awq",
        name="Qwen AWQ",
        artifact_path="./storage/models/qwen-awq",
        min_vram_mb=2000,
        quantization="awq",
    )
    runner = VLLMRunner(spec=awq_spec)
    cmd = runner.build_run_command()

    assert "--quantization" in cmd
    assert "awq" in cmd


def test_start_success():
    """Verify start returns True on zero exit code."""
    runner = VLLMRunner(spec=QWEN_0_5B_SPEC)
    mock_result = MagicMock(returncode=0, stdout="container-hash-12345", stderr="")

    with patch("subprocess.run", return_value=mock_result):
        assert runner.start() is True
        assert runner.last_error is None


def test_start_failure():
    """Verify start captures stderr on failure."""
    runner = VLLMRunner(spec=QWEN_0_5B_SPEC)
    mock_result = MagicMock(returncode=125, stdout="", stderr="Error: port 8000 is already in use")

    with patch("subprocess.run", return_value=mock_result):
        assert runner.start() is False
        assert "port 8000 is already in use" in runner.last_error


def test_stop():
    """Verify stop invokes docker stop and docker rm."""
    runner = VLLMRunner(spec=QWEN_0_5B_SPEC)
    with patch("subprocess.run") as mock_run:
        assert runner.stop() is True
        assert mock_run.call_count == 2


def test_is_running():
    """Verify is_running parses docker ps output accurately."""
    runner = VLLMRunner(spec=QWEN_0_5B_SPEC)

    # When container is present in output:
    mock_active = MagicMock(returncode=0, stdout=f"{runner.container_name}\n")
    with patch("subprocess.run", return_value=mock_active):
        assert runner.is_running() is True

    # When container is not present in output:
    mock_inactive = MagicMock(returncode=0, stdout="some-other-container\n")
    with patch("subprocess.run", return_value=mock_inactive):
        assert runner.is_running() is False


def test_wait_until_ready_success():
    """Verify wait_until_ready returns True when /health returns HTTP 200."""
    runner = VLLMRunner(spec=QWEN_0_5B_SPEC)
    mock_response = MagicMock(status=200)
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        assert runner.wait_until_ready(timeout_seconds=2, poll_interval=0.1) is True


def test_wait_until_ready_timeout():
    """Verify wait_until_ready times out if the server never becomes healthy."""
    runner = VLLMRunner(spec=QWEN_0_5B_SPEC)

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        assert runner.wait_until_ready(timeout_seconds=0.2, poll_interval=0.05) is False

