import pytest
from datetime import timezone
from pydantic import ValidationError

from core.preflight.models import CheckItem, PreflightResult


def test_check_item_defaults():
    """Verify that CheckItem sets defaults properly for optional fields."""
    item = CheckItem(name="docker", passed=True, message="Docker is active")
    assert item.name == "docker"
    assert item.passed is True
    assert item.message == "Docker is active"
    assert item.details == {}


def test_preflight_result_defaults():
    """Verify that PreflightResult defaults to an empty check list and a UTC timestamp."""
    result = PreflightResult(passed=True)
    assert result.passed is True
    assert result.checks == []
    assert result.timestamp.tzinfo == timezone.utc


def test_preflight_result_with_checks():
    """Verify that CheckItem instances can be properly nested inside PreflightResult."""
    item = CheckItem(name="gpu_vram", passed=True, message="4096 MB VRAM detected", details={"vram_mb": 4096})
    result = PreflightResult(passed=True, checks=[item])

    assert len(result.checks) == 1
    assert result.checks[0].name == "gpu_vram"
    assert result.checks[0].details["vram_mb"] == 4096


def test_check_item_validation_error():
    """Negative test: Verify that omitting mandatory fields raises a ValidationError."""
    with pytest.raises(ValidationError):
        # Missing 'passed' and 'message'
        CheckItem(name="docker")  # type: ignore[call-arg]
