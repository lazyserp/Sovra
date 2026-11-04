import pytest
from pydantic import ValidationError

from core.models.specs import QWEN_0_5B_SPEC, ModelSpec


def test_model_spec_creation():
    """Verify that ModelSpec correctly stores all fields."""
    spec = ModelSpec(
        id="test-model",
        name="Test Model",
        artifact_path="/path/to/weights",
        min_vram_mb=4000,
        min_disk_gb=10.0,
        max_context_length=8192,
        engine_type="vllm",
        quantization="awq",
    )
    assert spec.id == "test-model"
    assert spec.name == "Test Model"
    assert spec.artifact_path == "/path/to/weights"
    assert spec.min_vram_mb == 4000
    assert spec.min_disk_gb == 10.0
    assert spec.max_context_length == 8192
    assert spec.engine_type == "vllm"
    assert spec.quantization == "awq"


def test_model_spec_defaults():
    """Verify default values for optional parameters."""
    spec = ModelSpec(
        id="test-model",
        name="Test Model",
        artifact_path="/path/to/weights",
        min_vram_mb=2000,
    )
    assert spec.min_disk_gb == 2.0
    assert spec.max_context_length == 4096
    assert spec.engine_type == "vllm"
    assert spec.quantization is None


def test_model_spec_validation_error():
    """Verify omitting mandatory fields raises ValidationError."""
    with pytest.raises(ValidationError):
        ModelSpec(id="incomplete")  # type: ignore[call-arg]


def test_qwen_preset_validity():
    """Verify the pre-configured Qwen 0.5B preset matches hardware targets."""
    assert QWEN_0_5B_SPEC.id == "qwen-2.5-0.5b-instruct"
    assert QWEN_0_5B_SPEC.min_vram_mb == 1500
    assert QWEN_0_5B_SPEC.min_vram_mb <= 4096  # Must fit in host GPU

