import uuid
from datetime import datetime, timezone
from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy 2.0 declarative models."""
    pass


class NodeRecord(Base):
    """Represents a physical or virtual GPU compute node managed by Sovra."""
    __tablename__ = "nodes"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    hostname: Mapped[str] = mapped_column(String, nullable=False)
    ip_address: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="online")  
    gpu_name: Mapped[str | None] = mapped_column(String, nullable=True)
    total_vram_mb: Mapped[int] = mapped_column(default=0)
    free_vram_mb: Mapped[int] = mapped_column(default=0)
    last_heartbeat: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class ModelRecord(Base):
    """Represents an AI model registered in the Sovra platform."""
    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String, primary_key=True)  
    name: Mapped[str] = mapped_column(String, nullable=False)
    artifact_path: Mapped[str] = mapped_column(String, nullable=False)
    min_vram_mb: Mapped[int] = mapped_column(nullable=False)
    min_disk_gb: Mapped[float] = mapped_column(default=2.0)
    max_context_length: Mapped[int] = mapped_column(default=4096)
    engine_type: Mapped[str] = mapped_column(String, default="vllm")
    quantization: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class DeploymentRecord(Base):
    """Represents an active or historic model deployment instance bound to a node."""
    __tablename__ = "deployments"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    model_id: Mapped[str] = mapped_column(ForeignKey("models.id"), nullable=False)
    node_id: Mapped[str] = mapped_column(ForeignKey("nodes.id"), nullable=False)
    status: Mapped[str] = mapped_column(String, default="pending")  
    port: Mapped[int] = mapped_column(default=8000)
    container_id: Mapped[str | None] = mapped_column(String, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
