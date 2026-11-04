import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from core.db.models import Base, DeploymentRecord, ModelRecord, NodeRecord
from core.db.session import init_db


@pytest.fixture
def test_db_session():
    """Create an isolated in-memory SQLite database for testing."""
    test_engine = create_engine("sqlite:///:memory:")
    init_db(target_engine=test_engine)
    test_session_factory = sessionmaker(test_engine, expire_on_commit=False)

    with test_session_factory() as session:
        yield session

    Base.metadata.drop_all(bind=test_engine)


def test_create_and_query_node(test_db_session: Session):
    """Verify storing and querying a GPU Node record."""
    node = NodeRecord(
        id="node-rtx3050-01",
        hostname="sovra-workstation",
        ip_address="192.168.1.50",
        status="online",
        gpu_name="NVIDIA GeForce RTX 3050 Laptop GPU",
        total_vram_mb=4096,
        free_vram_mb=3962,
    )
    test_db_session.add(node)
    test_db_session.commit()

    queried_node = test_db_session.get(NodeRecord, "node-rtx3050-01")
    assert queried_node is not None
    assert queried_node.hostname == "sovra-workstation"
    assert queried_node.total_vram_mb == 4096
    assert queried_node.status == "online"
    assert queried_node.last_heartbeat is not None


def test_create_and_query_model(test_db_session: Session):
    """Verify storing and querying a Model definition record."""
    model = ModelRecord(
        id="qwen-2.5-0.5b-instruct",
        name="Qwen 2.5 0.5B Instruct",
        artifact_path="./storage/models/qwen2.5-0.5b-instruct",
        min_vram_mb=1500,
        min_disk_gb=2.0,
        max_context_length=4096,
        engine_type="vllm",
    )
    test_db_session.add(model)
    test_db_session.commit()

    queried_model = test_db_session.get(ModelRecord, "qwen-2.5-0.5b-instruct")
    assert queried_model is not None
    assert queried_model.name == "Qwen 2.5 0.5B Instruct"
    assert queried_model.min_vram_mb == 1500
    assert queried_model.created_at is not None


def test_create_and_query_deployment(test_db_session: Session):
    """Verify storing and querying a Deployment record linking node and model."""
    # 1. Create prerequisites
    node = NodeRecord(
        id="node-test",
        hostname="test-host",
        ip_address="127.0.0.1",
        status="online",
    )
    model = ModelRecord(
        id="model-test",
        name="Test Model",
        artifact_path="/tmp/test",
        min_vram_mb=1000,
    )
    test_db_session.add_all([node, model])
    test_db_session.commit()

    # 2. Create deployment
    deployment = DeploymentRecord(
        id="deploy-12345",
        model_id="model-test",
        node_id="node-test",
        status="running",
        port=8000,
        container_id="docker-container-abc",
    )
    test_db_session.add(deployment)
    test_db_session.commit()

    # 3. Query deployment
    queried_deployment = test_db_session.get(DeploymentRecord, "deploy-12345")
    assert queried_deployment is not None
    assert queried_deployment.model_id == "model-test"
    assert queried_deployment.node_id == "node-test"
    assert queried_deployment.status == "running"
    assert queried_deployment.port == 8000
    assert queried_deployment.created_at is not None

