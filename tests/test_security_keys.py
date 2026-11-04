from datetime import datetime, timedelta, timezone
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.db.models import ApiKeyRecord, Base
from core.db.session import init_db
from core.security.keys import generate_api_key, hash_key, verify_api_key


@pytest.fixture
def db_session():
    """Isolated in-memory test database session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(target_engine=engine)
    session_factory = sessionmaker(engine, expire_on_commit=False)

    with session_factory() as session:
        yield session

    Base.metadata.drop_all(bind=engine)


def test_generate_api_key_format():
    """Verify generated keys meet cryptographic format expectations."""
    plain_key, key_hash, prefix = generate_api_key()

    assert plain_key.startswith("sovra_sk_live_")
    assert len(key_hash) == 64  # SHA-256 produces 64 hex characters
    assert hash_key(plain_key) == key_hash
    assert prefix == plain_key[:16]


def test_verify_api_key_valid(db_session):
    """Verify an active, non-expired key is accepted."""
    plain_key, key_hash, prefix = generate_api_key()
    record = ApiKeyRecord(
        name="test-service",
        key_hash=key_hash,
        prefix=prefix,
        is_active=True,
    )
    db_session.add(record)
    db_session.commit()

    verified = verify_api_key(plain_key, db_session)
    assert verified is not None
    assert verified.id == record.id
    assert verified.name == "test-service"


def test_verify_api_key_unknown(db_session):
    """Verify an unknown key returns None without crashing."""
    verified = verify_api_key("sovra_sk_live_totally_fake_key", db_session)
    assert verified is None


def test_verify_api_key_revoked(db_session):
    """Verify a revoked key (is_active=False) is rejected."""
    plain_key, key_hash, prefix = generate_api_key()
    record = ApiKeyRecord(
        name="revoked-service",
        key_hash=key_hash,
        prefix=prefix,
        is_active=False,
    )
    db_session.add(record)
    db_session.commit()

    verified = verify_api_key(plain_key, db_session)
    assert verified is None


def test_verify_api_key_expired(db_session):
    """Verify a key past its TTL expiration is rejected."""
    plain_key, key_hash, prefix = generate_api_key()
    record = ApiKeyRecord(
        name="expired-service",
        key_hash=key_hash,
        prefix=prefix,
        is_active=True,
        expires_at=datetime.now(timezone.utc) - timedelta(hours=1),  # Expired 1 hr ago
    )
    db_session.add(record)
    db_session.commit()

    verified = verify_api_key(plain_key, db_session)
    assert verified is None

