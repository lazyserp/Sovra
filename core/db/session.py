from collections.abc import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from core.db.models import Base

DATABASE_URL = "sqlite:///./sovra.db"

# SQLAlchemy 2.0 Engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    echo=False,
)

# SQLAlchemy 2.0 Session Factory
SessionLocal = sessionmaker(engine, expire_on_commit=False)


def init_db(target_engine=None) -> None:
    """Create all database tables."""
    eng = target_engine or engine
    Base.metadata.create_all(bind=eng)


def get_db() -> Generator[Session, None, None]:
    """Dependency that provides a transactional database session per request."""
    with SessionLocal() as session:
        yield session
