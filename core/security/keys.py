import hashlib
import secrets
from datetime import datetime,timezone
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.db.models import ApiKeyRecord

def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

def generate_api_key(prefix: str = "sovra_sk_live") -> tuple[str,str,str]:
    """
    Generate a cryptographically secure API key.
    
    Returns:
        (plain_key, key_hash, prefix)
        - plain_key: Shown to the user ONCE (e.g. 'sovra_sk_live_AbCd123...')
        - key_hash: Stored in the database
        - prefix: Safe display prefix (e.g. 'sovra_sk_live_AbCd')
    """

    token  = secrets.token_urlsafe(32)
    plain_key = f"{prefix}{token}"
    key_hash = hash_key(plain_key)
    safe_prefix = plain_key[:16]
    return plain_key,key_hash,safe_prefix


def verify_api_key(raw_key: str, db:Session) -> ApiKeyRecord | None:
    """
    Verify an incoming raw API key against the database.
    
    Returns ApiKeyRecord if valid, active, and not expired. Otherwise None.
    """
    key_hash = hash_key(raw_key)
    stmt = select(ApiKeyRecord).where(ApiKeyRecord.key_hash == key_hash)
    record = db.scalar(stmt)

    if not record:
        return None

    if record.expires_at is not None:
        if record.expires_at < datetime.now(timezone.utc):
            return None
    return record