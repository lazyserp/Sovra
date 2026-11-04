from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field


class CheckItem(BaseModel):
    name: str
    passed: bool
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class PreflightResult(BaseModel):
    passed: bool
    checks: list[CheckItem] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
