from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EnquiryIn(BaseModel):
    text: str = Field(..., min_length=1)


class IngestRequest(BaseModel):
    enquiries: List[EnquiryIn]


class ThemeSummary(BaseModel):
    theme: str
    count: int


class PatternResponse(BaseModel):
    theme: str
    pattern: Dict[str, Any]
    updated_at: datetime


class GenerateRequest(BaseModel):
    theme: str
    kind: str
    count: int = 5


class GeneratedItem(BaseModel):
    id: int
    theme: str
    kind: str
    content: Dict[str, Any]
    created_at: datetime


class AuditItem(BaseModel):
    id: int
    event_type: str
    payload: Dict[str, Any]
    created_at: datetime


class ResetResponse(BaseModel):
    status: str
    cleared: bool


class OllamaStatus(BaseModel):
    available: bool
    detail: Optional[str] = None


class SettingsResponse(BaseModel):
    k_threshold: int
