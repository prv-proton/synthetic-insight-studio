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
    count_total: int
    meets_k: bool


class ThreadContext(BaseModel):
    stage: str
    actor_role: str
    tone: str
    goals: List[str]
    constraints: List[str]
    blockers: List[str]
    decision_points: List[str]
    what_they_tried: List[str]
    what_they_are_asking: List[str]
    attachments_mentioned: List[str]
    agencies_or_roles: List[str]
    timeline_signals: List[str]


class ThreadSummary(BaseModel):
    one_sentence: str
    stage: str
    goals: List[str]
    constraints: List[str]
    blockers: List[str]
    decision_points: List[str]
    attachments_mentioned: List[str]
    agencies_or_roles: List[str]
    timeline_signals: List[str]


class Persona(BaseModel):
    persona_name: str
    from_role: str
    experience_level: str
    primary_motivation: str
    frustrations: List[str]
    needs: List[str]
    tone: str


class NextQuestions(BaseModel):
    to_clarify: List[str]
    to_unblock: List[str]
    risks_if_ignored: List[str]


class PseudoEmailModel(BaseModel):
    subject: str
    from_role: str
    tone: str
    body: str
    attachments_mentioned: List[str]
    motivations: List[str]
    decision_points: List[str]
    assumptions: List[str]


class PatternResponse(BaseModel):
    theme: str
    pattern: Dict[str, Any]
    updated_at: datetime


class GenerateRequest(BaseModel):
    theme: str
    kind: str
    count: int = 5
    allow_below_threshold: bool = True


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
