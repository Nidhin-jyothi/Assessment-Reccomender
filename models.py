"""
models.py - All Pydantic schemas.

"""

from __future__ import annotations

from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


#  Request 

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: Annotated[list[Message], Field(min_length=1)]

    @field_validator("messages")
    @classmethod
    def at_least_one_user(cls, v: list[Message]) -> list[Message]:
        if not any(m.role == "user" for m in v):
            raise ValueError("messages must contain at least one user turn")
        return v


#  Response 

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str   # single primary letter e.g. "K", "P", "A"

    @field_validator("url")
    @classmethod
    def must_be_shl_url(cls, v: str) -> str:
        if "shl.com" not in v:
            raise ValueError(f"URL must be from shl.com, got: {v}")
        return v

    @field_validator("test_type")
    @classmethod
    def valid_test_type(cls, v: str) -> str:
        valid = {"A", "B", "C", "D", "E", "K", "P", "S"}
        # Take first letter if multiple passed accidentally
        letter = v.strip().upper()[0] if v.strip() else "K"
        if letter not in valid:
            raise ValueError(f"test_type '{letter}' not in {valid}")
        return letter


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False

    @model_validator(mode="after")
    def recommendations_bounded(self) -> "ChatResponse":
        n = len(self.recommendations)
        if n > 10:
            raise ValueError(f"recommendations must have 10 items, got {n}")
        return self


#  Health 

class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


#  Internal agent state (not serialized to API) 

class Constraint(BaseModel):
    """Accumulated hiring context extracted across conversation turns."""
    role_title:       Optional[str]  = None
    role_description: Optional[str]  = None
    seniority:        Optional[str]  = None
    purpose:          Optional[str]  = None   # "selection" | "development"
    languages:        list[str]      = Field(default_factory=list)
    skills_needed:    list[str]      = Field(default_factory=list)


class AgentState(BaseModel):
    """Full LangGraph node state passed between nodes."""
    messages:         list[Message]
    constraints:      Constraint              = Field(default_factory=Constraint)
    intent:           Optional[str]           = None   # vague|ready|refine|compare|off_scope|satisfied
    current_shortlist: list[Recommendation]   = Field(default_factory=list)
    turn_count:       int                     = 0
    clarifying_asked: list[str]               = Field(default_factory=list)  # avoid repeating same Q
    raw_retrieval:    list[dict]              = Field(default_factory=list)  # for compare node
