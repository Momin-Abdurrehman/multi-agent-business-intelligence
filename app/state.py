"""
Shared state schema and Pydantic output models for the research graph.

ResearchState is the single source of truth passed between all nodes.
Pydantic models are used with .with_structured_output() to eliminate
fragile JSON string parsing and get type-safe LLM responses.
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class ResearchState(TypedDict):
    """
    Shared state passed between every node in the research graph.

    The `messages` field uses the `add_messages` reducer, which means new
    messages are APPENDED to (not replaced in) the checkpoint on every
    invocation — giving us persistent conversation history across turns.
    All other fields use the default "last value wins" reducer and are
    reset per query.
    """

    messages: Annotated[list, add_messages]
    current_query: str            # search query (may be reformulated from original)
    company_name: str             # canonical display name, e.g. "Apple" (set by clarity agent)
    clarity_status: str           # "clear" | "needs_clarification"
    research_findings: Optional[str]
    confidence_score: float       # 0–10
    validation_result: Optional[str]  # "sufficient" | "insufficient"
    research_attempts: int
    low_confidence: bool          # True when synthesis must give a best-effort answer
    clarification_question: str   # dynamic disambiguation question from clarity agent


# ── Pydantic structured-output schemas ────────────────────────────────────────


class ClarityOutput(BaseModel):
    """Returned by the Clarity Agent via .with_structured_output()."""

    clarity_status: Literal["clear", "needs_clarification"]
    reason: str = Field(description="Brief explanation of the clarity decision")
    company_name: str = Field(
        default="",
        description=(
            "The canonical display name of the company or person being researched "
            "(e.g. 'Apple', 'Tesla', 'Elon Musk'). "
            "Resolved from conversation history for follow-up queries. "
            "Empty when clarity_status is needs_clarification."
        ),
    )
    clarification_question: str = Field(
        default="",
        description=(
            "When needs_clarification, a specific question naming the distinct entities "
            "the user might mean. Empty string when clarity_status is clear."
        ),
    )


class ResearchOutput(BaseModel):
    """Returned by the Research Agent via .with_structured_output()."""

    findings: str = Field(description="Detailed research report on the company")
    confidence_score: float = Field(
        ge=0, le=10, description="Confidence in the quality of findings (0–10)"
    )


class ValidationOutput(BaseModel):
    """Returned by the Validator Agent via .with_structured_output()."""

    validation_result: Literal["sufficient", "insufficient"]
    reason: str = Field(description="Explanation of the validation decision")
