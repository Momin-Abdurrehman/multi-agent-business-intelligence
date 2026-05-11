"""
Unit tests for Pydantic output schemas.

Validates that the structured-output contracts enforced by each agent
reject invalid data at the schema level — not silently at runtime.
Run with: uv run pytest tests/
"""
import pytest
from pydantic import ValidationError

from app.state import ClarityOutput, ResearchOutput, ValidationOutput


# ── ClarityOutput ──────────────────────────────────────────────────────────────

def test_clarity_output_accepts_clear():
    obj = ClarityOutput(clarity_status="clear", reason="Company name present")
    assert obj.clarity_status == "clear"


def test_clarity_output_accepts_needs_clarification():
    obj = ClarityOutput(clarity_status="needs_clarification", reason="No company named")
    assert obj.clarity_status == "needs_clarification"


def test_clarity_output_clarification_question_defaults_empty():
    obj = ClarityOutput(clarity_status="clear", reason="Apple named")
    assert obj.clarification_question == ""


def test_clarity_output_accepts_clarification_question():
    obj = ClarityOutput(
        clarity_status="needs_clarification",
        reason="Ambiguous name",
        clarification_question="Are you asking about Mercury the fintech or the logistics company?",
    )
    assert "Mercury" in obj.clarification_question


def test_clarity_output_rejects_invalid_status():
    with pytest.raises(ValidationError):
        ClarityOutput(clarity_status="maybe", reason="x")


# ── ResearchOutput ─────────────────────────────────────────────────────────────

def test_research_output_accepts_valid():
    obj = ResearchOutput(findings="Apple is a tech company.", confidence_score=8.0)
    assert obj.confidence_score == 8.0


def test_research_output_rejects_score_above_10():
    with pytest.raises(ValidationError):
        ResearchOutput(findings="x", confidence_score=11.0)


def test_research_output_rejects_score_below_0():
    with pytest.raises(ValidationError):
        ResearchOutput(findings="x", confidence_score=-1.0)


def test_research_output_accepts_boundary_values():
    assert ResearchOutput(findings="x", confidence_score=0.0).confidence_score == 0.0
    assert ResearchOutput(findings="x", confidence_score=10.0).confidence_score == 10.0


# ── ValidationOutput ───────────────────────────────────────────────────────────

def test_validation_output_accepts_sufficient():
    obj = ValidationOutput(validation_result="sufficient", reason="Data is specific")
    assert obj.validation_result == "sufficient"


def test_validation_output_accepts_insufficient():
    obj = ValidationOutput(validation_result="insufficient", reason="Too vague")
    assert obj.validation_result == "insufficient"


def test_validation_output_rejects_invalid_result():
    with pytest.raises(ValidationError):
        ValidationOutput(validation_result="partial", reason="x")
