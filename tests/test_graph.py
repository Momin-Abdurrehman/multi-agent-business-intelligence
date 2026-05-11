"""
Unit tests for the graph routing functions.

These are pure functions — no LLM calls, no API keys, no mocking required.
Run with: uv run pytest tests/
"""
import pytest

from app.graph import route_clarity, route_research, route_validation


# ── route_clarity ──────────────────────────────────────────────────────────────

def test_route_clarity_clear():
    assert route_clarity({"clarity_status": "clear"}) == "research_agent"


def test_route_clarity_needs_clarification():
    assert route_clarity({"clarity_status": "needs_clarification"}) == "human_clarification"


# ── route_research ─────────────────────────────────────────────────────────────

def test_route_research_high_confidence():
    assert route_research({"confidence_score": 7.0}) == "synthesis_agent"


def test_route_research_at_threshold():
    assert route_research({"confidence_score": 6.0}) == "synthesis_agent"


def test_route_research_low_confidence():
    assert route_research({"confidence_score": 5.9}) == "validator_agent"


def test_route_research_zero_confidence():
    assert route_research({"confidence_score": 0.0}) == "validator_agent"


# ── route_validation ───────────────────────────────────────────────────────────

def test_route_validation_sufficient():
    state = {"validation_result": "sufficient", "research_attempts": 1}
    assert route_validation(state) == "synthesis_agent"


def test_route_validation_insufficient_retries():
    state = {"validation_result": "insufficient", "research_attempts": 1}
    assert route_validation(state) == "research_agent"


def test_route_validation_insufficient_at_max():
    state = {"validation_result": "insufficient", "research_attempts": 3}
    assert route_validation(state) == "set_low_confidence"


def test_route_validation_insufficient_below_max():
    state = {"validation_result": "insufficient", "research_attempts": 2}
    assert route_validation(state) == "research_agent"


def test_route_validation_sufficient_overrides_max_attempts():
    # Even at max attempts, sufficient validation goes to synthesis — not the fallback
    state = {"validation_result": "sufficient", "research_attempts": 3}
    assert route_validation(state) == "synthesis_agent"
