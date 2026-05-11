"""
LangGraph graph construction for the multi-agent research pipeline.

Graph topology
──────────────
START
  └─► clarity_agent
        ├─► human_clarification  (query is vague → interrupt → user clarifies → re-evaluate)
        │       └─► clarity_agent
        └─► research_agent       (query is clear)
              ├─► validator_agent          (confidence < 6)
              │     ├─► research_agent     (insufficient AND attempts < MAX)
              │     └─► set_low_confidence (insufficient AND attempts ≥ MAX)
              │               └─► synthesis_agent
              └─► synthesis_agent          (confidence ≥ 6 OR validation sufficient)
                    └─► END

The graph is compiled with MemorySaver so:
  • conversation history persists across turns (add_messages reducer)
  • interrupt() can pause execution and resume with Command(resume=...)
  For production, swap MemorySaver for SqliteSaver / PostgresSaver.
"""
from __future__ import annotations

import functools
from typing import Any

from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.agents.clarity import clarity_agent
from app.agents.research import research_agent
from app.agents.synthesis import synthesis_agent
from app.agents.validator import validator_agent
from app.state import ResearchState

MAX_RESEARCH_ATTEMPTS = 3


# ── Special nodes ──────────────────────────────────────────────────────────────


def human_clarification_node(state: ResearchState) -> dict:
    """
    Pause the graph and ask the user to name a specific company.

    LangGraph's interrupt() saves graph state to the checkpointer and
    returns control to the caller. Execution resumes when the caller
    invokes the graph with Command(resume=<user_reply>).
    """
    question = state.get("clarification_question") or (
        "I need a bit more detail. Which specific company are you asking about?"
    )
    clarification = interrupt(question)
    return {"current_query": clarification}


def set_low_confidence_node(state: ResearchState) -> dict:
    """
    Flag that the synthesis agent should disclose limited data quality.

    This node is reached only when the validator repeatedly finds the
    research insufficient AND max retry attempts have been exhausted.
    Rather than silently returning poor results, we surface an honest
    disclosure note — a human-centric AI design principle.
    """
    return {"low_confidence": True}


# ── Routing functions ──────────────────────────────────────────────────────────


def route_clarity(state: ResearchState) -> str:
    """Route based on whether the query is specific enough to research."""
    return (
        "research_agent"
        if state["clarity_status"] == "clear"
        else "human_clarification"
    )


def route_research(state: ResearchState) -> str:
    """Route based on the research agent's self-reported confidence score."""
    return (
        "synthesis_agent"
        if state["confidence_score"] >= 6
        else "validator_agent"
    )


def route_validation(state: ResearchState) -> str:
    """
    Route based on validation result and retry count.

    Exits the retry loop (via set_low_confidence) once MAX_RESEARCH_ATTEMPTS
    is reached, preventing infinite loops and ensuring the user always gets
    a response — even if it comes with a quality caveat.
    """
    if state["validation_result"] == "sufficient":
        return "synthesis_agent"
    if state.get("research_attempts", 0) >= MAX_RESEARCH_ATTEMPTS:
        return "set_low_confidence"
    return "research_agent"


# ── Graph factory ──────────────────────────────────────────────────────────────


def build_graph(tools: list[BaseTool], checkpointer=None) -> Any:
    """
    Build and compile the multi-agent research graph.

    Tools are injected into the research node via functools.partial so the
    node signature stays compatible with LangGraph's (state) -> dict contract.
    """
    research_node = functools.partial(research_agent, tools=tools)

    builder = StateGraph(ResearchState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    builder.add_node("clarity_agent", clarity_agent)
    builder.add_node("human_clarification", human_clarification_node)
    builder.add_node("research_agent", research_node)
    builder.add_node("validator_agent", validator_agent)
    builder.add_node("set_low_confidence", set_low_confidence_node)
    builder.add_node("synthesis_agent", synthesis_agent)

    # ── Edges ───────────────────────────────────────────────────────────────
    builder.add_edge(START, "clarity_agent")

    builder.add_conditional_edges("clarity_agent", route_clarity)
    # After the user clarifies, re-evaluate clarity (handles still-vague replies)
    builder.add_edge("human_clarification", "clarity_agent")

    builder.add_conditional_edges("research_agent", route_research)
    builder.add_conditional_edges("validator_agent", route_validation)

    # set_low_confidence is a pass-through; always proceeds to synthesis
    builder.add_edge("set_low_confidence", "synthesis_agent")
    builder.add_edge("synthesis_agent", END)

    # Caller provides an async-compatible checkpointer (AsyncSqliteSaver in production).
    # Falls back to MemorySaver when none is supplied (tests, local dev without DB).
    return builder.compile(checkpointer=checkpointer or MemorySaver())
