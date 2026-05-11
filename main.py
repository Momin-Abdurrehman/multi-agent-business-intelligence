"""
Business Research Assistant — Chainlit entry point.

Run with: chainlit run main.py

UI ordering strategy (two-pass)
────────────────────────────────
Chainlit renders items in the order they are first sent to the client.
Streaming a message mid-pipeline races with step sends and produces
non-deterministic ordering. We solve this with a deliberate two-pass design:

  Pass 1 — graph.astream_events() runs the full pipeline.
            Agent outputs are recorded; synthesis tokens are buffered.
            A live status message shows the active agent while this runs.

  Pass 2 — after the graph completes, render in strict order:
            a) Collapsible nested pipeline steps  (sent first → appears first)
            b) Synthesis response streamed from buffer (sent second → appears second)

This guarantees the pipeline trace always sits above the response.

Interrupt flow
──────────────
If the Clarity Agent marks a query as needs_clarification, interrupt() pauses
the graph. astream_events() finishes early and get_state() reveals the interrupt.
We send the clarification prompt and store interrupted=True in the session.
The user's reply resumes the graph via Command(resume=...) on the same thread_id,
and the graph continues from exactly where it paused.
"""
from __future__ import annotations

import logging
import os
import uuid

import aiosqlite
import chainlit as cl
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.graph import build_graph
from app.state import ResearchState
from app.tools import get_search_tools

logging.basicConfig(level=logging.WARNING)

# Module-level singleton — created once on first session, shared across all users.
_checkpointer: AsyncSqliteSaver | None = None

_AGENT_NODES = {
    "clarity_agent", "research_agent", "validator_agent",
    "set_low_confidence", "synthesis_agent", "human_clarification",
}

_NODE_LABELS = {
    "clarity_agent": "Clarity Agent",
    "research_agent": "Research Agent",
    "validator_agent": "Validator Agent",
    "set_low_confidence": "Quality Check",
    "synthesis_agent": "Synthesis Agent",
    "human_clarification": "Awaiting clarification",
}


_WELCOME = """\
Ask me anything about a company — news, financials, leadership, strategy, competitors.

| Agent | Role |
|---|---|
| 🔍 **Clarity** | Checks whether the query names a specific company |
| 🌐 **Research** | Searches the web via Tavily MCP |
| ✅ **Validator** | Quality-gates findings; loops back up to 3× if needed |
| ✍️ **Synthesis** | Writes the final structured response |
"""


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> cl.User | None:
    """Validate login credentials from environment variables."""
    valid_user = os.environ.get("APP_USERNAME", "admin")
    valid_pass = os.environ.get("APP_PASSWORD", "demo1234")
    if username == valid_user and password == valid_pass:
        return cl.User(identifier=username, metadata={"role": "user"})
    return None


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialise a new session with its own thread_id and graph instance."""
    global _checkpointer
    if _checkpointer is None:
        conn = await aiosqlite.connect("conversations.db")
        _checkpointer = AsyncSqliteSaver(conn)
        await _checkpointer.setup()

    await cl.Message(content=_WELCOME).send()

    tools = await get_search_tools()
    graph = build_graph(tools, checkpointer=_checkpointer)

    # Namespace thread_id by user so each account has isolated history
    user = cl.user_session.get("user")
    username = user.identifier if user else "default"

    cl.user_session.set("graph", graph)
    cl.user_session.set("thread_id", f"{username}-{uuid.uuid4()}")
    cl.user_session.set("interrupted", False)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle each user message using the two-pass render strategy."""
    # ── Input validation ───────────────────────────────────────────────────────
    content = message.content.strip()
    if not content:
        await cl.Message(content="⚠️ Please enter a query.").send()
        return
    if len(content) > 2000:
        await cl.Message(
            content="⚠️ Query too long — please keep it under 2000 characters."
        ).send()
        return

    graph = cl.user_session.get("graph")
    thread_id = cl.user_session.get("thread_id")
    is_interrupted = cl.user_session.get("interrupted", False)
    config = {"configurable": {"thread_id": thread_id}}

    if is_interrupted:
        input_data: ResearchState | Command = Command(resume=content)
        cl.user_session.set("interrupted", False)
    else:
        input_data = {
            "messages": [HumanMessage(content=content)],
            "current_query": content,
            "company_name": "",
            "clarity_status": "",
            "research_findings": None,
            "confidence_score": 0.0,
            "validation_result": None,
            "research_attempts": 0,
            "low_confidence": False,
            "clarification_question": "",
        }

    # ── Pass 1: run the graph, collect trace + buffer synthesis tokens ─────
    status = cl.Message(content="⏳ Starting…")
    await status.send()

    agent_trace: list[tuple[str, dict]] = []  # completed (node_name, output) pairs
    synthesis_tokens: list[str] = []          # buffered tokens for streaming in pass 2

    graph_error: Exception | None = None
    try:
        async for event in graph.astream_events(input_data, config, version="v2"):
            kind = event["event"]
            name = event.get("name", "")
            node = event.get("metadata", {}).get("langgraph_node", "")

            if kind == "on_chain_start" and name in _AGENT_NODES:
                status.content = f"⏳ **{_NODE_LABELS.get(name, name)}** running…"
                await status.update()

            elif kind == "on_chain_end" and name in _AGENT_NODES:
                output = event.get("data", {}).get("output", {})
                agent_trace.append((name, output if isinstance(output, dict) else {}))

            elif kind == "on_chat_model_stream" and node == "synthesis_agent":
                token = getattr(event["data"]["chunk"], "content", "")
                if token:
                    synthesis_tokens.append(token)

    except Exception as exc:
        graph_error = exc
        logging.error("Graph execution error: %s", exc, exc_info=True)
    finally:
        await status.remove()

    if graph_error:
        await cl.Message(content=f"⚠️ Pipeline error: `{graph_error}`").send()
        return

    # ── Interrupt check ────────────────────────────────────────────────────
    snapshot = await graph.aget_state(config)
    if snapshot.next:
        cl.user_session.set("interrupted", True)
        interrupt_value: str = snapshot.tasks[0].interrupts[0].value
        await cl.Message(content=f"❓ **{interrupt_value}**").send()
        return

    # ── Pass 2a: pipeline trace (rendered first → sits above response) ─────
    # Nested async-with blocks are automatically parent-linked by Chainlit.
    async with cl.Step(name="Agent Pipeline", type="run") as pipeline:
        pipeline.input = message.content
        for node_name, output in agent_trace:
            async with cl.Step(
                name=_NODE_LABELS.get(node_name, node_name), type="tool"
            ) as step:
                for key in (
                    "clarity_status", "validation_result",
                    "confidence_score", "low_confidence",
                ):
                    if key in output:
                        step.output = f"{key}: {output[key]}"
                        break
                else:
                    step.output = "✓"
        pipeline.output = "Complete"

    # ── Pass 2b: stream synthesis response (rendered second → sits below) ──
    if synthesis_tokens:
        final_msg = cl.Message(content="")
        for token in synthesis_tokens:
            await final_msg.stream_token(token)
        await final_msg.send()
    else:
        # Fallback: streaming events didn't fire — read final AI message from state
        state_values = (await graph.aget_state(config)).values
        messages = state_values.get("messages", [])
        if messages and hasattr(messages[-1], "content") and messages[-1].content:
            await cl.Message(content=messages[-1].content).send()
