"""
Research Agent — searches for company information and scores its own confidence.

Calls the Tavily search tool directly (injected via functools.partial from graph.py)
then asks the LLM to synthesize findings and assign a 0–10 confidence score.
On retry runs, prior findings are included so the agent builds on previous work
rather than starting from scratch.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI

from app.state import ResearchOutput, ResearchState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the Research Agent in a business intelligence pipeline. You \
receive web search results and synthesise them into a structured report about a specific \
company or entity.

## Primary rule
Write ONLY about the entity named in the query. If the search results discuss a different \
company, a generic concept, or industry trends without mentioning the queried entity \
directly — do not include that content.

## Exception: comparison queries
If the query explicitly compares two or more companies (e.g. "how does X compare to Y", \
"X vs Y revenue"), include data for ALL named companies. Filtering out the comparison \
company makes the answer useless.

## Report structure
Include only sections for which you have real data. Omit any section entirely if you \
have nothing specific to say — do not write "data not available" under a heading.

Sections to use as needed:
- **Overview** — what the company does, when founded, where based
- **Recent News & Developments** — notable events, product launches, announcements
- **Financials** — revenue, funding rounds, market cap, growth (only if specific figures exist)
- **Leadership** — founders and key executives by name
- **Products & Competitive Position** — core offerings, market position, named competitors
- **Additional Insights** — anything else directly relevant to the query

## When search results do not cover the queried entity
If the results contain nothing specific about the company asked about, write exactly:
"No relevant results found for [company name]. Search returned results about \
[brief description of what was returned] instead."
Then set confidence_score to 1.

## Confidence score (0–10) — score conservatively
The score signals quality to the Validator. Inflating it locks in poor findings.

- 9–10  Multiple sources, specific figures, names, and dates — fully answers the query
- 7–8   Good coverage with minor gaps
- 5–6   Useful but notable gaps; some aspects of the query not addressed
- 3–4   Superficial or partially relevant; little specific detail
- 1–2   Almost nothing useful; results are off-topic or about a different entity
- 0     Search tool failed or returned empty

A conservative score that triggers one good retry is better than an optimistic score \
that locks in weak findings.

## Length
200–400 words. Specific and factual. No padding with general industry context."""

_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2).with_structured_output(ResearchOutput)
_query_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)


def _build_search_query(
    query: str,
    messages: list,
    attempt: int = 1,
    prior_findings: str = "",
) -> str:
    """Generate a self-contained Tavily search query from conversation context.

    On retries, instructs the LLM to use a different search angle so each
    attempt brings genuinely new information rather than repeating the same query.
    """
    history = "\n".join(f"{m.type.upper()}: {m.content}" for m in messages[-4:])
    retry_note = ""
    if attempt > 1 and prior_findings:
        retry_note = (
            f"\n\nPrevious attempt found: {prior_findings[:400]}\n"
            "Generate a DIFFERENT search query — vary the keywords or try a related angle."
        )
    prompt = (
        f"Conversation:\n{history}\n\n"
        f"User query: {query}{retry_note}\n\n"
        "Write a single web search query (8 words max) to find this information.\n"
        "Rules:\n"
        "- Include the company name even if the user used pronouns or ellipsis\n"
        "- Resolve the company from conversation history if not stated in the query\n"
        "- Add '2025' for news, financials, earnings, or time-sensitive topics\n"
        "- Return ONLY the search query — no explanation, no quotes"
    )
    try:
        result = _query_llm.invoke([HumanMessage(content=prompt)])
        return result.content.strip().strip('"').strip("'")
    except Exception:
        return query  # safe fallback


def research_agent(state: ResearchState, tools: list[BaseTool]) -> dict:
    """Search for company data and synthesise findings with a confidence score."""
    search_tool: BaseTool | None = next(
        (t for t in tools if "search" in t.name.lower()), None
    )

    query = state["current_query"]
    attempts = state.get("research_attempts", 0) + 1
    prior_findings = state.get("research_findings") or ""

    # --- Generate a self-contained, context-aware search query ---------------
    search_query = _build_search_query(
        query, state["messages"], attempt=attempts, prior_findings=prior_findings
    )

    # --- Run the search ------------------------------------------------------
    search_text = _run_search(search_tool, search_query)

    # Include prior findings on retry so the LLM can fill gaps
    prior_section = ""
    if prior_findings:
        prior_section = f"\n\nPrevious research attempt (attempt {attempts - 1}):\n{prior_findings}"

    prompt = (
        f"Research query: {query}\n\n"
        f"Search results:\n{search_text}"
        f"{prior_section}"
    )

    # --- LLM synthesis ---------------------------------------------------
    try:
        result: ResearchOutput = _llm.invoke(
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        return {
            "research_findings": result.findings,
            "confidence_score": result.confidence_score,
            "research_attempts": attempts,
        }
    except Exception as exc:
        logger.error("Research agent LLM error: %s", exc)
        # Return raw search text with a low confidence so the validator retries
        return {
            "research_findings": search_text,
            "confidence_score": 3.0,
            "research_attempts": attempts,
        }


def _run_search(tool: BaseTool | None, query: str) -> str:
    """Invoke the search tool and normalise the result to a plain string.

    MCP tools from langchain-mcp-adapters are async-only (StructuredTool).
    Since this function runs inside a thread-pool executor (via cl.make_async),
    there is no running event loop in this thread, so asyncio.run() is safe.
    """
    import asyncio

    if tool is None:
        return "Search tool unavailable."
    try:
        raw: Any = asyncio.run(tool.ainvoke({"query": query}))
        if isinstance(raw, list):
            return "\n\n".join(
                f"Source: {r.get('url', 'N/A')}\n{r.get('content', str(r))}"
                if isinstance(r, dict)
                else str(r)
                for r in raw
            )
        return str(raw)
    except Exception as exc:
        logger.error("Search tool error: %s", exc)
        return "Search returned no results."
