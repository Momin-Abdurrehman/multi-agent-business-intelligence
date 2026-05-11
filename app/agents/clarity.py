"""
Clarity Agent — determines whether the user's query is specific enough to research.

Uses Pydantic structured output to guarantee a typed response with no JSON parsing.
Temperature 0 ensures deterministic classification.
"""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.state import ClarityOutput, ResearchState

logger = logging.getLogger(__name__)

_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0).with_structured_output(ClarityOutput)

_SYSTEM_PROMPT = """You are the Clarity Agent in a business research pipeline. Your only \
job is to decide whether the user's request identifies a specific company or named entity \
that can be researched.

## Process — follow these steps in order

1. Read the full conversation history.
2. Ask: does the current query name a specific company, brand, startup, or person?
3. Ask: if not, does it refer to one already named earlier in the conversation?
4. Apply the rules below and give your decision.

## Mark CLEAR when:
- A specific company, brand, startup, or organisation is named — even if you have never
  heard of it. Unfamiliar or obscure names are valid targets.
  ✓ "Tell me about Apple"
  ✓ "What is Riverstone Holdings?"
  ✓ "Explain Merchaint's business model"
- The query is a follow-up using pronouns (they/their/it/its) and a company was already
  named earlier in the conversation.
  ✓ "What about their CEO?" — after discussing Microsoft
  ✓ "Is it profitable?" — after discussing Tesla
- The query names a person rather than a company — research the person.
  ✓ "Tell me about Elon Musk"

## Mark NEEDS_CLARIFICATION (ambiguous name) when:
- The name in the query is a common word or short name that refers to multiple distinct,
  well-known companies or entities (e.g. "Mercury", "Aurora", "Titan", "Genesis", "Apex").
- Only trigger this when you can confidently name at least 2 distinct well-known companies
  that share the name. Do NOT trigger for obscure or unique names.
- When triggered, set clarification_question to a specific question that names the most
  likely candidates. Example:
  "Are you asking about Mercury the fintech banking platform, Mercury Gate the logistics
  company, or the Ford Mercury automotive brand?"

## Mark NEEDS_CLARIFICATION (no entity) when:
- No company or person name appears anywhere in the query or conversation history.
  ✗ "Tell me about a big tech company"
  ✗ "Which startup should I invest in?"
  ✗ "What do you think about that company?" with no prior company mentioned
- The query names only an industry or concept, not a specific entity.
  ✗ "Tell me about fintech in Singapore"
  ✗ "What is private equity?"
- For these cases, set clarification_question to:
  "Which specific company are you asking about?"

## Defaults
- If you are uncertain, choose CLEAR. A search that finds nothing is better than \
blocking a valid user request.
- Your "reason" must be exactly one sentence stating which specific name was found, \
which were the ambiguous candidates, or what was missing."""


def clarity_agent(state: ResearchState) -> dict:
    """Assess whether the current query is specific enough to research."""
    # Include the last 6 messages so the agent understands follow-up context
    recent_messages = state["messages"][-6:]
    history_text = "\n".join(
        f"{m.type.upper()}: {m.content}" for m in recent_messages
    )

    prompt = (
        f"Conversation history:\n{history_text}\n\n"
        f"Evaluate this query: {state['current_query']}"
    )

    try:
        result: ClarityOutput = _llm.invoke(
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        return {
            "clarity_status": result.clarity_status,
            "clarification_question": result.clarification_question,
        }
    except Exception as exc:
        # Fail safe: assume clear so a parsing error doesn't block an obvious query
        logger.warning("Clarity agent error: %s — defaulting to clear.", exc)
        return {"clarity_status": "clear", "clarification_question": ""}
