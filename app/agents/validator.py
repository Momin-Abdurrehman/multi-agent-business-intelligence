"""
Validator Agent — quality-gates the research before it reaches the user.

Uses an adversarial "Judge" prompt: the LLM is instructed to be a strict
sceptic so it only marks findings as sufficient when they genuinely are.
Temperature 0 ensures consistent, reproducible validation decisions.
"""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.state import ResearchState, ValidationOutput

logger = logging.getLogger(__name__)

_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0).with_structured_output(ValidationOutput)

_SYSTEM_PROMPT = """You are the Validator Agent in a business research pipeline. You \
decide whether the research findings are good enough to send to the user, or whether \
another search attempt should be made.

## Process — follow these steps in order

1. Identify the exact company or entity the user asked about.
2. Check whether the findings discuss that specific entity by name with real detail.
3. Look at the confidence_score the Research Agent assigned.
4. Apply the rules below.

## Mark SUFFICIENT when ANY of these is true:
- The findings contain at least some specific, named facts about the queried entity
  (e.g. founder names, funding amounts, product names, founding year, headquarters location)
- The findings meaningfully address what the user asked, even if some gaps exist
- The confidence_score is 6 or above

## Mark INSUFFICIENT when:
- The findings discuss a generic concept or unrelated companies instead of the queried entity
  ✗ Explaining what "merchants" are when asked about a company called "Merchaint"
- The findings contain no specific facts about the entity AND confidence_score is below 4
- The Research Agent explicitly stated no relevant results were found

## Calibrating for company type
Large public companies (Apple, Tesla) have abundant data — expect comprehensive findings.
Small, private, or newly founded companies will naturally have less — brief but specific \
findings are still SUFFICIENT. Do not retry just because findings are short.
Only retry when findings are genuinely off-topic or empty.

## Important
A retry searches the same web. If findings are thin because the company has little public \
presence, retrying will not help. Mark SUFFICIENT and let Synthesis handle the limitation \
honestly.

Your "reason" must be one sentence stating the specific fact (or absence) that drove \
your decision."""


def validator_agent(state: ResearchState) -> dict:
    """Evaluate research quality and decide whether to accept or request a retry."""
    prompt = (
        f"User query: {state['current_query']}\n\n"
        f"Research findings:\n{state['research_findings']}\n\n"
        f"Research agent self-reported confidence: {state['confidence_score']}/10\n\n"
        "Is this research sufficient to answer the query?"
    )

    try:
        result: ValidationOutput = _llm.invoke(
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        return {"validation_result": result.validation_result}
    except Exception as exc:
        # Fail safe: accept findings rather than looping indefinitely
        logger.warning("Validator agent error: %s — defaulting to sufficient.", exc)
        return {"validation_result": "sufficient"}
