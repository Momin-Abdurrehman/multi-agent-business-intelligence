"""
Synthesis Agent — turns validated research findings into a polished user response.

Free-form markdown output (no structured output) so the response reads naturally.
Tokens are streamed to the Chainlit UI in real time via on_chat_model_stream events.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.state import ResearchState

_SYSTEM_PROMPT = """You are the Synthesis Agent — the final step in a business research \
pipeline. Your job is to turn research findings into a clear, direct response that \
answers the user's specific question.

## Output rules
- Use markdown: ## headers, bullet points, **bold** for key facts
- Be factual — only state information present in the research findings
- Target 150–400 words. Be concise; the user can ask for more detail
- For follow-up questions, answer only the new aspect asked — do not re-summarise \
everything already covered in the conversation

## When research found nothing specific about the company
If the findings state that no relevant results were found, respond with this and nothing else:
> ⚠️ I couldn't find reliable information about **{company}**.
> It may be very new, private, or not yet indexed online.
>
> Suggested next steps:
> - **LinkedIn** — search for the company profile and team
> - **Crunchbase** — startup funding and founding data
> - **Their official website** — if you have the URL

Do not fill the gap with generic industry background. Honest and brief beats \
confident and wrong.

## When [LOW_CONFIDENCE] appears in the message
Open your response with:
> ⚠️ After multiple search attempts, available data on this company is limited.
> Treat the following as a best-effort summary and verify key details independently.

## Tone
Professional but conversational. Write for a business analyst who values precision.

## Close every response with:
---
*Want to explore a specific aspect further? Just ask.*"""


_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)


def synthesis_agent(state: ResearchState) -> dict:
    """Generate a structured final answer from the validated research findings."""
    recent_messages = state["messages"][-8:]
    history_text = "\n".join(
        f"{m.type.upper()}: {m.content}" for m in recent_messages
    )

    # Inject a clean signal rather than a verbose paragraph — the system prompt
    # explains what [LOW_CONFIDENCE] means so the model acts on it correctly.
    low_confidence_flag = "\n\n[LOW_CONFIDENCE]" if state.get("low_confidence") else ""

    prompt = (
        f"Conversation history:\n{history_text}\n\n"
        f"Current query: {state['current_query']}\n\n"
        f"Research findings:\n{state['research_findings']}"
        f"{low_confidence_flag}\n\n"
        "Answer the current query directly and concisely."
    )

    # Fill the {company} placeholder so the model never outputs it literally
    system = _SYSTEM_PROMPT.replace("{company}", state["current_query"])

    response = _llm.invoke(
        [SystemMessage(content=system), HumanMessage(content=prompt)]
    )
    return {"messages": [AIMessage(content=response.content)]}
