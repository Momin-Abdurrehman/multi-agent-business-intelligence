# AI Prompts Log & Reasoning

This document satisfies the assignment requirement to detail AI assistance used
and the reasoning behind key decisions.

---

## 1. Project Architecture

**Prompt used:**
> Design a production-grade multi-agent LangGraph system for business research.
> Requirements: free LLM, Tavily MCP search, Chainlit UI, Pydantic structured outputs,
> LangSmith observability, uv packaging, and a Human-in-the-Loop interrupt pattern.

**Reasoning:**
Rather than just meeting the spec, the goal was to demonstrate AI Engineering
proficiency — type-safe state management, observability from day one, and modern
tooling choices that reflect real 2025/2026 production practices.

---

## 2. System Prompts

### Clarity Agent (`app/agents/clarity.py`)

```
You are a Clarity Agent for a business research assistant.
Your sole job is to decide whether the user's query is specific enough to act on.

A query is CLEAR when:
- A specific company name is mentioned
- It is a follow-up that clearly refers to a company already discussed

A query NEEDS_CLARIFICATION when:
- No company name is mentioned or implied
- The query is too vague to research

Study the full conversation history carefully — follow-up questions that use
pronouns like "they", "their", or "it" are often CLEAR when context exists.
```

**Reasoning:**
- `temperature=0` for deterministic classification (no hallucinated ambiguity).
- `.with_structured_output(ClarityOutput)` eliminates JSON parsing fragility; Pydantic
  enforces the `Literal["clear", "needs_clarification"]` constraint at the type level.
- Last 6 messages provided as context so follow-ups like "What about their CEO?" are
  correctly classified as CLEAR when a company was discussed earlier.

---

### Research Agent (`app/agents/research.py`)

```
You are a Research Agent specialising in business intelligence.
You have received search results about a company. Synthesise them into a structured report:
## Company Overview / ## Recent News / ## Financial Highlights / ## Leadership / ...
Assign a confidence_score (0–10): 8–10 rich data, 5–7 partial, 0–4 vague.
Be honest about the score. It drives whether the Validator will request a retry.
```

**Reasoning:**
- `temperature=0.2` allows slight creativity in synthesizing findings while staying grounded.
- Explicit confidence score bands (rather than "rate from 0-10") give the LLM concrete
  anchors, producing more consistent scores.
- Prior findings are included on retry runs so the agent builds on previous work rather
  than duplicating searches.
- The Tavily MCP tool is called directly (not via LLM tool-calling loop) — cleaner,
  faster, and more predictable for a pipeline architecture.

---

### Validator Agent (`app/agents/validator.py`)

```
You are a Validator Agent acting as a strict, sceptical quality reviewer.
SUFFICIENT: findings are specific, factual, genuinely useful.
INSUFFICIENT: vague, off-topic, or clearly misses what the user asked.
Be honest and strict. Do not mark as sufficient just because some text was returned.
```

**Reasoning:**
- Adversarial "Judge" framing ("be strict") consistently improves LLM evaluation quality
  vs. neutral framing in research literature.
- `temperature=0` ensures reproducible validation — the same findings always get the
  same verdict.
- Receiving the research agent's self-reported confidence score as additional signal
  lets the validator calibrate against an internal quality measure.

---

### Synthesis Agent (`app/agents/synthesis.py`)

```
You are a Synthesis Agent. Generate a clear, well-structured response.
Use markdown: ## headers, bullet points, **bold** for key facts.
Be factual — only include information present in the research findings.
If low_confidence is noted, open with the quality disclosure note.
Close with: "Want to explore a specific aspect further? Just ask."
```

**Reasoning:**
- `temperature=0.3` allows natural language variation while staying grounded in facts.
- Explicit markdown formatting instructions produce consistently readable output in Chainlit.
- The `low_confidence` disclosure note is a **human-centric AI** design choice: being
  transparent about data limitations builds user trust rather than presenting poor
  findings as authoritative.
- The closing prompt encourages multi-turn conversation, fulfilling the spec requirement.

---

## 3. Key Architectural Decisions

### Why Groq (Llama 3.3 70B)?
Free tier with generous rate limits, sub-second inference, and Llama 3.3 70B is
competitive with GPT-4o on instruction-following. Enables development and demo
without API costs.

### Why Tavily MCP over direct SDK?
The assignment explicitly preferred Tavily MCP. The Model Context Protocol is the
emerging standard (2025/2026) for decoupling tool logic from agent logic — it makes
tools composable and swappable without changing agent code. We use the remote HTTP
endpoint to avoid a Node.js dependency; a direct-SDK fallback ensures robustness.

### Why Chainlit over Gradio or Streamlit?
Chainlit is purpose-built for LLM agent UIs. The `LangchainCallbackHandler` creates
collapsible agent steps automatically from LangGraph's callback events — making the
multi-agent pipeline visible to the user without any custom CSS. This is the key
feature for demonstrating the system's architecture in the Loom recording.

### Why Pydantic V2 structured outputs?
`llm.with_structured_output(PydanticModel)` eliminates brittle JSON parsing.
If the LLM returns unexpected output, Pydantic raises a typed validation error at the
boundary — not a cryptic `KeyError` deep in business logic. All three classification
agents (Clarity, Research, Validator) use this pattern.

### Why MemorySaver?
Sufficient for demo purposes: in-memory, zero config, supports both `interrupt()`
and `add_messages` accumulation. A production deployment would use `SqliteSaver` or
`PostgresSaver` for cross-session persistence. This is noted in a code comment.

### Why LangSmith?
Zero-code observability — two environment variables and every agent invocation is
traced automatically. The trace URL can be included in the submission to demonstrate
production debugging practices. Free tier is sufficient for a demo workload.

### Why uv + pyproject.toml?
`uv` is the fastest Python package manager in 2025/2026 and signals familiarity with
modern Python tooling. `pyproject.toml` is the PEP 517/518 standard; `requirements.txt`
is legacy. No functional difference for this project, but a clear signal of craft.

---

## 4. Human-in-the-Loop Design

The interrupt pattern is the centrepiece of the assignment spec. Implementation:

1. `human_clarification_node` calls `langgraph.types.interrupt(message)` — this
   saves the entire graph state to the MemorySaver checkpointer and returns control
   to the caller.
2. `main.py` detects the pause via `graph.get_state(config).next` being non-empty.
3. The interrupt message is displayed to the user in Chainlit.
4. The user's reply is sent back via `graph.invoke(Command(resume=reply), config)`.
5. The graph resumes from exactly the point it paused — no state is lost.
6. After clarification, the graph routes back through the Clarity Agent to re-evaluate
   the new query (handles the case where the user is still vague).

---

## 5. LangSmith Trace

**Project dashboard:** https://smith.langchain.com/o/a2e055ee-ddad-4fe2-9581-8e8151fbf8ec/projects/p/c74e7251-d8d3-48a4-919d-a4059f99f9da

The LangSmith project captures every run automatically via the `LANGCHAIN_TRACING_V2=true` environment variable — no code changes required. Each run shows the full agent execution tree: node invocations, LLM calls, token counts, and latency per step.
