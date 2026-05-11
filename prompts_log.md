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
You are the Clarity Agent in a business research pipeline. Your job is to decide
whether the user's request identifies a specific company or named entity that can
be researched, and to extract the canonical company name.

## Process — follow these steps in order

1. Read the full conversation history.
2. Ask: does the current query name a specific company, brand, startup, or person?
3. Ask: if not, does it refer to one already named earlier in the conversation?
4. Apply the rules below and give your decision.
5. Always fill company_name with the canonical name of the entity being researched.

## Mark CLEAR when:
- A specific company, brand, startup, or organisation is named — even if you have
  never heard of it. Unfamiliar or obscure names are valid targets.
- The query is a follow-up and a company was already named earlier in the conversation.
  Resolve the company from history and set company_name accordingly.
- The query names a person rather than a company — research the person.

## Mark NEEDS_CLARIFICATION (ambiguous name) when:
- The name refers to multiple distinct, well-known companies (e.g. "Mercury", "Aurora").
- Only trigger when you can confidently name at least 2 distinct well-known companies
  sharing that name. Set clarification_question naming the most likely candidates.

## Mark NEEDS_CLARIFICATION (no entity) when:
- No company or person name appears anywhere in the query or conversation history.
- The query names only an industry or concept, not a specific entity.
- Set clarification_question to: "Which specific company are you asking about?"

## Defaults
- If uncertain, choose CLEAR. A search that finds nothing is better than blocking
  a valid user request.
```

**Reasoning:**
- `temperature=0` for deterministic classification.
- `.with_structured_output(ClarityOutput)` enforces `Literal["clear", "needs_clarification"]`
  at the type level — no JSON parsing fragility.
- Last 6 messages provided as context so follow-ups using pronouns ("their CEO?", "is it
  profitable?") resolve correctly to the company in history.
- `company_name` field extracted here so the Synthesis Agent always has a clean display
  name (e.g. "Apple") even when the search query was reformulated to "Apple profit 2025".
- Fail-safe defaults to `"clear"` — a failed search is recoverable; a blocked valid query
  is not.

---

### Research Agent (`app/agents/research.py`)

**Search query generation sub-prompt (internal, called before Tavily):**

```
Conversation:
{last 4 messages}

User query: {current_query}
[On retry: Previous findings (which were insufficient): {prior_findings[:400]}
Generate a DIFFERENT search query to find new information.]

Write a single web search query (8 words max) to find this information.
Rules:
- Include the company name even if the user used pronouns
- Resolve the company from conversation history if not stated in the query
- Add '2025' for news, financials, earnings, or time-sensitive topics
- Return ONLY the search query — no explanation, no quotes
```

**Synthesis system prompt:**

```
You are the Research Agent in a business intelligence pipeline. You receive web
search results and synthesise them into a structured report about a specific
company or entity.

## Primary rule
Write ONLY about the entity named in the query. If the search results discuss a
different company, a generic concept, or industry trends without mentioning the
queried entity directly — do not include that content.

## Report structure
Include only sections for which you have real data. Omit any section entirely if
you have nothing specific to say.
Sections: Overview / Recent News & Developments / Financials / Leadership /
Products & Competitive Position / Additional Insights

## When search results do not cover the queried entity
Write exactly: "No relevant results found for [company name]. Search returned
results about [brief description] instead." Then set confidence_score to 1.

## Confidence score (0–10) — score conservatively
- 9–10  Multiple sources, specific figures, names, and dates
- 7–8   Good coverage with minor gaps
- 5–6   Useful but notable gaps
- 3–4   Superficial or partially relevant
- 1–2   Almost nothing useful; off-topic results
- 0     Search tool failed or returned empty

## Length: 200–400 words. Specific and factual. No padding.
```

**Reasoning:**
- The research agent owns its own search query generation — this is the right layer
  (the searcher should control the search query, not the classifier).
- On retry, `_build_search_query()` receives prior findings and is explicitly instructed
  to use a different angle — ensuring each attempt brings genuinely new information.
- `temperature=0.2` allows slight creativity in synthesising findings while staying grounded.
- Explicit confidence score bands give concrete anchors, producing more consistent scores
  than an open-ended "rate 0–10" instruction.
- Tavily is called directly via `asyncio.run(tool.ainvoke(...))` (not via an LLM tool-use
  loop) — cleaner, faster, and more predictable for a pipeline architecture.

---

### Validator Agent (`app/agents/validator.py`)

```
You are the Validator Agent in a business research pipeline. You decide whether
the research findings are good enough to send to the user, or whether another
search attempt should be made.

## Process
1. Identify the exact company or entity the user asked about.
2. Check whether the findings discuss that specific entity by name with real detail.
3. Look at the confidence_score the Research Agent assigned.
4. Apply the rules below.

## Mark SUFFICIENT when ANY of these is true:
- The findings contain at least some specific, named facts about the queried entity
  (founder names, funding amounts, product names, founding year, headquarters location)
- The findings meaningfully address what the user asked, even if some gaps exist
- The confidence_score is 6 or above

## Mark INSUFFICIENT when:
- The findings discuss a generic concept or unrelated companies instead of the entity
- The findings discuss a non-company entity (planet, element) when a company was queried
- The findings contain no specific facts AND confidence_score is below 4
- The Research Agent explicitly stated no relevant results were found

## Calibrating for company type
Large public companies (Apple, Tesla) have abundant data — expect comprehensive findings.
Small, private, or newly founded companies will naturally have less — brief but specific
findings are still SUFFICIENT. Only retry when findings are genuinely off-topic or empty.

## Important
A retry searches the same web. If findings are thin because the company has little
public presence, retrying will not help. Mark SUFFICIENT and let Synthesis handle
the limitation honestly.
```

**Reasoning:**
- Adversarial "Judge" framing ("be strict") consistently improves LLM evaluation quality
  vs. neutral framing — well-documented in LLM-as-judge research.
- `temperature=0` ensures reproducible validation — same findings always get same verdict.
- Explicit examples of INSUFFICIENT (planet Mercury, generic merchant content) prevent the
  validator from accepting off-topic results that happen to contain relevant-looking words.
- The "calibrate for company type" instruction prevents over-retrying for obscure startups,
  which would loop indefinitely since the web has little data on them regardless.
- Fail-safe defaults to `"sufficient"` — prevents the retry loop running past max attempts
  on a validator error.

---

### Synthesis Agent (`app/agents/synthesis.py`)

```
You are the Synthesis Agent — the final step in a business research pipeline.
Your job is to turn research findings into a clear, direct response that answers
the user's specific question.

## Output rules
- Use markdown: ## headers, bullet points, **bold** for key facts
- Be factual — only state information present in the research findings
- Target 150–400 words. Be concise; the user can ask for more detail
- For follow-up questions, answer only the new aspect asked — do not re-summarise
  everything already covered in the conversation

## When research found nothing specific about the company
If the findings state that no relevant results were found, respond with:
> ⚠️ I couldn't find reliable information about **{company}**.
> It may be very new, private, or not yet indexed online.
> Suggested next steps: LinkedIn / Crunchbase / Their official website

## When [LOW_CONFIDENCE] appears in the message
Open your response with:
> ⚠️ After multiple search attempts, available data on this company is limited.
> Treat the following as a best-effort summary and verify key details independently.

## Tone
Professional but conversational. Write for a business analyst who values precision.

## Close every response with:
---
*Want to explore a specific aspect further? Just ask.*
```

**Reasoning:**
- `temperature=0.3` allows natural language variation while staying grounded in facts.
- `{company}` is filled at runtime with `state["company_name"]` (extracted by the Clarity
  Agent) — not `current_query` — so the not-found message always shows "Apple" rather than
  the reformulated search phrase "Apple profit revenue 2025".
- `[LOW_CONFIDENCE]` is injected as a flag in the human message (not a paragraph in the
  system prompt) — cleaner signal that the model acts on reliably.
- The closing prompt encourages multi-turn conversation, fulfilling the spec requirement.
- Conversation history (last 8 messages) is provided so follow-up responses don't
  re-summarise what was already covered.

---

## 3. Key Architectural Decisions

### Why Google Gemini 2.0 Flash?
Free tier with no hard rate limits for development, sub-second inference, and strong
instruction-following. The project originally used Groq (Llama 3.3 70B) but hit the
100k tokens/day limit during development. Gemini 2.0 Flash provides comparable quality
with higher free-tier headroom. The LLM is abstracted behind LangChain's `ChatGoogleGenerativeAI`
— swapping models requires changing one line.

### Why Research Agent Owns Query Generation?
Follow-up queries like "what about its profit?" contain no company name. The naive approach
is to reformulate the query in the Clarity Agent — but this conflates two jobs (classify
intent vs. generate search terms) and can't improve on retry. Instead, the Research Agent
calls `_build_search_query()` before every Tavily search: it sees the full conversation
history and generates a context-aware query. On retries, it receives prior findings and
generates a different angle. This handles all follow-up patterns (pronouns, ellipsis,
implicit references, multi-entity comparisons) without enumerating them.

### Why Tavily MCP over direct SDK?
The assignment explicitly preferred Tavily MCP. The Model Context Protocol is the
emerging standard (2025/2026) for decoupling tool logic from agent logic — it makes
tools composable and swappable without changing agent code. We use the remote HTTP
endpoint to avoid a Node.js dependency; a direct-SDK fallback ensures robustness.

### Why Chainlit over Gradio or Streamlit?
Chainlit is purpose-built for LLM agent UIs. `cl.Step` creates collapsible agent steps
natively, making the multi-agent pipeline visible to the user without any custom CSS.
Built-in `@cl.password_auth_callback` handles authentication. Token streaming is a
first-class feature via `cl.Message.stream_token()`.

### Why Two-Pass Rendering?
Chainlit renders items in the order they are first sent. Streaming synthesis tokens while
simultaneously sending agent steps produces non-deterministic ordering. The two-pass
design solves this: Pass 1 runs the full pipeline via `graph.astream_events()`, buffering
synthesis tokens. Pass 2 sends agent steps first (they always appear above), then streams
the buffered response. Pipeline trace always sits above the answer.

### Why AsyncSqliteSaver?
`MemorySaver` loses all state on server restart. `AsyncSqliteSaver` (from
`langgraph-checkpoint-sqlite`) persists every agent step to `conversations.db` with
zero infrastructure — no separate database process needed. The async variant is required
because Chainlit's event loop calls `graph.aget_state()` and the checkpointer must
support async access. The module-level singleton pattern (`_checkpointer`) ensures one
DB connection is shared across all sessions.

### Why Pydantic V2 Structured Outputs?
`llm.with_structured_output(PydanticModel)` eliminates brittle JSON parsing.
If the LLM returns unexpected output, Pydantic raises a typed validation error at the
boundary — not a cryptic `KeyError` deep in business logic. All three classification
agents (Clarity, Research, Validator) use this pattern.

### Why uv + pyproject.toml?
`uv` is the fastest Python package manager in 2025/2026 and signals familiarity with
modern Python tooling. `pyproject.toml` is the PEP 517/518 standard. Optional dev
dependencies (`pytest`) are kept separate under `[project.optional-dependencies]` and
installed with `uv sync --extra dev`.

---

## 4. Human-in-the-Loop Design

The interrupt pattern is the centrepiece of the assignment spec. Implementation:

1. `human_clarification_node` calls `langgraph.types.interrupt(question)` — this saves
   the entire graph state to the checkpointer and returns control to the caller.
2. `main.py` detects the pause via `await graph.aget_state(config)` — `.next` is non-empty
   when interrupted. (`aget_state` is required; the sync `get_state` silently fails with
   an async checkpointer.)
3. The interrupt message (a specific disambiguation question from the Clarity Agent) is
   displayed to the user in Chainlit.
4. The user's reply is sent back via `Command(resume=reply)` on the same `thread_id`.
5. The graph resumes from exactly the point it paused — no state is lost.
6. After clarification, the graph routes back through the Clarity Agent to re-evaluate
   the new query (handles the case where the user is still vague).

---

## 5. LangSmith Trace

**Project dashboard:** https://smith.langchain.com/o/a2e055ee-ddad-4fe2-9581-8e8151fbf8ec/projects/p/c74e7251-d8d3-48a4-919d-a4059f99f9da

The LangSmith project captures every run automatically via the `LANGCHAIN_TRACING_V2=true`
environment variable — no code changes required. Each run shows the full agent execution
tree: node invocations, LLM calls, token counts, and latency per step.
