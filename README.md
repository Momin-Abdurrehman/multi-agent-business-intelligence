# Business Research Assistant

A production-grade multi-agent research assistant built with LangGraph, Groq, Tavily, and Chainlit.

## Architecture

```
User Query
    │
    ▼
┌───────────────┐     vague      ┌──────────────────┐
│ Clarity Agent │ ─────────────► │ Human Clarif.    │ ◄─ interrupt()
│               │                │ (HITL interrupt) │    user replies
└───────┬───────┘                └────────┬─────────┘
        │ clear                           │ resume
        │ ◄───────────────────────────────┘
        ▼
┌───────────────┐
│ Research Agent│  (Tavily MCP search → LLM synthesis → confidence 0–10)
└───────┬───────┘
        │ confidence ≥ 6                  confidence < 6
        ├────────────────────────────────► ┌──────────────────┐
        │                                  │ Validator Agent  │
        │                                  └────────┬─────────┘
        │                        sufficient         │         insufficient
        │ ◄─────────────────────────────────────────┤         AND attempts < 3
        │                                           │              │
        │                                           │         ┌────▼──────────┐
        │                                           │         │ research_agent│ (retry)
        │                                           │         └───────────────┘
        │                          max attempts     │
        │ ◄───────────────────────── set_low_confidence ◄─────┘
        ▼
┌───────────────┐
│ Synthesis     │  (markdown answer, streamed token-by-token)
│ Agent         │
└───────────────┘
        │
       END
```

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM | Groq — Llama 3.3 70B (free tier) |
| Search | Tavily MCP (remote HTTP, no Node.js required) |
| UI | Chainlit — native agent step visibility + streaming |
| State schemas | Pydantic V2 structured outputs |
| Observability | LangSmith (free tier, auto-traced) |
| Packaging | uv + pyproject.toml |

## Setup

### 1. Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install dependencies
```bash
uv sync
```

### 3. Configure API keys
```bash
cp .env.example .env
```
Edit `.env` and fill in:
- `GROQ_API_KEY` — free at [console.groq.com](https://console.groq.com)
- `TAVILY_API_KEY` — your Tavily API key
- `LANGCHAIN_API_KEY` — free at [smith.langchain.com](https://smith.langchain.com) (optional but recommended)

### 4. Run the app
```bash
chainlit run main.py
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

## Key Features

### Multi-turn Conversation
The `add_messages` reducer in `ResearchState` automatically accumulates messages across invocations on the same `thread_id`. Follow-up questions like "What about their competitors?" are handled by passing the last 6 messages as context to the Clarity Agent.

### Human-in-the-Loop Interrupt
When a query is vague, `langgraph.types.interrupt()` pauses the entire graph state. The user is prompted for clarification. On reply, `Command(resume=...)` resumes from exactly where execution paused — no state is lost.

### Agent Step Visibility
`LangchainCallbackHandler` with `stream_final_answer=True` creates a collapsible Chainlit step for every agent node, making the pipeline visible in the UI. The Synthesis Agent's tokens are streamed word-by-word.

### Apology / Best-effort Logic
If the Validator Agent finds research insufficient after 3 retries, the `set_low_confidence` node flags the state. The Synthesis Agent opens its response with a transparent quality disclosure rather than presenting poor data as authoritative.

### Live Pipeline Diagram
On chat start, `graph.get_graph().draw_mermaid()` generates the actual graph topology as a Mermaid diagram rendered natively by Chainlit — so users see the exact agent pipeline before asking their first question.

## Project Structure

```
├── main.py              # Chainlit entry point
├── pyproject.toml       # uv dependencies
├── .env.example         # API key template
├── chainlit.md          # Chainlit welcome screen
├── prompts_log.md       # AI prompts & architectural reasoning (required by assignment)
└── app/
    ├── state.py         # ResearchState TypedDict + Pydantic output schemas
    ├── tools.py         # Tavily MCP client (with direct-SDK fallback)
    ├── graph.py         # Graph construction, routing, interrupt node
    └── agents/
        ├── clarity.py   # Clarity Agent
        ├── research.py  # Research Agent
        ├── validator.py # Validator Agent
        └── synthesis.py # Synthesis Agent
```
