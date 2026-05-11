<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangGraph-0.2+-FF6B35?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Chainlit-2.0+-00C4CC?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Gemini-2.0_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge"/>
</p>

<h1 align="center">Multi-Agent Business Intelligence</h1>

<p align="center">
  <strong>A production-grade multi-agent system for deep business research.</strong><br/>
  Ask anything about a company — financials, leadership, news, strategy, competitors.<br/>
  A pipeline of specialised AI agents searches, validates, and synthesises the answer.
</p>

---

## Overview

Company Intel Agents orchestrates four specialised AI agents in a stateful LangGraph pipeline to answer company research queries with precision and transparency.

Rather than a single LLM call, each query passes through a purpose-built chain: a **Clarity Agent** that understands intent, a **Research Agent** that generates context-aware search queries and retrieves live web data via Tavily, a **Validator Agent** that quality-gates findings, and a **Synthesis Agent** that writes a clean, factual response. If research quality falls short, the pipeline retries with a different search angle. If it repeatedly fails, the system discloses this honestly rather than padding the answer with generic content.

The system maintains full conversation history, so follow-up questions like *"how does it compare to Google?"* or *"what about their latest earnings?"* are understood in context — no need to repeat the company name.

---

## Features

| | |
|---|---|
| 🔁 **Multi-turn context** | Follow-up queries resolve correctly using conversation history — pronouns, ellipsis, multi-entity references all handled |
| 🔍 **Adaptive search** | Research Agent generates a context-aware Tavily query before every search; retry attempts use a different angle automatically |
| ✅ **Quality gating** | Validator Agent inspects every result; insufficient findings trigger up to 3 retry attempts before synthesis |
| 🛑 **Human-in-the-loop** | Ambiguous or vague queries pause the pipeline and prompt for clarification — the graph resumes exactly where it paused |
| 🧠 **Honest disclosure** | When data is limited, the system says so — no hallucinated padding |
| 💾 **Persistent memory** | Every conversation step is checkpointed to SQLite; history survives server restarts |
| 🔐 **Authentication** | Built-in login screen; credentials managed via environment variables |
| 📊 **Pipeline visibility** | Every agent step is shown as a collapsible trace in the UI — you see exactly what ran |
| 🔭 **Observability** | Full LangSmith tracing out of the box — latency, token counts, and agent execution trees |

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐   ambiguous / vague   ┌──────────────────────┐
│  Clarity Agent  │ ─────────────────────► │  Human Clarification │ ◄─ interrupt()
│                 │                        │  (HITL pause)        │
└────────┬────────┘                        └──────────┬───────────┘
         │ clear                                      │ user replies → resume
         ◄────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Research Agent │   generate search query from context
│                 │ → Tavily MCP live web search
│                 │ → synthesise findings + confidence score (0–10)
└────────┬────────┘
         │
         │  score ≥ 6                         score < 6
         ├───────────────────────────────►  ┌──────────────────┐
         │                                  │  Validator Agent │
         │                                  └────────┬─────────┘
         │                       sufficient          │        insufficient (< 3 retries)
         ◄────────────────────────────────────────────┤              │
         │                                            │        ┌─────▼──────────┐
         │                                            │        │ Research Agent │ (new angle)
         │                                            │        └────────────────┘
         │                          max retries       │
         ◄──────────────────── set_low_confidence ◄───┘
         │
         ▼
┌─────────────────┐
│ Synthesis Agent │   streams markdown response token-by-token
└─────────────────┘
         │
        END
```

---

## Quick Start

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and install dependencies

```bash
git clone https://github.com/Momin-Abdurrehman/multi-agent-business-intelligence.git
cd multi-agent-business-intelligence
uv sync
```

### 3. Configure environment

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | ✅ | [Google AI Studio](https://aistudio.google.com) — free |
| `TAVILY_API_KEY` | ✅ | Tavily search API key |
| `APP_USERNAME` | ✅ | Login username for the app |
| `APP_PASSWORD` | ✅ | Login password for the app |
| `CHAINLIT_AUTH_SECRET` | ✅ | Run `chainlit create-secret` to generate |
| `LANGCHAIN_API_KEY` | ☑️ | [LangSmith](https://smith.langchain.com) — optional, enables tracing |
| `LANGCHAIN_TRACING_V2` | ☑️ | Set to `true` to enable LangSmith tracing |

### 4. Run

```bash
chainlit run main.py
```

Open [http://localhost:8000](http://localhost:8000), log in, and start asking.

### 5. Run tests

```bash
uv run pytest tests/
```

---

## Example Queries

```
Tell me about Stripe.

What is their current valuation and who are the founders?

How does it compare to Adyen?

Tell me about Merchain Singapore.

What happened to Mercury last quarter?
```

The system handles obscure startups, ambiguous names (Mercury the fintech vs. the car brand), and multi-entity comparisons equally well.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) | Stateful agent graph with conditional routing and interrupt support |
| **LLM** | Google Gemini 2.0 Flash | Fast, capable, generous free tier |
| **Search** | [Tavily MCP](https://tavily.com) (remote HTTP) | Real-time web search via Model Context Protocol — no Node.js required |
| **UI** | [Chainlit](https://chainlit.io) | Purpose-built for LLM agents — native step visibility, streaming, auth |
| **State schemas** | Pydantic V2 | Type-safe structured outputs; validation errors at boundaries, not deep in logic |
| **Persistence** | AsyncSqliteSaver | SQLite checkpointing; zero infrastructure overhead |
| **Observability** | [LangSmith](https://smith.langchain.com) | Auto-traces every run — latency, token counts, agent execution trees |
| **Packaging** | [uv](https://github.com/astral-sh/uv) + pyproject.toml | Fast, modern Python dependency management |

---

## Project Structure

```
company-intel-agents/
├── main.py                  # Chainlit entry point — auth, session management, two-pass render
├── pyproject.toml           # Dependencies (uv)
├── .env.example             # Environment variable template
├── .chainlit/
│   └── config.toml          # Chainlit UI and feature configuration
└── app/
    ├── state.py             # ResearchState TypedDict + Pydantic output schemas
    ├── tools.py             # Tavily MCP client with direct-SDK fallback
    ├── graph.py             # Graph construction, conditional routing, interrupt node
    └── agents/
        ├── clarity.py       # Classifies intent, extracts canonical company name
        ├── research.py      # Generates search query, calls Tavily, synthesises findings
        ├── validator.py     # Quality-gates findings, triggers retries
        └── synthesis.py     # Streams the final markdown response
```

---

## Design Decisions

**Why does the Research Agent generate its own search query?**
Follow-up questions like *"what about its profit this year?"* contain no company name. Rather than trying to enumerate every follow-up pattern in the Clarity Agent, the Research Agent generates a self-contained Tavily query from conversation history before every search. On retries, it receives the previous findings and automatically uses a different search angle — giving each attempt a genuine chance of finding new information.

**Why is the Validator Agent strict by default?**
Without active quality-gating, an LLM synthesiser will generate plausible-sounding responses even from off-topic search results. The Validator uses an adversarial "judge" prompt and receives both the findings and the Research Agent's self-reported confidence score. Findings about the planet Mercury get rejected when the query was about a fintech company; generic merchant definitions get rejected when the query named a specific startup.

**Why two-pass rendering in Chainlit?**
Chainlit renders messages in the order they are first sent. Streaming synthesis tokens while simultaneously sending agent step traces produces non-deterministic UI ordering. The two-pass design runs the full pipeline first (buffering synthesis tokens), then sends agent steps (always above) followed by the streamed response (always below) — guaranteed ordering regardless of LLM latency.

---

## License

MIT
