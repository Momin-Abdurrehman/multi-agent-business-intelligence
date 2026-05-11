"""
Pytest configuration — sets stub environment variables so module-level LLM
clients can initialise without real API keys. Tests that don't call the LLM
(routing functions, Pydantic schemas) run entirely offline.
"""
import os

os.environ.setdefault("GROQ_API_KEY", "test-key-stub")
os.environ.setdefault("TAVILY_API_KEY", "test-key-stub")
os.environ.setdefault("GOOGLE_API_KEY", "test-key-stub")
