"""Local model configuration shared by helpdesk agents."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def build_llm():
    load_dotenv(ROOT / ".env")
    # Avoid fetching remote pricing metadata for this local-only application.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from google.adk.models.lite_llm import LiteLlm
    return LiteLlm(
        model="ollama_chat/" + os.getenv("LLM_MODEL", "qwen3:8b"),
        api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
        temperature=0,
        num_ctx=8192,
        max_tokens=2400,
        reasoning_effort="none",
    )


def helpdesk_agent_mode(mode=None):
    """Select architecture without changing the model or retrieval settings."""
    load_dotenv(ROOT / ".env")
    selected = mode if mode is not None else os.getenv("HELPDESK_AGENT_MODE", "single")
    selected = selected.strip().lower()
    if selected not in {"single", "multi"}:
        raise ValueError("HELPDESK_AGENT_MODE must be single or multi")
    return selected
