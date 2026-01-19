from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class LLMConfig:
    provider: str  # "none" | "openai" | "gemini"
    model: str
    api_key: Optional[str]
    enabled: bool

def load_llm_config() -> LLMConfig:
    provider = os.getenv("UCHKO_LLM_PROVIDER", "none").strip().lower()
    model = os.getenv("UCHKO_LLM_MODEL", "").strip()
    enabled = os.getenv("UCHKO_LLM_ENABLED", "0").strip() == "1"

    api_key = None
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not model:
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not model:
            model = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash")

    return LLMConfig(provider=provider, model=model, api_key=api_key, enabled=enabled)
