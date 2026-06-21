"""Centralized config + env loading. All modules import from here."""
from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    # DeepSeek
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    compressor_model: str = "deepseek-chat"          # V4 Pro — query-aware rewriter
    solver_model: str = "deepseek-chat"               # V4 Flash — frozen answerer (update ID when confirmed)

    # bear-1.1 (TheTokenCompany SDK — no base URL, just api key)
    bear_api_key: str = os.getenv("BEAR_API_KEY", "")

    # Anthropic (offline gate)
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    gate_model_loop: str = "claude-sonnet-4-20250514"
    gate_model_stamp: str = "claude-opus-4-20250514"

    # Eval
    seed: int = 42
    n_instances: int = 50
    bootstrap_iters: int = 1000


CFG = Config()
