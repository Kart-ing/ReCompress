"""Shared API client setup with retries + long timeouts."""
from __future__ import annotations
from openai import OpenAI
from src.config import CFG

_client = OpenAI(
    api_key=CFG.deepseek_api_key,
    base_url=CFG.deepseek_base_url,
    timeout=120.0,
    max_retries=5,
)

def get_client() -> OpenAI:
    return _client
