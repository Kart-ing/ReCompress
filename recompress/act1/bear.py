"""bear-2 baseline: blind character-for-character deletion.
PRD §2: bear deletes low-value tokens, char-for-char, blind to the query.
Uses the official TheTokenCompany SDK.

The TTC endpoint enforces 60 requests/minute. With parallel eval workers and
stacked bars (bear, bear→ours, ours→bear) that ceiling is easy to blow, so every
call goes through a single process-wide token-bucket limiter + bounded retry.
"""
from __future__ import annotations
import threading
import time

from thetokencompany import TheTokenCompany
from recompress.config import CFG
from recompress.act1.tokens import count_tokens, truncate_to_tokens

_client = TheTokenCompany(api_key=CFG.bear_api_key) if CFG.bear_api_key else None

# --- process-wide rate limit: stay under the TTC 60 req/min cap ---
_RPM = 55  # headroom below the hard 60/min
_MIN_INTERVAL = 60.0 / _RPM
_rl_lock = threading.Lock()
_next_allowed = 0.0


def _throttle() -> None:
    """Block until the next request is allowed (serializes spacing across threads)."""
    global _next_allowed
    while True:
        with _rl_lock:
            now = time.monotonic()
            if now >= _next_allowed:
                _next_allowed = max(now, _next_allowed) + _MIN_INTERVAL
                return
            wait = _next_allowed - now
        time.sleep(wait)


def _compress_with_retry(text: str, max_retries: int = 6):
    """Call the SDK through the limiter; back off and retry on rate-limit errors."""
    last_exc = None
    for attempt in range(max_retries):
        _throttle()
        try:
            return _client.compress(text)
        except Exception as e:  # SDK raises RateLimitError among others
            last_exc = e
            if "rate limit" in str(e).lower() or "429" in str(e):
                time.sleep(min(2.0 * (attempt + 1), 15.0))
                continue
            raise
    raise last_exc


def compress_bear(text: str, ratio: float = 0.3, question: str | None = None) -> str:
    """Blind deletion baseline. `question` is IGNORED by design (bear is query-blind).

    bear doesn't take a ratio directly — it compresses at its default rate and we
    report whatever it returns. To match budget across bars we truncate to the
    same target token count the other bars use.
    """
    if _client is None:
        raise RuntimeError("BEAR_API_KEY not set — fill .env")
    result = _compress_with_retry(text)
    out = result.output
    target_tokens = max(1, int(count_tokens(text) * ratio))
    return truncate_to_tokens(out, target_tokens)
