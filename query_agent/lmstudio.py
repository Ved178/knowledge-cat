"""LM Studio HTTP helpers (OpenAI-compatible API, stdlib-only, no extra deps)."""

from __future__ import annotations

import json
import queue
import threading
import urllib.request

from query_agent.constants import DEFAULT_LM_STUDIO_BASE_URL, DEFAULT_LM_STUDIO_MODEL


def check_available(base_url: str = DEFAULT_LM_STUDIO_BASE_URL) -> bool:
    """Return True if LM Studio is reachable at the given base URL."""
    try:
        urllib.request.urlopen(f"{base_url}/models", timeout=3)
        return True
    except Exception:
        return False


def get_first_model(base_url: str = DEFAULT_LM_STUDIO_BASE_URL) -> str | None:
    """Return the best available chat model ID, preferring non-thinking models."""
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=3) as resp:
            data = json.loads(resp.read())
            ids = [m["id"] for m in (data.get("data") or []) if not _is_embedding_model(m["id"])]
            # Prefer non-thinking models — thinking models add latency unsuited for a REPL.
            non_thinking = [m for m in ids if not _is_thinking_model(m)]
            return (non_thinking or ids or [None])[0]
    except Exception:
        return None


def _is_embedding_model(model_id: str) -> bool:
    """Heuristic: skip models that are clearly embedding-only."""
    return "embed" in model_id.lower()


def _is_thinking_model(model_id: str) -> bool:
    """Heuristic: flag known thinking/reasoning model families."""
    lower = model_id.lower()
    return any(pat in lower for pat in ("qwen3", "deepseek-r1", "qwq", "/r1"))


def generate(
    prompt: str,
    *,
    model: str = DEFAULT_LM_STUDIO_MODEL,
    base_url: str = DEFAULT_LM_STUDIO_BASE_URL,
    timeout: float = 60.0,
) -> str:
    """Call LM Studio's OpenAI-compatible chat completions endpoint and return the reply.

    Uses a thread + wall-clock timeout so thinking models (Qwen3, DeepSeek-R1) that
    spend a long time before emitting the first token are cut off correctly.
    """
    # /no_think disables the CoT thinking phase on Qwen3 and compatible models.
    messages = [
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": prompt},
    ]
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": 0.2,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    result_queue: queue.Queue[str | Exception] = queue.Queue()

    def _call() -> None:
        try:
            with urllib.request.urlopen(req, timeout=timeout + 10) as resp:
                data = json.loads(resp.read())
                result_queue.put(data["choices"][0]["message"]["content"].strip())
        except Exception as exc:
            result_queue.put(exc)

    thread = threading.Thread(target=_call, daemon=True)
    thread.start()
    try:
        result = result_queue.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError(f"LM Studio did not respond within {timeout}s") from None

    if isinstance(result, Exception):
        raise result
    return result
