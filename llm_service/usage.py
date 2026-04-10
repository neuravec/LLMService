"""
Token usage tracking and cost estimation.

Accumulates prompt/completion tokens across all requests in an LLMClient session.
Cost estimation uses configurable per-model pricing (USD per 1M tokens).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Pricing table — USD per 1M tokens (input, output)
# Update as Azure pricing changes. Unknown models → no cost estimate.
# ---------------------------------------------------------------------------

_PRICING: dict[str, tuple[float, float]] = {
    # GPT-4.1 family
    "gpt-4.1":       (2.00, 8.00),
    "gpt-4.1-mini":  (0.40, 1.60),
    "gpt-4.1-nano":  (0.10, 0.40),
    # GPT-4o family
    "gpt-4o":        (2.50, 10.00),
    "gpt-4o-mini":   (0.15, 0.60),
    # GPT-5.x family (estimated)
    "gpt-5.4-mini":  (1.00, 4.00),
    # o-series reasoning
    "o3":            (2.00, 8.00),
    "o3-mini":       (1.10, 4.40),
    "o4-mini":       (1.10, 4.40),
    "o1":            (15.00, 60.00),
    "o1-mini":       (1.10, 4.40),
}


def get_pricing(model_name: str) -> Optional[tuple[float, float]]:
    """Look up (input_per_1m, output_per_1m) for a model.
    Tries exact match first, then prefix match."""
    name = model_name.lower().strip()
    if name in _PRICING:
        return _PRICING[name]
    # Prefix match: "gpt-4.1-2025-04-14" → "gpt-4.1"
    for key in sorted(_PRICING.keys(), key=len, reverse=True):
        if name.startswith(key):
            return _PRICING[key]
    return None


# ---------------------------------------------------------------------------
# Single-request usage snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RequestUsage:
    """Token usage for a single API call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0          # hidden reasoning tokens (o-series, gpt-5.x)
    cached_prompt_tokens: int = 0      # prompt cache hits
    cost_usd: Optional[float] = None

    @classmethod
    def from_response(cls, data: dict[str, Any], model_name: str) -> "RequestUsage":
        """Extract usage from Azure OpenAI response body."""
        usage = data.get("usage")
        if not usage:
            return cls()

        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        total = usage.get("total_tokens", prompt + completion)

        # Detailed token breakdown (reasoning models, prompt caching)
        comp_details = usage.get("completion_tokens_details") or {}
        reasoning = comp_details.get("reasoning_tokens", 0)
        prompt_details = usage.get("prompt_tokens_details") or {}
        cached = prompt_details.get("cached_tokens", 0)

        pricing = get_pricing(model_name)
        cost = None
        if pricing:
            inp_rate, out_rate = pricing
            cost = (prompt * inp_rate + completion * out_rate) / 1_000_000

        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            reasoning_tokens=reasoning,
            cached_prompt_tokens=cached,
            cost_usd=cost,
        )


# ---------------------------------------------------------------------------
# Cumulative usage tracker (thread-safe for gather)
# ---------------------------------------------------------------------------

@dataclass
class UsageTracker:
    """Accumulates token usage across multiple requests.
    Thread-safe (asyncio.gather schedules on one thread, but just in case)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_prompt_tokens: int = 0
    cost_usd: float = 0.0
    request_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, req: RequestUsage) -> None:
        """Add a single request's usage to the running total."""
        with self._lock:
            self.prompt_tokens += req.prompt_tokens
            self.completion_tokens += req.completion_tokens
            self.total_tokens += req.total_tokens
            self.reasoning_tokens += req.reasoning_tokens
            self.cached_prompt_tokens += req.cached_prompt_tokens
            if req.cost_usd is not None:
                self.cost_usd += req.cost_usd
            self.request_count += 1

    def reset(self) -> None:
        """Zero out all counters."""
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.reasoning_tokens = 0
            self.cached_prompt_tokens = 0
            self.cost_usd = 0.0
            self.request_count = 0

    def summary(self) -> str:
        """Human-readable usage summary."""
        lines = [
            f"Requests: {self.request_count}",
            f"Tokens:   {self.total_tokens:,} (prompt: {self.prompt_tokens:,}, completion: {self.completion_tokens:,})",
        ]
        if self.reasoning_tokens > 0:
            lines.append(f"          reasoning: {self.reasoning_tokens:,} (included in completion)")
        if self.cached_prompt_tokens > 0:
            lines.append(f"          cached prompt: {self.cached_prompt_tokens:,}")
        if self.cost_usd > 0:
            lines.append(f"Cost:     ${self.cost_usd:.4f} USD")
        else:
            lines.append("Cost:     n/a (model not in pricing table)")
        return "\n".join(lines)
