# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`llm_service` is a Python toolkit for the team to standardize async communication with Azure OpenAI (Foundry). It provides a single importable module that handles config loading, model-aware request building, concurrency, and retries.

## Architecture

```
llm_service/
  __init__.py      — public API exports
  config.py        — YAML loading with ${ENV_VAR} substitution, dataclass validation
  models.py        — model capability registry (reasoning vs standard auto-detection)
  client.py        — async httpx client, semaphore, retry, LLMError, images support
  structured.py    — Pydantic ↔ JSON schema conversion, response parsing, strict mode
  pipeline.py      — declarative agent chains with DAG execution and dependency injection
  usage.py         — token counting, cost estimation, per-model pricing table
  vision.py        — image encoding (file/URL/bytes → base64), multimodal content builder
```

**Key design constraint:** async-first with `httpx.AsyncClient` + `asyncio.Semaphore` (default 8). This is a team requirement — never replace with synchronous `requests` or the official `openai` SDK.

**Model auto-detection:** `models.py` maps model name prefixes to capability flags (supports_temperature, supports_system_message, reasoning, etc.). When a reasoning model is detected (o-series, gpt-5.x), the client automatically drops temperature/top_p and uses `reasoning_effort` + `max_completion_tokens` instead. System messages become `"role": "developer"` for models that don't support system role.

**Config:** Only `api_key`, `endpoint`, `model_name` are required. Everything else is optional with sensible defaults. YAML configs support `${ENV_VAR}` and `${ENV_VAR:default}` syntax.

## Dependencies

- `httpx` — async HTTP
- `pyyaml` — config loading
- `pydantic` — structured output validation and schema generation

Intentionally does not use the `openai` Python SDK.

## Usage pattern

```python
from llm_service import LLMConfig, LLMClient

cfg = LLMConfig.from_yaml("config.yaml")
async with LLMClient(cfg) as llm:
    # plain text
    result = await llm.chat("Extract key facts", system="You are a data extractor.")
    # batch (concurrent, semaphore-bounded)
    results = await llm.batch(["prompt1", "prompt2"], system="...")
    # JSON dict
    data = await llm.chat_json("Return top 3 items as JSON")
    # structured output → Pydantic model
    obj = await llm.structured("Extract invoice data from ...", InvoiceModel)
    # batch structured
    objs = await llm.batch_structured(prompts, ContractModel, system="...")
```

## Structured output modes

`structured.py` supports two modes, both accessed via `LLMClient.structured()`:

1. **strict=True** (default) — sends Pydantic JSON schema in `response_format.json_schema` so Azure constrains the output. Uses `_enforce_strict()` to set `additionalProperties: false` and all-required recursively.
2. **strict=False** — uses `response_format: json_object` + injects schema hint into system message. More forgiving but no server-side constraint.

`lenient=True` on any structured call returns a raw dict instead of raising on validation failure.

## Pipeline (agent chains)

`pipeline.py` provides `Step` + `Pipeline` for multi-agent workflows:

- Steps without `depends_on` run in parallel (fan-out via `asyncio.gather`)
- Steps with `depends_on` wait for those results (fan-in)
- Execution order resolved via topological sort; cycles detected at construction time
- Prompts use `{step_name}` placeholders — auto-replaced with prior step outputs
- `{input}` placeholder refers to the original pipeline input
- Each step can output plain text (`output=None`) or structured (`output=PydanticModel`)
- `PipelineResult.trace()` returns human-readable execution log with per-step timing
- Pipelines are reusable: define once, run on many inputs (including via `asyncio.gather`)

## Token usage and cost tracking

`usage.py` provides automatic tracking:
- `llm.usage` — `UsageTracker` accumulates prompt/completion/reasoning tokens and estimated USD cost
- `RequestUsage` captures `reasoning_tokens` (hidden thinking tokens for o-series/gpt-5.x) and `cached_prompt_tokens`
- Cost is calculated from `_PRICING` dict in `usage.py` (USD per 1M tokens, input/output). Uses exact match then prefix match for versioned deployment names.
- Pipeline `trace()` includes per-step tokens and cost; `result.total_tokens` / `result.total_cost_usd` for totals
- When adding new model pricing, edit `_PRICING` in `usage.py`

## Vision / multimodal

`vision.py` handles image encoding for GPT-4.1, GPT-4o, and newer models:
- `images` parameter on `chat()`, `chat_json()`, `structured()` accepts list of: file paths (auto base64 + MIME detection), URLs (passthrough), or raw bytes
- `image_detail` param: `"auto"` (default), `"low"`, `"high"` — controls Azure resolution/cost tradeoff
- When no images are passed, content stays as plain string (no overhead)
- `build_content_parts()` and `encode_image()` are also exported for manual use

## Error handling

`client.py` raises `LLMError` (not generic exceptions) with:
- `status_code`, `error_code`, `error_type` — parsed from Azure response body
- `hint` — actionable advice mapped from known error codes (401, 404, 429, content_filter, etc.)
- `retry_history` — full log of each attempt with status, elapsed time, wait
- All transient errors (408, 429, 5xx, timeouts, connection errors) are retried with exponential backoff
- `chat()` handles `refusal` (model refused) and `content: null` (content filter)
- `chat_json()` detects truncated JSON from `finish_reason: length` and raises with actionable hint

## API compliance notes

- `max_completion_tokens` is used for ALL models (not deprecated `max_tokens`)
- `stop` is auto-stripped for models where `supports_stop=False` (o3, o4-mini)
- Reasoning models: `temperature`, `top_p`, `frequency_penalty`, `presence_penalty` are excluded
- System role → `developer` role for o-series models (as recommended by Azure docs)
- `reasoning_effort` values: o-series accepts `low|medium|high`; GPT-5.x accepts `none|minimal|low|medium|high|xhigh`

## When adding new models

Edit `_FAMILIES` list in `models.py`. Each entry is a regex pattern + `ModelCapabilities` dataclass. Order matters — first match wins. Unknown models fall back to standard (non-reasoning) defaults. Key flags: `reasoning`, `supports_temperature`, `supports_system_message`, `supports_stop`.

## Rules

- Config files with real keys (`config.yaml`, `.env`) are gitignored — never commit them.
- `config.example.yaml` is the template; keep it in sync with `LLMConfig` fields.
- All HTTP goes through `client.py._post()` which enforces semaphore + retry. Don't bypass it.
