# llm_service

Async Python toolkit for Azure OpenAI (Foundry). One module the team imports everywhere — handles config, model detection, concurrency, structured output, agent pipelines, vision, token tracking.

## Installation

```bash
pip install httpx pyyaml pydantic
```

Clone the repo and import directly, or copy `llm_service/` into your project.

## Quick start

```yaml
# config.yaml
api_key: ${AZURE_OPENAI_KEY}
endpoint: https://your-resource.openai.azure.com
model_name: gpt-4.1
```

```python
import asyncio
from llm_service import LLMConfig, LLMClient

async def main():
    cfg = LLMConfig.from_yaml("config.yaml")
    async with LLMClient(cfg) as llm:
        answer = await llm.chat(
            "Summarize this document.",
            system="You extract key facts."
        )
        print(answer)
        print(llm.usage.summary())

asyncio.run(main())
```

In a Jupyter notebook `asyncio.run()` is not needed — just `await` directly:

```python
cfg = LLMConfig.from_yaml("config.yaml")
async with LLMClient(cfg) as llm:
    answer = await llm.chat("Hello!")
```

## Features

### Batch (concurrent)

```python
results = await llm.batch(
    ["What is ETL?", "What is RAG?", "What is a vector DB?"],
    system="Answer in 1-2 sentences.",
)
```

Semaphore (default 8) controls how many requests fly in parallel.

### Structured output (Pydantic)

```python
from pydantic import BaseModel, Field

class Invoice(BaseModel):
    number: str = Field(description="Invoice number")
    total: float = Field(description="Gross amount")
    currency: str = Field(description="Currency code")

invoice = await llm.structured(
    f"Extract data from this invoice:\n\n{document}",
    Invoice,
)
print(invoice.number, invoice.total, invoice.currency)
```

Two modes:
- **`strict=True`** (default) — Azure constrains output to the schema server-side
- **`strict=False`** — JSON mode + post-validation with Pydantic

### JSON output (no schema)

```python
data = await llm.chat_json("Return top 3 cities in Poland as JSON with 'cities' key")
# → {"cities": [{"name": "Warsaw", ...}, ...]}
```

### Pipeline (agent chains)

Multiple agents working on the same input, with dependencies:

```python
from llm_service import Pipeline, Step

pipe = Pipeline(
    # Layer 1 — run in parallel (no depends_on)
    Step(name="analyst", system="You are a business analyst.", output=AnalystView),
    Step(name="lawyer",  system="You are a corporate lawyer.", output=LegalView),
    # Layer 2 — waits for both
    Step(
        name="critic",
        system="You are a critical reviewer.",
        prompt="Review:\n\n{analyst}\n\n{lawyer}\n\nOriginal:\n{input}",
        output=Verdict,
        depends_on=["analyst", "lawyer"],
    ),
)

async with LLMClient(cfg) as llm:
    result = await pipe.run(llm, input="Proposal to acquire startup ABC...")
    print(result.output("critic"))
    print(result.trace())
```

```
Pipeline completed in 4.2s

  [OK] analyst: 1.8s, 450 tokens, $0.0018
  [OK] lawyer: 2.1s, 520 tokens, $0.0022
  [OK] critic: 1.9s, 380 tokens, $0.0014

  Total: 1,350 tokens, $0.0054 USD
```

Pipelines are reusable — define once, run on many documents (including via `asyncio.gather`).

### Vision / multimodal

```python
# From file
answer = await llm.chat("Describe this image.", images=["photo.png"])

# Structured OCR from scan
invoice = await llm.structured(
    "Extract invoice data from this scan.",
    Invoice,
    images=["scan.jpg"],
    image_detail="high",
)

# Compare two images
result = await llm.structured(
    "Compare these two images.",
    ComparisonResult,
    images=["v1.png", "v2.png"],
)
```

Accepts file paths (auto base64), URLs (passthrough), or raw bytes.

### Token tracking and cost estimation

```python
async with LLMClient(cfg) as llm:
    await llm.batch(prompts)
    print(llm.usage.summary())
```

```
Requests: 50
Tokens:   12,340 (prompt: 8,200, completion: 4,140)
          reasoning: 1,200 (included in completion)
Cost:     $0.0495 USD
```

Built-in pricing for gpt-4.1, gpt-4o, o-series, gpt-5.x families.

### Model auto-detection

The client automatically adapts to the model:

| Model | temperature | system role | token param | reasoning_effort |
|---|---|---|---|---|
| gpt-4.1 | supported | `system` | `max_completion_tokens` | stripped |
| gpt-5.4-mini | stripped | `system` | `max_completion_tokens` | `medium` default |
| o3, o4-mini | stripped | `developer` | `max_completion_tokens` | `medium` default |

Unsupported params are silently dropped — no need for model-specific code.

### Error handling

```python
from llm_service import LLMError

try:
    result = await llm.chat("...")
except LLMError as e:
    print(e)
```

```
LLMError: Rate limit exceeded
  HTTP 429
  Azure code: RateLimitExceeded
  Model: gpt-4.1
  Hint: Rate limit — za duzo requestow. Zmniejsz concurrency lub dodaj retry.
  Retries: 3
    attempt 1: HTTP 429 after 1.2s
    attempt 2: HTTP 429 after 0.8s
    attempt 3: HTTP 429 after 0.9s
```

Retries (408, 429, 5xx, timeouts) with exponential backoff + Retry-After header support.

## Configuration

### YAML (recommended)

```yaml
api_key: ${AZURE_OPENAI_KEY}
endpoint: https://your-resource.openai.azure.com
model_name: gpt-4.1

# All optional:
# temperature: 0.7
# max_tokens: 4096
# reasoning_effort: medium     # for reasoning models: low|medium|high
# concurrency: 8               # max parallel requests
# timeout: 120                 # seconds per request
# retries: 3
```

Supports `${ENV_VAR}` and `${ENV_VAR:default}` substitution.

### Code

```python
cfg = LLMConfig(
    api_key=os.environ["AZURE_OPENAI_KEY"],
    endpoint="https://your-resource.openai.azure.com",
    model_name="gpt-4.1",
)
```

Only `api_key`, `endpoint`, `model_name` are required. Everything else has sensible defaults.

## Project structure

```
llm_service/
  __init__.py      — public API exports
  config.py        — YAML config loader with env-var substitution
  models.py        — model capability registry (reasoning vs standard)
  client.py        — async httpx client, semaphore, retry, error handling
  structured.py    — Pydantic ↔ JSON schema, response parsing
  pipeline.py      — agent chains with DAG execution
  usage.py         — token counting, cost estimation
  vision.py        — image encoding for multimodal requests

config.example.yaml  — config template
examples.py          — usage examples for .py scripts
examples.ipynb       — usage examples for Jupyter notebooks
```

## Examples

See [`examples.py`](examples.py) (scripts) and [`examples.ipynb`](examples.ipynb) (notebooks) for complete working examples covering all features.

## Dependencies

- `httpx` — async HTTP client
- `pyyaml` — YAML config loading
- `pydantic` — structured output validation

No dependency on the `openai` Python SDK. Direct HTTP to Azure OpenAI REST API.
