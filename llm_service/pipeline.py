"""
Pipeline — lightweight agent chains with dependency resolution.

Define steps declaratively, the pipeline handles:
  - Topological ordering (what can run in parallel, what must wait)
  - Automatic injection of dependency results into prompts
  - Per-step tracing (elapsed time, token usage hint, success/failure)
  - Structured or plain text output per step

No external dependencies beyond what llm_service already uses.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel

from .client import LLMClient

logger = logging.getLogger("llm_service")

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Step definition
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """A single agent step in a pipeline.

    Args:
        name: Unique identifier (used in depends_on references and results dict).
        system: System prompt — defines this agent's role.
        prompt: User prompt. Can contain ``{input}`` (replaced with pipeline input)
                and ``{step_name}`` placeholders (replaced with that step's output).
                If None, the pipeline input is used as-is.
        output: Optional Pydantic model — if set, uses structured() instead of chat().
        depends_on: List of step names whose results this step needs.
        strict: Structured output strict mode (default True).
        lenient: If True, validation errors return raw dict instead of raising.
        overrides: Extra body-level params for this step (temperature, etc.).
        transform: Optional post-processing function (result) -> result.
    """
    name: str
    system: str
    prompt: Optional[str] = None
    output: Optional[type[BaseModel]] = None
    depends_on: list[str] = field(default_factory=list)
    strict: bool = True
    lenient: bool = False
    overrides: dict[str, Any] = field(default_factory=dict)
    transform: Optional[Callable] = None


# ---------------------------------------------------------------------------
# Step result (trace entry)
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Result of a single pipeline step, including trace info."""
    name: str
    output: Any                        # str, BaseModel instance, or dict
    elapsed: float = 0.0              # seconds
    success: bool = True
    error: Optional[str] = None
    tokens: int = 0                    # total tokens used in this step
    cost_usd: float = 0.0             # estimated cost for this step

    @property
    def text(self) -> str:
        """Get result as string — works for str, BaseModel, and dict."""
        if isinstance(self.output, BaseModel):
            return self.output.model_dump_json(indent=2)
        if isinstance(self.output, dict):
            import json
            return json.dumps(self.output, ensure_ascii=False, indent=2)
        return str(self.output)


# ---------------------------------------------------------------------------
# Pipeline run result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Complete pipeline execution result with all step outputs and trace."""
    steps: dict[str, StepResult] = field(default_factory=dict)
    elapsed: float = 0.0

    def __getitem__(self, step_name: str) -> Any:
        """Get a step's output by name: result['analyst'].output"""
        return self.steps[step_name]

    def output(self, step_name: str) -> Any:
        """Shortcut: result.output('critic') → the Pydantic model or string."""
        return self.steps[step_name].output

    @property
    def total_tokens(self) -> int:
        return sum(sr.tokens for sr in self.steps.values())

    @property
    def total_cost_usd(self) -> float:
        return sum(sr.cost_usd for sr in self.steps.values())

    def trace(self) -> str:
        """Human-readable execution trace with token usage."""
        lines = [f"Pipeline completed in {self.elapsed:.1f}s", ""]
        for name, sr in self.steps.items():
            status = "OK" if sr.success else "FAILED"
            tokens_info = f", {sr.tokens:,} tokens" if sr.tokens else ""
            cost_info = f", ${sr.cost_usd:.4f}" if sr.cost_usd else ""
            lines.append(f"  [{status}] {name}: {sr.elapsed:.1f}s{tokens_info}{cost_info}")
            if sr.error:
                lines.append(f"         Error: {sr.error}")
        lines.append("")
        lines.append(f"  Total: {self.total_tokens:,} tokens, ${self.total_cost_usd:.4f} USD")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """Declarative agent chain with automatic parallelism and dependency injection.

    Usage::

        pipe = Pipeline(
            Step(name="analyst", system="You are a business analyst."),
            Step(name="lawyer",  system="You are a corporate lawyer."),
            Step(
                name="critic",
                system="You are a critical reviewer.",
                prompt="Review these analyses:\\n\\nAnalyst:\\n{analyst}\\n\\nLawyer:\\n{lawyer}",
                depends_on=["analyst", "lawyer"],
            ),
        )
        result = await pipe.run(llm, input="Analyze this contract: ...")
        print(result.output("critic"))
        print(result.trace())
    """

    def __init__(self, *steps: Step) -> None:
        self.steps = {s.name: s for s in steps}
        self._validate()

    def _validate(self) -> None:
        """Check for missing dependencies and cycles."""
        names = set(self.steps.keys())
        for step in self.steps.values():
            missing = set(step.depends_on) - names
            if missing:
                raise ValueError(
                    f"Step '{step.name}' depends on unknown steps: {missing}"
                )
        # Cycle detection via topological sort attempt
        self._topo_order()

    def _topo_order(self) -> list[list[str]]:
        """Return execution layers — each layer is a list of step names
        that can run in parallel. Steps in layer N+1 depend only on
        steps in layers 0..N.

        Raises ValueError on cycles.
        """
        in_degree: dict[str, int] = {
            name: len(step.depends_on) for name, step in self.steps.items()
        }
        dependents: dict[str, list[str]] = {name: [] for name in self.steps}
        for name, step in self.steps.items():
            for dep in step.depends_on:
                dependents[dep].append(name)

        layers: list[list[str]] = []
        ready = [n for n, d in in_degree.items() if d == 0]

        processed = 0
        while ready:
            layers.append(sorted(ready))  # sorted for determinism
            next_ready = []
            for name in ready:
                processed += 1
                for dep in dependents[name]:
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0:
                        next_ready.append(dep)
            ready = next_ready

        if processed != len(self.steps):
            raise ValueError("Cycle detected in pipeline dependencies")

        return layers

    async def run(
        self,
        client: LLMClient,
        input: str,
        **global_overrides: Any,
    ) -> PipelineResult:
        """Execute the full pipeline.

        Args:
            client: An already-opened LLMClient (inside async with).
            input: The base input text (available as {input} in step prompts).
            **global_overrides: Params applied to every step (step overrides win).
        """
        t0 = time.monotonic()
        layers = self._topo_order()
        results: dict[str, StepResult] = {}

        for layer in layers:
            # Skip steps whose dependencies failed
            runnable = []
            for name in layer:
                step = self.steps[name]
                failed_deps = [
                    d for d in step.depends_on
                    if d in results and not results[d].success
                ]
                if failed_deps:
                    results[name] = StepResult(
                        name=name,
                        output=None,
                        success=False,
                        error=f"Skipped — dependency failed: {', '.join(failed_deps)}",
                    )
                    logger.warning("Step '%s' skipped — failed dependencies: %s", name, failed_deps)
                else:
                    runnable.append(name)

            if not runnable:
                continue

            tasks = [
                self._run_step(
                    client, self.steps[name], input, results, global_overrides
                )
                for name in runnable
            ]
            layer_results = await asyncio.gather(*tasks, return_exceptions=True)

            for name, result in zip(runnable, layer_results):
                if isinstance(result, BaseException):
                    results[name] = StepResult(
                        name=name,
                        output=None,
                        success=False,
                        error=str(result),
                    )
                    logger.error("Step '%s' failed: %s", name, result)
                else:
                    results[name] = result

        return PipelineResult(
            steps=results,
            elapsed=time.monotonic() - t0,
        )

    async def _run_step(
        self,
        client: LLMClient,
        step: Step,
        input_text: str,
        prior_results: dict[str, StepResult],
        global_overrides: dict[str, Any],
    ) -> StepResult:
        """Execute a single step, injecting dependency results."""
        t0 = time.monotonic()

        # Build the prompt with placeholders replaced
        prompt = self._render_prompt(step, input_text, prior_results)

        # Merge overrides: global < step-level
        overrides = {**global_overrides, **step.overrides}

        # Call LLM
        if step.output is not None:
            result = await client.structured(
                prompt,
                step.output,
                system=step.system,
                strict=step.strict,
                lenient=step.lenient,
                **overrides,
            )
        else:
            result = await client.chat(
                prompt,
                system=step.system,
                **overrides,
            )

        # Grab per-request usage IMMEDIATELY after the await returns.
        # This is race-safe: between _post() storing _last_request_usage
        # and this read, there are zero await points (only sync code in
        # chat()/structured()), so no other coroutine can interleave.
        last_usage = client._last_request_usage
        step_tokens = last_usage.total_tokens
        step_cost = last_usage.cost_usd or 0.0

        # Optional post-processing
        if step.transform:
            result = step.transform(result)

        elapsed = time.monotonic() - t0
        logger.info("Step '%s' completed in %.1fs (%d tokens)", step.name, elapsed, step_tokens)

        return StepResult(
            name=step.name,
            output=result,
            elapsed=elapsed,
            tokens=step_tokens,
            cost_usd=step_cost,
        )

    def _render_prompt(
        self,
        step: Step,
        input_text: str,
        prior_results: dict[str, StepResult],
    ) -> str:
        """Replace {input} and {step_name} placeholders in the step prompt.

        Uses manual replacement instead of str.format() to avoid crashes
        when input text contains curly braces (JSON, code, etc.).
        """
        template = step.prompt if step.prompt is not None else "{input}"

        # Build substitution dict
        subs: dict[str, str] = {"input": input_text}
        for name, sr in prior_results.items():
            subs[name] = sr.text

        # Manual replacement — safe with JSON/code in values.
        # Sort by key length descending to prevent partial matches:
        # e.g. {input_validator} must be replaced before {input}
        result = template
        for key in sorted(subs, key=len, reverse=True):
            result = result.replace("{" + key + "}", subs[key])
        return result
