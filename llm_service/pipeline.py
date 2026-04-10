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
        name: Unique identifier. Used in ``depends_on`` references, ``result.output(name)``,
            and ``{name}`` placeholders in other steps' prompts.
        system: System prompt defining this agent's role/persona.
        prompt: User prompt template. Supports placeholders:
            ``{input}`` = original pipeline input,
            ``{step_name}`` = output of that step (auto-serialized to JSON for Pydantic models).
            If ``None``, the pipeline input is used as-is.
        output: Pydantic ``BaseModel`` subclass for structured output. If ``None``,
            the step returns plain text via ``chat()``.
        depends_on: List of step names that must complete before this step runs.
            Steps without dependencies run in parallel.
        strict: Use Azure Structured Outputs for schema enforcement (default ``True``).
        lenient: Return raw dict on validation failure instead of raising (default ``False``).
        overrides: Per-step API param overrides (e.g. ``{"temperature": 0.2}``).
        transform: Optional post-processing function ``(result) -> result`` applied
            after the LLM call.

    Example::

        Step(
            name="analyst",
            system="You are a business analyst. Perform SWOT analysis.",
            output=SwotAnalysis,
        )

        Step(
            name="critic",
            system="You are a critical reviewer.",
            prompt="Review this analysis:\\n{analyst}\\n\\nOriginal doc:\\n{input}",
            output=CriticVerdict,
            depends_on=["analyst"],
        )
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
    """Result of a single pipeline step, including trace info.

    Attributes:
        name: Step name.
        output: Step output — ``str``, Pydantic ``BaseModel`` instance, or ``dict``.
        elapsed: Execution time in seconds.
        success: ``True`` if step completed successfully.
        error: Error message if step failed or was skipped.
        tokens: Total tokens used by this step.
        cost_usd: Estimated cost in USD for this step.
    """
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
    """Complete pipeline execution result with all step outputs and trace.

    Attributes:
        steps: Dict of step name to ``StepResult``.
        elapsed: Total pipeline execution time in seconds.
        total_tokens: Sum of tokens across all steps.
        total_cost_usd: Sum of estimated cost across all steps.

    Example::

        result = await pipe.run(llm, input="...")
        print(result.output("critic"))       # Pydantic model or str
        print(result.total_tokens)           # 1350
        print(result.total_cost_usd)         # 0.0054
        print(result.trace())                # human-readable execution log
    """
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

    Define steps with roles, prompts, and dependencies. The pipeline
    automatically resolves execution order (topological sort), runs
    independent steps in parallel, and injects results into dependent steps.

    Args:
        *steps: Step instances. Order doesn't matter — execution order is
            determined by ``depends_on`` declarations.

    Raises:
        ValueError: If a step references an unknown dependency or a cycle is detected.

    Example::

        from llm_service import Pipeline, Step, LLMClient, LLMConfig

        pipe = Pipeline(
            Step(name="analyst", system="You are a business analyst.", output=Analysis),
            Step(name="lawyer",  system="You are a corporate lawyer.", output=LegalView),
            Step(
                name="critic",
                system="You are a critical reviewer.",
                prompt="Review:\\n{analyst}\\n{lawyer}\\nOriginal:\\n{input}",
                output=Verdict,
                depends_on=["analyst", "lawyer"],
            ),
        )

        cfg = LLMConfig.from_yaml("config.yaml")
        async with LLMClient(cfg) as llm:
            result = await pipe.run(llm, input="Proposal to acquire startup...")
            print(result.output("critic"))
            print(result.trace())

        # Reuse on multiple documents:
        async with LLMClient(cfg) as llm:
            all_results = await asyncio.gather(*[
                pipe.run(llm, input=doc) for doc in documents
            ])
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
            client: An already-opened ``LLMClient`` (inside ``async with``).
            input: The base input text, available as ``{input}`` in step prompts.
            **global_overrides: API params applied to every step. Per-step
                ``overrides`` take precedence over global ones.

        Returns:
            PipelineResult: Contains all step outputs, trace, and token/cost totals.

        Example::

            async with LLMClient(cfg) as llm:
                result = await pipe.run(llm, input="Document text here...")
                print(result.output("critic"))
                print(result.trace())
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
