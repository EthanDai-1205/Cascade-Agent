"""Task decomposition.

The planner turns one task into ordered steps and types each step. Step typing
is what lets Jev execute work instead of only routing it:

``kind = "generate"``  something must be written (prose, code, a diff)
``kind = "decide"``    the answer is one of a fixed set, a yes/no, or a position
                       on an ordered scale, so System One can answer it directly
                       with zero model tokens

A ``decide`` step carries ``answer_type`` plus either ``options`` (choice,
ordered levels for score) or nothing (noul, a single graded truth value).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .providers import Completion, Provider

DECIDE_TYPES = ("choice", "noul", "score")
SPLITTERS = (" and then ", "; ", "\n")
PLAN_SYSTEM = """You are the planner inside a cost-aware agent.

Split the task into the smallest number of concrete steps that can be executed
and checked independently. Type every step:

- "generate": a deliverable must be written (prose, code, a patch, a query).
- "decide": the answer is one of a fixed set of options, a yes/no, or a position
  on an ordered scale. These are answered by a small calibrated decision model
  and cost nothing, so use "decide" whenever it genuinely fits instead of
  wrapping a classification in prose.

Reply with JSON only, no prose and no code fences:

{"steps": [
  {"id": "s1",
   "goal": "one sentence describing exactly what this step produces",
   "kind": "generate",
   "criteria": "how a reviewer decides this step is correct",
   "context_needed": ["what this step must read from earlier steps"]},
  {"id": "s2",
   "goal": "label the ticket severity",
   "kind": "decide",
   "answer_type": "choice",
   "options": ["low", "medium", "high"],
   "criteria": "the label that best matches the described impact"}
]}

Rules: at most <<MAX_STEPS>> steps; ids are unique; "criteria" is never empty;
"options" has 2 or more entries when used; do not add steps that only restate
the task."""


@dataclass
class Step:
    id: str
    goal: str
    kind: str = "generate"
    criteria: str = ""
    answer_type: str = ""
    options: list[str] = field(default_factory=list)
    context_needed: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)

    @property
    def is_decision(self) -> bool:
        return self.kind == "decide"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "kind": self.kind,
            "criteria": self.criteria,
            "answer_type": self.answer_type,
            "options": self.options,
            "context_needed": self.context_needed,
            "depends_on": self.depends_on,
        }

@dataclass
class Plan:
    task: str
    steps: list[Step]
    tier: str = ""
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    stub: bool = False
    fell_back: bool = False
    error: str = ""


def parse_plan(text: str, task: str, max_steps: int) -> tuple[list[Step], str]:
    """Parse planner output. Returns (steps, error). Never raises."""

    payload = _extract_json(text)
    if payload is None:
        return [], "planner output was not valid JSON"
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        return [], "planner JSON had no non-empty 'steps' list"

    steps: list[Step] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_steps):
        if len(steps) >= max_steps:
            break
        if not isinstance(raw, dict):
            continue
        goal = str(raw.get("goal", "")).strip()
        if not goal:
            continue
        step_id = str(raw.get("id", "")).strip() or f"s{len(steps) + 1}"
        while step_id in seen:
            step_id = f"{step_id}b"
        seen.add(step_id)

        kind = str(raw.get("kind", "generate")).strip().lower()
        if kind not in ("generate", "decide"):
            kind = "generate"

        answer_type = str(raw.get("answer_type", "")).strip().lower()
        options = [str(option).strip() for option in (raw.get("options") or []) if str(option).strip()]
        if kind == "decide":
            if answer_type not in DECIDE_TYPES:
                answer_type = "choice" if len(options) >= 2 else "noul"
            if answer_type == "choice" and len(options) < 2:
                # A Choice question without options cannot be asked; degrade to a
                # single graded truth value rather than inventing options for it.
                answer_type = "noul"
                options = []
            if answer_type == "score" and len(options) < 2:
                answer_type = "noul"
                options = []
        else:
            answer_type = ""
            options = []

        criteria = str(raw.get("criteria", "")).strip()
        if not criteria:
            criteria = f"the output of step {step_id} satisfies: {goal}"

        context_needed = [
            str(item).strip() for item in (raw.get("context_needed") or []) if str(item).strip()
        ]
        depends_on = [
            str(item).strip() for item in (raw.get("depends_on") or []) if str(item).strip()
        ]
        steps.append(
            Step(
                id=step_id,
                goal=goal,
                kind=kind,
                criteria=criteria,
                answer_type=answer_type,
                options=options,
                context_needed=context_needed,
                depends_on=depends_on,
            )
        )
    if not steps:
        return [], "no usable steps in planner JSON"
    return steps, ""


def fallback_plan(task: str) -> Step:
    return Step(
        id="s1",
        goal=f"produce the deliverable for the task: {task}",
        kind="generate",
        criteria="the output answers the task directly and is complete",
    )


class LLMPlanner:
    """Asks a tier to decompose the task, then validates whatever came back.

    Planning is a structured-output job, so it asks for JSON mode and a token cap
    of its own. Both matter on reasoning models, where the chain of thought is
    charged against the same cap as the answer: too small a cap yields a reply that
    is all thinking and no JSON, which looks like a parse error but is really a
    budget problem. The failure is reported as what it is.
    """

    def __init__(self, provider: Provider, tier: str, max_steps: int, max_tokens: int = 2048) -> None:
        self.provider = provider
        self.tier = tier
        self.max_steps = max_steps
        self.max_tokens = max_tokens

    def plan(self, task: str, dry_run: bool = False) -> Plan:
        if dry_run:
            return HeuristicPlanner(self.tier).plan(task, dry_run=True)
        system = PLAN_SYSTEM.replace("<<MAX_STEPS>>", str(self.max_steps))
        try:
            completion: Completion = self.provider.complete(
                system, f"Task: {task}", max_tokens=self.max_tokens, json_mode=True
            )
        except Exception as exc:  # planner failure must not kill the run
            return Plan(
                task=task,
                steps=[fallback_plan(task)],
                tier=self.tier,
                fell_back=True,
                error=str(exc),
            )
        if completion.truncated:
            return Plan(
                task=task,
                steps=[fallback_plan(task)],
                tier=self.tier,
                model=completion.model,
                tokens_in=completion.tokens_in,
                tokens_out=completion.tokens_out,
                cost_usd=completion.cost_usd,
                stub=completion.stub,
                fell_back=True,
                error=(
                    f"planner output was cut off at max_tokens={self.max_tokens} "
                    f"(raised via [agent] planner_max_tokens)"
                ),
            )
        steps, error = parse_plan(completion.text, task, self.max_steps)
        if not steps:
            return Plan(
                task=task,
                steps=[fallback_plan(task)],
                tier=self.tier,
                model=completion.model,
                tokens_in=completion.tokens_in,
                tokens_out=completion.tokens_out,
                cost_usd=completion.cost_usd,
                stub=completion.stub,
                fell_back=True,
                error=f"{error}; reply began: {_excerpt(completion.text)}",
            )
        if any(step.is_decision for step in steps):
            # Decide first, act second: put the classification steps up front so
            # later steps can be routed with that knowledge already in context.
            steps.sort(key=lambda step: 0 if step.is_decision else 1)
        return Plan(
            task=task,
            steps=steps,
            tier=self.tier,
            model=completion.model,
            tokens_in=completion.tokens_in,
            tokens_out=completion.tokens_out,
            cost_usd=completion.cost_usd,
            stub=completion.stub,
        )


class HeuristicPlanner:
    """Offline planner used by --dry-run and by tests. Deterministic and dumb on purpose."""

    def __init__(self, tier: str) -> None:
        self.tier = tier

    def plan(self, task: str, dry_run: bool = True) -> Plan:
        steps: list[Step] = [
            Step(
                id="classify",
                goal="classify what kind of work this task is",
                kind="decide",
                answer_type="choice",
                options=["code-change", "analysis", "research", "writing"],
                criteria="the single label that best matches the task",
            )
        ]
        parts = _split(task)
        previous = ["classify"]
        for index, part in enumerate(parts, start=1):
            step_id = f"s{index}"
            steps.append(
                Step(
                    id=step_id,
                    goal=part,
                    kind="generate",
                    criteria=f"the output for '{part}' is complete and directly usable",
                    depends_on=list(previous),
                )
            )
            previous.append(step_id)
        return Plan(task=task, steps=steps, tier=self.tier, model="heuristic", stub=True)


def _excerpt(text: str, limit: int = 200) -> str:
    flat = " ".join(text.split())
    return repr(flat[:limit]) if flat else "(empty reply)"


def _split(task: str) -> list[str]:
    parts = [task.strip()]
    for separator in SPLITTERS:
        expanded: list[str] = []
        for part in parts:
            expanded.extend(chunk.strip() for chunk in part.split(separator) if chunk.strip())
        parts = expanded
    return [part for part in parts if len(part) > 3][:6] or [task.strip()]


def build_planner(config: Config, providers: dict[str, Provider]) -> LLMPlanner | HeuristicPlanner:
    tier = config.planner_tier()
    if config.agent.planner_mode == "heuristic" or tier.name not in providers:
        return HeuristicPlanner(tier.name)
    return LLMPlanner(
        providers[tier.name],
        tier.name,
        config.agent.max_steps,
        max_tokens=config.agent.planner_max_tokens,
    )


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a model reply, fences and prose included."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    try:
        decoded = json.loads(stripped)
        return decoded if isinstance(decoded, dict) else None
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        decoded = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def plan_to_json(plan: Plan) -> str:
    return json.dumps(
        {
            "task": plan.task,
            "tier": plan.tier,
            "model": plan.model,
            "fell_back": plan.fell_back,
            "error": plan.error,
            "steps": [step.to_dict() for step in plan.steps],
        },
        indent=2,
    )
