"""Shared test scaffolding: a programmable Jev and ready-made configs.

Everything here is offline. No test in this suite touches the network, so the
suite runs in CI and on a plane alike.
"""

from __future__ import annotations

from typing import Callable

from .config import Config, Price, config_from_dict
from .jev import JevResult
from .planner import Plan, Step
from .providers import ProviderError, ScriptedProvider

CHEAP_PRICE = {"input": 1.0, "output": 2.0, "currency": "USD"}
EXPENSIVE_PRICE = {"input": 10.0, "output": 20.0, "currency": "USD"}


def base_config(
    verify: bool = True,
    candidates: int = 1,
    escalate_below: float = 0.6,
    max_escalations: int = 1,
    same_tier_retries: int = 0,
    **agent_overrides,
) -> Config:
    """Two mock tiers with deliberately simple prices, plus a heuristic planner."""

    agent = {
        "planner_tier": "cheap",
        "planner_mode": "heuristic",
        "max_steps": 8,
        "max_cost_usd": 1.0,
        "verbose": False,
        "ledger_path": "",
    }
    agent.update(agent_overrides)
    return config_from_dict(
        {
            "agent": agent,
            "jev": {
                "verify": verify,
                "escalate_below": escalate_below,
                "candidates": candidates,
                "max_escalations": max_escalations,
                "same_tier_retries": same_tier_retries,
            },
            "tiers": [
                {
                    "name": "cheap",
                    "kind": "mock",
                    "model": "cheap-mock",
                    "description": "a small model",
                    "price": CHEAP_PRICE,
                },
                {
                    "name": "expensive",
                    "kind": "mock",
                    "model": "expensive-mock",
                    "description": "a strong model",
                    "price": EXPENSIVE_PRICE,
                },
            ],
        }
    )


def judge_config(judge_last: bool = True, **agent_overrides) -> Config:
    """Two mock executors plus a third mock tier that exists only to grade.

    Declared last by default, which is the shape that used to be dangerous: before
    ``judge_only`` existed, the eval took the last tier as the strong side of every
    pair, so a judge declared last would have been the thing being graded.
    """

    agent = {
        "planner_tier": "cheap",
        "planner_mode": "heuristic",
        "max_steps": 8,
        "max_cost_usd": 1.0,
        "verbose": False,
        "ledger_path": "",
    }
    agent.update(agent_overrides)
    cheap = {
        "name": "cheap",
        "kind": "mock",
        "model": "cheap-mock",
        "description": "a small model",
        "price": CHEAP_PRICE,
    }
    expensive = {
        "name": "expensive",
        "kind": "mock",
        "model": "expensive-mock",
        "description": "a strong model",
        "price": EXPENSIVE_PRICE,
    }
    judge = {
        "name": "judge",
        "kind": "mock",
        "model": "judge-mock",
        "description": "an outside model used only to grade",
        "price": EXPENSIVE_PRICE,
        "judge_only": True,
    }
    tiers = [cheap, expensive, judge] if judge_last else [cheap, judge, expensive]
    return config_from_dict(
        {
            "agent": agent,
            "jev": {"verify": True, "escalate_below": 0.6},
            "eval": {"judge_tier": "judge"},
            "tiers": tiers,
        }
    )


class ScriptedJev:
    """Deterministic System One stand-in with programmable answers.

    ``router`` sees the state and the offered options and returns one of them.
    ``verdicts`` are consumed in order by Noul questions (the verification gate).
    ``chooser`` answers Choice questions that pick among candidate drafts.
    """

    stub = True

    def __init__(
        self,
        router: Callable[[str, list[str]], str] | None = None,
        verdicts: list[float] | None = None,
        chooser: Callable[[str, list[str]], str] | None = None,
        price: Price | None = None,
    ) -> None:
        self.router = router or (lambda state, options: "jev" if "step_kind: decide" in state else "cheap")
        self.verdicts = list(verdicts or [])
        self.chooser = chooser or (lambda state, options: options[0] if options else "")
        self.price = price or Price(input=0.042, output=0.0)
        self.calls = 0
        self.states: list[str] = []
        self.questions: list[dict] = []
        self.total_tokens_in = 0
        self.total_cost_usd = 0.0

    def ask(self, state: str, questions: dict) -> JevResult:
        self.calls += 1
        self.states.append(state)
        self.questions.append(questions)
        answers: dict[str, dict] = {}
        for question_id, question in questions.items():
            options = _option_names(question)
            if question["type"] == "noul":
                p = self.verdicts.pop(0) if self.verdicts else 0.9
                answers[question_id] = {"type": "noul", "noul": p}
            elif question["type"] == "choice":
                if question_id == "executor":
                    pick = self.router(state, options)
                else:
                    pick = self.chooser(state, options)
                if pick not in options and options:
                    pick = options[0]
                answers[question_id] = {
                    "type": "choice",
                    "choice": pick,
                    "confidence": 0.9,
                    "probabilities": {option: (0.9 if option == pick else 0.1 / max(1, len(options) - 1)) for option in options},
                }
            elif question["type"] == "score":
                criteria = question["criteria"]
                answers[question_id] = {
                    "type": "score",
                    "score": float(len(criteria) - 1),
                    "confidence": 0.9,
                    "legend": {str(i): level for i, level in enumerate(criteria)},
                    "probabilities": {str(i): (0.9 if i == len(criteria) - 1 else 0.0) for i in range(len(criteria))},
                }
        tokens_in = max(1, len(state) // 4)
        self.total_tokens_in += tokens_in
        cost = self.price.cost(tokens_in, 8 * len(questions))
        self.total_cost_usd += cost
        return JevResult(
            answers=answers,
            model="scripted-jev",
            tokens_in=tokens_in,
            tokens_out=8 * len(questions),
            cost_usd=cost,
            latency_s=0.0,
            calls=1,
            stub=True,
        )


def _option_names(question: dict) -> list[str]:
    criteria = question.get("criteria")
    if question["type"] == "choice" and isinstance(criteria, dict):
        return list(criteria.keys())
    if question["type"] == "score" and isinstance(criteria, list):
        return list(criteria)
    return []


class StaticPlanner:
    """Returns a fixed plan. Tests use it to pin down exactly which steps run."""

    def __init__(self, steps: list[Step], tier: str = "cheap") -> None:
        self.steps = steps
        self.tier = tier

    def plan(self, task: str, dry_run: bool = True) -> Plan:
        return Plan(task=task, steps=list(self.steps), tier=self.tier, model="static", stub=True)


class FailingProvider:
    """Fails a fixed number of times, then behaves like a ScriptedProvider."""

    def __init__(self, name: str, outputs: list[str], failures: int = 1, price=None) -> None:
        self.tier = name
        self.stub = True
        self.model = f"{name}-flaky"
        self.failures = failures
        self.inner = ScriptedProvider(name, outputs, model=f"{name}-scripted", price=price)
        self.calls = 0
        self._error = ProviderError(f"{name} exploded", status=500)

    def complete(self, system: str, user: str):
        self.calls += 1
        if self.calls <= self.failures:
            raise self._error
        return self.inner.complete(system, user)
