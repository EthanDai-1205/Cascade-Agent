"""Split-quality eval: is the cheap tier good enough, and where should the gate sit?

The loop has one number that decides whether it saves money or quietly ships bad
work: ``escalate_below``. This module measures it instead of guessing.

For every generative step in a task set it produces a **pair**: the same prompt
sent to the cheap tier and to the strong tier. Then it asks four independent
questions about that pair:

1. What does Jev's Noul gate say about the cheap output?
2. What does Jev's Noul gate say about the strong output? (reference point)
3. What does Jev say about a deliberately **corrupted** version of the cheap
   output? (does the gate have teeth, or would it wave anything through)
4. Which output does a **blind judge** prefer, and is each one sufficient for the
   step's criteria?

The judge never learns which output came from which tier, and the presentation
order is randomized per pair so position bias can be measured rather than assumed.

With those labels the threshold sweep is arithmetic: for each candidate
threshold, how often would the gate have escalated, how much of that escalation
was wasted on an output that was already good enough, and how often would a bad
output have been kept. The recommendation is the cheapest threshold whose miss
rate stays inside a stated tolerance, and the whole curve is printed so the
trade-off is visible rather than hidden behind one number.

Known limitation, stated up front: the judge is a model, by default the same
strong tier that produced one side of the pair. A judge grading its own output
is biased toward it, which inflates how often the strong side "wins" and makes
the recommended threshold conservative (too eager to escalate). Point
``[eval] judge_tier`` at a different model family to remove that bias.
"""

from __future__ import annotations

import json
import random
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .agent import STEP_SYSTEM, CascadeAgent, _JevUse
from .config import Config, TierConfig
from .jev import Jev, choice_question, noul_question
from .planner import Step, _extract_json
from .providers import Provider

JUDGE_SYSTEM = """You grade two candidate outputs for one step of a job.

You do not know who or what produced each candidate, and you must not speculate
about it. Judge only against the stated criteria.

Be strict about correctness and completeness. Do not reward length, confident
tone, or formatting. An output that is incomplete, that misses part of the
criteria, or that would not be directly usable by the next step is NOT sufficient.

Reply with JSON only:
{"a_sufficient": true|false, "b_sufficient": true|false, "better": "A"|"B"|"tie",
 "reason": "one sentence"}"""


@dataclass
class TaskCase:
    id: str
    task: str
    note: str = ""


@dataclass
class PairSample:
    task_id: str
    step_id: str
    goal: str
    criteria: str
    cheap_text: str = ""
    strong_text: str = ""
    cheap_p: float = 0.0
    strong_p: float = 0.0
    corrupt_p: float = 0.0
    cheap_cost: float = 0.0
    strong_cost: float = 0.0
    judge_cost: float = 0.0
    jev_cost: float = 0.0
    verdict: str = "unparsed"
    judge_note: str = ""
    judge_order: str = ""
    route_choice: str = ""
    route_confidence: float = 0.0
    planner_fell_back: bool = False
    stub: bool = False

    @property
    def cheap_sufficient(self) -> bool | None:
        if self.verdict == "cheap-ok":
            return True
        if self.verdict in ("strong-better", "both-bad"):
            return False
        return None

    @property
    def strong_sufficient(self) -> bool | None:
        if self.verdict in ("cheap-ok",):
            return True
        if self.verdict == "strong-better":
            return True
        if self.verdict == "both-bad":
            return False
        return None


@dataclass
class ThresholdRow:
    threshold: float
    escalate_rate: float
    wasted_rate: float
    missed_rate: float
    corrupt_pass_rate: float
    expected_cost: float

    def to_dict(self) -> dict[str, float]:
        return {
            "threshold": round(self.threshold, 3),
            "escalate_rate": round(self.escalate_rate, 4),
            "wasted_rate": round(self.wasted_rate, 4),
            "missed_rate": round(self.missed_rate, 4),
            "corrupt_pass_rate": round(self.corrupt_pass_rate, 4),
            "expected_cost": round(self.expected_cost, 6),
        }


@dataclass
class EvalReport:
    samples: list[PairSample] = field(default_factory=list)
    rows: list[ThresholdRow] = field(default_factory=list)
    recommended: float = 0.0
    recommended_reason: str = ""
    judge_tier: str = ""
    cheap_tier: str = ""
    strong_tier: str = ""
    stub: bool = False
    model_cost_usd: float = 0.0
    jev_cost_usd: float = 0.0
    judge_cost_usd: float = 0.0
    planner_failures: int = 0
    mean_cheap_cost: float = 0.0
    mean_strong_cost: float = 0.0
    unparsed_judgements: int = 0
    judge_failures: int = 0
    generation_failures: int = 0
    gate_failures: int = 0

    @property
    def judged_verdicts(self) -> tuple[str, ...]:
        return ("cheap-ok", "strong-better", "both-bad")

    @property
    def verified_pairs(self) -> int:
        """Pairs the judge actually ruled on, which is the only usable sample."""

        return sum(1 for s in self.samples if s.verdict in self.judged_verdicts)

    # ------------------------------------------------------------- derived stats

    @property
    def total_cost_usd(self) -> float:
        return self.model_cost_usd + self.jev_cost_usd + self.judge_cost_usd

    def gate_separation(self) -> tuple[float, float, float]:
        """Mean gate probability for real outputs, for corrupted ones, and the gap."""

        if not self.samples:
            return (0.0, 0.0, 0.0)
        real = sum(s.cheap_p for s in self.samples) / len(self.samples)
        corrupt = sum(s.corrupt_p for s in self.samples) / len(self.samples)
        return (real, corrupt, real - corrupt)

    def discrimination(self) -> tuple[float, float, float]:
        """Mean gate probability on cheap outputs the judge called sufficient vs not.

        This is the number that decides whether any threshold can work. The gate only
        has to rank, so a positive gap means the signal is there; a gap near zero means
        the threshold is a coin flip dressed up as policy.
        """

        sufficient = [s.cheap_p for s in self.samples if s.cheap_sufficient is True]
        short = [s.cheap_p for s in self.samples if s.cheap_sufficient is False]
        if not sufficient or not short:
            only = sufficient or short
            mean = sum(only) / len(only) if only else 0.0
            return (mean, mean, 0.0)
        mean_ok = sum(sufficient) / len(sufficient)
        mean_short = sum(short) / len(short)
        return (mean_ok, mean_short, mean_ok - mean_short)

    def gate_is_decorative(self, epsilon: float = 0.05) -> bool:
        """True when the gate's probability fails to separate sufficient from insufficient.

        A threshold sitting on top of a signal that does not separate anything is
        decoration: it will either escalate everything or nothing, and either way the
        number in the config is not what is deciding the outcome.
        """

        _, _, gap = self.discrimination()
        sufficient = sum(1 for s in self.samples if s.cheap_sufficient is True)
        insufficient = sum(1 for s in self.samples if s.cheap_sufficient is False)
        if not sufficient or not insufficient:
            return True
        return abs(gap) < epsilon

    def position_bias(self) -> tuple[int, int, float]:
        """How often the judge preferred slot A, when slot A held the cheap output."""

        a_win = sum(1 for s in self.samples if s.judge_order == "cheap-first" and s.verdict == "cheap-ok")
        a_win += sum(
            1 for s in self.samples if s.judge_order == "strong-first" and s.verdict == "strong-better"
        )
        judged = sum(1 for s in self.samples if s.verdict in self.judged_verdicts)
        if not judged:
            return (a_win, 0, 0.0)
        return (a_win, judged, a_win / judged)

    def routing_stats(self) -> tuple[int, int, int]:
        """(under-routed, over-routed, judged) where under means cheap was not enough.

        Only judged pairs count: a pair whose generation or judgement failed says
        nothing about whether the router chose well.
        """

        judged = [s for s in self.samples if s.verdict in self.judged_verdicts]
        choices = [s.route_choice for s in judged if s.route_choice]
        cheapest = min(choices, default="")
        strongest = max(choices, default="")
        under = sum(1 for s in judged if s.route_choice == cheapest and s.verdict == "strong-better")
        over = sum(1 for s in judged if s.route_choice == strongest and s.verdict == "cheap-ok")
        return (under, over, len(judged))

    def worst_missed(self) -> float:
        return max((row.missed_rate for row in self.rows), default=0.0)

    def to_dict(self) -> dict[str, Any]:
        real, corrupt, gap = self.gate_separation()
        under, over, routed = self.routing_stats()
        a_win, judged, bias = self.position_bias()
        ok_p, short_p, discrimination_gap = self.discrimination()
        return {
            "judge_tier": self.judge_tier,
            "cheap_tier": self.cheap_tier,
            "strong_tier": self.strong_tier,
            "stub": self.stub,
            "pairs": len(self.samples),
            "unparsed_judgements": self.unparsed_judgements,
            "judge_failures": self.judge_failures,
            "generation_failures": self.generation_failures,
            "gate_failures": self.gate_failures,
            "planner_failures": self.planner_failures,
            "costs": {
                "model_usd": round(self.model_cost_usd, 6),
                "jev_usd": round(self.jev_cost_usd, 6),
                "judge_usd": round(self.judge_cost_usd, 6),
                "total_usd": round(self.total_cost_usd, 6),
            },
            "mean_cost_per_step": {
                "cheap": round(self.mean_cheap_cost, 6),
                "strong": round(self.mean_strong_cost, 6),
            },
            "gate": {
                "mean_p_real": round(real, 4),
                "mean_p_corrupted": round(corrupt, 4),
                "separation": round(gap, 4),
                "mean_p_when_sufficient": round(ok_p, 4),
                "mean_p_when_insufficient": round(short_p, 4),
                "discrimination": round(discrimination_gap, 4),
                "sufficient_samples": sum(1 for s in self.samples if s.cheap_sufficient is True),
                "insufficient_samples": sum(1 for s in self.samples if s.cheap_sufficient is False),
            },
            "routing": {"under_routed": under, "over_routed": over, "steps": routed},
            "judge_position": {"a_wins": a_win, "judged": judged, "a_win_rate": round(bias, 4)},
            "gate_is_decorative": self.gate_is_decorative(),
            "recommended_threshold": round(self.recommended, 3),
            "recommended_reason": self.recommended_reason,
            "sweep": [row.to_dict() for row in self.rows],
            "samples": [
                {
                    "task": s.task_id,
                    "step": s.step_id,
                    "goal": s.goal[:120],
                    "cheap_p": round(s.cheap_p, 4),
                    "strong_p": round(s.strong_p, 4),
                    "corrupt_p": round(s.corrupt_p, 4),
                    "verdict": s.verdict,
                    "route": s.route_choice,
                    "route_confidence": round(s.route_confidence, 3),
                    "judge_order": s.judge_order,
                    "judge_note": s.judge_note,
                }
                for s in self.samples
            ],
        }

    def render(self, config: Config) -> str:
        lines: list[str] = []
        if self.stub:
            lines.append("NOTE: stub tiers and/or stub Jev were used. These numbers are plumbing only.")
        cur = config.currency
        lines.append(
            f"pairs {len(self.samples)}  {self.cheap_tier} vs {self.strong_tier}  judge {self.judge_tier}  "
            f"model {self.model_cost_usd:.4f} {cur}  jev {self.jev_cost_usd:.4f} {cur}  "
            f"judge {self.judge_cost_usd:.4f} {cur}  total {self.total_cost_usd:.4f} {cur}"
        )
        if self.judge_tier and self.judge_tier == self.strong_tier:
            lines.append(
                "NOTE: the judge is the same tier that produced the strong side of every pair "
                "(self-grading). Expect escalation to look more necessary than it is."
            )
        elif self.judge_tier:
            lines.append(
                f"NOTE: the judge ({self.judge_tier}) is an outside tier, not one of the two sides, "
                "so these verdicts are not the strong tier grading itself."
            )
        if self.planner_failures:
            lines.append(f"planner JSON failures: {self.planner_failures} (the step list fell back to one step)")
        if self.unparsed_judgements:
            lines.append(f"unparsed judge replies: {self.unparsed_judgements} (excluded from the sweep)")
        if self.judge_failures:
            lines.append(
                f"judge calls that failed outright: {self.judge_failures} "
                f"(excluded from the sweep; usually a judge token cap that is too small)"
            )
        if self.generation_failures:
            lines.append(
                f"generations that failed outright: {self.generation_failures} "
                f"(excluded from the sweep; usually a reasoning model burning its whole cap)"
            )
        if self.gate_failures:
            lines.append(
                f"gate or routing calls that failed outright: {self.gate_failures} "
                f"(excluded from the sweep; usually a dropped connection or a timeout)"
            )

        real, corrupt, gap = self.gate_separation()
        lines.append("")
        lines.append(f"gate teeth: mean p on real outputs {real:.2f}, on corrupted outputs {corrupt:.2f}, "
                     f"separation {gap:+.2f}")
        verdicts = _tally(s.verdict for s in self.samples)
        lines.append(
            "judge verdicts: "
            + ", ".join(f"{name} {count}" for name, count in sorted(verdicts.items()))
        )
        ok_p, short_p, discrimination_gap = self.discrimination()
        sufficient = sum(1 for s in self.samples if s.cheap_sufficient is True)
        insufficient = sum(1 for s in self.samples if s.cheap_sufficient is False)
        if sufficient and insufficient:
            lines.append(
                f"gate discrimination: mean p {ok_p:.2f} on the {sufficient} outputs the judge called "
                f"sufficient, {short_p:.2f} on the {insufficient} it did not, gap {discrimination_gap:+.2f}"
            )
            if self.gate_is_decorative():
                lines.append(
                    "  WARNING: that gap is too small to be a signal on this sample. The threshold is "
                    "decoration here: it will escalate everything or nothing, and the choice between "
                    "tiers is really being made by the default tier, not by the gate. Either gather more "
                    "pairs, sharpen the step criteria, or read the choice as a plain cost/quality "
                    "trade-off between all-cheap and all-strong."
                )
        else:
            lines.append(
                f"gate discrimination: not measurable here ({sufficient} sufficient, {insufficient} "
                f"insufficient cheap outputs in this run)"
            )
        under, over, routed = self.routing_stats()
        lines.append(f"routing: under-routed {under}/{routed}, over-routed {over}/{routed} (against the judge)")
        a_win, judged, bias = self.position_bias()
        lines.append(f"judge position bias: slot A won {a_win}/{judged} ({bias:.0%}, blind order per pair)")

        lines.append("")
        header = (
            f"{'threshold':>9}{'escalates':>11}{'wasted':>9}{'missed':>9}"
            f"{'corrupt pass':>14}{'cost/step':>11}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for row in self.rows:
            marker = " <-- recommended" if abs(row.threshold - self.recommended) < 1e-9 else ""
            lines.append(
                f"{row.threshold:>9.2f}{row.escalate_rate:>10.0%}{row.wasted_rate:>9.0%}"
                f"{row.missed_rate:>9.0%}{row.corrupt_pass_rate:>13.0%}"
                f"{row.expected_cost:>11.5f}{marker}"
            )
        lines.append("")
        lines.append(
            f"escalates = share of steps the gate sends to the strong tier, wasted = escalated even though "
            f"the cheap output was sufficient,"
        )
        lines.append(
            "missed = kept a cheap output the judge called insufficient, corrupt pass = an obviously "
            "truncated output the gate waved through."
        )
        lines.append("")
        lines.append("cost/step counts model tokens only; the gate and judge calls are roughly constant per step.")
        lines.append("")
        lines.append(
            f"recommended escalate_below = {self.recommended:.2f}: {self.recommended_reason}"
        )
        lines.append(
            f"(judged pairs {self.verified_pairs}, floor {config.eval.min_judged_pairs}, "
            f"tolerance {config.eval.miss_tolerance:.0%})"
        )

        if self.mean_strong_cost > 0:
            all_strong = self.mean_strong_cost + self.judge_cost_usd / max(1, len(self.samples))
            row = next((r for r in self.rows if abs(r.threshold - self.recommended) < 1e-9), None)
            if row is not None:
                saved = 1.0 - (row.expected_cost + self.judge_cost_usd / max(1, len(self.samples))) / all_strong
                lines.append(
                    f"at that threshold a step costs about {row.expected_cost:.5f} {cur} against "
                    f"{self.mean_strong_cost:.5f} {cur} if everything went to the strong tier "
                    f"({saved:+.0%} on model spend)."
                )
        lines.append("")
        if self.judge_tier and self.judge_tier == self.strong_tier:
            lines.append(
                "Caveat: the judge is a model, and here it is the same strong tier that produced one "
                "side of each pair. That bias makes escalation look more necessary than it is."
            )
        else:
            lines.append(
                "Caveat: the judge is still a model. An outside judge removes self-grading bias but "
                "not its own family's preferences, so read one run as a direction, not a proof."
            )
        return "\n".join(lines)


def load_task_set(path: str | Path) -> list[TaskCase]:
    target = Path(path).expanduser()
    if not target.is_file():
        raise FileNotFoundError(f"task set not found: {target}")
    with target.open("rb") as handle:
        data = tomllib.load(handle)
    rows = data.get("tasks") or []
    cases: list[TaskCase] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        task = str(row.get("task", "")).strip()
        if not task:
            continue
        cases.append(
            TaskCase(
                id=str(row.get("id", f"task{index + 1}")),
                task=task,
                note=str(row.get("note", "")),
            )
        )
    if not cases:
        raise ValueError(f"no usable [[tasks]] entries in {target}")
    return cases


def corrupt(text: str, ratio: float) -> str:
    """Deterministically truncate an output, the most common real failure mode."""

    if not text.strip():
        return "<empty>"
    keep = max(1, int(len(text) * (1.0 - ratio)))
    return text[:keep].rstrip()


def judge_pair(
    provider: Provider,
    step: Step,
    cheap: str,
    strong: str,
    order_seed: int,
    max_tokens: int = 512,
) -> dict[str, Any]:
    """Ask the judge which candidate is better, blind, with a randomized order.

    The token cap matters more than it looks: a reasoning judge spends its budget
    thinking before it answers, so too small a cap yields an empty reply rather than
    a wrong verdict, and the pair has to be dropped.
    """

    rng = random.Random(order_seed)
    cheap_first = rng.random() < 0.5
    slot_a, slot_b = (cheap, strong) if cheap_first else (strong, cheap)
    user = "\n".join(
        [
            f"Step goal: {step.goal}",
            f"Criteria this step must satisfy: {step.criteria}",
            "",
            "-- CANDIDATE A --",
            slot_a or "<empty>",
            "",
            "-- CANDIDATE B --",
            slot_b or "<empty>",
        ]
    )
    completion = provider.complete(JUDGE_SYSTEM, user, max_tokens=max_tokens, json_mode=True)
    parsed = _extract_json(completion.text) or {}
    a_sufficient = _as_bool(parsed.get("a_sufficient"))
    b_sufficient = _as_bool(parsed.get("b_sufficient"))
    better = str(parsed.get("better", "")).strip().lower()

    if a_sufficient is None or b_sufficient is None:
        verdict = "unparsed"
    else:
        cheap_sufficient = a_sufficient if cheap_first else b_sufficient
        if cheap_sufficient:
            verdict = "cheap-ok"
        elif (b_sufficient if cheap_first else a_sufficient):
            verdict = "strong-better"
        else:
            verdict = "both-bad"

    return {
        "verdict": verdict,
        "order": "cheap-first" if cheap_first else "strong-first",
        "note": str(parsed.get("reason", ""))[:200],
        "better": better,
        "cost_usd": completion.cost_usd,
        "tokens_in": completion.tokens_in,
        "tokens_out": completion.tokens_out,
        "stub": completion.stub,
    }


def run_eval(
    config: Config,
    task_set: str | Path | None = None,
    providers: dict[str, Provider] | None = None,
    jev: Jev | None = None,
    dry_run: bool = False,
    max_pairs: int | None = None,
    on_event: Callable[[str], None] | None = None,
) -> EvalReport:
    """Build pairs across the task set and measure where the gate should sit."""

    def emit(message: str) -> None:
        if on_event is not None:
            on_event(message)
        elif config.agent.verbose:
            print(message, flush=True)

    if len(config.executor_tiers) < 2:
        raise ValueError(
            "the eval needs at least two tiers that can run a step (one cheap, one strong) to "
            f"compare; this config defines {config.executor_names} as executors"
            + (
                f" and {[t.name for t in config.tiers if t.judge_only]} as judge-only"
                if any(t.judge_only for t in config.tiers)
                else ""
            )
        )

    cases = load_task_set(task_set or config.eval.task_set)
    budget = max_pairs or config.eval.max_pairs
    agent = CascadeAgent(config, providers=providers, jev=jev, dry_run=dry_run, on_event=on_event)
    provider_map = agent.providers
    jev_engine = agent.jev
    cheap_tier = config.executor_tiers[0]
    strong_tier = config.strong_tier(config.eval.strong_tier)
    judge_tier = config.tier(config.eval.judge_tier)
    judge = provider_map[config.eval.judge_tier]
    if judge_tier.judge_only:
        emit(
            f"  judge {judge_tier.name} ({judge_tier.model}) is judge_only: it does not run steps "
            f"and the pairs compare {cheap_tier.name} against {strong_tier.name}"
        )
    report = EvalReport(
        judge_tier=judge_tier.name,
        cheap_tier=cheap_tier.name,
        strong_tier=strong_tier.name,
        stub=agent.stubbed,
    )

    cheap_costs: list[float] = []
    strong_costs: list[float] = []
    seed = config.eval.seed

    for case in cases:
        if len(report.samples) >= budget:
            break
        try:
            plan = agent.plan(case.task)
        except Exception as exc:
            emit(f"  plan failed for {case.id}: {exc}")
            continue
        if plan.fell_back:
            report.planner_failures += 1
        generate_steps = [step for step in plan.steps if not step.is_decision]
        for step in generate_steps:
            if len(report.samples) >= budget:
                break
            index = len(report.samples)
            emit(f"  {case.id}/{step.id}: generating a pair")
            context = agent._context(step, {})  # noqa: SLF001 (same package, shared prompt shape)
            user = agent._state(case.task, step, context)  # noqa: SLF001
            sample = PairSample(
                task_id=case.id,
                step_id=step.id,
                goal=step.goal,
                criteria=step.criteria,
                planner_fell_back=plan.fell_back,
            )

            failed_generation = ""
            failed_gate = ""
            for tier, attr_text, attr_p, attr_cost in (
                (cheap_tier, "cheap_text", "cheap_p", "cheap_cost"),
                (strong_tier, "strong_text", "strong_p", "strong_cost"),
            ):
                try:
                    completion = provider_map[tier.name].complete(
                        STEP_SYSTEM, user, max_tokens=config.eval.max_tokens or None
                    )
                except Exception as exc:
                    report.generation_failures += 1
                    failed_generation = f"{tier.name}: {exc}"
                    emit(f"    {tier.name} generation failed on {case.id}/{step.id}: {exc}")
                    break
                setattr(sample, attr_text, completion.text)
                setattr(sample, attr_cost, completion.cost_usd)
                report.model_cost_usd += completion.cost_usd
                sample.stub = sample.stub or completion.stub
                try:
                    gate = jev_engine.ask(
                        agent._state(case.task, step, "", candidate=completion.text),  # noqa: SLF001
                        {"meets": noul_question(step.criteria, "the criteria are not satisfied")},
                    )
                except Exception as exc:
                    # One unusable gate call drops one pair. It must not end the run:
                    # a flaky connection would otherwise take the whole measurement with it.
                    report.gate_failures += 1
                    failed_gate = f"{tier.name} gate: {exc}"
                    emit(f"    {tier.name} gate failed on {case.id}/{step.id}: {exc}")
                    break
                setattr(sample, attr_p, gate.noul("meets"))
                report.jev_cost_usd += gate.cost_usd
                sample.jev_cost += gate.cost_usd

            if failed_generation or failed_gate:
                sample.verdict = "gen-error" if failed_generation else "gate-error"
                sample.judge_note = (failed_generation or failed_gate)[:200]
                report.samples.append(sample)
                emit(f"    pair dropped: {(failed_generation or failed_gate)[:120]}")
                continue

            corrupted = corrupt(sample.cheap_text, config.eval.corrupt_ratio)
            try:
                gate = jev_engine.ask(
                    agent._state(case.task, step, "", candidate=corrupted),  # noqa: SLF001
                    {"meets": noul_question(step.criteria, "the criteria are not satisfied")},
                )
            except Exception as exc:
                report.gate_failures += 1
                sample.verdict = "gate-error"
                sample.judge_note = f"corrupt gate: {exc}"[:200]
                report.samples.append(sample)
                emit(f"    corrupt gate failed on {case.id}/{step.id}: {exc}")
                continue
            sample.corrupt_p = gate.noul("meets")
            report.jev_cost_usd += gate.cost_usd
            sample.jev_cost += gate.cost_usd

            try:
                picked, confidence, route_use = agent.route(case.task, step, context)
            except Exception as exc:
                # Routing is a statistic here, not a result, so a failed route costs the
                # pair its routing datapoint rather than the whole pair.
                report.gate_failures += 1
                picked, confidence, route_use = cheap_tier.name, 0.0, _JevUse()
                emit(f"    routing failed on {case.id}/{step.id} ({exc}); recorded as unrouted")
            sample.route_choice = picked
            sample.route_confidence = confidence
            report.jev_cost_usd += route_use.cost_usd
            sample.jev_cost += route_use.cost_usd

            try:
                verdict = judge_pair(
                    judge,
                    step,
                    sample.cheap_text,
                    sample.strong_text,
                    seed + index,
                    max_tokens=config.eval.judge_max_tokens,
                )
            except Exception as exc:
                # One failed judge call must not void the whole measurement.
                report.judge_failures += 1
                sample.verdict = "judge-error"
                sample.judge_note = str(exc)[:200]
                verdict = {"verdict": "judge-error", "order": "", "note": str(exc)[:200], "cost_usd": 0.0}
                emit(f"    judge failed on {case.id}/{step.id}: {exc}")
            sample.verdict = verdict["verdict"]
            sample.judge_order = verdict["order"]
            sample.judge_note = verdict["note"]
            sample.judge_cost = verdict["cost_usd"]
            report.judge_cost_usd += verdict["cost_usd"]
            if verdict["verdict"] == "unparsed":
                report.unparsed_judgements += 1

            cheap_costs.append(sample.cheap_cost)
            strong_costs.append(sample.strong_cost)
            report.samples.append(sample)
            emit(
                f"    cheap p={sample.cheap_p:.2f} (corrupt {sample.corrupt_p:.2f}), "
                f"strong p={sample.strong_p:.2f}, judge: {sample.verdict}, route: {picked}"
            )

    report.mean_cheap_cost = sum(cheap_costs) / len(cheap_costs) if cheap_costs else 0.0
    report.mean_strong_cost = sum(strong_costs) / len(strong_costs) if strong_costs else 0.0
    report.rows = sweep(report, config)
    report.recommended, report.recommended_reason = _recommend(report, config)
    agent.ledger.close()
    return report


def sweep(report: EvalReport, config: Config) -> list[ThresholdRow]:
    """For each candidate threshold, work out what the gate would have done."""

    usable = [s for s in report.samples if s.verdict in report.judged_verdicts]
    if not usable:
        return []
    mean_cheap = sum(s.cheap_cost for s in usable) / len(usable)
    mean_strong = sum(s.strong_cost for s in usable) / len(usable)
    rows: list[ThresholdRow] = []
    step = config.eval.threshold_step
    counts = max(1, round(1.0 / step))
    for index in range(1, counts + 1):
        threshold = round(index * step, 3)
        escalated = [s for s in usable if s.cheap_p < threshold]
        rows.append(
            ThresholdRow(
                threshold=threshold,
                escalate_rate=len(escalated) / len(usable),
                wasted_rate=sum(1 for s in escalated if s.verdict == "cheap-ok") / len(usable),
                missed_rate=sum(
                    1 for s in usable if s.cheap_p >= threshold and s.verdict == "strong-better"
                )
                / len(usable),
                corrupt_pass_rate=sum(1 for s in usable if s.corrupt_p >= threshold) / len(usable),
                expected_cost=mean_cheap + (len(escalated) / len(usable)) * mean_strong,
            )
        )
    return rows


def _recommend(report: EvalReport, config: Config) -> tuple[float, str]:
    """The cheapest threshold whose miss rate stays inside the stated tolerance.

    Three ways this refuses to answer, all of them deliberate:

    * too few judged pairs (``[eval] min_judged_pairs``): the honest output of a tiny
      sample is "not enough data", not a number with a decimal point
    * a threshold that escalates every step is excluded, because that is not a cascade,
      it is paying for the strong model with extra steps
    * if nothing stays inside the tolerance, the least-bad threshold is reported as such
    """

    if not report.rows:
        return (config.jev.escalate_below, "no usable pairs, keeping the configured threshold")
    if report.verified_pairs < config.eval.min_judged_pairs:
        return (
            config.jev.escalate_below,
            f"only {report.verified_pairs} judged pairs, below the {config.eval.min_judged_pairs} "
            f"needed to recommend a change; keeping the configured threshold",
        )

    tolerance = config.eval.miss_tolerance
    candidates = [row for row in report.rows if row.escalate_rate < 1.0] or report.rows
    acceptable = [row for row in candidates if row.missed_rate <= tolerance]
    if acceptable:
        best = min(acceptable, key=lambda row: row.expected_cost)
        return (
            best.threshold,
            f"cheapest threshold that keeps missed shortfalls at or under {tolerance:.0%} "
            f"(missed {best.missed_rate:.0%} here)",
        )
    best = min(candidates, key=lambda row: (row.missed_rate, row.expected_cost))
    return (
        best.threshold,
        f"no threshold held missed shortfalls under {tolerance:.0%}; this one missed least "
        f"({best.missed_rate:.0%})",
    )


def _tally(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return out


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "sufficient"):
            return True
        if lowered in ("false", "no", "insufficient"):
            return False
    return None


def report_to_json(report: EvalReport) -> str:
    return json.dumps(report.to_dict(), indent=2)
