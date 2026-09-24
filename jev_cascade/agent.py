"""The agent loop: decompose, route, execute, verify, escalate.

Jev (System One) shows up in five places, not just routing:

1. **Step typing side** the planner: steps whose answer is a fixed choice, a
   yes/no, or a position on a scale are marked ``decide``.
2. **Routing**: one Choice question picks the executor for the step from the
   configured tiers, plus ``jev`` itself when the step is a decision.
3. **Execution**: ``decide`` steps are answered by Jev directly, with no model
   tokens spent at all.
4. **Candidate selection**: when several cheap drafts exist, Jev picks the best
   one instead of paying a strong model to choose.
5. **Verification**: a Noul question scores the output against the step's own
   criteria, and a low probability triggers exactly one escalation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable

from .config import Config, TierConfig
from .jev import Jev, build_jev, choice_question, noul_question, score_question
from .ledger import Ledger, StepRecord
from .planner import Plan, Step, build_planner, plan_to_json
from .providers import Provider, ProviderError, build_provider

STEP_SYSTEM = """You are one step inside a larger plan.

Rules:
- produce ONLY this step's deliverable, nothing about other steps
- no preamble, no sign-off, no restating the plan
- if the step asks for code, return the code, minimally commented
- the step's acceptance criteria are shown; satisfy them exactly"""


@dataclass
class _JevUse:
    """What one or more System One calls cost, in calls, tokens and dollars."""

    cost_usd: float = 0.0
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    @classmethod
    def of(cls, result) -> _JevUse:
        return cls(result.cost_usd, result.calls, result.tokens_in, result.tokens_out)


@dataclass
class RunResult:
    task: str
    plan: Plan | None = None
    outputs: dict[str, str] = field(default_factory=dict)
    ledger: Ledger | None = None
    aborted: str = ""
    error: str = ""


class CascadeAgent:
    """Cheap by default, strong only when a verification gate says so."""

    def __init__(
        self,
        config: Config,
        providers: dict[str, Provider] | None = None,
        jev: Jev | None = None,
        dry_run: bool = False,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.dry_run = dry_run
        self.jev: Jev = jev if jev is not None else build_jev(config.jev, dry_run=dry_run)
        self.providers: dict[str, Provider] = (
            providers
            if providers is not None
            else {tier.name: build_provider(tier, dry_run=dry_run) for tier in config.tiers}
        )
        self.ledger = Ledger(path=config.agent.ledger_path)
        self.on_event = on_event
        self.planner = build_planner(config, self.providers)

    # ------------------------------------------------------------------ helpers

    def _emit(self, message: str) -> None:
        if self.on_event is not None:
            self.on_event(message)
        elif self.config.agent.verbose:
            print(message, flush=True)

    @property
    def stubbed(self) -> bool:
        return bool(self.jev.stub) or any(provider.stub for provider in self.providers.values())

    def _state(self, task: str, step: Step, context: str, candidate: str | None = None) -> str:
        lines = [
            f"task: {task}",
            f"step_id: {step.id}",
            f"step_kind: {step.kind}",
            f"step_goal: {step.goal}",
            f"step_criteria: {step.criteria}",
        ]
        if context:
            lines.append(f"context:\n{context}")
        if candidate is not None:
            lines.append(f"candidate output:\n{candidate or '<empty>'}")
        return "\n".join(lines)

    def _context(self, step: Step, outputs: dict[str, str]) -> str:
        parts = []
        for other_id, text in outputs.items():
            if other_id == step.id:
                continue
            # Honour an explicit dependency list; otherwise pass everything the
            # run has produced so far.
            if step.depends_on and other_id not in step.depends_on:
                continue
            parts.append(f"[{other_id}] {text}")
        if not parts:
            return ""
        joined = "\n\n".join(parts)
        if len(joined) > 6000:
            joined = joined[:3000] + "\n[...context elided...]\n" + joined[-3000:]
        return joined

    def _tier_options(self, step: Step) -> dict[str, str | None]:
        options: dict[str, str | None] = {}
        # Only tiers that can run a step are offered to the router; a judge-only tier
        # must never be picked, or the judge would author one side of the pairs it grades.
        for tier in self.config.executor_tiers:
            options[tier.name] = tier.description or f"model {tier.model} ({tier.name} tier)"
        if step.is_decision:
            options["jev"] = (
                "decide directly with System One: no model tokens, calibrated answer for a "
                "fixed choice, a yes/no, or an ordered scale"
            )
        return options

    # -------------------------------------------------------------------- stages

    def plan(self, task: str) -> Plan:
        plan = self.planner.plan(task, dry_run=self.dry_run)
        self.ledger.record(
            {
                "type": "plan",
                "tier": plan.tier,
                "model": plan.model,
                "cost_usd": plan.cost_usd,
                "fell_back": plan.fell_back,
                "error": plan.error,
                "steps": [step.to_dict() for step in plan.steps],
            }
        )
        self._emit(
            f"plan: {len(plan.steps)} step(s) via {plan.tier} ({plan.model or 'heuristic'})"
            + (" [fell back to a single step]" if plan.fell_back else "")
        )
        for step in plan.steps:
            tag = f"decide/{step.answer_type}" if step.is_decision else "generate"
            self._emit(f"  - {step.id} [{tag}] {step.goal}")
        return plan

    def route(self, task: str, step: Step, context: str) -> tuple[str, float, _JevUse]:
        """Jev Choice picks the executor for this step."""

        if self.config.agent.force_tier:
            return self.config.agent.force_tier, 1.0, _JevUse()
        options = self._tier_options(step)
        if len(options) == 1:
            return next(iter(options)), 1.0, _JevUse()
        result = self.jev.ask(
            self._state(task, step, context),
            {
                "executor": choice_question(
                    options, "Which single executor should run this step?"
                )
            },
        )
        picked, confidence = result.choice("executor")
        if picked not in options:
            picked = next(iter(options))
            confidence = 0.0
        return picked, confidence, _JevUse.of(result)

    def execute_decision(self, task: str, step: Step, context: str) -> tuple[str, _JevUse]:
        """System One answers the step itself, with no model tokens spent."""

        state = self._state(task, step, context)
        if step.answer_type == "noul":
            yes = step.criteria or step.goal
            questions = {
                "answer": noul_question(
                    yes,
                    f"the statement above is not supported by the state: {yes}",
                )
            }
        elif step.answer_type == "score":
            questions = {
                "answer": score_question(
                    step.options, f"Where does this belong on the scale: {step.goal}?"
                )
            }
        else:
            questions = {
                "answer": choice_question(
                    {option: None for option in step.options}, f"Which option: {step.goal}?"
                )
            }
        result = self.jev.ask(state, questions)

        if step.answer_type == "noul":
            probability = result.noul("answer")
            text = f"{'yes' if probability >= 0.5 else 'no'} (p={probability:.2f})"
        elif step.answer_type == "score":
            score, confidence = result.score("answer")
            legend = result.answers.get("answer", {}).get("legend") or {}
            level = legend.get(str(round(score))) or legend.get(str(int(score))) or "n/a"
            text = f"{level} (score={score:.2f}, confidence={confidence:.2f})"
        else:
            choice, confidence = result.choice("answer")
            text = f"{choice} (confidence={confidence:.2f})"
        return text, _JevUse.of(result)

    def execute_generate(self, task: str, step: Step, context: str, tier: str) -> tuple[str, _Draft]:
        """Run a generative step on a tier, possibly drafting candidates first."""

        provider = self.providers[tier]
        system = STEP_SYSTEM
        user = self._state(task, step, context)
        drafts = 1
        if tier == self.config.tiers[0].name and self.config.jev.candidates > 1:
            drafts = self.config.jev.candidates

        completions = [provider.complete(system, user) for _ in range(drafts)]
        tokens_in = sum(c.tokens_in for c in completions)
        tokens_out = sum(c.tokens_out for c in completions)
        cost = sum(c.cost_usd for c in completions)
        stub = all(c.stub for c in completions)
        latency = sum(c.latency_s for c in completions)

        if drafts == 1:
            return completions[0].text, _Draft(tokens_in, tokens_out, cost, latency, stub, 1, _JevUse())

        options = {
            f"candidate-{index + 1}": (completion.text[:400] or "<empty>")
            for index, completion in enumerate(completions)
        }
        picked, confidence = "", 0.0
        jev_use = _JevUse()
        try:
            result = self.jev.ask(
                self._state(task, step, context),
                {
                    "best": choice_question(
                        options,
                        "Which single draft best satisfies this step, in full? "
                        f"Step criteria: {step.criteria}",
                    )
                },
            )
            picked, confidence = result.choice("best")
            jev_use = _JevUse.of(result)
        except Exception as exc:  # selection is an optimisation, never a hard dependency
            self._emit(f"    candidate selection skipped: {exc}")
        index = _candidate_index(picked)
        if index is None or index >= len(completions):
            index = 0
        self._emit(
            f"    chose candidate-{index + 1} of {drafts} (confidence={confidence:.2f})"
        )
        return (
            completions[index].text,
            _Draft(tokens_in, tokens_out, cost, latency, stub, drafts, jev_use),
        )

    def verify(self, task: str, step: Step, output: str) -> tuple[float, _JevUse]:
        """Noul gate against the step's own criteria.

        The question is phrased to be adversarial rather than agreeable. Asking a
        calibrated model "is this good" invites a high probability for anything
        plausible; asking it to check each requirement, and to say no when any part is
        missing or merely gestured at, is what gives the threshold something to sit on.
        """

        result = self.jev.ask(
            self._state(task, step, "", candidate=output),
            {
                "meets": noul_question(
                    "Every requirement stated in the step criteria is met, in full: "
                    f"{step.criteria}. Concretely: the output covers all of it, says something "
                    "specific rather than gesturing at it, and could be handed to the next step "
                    "as-is with nothing important still missing or left as an exercise.",
                    "At least one requirement in the step criteria is not met, or is met only "
                    f"partially, vaguely, or as a placeholder: {step.criteria}. Answer this way if "
                    "any element is missing, wrong, generic where the step asked for something "
                    "specific, or would need more work before the next step could use it.",
                    "Does the output satisfy every requirement of this step, in full?",
                )
            },
        )
        return result.noul("meets"), _JevUse.of(result)

    # --------------------------------------------------------------------- loop

    def run(self, task: str) -> RunResult:
        result = RunResult(task=task)
        try:
            plan = self.plan(task)
        except Exception as exc:
            result.error = f"planning failed: {exc}"
            return result
        result.plan = plan

        outputs: dict[str, str] = {}
        started = time.monotonic()

        for step in plan.steps:
            if time.monotonic() - started > self.config.agent.task_timeout_s:
                result.aborted = f"task timeout after {self.config.agent.task_timeout_s}s"
                self._skip(step, result.aborted)
                continue
            if self.ledger.total_cost_usd >= self.config.agent.max_cost_usd:
                result.aborted = f"cost budget ${self.config.agent.max_cost_usd:.4f} reached"
                self._skip(step, result.aborted)
                continue
            context = self._context(step, outputs)
            self._run_step(task, step, context, outputs, result)

        result.outputs = outputs
        result.ledger = self.ledger
        return result

    def _skip(self, step: Step, reason: str) -> None:
        self.ledger.add_step(
            StepRecord(step_id=step.id, goal=step.goal, kind=step.kind, tier="-", status="skipped")
        )
        self._emit(f"  - {step.id} skipped ({reason})")

    def _run_step(
        self,
        task: str,
        step: Step,
        context: str,
        outputs: dict[str, str],
        result: RunResult,
    ) -> None:
        record = StepRecord(step_id=step.id, goal=step.goal, kind=step.kind, tier="")

        try:
            picked, confidence, route_use = self.route(task, step, context)
        except Exception as exc:
            picked, confidence, route_use = self.config.tiers[0].name, 0.0, _JevUse()
            self._emit(f"    routing fell back to {picked}: {exc}")
        record.route_confidence = confidence
        record.routed_by = "stub-jev" if self.jev.stub else "jev"
        _add_jev(record, route_use)

        if step.is_decision and picked == "jev":
            try:
                text, decision_use = self.execute_decision(task, step, context)
            except Exception as exc:
                self._emit(f"    Jev could not answer {step.id} ({exc}); falling back to a tier")
                picked = self.config.tiers[0].name
            else:
                record.tier = "jev"
                record.model = "systemone"
                _add_jev(record, decision_use)
                record.status = "done"
                record.output_chars = len(text)
                record.candidates = 0
                outputs[step.id] = text
                self.ledger.add_step(record)
                self._emit(f"  - {step.id} answered by Jev: {_one_line(text)}")
                return

        tier = self.config.tier(picked)
        try:
            final = self._generate_and_verify(task, step, context, tier, record)
        except ProviderError as exc:
            stronger = self.config.stronger_than(tier.name)
            if stronger is not None:
                self._emit(f"    {tier.name} failed ({exc}); escalating to {stronger.name}")
                record.escalations += 1
                record.escalated_from = tier.name
                try:
                    final = self._generate_and_verify(task, step, context, stronger, record)
                except Exception as exc2:
                    record.status = "failed"
                    record.tier = stronger.name
                    self.ledger.add_step(record)
                    result.error = f"step {step.id} failed on {stronger.name}: {exc2}"
                    self._emit(f"  - {step.id} FAILED: {exc2}")
                    return
            else:
                record.status = "failed"
                record.tier = tier.name
                self.ledger.add_step(record)
                result.error = f"step {step.id} failed on {tier.name}: {exc}"
                self._emit(f"  - {step.id} FAILED: {exc}")
                return

        text, final_tier, p_final = final
        record.tier = final_tier.name
        record.verify_p = p_final
        record.output_chars = len(text)
        outputs[step.id] = text
        self.ledger.add_step(record)
        verdict = "n/a" if p_final is None else f"p={p_final:.2f}"
        self._emit(f"  - {step.id} done on {final_tier.name} ({verdict})")

    def _generate_and_verify(
        self,
        task: str,
        step: Step,
        context: str,
        tier: TierConfig,
        record: StepRecord,
    ) -> tuple[str, TierConfig, float | None]:
        """Generate, verify with Jev, then retry, escalate, or keep the best attempt.

        Policy ladder, cheapest move first:
        1. accept when the gate passes
        2. retry the same tier (``same_tier_retries``) before paying for a bigger model,
           and retry the same tier on a transient provider error (``error_retries``)
        3. escalate one tier up (``max_escalations``), keeping whichever result the
           gate liked better
        4. stop escalating once the run is close to its budget (``budget_guard_ratio``)
        """

        threshold = self.config.jev.escalate_below
        best: tuple[str, TierConfig, float | None] = ("", tier, None)

        while True:
            text, draft = self._generate_with_error_retries(task, step, context, tier, record)
            self._absorb(record, draft, tier)

            if not self.config.jev.verify:
                best = (text, tier, None)
                break

            p, gate_use = self.verify(task, step, text)
            _add_jev(record, gate_use)
            if record.verify_p_initial is None:
                record.verify_p_initial = p
            if best[2] is None or p > (best[2] or 0.0):
                best = (text, tier, p)

            if p >= threshold:
                break

            retries_left = self.config.jev.same_tier_retries - record.retries
            if retries_left > 0 and not self._budget_guard_active(record):
                record.retries += 1
                self._emit(
                    f"    {tier.name} verified at p={p:.2f} < {threshold:.2f}; "
                    f"retrying the same tier (retry {record.retries}/{self.config.jev.same_tier_retries})"
                )
                continue
            break

        if not self.config.jev.verify:
            return best

        if best[2] is not None and best[2] >= threshold:
            return best

        text, final_tier, p = best
        if record.escalations >= self.config.jev.max_escalations:
            return best
        if self._budget_guard_active(record):
            self._emit("    escalation skipped: the run is inside its budget guard band")
            return best
        stronger = self.config.stronger_than(final_tier.name)
        if stronger is None:
            return best

        self._emit(
            f"    {final_tier.name} verified at p={(p or 0.0):.2f} < {threshold:.2f}; "
            f"escalating to {stronger.name}"
        )
        record.escalations += 1
        record.escalated_from = final_tier.name
        better_text, better_draft = self.execute_generate(task, step, context, stronger.name)
        self._absorb(record, better_draft, stronger)

        p2, gate_use2 = self.verify(task, step, better_text)
        _add_jev(record, gate_use2)
        if p2 >= (p or 0.0):
            return better_text, stronger, p2
        return best

    def _generate_with_error_retries(
        self, task: str, step: Step, context: str, tier: TierConfig, record: StepRecord
    ) -> tuple[str, _Draft]:
        """Run one generation, retrying the same tier on a transient provider error.

        A reasoning model that burns its whole output cap on thinking, a rate limit, or
        a dropped connection are all transient. Retrying the same tier is far cheaper
        than escalating, and escalation is what happens if this keeps failing.
        """

        attempts = self.config.agent.error_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return self.execute_generate(task, step, context, tier.name)
            except ProviderError as exc:
                if attempt >= attempts:
                    raise
                record.error_retries += 1
                self._emit(f"    {tier.name} failed ({exc}); retrying the same tier (attempt {attempt + 1}/{attempts})")
        raise AssertionError("unreachable")

    def _absorb(self, record: StepRecord, draft: _Draft, tier: TierConfig) -> None:
        """Add one generation round's spend to the record."""

        record.model = tier.model
        record.tokens_in += draft.tokens_in
        record.tokens_out += draft.tokens_out
        record.cost_usd += draft.cost_usd
        record.latency_s += draft.latency_s
        record.stub = record.stub or draft.stub
        record.candidates = max(record.candidates, draft.drafts)
        _add_jev(record, draft.jev)

    def _budget_guard_active(self, record: StepRecord | None = None) -> bool:
        """True once a run has spent enough of its budget that escalation is unwise.

        The in-flight step's own spend counts: its record is not in the ledger yet,
        and ignoring it would let the first step escalate past the guard band.
        """

        ratio = self.config.agent.budget_guard_ratio
        if ratio >= 1.0:
            return False
        spent = self.ledger.total_cost_usd
        if record is not None:
            spent += record.cost_usd + record.jev_cost_usd
        return spent >= ratio * self.config.agent.max_cost_usd


@dataclass
class _Draft:
    """Bookkeeping for one generation round, including its candidate selection."""

    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    stub: bool = False
    drafts: int = 1
    jev: _JevUse = field(default_factory=_JevUse)


def _add_jev(record: StepRecord, use: _JevUse) -> None:
    record.jev_cost_usd += use.cost_usd
    record.jev_calls += use.calls
    record.jev_tokens_in += use.tokens_in
    record.jev_tokens_out += use.tokens_out


def _candidate_index(picked: str) -> int | None:
    if not picked.startswith("candidate-"):
        return None
    try:
        return int(picked.split("-", 1)[1]) - 1
    except (IndexError, ValueError):
        return None


def _one_line(text: str, limit: int = 120) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else f"{flat[:limit]}..."


def run_task(
    config: Config,
    task: str,
    dry_run: bool = False,
    providers: dict[str, Provider] | None = None,
    jev: Jev | None = None,
    on_event: Callable[[str], None] | None = None,
) -> RunResult:
    agent = CascadeAgent(config, providers=providers, jev=jev, dry_run=dry_run, on_event=on_event)
    return agent.run(task)


def result_to_json(result: RunResult, config: Config, stub: bool = False) -> str:
    return json.dumps(
        {
            "task": result.task,
            "aborted": result.aborted,
            "error": result.error,
            "plan": json.loads(plan_to_json(result.plan)) if result.plan else None,
            "outputs": result.outputs,
            "summary": result.ledger.summary(config, stub=stub) if result.ledger else None,
        },
        indent=2,
    )
