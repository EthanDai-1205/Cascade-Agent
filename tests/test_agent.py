"""The loop: routing, Jev-executed steps, verification, escalation, budget, ledger."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_cascade.testing import (  # noqa: E402
    FailingProvider,
    ScriptedJev,
    StaticPlanner,
    base_config,
    judge_config,
)
from jev_cascade.agent import CascadeAgent  # noqa: E402
from jev_cascade.config import config_from_dict  # noqa: E402
from jev_cascade.planner import Step  # noqa: E402
from jev_cascade.providers import ProviderError, ScriptedProvider  # noqa: E402

LABEL_STEP = Step(
    id="label",
    goal="label the issue",
    kind="decide",
    answer_type="choice",
    options=["bug", "feature"],
    criteria="the label that best matches the reported behavior",
)
WRITE_STEP = Step(
    id="write",
    goal="write the reply",
    kind="generate",
    criteria="the reply is specific and polite",
)


def make_agent(
    config,
    steps: list[Step] | None = None,
    jev=None,
    cheap_outputs: list[str] | None = None,
    expensive_outputs: list[str] | None = None,
    cheap_provider=None,
    expensive_provider=None,
    events: list[str] | None = None,
):
    providers = {
        "cheap": cheap_provider
        or ScriptedProvider("cheap", cheap_outputs or ["CHEAP-OUT"], price=config.tier("cheap").price),
        "expensive": expensive_provider
        or ScriptedProvider(
            "expensive", expensive_outputs or ["EXPENSIVE-OUT"], price=config.tier("expensive").price
        ),
    }
    agent = CascadeAgent(
        config,
        providers=providers,
        jev=jev or ScriptedJev(),
        dry_run=True,
        on_event=(events.append if events is not None else (lambda _message: None)),
    )
    if steps is not None:
        agent.planner = StaticPlanner(steps)
    return agent, providers


class TestJevExecutesDecisionSteps(unittest.TestCase):
    def test_decision_step_is_answered_by_jev_and_costs_no_model_tokens(self) -> None:
        jev = ScriptedJev()
        agent, providers = make_agent(base_config(), steps=[LABEL_STEP], jev=jev)

        result = agent.run("triage this report")

        self.assertEqual(result.outputs["label"], "bug (confidence=0.90)")
        self.assertEqual(providers["cheap"].calls, 0)
        self.assertEqual(providers["expensive"].calls, 0)
        record = result.ledger.records[0]
        self.assertEqual(record.tier, "jev")
        self.assertEqual(record.model, "systemone")
        self.assertEqual(record.cost_usd, 0.0)
        self.assertGreater(record.jev_calls, 0)
        self.assertEqual(result.ledger.model_cost_usd, 0.0)

    def test_the_state_jev_receives_carries_the_goal_and_the_criteria(self) -> None:
        jev = ScriptedJev()
        agent, _ = make_agent(base_config(), steps=[LABEL_STEP], jev=jev)
        agent.run("triage this report")
        decision_state = [s for s in jev.states if "step_kind: decide" in s][0]
        self.assertIn("step_goal: label the issue", decision_state)
        self.assertIn("step_criteria: the label that best matches", decision_state)
        self.assertIn("task: triage this report", decision_state)

    def test_noul_and_score_steps_are_also_answered_by_jev(self) -> None:
        jev = ScriptedJev(verdicts=[0.87])
        steps = [
            Step(id="urgent", goal="is this urgent", kind="decide", answer_type="noul", criteria="the report describes data loss"),
            Step(
                id="severity",
                goal="rate the severity",
                kind="decide",
                answer_type="score",
                options=["cosmetic", "annoying", "blocking"],
                criteria="how much it blocks the user",
            ),
        ]
        agent, providers = make_agent(base_config(), steps=steps, jev=jev)

        result = agent.run("triage this report")

        self.assertEqual(result.outputs["urgent"], "yes (p=0.87)")
        self.assertEqual(result.outputs["severity"], "blocking (score=2.00, confidence=0.90)")
        self.assertEqual(providers["cheap"].calls, 0)

    def test_jev_failure_on_a_decision_step_falls_back_to_a_tier(self) -> None:
        class BrokenJev(ScriptedJev):
            def ask(self, state, questions):
                raise RuntimeError("system one is down")

        jev = BrokenJev()
        agent, providers = make_agent(
            base_config(verify=False), steps=[LABEL_STEP], jev=jev, cheap_outputs=["FALLBACK"]
        )
        result = agent.run("triage this report")

        self.assertEqual(result.outputs["label"], "FALLBACK")
        self.assertEqual(providers["cheap"].calls, 1)
        self.assertEqual(result.ledger.records[0].tier, "cheap")


class TestJudgeOnlyTiersNeverRunSteps(unittest.TestCase):
    """A judge-only tier grades; it must never author the work it grades."""

    def test_a_judge_only_tier_is_never_offered_to_the_router(self) -> None:
        config = judge_config()
        agent, _ = make_agent(config, steps=[WRITE_STEP])
        options = agent._tier_options(WRITE_STEP)
        self.assertEqual(set(options), {"cheap", "expensive"})
        self.assertNotIn("judge", options)

    def test_a_decision_step_still_offers_jev_but_not_the_judge(self) -> None:
        config = judge_config()
        agent, _ = make_agent(config, steps=[LABEL_STEP])
        options = agent._tier_options(LABEL_STEP)
        self.assertIn("jev", options)
        self.assertNotIn("judge", options)

    def test_escalation_never_lands_on_a_judge_only_tier(self) -> None:
        config = judge_config()
        judge_provider = ScriptedProvider(
            "judge", ["JUDGE-SHOULD-NEVER-RUN"], price=config.tier("judge").price
        )
        providers = {
            "cheap": FailingProvider("cheap", ["never"], failures=99, price=config.tier("cheap").price),
            "expensive": ScriptedProvider(
                "expensive", ["STRONG"], price=config.tier("expensive").price
            ),
            "judge": judge_provider,
        }
        agent = CascadeAgent(
            config, providers=providers, jev=ScriptedJev(verdicts=[0.9]), dry_run=True
        )
        agent.planner = StaticPlanner([WRITE_STEP])

        result = agent.run("write the reply")

        record = result.ledger.records[0]
        self.assertEqual(record.tier, "expensive")
        self.assertEqual(record.escalated_from, "cheap")
        self.assertEqual(judge_provider.calls, 0)
        self.assertEqual(result.outputs["write"], "STRONG")

    def test_an_unroutable_reply_still_falls_back_to_the_cheap_tier(self) -> None:
        config = judge_config()
        # A router that answers with the judge tier's name, which is not an option.
        jev = ScriptedJev(router=lambda state, options: "judge")
        agent, providers = make_agent(config, steps=[WRITE_STEP], jev=jev, cheap_outputs=["CHEAP-OUT"])

        result = agent.run("write the reply")

        self.assertEqual(result.ledger.records[0].tier, "cheap")
        self.assertEqual(result.outputs["write"], "CHEAP-OUT")


class TestVerificationGate(unittest.TestCase):
    def test_confident_cheap_output_is_not_escalated(self) -> None:
        jev = ScriptedJev(verdicts=[0.95])
        agent, providers = make_agent(
            base_config(), steps=[WRITE_STEP], jev=jev, cheap_outputs=["GOOD-ENOUGH"]
        )
        result = agent.run("write the reply")
        record = result.ledger.records[0]

        self.assertEqual(result.outputs["write"], "GOOD-ENOUGH")
        self.assertEqual(record.tier, "cheap")
        self.assertEqual(record.escalations, 0)
        self.assertAlmostEqual(record.verify_p, 0.95)
        self.assertEqual(providers["expensive"].calls, 0)

    def test_failed_gate_escalates_once_and_keeps_the_stronger_output(self) -> None:
        jev = ScriptedJev(verdicts=[0.2, 0.9])
        agent, providers = make_agent(
            base_config(),
            steps=[WRITE_STEP],
            jev=jev,
            cheap_outputs=["WEAK"],
            expensive_outputs=["STRONG"],
        )
        result = agent.run("write the reply")
        record = result.ledger.records[0]

        self.assertEqual(result.outputs["write"], "STRONG")
        self.assertEqual(record.tier, "expensive")
        self.assertEqual(record.escalations, 1)
        self.assertAlmostEqual(record.verify_p, 0.9)
        self.assertAlmostEqual(record.verify_p_initial, 0.2)
        self.assertEqual(providers["cheap"].calls, 1)
        self.assertEqual(providers["expensive"].calls, 1)
        self.assertGreater(record.tokens_in, 0)

    def test_escalation_keeps_the_cheap_output_when_the_strong_one_is_worse(self) -> None:
        jev = ScriptedJev(verdicts=[0.3, 0.1])
        agent, _ = make_agent(
            base_config(),
            steps=[WRITE_STEP],
            jev=jev,
            cheap_outputs=["FIRST-ATTEMPT"],
            expensive_outputs=["WORSE"],
        )
        result = agent.run("write the reply")
        record = result.ledger.records[0]

        self.assertEqual(result.outputs["write"], "FIRST-ATTEMPT")
        self.assertEqual(record.tier, "cheap")
        self.assertAlmostEqual(record.verify_p, 0.3)
        self.assertEqual(record.escalations, 1)

    def test_no_escalation_when_the_policy_forbids_it(self) -> None:
        jev = ScriptedJev(verdicts=[0.1])
        agent, providers = make_agent(
            base_config(max_escalations=0), steps=[WRITE_STEP], jev=jev, cheap_outputs=["WEAK"]
        )
        result = agent.run("write the reply")
        record = result.ledger.records[0]

        self.assertEqual(record.tier, "cheap")
        self.assertEqual(record.escalations, 0)
        self.assertEqual(providers["expensive"].calls, 0)

    def test_top_tier_result_is_kept_even_when_it_fails_the_gate(self) -> None:
        jev = ScriptedJev(
            router=lambda state, options: "expensive" if "step_kind: generate" in state else "jev",
            verdicts=[0.05],
        )
        agent, _ = make_agent(base_config(), steps=[WRITE_STEP], jev=jev, expensive_outputs=["TOP"])
        result = agent.run("write the reply")
        record = result.ledger.records[0]

        self.assertEqual(result.outputs["write"], "TOP")
        self.assertEqual(record.tier, "expensive")
        self.assertEqual(record.escalations, 0)
        self.assertAlmostEqual(record.verify_p, 0.05)

    def test_verification_can_be_switched_off(self) -> None:
        jev = ScriptedJev(verdicts=[0.01])
        agent, _ = make_agent(
            base_config(verify=False), steps=[WRITE_STEP], jev=jev, cheap_outputs=["UNAUDITED"]
        )
        result = agent.run("write the reply")
        record = result.ledger.records[0]

        self.assertIsNone(record.verify_p)
        self.assertEqual(record.escalations, 0)
        self.assertFalse(
            any(question.get("type") == "noul" for call in jev.questions for question in call.values())
        )

    def test_the_gate_sees_the_candidate_output(self) -> None:
        jev = ScriptedJev(verdicts=[0.9])
        agent, _ = make_agent(base_config(), steps=[WRITE_STEP], jev=jev, cheap_outputs=["DRAFT-TEXT"])
        agent.run("write the reply")
        gate_state = [s for s in jev.states if "candidate output:" in s][0]
        self.assertIn("DRAFT-TEXT", gate_state)
        self.assertIn("step_criteria: the reply is specific and polite", gate_state)


class TestRouting(unittest.TestCase):
    def test_jev_can_send_a_generative_step_to_the_strong_tier(self) -> None:
        jev = ScriptedJev(router=lambda state, options: "expensive")
        agent, providers = make_agent(
            base_config(), steps=[WRITE_STEP], jev=jev, expensive_outputs=["HARD"]
        )
        result = agent.run("refactor the parser")

        self.assertEqual(result.outputs["write"], "HARD")
        self.assertEqual(providers["expensive"].calls, 1)
        self.assertEqual(providers["cheap"].calls, 0)
        self.assertEqual(result.ledger.records[0].routed_by, "stub-jev")

    def test_jev_is_offered_as_an_executor_only_for_decision_steps(self) -> None:
        jev = ScriptedJev()
        agent, _ = make_agent(base_config(), steps=[LABEL_STEP, WRITE_STEP], jev=jev)
        agent.run("triage and reply")

        offered = [
            list(list(call.values())[0]["criteria"].keys())
            for call in jev.questions
            if "executor" in call
        ]
        self.assertIn("jev", offered[0])
        self.assertNotIn("jev", offered[1])

    def test_an_invalid_route_falls_back_to_the_cheap_tier(self) -> None:
        class GhostPickingJev(ScriptedJev):
            def ask(self, state, questions):
                result = super().ask(state, questions)
                for answer in result.answers.values():
                    if answer.get("type") == "choice":
                        answer["choice"] = "tier-that-does-not-exist"
                return result

        jev = GhostPickingJev()
        agent, providers = make_agent(base_config(), steps=[WRITE_STEP], jev=jev)
        result = agent.run("write the reply")

        self.assertEqual(result.ledger.records[0].tier, "cheap")
        self.assertEqual(providers["cheap"].calls, 1)
        self.assertEqual(result.ledger.records[0].route_confidence, 0.0)

    def test_routing_fallback_when_jev_raises(self) -> None:
        class BrokenRouter(ScriptedJev):
            def ask(self, state, questions):
                if "executor" in questions:
                    raise RuntimeError("no routing today")
                return super().ask(state, questions)

        agent, providers = make_agent(base_config(), steps=[WRITE_STEP], jev=BrokenRouter())
        result = agent.run("write the reply")

        self.assertEqual(result.ledger.records[0].tier, "cheap")
        self.assertEqual(providers["cheap"].calls, 1)

    def test_forced_tier_bypasses_routing_entirely(self) -> None:
        jev = ScriptedJev()
        agent, providers = make_agent(
            base_config(force_tier="expensive"), steps=[WRITE_STEP], jev=jev, expensive_outputs=["PINNED"]
        )
        result = agent.run("write the reply")

        self.assertEqual(result.outputs["write"], "PINNED")
        self.assertEqual(providers["cheap"].calls, 0)
        self.assertFalse(any("executor" in call for call in jev.questions))

    def test_single_candidate_needs_no_selection_call(self) -> None:
        jev = ScriptedJev()
        agent, _ = make_agent(base_config(candidates=1), steps=[WRITE_STEP], jev=jev)
        agent.run("write the reply")
        self.assertFalse(any("best" in call for call in jev.questions))


class TestCandidateDrafting(unittest.TestCase):
    def test_drafts_are_generated_then_jev_picks_one(self) -> None:
        jev = ScriptedJev(chooser=lambda state, options: "candidate-3")
        agent, providers = make_agent(
            base_config(candidates=3),
            steps=[WRITE_STEP],
            jev=jev,
            cheap_outputs=["DRAFT-A", "DRAFT-B", "DRAFT-C"],
        )
        result = agent.run("write the reply")
        record = result.ledger.records[0]

        self.assertEqual(providers["cheap"].calls, 3)
        self.assertEqual(result.outputs["write"], "DRAFT-C")
        self.assertEqual(record.candidates, 3)
        self.assertGreater(record.tokens_out, len("DRAFT-C") // 4)

        selection = [call["best"] for call in jev.questions if "best" in call]
        self.assertEqual(len(selection), 1, "one selection question")
        options = selection[0]["criteria"]
        self.assertEqual(list(options), ["candidate-1", "candidate-2", "candidate-3"])
        self.assertIn("DRAFT-B", options["candidate-2"])

    def test_a_dead_selection_call_falls_back_to_the_first_draft(self) -> None:
        class PickerlessJev(ScriptedJev):
            def ask(self, state, questions):
                if "best" in questions:
                    raise RuntimeError("selection unavailable")
                return super().ask(state, questions)

        agent, _ = make_agent(
            base_config(candidates=2),
            steps=[WRITE_STEP],
            jev=PickerlessJev(),
            cheap_outputs=["FIRST", "SECOND"],
        )
        result = agent.run("write the reply")

        self.assertEqual(result.outputs["write"], "FIRST")

    def test_drafting_is_not_used_for_the_strong_tier(self) -> None:
        jev = ScriptedJev(router=lambda state, options: "expensive")
        agent, providers = make_agent(
            base_config(candidates=3),
            steps=[WRITE_STEP],
            jev=jev,
            expensive_outputs=["ONLY-ONE"],
        )
        result = agent.run("write the reply")

        self.assertEqual(providers["expensive"].calls, 1)
        self.assertEqual(result.ledger.records[0].candidates, 1)


class TestPolicyLadder(unittest.TestCase):
    """Cheapest move first: accept, retry the same tier, escalate, or stop on the budget guard."""

    def test_a_same_tier_retry_recovers_without_escalating(self) -> None:
        jev = ScriptedJev(verdicts=[0.2, 0.9])
        agent, providers = make_agent(
            base_config(same_tier_retries=1),
            steps=[WRITE_STEP],
            jev=jev,
            cheap_outputs=["WEAK", "BETTER"],
            expensive_outputs=["NEVER-USED"],
        )
        result = agent.run("write the reply")
        record = result.ledger.records[0]

        self.assertEqual(result.outputs["write"], "BETTER")
        self.assertEqual(record.tier, "cheap")
        self.assertEqual(record.retries, 1)
        self.assertEqual(record.escalations, 0)
        self.assertEqual(providers["cheap"].calls, 2)
        self.assertEqual(providers["expensive"].calls, 0)
        self.assertAlmostEqual(record.verify_p, 0.9)
        self.assertAlmostEqual(record.verify_p_initial, 0.2)

    def test_escalation_follows_an_exhausted_retry(self) -> None:
        jev = ScriptedJev(verdicts=[0.2, 0.3, 0.9])
        agent, providers = make_agent(
            base_config(same_tier_retries=1),
            steps=[WRITE_STEP],
            jev=jev,
            cheap_outputs=["W1", "W2"],
            expensive_outputs=["STRONG"],
        )
        result = agent.run("write the reply")
        record = result.ledger.records[0]

        self.assertEqual(result.outputs["write"], "STRONG")
        self.assertEqual(record.tier, "expensive")
        self.assertEqual(record.retries, 1)
        self.assertEqual(record.escalations, 1)
        self.assertEqual(providers["cheap"].calls, 2)
        self.assertEqual(providers["expensive"].calls, 1)

    def test_retries_are_capped(self) -> None:
        jev = ScriptedJev(verdicts=[0.1, 0.1, 0.1, 0.9])
        agent, providers = make_agent(
            base_config(same_tier_retries=3),
            steps=[WRITE_STEP],
            jev=jev,
            cheap_outputs=["A", "B", "C", "D"],
        )
        result = agent.run("write the reply")
        record = result.ledger.records[0]

        self.assertEqual(record.retries, 3)
        self.assertEqual(providers["cheap"].calls, 4)
        self.assertAlmostEqual(record.verify_p, 0.9)

    def test_the_best_attempt_is_kept_when_nothing_passes(self) -> None:
        jev = ScriptedJev(verdicts=[0.2, 0.5])
        agent, _ = make_agent(
            base_config(same_tier_retries=1, max_escalations=0),
            steps=[WRITE_STEP],
            jev=jev,
            cheap_outputs=["FIRST", "SECOND"],
        )
        result = agent.run("write the reply")
        record = result.ledger.records[0]

        self.assertEqual(result.outputs["write"], "SECOND")
        self.assertAlmostEqual(record.verify_p, 0.5)

    def test_budget_guard_stops_retries_and_escalation(self) -> None:
        events: list[str] = []
        agent, providers = make_agent(
            # a sub-cent budget with a 50% guard band: the first generation alone
            # crosses it, so retries and escalation must both be refused
            base_config(same_tier_retries=2, budget_guard_ratio=0.5, max_cost_usd=0.0001),
            steps=[WRITE_STEP],
            jev=ScriptedJev(verdicts=[0.1, 0.9]),
            cheap_outputs=["ONLY-ONE", "SHOULD-NOT-RUN"],
            expensive_outputs=["SHOULD-NOT-RUN"],
            events=events,
        )
        result = agent.run("write the reply")
        record = result.ledger.records[0]

        self.assertEqual(result.outputs["write"], "ONLY-ONE")
        self.assertEqual(record.tier, "cheap")
        self.assertEqual(record.retries, 0)
        self.assertEqual(record.escalations, 0)
        self.assertEqual(providers["cheap"].calls, 1)
        self.assertEqual(providers["expensive"].calls, 0)
        self.assertGreater(record.cost_usd + record.jev_cost_usd, 0.5 * 0.0001)
        self.assertIn("budget guard", "\n".join(events))

    def test_a_transient_error_retries_the_same_tier_before_escalating(self) -> None:
        cheap = FailingProvider("cheap", ["RECOVERED"], failures=1, price=None)
        agent, providers = make_agent(
            base_config(verify=False),
            steps=[WRITE_STEP],
            jev=ScriptedJev(),
            cheap_provider=cheap,
            expensive_outputs=["NEVER-USED"],
        )
        result = agent.run("write the reply")
        record = result.ledger.records[0]

        self.assertEqual(result.outputs["write"], "RECOVERED")
        self.assertEqual(record.tier, "cheap")
        self.assertEqual(record.error_retries, 1)
        self.assertEqual(record.escalations, 0)
        self.assertEqual(providers["expensive"].calls, 0)

    def test_errors_are_capped_then_escalated(self) -> None:
        cheap = FailingProvider("cheap", ["NEVER-REACHED"], failures=99)
        agent, providers = make_agent(
            base_config(verify=False),
            steps=[WRITE_STEP],
            jev=ScriptedJev(),
            cheap_provider=cheap,
            expensive_outputs=["RESCUED"],
        )
        result = agent.run("write the reply")
        record = result.ledger.records[0]

        self.assertEqual(result.outputs["write"], "RESCUED")
        self.assertEqual(record.tier, "expensive")
        self.assertEqual(record.error_retries, 1)
        self.assertEqual(cheap.calls, 2, "one try plus one retry")
        self.assertEqual(providers["expensive"].calls, 1)

    def test_error_retries_can_be_switched_off(self) -> None:
        cheap = FailingProvider("cheap", ["NEVER-REACHED"], failures=99)
        agent, providers = make_agent(
            base_config(verify=False, error_retries=0),
            steps=[WRITE_STEP],
            jev=ScriptedJev(),
            cheap_provider=cheap,
            expensive_outputs=["RESCUED"],
        )
        result = agent.run("write the reply")

        self.assertEqual(cheap.calls, 1)
        self.assertEqual(result.ledger.records[0].error_retries, 0)

    def test_budget_guard_is_off_when_the_ratio_is_one(self) -> None:
        agent, providers = make_agent(
            base_config(budget_guard_ratio=1.0),
            steps=[WRITE_STEP],
            jev=ScriptedJev(verdicts=[0.1, 0.9]),
            cheap_outputs=["WEAK"],
            expensive_outputs=["STRONG"],
        )
        result = agent.run("write the reply")
        self.assertEqual(result.ledger.records[0].escalations, 1)
        self.assertEqual(providers["expensive"].calls, 1)


class TestFailureHandling(unittest.TestCase):
    def test_a_failing_tier_escalates_instead_of_dying(self) -> None:
        jev = ScriptedJev()
        cheap = ScriptedProvider("cheap", ["never used"])
        cheap.complete = _raiser(ProviderError("cheap is down", status=500))  # type: ignore[method-assign]
        agent, providers = make_agent(
            base_config(),
            steps=[WRITE_STEP],
            jev=jev,
            cheap_provider=cheap,
            expensive_outputs=["RESCUED"],
        )
        result = agent.run("write the reply")
        record = result.ledger.records[0]

        self.assertEqual(result.outputs["write"], "RESCUED")
        self.assertEqual(record.tier, "expensive")
        self.assertEqual(record.escalations, 1)
        self.assertEqual(result.error, "")
        self.assertEqual(providers["expensive"].calls, 1)

    def test_failure_at_the_top_tier_marks_the_step_failed(self) -> None:
        config = config_from_dict(
            {
                "agent": {"planner_mode": "heuristic", "planner_tier": "only", "verbose": False},
                "tiers": [{"name": "only", "kind": "mock", "model": "m"}],
            }
        )
        provider = ScriptedProvider("only", ["never used"])
        provider.complete = _raiser(ProviderError("only tier is down", status=500))  # type: ignore[method-assign]
        agent = CascadeAgent(
            config,
            providers={"only": provider},
            jev=ScriptedJev(router=lambda state, options: "only"),
            dry_run=True,
            on_event=lambda _message: None,
        )
        agent.planner = StaticPlanner([WRITE_STEP])
        result = agent.run("write the reply")

        self.assertEqual(result.outputs, {})
        self.assertEqual(result.ledger.records[0].status, "failed")
        self.assertIn("only tier is down", result.error)


class TestBudgetAndSkips(unittest.TestCase):
    def test_the_budget_stops_the_run_and_marks_the_rest_skipped(self) -> None:
        config = base_config(max_cost_usd=0.00001)
        jev = ScriptedJev()
        agent, providers = make_agent(
            config,
            steps=[WRITE_STEP, Step(id="second", goal="second thing", kind="generate")],
            jev=jev,
            cheap_outputs=["FIRST", "SECOND"],
        )
        result = agent.run("write two things")

        self.assertIn("budget", result.aborted)
        statuses = [record.status for record in result.ledger.records]
        self.assertEqual(statuses, ["done", "skipped"])
        self.assertEqual(providers["cheap"].calls, 1)

    def test_budget_of_zero_skips_everything(self) -> None:
        agent, providers = make_agent(
            base_config(max_cost_usd=0.0), steps=[WRITE_STEP], jev=ScriptedJev()
        )
        result = agent.run("write the reply")

        self.assertEqual(result.outputs, {})
        self.assertEqual(result.ledger.records[0].status, "skipped")
        self.assertEqual(providers["cheap"].calls, 0)


class TestContextPropagation(unittest.TestCase):
    def test_later_steps_see_earlier_step_outputs(self) -> None:
        jev = ScriptedJev()
        agent, providers = make_agent(
            base_config(), jev=jev, cheap_outputs=["FIRST-OUTPUT", "SECOND-OUTPUT"]
        )
        result = agent.run("write A and then write B")

        prompts = providers["cheap"].prompts
        self.assertEqual(len(prompts), 2)
        self.assertIn("FIRST-OUTPUT", prompts[1])
        self.assertNotIn("FIRST-OUTPUT", prompts[0])
        self.assertIn("SECOND-OUTPUT", result.outputs["s2"])

    def test_the_classification_from_a_decision_step_reaches_the_writer(self) -> None:
        jev = ScriptedJev()
        agent, providers = make_agent(base_config(), jev=jev, cheap_outputs=["WRITTEN"])
        result = agent.run("write the reply")

        self.assertIn("code-change (confidence=0.90)", providers["cheap"].prompts[0])
        self.assertEqual(result.outputs["classify"], "code-change (confidence=0.90)")


class TestLedger(unittest.TestCase):
    def test_costs_are_attributed_per_tier_and_repriced(self) -> None:
        config = base_config()
        jev = ScriptedJev(verdicts=[0.2, 0.9])
        agent, _ = make_agent(
            config,
            steps=[WRITE_STEP],
            jev=jev,
            cheap_provider=ScriptedProvider("cheap", ["WEAK"], price=config.tier("cheap").price, stub=False),
            expensive_provider=ScriptedProvider(
                "expensive", ["STRONG"], price=config.tier("expensive").price, stub=False
            ),
        )
        result = agent.run("write the reply")
        summary = result.ledger.summary(config, stub=False)

        self.assertGreater(summary["model_cost_usd"], 0.0)
        self.assertGreater(summary["jev_cost_usd"], 0.0)
        self.assertAlmostEqual(
            summary["total_cost_usd"],
            round(summary["model_cost_usd"] + summary["jev_cost_usd"], 6),
            places=6,
        )
        self.assertEqual(set(summary["by_tier"]), {"expensive"})
        self.assertEqual(result.ledger.records[0].escalated_from, "cheap")
        self.assertIn("cheap", summary["repriced_usd"])
        self.assertGreater(
            summary["repriced_usd"]["expensive"],
            summary["repriced_usd"]["cheap"],
            "the strong tier's list price is 10x higher",
        )
        self.assertEqual(summary["escalations"], 1)
        self.assertFalse(summary["stub"])

    def test_repricing_never_quotes_a_judge_only_tier(self) -> None:
        config = judge_config()
        provider = ScriptedProvider("cheap", ["OUT"], price=config.tier("cheap").price, stub=False)
        agent, _ = make_agent(config, steps=[WRITE_STEP], cheap_provider=provider)
        agent.run("write the reply")

        repriced = agent.ledger.repriced(config)

        # a judge can grade this run's tokens, but it cannot have written them
        self.assertEqual(set(repriced), {"cheap", "expensive"})

    def test_the_jev_row_reports_system_one_tokens(self) -> None:
        config = base_config()
        agent, _ = make_agent(config, steps=[LABEL_STEP], jev=ScriptedJev())
        result = agent.run("triage")
        bucket = result.ledger.summary(config, stub=True)["by_tier"]["jev"]

        self.assertEqual(int(bucket["steps"]), 1)
        self.assertGreater(int(bucket["tokens_in"]), 0, "System One usage is billed and logged")
        jev_calls = result.ledger.records[0].jev_calls
        self.assertEqual(int(bucket["tokens_out"]), 8 * jev_calls)

    def test_jev_only_steps_still_appear_in_the_ledger(self) -> None:
        config = base_config()
        agent, _ = make_agent(config, steps=[LABEL_STEP], jev=ScriptedJev())
        result = agent.run("triage")
        summary = result.ledger.summary(config, stub=True)

        self.assertEqual(summary["steps"], 1)
        self.assertIn("jev", summary["by_tier"])
        self.assertGreater(summary["jev_cost_usd"], 0.0)
        self.assertEqual(summary["model_cost_usd"], 0.0)

    def test_events_are_written_as_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = base_config(ledger_path=str(Path(tmp) / "run.jsonl"))
            agent, _ = make_agent(config, steps=[WRITE_STEP], jev=ScriptedJev())
            agent.run("write the reply")
            agent.ledger.close()

            lines = (Path(tmp) / "run.jsonl").read_text().strip().splitlines()
            events = [json.loads(line) for line in lines]
            kinds = [event["type"] for event in events]
            self.assertEqual(kinds[0], "plan")
            self.assertIn("step", kinds)
            self.assertTrue(all("ts" in event for event in events))

    def test_the_rendered_summary_is_honest_about_stubs(self) -> None:
        config = base_config()
        agent, _ = make_agent(config, steps=[WRITE_STEP], jev=ScriptedJev())
        result = agent.run("write the reply")
        text = result.ledger.render_summary(config, stub=True)

        self.assertIn("stub Jev", text)
        self.assertIn("steps 1", text)
        self.assertIn("[write]", text)


class TestEvents(unittest.TestCase):
    def test_a_trace_is_emitted_per_step(self) -> None:
        events: list[str] = []
        agent, _ = make_agent(
            base_config(), steps=[LABEL_STEP, WRITE_STEP], jev=ScriptedJev(), events=events
        )
        agent.run("triage and reply")

        joined = "\n".join(events)
        self.assertIn("plan: 2 step(s)", joined)
        self.assertIn("answered by Jev", joined)
        self.assertIn("done on cheap", joined)


def _raiser(error: Exception):
    def boom(*_args, **_kwargs):
        raise error

    return boom


if __name__ == "__main__":
    unittest.main()
