"""The split eval: pair building, blind judging, threshold sweep, recommendation."""

from __future__ import annotations

import dataclasses
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jev_cascade.config import config_from_dict  # noqa: E402
from jev_cascade.jev import JevError  # noqa: E402
from jev_cascade.eval import (  # noqa: E402
    EvalReport,
    PairSample,
    TaskCase,
    corrupt,
    judge_pair,
    load_task_set,
    run_eval,
    sweep,
)
from jev_cascade.planner import Step  # noqa: E402
from jev_cascade.providers import ProviderError, ScriptedProvider  # noqa: E402
from jev_cascade.testing import ScriptedJev, base_config, judge_config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

STEP = Step(
    id="s1",
    goal="write the reply",
    kind="generate",
    criteria="the reply is specific and polite",
)


def sample(cheap_p: float, corrupt_p: float, verdict: str, cheap_cost: float = 1.0, strong_cost: float = 3.0):
    return PairSample(
        task_id="t",
        step_id="s1",
        goal="g",
        criteria="c",
        cheap_p=cheap_p,
        corrupt_p=corrupt_p,
        verdict=verdict,
        cheap_cost=cheap_cost,
        strong_cost=strong_cost,
    )


class TestCorrupt(unittest.TestCase):
    def test_truncates_deterministically(self) -> None:
        self.assertEqual(corrupt("abcdefghij", 0.5), "abcde")
        self.assertEqual(corrupt("abcdefghij", 0.5), "abcde")

    def test_short_and_empty_inputs_are_safe(self) -> None:
        self.assertEqual(corrupt("", 0.5), "<empty>")
        self.assertEqual(corrupt("   ", 0.5), "<empty>")
        self.assertTrue(corrupt("a", 0.9))

    def test_full_ratio_still_leaves_something(self) -> None:
        self.assertTrue(corrupt("abcd", 0.99))


class TestLoadTaskSet(unittest.TestCase):
    def test_reads_the_bundled_set(self) -> None:
        cases = load_task_set(ROOT / "evals" / "tasks.toml")
        self.assertGreaterEqual(len(cases), 4)
        self.assertTrue(all(isinstance(case, TaskCase) for case in cases))
        self.assertTrue(all(case.task for case in cases))
        ids = [case.id for case in cases]
        self.assertEqual(len(set(ids)), len(ids))

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_task_set(ROOT / "evals" / "nope.toml")


class TestJudgePair(unittest.TestCase):
    def test_cheap_ok_verdict_regardless_of_slot(self) -> None:
        for seed in range(8):
            judge = ScriptedProvider(
                "expensive",
                [json.dumps({"a_sufficient": True, "b_sufficient": True, "better": "A", "reason": "both fine"})],
            )
            result = judge_pair(judge, STEP, "cheap text", "strong text", seed)
            self.assertEqual(result["verdict"], "cheap-ok", f"seed {seed}")

    def test_maps_both_slots_correctly(self) -> None:
        seen_orders = set()
        for seed in range(16):
            judge = ScriptedProvider(
                "expensive",
                [
                    json.dumps(
                        {"a_sufficient": False, "b_sufficient": True, "better": "B", "reason": "b is complete"}
                    )
                ],
            )
            result = judge_pair(judge, STEP, "cheap text", "strong text", seed)
            seen_orders.add(result["order"])
            if result["order"] == "cheap-first":
                self.assertEqual(result["verdict"], "strong-better")
            else:
                self.assertEqual(result["verdict"], "cheap-ok")
        self.assertEqual(seen_orders, {"cheap-first", "strong-first"}, "order must actually be randomized")

    def test_both_bad_verdict(self) -> None:
        judge = ScriptedProvider(
            "expensive", [json.dumps({"a_sufficient": False, "b_sufficient": False, "better": "tie"})]
        )
        result = judge_pair(judge, STEP, "c", "s", 1)
        self.assertEqual(result["verdict"], "both-bad")

    def test_unparsable_reply_is_flagged_not_guessed(self) -> None:
        judge = ScriptedProvider("expensive", ["I think candidate A is better, honestly."])
        result = judge_pair(judge, STEP, "c", "s", 1)
        self.assertEqual(result["verdict"], "unparsed")

    def test_the_prompt_is_blind(self) -> None:
        judge = ScriptedProvider("expensive", [json.dumps({"a_sufficient": True, "b_sufficient": True})])
        judge_pair(judge, STEP, "MARKER-ONE", "MARKER-TWO", 3)
        prompt = judge.prompts[0]
        for leak in ("cheap", "strong", "tier", "model", "gpt", "reasoner"):
            self.assertNotIn(leak, prompt.lower(), f"the prompt must not reveal {leak!r}")
        self.assertIn("CANDIDATE A", prompt)
        self.assertIn("CANDIDATE B", prompt)


class TestSweepAndRecommend(unittest.TestCase):
    def build(self, samples: list[PairSample], tolerance: float = 0.05) -> EvalReport:
        config = base_config()
        config = config.__class__(
            tiers=config.tiers,
            jev=config.jev,
            agent=config.agent,
            eval=config.eval.__class__(
                miss_tolerance=tolerance, threshold_step=0.1, min_judged_pairs=1
            ),
            source_path=config.source_path,
        )
        report = EvalReport(samples=samples)
        report.rows = sweep(report, config)
        from jev_cascade.eval import _recommend

        report.recommended, report.recommended_reason = _recommend(report, config)
        return report

    def test_sweep_counts_escalation_waste_and_misses(self) -> None:
        report = self.build(
            [
                sample(0.9, 0.9, "cheap-ok"),      # safe, no escalation
                sample(0.3, 0.2, "cheap-ok"),      # gate escalates, but wastefully
                sample(0.7, 0.1, "strong-better"), # kept, and that was a mistake
                sample(0.2, 0.1, "strong-better"), # escalated, correctly
            ]
        )
        row = next(r for r in report.rows if abs(r.threshold - 0.6) < 1e-9)
        self.assertAlmostEqual(row.escalate_rate, 0.5)      # two of four are under 0.6
        self.assertAlmostEqual(row.wasted_rate, 0.25)       # one of those two was already good
        self.assertAlmostEqual(row.missed_rate, 0.25)       # one kept output was insufficient
        self.assertAlmostEqual(row.expected_cost, 1.0 + 0.5 * 3.0)

    def test_corrupt_pass_rate_tracks_the_threshold(self) -> None:
        report = self.build([sample(0.9, 0.8, "cheap-ok"), sample(0.9, 0.2, "cheap-ok")])
        high = next(r for r in report.rows if abs(r.threshold - 0.9) < 1e-9)
        low = next(r for r in report.rows if abs(r.threshold - 0.2) < 1e-9)
        # a higher bar means fewer obviously-broken outputs get waved through
        self.assertLess(high.corrupt_pass_rate, low.corrupt_pass_rate)

    def test_recommendation_is_the_cheapest_acceptable_threshold(self) -> None:
        report = self.build(
            [
                sample(0.9, 0.9, "cheap-ok"),
                sample(0.8, 0.8, "cheap-ok"),
                sample(0.2, 0.1, "strong-better"),
            ],
            tolerance=0.0,
        )
        # The only missed shortfall sits at p=0.2, so any threshold above 0.2 is perfect;
        # 0.3 is the cheapest of those.
        self.assertAlmostEqual(report.recommended, 0.3)
        self.assertIn("missed", report.recommended_reason)

    def test_too_few_judged_pairs_refuses_to_recommend_a_change(self) -> None:
        config = base_config()
        config = config.__class__(
            tiers=config.tiers,
            jev=config.jev,
            agent=config.agent,
            eval=config.eval.__class__(min_judged_pairs=8),
            source_path=config.source_path,
        )
        report = EvalReport(samples=[sample(0.2, 0.1, "strong-better"), sample(0.9, 0.9, "cheap-ok")])
        report.rows = sweep(report, config)
        from jev_cascade.eval import _recommend

        threshold, reason = _recommend(report, config)
        self.assertEqual(threshold, config.jev.escalate_below)
        self.assertIn("only 2 judged pairs", reason)
        self.assertIn("below the 8", reason)

    def test_an_always_escalating_threshold_is_never_recommended(self) -> None:
        # every cheap output is insufficient, so only t=1.0 has zero misses, and that
        # threshold buys nothing over just using the strong model
        samples = [sample(0.9, 0.9, "strong-better") for _ in range(3)]
        report = self.build(samples, tolerance=0.0)
        self.assertLess(report.recommended, 1.0)

    def test_impossible_tolerance_reports_the_least_bad_threshold(self) -> None:
        # p=1.0 is never below any threshold on the grid, so the gate never escalates
        # and the miss is unavoidable at every candidate threshold.
        report = self.build([sample(1.0, 0.9, "strong-better")], tolerance=0.0)
        self.assertAlmostEqual(report.recommended, 0.1)
        self.assertIn("missed least", report.recommended_reason)

    def test_gate_separation_compares_real_and_corrupted(self) -> None:
        report = self.build([sample(0.9, 0.2, "cheap-ok"), sample(0.7, 0.4, "cheap-ok")])
        real, corrupted, gap = report.gate_separation()
        self.assertAlmostEqual(real, 0.8)
        self.assertAlmostEqual(corrupted, 0.3)
        self.assertAlmostEqual(gap, 0.5)

    def test_routing_stats_use_the_judge_as_ground_truth(self) -> None:
        under = sample(0.9, 0.9, "strong-better")
        under.route_choice = "cheap"
        over = sample(0.9, 0.9, "cheap-ok")
        over.route_choice = "expensive"
        plain = sample(0.9, 0.9, "cheap-ok")
        plain.route_choice = "cheap"
        report = self.build([under, over, plain])
        counted_under, counted_over, routed = report.routing_stats()
        self.assertEqual((counted_under, counted_over, routed), (1, 1, 3))

    def test_gate_discrimination_measures_the_signal(self) -> None:
        sufficient = sample(0.9, 0.9, "cheap-ok")
        other = sample(0.7, 0.7, "cheap-ok")
        short = sample(0.2, 0.1, "strong-better")
        report = self.build([sufficient, other, short])
        ok_p, short_p, gap = report.discrimination()
        self.assertAlmostEqual(ok_p, 0.8)
        self.assertAlmostEqual(short_p, 0.2)
        self.assertAlmostEqual(gap, 0.6)
        blob = report.to_dict()["gate"]
        self.assertEqual(blob["sufficient_samples"], 2)
        self.assertEqual(blob["insufficient_samples"], 1)
        self.assertAlmostEqual(blob["discrimination"], 0.6)

    def test_discrimination_is_reported_as_unmeasurable_when_one_side_is_empty(self) -> None:
        report = self.build([sample(0.9, 0.9, "cheap-ok")])
        ok_p, short_p, gap = report.discrimination()
        self.assertAlmostEqual(ok_p, short_p)
        self.assertEqual(gap, 0.0)
        self.assertIn("not measurable", report.render(base_config()))

    def test_position_bias_is_measurable(self) -> None:
        # A judge that always picks slot A: when A held the strong output that looks like
        # a real preference, and when A held the cheap output it looks the other way.
        strong_in_a = sample(0.9, 0.9, "strong-better")
        strong_in_a.judge_order = "strong-first"
        cheap_in_a = sample(0.9, 0.9, "cheap-ok")
        cheap_in_a.judge_order = "cheap-first"
        a_loses = sample(0.9, 0.9, "strong-better")
        a_loses.judge_order = "cheap-first"          # A held the cheap output here
        report = self.build([strong_in_a, cheap_in_a, a_loses])
        a_win, judged, rate = report.position_bias()
        self.assertEqual((a_win, judged), (2, 3))
        self.assertAlmostEqual(rate, 2 / 3)

    def test_no_samples_gives_an_empty_sweep_and_a_kept_threshold(self) -> None:
        report = self.build([])
        self.assertEqual(report.rows, [])
        self.assertAlmostEqual(report.recommended, 0.6)
        self.assertIn("no usable pairs", report.recommended_reason)


class TestRunEvalEndToEnd(unittest.TestCase):
    def test_pairs_are_built_only_for_generative_steps(self) -> None:
        config = base_config(candidates=1)
        # per pair the strong provider is used twice: once to generate, once to judge,
        # so its outputs interleave. Jev answers cheap-gate, strong-gate, corrupt-gate.
        strong_and_judge = [
            "STRONG-ONE",
            json.dumps({"a_sufficient": True, "b_sufficient": True, "better": "A", "reason": "ok"}),
            "STRONG-TWO",
            json.dumps({"a_sufficient": False, "b_sufficient": True, "better": "B", "reason": "b complete"}),
        ]
        providers = {
            "cheap": ScriptedProvider("cheap", ["CHEAP-ONE", "CHEAP-TWO"], price=config.tier("cheap").price),
            "expensive": ScriptedProvider(
                "expensive", strong_and_judge, price=config.tier("expensive").price
            ),
        }
        jev = ScriptedJev(verdicts=[0.9, 0.95, 0.9, 0.2, 0.4, 0.1])
        report = run_eval(config, providers=providers, jev=jev, dry_run=True, max_pairs=2)
        self.assertEqual(len(report.samples), 2)
        # two pairs, one generative step each from two different tasks
        self.assertEqual(len({s.task_id for s in report.samples}), 2)
        self.assertEqual({s.step_id for s in report.samples}, {"s1"})
        self.assertNotEqual(report.samples[0].task_id, report.samples[1].task_id)
        self.assertEqual(report.samples[0].cheap_p, 0.9)
        self.assertEqual(report.samples[0].corrupt_p, 0.9)
        self.assertEqual(report.samples[1].cheap_p, 0.2)
        self.assertEqual(report.samples[1].corrupt_p, 0.1)
        # the judge said slot B was sufficient; which tier that is depends on the
        # seed-driven presentation order, so assert the un-mapping, not a fixed string
        second = report.samples[1]
        expected = "strong-better" if second.judge_order == "cheap-first" else "cheap-ok"
        self.assertEqual(second.verdict, expected)
        # B was sufficient; B is the cheap output exactly when the strong one sat in slot A
        self.assertEqual(second.cheap_sufficient, second.judge_order == "strong-first")
        self.assertEqual(report.samples[0].cheap_text, "CHEAP-ONE")
        self.assertEqual(report.planner_failures, 0)
        self.assertTrue(report.rows)
        self.assertTrue(report.stub)
        self.assertGreater(report.total_cost_usd, 0.0)
        self.assertEqual(report.unparsed_judgements, 0)
        self.assertFalse(any(s.verdict == "unparsed" for s in report.samples))

    def test_json_report_has_the_expected_shape(self) -> None:
        config = base_config(candidates=1)
        providers = {
            "cheap": ScriptedProvider("cheap", ["CHEAP"], price=config.tier("cheap").price),
            "expensive": ScriptedProvider(
                "expensive",
                ["STRONG", json.dumps({"a_sufficient": True, "b_sufficient": True, "better": "A"})],
                price=config.tier("expensive").price,
            ),
        }
        jev = ScriptedJev(verdicts=[0.9, 0.95, 0.9])
        report = run_eval(config, providers=providers, jev=jev, dry_run=True, max_pairs=1)
        blob = report.to_dict()
        self.assertEqual(blob["pairs"], 1)
        self.assertIn("sweep", blob)
        self.assertIn("recommended_threshold", blob)
        self.assertIn("gate", blob)
        self.assertIn("routing", blob)
        self.assertIn("judge_position", blob)
        self.assertEqual(blob["samples"][0]["verdict"], "cheap-ok")
        json.dumps(blob)  # must be serializable

    def test_render_includes_the_honesty_caveat(self) -> None:
        config = base_config(candidates=1)
        providers = {
            "cheap": ScriptedProvider("cheap", ["CHEAP"], price=config.tier("cheap").price),
            "expensive": ScriptedProvider(
                "expensive",
                ["STRONG", json.dumps({"a_sufficient": True, "b_sufficient": True, "better": "A"})],
                price=config.tier("expensive").price,
            ),
        }
        jev = ScriptedJev(verdicts=[0.9, 0.95, 0.9])
        report = run_eval(config, providers=providers, jev=jev, dry_run=True, max_pairs=1)
        text = report.render(config)
        self.assertIn("stub", text.lower())
        self.assertIn("judge is a model", text)
        self.assertIn("gate teeth", text)

    def test_a_failed_judge_call_is_recorded_and_the_run_continues(self) -> None:
        class ExplodingJudge(ScriptedProvider):
            """Generates fine, but burns its whole cap thinking when asked to grade."""

            def complete(self, system, user, max_tokens=None, json_mode=False):
                if json_mode:
                    raise ProviderError("judge burned its cap thinking", status=None)
                return super().complete(system, user, max_tokens, json_mode)

        config = base_config(candidates=1)
        providers = {
            "cheap": ScriptedProvider("cheap", ["CHEAP"], price=config.tier("cheap").price),
            "expensive": ExplodingJudge("expensive", ["STRONG"], price=config.tier("expensive").price),
        }
        jev = ScriptedJev(verdicts=[0.9, 0.95, 0.9])
        report = run_eval(config, providers=providers, jev=jev, dry_run=True, max_pairs=1)

        self.assertEqual(report.judge_failures, 1)
        self.assertEqual(report.samples[0].verdict, "judge-error")
        self.assertIn("cap thinking", report.samples[0].judge_note)
        self.assertEqual(report.rows, [], "an unjudged pair cannot be swept")
        self.assertIn("failed outright", report.render(config))
        self.assertEqual(report.verified_pairs, 0)

    def test_a_failed_generation_drops_the_pair_without_voiding_the_run(self) -> None:
        class ExplodingCheap(ScriptedProvider):
            def complete(self, system, user, max_tokens=None, json_mode=False):
                raise ProviderError("cheap burned its cap thinking")

        config = base_config(candidates=1)
        providers = {
            "cheap": ExplodingCheap("cheap", [], price=config.tier("cheap").price),
            "expensive": ScriptedProvider("expensive", [], price=config.tier("expensive").price),
        }
        report = run_eval(
            config, providers=providers, jev=ScriptedJev(verdicts=[0.9]), dry_run=True, max_pairs=2
        )
        self.assertEqual(report.generation_failures, 2)
        self.assertTrue(all(s.verdict == "gen-error" for s in report.samples))
        self.assertEqual(report.rows, [])
        self.assertIn("generations that failed outright", report.render(config))
        self.assertIn("burned its cap", report.samples[0].judge_note)

    def test_the_judge_gets_its_own_larger_cap(self) -> None:
        config = base_config(candidates=1)
        judge = ScriptedProvider(
            "expensive",
            ["STRONG", json.dumps({"a_sufficient": True, "b_sufficient": True, "better": "A"})],
            price=config.tier("expensive").price,
        )
        providers = {
            "cheap": ScriptedProvider("cheap", ["CHEAP"], price=config.tier("cheap").price),
            "expensive": judge,
        }
        run_eval(config, providers=providers, jev=ScriptedJev(verdicts=[0.9, 0.95, 0.9]), dry_run=True, max_pairs=1)
        # first call is the strong generation, second is the judge. With max_tokens = 0
        # the eval passes no override, so each tier's own cap is what gets measured.
        self.assertIsNone(judge.caps[0])
        self.assertEqual(judge.caps[1], config.eval.judge_max_tokens)
        self.assertEqual(judge.json_modes, [False, True])

    def test_a_single_tier_config_is_rejected_with_a_clear_error(self) -> None:
        from jev_cascade.config import config_from_dict

        config = config_from_dict(
            {
                "agent": {"planner_mode": "heuristic", "planner_tier": "only", "verbose": False},
                "tiers": [{"name": "only", "kind": "mock", "model": "m"}],
            }
        )
        with self.assertRaises(ValueError) as ctx:
            run_eval(config, providers={"only": ScriptedProvider("only", ["x"])}, jev=ScriptedJev(), dry_run=True)
        self.assertIn("at least two tiers", str(ctx.exception))


class TestJudgeOnlyTiersInTheEval(unittest.TestCase):
    """The outside judge grades the pairs; it must not be one side of them."""

    def _providers(self, config, judge_output: str):
        return {
            "cheap": ScriptedProvider("cheap", ["CHEAP"], price=config.tier("cheap").price),
            "expensive": ScriptedProvider("expensive", ["STRONG"], price=config.tier("expensive").price),
            "judge": ScriptedProvider("judge", [judge_output], price=config.tier("judge").price),
        }

    def test_the_pair_compares_the_executors_even_with_the_judge_declared_last(self) -> None:
        config = judge_config()  # judge is the last tier AND judge_only
        judge_json = json.dumps({"a_sufficient": True, "b_sufficient": True, "better": "A"})
        providers = self._providers(config, judge_json)

        report = run_eval(
            config, providers=providers, jev=ScriptedJev(verdicts=[0.9, 0.95, 0.9]), dry_run=True, max_pairs=1
        )

        self.assertEqual(report.cheap_tier, "cheap")
        self.assertEqual(report.strong_tier, "expensive")
        self.assertEqual(report.judge_tier, "judge")
        # the strong side is the expensive tier's output, not the judge's
        self.assertEqual(report.samples[0].cheap_text, "CHEAP")
        self.assertEqual(report.samples[0].strong_text, "STRONG")
        # and the judge was consulted exactly once: to grade
        self.assertEqual(providers["judge"].calls, 1)

    def test_an_explicit_strong_tier_is_honoured(self) -> None:
        config = judge_config()
        config = dataclasses.replace(config, eval=dataclasses.replace(config.eval, strong_tier="cheap"))
        judge_json = json.dumps({"a_sufficient": True, "b_sufficient": True, "better": "A"})
        providers = self._providers(config, judge_json)
        # the cheap tier now plays both sides, so it needs one output per side
        providers["cheap"] = ScriptedProvider(
            "cheap", ["CHEAP-A", "CHEAP-B"], price=config.tier("cheap").price
        )

        report = run_eval(
            config, providers=providers, jev=ScriptedJev(verdicts=[0.9, 0.95, 0.9]), dry_run=True, max_pairs=1
        )

        self.assertEqual(report.strong_tier, "cheap")
        self.assertEqual(report.samples[0].cheap_text, "CHEAP-A")
        self.assertEqual(report.samples[0].strong_text, "CHEAP-B")
        # the expensive tier was never asked to run a step this time
        self.assertEqual(providers["expensive"].calls, 0)

    def test_the_report_says_the_judge_is_outside_the_pair(self) -> None:
        config = judge_config()
        judge_json = json.dumps({"a_sufficient": False, "b_sufficient": True, "better": "B"})
        providers = self._providers(config, judge_json)
        report = run_eval(
            config, providers=providers, jev=ScriptedJev(verdicts=[0.9, 0.95, 0.9]), dry_run=True, max_pairs=1
        )

        text = report.render(config)
        self.assertIn("outside tier", text)
        self.assertIn("cheap vs expensive", text)
        self.assertNotIn("is the same tier that produced", text)
        blob = report.to_dict()
        self.assertEqual(blob["cheap_tier"], "cheap")
        self.assertEqual(blob["strong_tier"], "expensive")
        self.assertEqual(blob["judge_tier"], "judge")

    def test_a_self_grading_run_says_so(self) -> None:
        config = base_config(candidates=1)  # judge_tier defaults to the strong tier
        providers = {
            "cheap": ScriptedProvider("cheap", ["CHEAP"], price=config.tier("cheap").price),
            "expensive": ScriptedProvider(
                "expensive",
                ["STRONG", json.dumps({"a_sufficient": True, "b_sufficient": True, "better": "A"})],
                price=config.tier("expensive").price,
            ),
        }
        report = run_eval(
            config, providers=providers, jev=ScriptedJev(verdicts=[0.9, 0.95, 0.9]), dry_run=True, max_pairs=1
        )

        text = report.render(config)
        self.assertIn("self-grading", text)
        self.assertIn("judge is a model", text)

    def test_a_config_with_one_executor_and_one_judge_is_rejected(self) -> None:
        config = config_from_dict(
            {
                "agent": {"planner_mode": "heuristic", "planner_tier": "cheap", "verbose": False},
                "tiers": [
                    {"name": "cheap", "kind": "mock", "model": "m"},
                    {"name": "judge", "kind": "mock", "model": "g", "judge_only": True},
                ],
            }
        )
        with self.assertRaises(ValueError) as ctx:
            run_eval(
                config,
                providers={name: ScriptedProvider(name, ["x"]) for name in config.tier_names},
                jev=ScriptedJev(),
                dry_run=True,
            )
        self.assertIn("judge-only", str(ctx.exception))


class ExplodingGate(ScriptedJev):
    """Answers routing, but every verification gate dies like a dropped connection."""

    def ask(self, state, questions):
        if "meets" in questions:
            raise JevError("System One connection failed: RemoteDisconnected")
        return super().ask(state, questions)


class TestFailedGateIsCounted(unittest.TestCase):
    def test_a_failed_gate_drops_pairs_instead_of_ending_the_run(self) -> None:
        config = base_config(candidates=1)
        providers = {
            "cheap": ScriptedProvider("cheap", ["CHEAP"] * 4, price=config.tier("cheap").price),
            "expensive": ScriptedProvider(
                "expensive", ["STRONG"] * 4, price=config.tier("expensive").price
            ),
        }

        report = run_eval(
            config, providers=providers, jev=ExplodingGate(), dry_run=True, max_pairs=2
        )

        # the run survives, and the failures are visible rather than silent
        self.assertGreater(report.gate_failures, 0)
        self.assertEqual(report.gate_failures, len(report.samples))
        self.assertTrue(all(s.verdict == "gate-error" for s in report.samples))
        self.assertEqual(report.verified_pairs, 0)
        self.assertEqual(report.judge_failures, 0)
        self.assertIn("a dropped connection", report.render(config))
        self.assertEqual(report.to_dict()["gate_failures"], report.gate_failures)
        # nothing was judged, so nothing can be recommended
        self.assertEqual(report.rows, [])

    def test_a_failed_route_costs_only_the_routing_datapoint(self) -> None:
        class ExplodingRouter(ScriptedJev):
            def ask(self, state, questions):
                if "executor" in questions:
                    raise JevError("System One connection failed: RemoteDisconnected")
                return super().ask(state, questions)

        config = base_config(candidates=1)
        providers = {
            "cheap": ScriptedProvider("cheap", ["CHEAP"], price=config.tier("cheap").price),
            "expensive": ScriptedProvider(
                "expensive",
                ["STRONG", json.dumps({"a_sufficient": True, "b_sufficient": True, "better": "A"})],
                price=config.tier("expensive").price,
            ),
        }

        report = run_eval(
            config, providers=providers, jev=ExplodingRouter(verdicts=[0.9, 0.95, 0.9]),
            dry_run=True, max_pairs=1,
        )

        # the pair is still measured; only its routing label is lost
        self.assertEqual(report.gate_failures, 1)
        self.assertEqual(report.verified_pairs, 1)
        self.assertEqual(report.samples[0].route_choice, "cheap")


if __name__ == "__main__":
    unittest.main()
