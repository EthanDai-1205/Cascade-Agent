"""Step typing and JSON robustness."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_cascade.testing import StaticPlanner, base_config  # noqa: E402
from jev_cascade.planner import (  # noqa: E402
    HeuristicPlanner,
    LLMPlanner,
    Step,
    parse_plan,
    plan_to_json,
)
from jev_cascade.providers import Completion, ScriptedProvider  # noqa: E402

GOOD_PLAN = """Here is the plan you asked for:

```json
{"steps": [
  {"id": "s1", "goal": "label the ticket", "kind": "decide", "answer_type": "choice",
   "options": ["low", "high"], "criteria": "best matching label"},
  {"id": "s2", "goal": "write the reply", "kind": "generate",
   "criteria": "polite and specific", "context_needed": ["the label"]}
]}
```
"""


class TestParsePlan(unittest.TestCase):
    def test_parses_fenced_json_with_prose(self) -> None:
        steps, error = parse_plan(GOOD_PLAN, "task", 8)
        self.assertEqual(error, "")
        self.assertEqual([step.id for step in steps], ["s1", "s2"])
        self.assertTrue(steps[0].is_decision)
        self.assertEqual(steps[0].options, ["low", "high"])
        self.assertFalse(steps[1].is_decision)

    def test_rejects_non_json(self) -> None:
        steps, error = parse_plan("I cannot help with that", "task", 8)
        self.assertEqual(steps, [])
        self.assertIn("not valid JSON", error)

    def test_rejects_json_without_steps(self) -> None:
        steps, error = parse_plan('{"plan": []}', "task", 8)
        self.assertEqual(steps, [])
        self.assertIn("steps", error)

    def test_caps_step_count(self) -> None:
        blob = {"steps": [{"id": f"s{i}", "goal": f"goal {i}", "kind": "generate"} for i in range(20)]}
        steps, _ = parse_plan(__import__("json").dumps(blob), "task", 3)
        self.assertEqual(len(steps), 3)

    def test_unknown_kind_becomes_generate(self) -> None:
        steps, _ = parse_plan('{"steps":[{"goal":"do it","kind":"vibes"}]}', "task", 5)
        self.assertEqual(steps[0].kind, "generate")

    def test_choice_with_one_option_degrades_to_noul(self) -> None:
        steps, _ = parse_plan(
            '{"steps":[{"goal":"is it urgent","kind":"decide","answer_type":"choice","options":["only"]}]}',
            "task",
            5,
        )
        self.assertEqual(steps[0].answer_type, "noul")
        self.assertEqual(steps[0].options, [])

    def test_decide_without_answer_type_uses_the_options(self) -> None:
        steps, _ = parse_plan(
            '{"steps":[{"goal":"pick","kind":"decide","options":["a","b"]}]}', "task", 5
        )
        self.assertEqual(steps[0].answer_type, "choice")

    def test_criteria_defaults_instead_of_staying_empty(self) -> None:
        steps, _ = parse_plan('{"steps":[{"goal":"do the thing","kind":"generate"}]}', "task", 5)
        self.assertIn("do the thing", steps[0].criteria)

    def test_duplicate_ids_are_disambiguated(self) -> None:
        steps, _ = parse_plan(
            '{"steps":[{"id":"x","goal":"one"},{"id":"x","goal":"two"}]}', "task", 5
        )
        self.assertEqual(len({step.id for step in steps}), 2)

    def test_blank_goals_are_dropped(self) -> None:
        steps, error = parse_plan('{"steps":[{"goal":"  "},{"goal":"real step"}]}', "task", 5)
        self.assertEqual(len(steps), 1)
        self.assertEqual(error, "")


class TestLLMPlanner(unittest.TestCase):
    def test_falls_back_to_one_step_when_the_model_rambles(self) -> None:
        provider = ScriptedProvider("cheap", ["no json here at all"])
        plan = LLMPlanner(provider, "cheap", 6).plan("do something")
        self.assertTrue(plan.fell_back)
        self.assertEqual(len(plan.steps), 1)
        self.assertIn("do something", plan.steps[0].goal)
        self.assertTrue(plan.error)

    def test_falls_back_when_the_provider_raises(self) -> None:
        class Boom:
            tier = "cheap"
            stub = True

            def complete(self, system: str, user: str, max_tokens=None, json_mode=False) -> Completion:
                raise RuntimeError("provider down")

        plan = LLMPlanner(Boom(), "cheap", 6).plan("do something")
        self.assertTrue(plan.fell_back)
        self.assertIn("provider down", plan.error)

    def test_decision_steps_are_hoisted_to_the_front(self) -> None:
        provider = ScriptedProvider(
            "cheap",
            ['{"steps":[{"id":"a","goal":"write it","kind":"generate"},'
             '{"id":"b","goal":"label it","kind":"decide","answer_type":"choice","options":["x","y"]}]}'],
        )
        plan = LLMPlanner(provider, "cheap", 6).plan("task")
        self.assertEqual([step.id for step in plan.steps], ["b", "a"])

    def test_records_planner_cost_and_tokens(self) -> None:
        provider = ScriptedProvider("cheap", ['{"steps":[{"goal":"g","kind":"generate"}]}'])
        plan = LLMPlanner(provider, "cheap", 6).plan("task")
        self.assertGreater(plan.tokens_in, 0)
        self.assertGreater(plan.tokens_out, 0)
        self.assertEqual(plan.tier, "cheap")

    def test_planner_asks_for_json_mode_and_its_own_token_cap(self) -> None:
        provider = ScriptedProvider("cheap", ['{"steps":[{"goal":"g","kind":"generate"}]}'])
        LLMPlanner(provider, "cheap", 6, max_tokens=1234).plan("task")
        self.assertEqual(provider.json_modes, [True])
        self.assertEqual(provider.caps, [1234])

    def test_a_cut_off_plan_is_reported_as_a_budget_problem(self) -> None:
        class Truncating(ScriptedProvider):
            def complete(self, system, user, max_tokens=None, json_mode=False) -> Completion:
                completion = super().complete(system, user, max_tokens, json_mode)
                completion.truncated = True
                return completion

        provider = Truncating("cheap", ['{"steps": [{"goal": "g"'])
        plan = LLMPlanner(provider, "cheap", 6, max_tokens=512).plan("task")
        self.assertTrue(plan.fell_back)
        self.assertIn("cut off at max_tokens=512", plan.error)
        self.assertIn("planner_max_tokens", plan.error)

    def test_an_unparseable_plan_includes_an_excerpt(self) -> None:
        provider = ScriptedProvider("cheap", ["Sure! Here is a plan: I would start by renaming things."])
        plan = LLMPlanner(provider, "cheap", 6).plan("task")
        self.assertTrue(plan.fell_back)
        self.assertIn("reply began", plan.error)
        self.assertIn("Sure! Here is a plan", plan.error)

    def test_dry_run_uses_the_heuristic_planner(self) -> None:
        plan = LLMPlanner(ScriptedProvider("cheap", ["junk"]), "cheap", 6).plan("task", dry_run=True)
        self.assertFalse(plan.fell_back)
        self.assertEqual(plan.model, "heuristic")


class TestHeuristicPlanner(unittest.TestCase):
    def test_always_starts_with_a_decision_step(self) -> None:
        plan = HeuristicPlanner("cheap").plan("rename the flag")
        self.assertTrue(plan.steps[0].is_decision)
        self.assertEqual(plan.steps[0].answer_type, "choice")
        self.assertGreaterEqual(len(plan.steps[0].options), 2)

    def test_splits_on_separators(self) -> None:
        plan = HeuristicPlanner("cheap").plan("write A and then write B; and also write C")
        generate_steps = [step for step in plan.steps if not step.is_decision]
        self.assertEqual(len(generate_steps), 3)

    def test_dependency_chain_is_accumulative(self) -> None:
        plan = HeuristicPlanner("cheap").plan("first thing and then second thing")
        generate_steps = [step for step in plan.steps if not step.is_decision]
        self.assertEqual(generate_steps[0].depends_on, ["classify"])
        self.assertEqual(generate_steps[1].depends_on, ["classify", "s1"])


class TestSerialisation(unittest.TestCase):
    def test_plan_json_round_trips_step_fields(self) -> None:
        plan = StaticPlanner([Step(id="a", goal="g", kind="decide", answer_type="noul")]).plan("t")
        blob = plan_to_json(plan)
        self.assertIn('"answer_type": "noul"', blob)
        self.assertIn('"id": "a"', blob)

    def test_config_has_a_usable_planner_tier(self) -> None:
        config = base_config()
        self.assertEqual(config.planner_tier().name, "cheap")


if __name__ == "__main__":
    unittest.main()
