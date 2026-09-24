"""Config loading, validation, and tier ordering."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_cascade.config import (  # noqa: E402
    ConfigError,
    Price,
    config_from_dict,
    expand_vars,
    load_config,
)

ROOT = Path(__file__).resolve().parent.parent


class TestExpandVars(unittest.TestCase):
    def test_plain_var(self) -> None:
        os.environ["JC_TEST_VALUE"] = "abc"
        self.assertEqual(expand_vars("x-${JC_TEST_VALUE}-y"), "x-abc-y")

    def test_default_when_missing(self) -> None:
        os.environ.pop("JC_TEST_MISSING", None)
        self.assertEqual(expand_vars("${JC_TEST_MISSING:-fallback}"), "fallback")

    def test_empty_when_no_default(self) -> None:
        os.environ.pop("JC_TEST_MISSING", None)
        self.assertEqual(expand_vars("[${JC_TEST_MISSING}]"), "[]")


class TestPrice(unittest.TestCase):
    def test_cost_math(self) -> None:
        price = Price(input=1.0, output=2.0)
        self.assertAlmostEqual(price.cost(1_000_000, 1_000_000), 3.0)
        self.assertAlmostEqual(price.cost(500_000, 0), 0.5)

    def test_zero_price_is_reported_unknown(self) -> None:
        self.assertFalse(Price().known)
        self.assertTrue(Price(input=1.0).known)


class TestConfigValidation(unittest.TestCase):
    def test_needs_at_least_one_tier(self) -> None:
        with self.assertRaises(ConfigError):
            config_from_dict({"tiers": []})

    def test_rejects_duplicate_tier_names(self) -> None:
        with self.assertRaises(ConfigError):
            config_from_dict({"tiers": [{"name": "a"}, {"name": "a"}]})

    def test_rejects_unknown_kind(self) -> None:
        with self.assertRaises(ConfigError):
            config_from_dict({"tiers": [{"name": "a", "kind": "telepathy"}]})

    def test_rejects_bad_planner_tier(self) -> None:
        with self.assertRaises(ConfigError):
            config_from_dict({"agent": {"planner_tier": "nope"}, "tiers": [{"name": "cheap"}]})

    def test_rejects_escalate_threshold_out_of_range(self) -> None:
        with self.assertRaises(ConfigError):
            config_from_dict({"jev": {"escalate_below": 1.5}, "tiers": [{"name": "cheap"}]})

    def test_escalation_walks_declared_order(self) -> None:
        config = config_from_dict(
            {"tiers": [{"name": "cheap"}, {"name": "middle"}, {"name": "expensive"}]}
        )
        self.assertEqual(config.stronger_than("cheap").name, "middle")
        self.assertEqual(config.stronger_than("middle").name, "expensive")
        self.assertIsNone(config.stronger_than("expensive"))

    def test_unknown_tier_lookup_raises(self) -> None:
        config = config_from_dict({"tiers": [{"name": "cheap"}]})
        with self.assertRaises(ConfigError):
            config.tier("nope")


class TestComputerSection(unittest.TestCase):
    def test_defaults_exist_without_the_section(self) -> None:
        config = config_from_dict({"tiers": [{"name": "cheap"}]})
        self.assertEqual(config.computer.actor, "jev")
        self.assertEqual(config.computer.max_steps, 10)
        self.assertEqual(config.computer.allowed_apps, "")
        self.assertEqual(config.computer.writer_tier, "")

    def test_rejects_a_bad_actor(self) -> None:
        with self.assertRaises(ConfigError):
            config_from_dict({"computer": {"actor": "gpt"}, "tiers": [{"name": "cheap"}]})

    def test_allowed_apps_round_trips(self) -> None:
        config = config_from_dict(
            {"computer": {"allowed_apps": "Safari, Notes "}, "tiers": [{"name": "cheap"}]}
        )
        self.assertEqual(config.computer.allowed_apps, "Safari, Notes")


class TestWriterTier(unittest.TestCase):
    def test_a_writer_tier_must_name_a_tier_that_can_run_a_step(self) -> None:
        with self.assertRaises(ConfigError):
            config_from_dict({"browser": {"writer_tier": "nope"}, "tiers": [{"name": "cheap"}]})
        with self.assertRaises(ConfigError):
            config_from_dict({"computer": {"writer_tier": "nope"}, "tiers": [{"name": "cheap"}]})

    def test_a_judge_only_tier_cannot_write(self) -> None:
        with self.assertRaises(ConfigError):
            config_from_dict(
                {
                    "browser": {"writer_tier": "judge"},
                    "tiers": [
                        {"name": "cheap"},
                        {"name": "judge", "judge_only": True},
                    ],
                }
            )

    def test_a_valid_writer_tier_is_kept(self) -> None:
        config = config_from_dict(
            {"browser": {"writer_tier": "cheap"}, "computer": {"writer_tier": "cheap"},
             "tiers": [{"name": "cheap"}]}
        )
        self.assertEqual(config.browser.writer_tier, "cheap")
        self.assertEqual(config.computer.writer_tier, "cheap")


class TestJudgeOnlyTiers(unittest.TestCase):
    """A tier that exists to grade must not be reachable as an executor."""

    def test_judge_only_tier_is_not_in_the_ladder(self) -> None:
        config = config_from_dict(
            {
                "tiers": [
                    {"name": "cheap"},
                    {"name": "expensive"},
                    {"name": "judge", "judge_only": True},
                ]
            }
        )
        self.assertEqual(config.tier_names, ["cheap", "expensive", "judge"])
        self.assertEqual(config.executor_names, ["cheap", "expensive"])
        self.assertEqual(config.strongest_executor.name, "expensive")
        self.assertEqual(config.stronger_than("cheap").name, "expensive")
        self.assertIsNone(config.stronger_than("expensive"))

    def test_judge_only_tier_in_the_middle_is_skipped(self) -> None:
        config = config_from_dict(
            {
                "tiers": [
                    {"name": "cheap"},
                    {"name": "judge", "judge_only": True},
                    {"name": "expensive"},
                ]
            }
        )
        self.assertEqual(config.stronger_than("cheap").name, "expensive")

    def test_stronger_than_returns_none_for_a_judge_tier(self) -> None:
        config = config_from_dict(
            {"tiers": [{"name": "cheap"}, {"name": "judge", "judge_only": True}]}
        )
        self.assertIsNone(config.stronger_than("judge"))

    def test_first_tier_cannot_be_judge_only(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            config_from_dict({"tiers": [{"name": "judge", "judge_only": True}, {"name": "cheap"}]})
        self.assertIn("first tier", str(ctx.exception))

    def test_a_config_with_no_executor_is_rejected(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            config_from_dict({"tiers": [{"name": "judge", "judge_only": True}]})
        self.assertIn("judge_only", str(ctx.exception))

    def test_planner_tier_cannot_be_a_judge_tier(self) -> None:
        with self.assertRaises(ConfigError):
            config_from_dict(
                {
                    "agent": {"planner_tier": "judge"},
                    "tiers": [{"name": "cheap"}, {"name": "judge", "judge_only": True}],
                }
            )

    def test_force_tier_cannot_be_a_judge_tier(self) -> None:
        with self.assertRaises(ConfigError):
            config_from_dict(
                {
                    "agent": {"force_tier": "judge"},
                    "tiers": [{"name": "cheap"}, {"name": "judge", "judge_only": True}],
                }
            )

    def test_strong_tier_defaults_to_the_strongest_executor(self) -> None:
        config = config_from_dict(
            {
                "tiers": [
                    {"name": "cheap"},
                    {"name": "expensive"},
                    {"name": "judge", "judge_only": True},
                ]
            }
        )
        # not "judge", which is declared last
        self.assertEqual(config.strong_tier().name, "expensive")

    def test_strong_tier_override_must_be_an_executor(self) -> None:
        tiers = [
            {"name": "cheap"},
            {"name": "expensive"},
            {"name": "judge", "judge_only": True},
        ]
        ok = config_from_dict({"eval": {"strong_tier": "cheap"}, "tiers": tiers})
        self.assertEqual(ok.strong_tier(ok.eval.strong_tier).name, "cheap")
        config = config_from_dict({"tiers": tiers})
        with self.assertRaises(ConfigError):
            config.strong_tier("judge")
        with self.assertRaises(ConfigError):
            config_from_dict({"eval": {"strong_tier": "judge"}, "tiers": tiers})


class TestExampleConfig(unittest.TestCase):
    def test_example_config_loads(self) -> None:
        config = load_config(ROOT / "config.example.toml")
        self.assertEqual(config.tier_names, ["cheap", "expensive"])
        self.assertEqual(config.agent.planner_tier, "cheap")
        self.assertTrue(config.jev.verify)
        self.assertEqual(config.source_path, str(ROOT / "config.example.toml"))

    def test_missing_file_is_a_clear_error(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            load_config(ROOT / "does-not-exist.toml")
        self.assertIn("not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
