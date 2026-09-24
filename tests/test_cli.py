"""CLI wiring: overrides that change which tiers the eval compares and who grades."""

from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_cascade.cli import main  # noqa: E402

CONFIG = """
[agent]
planner_tier = "cheap"
planner_mode = "heuristic"
verbose = false

[jev]
verify = true

[eval]
judge_tier = "expensive"
task_set = "evals/tasks.toml"

[[tiers]]
name = "cheap"
kind = "mock"
model = "cheap-mock"

[[tiers]]
name = "expensive"
kind = "mock"
model = "expensive-mock"

[[tiers]]
name = "judge"
kind = "mock"
model = "judge-mock"
judge_only = true
"""


class EvalFlagTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.config_path = Path(self.dir.name) / "config.toml"
        self.config_path.write_text(CONFIG)
        # the eval resolves task_set relative to the repo, so run from there
        self.root = Path(__file__).resolve().parent.parent

    def _eval(self, *extra: str) -> dict:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(
                [
                    "--config",
                    str(self.config_path),
                    "eval",
                    "--dry-run",
                    "--json",
                    "--pairs",
                    "1",
                    *extra,
                ]
            )
        self.assertEqual(code, 0)
        return json.loads(buffer.getvalue())

    def test_defaults_come_from_the_config(self) -> None:
        blob = self._eval()
        self.assertEqual(blob["cheap_tier"], "cheap")
        self.assertEqual(blob["strong_tier"], "expensive")
        self.assertEqual(blob["judge_tier"], "expensive")

    def test_judge_flag_points_the_eval_at_an_outside_tier(self) -> None:
        blob = self._eval("--judge", "judge")
        self.assertEqual(blob["judge_tier"], "judge")
        # the two sides are unchanged: an outside judge must not become one of them
        self.assertEqual(blob["cheap_tier"], "cheap")
        self.assertEqual(blob["strong_tier"], "expensive")

    def test_strong_flag_changes_the_pair_not_the_judge(self) -> None:
        blob = self._eval("--strong", "cheap", "--judge", "judge")
        self.assertEqual(blob["strong_tier"], "cheap")
        self.assertEqual(blob["judge_tier"], "judge")

    def test_an_unknown_judge_is_a_clear_error(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self._eval("--judge", "nope")
        self.assertIn("not configured", str(ctx.exception))

    def test_a_judge_only_tier_cannot_be_the_strong_side(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self._eval("--strong", "judge")
        self.assertIn("judge_only", str(ctx.exception))

    def test_a_judge_only_tier_cannot_be_pinned_for_run(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            main(
                [
                    "--config",
                    str(self.config_path),
                    "run",
                    "do something",
                    "--dry-run",
                    "--tier",
                    "judge",
                ]
            )
        self.assertIn("cannot run a step", str(ctx.exception))

    def test_a_pinnable_tier_still_works_for_run(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(
                [
                    "--config",
                    str(self.config_path),
                    "run",
                    "write a paragraph about teapots",
                    "--dry-run",
                    "--tier",
                    "expensive",
                ]
            )
        self.assertEqual(code, 0)


@unittest.skipUnless(shutil.which("node"), "the browser tool needs Node on PATH")
class BrowseCommandTests(unittest.TestCase):
    """The browse command, dry, against a local fixture. No network, no keys, no model."""

    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.config_path = Path(self.dir.name) / "config.toml"
        self.config_path.write_text(CONFIG)
        self.fixture = (
            Path(__file__).resolve().parent / "fixtures" / "browser" / "task.html"
        ).as_uri()

    def _browse(self, *extra: str) -> dict:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(
                [
                    "--config",
                    str(self.config_path),
                    "browse",
                    "open the report",
                    "--url",
                    self.fixture,
                    "--dry-run",
                    "--json",
                    *extra,
                ]
            )
        self.assertEqual(code, 0)
        return json.loads(buffer.getvalue())

    def test_a_dry_run_reports_one_step_and_does_not_act(self) -> None:
        blob = self._browse()
        self.assertFalse(blob["act"])
        self.assertEqual(len(blob["steps"]), 1)
        self.assertIn("dry run", blob["stop_reason"])
        self.assertTrue(blob["stub"], "the run says it used a stub engine")

    def test_the_report_names_the_goal_and_the_final_page(self) -> None:
        blob = self._browse()
        self.assertEqual(blob["goal"], "open the report")
        self.assertIn("task.html", blob["final"]["url"])

    def test_a_step_ceiling_can_be_lowered(self) -> None:
        blob = self._browse("--steps", "1")
        self.assertLessEqual(len(blob["steps"]), 1)


if __name__ == "__main__":
    unittest.main()
