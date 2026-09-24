"""The writer: draft cleaning, the gate, the retry, and the honest refusal."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_cascade.jev import JevError  # noqa: E402
from jev_cascade.providers import ProviderError, ScriptedProvider  # noqa: E402
from jev_cascade.testing import ScriptedJev, base_config  # noqa: E402
from jev_cascade.writer import clean_draft, compose_for_field, compose_text  # noqa: E402
from jev_cascade.ledger import Ledger  # noqa: E402


class TestCleanDraft(unittest.TestCase):
    def test_it_strips_wrapping_quotes(self) -> None:
        self.assertEqual(clean_draft('"Tokyo"'), "Tokyo")
        self.assertEqual(clean_draft("'Tokyo'"), "Tokyo")

    def test_it_takes_the_first_line_only(self) -> None:
        self.assertEqual(clean_draft("Tokyo\nbecause the goal says Japan"), "Tokyo")

    def test_an_empty_answer_stays_empty(self) -> None:
        self.assertEqual(clean_draft("   \n  "), "")


class TestComposeText(unittest.TestCase):
    def test_a_passing_draft_is_returned_with_its_verdict(self) -> None:
        config = base_config()
        tier = config.tiers[0]
        provider = ScriptedProvider("cheap", ["Tokyo"])
        draft = compose_text(
            "search flights to Japan", "the search field 'Origin'", "",
            tier, ScriptedJev(verdicts=[0.9]), provider=provider,
        )
        self.assertTrue(draft.ok)
        self.assertEqual(draft.text, "Tokyo")
        self.assertEqual(draft.verify_p, 0.9)
        self.assertEqual(draft.attempts, 1)
        self.assertEqual(provider.calls, 1)

    def test_a_failing_gate_gets_one_retry_then_refuses(self) -> None:
        config = base_config()
        tier = config.tiers[0]
        provider = ScriptedProvider("cheap", ["Tokyo", "Kyoto again"])
        draft = compose_text(
            "search flights to Japan", "the search field 'Origin'", "",
            tier, ScriptedJev(verdicts=[0.1, 0.2]), provider=provider,
        )
        self.assertFalse(draft.ok)
        self.assertEqual(draft.attempts, 2)
        self.assertEqual(provider.calls, 2)
        self.assertIn("gate scored", draft.error)

    def test_a_provider_error_is_reported_not_raised(self) -> None:
        config = base_config()
        tier = config.tiers[0]

        class Failing:
            tier = "cheap"
            stub = True

            def complete(self, *args, **kwargs):
                raise ProviderError("the tier is down", status=500)

        draft = compose_text(
            "goal", "the search field 'Origin'", "",
            tier, ScriptedJev(), provider=Failing(),
        )
        self.assertFalse(draft.ok)
        self.assertIn("the tier is down", draft.error)
        self.assertEqual(draft.attempts, 2)

    def test_a_failed_gate_stops_before_burning_a_second_draft(self) -> None:
        config = base_config()
        tier = config.tiers[0]

        class Broken(ScriptedJev):
            def ask(self, state, questions):
                raise JevError("the gate engine is down")

        provider = ScriptedProvider("cheap", ["Tokyo"])
        draft = compose_text(
            "goal", "the search field 'Origin'", "", tier, Broken(), provider=provider
        )
        self.assertFalse(draft.ok)
        self.assertIn("gate failed", draft.error)
        self.assertEqual(provider.calls, 1)


class TestComposeForField(unittest.TestCase):
    def test_the_spend_is_booked_under_the_writer_tier(self) -> None:
        config = base_config()
        book = Ledger()
        control = {"i": 2, "role": "text field", "label": "Search"}
        draft = compose_for_field(
            "search for jev",
            {"lines": ["A portal", "Sign in"]},
            control,
            (config.tiers[0], ScriptedProvider("cheap", ["jev cascade"])),
            ScriptedJev(verdicts=[0.9]),
            book,
            3,
            tool="computer",
        )
        self.assertTrue(draft.ok)
        writes = [r for r in book.records if r.kind == "write"]
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0].step_id, "computer-write-3")
        self.assertEqual(writes[0].tier, "cheap")
        self.assertEqual(writes[0].verify_p, 0.9)
        kinds = [event["type"] for event in book.events]
        self.assertIn("write", kinds)
        book.close()

    def test_a_refused_draft_is_booked_as_refused(self) -> None:
        config = base_config()
        book = Ledger()
        control = {"i": 1, "role": "text field", "label": "Body"}
        draft = compose_for_field(
            "goal",
            {"lines": []},
            control,
            (config.tiers[0], ScriptedProvider("cheap", ["a draft", "another draft"])),
            ScriptedJev(verdicts=[0.0, 0.1]),
            book,
            1,
            tool="browser",
        )
        self.assertFalse(draft.ok)
        self.assertEqual(book.records[0].status, "refused")
        book.close()
