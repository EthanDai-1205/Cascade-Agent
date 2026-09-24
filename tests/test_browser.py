"""The browser tool: option building, action parsing, the loop guards, and one real run.

It splits in two on purpose. The loop logic is tested against a scripted session, so it
runs anywhere with no browser at all. One test then drives the real bridge against a local
HTML fixture, and skips itself when Node or Playwright is not present, because the cascade
itself does not need either.
"""

from __future__ import annotations

import dataclasses
import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_cascade.browser import (  # noqa: E402
    BACK,
    ESCAPE,
    SCROLL,
    SCROLL_UP,
    STOP,
    BrowserSession,
    build_options,
    describe_action,
    parse_action,
    run_browser_task,
)
from jev_cascade.browser import _render_state  # noqa: E402
from jev_cascade.config import BrowserConfig  # noqa: E402
from jev_cascade.jev import JevError  # noqa: E402
from jev_cascade.testing import ScriptedJev, base_config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "browser" / "task.html"


def a_page(lines=None, controls=None, url="https://example.test/portal", focused="none"):
    return {
        "url": url,
        "title": "Portal",
        "lines": lines if lines is not None else ["Team portal", "Signed in as a guest."],
        "controls": controls
        if controls is not None
        else [
            {"i": 1, "role": "link", "label": "Open the report", "onscreen": True},
            {"i": 2, "role": "link", "label": "Billing settings", "onscreen": True},
        ],
        "focused": focused,
    }


class FakeSession:
    """A scripted page: each act moves to the next state."""

    def __init__(self, states, fail=()) -> None:
        self.states = list(states)
        self.actions: list[dict] = []
        self.fail = set(fail)
        self.closed = False
        self.index = 0

    def state(self):
        return self.states[min(self.index, len(self.states) - 1)]

    def act(self, action):
        self.actions.append(action)
        kind = action.get("kind")
        if kind in self.fail:
            return {"ok": False, "detail": f"the page refused {kind}"}
        if kind in ("click", "type", "scroll_down", "press_enter"):
            self.index += 1
        return {"ok": True, "detail": f"did {kind}"}

    def close(self):
        self.closed = True


class ScriptedChooser:
    """Consumes a list of option fragments, matching by prefix."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.seen: list[list[str]] = []

    def __call__(self, state, options):
        self.seen.append(list(options))
        if not self.answers:
            return options[0]
        want = self.answers.pop(0)
        for option in options:
            if option.startswith(want) or want == option:
                return option
        raise AssertionError(f"nothing on offer starts with {want!r}; offered {options}")


class TestOptions(unittest.TestCase):
    def test_every_option_names_its_target(self) -> None:
        options = build_options(a_page(), text="laya")
        self.assertIn("click the link 'Open the report'", options)
        self.assertIn("click the link 'Billing settings'", options)
        self.assertIn(SCROLL, options)
        self.assertIn(STOP, options)
        self.assertIn("type the prepared text into the focused field", options)

    def test_typing_options_appear_only_when_there_is_text(self) -> None:
        self.assertNotIn("type the prepared text into the focused field", build_options(a_page()))

    def test_off_screen_controls_are_labelled_rather_than_dropped(self) -> None:
        page = a_page(controls=[{"i": 1, "role": "link", "label": "More", "onscreen": False}])
        options = build_options(page)
        self.assertIn("click the link 'More' (below the fold)", options)

    def test_the_option_set_stays_mutually_exclusive(self) -> None:
        labels = list(build_options(a_page()))
        self.assertEqual(len(labels), len(set(labels)))
        self.assertEqual(len(labels), len({label.lower() for label in labels}))

    def test_the_click_options_are_capped(self) -> None:
        page = a_page(controls=[{"i": i, "role": "link", "label": f"Link {i}", "onscreen": True}
                               for i in range(1, 60)])
        # 5 click options plus the six fixed ones: scroll both ways, back, escape, wait, stop.
        self.assertEqual(len(build_options(page, max_controls=5)), 11)

    def test_field_options_appear_only_when_a_writer_is_configured(self) -> None:
        page = a_page(controls=[
            {"i": 1, "role": "text", "label": "Query", "onscreen": True, "field": True},
            {"i": 2, "role": "link", "label": "Open the report", "onscreen": True},
        ])
        self.assertNotIn("type into the text 'Query'", build_options(page))
        self.assertNotIn("type into the text 'Query'", build_options(page, text="prepared"))
        with_writer = build_options(page, writer=True)
        self.assertIn("type into the text 'Query'", with_writer)
        self.assertNotIn("type into the text 'Query' (below the fold)", with_writer)
        # A field is never offered twice, as a click and as a field.
        self.assertNotIn("click the text 'Query'", with_writer)

    def test_a_field_without_a_label_gets_no_writer_option(self) -> None:
        page = a_page(controls=[{"i": 1, "role": "text", "label": "", "onscreen": True, "field": True}])
        self.assertNotIn("type into the text ''", build_options(page, writer=True))

    def test_the_escape_back_and_scroll_up_options_are_always_on_offer(self) -> None:
        options = build_options(a_page())
        for option in (SCROLL_UP, BACK, ESCAPE):
            self.assertIn(option, options)

    def test_a_duplicated_control_label_is_skipped(self) -> None:
        page = a_page(controls=[{"i": 1, "role": "link", "label": "", "onscreen": True}])
        self.assertNotIn("click the link ''", build_options(page))


class TestActionParsing(unittest.TestCase):
    def test_a_click_option_becomes_a_click_on_that_index(self) -> None:
        page = a_page()
        action = parse_action("click the link 'Billing settings'", page)
        self.assertEqual(action, {"kind": "click", "index": 2})

    def test_the_stop_wait_and_scroll_options_map_cleanly(self) -> None:
        page = a_page()
        self.assertEqual(parse_action(STOP, page)["kind"], "done")
        self.assertEqual(parse_action(SCROLL, page)["kind"], "scroll_down")
        self.assertEqual(parse_action(SCROLL_UP, page)["kind"], "scroll_up")
        self.assertEqual(parse_action(BACK, page)["kind"], "back")
        self.assertEqual(parse_action(ESCAPE, page)["kind"], "press_escape")
        self.assertEqual(parse_action("type the prepared text into the focused field", page)["kind"], "type")

    def test_a_field_option_becomes_a_type_into_on_that_index(self) -> None:
        page = a_page(controls=[
            {"i": 3, "role": "text", "label": "Query", "onscreen": True, "field": True},
        ])
        action = parse_action("type into the text 'Query'", page)
        self.assertEqual(action, {"kind": "type_into", "index": 3})

    def test_describe_reads_the_new_actions(self) -> None:
        self.assertEqual(describe_action({"kind": "back"}), "back")
        self.assertEqual(describe_action({"kind": "press_escape"}), "press escape")
        self.assertEqual(
            describe_action({"kind": "type_into", "index": 3, "text": "hello"}),
            "type 5 characters into the field numbered 3",
        )
        self.assertEqual(
            describe_action({"kind": "type_into", "index": 3}),
            "compose text for the field numbered 3",
        )

    def test_an_option_that_matches_nothing_is_not_guessed(self) -> None:
        self.assertEqual(parse_action("click the link 'Not here'", a_page())["kind"], "unknown")


class TestStateRendering(unittest.TestCase):
    def test_the_prompt_carries_the_goal_the_page_and_the_controls(self) -> None:
        text = _render_state(a_page(), "open the report")
        self.assertIn("task: open the report", text)
        self.assertIn("link 'Open the report'", text)
        self.assertIn("Team portal", text)

    def test_a_focused_field_is_reported(self) -> None:
        self.assertIn("focused field: searchbox", _render_state(a_page(focused="searchbox"), "goal"))


class TestTheLoop(unittest.TestCase):
    def _run(self, session, answers, **kwargs):
        config = base_config()
        jev = ScriptedJev(chooser=ScriptedChooser(answers))
        return run_browser_task("open the report", config, actor=jev, session=session, **kwargs)

    def test_a_dry_run_reports_one_step_and_touches_nothing(self) -> None:
        session = FakeSession([a_page(), a_page(lines=["report is open"])])

        result = self._run(session, ["click the link 'Open the report'"], act=False)

        self.assertEqual(len(result.steps), 1)
        self.assertEqual(session.actions, [])
        self.assertIn("dry run", result.stop_reason)
        self.assertFalse(result.acted)
        self.assertIn("Nothing was clicked", result.render())

    def test_acting_clicks_then_stops_when_the_engine_says_done(self) -> None:
        session = FakeSession([a_page(), a_page(lines=["Quarterly report", "revenue of 42 units"])])

        result = self._run(
            session, ["click the link 'Open the report'", STOP], act=True, max_steps=5
        )

        self.assertEqual([s["choice"] for s in result.steps][0], "click the link 'Open the report'")
        self.assertEqual(session.actions, [{"kind": "click", "index": 1}])
        self.assertEqual(result.stop_reason, "the engine judged the goal already achieved")
        self.assertTrue(result.acted)
        self.assertIn("42 units", " ".join(result.final_state["lines"]))

    def test_two_identical_states_in_a_row_stop_the_loop(self) -> None:
        frozen = a_page()

        result = self._run(
            FakeSession([frozen, frozen, frozen]), ["click the link 'Open the report'"] * 4, act=True
        )

        self.assertEqual(result.stop_reason, "two steps with no change on the page")
        self.assertLessEqual(len(result.steps), 3)

    def test_a_successful_type_counts_as_change_even_when_the_page_does_not_move(self) -> None:
        # After the fill, the only difference is the focused field's value — exactly what
        # typing into a search box looks like on a page that otherwise stays put. The loop
        # must see it as progress, or the two-no-op guard ends the run before "press enter".
        blank = a_page(focused="none")
        typed = a_page(focused="Search = 'jev ai computeruse'")
        session = FakeSession([blank, typed])
        result = self._run(
            session, ["press enter", STOP], act=True, text="jev ai computeruse"
        )
        self.assertEqual(result.stop_reason, "the engine judged the goal already achieved")
        self.assertEqual([a["kind"] for a in session.actions], ["press_enter"])

    def test_a_refused_action_counts_as_no_change(self) -> None:
        frozen = a_page()

        result = self._run(
            FakeSession([frozen, frozen, frozen], fail={"click"}),
            ["click the link 'Open the report'"] * 4,
            act=True,
        )

        self.assertEqual(result.stop_reason, "two steps with no change on the page")

    def test_a_low_confidence_decision_stops_instead_of_acting(self) -> None:
        session = FakeSession([a_page()])
        jev = ScriptedJev(chooser=lambda state, options: options[0])
        result = run_browser_task(
            "open the report", base_config(), actor=jev, session=session, act=True,
            min_confidence=0.99,
        )
        self.assertEqual(session.actions, [])
        self.assertIn("not confident enough", result.stop_reason)

    def test_an_off_menu_choice_is_refused_rather_than_acted_on(self) -> None:
        class RogueJev(ScriptedJev):
            """Answers with an option that was never offered, as a broken engine might.

            ScriptedJev cannot express this: it coerces an off-menu pick back to the first
            option, which is the right behaviour for a stub and hides the guard.
            """

            def ask(self, state, questions):
                result = super().ask(state, questions)
                for answer in result.answers.values():
                    if answer.get("type") == "choice":
                        answer["choice"] = "click the link 'Invented'"
                return result

        session = FakeSession([a_page()])
        result = run_browser_task(
            "goal", base_config(), actor=RogueJev(), session=session, act=True
        )
        self.assertEqual(session.actions, [])
        self.assertIn("was not on offer", result.stop_reason)

    def test_a_failing_engine_stops_the_loop_with_the_reason(self) -> None:
        class BrokenJev(ScriptedJev):
            def ask(self, state, questions):
                raise JevError("System One unreachable")

        result = run_browser_task(
            "goal", base_config(), actor=BrokenJev(), session=FakeSession([a_page()]), act=True
        )
        self.assertEqual(result.steps, [])
        self.assertIn("System One unreachable", result.stop_reason)

    def test_the_step_ceiling_is_reported(self) -> None:
        pages = [a_page(url=f"https://example.test/{i}") for i in range(6)]
        result = self._run(
            FakeSession(pages), ["click the link 'Open the report'"] * 6, act=True, max_steps=3
        )
        self.assertEqual(result.stop_reason, "hit the 3-step ceiling")

    def test_the_ledger_gets_the_trace(self) -> None:
        from jev_cascade.ledger import Ledger

        book = Ledger()
        result = self._run(
            FakeSession([a_page(), a_page(lines=["done"])]),
            ["click the link 'Open the report'", STOP],
            act=True,
            ledger=book,
        )
        kinds = [event["type"] for event in book.events]
        self.assertEqual(kinds[0], "browser_start")
        self.assertIn("browser_step", kinds)
        self.assertIn("browser_stop", kinds)
        self.assertIs(result.ledger, book)

    def test_a_field_option_gets_its_words_from_the_writer(self) -> None:
        from jev_cascade.ledger import Ledger

        field_page = a_page(controls=[
            {"i": 1, "role": "link", "label": "Open the report", "onscreen": True},
            {"i": 2, "role": "text", "label": "Query", "onscreen": True, "field": True},
        ])
        book = Ledger()
        config = base_config()
        config = dataclasses.replace(config, browser=BrowserConfig(writer_tier="cheap"))
        result = run_browser_task(
            "search the portal for revenue",
            config,
            actor=ScriptedJev(chooser=ScriptedChooser(["type into the text 'Query'", STOP])),
            session=FakeSession([field_page, field_page, a_page(lines=["results"])]),
            act=True,
            writer_tier="cheap",
            ledger=book,
        )
        self.assertEqual(len(result.steps), 2)
        self.assertEqual(result.steps[0]["status"], "acted")
        writes = [r for r in book.records if r.kind == "write"]
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0].tier, "cheap")
        self.assertEqual(writes[0].status, "done")
        # The bridge action carries the composed text and targets the field by index.
        self.assertEqual(result.steps[0].get("action", {}).get("kind"), "type_into")
        self.assertEqual(result.steps[0]["action"]["index"], 2)
        self.assertTrue(result.steps[0]["action"].get("text"))
        book.close()

    def test_a_failed_writer_gate_stops_the_step_before_anything_is_typed(self) -> None:
        field_page = a_page(controls=[
            {"i": 1, "role": "text", "label": "Query", "onscreen": True, "field": True},
        ])
        session = FakeSession([field_page])
        result = run_browser_task(
            "search the portal",
            base_config(),
            actor=ScriptedJev(
                chooser=ScriptedChooser(["type into the text 'Query'"]), verdicts=[0.1, 0.2]
            ),
            session=session,
            act=True,
            writer_tier="cheap",
        )
        self.assertIn("the writer could not produce a verified string", result.stop_reason)
        self.assertEqual(session.actions, [])


class TestSessionWiring(unittest.TestCase):
    def test_the_session_gets_its_settings_from_the_config(self) -> None:
        import unittest.mock
        from jev_cascade import browser as browser_mod

        config = base_config()
        config = dataclasses.replace(
            config,
            browser=BrowserConfig(node="node-custom", headed=True, playwright="pw-spec"),
        )
        seen: dict = {}

        class RecordingSession(FakeSession):
            def __init__(self, url="about:blank", node="node", headed=False,
                        timeout_s=90.0, playwright_spec="") -> None:
                seen.update(url=url, node=node, headed=headed, playwright_spec=playwright_spec)
                super().__init__([a_page()])

        with unittest.mock.patch.object(browser_mod, "BrowserSession", RecordingSession):
            run_browser_task("goal", config, actor=ScriptedJev(), session=None, act=False)

        self.assertEqual(seen["node"], "node-custom")
        self.assertTrue(seen["headed"])
        self.assertEqual(seen["playwright_spec"], "pw-spec")


@unittest.skipUnless(shutil.which("node"), "the browser tool needs Node on PATH")
class TestAgainstARealBrowser(unittest.TestCase):
    """Drives the real bridge over a local fixture page. No network, no keys, no Jev."""

    def _session(self):
        try:
            session = BrowserSession(url=FIXTURE.as_uri())
        except Exception as exc:  # pragma: no cover - only when Playwright is absent
            self.skipTest(f"no usable Playwright: {exc}")
        self.addCleanup(session.close)
        return session

    def test_it_reads_the_page_and_clicks_the_right_control(self) -> None:
        session = self._session()
        options_seen: list[list[str]] = []

        def chooser(state, options):
            options_seen.append(list(options))
            if any(label.startswith("click the link 'Open the report'") for label in options):
                return next(o for o in options if o.startswith("click the link 'Open the report'"))
            return STOP

        result = run_browser_task(
            "open the report", base_config(), actor=ScriptedJev(chooser=chooser),
            session=session, act=True, max_steps=4,
        )

        first = options_seen[0]
        self.assertIn("click the link 'Open the report'", first)
        self.assertIn("click the link 'Billing settings'", first)
        self.assertEqual(result.stop_reason, "the engine judged the goal already achieved")
        body = " ".join(result.final_state["lines"])
        self.assertIn("42 units", body, "the click should have revealed the report text")
        self.assertNotIn("Open the report", [c["label"] for c in result.final_state["controls"]])

    def test_a_dry_run_leaves_the_page_untouched(self) -> None:
        session = self._session()

        result = run_browser_task(
            "open the report", base_config(),
            actor=ScriptedJev(chooser=lambda state, options: options[0]),
            session=session, act=False,
        )

        self.assertEqual(len(result.steps), 1)
        self.assertNotIn("42 units", " ".join(result.final_state["lines"]))


if __name__ == "__main__":
    unittest.main()
