"""The computer tool: option building, action parsing, the loop guards.

It follows the browser test's split: everything here runs against a scripted
desktop session, so it needs no Accessibility grant and no Swift. One class at
the end starts the real bridge read-only and skips itself when the Swift
toolchain is absent or the Accessibility permission has not been granted.
"""

from __future__ import annotations

import dataclasses
import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_cascade.computer import (  # noqa: E402
    ESCAPE,
    ENTER,
    SCROLL_DOWN,
    SCROLL_UP,
    STOP,
    WAIT,
    build_computer_options,
    check_permissions,
    parse_computer_action,
    run_computer_task,
)
from jev_cascade.computer import _render_computer_state  # noqa: E402
from jev_cascade.config import ComputerConfig  # noqa: E402
from jev_cascade.testing import ScriptedJev, base_config  # noqa: E402


def a_desktop(
    lines=None,
    controls=None,
    apps=None,
    app="Safari",
    bundle="com.apple.Safari",
    window="Start Page",
    focused="none",
) -> dict:
    return {
        "app": app,
        "bundle": bundle,
        "window": window,
        "lines": lines if lines is not None else ["Welcome to Safari", "Favorites"],
        "controls": controls
        if controls is not None
        else [
            {"i": 1, "role": "button", "label": "New Tab", "onscreen": True, "field": False},
            {"i": 2, "role": "text field", "label": "Search", "onscreen": True, "field": True, "value": ""},
        ],
        "apps": apps if apps is not None else ["Safari", "Notes", "ZCode"],
        "focused": focused,
    }


class FakeDesktop:
    """A scripted screen: each acted step moves to the next state."""

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
            return {"ok": False, "detail": f"the app refused {kind}"}
        if kind in ("click", "set_value", "type", "press_enter", "switch", "scroll"):
            self.index += 1
        return {"ok": True, "detail": f"did {kind}"}

    def close(self):
        self.closed = True


class ScriptedChooser:
    """Consumes a list of option fragments, matching by prefix."""

    def __init__(self, answers) -> None:
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


def desktop_config(**computer_overrides) -> object:
    computer = {"actor": "jev", "max_steps": 4, "min_confidence": 0.35, "max_controls": 25}
    computer.update(computer_overrides)
    return dataclasses.replace(base_config(), computer=ComputerConfig(**computer))


class TestOptions(unittest.TestCase):
    def test_every_option_names_its_target(self) -> None:
        options = build_computer_options(a_desktop(), writer=True)
        self.assertIn("click the button 'New Tab'", options)
        self.assertIn("set the text field 'Search'", options)

    def test_field_and_press_options_stay_mutually_exclusive(self) -> None:
        options = build_computer_options(a_desktop(), writer=True)
        self.assertEqual(len(options), len(set(options)))

    def test_a_field_is_never_offered_as_a_click_when_a_writer_exists(self) -> None:
        options = build_computer_options(a_desktop(), writer=True)
        self.assertNotIn("click the text field 'Search'", options)

    def test_without_a_writer_a_field_is_just_a_clickable_control(self) -> None:
        # No writer, no --text: there are no words to set, so the field is offered as a
        # plain click (to focus it) and never as a "set" that could not be carried out.
        options = build_computer_options(a_desktop())
        self.assertNotIn("set the text field 'Search'", options)
        self.assertIn("click the text field 'Search'", options)

    def test_running_apps_become_switch_options_except_the_frontmost(self) -> None:
        options = build_computer_options(a_desktop())
        self.assertIn("switch to the app 'Notes'", options)
        self.assertIn("switch to the app 'ZCode'", options)
        self.assertNotIn("switch to the app 'Safari'", options)

    def test_prepared_text_options_need_text_and_a_textable_focus(self) -> None:
        # With text but nothing focused, the option must not exist: keystrokes without
        # a field under them go into whatever the app has under the cursor.
        self.assertNotIn(
            "type the prepared text into the focused field", build_computer_options(a_desktop())
        )
        self.assertNotIn(
            "type the prepared text into the focused field",
            build_computer_options(a_desktop(), text="hello"),
        )
        self.assertNotIn(ENTER, build_computer_options(a_desktop(), text="hello"))
        ready = build_computer_options(a_desktop(focused="text area 'Body'"), text="hello")
        self.assertIn("type the prepared text into the focused field", ready)
        self.assertIn(ENTER, ready)

    def test_writer_mode_still_offers_enter_but_not_prepared_text(self) -> None:
        options = build_computer_options(a_desktop(), writer=True)
        self.assertIn(ENTER, options)
        self.assertNotIn("type the prepared text into the focused field", options)

    def test_off_screen_controls_are_labelled_rather_than_dropped(self) -> None:
        controls = [{"i": 1, "role": "button", "label": "Deep Setting", "onscreen": False, "field": False}]
        options = build_computer_options(a_desktop(controls=controls))
        self.assertIn("click the button 'Deep Setting' (off-screen)", options)

    def test_the_option_set_is_capped(self) -> None:
        controls = [
            {"i": i, "role": "button", "label": f"Button {i}", "onscreen": True, "field": False}
            for i in range(1, 60)
        ]
        # 5 control options plus the five fixed ones: scroll both ways, escape, wait, stop.
        self.assertEqual(len(build_computer_options(a_desktop(controls=controls, apps=[]), max_controls=5)), 10)

    def test_a_pinned_app_is_not_offered_app_switches(self) -> None:
        # --app pins where the work happens; a goal that says "note" beside a list of
        # running notes-apps is an invitation to wander.
        options = build_computer_options(a_desktop(), pinned_app=True)
        self.assertNotIn("switch to the app 'Notes'", options)
        self.assertNotIn("switch to the app 'ZCode'", options)

    def test_the_fixed_options_are_always_there(self) -> None:
        options = build_computer_options(a_desktop())
        for option in (SCROLL_DOWN, SCROLL_UP, ESCAPE, WAIT, STOP):
            self.assertIn(option, options)
        # Enter is only meaningful once something can be typed: with prepared text,
        # or with a writer configured to compose it.
        self.assertNotIn(ENTER, options)
        self.assertIn(ENTER, build_computer_options(a_desktop(focused="text field 'Q'"), text="hello"))
        self.assertIn(ENTER, build_computer_options(a_desktop(), writer=True))


class TestActionParsing(unittest.TestCase):
    def test_a_click_option_becomes_a_click_on_that_index(self) -> None:
        action = parse_computer_action("click the button 'New Tab'", a_desktop())
        self.assertEqual(action, {"kind": "click", "index": 1})

    def test_a_field_option_becomes_a_set_value_on_that_index(self) -> None:
        action = parse_computer_action("set the text field 'Search'", a_desktop())
        self.assertEqual(action, {"kind": "set_value", "index": 2})

    def test_a_switch_option_names_the_app(self) -> None:
        action = parse_computer_action("switch to the app 'Notes'", a_desktop())
        self.assertEqual(action, {"kind": "switch", "app": "Notes"})

    def test_the_fixed_options_map_cleanly(self) -> None:
        screen = a_desktop()
        self.assertEqual(parse_computer_action(STOP, screen)["kind"], "done")
        self.assertEqual(parse_computer_action(SCROLL_DOWN, screen), {"kind": "scroll", "direction": "down"})
        self.assertEqual(parse_computer_action(SCROLL_UP, screen), {"kind": "scroll", "direction": "up"})
        self.assertEqual(parse_computer_action(ESCAPE, screen)["kind"], "press_escape")

    def test_an_option_that_matches_nothing_is_not_guessed(self) -> None:
        self.assertEqual(parse_computer_action("click the button 'Not here'", a_desktop())["kind"], "unknown")


class TestStateRendering(unittest.TestCase):
    def test_the_prompt_carries_the_app_the_window_and_the_controls(self) -> None:
        text = _render_computer_state(a_desktop(), "open a private tab")
        self.assertIn("task: open a private tab", text)
        self.assertIn("app: Safari (com.apple.Safari)", text)
        self.assertIn("window: Start Page", text)
        self.assertIn("button 'New Tab'", text)

    def test_a_field_with_a_value_reports_it(self) -> None:
        controls = [
            {"i": 1, "role": "text field", "label": "Search", "onscreen": True, "field": True, "value": "jev"}
        ]
        text = _render_computer_state(a_desktop(controls=controls), "goal")
        self.assertIn("text field 'Search' = jev", text)

    def test_running_apps_are_listed_without_the_frontmost(self) -> None:
        text = _render_computer_state(a_desktop(), "goal")
        self.assertIn("running apps: Notes, ZCode", text)


class TestTheLoop(unittest.TestCase):
    def test_it_clicks_until_the_engine_says_stop(self) -> None:
        chooser = ScriptedChooser(["click the button 'New Tab'", STOP])
        session = FakeDesktop([a_desktop(), a_desktop(), a_desktop(focused="tab")])
        result = run_computer_task(
            "open a new tab", desktop_config(), actor=ScriptedJev(chooser=chooser),
            session=session, act=True,
        )
        self.assertEqual(result.stop_reason, "the engine judged the goal already achieved")
        self.assertEqual([a["kind"] for a in session.actions], ["click"])

    def test_two_steps_with_no_change_stop_the_loop(self) -> None:
        # Alternating actions, so the repeat guard stays out of the way and this test
        # exercises exactly the no-change guard.
        chooser = ScriptedChooser([
            "click the button 'New Tab'",
            "scroll down to see more of what is on screen",
            "click the button 'New Tab'",
        ])
        session = FakeDesktop([a_desktop()])  # acting never changes the state
        result = run_computer_task(
            "goal", desktop_config(), actor=ScriptedJev(chooser=chooser), session=session, act=True
        )
        self.assertEqual(result.stop_reason, "two steps with no change on the screen")

    def test_a_low_confidence_step_is_reported_and_stopped(self) -> None:
        chooser = ScriptedChooser(["click the button 'New Tab'"])
        session = FakeDesktop([a_desktop()])
        result = run_computer_task(
            "goal", desktop_config(), actor=ScriptedJev(chooser=chooser),
            session=session, act=True, min_confidence=0.99,
        )
        self.assertIn("not confident enough", result.stop_reason)
        self.assertEqual(session.actions, [])

    def test_an_off_menu_choice_is_refused_rather_than_guessed(self) -> None:
        class Rogue(ScriptedJev):
            def ask(self, state, questions):
                result = super().ask(state, questions)
                result.answers["action"]["choice"] = "delete the whole disk"
                return result

        session = FakeDesktop([a_desktop()])
        result = run_computer_task(
            "goal", desktop_config(), actor=Rogue(), session=session, act=True
        )
        self.assertIn("not on offer", result.stop_reason)
        self.assertEqual(session.actions, [])

    def test_a_failing_engine_stops_with_its_reason(self) -> None:
        from jev_cascade.jev import JevError

        class Broken(ScriptedJev):
            def ask(self, state, questions):
                raise JevError("the engine is down")

        session = FakeDesktop([a_desktop()])
        result = run_computer_task(
            "goal", desktop_config(), actor=Broken(), session=session, act=True
        )
        self.assertIn("the decision engine failed", result.stop_reason)

    def test_a_step_ceiling_can_be_lowered(self) -> None:
        chooser = ScriptedChooser([
            "click the button 'New Tab'",
            "scroll down to see more of what is on screen",
        ])
        # Every act lands somewhere new, so only the ceiling can end this run.
        session = FakeDesktop([a_desktop(window=f"Page {i}") for i in range(1, 10)])
        result = run_computer_task(
            "goal", desktop_config(), actor=ScriptedJev(chooser=chooser),
            session=session, act=True, max_steps=2,
        )
        self.assertIn("2-step ceiling", result.stop_reason)

    def test_an_identical_action_is_never_repeated(self) -> None:
        # Measured on Notes: the same phrase was typed five times into one note because
        # state reads lagged the app, so the no-change guard never fired. The repeat is
        # the thing to catch: identical action twice on a desktop is a stall that damages.
        chooser = ScriptedChooser(["click the button 'New Tab'"] * 5)
        session = FakeDesktop([a_desktop(window=f"Page {i}") for i in range(1, 10)])
        result = run_computer_task(
            "goal", desktop_config(), actor=ScriptedJev(chooser=chooser),
            session=session, act=True, max_steps=6,
        )
        self.assertIn("repeated the same action", result.stop_reason)
        self.assertEqual(len(session.actions), 1)

    def test_a_dry_run_reports_one_step_and_touches_nothing(self) -> None:
        chooser = ScriptedChooser(["click the button 'New Tab'"])
        session = FakeDesktop([a_desktop()])
        result = run_computer_task(
            "goal", desktop_config(), actor=ScriptedJev(chooser=chooser), session=session, act=False
        )
        self.assertIn("dry run", result.stop_reason)
        self.assertEqual(session.actions, [])

    def test_an_app_outside_the_allow_list_ends_the_run_before_anything_acts(self) -> None:
        chooser = ScriptedChooser(["click the button 'New Tab'"])
        session = FakeDesktop([a_desktop(app="Mail", bundle="com.apple.mail")])
        result = run_computer_task(
            "goal", desktop_config(), actor=ScriptedJev(chooser=chooser),
            session=session, act=True, allowed=frozenset({"safari"}),
        )
        self.assertIn("allowed_apps", result.stop_reason)
        self.assertEqual(session.actions, [])
        self.assertEqual(result.steps, [])

    def test_a_bundle_id_satisfies_the_allow_list(self) -> None:
        chooser = ScriptedChooser(["click the button 'New Tab'", STOP])
        session = FakeDesktop([a_desktop(), a_desktop(), a_desktop()])
        result = run_computer_task(
            "goal", desktop_config(), actor=ScriptedJev(chooser=chooser),
            session=session, act=True, allowed=frozenset({"com.apple.safari"}),
        )
        self.assertEqual(result.stop_reason, "the engine judged the goal already achieved")
        self.assertEqual(len(session.actions), 1)

    def test_a_step_records_what_was_on_offer(self) -> None:
        chooser = ScriptedChooser(["click the button 'New Tab'", STOP])
        session = FakeDesktop([a_desktop(), a_desktop(), a_desktop()])
        result = run_computer_task(
            "goal", desktop_config(), actor=ScriptedJev(chooser=chooser),
            session=session, act=True,
        )
        first = result.steps[0]
        self.assertEqual(first["options"], len(first["choices"]))
        self.assertIn("click the button 'New Tab'", first["choices"])

    def test_the_ledger_receives_start_step_and_stop_records(self) -> None:
        from jev_cascade.ledger import Ledger

        book = Ledger()
        chooser = ScriptedChooser(["click the button 'New Tab'", STOP])
        session = FakeDesktop([a_desktop(), a_desktop(), a_desktop()])
        run_computer_task(
            "goal", desktop_config(), actor=ScriptedJev(chooser=chooser),
            session=session, act=True, ledger=book,
        )
        kinds = [event["type"] for event in book.events]
        self.assertEqual(kinds[0], "computer_start")
        self.assertIn("computer_step", kinds)
        self.assertEqual(kinds[-1], "computer_stop")
        self.assertEqual([r.kind for r in book.records], ["computer"])
        book.close()

    def test_a_set_value_step_gets_its_words_from_the_writer(self) -> None:
        chooser = ScriptedChooser(["set the text field 'Search'", STOP])
        session = FakeDesktop([a_desktop(), a_desktop(), a_desktop()])
        config = desktop_config(writer_tier="cheap")
        result = run_computer_task(
            "search for jev", config, actor=ScriptedJev(chooser=chooser),
            session=session, act=True, writer_tier="cheap",
        )
        self.assertEqual(len(session.actions), 1)
        self.assertEqual(session.actions[0]["kind"], "set_value")
        self.assertTrue(session.actions[0].get("text"))

    def test_the_writer_spend_is_booked_in_the_ledger(self) -> None:
        from jev_cascade.ledger import Ledger

        book = Ledger()
        chooser = ScriptedChooser(["set the text field 'Search'", STOP])
        session = FakeDesktop([a_desktop(), a_desktop(), a_desktop()])
        config = desktop_config(writer_tier="cheap")
        run_computer_task(
            "search for jev", config, actor=ScriptedJev(chooser=chooser),
            session=session, act=True, writer_tier="cheap", ledger=book,
        )
        writes = [r for r in book.records if r.kind == "write"]
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0].tier, "cheap")
        self.assertEqual(writes[0].status, "done")
        self.assertIsNotNone(writes[0].verify_p)
        book.close()

    def test_a_failed_gate_stops_the_step_and_never_types(self) -> None:
        chooser = ScriptedChooser(["set the text field 'Search'"])
        session = FakeDesktop([a_desktop()])
        config = desktop_config(writer_tier="cheap")
        result = run_computer_task(
            "search for jev", config, actor=ScriptedJev(chooser=chooser, verdicts=[0.1, 0.2]),
            session=session, act=True, writer_tier="cheap",
        )
        self.assertIn("the writer could not produce a verified string", result.stop_reason)
        self.assertEqual(session.actions, [])

    def test_an_unknown_writer_tier_is_refused_before_the_loop_starts(self) -> None:
        with self.assertRaises(ValueError):
            run_computer_task(
                "goal", desktop_config(), actor=ScriptedJev(),
                session=FakeDesktop([a_desktop()]), act=False, writer_tier="nonexistent",
            )


class TestAgainstTheRealDesktop(unittest.TestCase):
    """Read-only: starts the real bridge, reads the screen, acts at nothing."""

    def test_it_reads_the_frontmost_app_and_reports_a_dry_run(self) -> None:
        if sys.platform != "darwin" or shutil.which("swift") is None:
            self.skipTest("the computer tool needs macOS and the Swift toolchain")
        from jev_cascade.computer import ComputerSession

        session = None
        try:
            try:
                session = ComputerSession(timeout_s=120.0)
                state = session.state()
            except Exception as exc:  # BridgeError: permission missing, or compile failed
                self.skipTest(str(exc)[:200])
            self.assertTrue(str(state.get("app", "")))
            self.assertIsInstance(state.get("controls", []), list)
            config = desktop_config()
            result = run_computer_task(
                "report what is on the screen", config,
                # A non-stop answer that exists in every state, so the run reaches the
                # dry-run branch and reports one step. Nothing is acted on. (The
                # frontmost app during a test run is often Terminal, which exposes no
                # clickable controls at all.)
                actor=ScriptedJev(chooser=ScriptedChooser(["wait because"])),
                session=session, act=False,
            )
            self.assertIn("dry run", result.stop_reason)
        finally:
            if session is not None:
                session.close()

    def test_check_permissions_answers_without_raising(self) -> None:
        if sys.platform != "darwin" or shutil.which("swift") is None:
            self.skipTest("the computer tool needs macOS and the Swift toolchain")
        verdict = check_permissions()
        self.assertIsInstance(verdict.get("trusted"), bool)
