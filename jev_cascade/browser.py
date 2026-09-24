"""Browser use: read the page, make one typed decision, take one action, repeat.

This is the cheap computer-use shape, in the cascade's own terms. The expensive way to
drive a browser is to ship a screenshot to a big model every step and ask it what to do.
The cheap way, and the one measured here, is:

1. **Perceive deterministically.** The page supplies its own text and its own controls,
   with roles and labels, which is what a browser can already tell you about itself. No
   vision model is involved, and nothing has to be guessed from pixels.
2. **Decide with a typed question.** One Choice over a fused, mutually exclusive option
   set, where every option names both the action and its target.

Two things in here come from measurement rather than taste (2026-09-21, see README):

* **Options name their target.** "click the link 'Open the report'" beats an abstract
  "click_item" followed by a separate question about which control, for both engines.
* **The action decision stays on the hosted Jev engine.** Laya, the local one, scored 1/6
  on situated action selection where Jev scored 5/6 on real pages, so ``actor`` defaults to
  Jev even when ``[jev] kind = "laya"``. Laya is for the classification-shaped calls.

Nothing here acts unless ``act`` is true. A dry run reports what it would do and stops.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .bridge import BridgeError, LineBridge
from .config import Config, JevConfig, TierConfig
from .jev import Jev, JevError, build_jev, choice_question
from .ledger import Ledger, StepRecord
from .providers import build_provider
from .writer import compose_for_field, resolve_writer_tier

BRIDGE = Path(__file__).resolve().parent.parent / "tools" / "browser_bridge.mjs"
STOP = "stop because the goal is already achieved"
WAIT = "wait because the page is still loading"
SCROLL = "scroll down to see more of the page"
SCROLL_UP = "scroll up to see what came before"
BACK = "go back to the previous page"
ESCAPE = "press escape to dismiss a dialog or overlay"


@dataclass
class BrowserResult:
    goal: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""
    acted: bool = False
    final_state: dict[str, Any] = field(default_factory=dict)
    ledger: Ledger | None = None
    stub: bool = False

    @property
    def jev_cost_usd(self) -> float:
        return self.ledger.jev_cost_usd if self.ledger else 0.0

    def render(self) -> str:
        lines = [f"goal: {self.goal}"]
        if not self.steps:
            lines.append("no steps were taken")
        for step in self.steps:
            mark = str(step.get("status") or ("acted" if step.get("acted") else "would"))
            detail = step.get("detail") or step.get("choice", "")
            lines.append(
                f"  {step['i']:>2}. {mark:<7} {detail[:84]:86} "
                f"conf={step.get('confidence', 0.0):.2f}"
            )
        lines.append("")
        lines.append(f"stopped: {self.stop_reason}")
        if not self.acted:
            lines.append(
                "NOTE: dry run. Nothing was clicked or typed. Pass --act to let it drive."
            )
        if self.stub:
            lines.append("NOTE: stub Jev was used. These decisions came from a script, not a model.")
        calls = sum(int(step.get("jev_calls") or 0) for step in self.steps)
        lines.append(f"decisions {calls}  engine {self.jev_cost_usd:.5f} USD")
        return "\n".join(lines)


class BrowserSession:
    """A persistent Playwright session behind a JSON line protocol.

    The bridge is a Node process rather than a Python import so that this package keeps its
    no-dependencies property, and it is held open across steps because relaunching the
    browser every step would lose form state.
    """

    def __init__(
        self,
        url: str = "about:blank",
        node: str = "node",
        bridge: Path | str = BRIDGE,
        headed: bool = False,
        timeout_s: float = 90.0,
        playwright_spec: str = "",
    ) -> None:
        self.bridge = Path(bridge)
        if not self.bridge.is_file():
            raise BridgeError(f"the browser bridge is missing: {self.bridge}")
        if shutil.which(node) is None:
            raise BridgeError(
                f"{node!r} is not on PATH; the browser tool needs Node and Playwright, "
                "and the rest of the cascade does not"
            )
        argv = [node, str(self.bridge), f"--url={url}"]
        if headed:
            argv.append("--headed")
        env_extra = {"JEV_BROWSER_PLAYWRIGHT": playwright_spec} if playwright_spec else None
        self.timeout_s = timeout_s
        self._link = LineBridge(argv, what="browser bridge", env_extra=env_extra)

    def state(self) -> dict[str, Any]:
        return self._link.exchange({"cmd": "state"})["state"]

    def act(self, action: dict[str, Any]) -> dict[str, Any]:
        reply = self._link.exchange({"cmd": "act", "action": action})
        return {"ok": bool(reply.get("ok")), "detail": str(reply.get("detail") or reply.get("error") or "")}

    def close(self) -> None:
        self._link.close()


TYPE_INTO = "type into the {role} '{label}'"
TEXT_FIELD_ROLES = frozenset(
    {"text", "search", "email", "url", "tel", "password", "number", "textarea", "input", "date"}
)


def is_text_field(control: dict[str, Any]) -> bool:
    """Whether a control can receive typed text.

    The bridge marks fields itself when it can (`field: true`); the role check
    is the fallback for states built by hand in tests.
    """

    if control.get("field") is True:
        return True
    if control.get("field") is False:
        return False
    return str(control.get("role", "")) in TEXT_FIELD_ROLES


def build_options(
    state: dict[str, Any],
    text: str = "",
    max_controls: int = 25,
    max_options: int = 80,
    writer: bool = False,
) -> dict[str, str]:
    """One mutually exclusive option per concrete action, target named in the label.

    Kept deliberately free of near-duplicates: the project this borrows from found that
    every stall traced back to two options that meant the same thing, because a
    calibrated model splits its probability between them and reads as doubt.
    """

    options: dict[str, str] = {}
    room = max(1, max_options - 7)  # scroll both ways, back, escape, wait, stop, and one spare
    for control in state.get("controls", [])[:max_controls]:
        label = str(control.get("label", "")).strip()
        if not label:
            continue
        role = str(control.get("role", "control"))
        where = "" if control.get("onscreen", True) else " (below the fold)"
        if writer and is_text_field(control):
            # A field with a writer behind it is offered once, as a field. Offering the
            # same control as a click and as a field is exactly the near-duplicate that
            # splits an engine's probability and reads as doubt.
            option = f"type into the {role} '{label}'{where}"
            options[option] = f"have the writer compose the text for this field: {label!r}{where}"
        else:
            option = f"click the {role} '{label}'{where}"
            options[option] = f"press this control: {role} {label!r}{where}"
        if len(options) >= room:
            break
    if text.strip():
        options["type the prepared text into the focused field"] = (
            "send the prepared text to the field that currently has focus"
        )
        options["press enter"] = "press the return key"
    elif writer:
        options["press enter"] = "press the return key"
    options[SCROLL] = "scroll down to reveal more of the page"
    options[SCROLL_UP] = "scroll up to reveal what scrolled past"
    options[BACK] = "return to the page this one came from"
    options[ESCAPE] = "close an overlay or dialog that is in the way"
    options[WAIT] = "the page has not finished loading"
    options[STOP] = "the goal is satisfied by what is on the page now"
    return options


def describe_action(action: dict[str, Any]) -> str:
    """A short human reading of an action, for the trace and the dry run."""

    kind = str(action.get("kind", ""))
    if kind == "click":
        return f"click the control numbered {action.get('index')}"
    if kind == "type":
        return f"type {len(str(action.get('text', '')))} characters into the focused field"
    if kind == "type_into":
        if action.get("text"):
            return f"type {len(str(action.get('text', '')))} characters into the field numbered {action.get('index')}"
        return f"compose text for the field numbered {action.get('index')}"
    if kind in ("scroll_down", "scroll_up", "press_enter", "press_escape", "back", "wait", "done"):
        return kind.replace("_", " ")
    return kind.replace("_", " ") if kind else "nothing"


def parse_action(choice: str, state: dict[str, Any]) -> dict[str, Any]:
    """Turn the chosen option back into an action the bridge understands."""

    if choice == STOP:
        return {"kind": "done"}
    if choice == WAIT:
        return {"kind": "wait"}
    if choice == SCROLL:
        return {"kind": "scroll_down"}
    if choice == SCROLL_UP:
        return {"kind": "scroll_up"}
    if choice == BACK:
        return {"kind": "back"}
    if choice == ESCAPE:
        return {"kind": "press_escape"}
    if choice == "press enter":
        return {"kind": "press_enter"}
    if choice == "type the prepared text into the focused field":
        return {"kind": "type"}
    if choice.startswith("type into the "):
        for control in state.get("controls", []):
            label = str(control.get("label", "")).strip()
            role = str(control.get("role", ""))
            if choice.startswith(f"type into the {role} '{label}'"):
                return {"kind": "type_into", "index": int(control["i"])}
    if choice.startswith("click the "):
        for control in state.get("controls", []):
            label = str(control.get("label", "")).strip()
            role = str(control.get("role", "control"))
            if choice.startswith(f"click the {role} '{label}'"):
                return {"kind": "click", "index": int(control["i"])}
    return {"kind": "unknown", "choice": choice}


def _render_state(state: dict[str, Any], goal: str, limit_lines: int = 16) -> str:
    """The state Jev reads: the page as text, plus what can be pressed on it."""

    lines = [f"task: {goal}", f"page: {state.get('url', '')}", f"title: {state.get('title', '')}"]
    if state.get("focused") and state["focused"] != "none":
        lines.append(f"focused field: {state['focused']}")
    body = state.get("lines") or []
    if body:
        lines.append("visible text:")
        lines.extend(f"  {line}" for line in body[:limit_lines])
    controls = state.get("controls") or []
    if controls:
        lines.append("controls on the page:")
        for control in controls[:25]:
            where = "" if control.get("onscreen", True) else " (below the fold)"
            lines.append(f"  {control['role']} '{control['label']}'{where}")
    return "\n".join(lines)


def run_browser_task(
    goal: str,
    config: Config,
    actor: Jev | None = None,
    session: BrowserSession | None = None,
    url: str = "about:blank",
    text: str = "",
    act: bool = False,
    max_steps: int = 12,
    min_confidence: float = 0.35,
    max_controls: int = 25,
    writer_tier: str = "",
    ledger: Ledger | None = None,
    on_event: Callable[[str], None] | None = None,
) -> BrowserResult:
    """Read, decide, act, until the decision engine says stop or a guard trips.

    ``actor`` is the decision engine. It defaults to the hosted Jev client on purpose:
    action selection is the one role the local engine was measured to be bad at.
    ``writer_tier`` names the tier that composes typed text; empty means the only words
    typed are the ones passed in as ``text``.
    """

    def emit(message: str) -> None:
        if on_event is not None:
            on_event(message)
        elif config.agent.verbose:
            print(message, flush=True)

    if actor is None:
        actor = build_jev(_actor_config(config))
    actor_name = getattr(actor, "model", None) or getattr(
        getattr(actor, "config", None), "model", "unknown"
    )
    writer: tuple[TierConfig, Any] | None = None
    if writer_tier:
        tier = resolve_writer_tier(writer_tier, config)
        writer = (tier, build_provider(tier))
    owns_ledger = ledger is None
    book = ledger or Ledger(path=config.agent.ledger_path)
    owns_session = session is None
    session = session or BrowserSession(
        url=url,
        node=config.browser.node,
        headed=config.browser.headed,
        playwright_spec=config.browser.playwright,
    )
    result = BrowserResult(goal=goal, acted=act, ledger=book, stub=bool(actor.stub))

    book.record({"type": "browser_start", "goal": goal, "url": url, "act": act, "actor": actor_name})

    consecutive_noops = 0
    previous_fingerprint = ""
    try:
        for index in range(1, max_steps + 1):
            state = session.state()
            # The focused field rides along in the fingerprint: its descriptor carries
            # the field's current value, so a successful type counts as change. Without
            # it, filling a search box reads as a no-op and the guard below ends the
            # run before the loop can press enter.
            fingerprint = json.dumps(
                [
                    state.get("url"),
                    state.get("title"),
                    state.get("focused"),
                    state.get("lines"),
                    state.get("controls"),
                ],
                sort_keys=True,
            )
            if fingerprint == previous_fingerprint:
                consecutive_noops += 1
            else:
                consecutive_noops = 0
            previous_fingerprint = fingerprint

            options = build_options(state, text=text, max_controls=max_controls, writer=bool(writer))
            prompt = _render_state(state, goal)
            step: dict[str, Any] = {"i": index, "url": state.get("url", ""), "options": len(options)}

            if consecutive_noops >= 2:
                result.stop_reason = "two steps with no change on the page"
                break

            try:
                decision = actor.ask(
                    prompt,
                    {
                        "action": choice_question(
                            options,
                            "Which single action moves toward the task, given what this page shows?",
                        )
                    },
                )
            except JevError as exc:
                result.stop_reason = f"the decision engine failed: {exc}"
                book.record({"type": "browser_error", "step": index, "error": str(exc)})
                break

            choice, confidence = decision.choice("action")
            step["choice"] = choice
            step["confidence"] = confidence
            step["jev_calls"] = decision.calls
            step["jev_cost_usd"] = decision.cost_usd
            step["jev_tokens_in"] = decision.tokens_in
            step["jev_tokens_out"] = decision.tokens_out
            emit(f"  step {index}: {choice[:80]} (conf={confidence:.2f})")

            if choice not in options:
                result.stop_reason = f"the engine chose {choice!r}, which was not on offer"
                step["status"] = "refused"
                step["detail"] = result.stop_reason
                result.steps.append(step)
                book.record({"type": "browser_step", **step, "acted": False})
                break

            if choice == STOP:
                result.stop_reason = "the engine judged the goal already achieved"
                step["status"] = "stopped"
                step["detail"] = STOP
                result.steps.append(step)
                book.record({"type": "browser_step", **step, "acted": False})
                break

            if confidence < min_confidence:
                result.stop_reason = (
                    f"the engine was not confident enough ({confidence:.2f} < {min_confidence:.2f})"
                )
                step["status"] = "stopped"
                step["detail"] = result.stop_reason
                result.steps.append(step)
                book.record({"type": "browser_step", **step, "acted": False})
                break

            action = parse_action(choice, state)
            if action["kind"] == "unknown":
                result.stop_reason = f"could not turn {choice!r} into an action"
                step["status"] = "refused"
                step["detail"] = result.stop_reason
                result.steps.append(step)
                book.record({"type": "browser_step", **step, "acted": False})
                break

            if action["kind"] == "type":
                action["text"] = text

            if action["kind"] == "type_into" and act:
                # The words come from the writer tier, gated by Jev, before anything
                # is typed. A draft that fails the gate stops the step: never type an
                # unverified string into a real page.
                control = next(
                    (c for c in state.get("controls", []) if int(c.get("i", -1)) == action["index"]),
                    {},
                )
                draft = compose_for_field(
                    goal,
                    state,
                    control,
                    writer,
                    actor,
                    book,
                    index,
                    tool="browser",
                    emit=emit,
                )
                if not draft.ok:
                    result.stop_reason = (
                        f"the writer could not produce a verified string for the field "
                        f"{str(control.get('label', ''))!r}: {draft.error}"
                    )
                    step["status"] = "refused"
                    step["detail"] = result.stop_reason
                    result.steps.append(step)
                    book.record({"type": "browser_step", **step, "acted": False})
                    break
                action["text"] = draft.text

            if not act:
                step["detail"] = describe_action(action)
                step["status"] = "would"
                step["acted"] = False
                result.steps.append(step)
                book.record({"type": "browser_step", **step, "acted": False})
                result.stop_reason = "dry run: one step reported, nothing was done"
                break

            outcome = session.act(action)
            step["action"] = action
            step["acted"] = outcome["ok"]
            step["status"] = "acted" if outcome["ok"] else "refused"
            step["detail"] = f"{describe_action(action)}: {outcome['detail']}"
            result.steps.append(step)
            book.record({"type": "browser_step", **step, "action": action})
            book.add_step(
                StepRecord(
                    step_id=f"browser-{index}",
                    goal=goal,
                    kind="browser",
                    tier="jev",
                    model=actor_name,
                    status=str(step.get("status") or "acted"),
                    output_chars=len(json.dumps(action)),
                    latency_s=float(decision.latency_s or 0.0),
                    stub=bool(actor.stub),
                    route_confidence=confidence,
                    jev_calls=int(decision.calls or 0),
                    jev_cost_usd=float(decision.cost_usd or 0.0),
                    jev_tokens_in=int(decision.tokens_in or 0),
                    jev_tokens_out=int(decision.tokens_out or 0),
                )
            )
            emit(f"    {outcome['detail'][:80]}")

            if not outcome["ok"]:
                emit(f"    the page refused it: {outcome['detail'][:70]}")
            else:
                time.sleep(0.2)
        else:
            result.stop_reason = f"hit the {max_steps}-step ceiling"

        if not result.stop_reason:
            result.stop_reason = "the loop ended without a reason, which is a bug"
        # Exactly one terminal record, so the JSONL trace always ends with a reason.
        book.record(
            {
                "type": "browser_stop",
                "reason": result.stop_reason,
                "steps": len(result.steps),
                "acted": result.acted,
            }
        )
        result.final_state = session.state()
    finally:
        if owns_session:
            session.close()
        if owns_ledger:
            book.close()
    return result


def _actor_config(config: Config) -> JevConfig:
    """The engine used for action selection, which is the hosted one by default.

    Rationale, measured: Laya scored 1/6 on choosing what to press where Jev scored 5/6,
    so a config that put Laya on the classification roles does not put it here. Set
    ``[browser] actor = "laya"`` to overrule that, knowing the number.
    """

    if config.browser.actor == "laya":
        from dataclasses import replace  # noqa: PLC0415

        return replace(config.jev, kind="laya")
    return config.jev
