"""Computer use: read the desktop's own controls, decide once, act once, repeat.

The same shape as the browser tool, pointed at the screen instead of a page. The
expensive way to drive a desktop is to ship screenshots to a big model every step.
The cheap way, and the one built here:

1. **Perceive deterministically.** The frontmost app reports its own controls,
   with roles and labels, through the macOS Accessibility tree — collected by a
   Swift bridge so this package keeps its zero dependencies. No vision is
   involved, no pixels are read, and Screen Recording is never requested.
2. **Decide with one typed question.** One Choice over a fused, mutually
   exclusive option set, every option naming both the action and its target.
3. **Act through the same tree.** AXPress where the control offers it, a real
   mouse event at the control's own position when it does not, and a direct
   AXValue write for text fields before any keystroke injection is considered.

Two guards a desktop needs that a page does not: nothing acts unless ``act`` is
true (a dry run reads and reports), and ``[computer] allowed_apps`` can name the
apps the loop may act on, so a confused engine cannot reach past the one app it
was aimed at.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .bridge import BridgeError, LineBridge
from .browser import STOP, BrowserResult
from .config import ComputerConfig, Config, JevConfig, TierConfig
from .jev import Jev, JevError, build_jev, choice_question
from .ledger import Ledger, StepRecord
from .providers import build_provider
from .writer import compose_for_field, resolve_writer_tier

SWIFT_BRIDGE = Path(__file__).resolve().parent.parent / "tools" / "macos_bridge.swift"
ENTER = "press enter"
ESCAPE = "press escape"
# Deliberately not the browser's strings: nothing here is a page.
SCROLL_DOWN = "scroll down to see more of what is on screen"
SCROLL_UP = "scroll up to see what scrolled past"
WAIT = "wait because the app is still working"
COMPUTER_FIELD_ROLES = frozenset({"text field", "search field", "text area", "combo box"})


class ComputerResult(BrowserResult):
    """The same shape as a browser run: steps, a stop reason, and what it cost."""


def _is_field(control: dict[str, Any]) -> bool:
    if control.get("field") is True:
        return True
    if control.get("field") is False:
        return False
    return str(control.get("role", "")) in COMPUTER_FIELD_ROLES


class ComputerSession:
    """The desktop's Accessibility tree behind the same JSON line protocol.

    The bridge is a Swift file run by the ``swift`` interpreter, compiled once to
    a cached binary next to it when possible, because interpreter startup is a
    second or two per run and the compile is paid once. Like the browser bridge,
    it is one long-lived process: the desktop is the state that must survive
    between steps.
    """

    def __init__(
        self,
        computer: ComputerConfig | None = None,
        app: str = "",
        timeout_s: float = 90.0,
        check_only: bool = False,
    ) -> None:
        if sys.platform != "darwin":
            raise BridgeError(
                "the computer tool drives macOS through its Accessibility tree; "
                "there is nothing for it to read on this platform"
            )
        cfg = computer or ComputerConfig()
        source = Path(cfg.bridge).expanduser() if cfg.bridge else SWIFT_BRIDGE
        if not source.is_file():
            raise BridgeError(f"the computer bridge is missing: {source}")
        if shutil.which(cfg.swift) is None:
            raise BridgeError(
                f"{cfg.swift!r} is not on PATH; the computer tool needs the Swift "
                "toolchain (Xcode Command Line Tools), and the rest of the cascade does not"
            )
        argv = self._launch_argv(cfg, source)
        if app:
            argv.append(f"--app={app}")
        if check_only:
            argv.append("--check-only")
        env_extra = {
            "JEV_COMPUTER_MAX_CONTROLS": str(cfg.max_controls),
            "JEV_COMPUTER_MAX_ELEMENTS": str(cfg.max_elements),
        }
        self.timeout_s = timeout_s
        self._link = LineBridge(argv, what="computer bridge", env_extra=env_extra)
        if app:
            self._bring_to_front(app)

    def _bring_to_front(self, app: str, attempts: int = 3) -> None:
        """Verify the named app really is frontmost before the loop starts.

        Activation requests are asynchronous and silently denied while the user is
        working in another app, and an engine that reads the wrong app's tree decides
        the wrong app's actions. So this verifies rather than hopes: activate via
        osascript, re-read the state, retry, and fail fast with what is frontmost.
        """

        import subprocess as sp  # noqa: PLC0415

        for _ in range(attempts):
            sp.run(
                ["osascript", "-e", f'tell application "{app}" to activate'],
                capture_output=True,
                timeout=15,
            )
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if str(self.state().get("app", "")).lower() == app.lower():
                    return
                time.sleep(0.4)
        front = str(self.state().get("app", "unknown"))
        raise BridgeError(
            f"could not bring {app!r} to the front (frontmost is {front!r}); "
            "close what is in the way and try again"
        )

    @staticmethod
    def _launch_argv(cfg: ComputerConfig, source: Path) -> list[str]:
        """A cached compiled binary when one can be built, else the interpreter."""

        binary = source.parent / f".{source.stem}"
        stale = not binary.is_file() or binary.stat().st_mtime < source.stat().st_mtime
        if stale and shutil.which("swiftc") is not None:
            built = subprocess.run(
                ["swiftc", "-O", str(source), "-o", str(binary)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if built.returncode != 0:
                # A compile failure is not fatal: the interpreter can still run the
                # same file, slower, and the error says which path was taken.
                argv = [cfg.swift, str(source)]
            else:
                argv = [str(binary)]
        elif not stale:
            argv = [str(binary)]
        else:
            argv = [cfg.swift, str(source)]
        return argv

    def state(self) -> dict[str, Any]:
        return self._link.exchange({"cmd": "state"})["state"]

    def act(self, action: dict[str, Any]) -> dict[str, Any]:
        reply = self._link.exchange({"cmd": "act", "action": action})
        return {"ok": bool(reply.get("ok")), "detail": str(reply.get("detail") or reply.get("error") or "")}

    def close(self) -> None:
        self._link.close()


def check_permissions(computer: ComputerConfig | None = None) -> dict[str, Any]:
    """Ask macOS about the Accessibility grant, showing its dialog if missing.

    Runs the bridge in check-only mode, where the startup guard is skipped so the
    ``check`` command itself can trigger the system prompt. Returns what the
    system said rather than raising, because the whole point is to print advice.
    """

    probe = ComputerSession(computer, check_only=True)
    try:
        reply = probe._link.exchange({"cmd": "check"})
        return {"trusted": bool(reply.get("trusted")), "hint": str(reply.get("hint") or "")}
    except BridgeError as exc:
        return {"trusted": False, "hint": str(exc)}
    finally:
        probe.close()


def allowed_apps(config: ComputerConfig) -> frozenset[str] | None:
    """The app-name allow list as a case-folded set, or None for "any app"."""

    if not config.allowed_apps.strip():
        return None
    return frozenset(part.strip().lower() for part in config.allowed_apps.split(",") if part.strip())


def build_computer_options(
    state: dict[str, Any],
    text: str = "",
    max_controls: int = 25,
    max_options: int = 80,
    writer: bool = False,
    pinned_app: bool = False,
) -> dict[str, str]:
    """One mutually exclusive option per concrete action, target named in the label.

    ``pinned_app`` is a run started with an explicit ``--app``: the caller said where
    the work happens, so app-switch options are not doors this loop should offer —
    and a goal that says "note" next to a list of running apps reads as an invitation
    to wander.
    """

    options: dict[str, str] = {}
    room = max(1, max_options - 8)  # scroll both ways, enter, escape, wait, stop, and two spares
    for control in state.get("controls", [])[:max_controls]:
        label = str(control.get("label", "")).strip()
        if not label:
            continue
        role = str(control.get("role", "control"))
        where = "" if control.get("onscreen", True) else " (off-screen)"
        if _is_field(control) and writer:
            # A field with a writer behind it is offered once, as a field: the words will
            # be composed and gated. Without a writer there are no words to set, so the
            # field is offered as a plain click (to focus it) like any other control —
            # the loop never reaches for text it has no honest source for.
            option = f"set the {role} '{label}'{where}"
            options[option] = "replace this field's contents with text written for the goal"
        else:
            option = f"click the {role} '{label}'{where}"
            options[option] = f"press this control: {role} {label!r}{where}"
        if len(options) >= room:
            break
    if not pinned_app:
        for name in state.get("apps", [])[:6]:
            if name and name != state.get("app"):
                options[f"switch to the app '{name}'"] = "bring this running app to the front"
    focused = str(state.get("focused", "none")).lower()
    texty_focus = any(role in focused for role in ("text field", "text area", "search field", "combo box"))
    if text.strip() and texty_focus:
        # Keystrokes without a textable focus go straight into whatever the app has
        # under the cursor — a list, a table — so the option only exists when the state
        # says a field is focused. The bridge refuses as a second line of defence.
        options["type the prepared text into the focused field"] = (
            "send the prepared text to the field that currently has focus"
        )
        options[ENTER] = "press the return key"
    elif writer:
        options[ENTER] = "press the return key"
    options[SCROLL_DOWN] = "scroll down to reveal more of what is on screen"
    options[SCROLL_UP] = "scroll up to reveal what scrolled past"
    options[ESCAPE] = "close a dialog or menu that is in the way"
    options[WAIT] = "the app has not finished what it is doing"
    options[STOP] = "the goal is satisfied by what is on the screen now"
    return options


def describe_computer_action(action: dict[str, Any]) -> str:
    """A short human reading of an action, for the trace and the dry run."""

    kind = str(action.get("kind", ""))
    if kind == "click":
        return f"click the control numbered {action.get('index')}"
    if kind == "set_value":
        if action.get("text"):
            return f"set {len(str(action.get('text', '')))} characters into the field numbered {action.get('index')}"
        return f"compose text for the field numbered {action.get('index')}"
    if kind == "type":
        return f"type {len(str(action.get('text', '')))} characters into the focused field"
    if kind == "switch":
        return f"switch to {action.get('app')}"
    if kind in ("press_enter", "press_escape", "scroll_down", "scroll_up", "wait", "done"):
        return kind.replace("_", " ")
    return kind.replace("_", " ") if kind else "nothing"


def parse_computer_action(choice: str, state: dict[str, Any]) -> dict[str, Any]:
    """Turn the chosen option back into an action the bridge understands."""

    if choice == STOP:
        return {"kind": "done"}
    if choice == WAIT:
        return {"kind": "wait"}
    if choice == SCROLL_DOWN:
        return {"kind": "scroll", "direction": "down"}
    if choice == SCROLL_UP:
        return {"kind": "scroll", "direction": "up"}
    if choice == ENTER:
        return {"kind": "press_enter"}
    if choice == ESCAPE:
        return {"kind": "press_escape"}
    if choice == "type the prepared text into the focused field":
        return {"kind": "type"}
    if choice.startswith("switch to the app '"):
        name = choice[len("switch to the app '") :].rstrip("'")
        if name:
            return {"kind": "switch", "app": name}
    if choice.startswith("set the "):
        for control in state.get("controls", []):
            label = str(control.get("label", "")).strip()
            role = str(control.get("role", ""))
            if choice.startswith(f"set the {role} '{label}'"):
                return {"kind": "set_value", "index": int(control["i"])}
    if choice.startswith("click the "):
        for control in state.get("controls", []):
            label = str(control.get("label", "")).strip()
            role = str(control.get("role", "control"))
            if choice.startswith(f"click the {role} '{label}'"):
                return {"kind": "click", "index": int(control["i"])}
    return {"kind": "unknown", "choice": choice}


def _render_computer_state(state: dict[str, Any], goal: str, limit_lines: int = 14) -> str:
    """The state Jev reads: the app, its window, its text, and what can be pressed."""

    bundle = state.get("bundle") or ""
    app_line = f"app: {state.get('app', 'unknown')}" + (f" ({bundle})" if bundle else "")
    lines = [
        f"task: {goal}",
        app_line,
        f"window: {state.get('window', '')}",
    ]
    if state.get("focused") and state["focused"] != "none":
        lines.append(f"focused: {state['focused']}")
    body = state.get("lines") or []
    if body:
        lines.append("visible text:")
        lines.extend(f"  {line}" for line in body[:limit_lines])
    controls = state.get("controls") or []
    if controls:
        lines.append("controls on the desktop:")
        for control in controls[:25]:
            where = "" if control.get("onscreen", True) else " (off-screen)"
            value = f" = {control['value']}" if control.get("value") else ""
            lines.append(f"  {control['role']} '{control['label']}'{where}{value}")
    apps = [name for name in (state.get("apps") or []) if name and name != state.get("app")]
    if apps:
        lines.append(f"running apps: {', '.join(apps[:8])}")
    return "\n".join(lines)


def _actor_config(config: Config) -> JevConfig:
    """The engine used for action selection, which is the hosted one by default.

    Same measured rule as the browser: Laya scored 1/6 on choosing what to press
    where Jev scored 5/6. Set ``[computer] actor = "laya"`` to overrule that,
    knowing the number.
    """

    if config.computer.actor == "laya":
        from dataclasses import replace  # noqa: PLC0415

        return replace(config.jev, kind="laya")
    return config.jev


def run_computer_task(
    goal: str,
    config: Config,
    actor: Jev | None = None,
    session: ComputerSession | None = None,
    text: str = "",
    act: bool = False,
    max_steps: int = 10,
    min_confidence: float = 0.35,
    max_controls: int = 25,
    writer_tier: str = "",
    app: str = "",
    allowed: frozenset[str] | None = None,
    ledger: Ledger | None = None,
    on_event: Callable[[str], None] | None = None,
) -> ComputerResult:
    """Read, decide, act, until the decision engine says stop or a guard trips.

    The same ladder of guards as the browser: two no-ops, the confidence floor,
    the step ceiling, the off-menu refusal. ``allowed`` is the desktop's extra
    guard: when it names apps, a step ends rather than act on another one.
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
    session = session or ComputerSession(config.computer)
    result = ComputerResult(goal=goal, acted=act, ledger=book, stub=bool(actor.stub))

    book.record(
        {
            "type": "computer_start",
            "goal": goal,
            "act": act,
            "actor": actor_name,
            "allowed_apps": sorted(allowed) if allowed else [],
        }
    )

    consecutive_noops = 0
    previous_fingerprint = ""
    last_action_json = ""
    try:
        for index in range(1, max_steps + 1):
            state = session.state()
            front = str(state.get("app", "unknown"))
            bundle = str(state.get("bundle", ""))
            fingerprint = json.dumps(
                [
                    front,
                    state.get("window"),
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

            step: dict[str, Any] = {"i": index, "app": front}

            if allowed and front.lower() not in allowed and bundle.lower() not in allowed:
                result.stop_reason = (
                    f"the frontmost app {front!r} is not on [computer] allowed_apps; "
                    "the loop stops rather than act there"
                )
                break

            if consecutive_noops >= 2:
                result.stop_reason = "two steps with no change on the screen"
                break

            options = build_computer_options(
                state, text=text, max_controls=max_controls, writer=bool(writer),
                pinned_app=bool(app),
            )
            prompt = _render_computer_state(state, goal)

            try:
                decision = actor.ask(
                    prompt,
                    {
                        "action": choice_question(
                            options,
                            "Which single action moves toward the task, given what is on the screen?",
                        )
                    },
                )
            except JevError as exc:
                result.stop_reason = f"the decision engine failed: {exc}"
                book.record({"type": "computer_error", "step": index, "error": str(exc)})
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
                book.record({"type": "computer_step", **step, "acted": False})
                break

            if choice == STOP:
                result.stop_reason = "the engine judged the goal already achieved"
                step["status"] = "stopped"
                step["detail"] = STOP
                result.steps.append(step)
                book.record({"type": "computer_step", **step, "acted": False})
                break

            if confidence < min_confidence:
                result.stop_reason = (
                    f"the engine was not confident enough ({confidence:.2f} < {min_confidence:.2f})"
                )
                step["status"] = "stopped"
                step["detail"] = result.stop_reason
                result.steps.append(step)
                book.record({"type": "computer_step", **step, "acted": False})
                break

            action = parse_computer_action(choice, state)
            if action["kind"] == "unknown":
                result.stop_reason = f"could not turn {choice!r} into an action"
                step["status"] = "refused"
                step["detail"] = result.stop_reason
                result.steps.append(step)
                book.record({"type": "computer_step", **step, "acted": False})
                break

            if action["kind"] == "type":
                action["text"] = text

            if action["kind"] == "set_value" and act:
                # The words come from the writer tier, gated by Jev, before anything
                # is set. A draft that fails the gate stops the step: never write an
                # unverified string into a real app.
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
                    tool="computer",
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
                    book.record({"type": "computer_step", **step, "acted": False})
                    break
                action["text"] = draft.text

            if not act:
                step["detail"] = describe_computer_action(action)
                step["status"] = "would"
                step["acted"] = False
                result.steps.append(step)
                book.record({"type": "computer_step", **step, "acted": False})
                result.stop_reason = "dry run: one step reported, nothing was done"
                break

            # Keystrokes are not idempotent and state reads can lag the app, so the
            # no-change guard alone cannot catch a repeat. Measured on Notes (2026-09-24):
            # the same phrase was typed five times into one note because every state read
            # looked alike. An identical action twice is a stall, and on a desktop a
            # stall damages.
            action_json = json.dumps(action, sort_keys=True)
            if action_json == last_action_json:
                result.stop_reason = (
                    "the engine repeated the same action; stopping rather than do it twice"
                )
                step["status"] = "refused"
                step["detail"] = result.stop_reason
                result.steps.append(step)
                book.record({"type": "computer_step", **step, "acted": False})
                break

            outcome = session.act(action)
            last_action_json = action_json
            step["action"] = action
            step["acted"] = outcome["ok"]
            step["status"] = "acted" if outcome["ok"] else "refused"
            step["detail"] = f"{describe_computer_action(action)}: {outcome['detail']}"
            result.steps.append(step)
            book.record({"type": "computer_step", **step, "action": action})
            book.add_step(
                StepRecord(
                    step_id=f"computer-{index}",
                    goal=goal,
                    kind="computer",
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
                emit(f"    the app refused it: {outcome['detail'][:70]}")
            else:
                time.sleep(0.2)
        else:
            result.stop_reason = f"hit the {max_steps}-step ceiling"

        if not result.stop_reason:
            result.stop_reason = "the loop ended without a reason, which is a bug"
        # Exactly one terminal record, so the JSONL trace always ends with a reason.
        book.record(
            {
                "type": "computer_stop",
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
