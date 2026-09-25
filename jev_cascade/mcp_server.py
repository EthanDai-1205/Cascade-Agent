"""MCP server: the cascade as tools for any agent client.

Speaks the Model Context Protocol's stdio transport by hand — JSON-RPC 2.0 over
line-delimited JSON on stdin and stdout — so the project keeps its zero-dependencies
property. This is the house pattern for a third time: the browser lives behind a Node
process that speaks JSON lines, the desktop behind a Swift one, and now an agent client
behind this one. The protocol layer is one pure function over a small state object,
which makes every handshake testable in-process with no subprocess and no client.

Only the surface an agent needs is implemented, and deliberately no more:

    initialize                  -> protocol version + capabilities
    notifications/initialized   -> silence (notifications never get replies)
    ping                        -> {}
    tools/list                  -> the tool schemas
    tools/call                  -> the tool, its result, or an honest error

Tool execution failures come back as results with ``isError: true`` — a broken config
is a fact about the world the agent can read and react to, not a dead server. Protocol
violations (unknown method, unknown tool, malformed JSON) are JSON-RPC errors. If the
spec ever drifts far enough to hurt, this module is the single file to swap for the
official SDK.

The running loop is :func:`serve_stdio`; the tools it serves are built in
:func:`build_tools` and wrap the same functions the CLI calls — no logic duplicated.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

# Protocol versions this server speaks, oldest first. A client's requested version is
# honoured when known and answered with our latest when not, which is the whole of
# version negotiation.
SUPPORTED_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")
LATEST_VERSION = SUPPORTED_VERSIONS[-1]

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602


def _reply(message_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}


def _text_result(text: str, structured: dict[str, Any] | None = None) -> dict[str, Any]:
    """A successful tool result: the human trace, plus the machine summary beside it."""

    result: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
    if structured is not None:
        result["structuredContent"] = structured
    return result


def _error_result(text: str) -> dict[str, Any]:
    """A tool that ran and failed: the fact travels to the agent, the server stays up."""

    return {"content": [{"type": "text", "text": text}], "isError": True}


@dataclass
class Tool:
    """One agent-callable tool: a schema for the client, a handler for us.

    ``handler`` takes the call's arguments dict and returns a result dict shaped by
    :func:`_text_result` / :func:`_error_result`. Any exception the handler raises is
    reported as an ``isError`` result — the agent reads what went wrong and decides
    what to do, and the server never dies mid-session on one bad call.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]

    def schema_entry(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {"type": "object", **self.input_schema},
        }

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.handler(arguments)
        except Exception as exc:  # noqa: BLE001 - a tool failure is a result, not a crash
            return _error_result(f"{type(exc).__name__}: {exc}")


class McpServer:
    """Protocol state (which client, which version) plus the tool registry."""

    def __init__(self, tools: list[Tool] | None = None, version: str = "0.1.0") -> None:
        self.tools: dict[str, Tool] = {tool.name: tool for tool in (tools or [])}
        self.version = version
        self.client_version: str | None = None

    def handle_message(self, message: Any) -> dict[str, Any] | None:
        """One decoded JSON message in, one reply out — or None for notifications.

        The whole protocol is this function, which is why it takes decoded input and
        returns a dict: tests drive it without a pipe, and the running loop below is
        only JSON decoding and line discipline around it.
        """

        if not isinstance(message, dict):
            return _error(None, INVALID_REQUEST, "a message must be a JSON object")
        message_id = message.get("id")
        method = message.get("method")
        is_notification = "id" not in message
        if not isinstance(method, str) or not method:
            if is_notification:
                return None
            return _error(message_id, INVALID_REQUEST, "a request needs a method")

        if method == "initialize":
            return self._initialize(message_id, message.get("params") or {})
        if method == "ping":
            return _reply(message_id, {})
        if method == "tools/list":
            return _reply(message_id, {"tools": [tool.schema_entry() for tool in self.tools.values()]})
        if method == "tools/call":
            return self._call_tool(message_id, message.get("params") or {})

        # notifications/initialized lands here, as does anything we never learned.
        if is_notification:
            return None
        return _error(message_id, METHOD_NOT_FOUND, f"method not found: {method}")

    def _initialize(self, message_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        requested = str(params.get("protocolVersion") or "")
        self.client_version = requested if requested in SUPPORTED_VERSIONS else LATEST_VERSION
        return _reply(
            message_id,
            {
                "protocolVersion": self.client_version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "jev-cascade", "version": self.version},
            },
        )

    def _call_tool(self, message_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        tool = self.tools.get(str(name or ""))
        if tool is None:
            return _error(message_id, INVALID_PARAMS, f"unknown tool: {name!r}")
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return _error(message_id, INVALID_PARAMS, f"arguments for {name!r} must be an object")
        return _reply(message_id, tool.run(arguments))


def serve_stdio(
    server: McpServer,
    stdin: Any = None,
    stdout: Any = None,
) -> None:
    """The running loop: one JSON message per line in, one reply per request out.

    Runs until stdin closes, which is how an MCP client manages the server's life.
    Malformed JSON gets a parse-error reply with a null id and the loop carries on —
    one bad line never ends a session.
    """

    source = stdin if stdin is not None else sys.stdin
    sink = stdout if stdout is not None else sys.stdout
    for line in source:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            reply = _error(None, PARSE_ERROR, f"could not parse message as JSON: {exc.msg}")
        else:
            reply = server.handle_message(message)
        if reply is not None:
            sink.write(json.dumps(reply) + "\n")
            sink.flush()


# ---------------------------------------------------------------------------
# The tool layer: thin wrappers over the same functions the CLI calls. Handlers
# import their machinery lazily so a server that only ever answers tools/list
# never pays for the agent stack, and config is loaded per call so a fix to
# config.toml takes effect without restarting the server.

SCHEMA_STRING = {"type": "string"}
SCHEMA_BOOL = {"type": "boolean"}
SCHEMA_INT = {"type": "integer", "minimum": 1}


def _load_config():
    from .config import load_config

    return load_config()  # find_config: $JEVCASCADE_CONFIG, then ./config.toml, then example


def _need_actor(config):
    """The decision engine for a loop that will act, or the honest reason there isn't one."""

    from .jev import build_jev

    if config.jev.kind == "http" and not config.jev.api_key():
        raise RuntimeError(
            f"the cascade needs a decision engine: set ${config.jev.api_key_env} "
            "for the hosted Jev, or set [jev] kind = \"laya\" for the local one"
        )
    return build_jev(config.jev)


def _actor_for(config, act: bool):
    """Dry runs degrade to the stub engine without a key (flagged in the trace);
    anything that acts needs a real one, or the honest refusal above."""

    from .jev import build_jev

    if not act and config.jev.kind == "http" and not config.jev.api_key():
        return build_jev(config.jev, dry_run=True)
    return _need_actor(config)


def _tool_run(args: dict) -> dict:
    from .agent import CascadeAgent

    config = _load_config()
    dry_run = bool(args.get("dry_run", False))
    agent = CascadeAgent(config, dry_run=dry_run, on_event=lambda message: None)
    try:
        result = agent.run(str(args.get("task", "")).strip() or "-")
        outputs = result.outputs or {}
        ledger = result.ledger
        parts = []
        for step_id, text in outputs.items():
            parts.append(f"[{step_id}]\n{text.rstrip()}")
        if ledger is not None:
            parts.append(ledger.render_summary(config, stub=agent.stubbed))
        if result.error:
            parts.append(f"error: {result.error}")
        structured = {
            "task": args.get("task", ""),
            "outputs": outputs,
            "dry_run": dry_run,
            "steps": len(ledger.records) if ledger else 0,
            "total_cost_usd": round(ledger.total_cost_usd, 6) if ledger else 0.0,
            "error": result.error or "",
            "stub": agent.stubbed,
            "ledger_path": config.agent.ledger_path,
        }
        return _text_result("\n\n".join(parts) or "the run produced no outputs", structured)
    finally:
        agent.ledger.close()


def _tool_plan(args: dict) -> dict:
    import json as _json

    from .config import load_config
    from .planner import build_planner, plan_to_json
    from .providers import build_provider

    config = _load_config()
    providers = {tier.name: build_provider(tier) for tier in config.tiers}
    planner = build_planner(config, providers)
    plan = planner.plan(str(args.get("task", "")), dry_run=False)
    structured = _json.loads(plan_to_json(plan))  # plan_to_json returns JSON text
    return _text_result(_json.dumps(structured, indent=2), structured)


def _dry_run_note(act: bool) -> str:
    return "" if act else (
        " NOTE: this was a dry run - nothing was clicked or typed. "
        "Ask the user first, then call again with act=true to act."
    )


def _tool_browse(args: dict) -> dict:
    import dataclasses

    from .browser import BridgeError, run_browser_task
    from .cli import browser_result_to_json

    config = _load_config()
    browser = config.browser
    if args.get("steps") is not None:
        browser = dataclasses.replace(browser, max_steps=max(1, int(args["steps"])))
    if args.get("writer_tier"):
        browser = dataclasses.replace(browser, writer_tier=str(args["writer_tier"]))
    config = dataclasses.replace(config, browser=browser)

    act = bool(args.get("act", False))
    try:
        actor = _need_actor(config) if act else _actor_for(config, act=False)
        result = run_browser_task(
            str(args.get("goal", "")),
            config,
            actor=actor,
            url=str(args.get("url", "") or "about:blank"),
            text=str(args.get("text", "")),
            act=act,
            max_steps=browser.max_steps,
            min_confidence=browser.min_confidence,
            max_controls=browser.max_controls,
            writer_tier=browser.writer_tier,
            on_event=lambda message: None,
        )
    except BridgeError as exc:
        return _error_result(f"browser error: {exc}")
    return _text_result(
        result.render() + _dry_run_note(act), browser_result_to_json(result)
    )


def _tool_computer(args: dict) -> dict:
    import dataclasses

    from .cli import computer_result_to_json
    from .computer import BridgeError, allowed_apps, run_computer_task

    config = _load_config()
    cfg = config.computer
    if args.get("steps") is not None:
        cfg = dataclasses.replace(cfg, max_steps=max(1, int(args["steps"])))
    if args.get("writer_tier"):
        cfg = dataclasses.replace(cfg, writer_tier=str(args["writer_tier"]))
    config = dataclasses.replace(config, computer=cfg)

    act = bool(args.get("act", False))
    try:
        actor = _need_actor(config) if act else _actor_for(config, act=False)
        result = run_computer_task(
            str(args.get("goal", "")),
            config,
            actor=actor,
            text=str(args.get("text", "")),
            act=act,
            max_steps=cfg.max_steps,
            min_confidence=cfg.min_confidence,
            max_controls=cfg.max_controls,
            writer_tier=cfg.writer_tier,
            app=str(args.get("app", "")),
            allowed=allowed_apps(cfg),
            on_event=lambda message: None,
        )
    except BridgeError as exc:
        return _error_result(f"computer error: {exc}")
    return _text_result(
        result.render() + _dry_run_note(act), computer_result_to_json(result)
    )


def _tool_check(args: dict) -> dict:
    import contextlib
    import io

    from .cli import _cmd_check

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        _cmd_check(_load_config())
    return _text_result(buffer.getvalue(), {"ok": True})


def build_tools() -> list[Tool]:
    """The five agent-facing tools. Offline measurement (eval, selftest, demo)
    stays in the CLI on purpose: an agent has no use for them."""

    act_note = (
        " DEFAULTS TO A DRY RUN: one step is reported and nothing is touched. "
        "Ask the user first, then call again with act=true to act."
    )
    return [
        Tool(
            name="cascade_run",
            description=(
                "Run a text task through the cascade: decompose into steps, let the cheap "
                "System One engine route and verify each one, execute on the cheapest tier "
                "that earns it, escalate only when the gate fails. Every decision and cost "
                "is in the returned trace."
            ),
            input_schema={
                "properties": {"task": SCHEMA_STRING, "dry_run": SCHEMA_BOOL},
                "required": ["task"],
            },
            handler=_tool_run,
        ),
        Tool(
            name="cascade_plan",
            description="Decompose a task into typed steps without executing anything.",
            input_schema={"properties": {"task": SCHEMA_STRING}, "required": ["task"]},
            handler=_tool_plan,
        ),
        Tool(
            name="cascade_browse",
            description=(
                "Drive a real browser toward a goal for fractions of a cent: the page "
                "reports its own text and controls, one typed decision per step, no vision "
                "model. Best for navigation-shaped goals (find a page, click through, "
                "search, fill one prepared string)." + act_note
            ),
            input_schema={
                "properties": {
                    "goal": SCHEMA_STRING,
                    "url": SCHEMA_STRING,
                    "text": SCHEMA_STRING,
                    "act": SCHEMA_BOOL,
                    "steps": SCHEMA_INT,
                    "writer_tier": SCHEMA_STRING,
                },
                "required": ["goal"],
            },
            handler=_tool_browse,
        ),
        Tool(
            name="cascade_computer",
            description=(
                "Drive the macOS desktop toward a goal through the Accessibility tree, one "
                "typed decision per step, no pixels read. Best for simple app actions "
                "(menus, buttons, one field)." + act_note
            ),
            input_schema={
                "properties": {
                    "goal": SCHEMA_STRING,
                    "app": SCHEMA_STRING,
                    "text": SCHEMA_STRING,
                    "act": SCHEMA_BOOL,
                    "steps": SCHEMA_INT,
                    "writer_tier": SCHEMA_STRING,
                },
                "required": ["goal"],
            },
            handler=_tool_computer,
        ),
        Tool(
            name="cascade_check",
            description=(
                "Validate the cascade config and report which API keys are present. Call "
                "this first when another cascade tool fails."
            ),
            input_schema={"properties": {}, "required": []},
            handler=_tool_check,
        ),
    ]
