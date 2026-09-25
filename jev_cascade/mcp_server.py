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
