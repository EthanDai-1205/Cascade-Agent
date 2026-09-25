"""The MCP server's protocol layer: handshake, listing, dispatch, error discipline.

Everything here drives ``McpServer.handle_message`` in-process — no subprocess, no
pipe, no client — because the protocol was designed as one pure function precisely so
these tests could exist. The tools themselves are stubs; the real ones have their own
tests.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_cascade.mcp_server import (  # noqa: E402
    INVALID_PARAMS,
    INVALID_REQUEST,
    LATEST_VERSION,
    METHOD_NOT_FOUND,
    SUPPORTED_VERSIONS,
    McpServer,
    Tool,
    _error_result,
    _text_result,
    serve_stdio,
)


def a_tool(name: str = "demo", text: str = "did the thing") -> Tool:
    def handler(arguments):
        if arguments.get("fail"):
            raise RuntimeError("the demo tool failed")
        return _text_result(text, {"ok": True})

    return Tool(
        name=name,
        description="a stub tool",
        input_schema={"properties": {"fail": {"type": "boolean"}}, "required": []},
        handler=handler,
    )


def a_server(*tools: Tool) -> McpServer:
    return McpServer(tools=list(tools), version="9.9.9")


def initialize(server: McpServer, version: str = "2025-06-18") -> dict:
    reply = server.handle_message(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": version, "clientInfo": {"name": "test"}}}
    )
    assert reply is not None
    return reply


class TestHandshake(unittest.TestCase):
    def test_initialize_answers_with_a_known_version_and_capabilities(self) -> None:
        reply = initialize(a_server())
        self.assertEqual(reply["jsonrpc"], "2.0")
        self.assertEqual(reply["id"], 1)
        self.assertIn(reply["result"]["protocolVersion"], SUPPORTED_VERSIONS)
        self.assertEqual(reply["result"]["capabilities"], {"tools": {}})
        self.assertEqual(reply["result"]["serverInfo"]["name"], "jev-cascade")
        self.assertEqual(reply["result"]["serverInfo"]["version"], "9.9.9")

    def test_an_unknown_requested_version_gets_our_latest(self) -> None:
        reply = initialize(a_server(), version="1999-01-01")
        self.assertEqual(reply["result"]["protocolVersion"], LATEST_VERSION)

    def test_each_supported_version_is_echoed_back(self) -> None:
        for version in SUPPORTED_VERSIONS:
            reply = initialize(a_server(), version=version)
            self.assertEqual(reply["result"]["protocolVersion"], version)


class TestDiscipline(unittest.TestCase):
    def test_a_notification_never_gets_a_reply(self) -> None:
        server = a_server()
        self.assertIsNone(
            server.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"})
        )
        self.assertIsNone(
            server.handle_message({"jsonrpc": "2.0", "method": "notifications/whatever"})
        )

    def test_ping_answers_an_empty_object(self) -> None:
        reply = a_server().handle_message({"jsonrpc": "2.0", "id": 7, "method": "ping"})
        assert reply is not None
        self.assertEqual(reply["result"], {})

    def test_an_unknown_request_is_method_not_found(self) -> None:
        reply = a_server().handle_message({"jsonrpc": "2.0", "id": 2, "method": "resources/list"})
        assert reply is not None
        self.assertEqual(reply["error"]["code"], METHOD_NOT_FOUND)

    def test_a_request_without_a_method_is_invalid(self) -> None:
        reply = a_server().handle_message({"jsonrpc": "2.0", "id": 3})
        assert reply is not None
        self.assertEqual(reply["error"]["code"], INVALID_REQUEST)

    def test_a_non_object_message_is_invalid(self) -> None:
        reply = a_server().handle_message([1, 2, 3])
        assert reply is not None
        self.assertEqual(reply["error"]["code"], INVALID_REQUEST)
        self.assertIsNone(reply["id"])


class TestToolCalls(unittest.TestCase):
    def test_tools_list_exposes_the_registry_schemas(self) -> None:
        server = a_server(a_tool("alpha"), a_tool("beta"))
        reply = server.handle_message({"jsonrpc": "2.0", "id": 4, "method": "tools/list"})
        assert reply is not None
        tools = reply["result"]["tools"]
        self.assertEqual([tool["name"] for tool in tools], ["alpha", "beta"])
        schema = tools[0]["inputSchema"]
        self.assertEqual(schema["type"], "object")
        self.assertIn("properties", schema)
        self.assertEqual(tools[0]["description"], "a stub tool")

    def test_a_call_reaches_the_tool_and_returns_its_result(self) -> None:
        reply = a_server(a_tool()).handle_message(
            {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
             "params": {"name": "demo", "arguments": {}}}
        )
        assert reply is not None
        result = reply["result"]
        self.assertNotIn("isError", result)
        self.assertEqual(result["content"][0]["text"], "did the thing")
        self.assertEqual(result["structuredContent"], {"ok": True})

    def test_an_unknown_tool_is_invalid_params_not_a_crash(self) -> None:
        reply = a_server().handle_message(
            {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "nope"}}
        )
        assert reply is not None
        self.assertEqual(reply["error"]["code"], INVALID_PARAMS)

    def test_a_raising_tool_becomes_an_is_error_result(self) -> None:
        reply = a_server(a_tool()).handle_message(
            {"jsonrpc": "2.0", "id": 8, "method": "tools/call",
             "params": {"name": "demo", "arguments": {"fail": True}}}
        )
        assert reply is not None
        result = reply["result"]
        self.assertTrue(result["isError"])
        self.assertIn("the demo tool failed", result["content"][0]["text"])

    def test_missing_arguments_become_an_empty_dict(self) -> None:
        reply = a_server(a_tool()).handle_message(
            {"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": "demo"}}
        )
        assert reply is not None
        self.assertNotIn("isError", reply["result"])

    def test_non_object_arguments_are_invalid_params(self) -> None:
        reply = a_server(a_tool()).handle_message(
            {"jsonrpc": "2.0", "id": 10, "method": "tools/call",
             "params": {"name": "demo", "arguments": ["not", "an", "object"]}}
        )
        assert reply is not None
        self.assertEqual(reply["error"]["code"], INVALID_PARAMS)


class TestServeLoop(unittest.TestCase):
    def test_lines_in_replies_out_until_eof(self) -> None:
        import io

        server = a_server(a_tool())
        lines = "\n".join([
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
            "",  # blank lines are skipped, not errors
            "not json at all",  # a parse error with a null id, and the loop carries on
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                        "params": {"name": "demo", "arguments": {}}}),
        ])
        out = io.StringIO()
        serve_stdio(server, stdin=io.StringIO(lines), stdout=out)
        replies = [json.loads(line) for line in out.getvalue().splitlines()]
        # Three replies: initialize, the parse error, the tool call. The notification
        # is silent even though it sits between them.
        self.assertEqual([r.get("id") for r in replies], [1, None, 2])
        self.assertEqual(replies[1]["error"]["code"], -32700)
        self.assertEqual(replies[2]["result"]["content"][0]["text"], "did the thing")

    def test_the_error_and_result_helpers_shape_what_the_spec_asks_for(self) -> None:
        good = _text_result("trace", {"steps": 2})
        self.assertEqual(good["content"], [{"type": "text", "text": "trace"}])
        self.assertEqual(good["structuredContent"], {"steps": 2})
        bad = _error_result("no key")
        self.assertTrue(bad["isError"])
        self.assertNotIn("structuredContent", bad)


if __name__ == "__main__":
    unittest.main()
