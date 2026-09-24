"""Provider clients against a local HTTP server, for both wire protocols."""

from __future__ import annotations

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jev_cascade.config import Price, TierConfig  # noqa: E402
from jev_cascade.providers import (  # noqa: E402
    AnthropicProvider,
    MockProvider,
    OpenAICompatibleProvider,
    ProviderError,
    build_provider,
)


class _Recorder(BaseHTTPRequestHandler):
    status = 200
    response: object = {}
    seen: list[dict] = []
    paths: list[str] = []
    headers_seen: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", 0))
        raw = self.rfile.read(length).decode("utf-8")
        _Recorder.paths.append(self.path)
        _Recorder.seen.append(json.loads(raw))
        _Recorder.headers_seen.append({k.lower(): v for k, v in self.headers.items()})
        payload = json.dumps(_Recorder.response).encode("utf-8")
        self.send_response(_Recorder.status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:
        return


class _Server:
    def __init__(self, response: object, status: int = 200) -> None:
        _Recorder.response = response
        _Recorder.status = status
        _Recorder.seen = []
        _Recorder.paths = []
        _Recorder.headers_seen = []
        self.httpd = HTTPServer(("127.0.0.1", 0), _Recorder)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        host, port = self.httpd.server_address[:2]
        self.base_url = f"http://{host}:{port}"

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def tier(name: str, kind: str, base_url: str, model: str = "m", env: str = "JC_TEST_KEY") -> TierConfig:
    return TierConfig(
        name=name,
        kind=kind,
        base_url=base_url,
        model=model,
        api_key_env=env,
        price=Price(input=1.0, output=2.0),
        max_tokens=64,
    )


class TestOpenAICompatibleProvider(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["JC_TEST_KEY"] = "test-key"

    def test_request_shape_and_usage(self) -> None:
        server = _Server(
            {
                "model": "deepseek-flash",
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
            }
        )
        try:
            provider = OpenAICompatibleProvider(tier("cheap", "openai", server.base_url))
            completion = provider.complete("sys", "user")
        finally:
            server.stop()

        self.assertEqual(_Recorder.paths[0], "/chat/completions")
        self.assertEqual(_Recorder.headers_seen[0]['authorization'], "Bearer test-key")
        self.assertEqual(_Recorder.seen[0]["messages"][0]["content"], "sys")
        self.assertEqual(_Recorder.seen[0]["max_tokens"], 64)
        self.assertEqual(completion.text, "hello")
        self.assertEqual(completion.tokens_in, 1000)
        self.assertEqual(completion.tokens_out, 500)
        self.assertAlmostEqual(completion.cost_usd, 1000 / 1e6 * 1.0 + 500 / 1e6 * 2.0, places=9)
        self.assertFalse(completion.stub)

    def test_missing_key_fails_before_any_request(self) -> None:
        provider = OpenAICompatibleProvider(
            tier("cheap", "openai", "http://127.0.0.1:1", env="JC_TEST_ABSENT_KEY")
        )
        os.environ.pop("JC_TEST_ABSENT_KEY", None)
        with self.assertRaises(ProviderError) as ctx:
            provider.complete("s", "u")
        self.assertEqual(ctx.exception.status, 401)

    def test_401_on_the_wire_is_not_retried(self) -> None:
        server = _Server({"error": {"message": "bad key"}}, status=401)
        try:
            provider = OpenAICompatibleProvider(tier("cheap", "openai", server.base_url))
            with self.assertRaises(ProviderError) as ctx:
                provider.complete("s", "u")
        finally:
            server.stop()
        self.assertEqual(ctx.exception.status, 401)
        self.assertEqual(len(_Recorder.seen), 1)

    def test_object_without_choices_is_an_error(self) -> None:
        server = _Server({"model": "m", "usage": {}})
        try:
            provider = OpenAICompatibleProvider(tier("cheap", "openai", server.base_url))
            with self.assertRaises(ProviderError) as ctx:
                provider.complete("s", "u")
        finally:
            server.stop()
        self.assertIn("no choices", str(ctx.exception))


class TestCallOptionsAndEmptyAnswers(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["JC_TEST_KEY"] = "test-key"

    def _provider(self, server: _Server) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(tier("cheap", "openai", server.base_url, model="m"))

    def test_json_mode_and_token_override_reach_the_wire(self) -> None:
        server = _Server(
            {
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        )
        try:
            provider = self._provider(server)
            provider.complete("s", "u", max_tokens=99, json_mode=True)
        finally:
            server.stop()
        body = _Recorder.seen[0]
        self.assertEqual(body["max_tokens"], 99)
        self.assertEqual(body["response_format"], {"type": "json_object"})

    def test_default_cap_is_used_when_no_override_is_given(self) -> None:
        server = _Server(
            {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        )
        try:
            provider = self._provider(server)
            provider.complete("s", "u")
        finally:
            server.stop()
        self.assertEqual(_Recorder.seen[0]["max_tokens"], 64)
        self.assertNotIn("response_format", _Recorder.seen[0])

    def test_truncated_reply_is_flagged(self) -> None:
        server = _Server(
            {
                "choices": [{"message": {"content": '{"steps": ['}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 64},
            }
        )
        try:
            completion = self._provider(server).complete("s", "u")
        finally:
            server.stop()
        self.assertTrue(completion.truncated)
        self.assertEqual(completion.finish_reason, "length")

    def test_empty_answer_is_an_error_not_a_deliverable(self) -> None:
        server = _Server(
            {
                "choices": [
                    {
                        "message": {"content": "", "reasoning_content": "thinking hard about it"},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1024},
            }
        )
        try:
            provider = self._provider(server)
            with self.assertRaises(ProviderError) as ctx:
                provider.complete("s", "u")
        finally:
            server.stop()
        message = str(ctx.exception)
        self.assertIn("empty answer", message)
        self.assertIn("budget thinking", message)
        self.assertIn("max_tokens", message)

    def test_reasoning_only_answer_is_accepted_when_it_was_not_cut_off(self) -> None:
        server = _Server(
            {
                "choices": [
                    {
                        "message": {"content": "", "reasoning_content": "the answer is 42"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            }
        )
        try:
            completion = self._provider(server).complete("s", "u")
        finally:
            server.stop()
        self.assertEqual(completion.text, "the answer is 42")
        self.assertFalse(completion.truncated)

    def test_anthropic_max_tokens_maps_to_truncated(self) -> None:
        server = _Server({"content": [{"type": "text", "text": "partial"}], "stop_reason": "max_tokens"})
        try:
            completion = AnthropicProvider(tier("judge", "anthropic", server.base_url)).complete("s", "u")
        finally:
            server.stop()
        self.assertTrue(completion.truncated)
        self.assertEqual(completion.text, "partial")


class TestAnthropicProvider(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["JC_TEST_KEY"] = "test-key"

    def test_request_shape_and_text_blocks(self) -> None:
        server = _Server(
            {
                "model": "claude-fable-5",
                "content": [
                    {"type": "text", "text": "part one "},
                    {"type": "thinking", "thinking": "ignored"},
                    {"type": "text", "text": "part two"},
                ],
                "usage": {"input_tokens": 200, "output_tokens": 100},
            }
        )
        try:
            provider = AnthropicProvider(tier("judge", "anthropic", server.base_url))
            completion = provider.complete("sys prompt", "user prompt")
        finally:
            server.stop()

        self.assertEqual(_Recorder.paths[0], "/v1/messages")
        headers = _Recorder.headers_seen[0]
        self.assertEqual(headers["x-api-key"], "test-key")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        body = _Recorder.seen[0]
        self.assertEqual(body["system"], "sys prompt")
        self.assertEqual(body["messages"], [{"role": "user", "content": "user prompt"}])
        self.assertNotIn("stream", body)
        self.assertEqual(completion.text, "part one part two")
        self.assertEqual(completion.tokens_in, 200)
        self.assertAlmostEqual(completion.cost_usd, 200 / 1e6 * 1.0 + 100 / 1e6 * 2.0, places=9)

    def test_non_list_content_is_an_error(self) -> None:
        server = _Server({"content": "not a list"})
        try:
            provider = AnthropicProvider(tier("judge", "anthropic", server.base_url))
            with self.assertRaises(ProviderError) as ctx:
                provider.complete("s", "u")
        finally:
            server.stop()
        self.assertIn("no content blocks", str(ctx.exception))


class TestMockProvider(unittest.TestCase):
    """A dry run must exercise the planner and the judge, not fail at their parsers."""

    def _mock(self) -> MockProvider:
        return MockProvider(tier("cheap", "mock", ""))

    def test_plain_reply_stays_a_placeholder(self) -> None:
        completion = self._mock().complete("sys", "step_goal: rename the variable")
        self.assertTrue(completion.stub)
        self.assertIn("rename the variable", completion.text)
        self.assertIn("no model was called", completion.text)

    def test_json_mode_answers_the_planner_with_a_step(self) -> None:
        completion = self._mock().complete(
            "You are the planner inside a cost-aware agent. Split the task into steps.",
            "Task: rename the variable",
            json_mode=True,
        )
        payload = json.loads(completion.text)
        self.assertIn("steps", payload)
        self.assertEqual(payload["steps"][0]["kind"], "generate")
        self.assertTrue(payload["steps"][0]["criteria"])
        self.assertTrue(completion.stub)

    def test_json_mode_answers_the_judge_with_a_verdict(self) -> None:
        completion = self._mock().complete(
            "You grade two candidate outputs for one step of a job.",
            "-- CANDIDATE A --\nx\n-- CANDIDATE B --\ny",
            json_mode=True,
        )
        payload = json.loads(completion.text)
        self.assertIn("a_sufficient", payload)
        self.assertIn("better", payload)

    def test_json_mode_records_the_cap_and_the_flag(self) -> None:
        mock = self._mock()
        mock.complete("sys", "user", max_tokens=321, json_mode=True)
        self.assertEqual(mock.caps, [321])
        self.assertEqual(mock.json_modes, [True])


class TestBuildProvider(unittest.TestCase):
    def test_kind_dispatch(self) -> None:
        self.assertIsInstance(build_provider(tier("a", "openai", "http://x")), OpenAICompatibleProvider)
        self.assertIsInstance(build_provider(tier("a", "anthropic", "http://x")), AnthropicProvider)
        self.assertTrue(build_provider(tier("a", "mock", "")).stub)
        self.assertTrue(build_provider(tier("a", "openai", "http://x"), dry_run=True).stub)


if __name__ == "__main__":
    unittest.main()
