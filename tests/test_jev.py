"""Jev client: request shape, error handling, state clipping, and the stub."""

from __future__ import annotations

import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_cascade.config import JevConfig  # noqa: E402
from jev_cascade.jev import (  # noqa: E402
    JevClient,
    JevError,
    StubJev,
    choice_question,
    noul_question,
    score_question,
)


class _Recorder(BaseHTTPRequestHandler):
    """A local stand-in for api.typesafe.ai that records what it was sent."""

    status = 200
    response: dict = {"answers": {}, "usage": {}}
    seen: list[dict] = []
    headers_seen: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", 0))
        body = self.rfile.read(length).decode("utf-8")
        _Recorder.seen.append(json.loads(body))
        _Recorder.headers_seen.append({k.lower(): v for k, v in self.headers.items()})
        payload = json.dumps(_Recorder.response).encode("utf-8")
        self.send_response(_Recorder.status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:  # silence the test output
        return


class _Server:
    def __init__(self, response: dict, status: int = 200) -> None:
        _Recorder.response = response
        _Recorder.status = status
        _Recorder.seen = []
        _Recorder.headers_seen = []
        self.httpd = HTTPServer(("127.0.0.1", 0), _Recorder)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.httpd.server_address[:2]
        self.url = f"http://{host}:{port}/v1/systemone"

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class _DropConnection(BaseHTTPRequestHandler):
    """Accepts the request, then closes the socket with no response at all.

    This is what a flaky proxy does, and the client sees it as
    ``http.client.RemoteDisconnected``, which is a ``ConnectionError`` rather than a
    ``URLError``: it used to escape the retry loop and end the whole caller.
    """

    attempts = 0

    def do_POST(self) -> None:  # noqa: N802
        _DropConnection.attempts += 1
        self.close_connection = True
        self.connection.close()

    def log_message(self, *args) -> None:  # silence the test output
        return


class TestDroppedConnections(unittest.TestCase):
    def _serve(self) -> str:
        _DropConnection.attempts = 0
        httpd = HTTPServer(("127.0.0.1", 0), _DropConnection)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        host, port = httpd.server_address[:2]
        return f"http://{host}:{port}/v1/systemone"

    def test_a_dropped_socket_is_retried_then_raised_as_a_jev_error(self) -> None:
        url = self._serve()
        client = JevClient(JevConfig(base_url=url, model="jev-latest", max_retries=1), api_key="k")

        with self.assertRaises(JevError) as ctx:
            client.ask("state", {"q": noul_question("a", "b")})

        self.assertIn("connection failed", str(ctx.exception))
        self.assertEqual(_DropConnection.attempts, 2, "1 try plus 1 retry")
        self.assertEqual(client.calls, 0, "a failed call is not billed as a call")

    def test_no_retries_still_raises_the_typed_error(self) -> None:
        url = self._serve()
        client = JevClient(JevConfig(base_url=url, model="jev-latest", max_retries=0), api_key="k")

        with self.assertRaises(JevError):
            client.ask("state", {"q": noul_question("a", "b")})

        self.assertEqual(_DropConnection.attempts, 1)


class TestQuestionBuilders(unittest.TestCase):
    def test_noul_shape(self) -> None:
        question = noul_question("it works", "it does not work")
        self.assertEqual(question["type"], "noul")
        self.assertEqual(set(question["criteria"]), {"true", "false"})

    def test_choice_shape_preserves_option_order(self) -> None:
        question = choice_question({"b": "second", "a": None})
        self.assertEqual(question["type"], "choice")
        self.assertEqual(list(question["criteria"]), ["b", "a"])

    def test_score_shape_is_an_ordered_list(self) -> None:
        question = score_question(["bad", "ok", "good"])
        self.assertIsInstance(question["criteria"], list)
        self.assertEqual(question["criteria"], ["bad", "ok", "good"])

    def test_score_needs_levels(self) -> None:
        with self.assertRaises(ValueError):
            score_question([])


class TestJevClientAgainstLocalServer(unittest.TestCase):
    def _config(self, url: str, **overrides) -> JevConfig:
        base = {"base_url": url, "model": "jev-latest", "max_retries": 0}
        base.update(overrides)
        return JevConfig(**base)

    def test_sends_the_documented_payload_and_parses_answers(self) -> None:
        server = _Server(
            {
                "model": "jev-1.13.0",
                "answers": {"pick": {"type": "choice", "choice": "cheap", "confidence": 0.93, "probabilities": {"cheap": 0.93, "expensive": 0.07}}},
                "usage": {"input_tokens": 1000, "output_tokens": 20},
            }
        )
        try:
            client = JevClient(self._config(server.url), api_key="test-key")
            result = client.ask("state text", {"pick": choice_question({"cheap": None, "expensive": None})})
        finally:
            server.stop()

        sent = _Recorder.seen[0]
        self.assertEqual(sent["model"], "jev-latest")
        self.assertEqual(sent["state"], "state text")
        self.assertIn("pick", sent["questions"])
        self.assertEqual(_Recorder.headers_seen[0]["authorization"], "Bearer test-key")

        self.assertEqual(result.choice("pick"), ("cheap", 0.93))
        self.assertEqual(result.model, "jev-1.13.0")
        self.assertEqual(result.tokens_in, 1000)
        self.assertAlmostEqual(result.cost_usd, 1000 / 1_000_000 * 0.042, places=9)
        self.assertFalse(result.stub)

    def test_score_answer_is_a_weighted_float_with_a_legend(self) -> None:
        server = _Server(
            {
                "model": "jev-1.13.0",
                "answers": {
                    "q": {
                        "type": "score",
                        "score": 2.77,
                        "confidence": 0.77,
                        "legend": {"0": "broken", "1": "meh", "2": "ok", "3": "clean"},
                        "probabilities": {"3": 0.89},
                    }
                },
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }
        )
        try:
            client = JevClient(self._config(server.url), api_key="test-key")
            result = client.ask("s", {"q": score_question(["broken", "meh", "ok", "clean"])})
        finally:
            server.stop()
        score, confidence = result.score("q")
        self.assertAlmostEqual(score, 2.77)
        self.assertAlmostEqual(confidence, 0.77)

    def test_missing_key_fails_fast(self) -> None:
        client = JevClient(self._config("http://127.0.0.1:1/v1/systemone"), api_key="")
        with self.assertRaises(JevError) as ctx:
            client.ask("state", {"q": noul_question("a", "b")})
        self.assertIn("missing API key", str(ctx.exception))

    def test_bad_key_is_not_retried(self) -> None:
        server = _Server({"error": {"message": "Cannot authenticate with the server"}}, status=401)
        try:
            client = JevClient(self._config(server.url), api_key="dead")
            with self.assertRaises(JevError) as ctx:
                client.ask("state", {"q": noul_question("a", "b")})
        finally:
            server.stop()
        self.assertEqual(ctx.exception.status, 401)
        self.assertEqual(len(_Recorder.seen), 1)

    def test_overloaded_is_retried_then_raised(self) -> None:
        server = _Server({"error": {"message": "overloaded"}}, status=529)
        try:
            client = JevClient(self._config(server.url, max_retries=2), api_key="k")
            with self.assertRaises(JevError) as ctx:
                client.ask("state", {"q": noul_question("a", "b")})
        finally:
            server.stop()
        self.assertEqual(ctx.exception.status, 529)
        self.assertEqual(len(_Recorder.seen), 3, "1 try plus 2 retries")

    def test_non_json_body_raises_clearly(self) -> None:
        server = _Server({})
        try:
            _Recorder.response = "<html>gateway</html>"  # type: ignore[assignment]
            client = JevClient(self._config(server.url), api_key="k")
            with self.assertRaises(JevError) as ctx:
                client.ask("state", {"q": noul_question("a", "b")})
        finally:
            _Recorder.response = {}
            server.stop()
        self.assertIn("not an object", str(ctx.exception))

    def test_long_state_is_clipped_and_marked(self) -> None:
        server = _Server({"answers": {}, "usage": {}})
        try:
            client = JevClient(self._config(server.url, max_state_chars=100), api_key="k")
            client.ask("x" * 5000, {"q": noul_question("a", "b")})
        finally:
            server.stop()
        sent = _Recorder.seen[0]["state"]
        self.assertLess(len(sent), 200)
        self.assertIn("characters elided", sent)

    def test_usage_accumulates_on_the_client(self) -> None:
        server = _Server({"answers": {}, "usage": {"input_tokens": 500, "output_tokens": 0}})
        try:
            client = JevClient(self._config(server.url), api_key="k")
            for _ in range(3):
                client.ask("state", {"q": noul_question("a", "b")})
        finally:
            server.stop()
        self.assertEqual(client.calls, 3)
        self.assertEqual(client.total_tokens_in, 1500)
        self.assertAlmostEqual(client.total_cost_usd, 1500 / 1_000_000 * 0.042, places=9)


class TestStubJev(unittest.TestCase):
    def test_routes_decision_steps_to_jev(self) -> None:
        stub = StubJev()
        result = stub.ask(
            "step_kind: decide\nstep_goal: pick a label",
            {"executor": choice_question({"cheap": None, "expensive": None, "jev": None})},
        )
        self.assertEqual(result.choice("executor")[0], "jev")

    def test_hard_looking_work_routes_expensive(self) -> None:
        stub = StubJev()
        result = stub.ask(
            "step_kind: generate\nstep_goal: debug the root cause of a protocol race",
            {"executor": choice_question({"cheap": None, "expensive": None})},
        )
        self.assertEqual(result.choice("executor")[0], "expensive")

    def test_plain_work_routes_cheap(self) -> None:
        stub = StubJev()
        result = stub.ask(
            "step_kind: generate\nstep_goal: rename a variable",
            {"executor": choice_question({"cheap": None, "expensive": None})},
        )
        self.assertEqual(result.choice("executor")[0], "cheap")

    def test_verification_trips_on_a_marked_flaw(self) -> None:
        stub = StubJev()
        good = stub.ask("candidate output:\nfine text", {"meets": noul_question("a", "b")})
        bad = stub.ask("candidate output:\nFLAWED-OUTPUT", {"meets": noul_question("a", "b")})
        self.assertGreater(good.noul("meets"), 0.6)
        self.assertLess(bad.noul("meets"), 0.6)

    def test_score_stub_returns_an_ordered_legend(self) -> None:
        stub = StubJev()
        result = stub.ask("state", {"q": score_question(["low", "mid", "high"])})
        answer = result.answers["q"]
        self.assertEqual(answer["legend"], {"0": "low", "1": "mid", "2": "high"})

    def test_stub_is_always_flagged(self) -> None:
        result = StubJev().ask("state", {"q": noul_question("a", "b")})
        self.assertTrue(result.stub)


if __name__ == "__main__":
    unittest.main()
