"""TypeSafe System One (Jev) client.

Jev is not a chat model. You hand it state plus typed questions and it returns
typed answers with calibrated probabilities. Three primitives are used here:

``choice``  pick one of the options you supply (it cannot invent one)
``noul``    probability that yes (a single graded truth value)
``score``   probability-weighted position on an ordered scale

Endpoint: ``POST https://api.typesafe.ai/v1/systemone``.
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import JevConfig

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class JevError(RuntimeError):
    """Any failure talking to System One."""

    def __init__(self, message: str, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass
class JevResult:
    """One System One response."""

    answers: dict[str, dict[str, Any]] = field(default_factory=dict)
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    calls: int = 1
    stub: bool = False

    def choice(self, question_id: str) -> tuple[str, float]:
        answer = self.answers.get(question_id, {})
        return str(answer.get("choice", "")), float(answer.get("confidence") or 0.0)

    def noul(self, question_id: str) -> float:
        answer = self.answers.get(question_id, {})
        value = answer.get("noul")
        return float(value) if value is not None else 0.0

    def score(self, question_id: str) -> tuple[float, float]:
        answer = self.answers.get(question_id, {})
        return float(answer.get("score") or 0.0), float(answer.get("confidence") or 0.0)

    def probabilities(self, question_id: str) -> dict[str, float]:
        answer = self.answers.get(question_id, {})
        return {str(k): float(v) for k, v in dict(answer.get("probabilities") or {}).items()}


def choice_question(
    options: dict[str, str | None], instructions: str | None = None
) -> dict[str, Any]:
    """A Choice question. Descriptions are optional per option.

    ``instructions`` is the one-line framing of the question. It is omitted from the
    payload when the caller gives none, so an existing call sends exactly the bytes it
    always did; both engines accept it when it is there, and the local Laya engine leans
    on it more than the hosted one does.
    """

    question: dict[str, Any] = {"type": "choice", "criteria": dict(options)}
    if instructions:
        question["instructions"] = instructions
    return question


def noul_question(
    true_case: str, false_case: str, instructions: str | None = None
) -> dict[str, Any]:
    """A Noul question: probability that the true case holds."""

    question: dict[str, Any] = {"type": "noul", "criteria": {"true": true_case, "false": false_case}}
    if instructions:
        question["instructions"] = instructions
    return question


def score_question(levels: list[str], instructions: str | None = None) -> dict[str, Any]:
    """A Score question. The levels list is ORDERED; a map is rejected by the API."""

    if not levels:
        raise ValueError("score_question needs at least one level")
    question: dict[str, Any] = {"type": "score", "criteria": list(levels)}
    if instructions:
        question["instructions"] = instructions
    return question


class Jev(Protocol):
    """What the agent needs from a decision engine."""

    stub: bool
    calls: int

    def ask(self, state: str, questions: dict[str, dict[str, Any]]) -> JevResult: ...


class JevClient:
    """Live System One client. Stdlib only, so it runs anywhere Python does."""

    def __init__(self, config: JevConfig, api_key: str | None = None) -> None:
        self.config = config
        self._api_key = api_key if api_key is not None else config.api_key()
        self.calls = 0
        self.stub = False
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_cost_usd = 0.0

    @property
    def key_present(self) -> bool:
        return bool(self._api_key)

    def ask(self, state: str, questions: dict[str, dict[str, Any]]) -> JevResult:
        if not questions:
            raise JevError("at least one question is required")
        if not self._api_key:
            raise JevError(
                f"missing API key: set ${self.config.api_key_env} (or use --dry-run to stub Jev)"
            )
        payload = {
            "model": self.config.model,
            "state": self._clip(state),
            "questions": questions,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.config.base_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        started = time.monotonic()
        attempt = 0
        while True:
            attempt += 1
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
                    raw = response.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")
                exc.close()
                if exc.code in RETRYABLE_STATUS and attempt <= self.config.max_retries:
                    time.sleep(min(2.0 ** (attempt - 1), 8.0))
                    continue
                raise JevError(
                    f"System One returned HTTP {exc.code}: {_short(detail)}", exc.code, detail
                ) from exc
            except urllib.error.URLError as exc:
                if attempt <= self.config.max_retries:
                    time.sleep(min(2.0 ** (attempt - 1), 8.0))
                    continue
                raise JevError(f"System One unreachable: {exc.reason}") from exc
            except (OSError, http.client.HTTPException) as exc:
                # A socket dropped mid-request, a reset connection, or a truncated
                # response body. ``RemoteDisconnected`` is a ``ConnectionError``, not a
                # ``URLError``, so it used to escape this retry loop entirely and kill
                # the caller: one dropped socket ended a whole eval run. These are
                # transient in the same way a 529 is, and retry on the same budget.
                if attempt <= self.config.max_retries:
                    time.sleep(min(2.0 ** (attempt - 1), 8.0))
                    continue
                raise JevError(
                    f"System One connection failed: {type(exc).__name__}: {exc}"
                ) from exc
        latency = time.monotonic() - started

        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise JevError(f"System One returned non-JSON: {_short(raw)}") from exc
        if not isinstance(decoded, dict):
            raise JevError(f"System One returned a JSON {type(decoded).__name__}, not an object: {_short(raw)}")

        usage = decoded.get("usage") or {}
        tokens_in = int(usage.get("input_tokens") or 0)
        tokens_out = int(usage.get("output_tokens") or 0)
        cost = self.config.price.cost(tokens_in, tokens_out)

        self.calls += 1
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out
        self.total_cost_usd += cost

        return JevResult(
            answers=dict(decoded.get("answers") or {}),
            model=str(decoded.get("model", self.config.model)),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            latency_s=latency,
            calls=1,
        )

    def _clip(self, state: str) -> str:
        limit = self.config.max_state_chars
        if len(state) <= limit:
            return state
        head = limit // 2
        tail = limit - head
        return f"{state[:head]}\n[... {len(state) - limit} characters elided ...]\n{state[-tail:]}"


class StubJev:
    """Offline heuristic stand-in with the same interface.

    It is deterministic and rule based, reads the structural markers the agent
    puts in the state (``step_kind:``, ``step_goal:``), and never touches the
    network. Every result it returns is flagged ``stub=True`` so no run can
    mistake it for real System One output.
    """

    def __init__(self, config: JevConfig | None = None) -> None:
        self.config = config or JevConfig()
        self.calls = 0
        self.stub = True
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_cost_usd = 0.0

    HARD_WORDS = (
        "design",
        "architect",
        "refactor",
        "why",
        "debug",
        "root cause",
        "protocol",
        "migrat",
        "security",
        "trade-off",
        "tradeoff",
    )

    def ask(self, state: str, questions: dict[str, dict[str, Any]]) -> JevResult:
        self.calls += 1
        lowered = state.lower()
        answers: dict[str, dict[str, Any]] = {}
        for question_id, question in questions.items():
            kind = question.get("type")
            criteria = question.get("criteria")
            if kind == "choice":
                answers[question_id] = self._stub_choice(lowered, dict(criteria or {}))
            elif kind == "noul":
                answers[question_id] = {"type": "noul", "noul": self._stub_noul(lowered)}
            elif kind == "score":
                answers[question_id] = self._stub_score(list(criteria or []))
            else:
                raise JevError(f"stub does not implement question type {kind!r}")
        tokens_in = max(1, len(state) // 4)
        tokens_out = 8 * len(questions)
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out
        cost = self.config.price.cost(tokens_in, tokens_out)
        self.total_cost_usd += cost
        return JevResult(
            answers=answers,
            model="stub-jev",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            latency_s=0.0,
            calls=1,
            stub=True,
        )

    def _stub_choice(self, state: str, options: dict[str, Any]) -> dict[str, Any]:
        keys = list(options.keys())
        decision_step = "step_kind: decide" in state
        hard = any(word in state for word in self.HARD_WORDS)

        if decision_step and "jev" in keys:
            pick = "jev"
        elif hard and "expensive" in keys:
            pick = "expensive"
        elif "cheap" in keys:
            pick = "cheap"
        else:
            pick = keys[0] if keys else ""
        probabilities = {key: (0.85 if key == pick else round(0.15 / max(1, len(keys) - 1), 3)) for key in keys}
        return {
            "type": "choice",
            "choice": pick,
            "confidence": 0.85,
            "probabilities": probabilities,
        }

    def _stub_noul(self, state: str) -> float:
        if "flawed-output" in state.lower() or "candidate output: <empty>" in state.lower():
            return 0.15
        if "candidate output:" in state:
            return 0.9
        return 0.7

    def _stub_score(self, levels: list[str]) -> dict[str, Any]:
        if not levels:
            return {"type": "score", "score": 0.0, "confidence": 0.0, "legend": {}, "probabilities": {}}
        index = len(levels) - 2 if len(levels) > 1 else 0
        legend = {str(i): level for i, level in enumerate(levels)}
        probabilities = {str(i): round(0.05 / max(1, len(levels) - 1), 3) for i in range(len(levels))}
        probabilities[str(index)] = 0.95
        return {
            "type": "score",
            "score": float(index),
            "confidence": 0.95,
            "legend": legend,
            "probabilities": probabilities,
        }


def build_jev(config: JevConfig, dry_run: bool = False) -> Jev:
    """Return the engine named by ``[jev] kind``, or the stub on a dry run.

    The two engines answer the same three typed questions, so the caller cannot tell them
    apart. What differs is measured, not theoretical: the local one is free, fast and good
    at classification-shaped calls, and it is not reliable at choosing an action from a
    screen. Point roles at it accordingly.
    """

    if dry_run:
        return StubJev(config)
    if config.kind == "laya":
        from .laya import LayaJev  # noqa: PLC0415 (keeps the optional runtime out of the import path)

        return LayaJev(config)
    return JevClient(config)


def _short(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[:limit]}..."
