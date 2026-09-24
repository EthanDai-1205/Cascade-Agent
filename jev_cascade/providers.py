"""Model tiers.

Every tier is reached over either the OpenAI chat-completions protocol (which
covers DeepSeek, OpenRouter, Groq, vLLM, Ollama, LM Studio and anything else of
that shape) or the Anthropic messages protocol (Claude-compatible proxies).
Adding a provider is a config edit, not a code change.

``MockProvider`` and ``ScriptedProvider`` keep the whole pipeline runnable with
no keys and no network, which is what the tests and ``--dry-run`` use.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .config import Price, TierConfig

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
MAX_ATTEMPTS = 4


class ProviderError(RuntimeError):
    """Any failure talking to a tier."""

    def __init__(self, message: str, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass
class Completion:
    text: str
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    stub: bool = False
    finish_reason: str = ""
    truncated: bool = False

    @property
    def blank(self) -> bool:
        return not self.text.strip()


@dataclass
class _Extracted:
    """What a provider pulls out of one response body."""

    text: str
    tokens_in: int
    tokens_out: int
    finish_reason: str = ""
    had_reasoning: bool = False


class Provider(Protocol):
    tier: str
    stub: bool

    def complete(
        self, system: str, user: str, max_tokens: int | None = None, json_mode: bool = False
    ) -> Completion: ...


class _HTTPProvider:
    """Shared plumbing: retries, error mapping, usage accounting, cost."""

    def __init__(self, config: TierConfig) -> None:
        self.config = config
        self.tier = config.name
        self.stub = False
        self.calls = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_cost_usd = 0.0

    def _url(self) -> str:
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:
        raise NotImplementedError

    def _payload(self, system: str, user: str) -> dict[str, Any]:
        raise NotImplementedError

    def _extract(self, decoded: dict[str, Any]) -> _Extracted:
        """Pull the answer text and usage out of one response body."""
        raise NotImplementedError

    def complete(
        self, system: str, user: str, max_tokens: int | None = None, json_mode: bool = False
    ) -> Completion:
        if not self.config.api_key():
            raise ProviderError(
                f"tier {self.config.name!r} has no API key: set ${self.config.api_key_env}",
                status=401,
            )

        request = urllib.request.Request(
            self._url(),
            data=json.dumps(self._payload(system, user, max_tokens, json_mode)).encode("utf-8"),
            headers=self._headers(),
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
                if exc.code in RETRYABLE_STATUS and attempt < MAX_ATTEMPTS:
                    time.sleep(min(2.0 ** (attempt - 1), 8.0))
                    continue
                raise ProviderError(
                    f"tier {self.config.name!r} returned HTTP {exc.code}: {_short(detail)}",
                    exc.code,
                    detail,
                ) from exc
            except urllib.error.URLError as exc:
                if attempt < MAX_ATTEMPTS:
                    time.sleep(min(2.0 ** (attempt - 1), 8.0))
                    continue
                raise ProviderError(
                    f"tier {self.config.name!r} unreachable at {self._url()}: {exc.reason}"
                ) from exc
        latency = time.monotonic() - started

        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"tier {self.config.name!r} returned non-JSON: {_short(raw)}") from exc
        if not isinstance(decoded, dict):
            raise ProviderError(
                f"tier {self.config.name!r} returned a JSON {type(decoded).__name__}, not an object: "
                f"{_short(raw)}",
                body=raw,
            )

        extracted = self._extract(decoded)
        cost = self.config.price.cost(extracted.tokens_in, extracted.tokens_out)
        self.calls += 1
        self.total_tokens_in += extracted.tokens_in
        self.total_tokens_out += extracted.tokens_out
        self.total_cost_usd += cost

        if not extracted.text.strip():
            # A blank answer is a failure, never a deliverable. Reasoning models count
            # their thinking against max_tokens, so the usual cause is a cap that is too
            # small: say so instead of handing an empty string to the caller.
            hint = (
                "the model spent its whole budget thinking"
                if extracted.had_reasoning
                else "the response carried no text"
            )
            effective_cap = max_tokens or self.config.max_tokens
            raise ProviderError(
                f"tier {self.config.name!r} returned an empty answer "
                f"(finish_reason={extracted.finish_reason or 'unknown'}; {hint}). "
                f"Raise max_tokens (this call used {effective_cap}): thinking counts against it."
            )

        return Completion(
            text=extracted.text,
            model=str(decoded.get("model", self.config.model)),
            tokens_in=extracted.tokens_in,
            tokens_out=extracted.tokens_out,
            cost_usd=cost,
            latency_s=latency,
            finish_reason=extracted.finish_reason,
            truncated=extracted.finish_reason == "length",
        )


class OpenAICompatibleProvider(_HTTPProvider):
    """A tier behind an OpenAI-compatible /chat/completions endpoint."""

    def _url(self) -> str:
        return f"{self.config.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key()}",
            "Content-Type": "application/json",
        }

    def _payload(
        self, system: str, user: str, max_tokens: int | None = None, json_mode: bool = False
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _extract(self, decoded: dict[str, Any]) -> _Extracted:
        choices = decoded.get("choices") or []
        if not choices:
            raise ProviderError(
                f"tier {self.config.name!r} returned no choices: {_short(json.dumps(decoded))}",
                body=json.dumps(decoded),
            )
        choice = choices[0]
        message = choice.get("message") or {}
        reasoning = message.get("reasoning_content")
        had_reasoning = bool(isinstance(reasoning, str) and reasoning.strip())
        finish_reason = str(choice.get("finish_reason") or "")
        text = _openai_text(choice)

        if not text.strip() and had_reasoning and finish_reason != "length":
            # Some OpenAI-shaped servers answer inside reasoning_content and leave
            # content empty. Accept that, but never when the reply was cut off: a
            # truncated chain of thought is thinking, not an answer.
            text = reasoning if isinstance(reasoning, str) else ""

        usage = decoded.get("usage") or {}
        return _Extracted(
            text=text,
            tokens_in=int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
            finish_reason=finish_reason,
            had_reasoning=had_reasoning,
        )


class AnthropicProvider(_HTTPProvider):
    """A tier behind an Anthropic-compatible /v1/messages endpoint.

    Useful for Claude-shaped proxies, and for using a different model family as
    an independent judge, since a model judging its own output is biased toward it.
    """

    def _url(self) -> str:
        return f"{self.config.base_url}/v1/messages"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.config.api_key(),
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _payload(
        self, system: str, user: str, max_tokens: int | None = None, json_mode: bool = False
    ) -> dict[str, Any]:
        # json_mode has no Anthropic equivalent that is safe to assume here; the
        # planner prompt already states the required shape.
        return {
            "model": self.config.model,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": self.config.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }

    def _extract(self, decoded: dict[str, Any]) -> _Extracted:
        blocks = decoded.get("content")
        if not isinstance(blocks, list):
            raise ProviderError(
                f"tier {self.config.name!r} returned no content blocks: {_short(json.dumps(decoded))}",
                body=json.dumps(decoded),
            )
        parts = [
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        usage = decoded.get("usage") or {}
        stop = str(decoded.get("stop_reason") or "")
        return _Extracted(
            text="".join(parts),
            tokens_in=int(usage.get("input_tokens") or 0),
            tokens_out=int(usage.get("output_tokens") or 0),
            finish_reason="length" if stop == "max_tokens" else stop,
            had_reasoning=any(
                isinstance(block, dict) and block.get("type") == "thinking" for block in blocks
            ),
        )


class MockProvider:
    """Deterministic, offline, free. Used by --dry-run and by the test suite.

    When a caller asks for JSON mode it answers with a shaped reply rather than a
    sentence, so a dry run exercises the planner and the eval's judge instead of
    failing at the parse step. The replies are canned: they prove plumbing, never
    quality, and every completion is flagged as a stub.
    """

    JUDGE_MARKERS = ("a_sufficient", "grade two candidate")  # same list the reply logic checks

    def __init__(self, config: TierConfig) -> None:
        self.config = config
        self.tier = config.name
        self.stub = True
        self.calls = 0
        self.prompts: list[str] = []
        self.caps: list[int | None] = []
        self.json_modes: list[bool] = []

    def complete(
        self, system: str, user: str, max_tokens: int | None = None, json_mode: bool = False
    ) -> Completion:
        self.calls += 1
        self.prompts.append(user)
        self.caps.append(max_tokens)
        self.json_modes.append(json_mode)
        goal = _marker(user, "step_goal") or "the step"
        if json_mode:
            return self._json_reply(system, user, goal)
        body = (
            f"[{self.tier} mock] deliverable for: {goal}\n"
            f"- produced offline by MockProvider, no model was called\n"
            f"- prompt characters: {len(system) + len(user)}"
        )
        return Completion(
            text=body,
            model=f"{self.config.model or self.tier}-mock",
            tokens_in=max(1, (len(system) + len(user)) // 4),
            tokens_out=max(1, len(body) // 4),
            cost_usd=0.0,
            latency_s=0.0,
            stub=True,
        )


    def _json_reply(self, system: str, user: str, goal: str) -> Completion:
        haystack = f"{system}\n{user}".lower()
        if any(marker in haystack for marker in ("a_sufficient", "grade two candidate")):
            body = json.dumps(
                {
                    "a_sufficient": True,
                    "b_sufficient": True,
                    "better": "tie",
                    "reason": "stub judge: both candidates are placeholders",
                }
            )
        else:
            body = json.dumps(
                {
                    "steps": [
                        {
                            "id": "s1",
                            "goal": goal[:160],
                            "kind": "generate",
                            "criteria": "the output satisfies the goal and is directly usable",
                        }
                    ]
                }
            )
        return Completion(
            text=body,
            model=f"{self.config.model or self.tier}-mock",
            tokens_in=max(1, (len(system) + len(user)) // 4),
            tokens_out=max(1, len(body) // 4),
            cost_usd=0.0,
            latency_s=0.0,
            stub=True,
            finish_reason="stop",
        )


class ScriptedProvider:
    """Returns prepared outputs in order. Tests use this to force specific outcomes."""

    def __init__(
        self,
        name: str,
        outputs: list[str],
        model: str = "scripted",
        price: Price | None = None,
        stub: bool = True,
    ) -> None:
        self.tier = name
        self.stub = stub
        self.model = model
        self.price = price
        self.outputs = list(outputs)
        self.calls = 0
        self.prompts: list[str] = []
        self.caps: list[int | None] = []
        self.json_modes: list[bool] = []
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_cost_usd = 0.0

    def complete(
        self, system: str, user: str, max_tokens: int | None = None, json_mode: bool = False
    ) -> Completion:
        self.calls += 1
        self.prompts.append(user)
        self.caps.append(max_tokens)
        self.json_modes.append(json_mode)
        text = self.outputs.pop(0) if self.outputs else "[scripted] exhausted"
        tokens_in = max(1, (len(system) + len(user)) // 4)
        tokens_out = max(1, len(text) // 4)
        cost = self.price.cost(tokens_in, tokens_out) if self.price is not None else 0.0
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out
        self.total_cost_usd += cost
        return Completion(
            text=text,
            model=self.model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            latency_s=0.0,
            stub=self.stub,
        )


def build_provider(config: TierConfig, dry_run: bool = False) -> Provider:
    """Live provider unless the caller asked for a dry run or the tier is mock."""

    if dry_run or config.kind == "mock":
        return MockProvider(config)
    if config.kind == "anthropic":
        return AnthropicProvider(config)
    return OpenAICompatibleProvider(config)


def _openai_text(choice: dict) -> str:
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        joined = "".join(parts)
        if joined.strip():
            return joined
    return ""


def _marker(text: str, key: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(f"{key}:"):
            return stripped.split(":", 1)[1].strip()
    return ""


def _short(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[:limit]}..."
