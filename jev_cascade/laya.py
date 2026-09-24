"""A local Laya decision engine behind the same one-method protocol as Jev.

Laya (Convai Innovations) answers the same three typed questions Jev does, ``choice``,
``score`` and ``noul``, so it can stand in for the hosted engine wherever the agent needs
a decision. This module is the port for ``laya-mlx``, the Apple Silicon MLX runtime, which
runs the model locally: no API key, no network, and no cost per call.

What this is for, measured rather than assumed (2026-09-21, 8 to 16 decisions per
condition, positive control 7/8 in Laya's own trained style):

* **Good at classification-shaped calls.** Routing a request to a category scored 7/8, and
  a yes/no triage scored 7/8. That covers the tier router, the verification gate and
  ``decide`` steps, which are most of the calls a cascade makes.
* **Weak at situated action selection.** Choosing what to press on a live screen scored
  1/6 where Jev scored 5/6, and on synthetic screens it bails out to "wait" or "done"
  rather than reading the state. It prefers actions that name no target, and its item head
  contradicted its action head in the same request. Do not use it to pick an action.

Two operational limits worth knowing before pointing a role at it:

* The context is small. The published checkpoints allow 512 tokens for the English one and
  1024 for multilingual and typed-decisions, and that budget holds the instructions, every
  option label and the state together. Above roughly 126 options the runtime refuses the
  question outright rather than truncating the choices.
* Given a large enough state it truncates silently, so this port clips the state itself and
  says so, instead of letting the encoder quietly drop the end of the prompt.

``[jev] max_len`` and ``head_max_len`` exist because raising them is the one measured way
to take a bigger question set: on this machine 255 options were refused at the checkpoint's
default 512/192 and answered at 2048/1024 (111 ms, 1064 tokens). That is outside the
published validation, so it is opt-in and off by default.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from .config import JevConfig
from .jev import JevError, JevResult

DEFAULT_CHECKPOINT = "aac6fef/laya-mlx"
DTYPE = "float16"

# Roughly four characters per token for English text. The state is clipped to a fraction of
# the window because the question prefix (instructions plus every option label) shares it.
CHARS_PER_TOKEN = 4
STATE_SHARE = 0.55

_PATCHED: dict[tuple[str, int, int], str] = {}


def checkpoint_for(config: JevConfig) -> str:
    """The checkpoint to load for this config.

    ``JevConfig.model`` defaults to the hosted model name, so a caller that builds the
    config directly (``JevConfig(kind="laya")``, which tests and embedders do) would
    otherwise send "jev-latest" to the Hugging Face resolver. Anything that is not a Laya
    checkpoint name falls back to the published one rather than failing on a download.
    """

    model = (config.model or "").strip()
    if not model or model == JevConfig.model or "/" not in model:
        return DEFAULT_CHECKPOINT
    return model


class LayaJev:
    """Local MLX Laya, answering through the ``Jev`` protocol.

    Costs nothing per call, so the ledger records zero for it and the price is not consulted.
    """

    stub = False

    def __init__(self, config: JevConfig, agent: Any | None = None) -> None:
        self.config = config
        self.calls = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_cost_usd = 0.0
        self.truncated_states = 0
        self._agent = agent if agent is not None else self._load()

    # ------------------------------------------------------------------ lifecycle

    def _load(self) -> Any:
        # Configuration is checked before the dependency, so a misconfigured engine says so
        # even on a machine where the optional runtime is not installed.
        target = checkpoint_for(self.config)
        if bool(self.config.max_len) != bool(self.config.head_max_len):
            raise JevError(
                "[jev] max_len and head_max_len must be set together: the Laya runtime takes "
                "both context limits from the checkpoint and rejects anything else"
            )
        if self.config.max_len and (
            self.config.max_len <= self.config.head_max_len or self.config.head_max_len <= 4
        ):
            raise JevError("[jev] needs 4 < head_max_len < max_len")

        try:
            import laya_mlx  # noqa: PLC0415 (optional dependency, imported on demand)
        except ImportError as exc:  # pragma: no cover - exercised by the error test
            raise JevError(
                "the Laya engine needs the optional laya-mlx package: "
                "pip install laya-mlx (Apple Silicon, macOS 14+, Python 3.11+). "
                "Use [jev] kind = \"http\" for the hosted engine instead."
            ) from exc

        if self.config.max_len:
            target = _patched_checkpoint(
                laya_mlx, target, self.config.max_len, self.config.head_max_len
            )
        try:
            return laya_mlx.load(target, dtype=DTYPE)
        except Exception as exc:
            raise JevError(f"could not load the Laya checkpoint {target!r}: {exc}") from exc

    @property
    def model(self) -> str:
        return "laya-mlx"

    # ---------------------------------------------------------------------- ask

    def ask(self, state: str, questions: dict[str, dict[str, Any]]) -> JevResult:
        if not questions:
            raise JevError("at least one question is required")

        clipped, was_clipped = self._clip_state(state)
        if was_clipped:
            self.truncated_states += 1

        payload = {qid: _translate(qid, question) for qid, question in questions.items()}
        started = time.monotonic()
        try:
            raw = self._agent.predict(clipped, payload)
        except ValueError as exc:
            # The runtime's own guard, e.g. "too many options for the token budget".
            raise JevError(f"Laya refused the question: {exc}") from exc
        except Exception as exc:
            raise JevError(f"Laya failed: {type(exc).__name__}: {exc}") from exc
        latency = time.monotonic() - started

        usage = raw.get("usage") or {}
        tokens_in = int(usage.get("input_tokens") or 0)
        tokens_out = int(usage.get("output_tokens") or 0)
        self.calls += 1
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out

        return JevResult(
            answers=_normalise(raw.get("answers") or {}),
            model=str(raw.get("model") or "laya"),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=0.0,
            latency_s=latency,
            calls=1,
            stub=False,
        )

    def _clip_state(self, state: str) -> tuple[str, bool]:
        """Clip to what the window can hold once the options have taken their share.

        Laya truncates silently, which would drop the end of the prompt (often the goal)
        without telling anyone. Clipping here makes that visible in the counters instead.
        """

        window = self.config.max_len or 512
        budget = max(200, int(window * STATE_SHARE)) * CHARS_PER_TOKEN
        if len(state) <= budget:
            return state, False
        return state[:budget], True


def _translate(qid: str, question: dict[str, Any]) -> dict[str, Any]:
    """Jev question shape to Laya question shape.

    They agree on ``type`` and on a dictionary of options, so the only real work is the
    instruction, which Laya leans on and Jev questions often leave out.
    """

    out: dict[str, Any] = {"type": str(question["type"])}
    if question.get("criteria") is not None:
        out["criteria"] = question["criteria"]
    out["instructions"] = str(question.get("instructions") or _default_instruction(qid, question))
    return out


def _default_instruction(qid: str, question: dict[str, Any]) -> str:
    kind = str(question.get("type", ""))
    criteria = question.get("criteria")
    if kind == "noul" and isinstance(criteria, dict) and criteria.get("true"):
        return f"Is this true: {criteria['true']}?"
    if kind == "choice":
        return f"Which option is correct for {qid.replace('_', ' ')}?"
    if kind == "score":
        return f"How does this score on the rubric for {qid.replace('_', ' ')}?"
    return f"Answer {qid.replace('_', ' ')}."


def _normalise(answers: dict[str, Any]) -> dict[str, Any]:
    """Laya's answer dict is already close to Jev's; keep the keys the callers read."""

    out: dict[str, Any] = {}
    for qid, answer in answers.items():
        if not isinstance(answer, dict):
            continue
        kept: dict[str, Any] = {"type": answer.get("type", "")}
        for key in ("choice", "confidence", "probabilities", "noul", "score", "legend"):
            if key in answer:
                kept[key] = answer[key]
        out[str(qid)] = kept
    return out


def _patched_checkpoint(laya_mlx: Any, model: str, max_len: int, head_max_len: int) -> str:
    """A copy of the checkpoint whose runtime config raises the context limits.

    The limits come from ``rl_agent_config.json`` inside the checkpoint and the runtime
    takes no argument for them, so the only way to raise them is to load from a directory
    that holds an edited copy. The weights are symlinked rather than copied.
    """

    key = (model, max_len, head_max_len)
    if key in _PATCHED:
        return _PATCHED[key]

    try:
        source = Path(laya_mlx.agent.resolve_model(model))
    except Exception as exc:
        raise JevError(f"could not locate the Laya checkpoint {model!r}: {exc}") from exc

    config_path = source / "rl_agent_config.json"
    if not config_path.is_file():
        raise JevError(f"the checkpoint {model!r} has no rl_agent_config.json to raise")

    target = Path(tempfile.mkdtemp(prefix="jev-cascade-laya-"))
    for name in ("encoder", "tokenizer"):
        shutil.copytree(source / name, target / name, symlinks=False, dirs_exist_ok=True)
    for name in ("mlx_config.json", "rl_agent_config.json"):
        if (source / name).is_file():
            shutil.copy2(source / name, target / name)
    weights = source / "model.safetensors"
    if weights.is_file():
        (target / "model.safetensors").symlink_to(weights)

    settings = json.loads(config_path.read_text())
    settings["max_len"] = int(max_len)
    settings["head_max_len"] = int(head_max_len)
    (target / "rl_agent_config.json").write_text(json.dumps(settings))

    _PATCHED[key] = str(target)
    return str(target)
