"""The writer: a cheap tier composes the text a step wants to type, Jev gates it.

Neither decision engine can generate text — that is the one thing System One is
never asked for — but the cascade already carries tiers that can. The writer is
the smallest possible use of one: when a chosen action needs a string (a search
query, a short message), the named tier writes it, a Noul gate checks the draft
against the goal before anything is typed, and the ledger records who wrote
what. One retry, then an honest refusal: the loop stops that step rather than
typing something unverified. Where no writer tier is configured, the words come
from ``--text`` or the step does not happen, so the loop never quietly invents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .config import TierConfig
from .jev import Jev, JevError, noul_question
from .ledger import Ledger, StepRecord
from .providers import Provider, ProviderError, build_provider

WRITER_SYSTEM = (
    "You write the exact string to type into one field of an interface. "
    "Answer with the string and nothing else: no quotes, no explanation, "
    "no leading label, one line. Keep it as short as the goal allows."
)
# Deliberately lower than the agent's measured escalate_below: this gate answers
# a much easier question (would typing this serve the goal here?) and a draft
# that fails it simply does not get typed.
WRITER_GATE_THRESHOLD = 0.6


@dataclass
class WriterDraft:
    text: str = ""
    tier: str = ""
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    verify_p: float | None = None
    attempts: int = 0
    error: str = ""
    stub: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.text)


def resolve_writer_tier(name: str, config) -> TierConfig:
    """The tier that will compose typed text, validated before a loop starts."""

    for tier in config.tiers:
        if tier.name == name:
            if tier.judge_only:
                raise ValueError(
                    f"writer tier {name!r} is judge_only: the judge grades work, it never writes it"
                )
            return tier
    raise ValueError(f"writer tier {name!r} is not a configured tier")


def clean_draft(raw: str) -> str:
    """The string, not the prose around it: one line, no wrapping quotes."""

    text = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'“”":
        text = text[1:-1].strip()
    return text


def compose_text(
    goal: str,
    target: str,
    context: str,
    tier: TierConfig,
    jev: Jev,
    provider: Provider | None = None,
    threshold: float = WRITER_GATE_THRESHOLD,
) -> WriterDraft:
    """Write one string, gate it, retry once, and give up honestly."""

    draft = WriterDraft(tier=tier.name)
    writer = provider or build_provider(tier)
    prompt = (
        f"goal: {goal}\n"
        f"field: {target}\n"
        + (f"what the interface shows:\n{context}\n" if context.strip() else "")
        + "Write exactly the string to put in this field."
    )
    for attempt in (1, 2):
        draft.attempts = attempt
        try:
            completion = writer.complete(WRITER_SYSTEM, prompt, max_tokens=256)
        except ProviderError as exc:
            draft.error = str(exc)
            continue
        draft.model = completion.model
        draft.tokens_in += completion.tokens_in
        draft.tokens_out += completion.tokens_out
        draft.cost_usd += completion.cost_usd
        draft.latency_s += completion.latency_s
        draft.stub = draft.stub or completion.stub
        text = clean_draft(completion.text)
        if not text:
            draft.error = "the writer returned nothing usable"
            continue
        try:
            decision = jev.ask(
                f"goal: {goal}\nfield: {target}\ndraft: {text}",
                {
                    "fit": noul_question(
                        "the string is exactly what belongs in this field to serve the goal",
                        "the string is empty, off-target, or not what this field needs",
                        "Would typing this string into this field move toward the goal?",
                    )
                },
            )
        except JevError as exc:
            draft.error = f"the gate failed: {exc}"
            return draft
        draft.verify_p = decision.noul("fit")
        draft.error = ""
        if draft.verify_p >= threshold:
            draft.text = text
            return draft
        draft.error = f"the gate scored the draft {draft.verify_p:.2f} < {threshold:.2f}"
    return draft


def compose_for_field(
    goal: str,
    state: dict[str, Any],
    control: dict[str, Any],
    writer: tuple[TierConfig, Provider],
    jev: Jev,
    ledger: Ledger,
    step_index: int,
    tool: str,
    emit: Callable[[str], None] | None = None,
) -> WriterDraft:
    """Compose the text for one chosen field control, and book the attempt.

    Shared by the browser and desktop loops: one write, one ledger line for the
    event and one StepRecord for the spend, so the by-tier summary shows the
    writer exactly like any other tier.
    """

    label = str(control.get("label", ""))
    role = str(control.get("role", "field"))
    tier, provider = writer
    draft = compose_text(
        goal=goal,
        target=f"the {role} '{label}'",
        context="\n".join(str(line) for line in (state.get("lines") or [])[:12]),
        tier=tier,
        jev=jev,
        provider=provider,
    )
    ledger.record(
        {
            "type": "write",
            "tool": tool,
            "step": step_index,
            "tier": draft.tier,
            "field": label,
            "verify_p": draft.verify_p,
            "attempts": draft.attempts,
            "chars": len(draft.text),
            "status": "done" if draft.ok else "refused",
            **({"error": draft.error} if draft.error and not draft.ok else {}),
        }
    )
    ledger.add_step(
        StepRecord(
            step_id=f"{tool}-write-{step_index}",
            goal=goal,
            kind="write",
            tier=draft.tier,
            model=draft.model,
            tokens_in=draft.tokens_in,
            tokens_out=draft.tokens_out,
            cost_usd=draft.cost_usd,
            latency_s=draft.latency_s,
            stub=draft.stub,
            verify_p=draft.verify_p,
            status="done" if draft.ok else "refused",
            output_chars=len(draft.text),
        )
    )
    if emit is not None and draft.ok:
        verdict = "-" if draft.verify_p is None else f"p={draft.verify_p:.2f}"
        emit(f"    writer {draft.tier} ({verdict}): {draft.text[:60]!r}")
    return draft
