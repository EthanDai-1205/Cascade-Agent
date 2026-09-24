"""Run ledger: what each step cost, which tier ran it, and why it escalated.

Every event is appended to JSONL so a run can be audited afterwards. The
summary deliberately separates facts from counterfactuals: what was actually
spent, versus what the same token counts would have cost on another tier
(re-pricing, not a prediction of what a different model would emit).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config


@dataclass
class StepRecord:
    step_id: str
    goal: str
    kind: str
    tier: str
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    jev_calls: int = 0
    jev_cost_usd: float = 0.0
    jev_tokens_in: int = 0
    jev_tokens_out: int = 0
    verify_p: float | None = None
    verify_p_initial: float | None = None
    route_confidence: float | None = None
    routed_by: str = ""
    escalations: int = 0
    retries: int = 0
    error_retries: int = 0
    escalated_from: str = ""
    candidates: int = 1
    latency_s: float = 0.0
    stub: bool = False
    status: str = "done"
    output_chars: int = 0


@dataclass
class Ledger:
    path: str = ""
    records: list[StepRecord] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self._handle = None
        if self.path:
            target = Path(self.path).expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
            self._handle = target.open("a", encoding="utf-8")

    def record(self, event: dict[str, Any]) -> None:
        event = {"ts": round(time.time(), 3), **event}
        self.events.append(event)
        if self._handle is not None:
            self._handle.write(json.dumps(event, sort_keys=True) + "\n")
            self._handle.flush()

    def add_step(self, record: StepRecord) -> StepRecord:
        self.records.append(record)
        self.record({"type": "step", **record.__dict__})
        return record

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    @property
    def model_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def jev_cost_usd(self) -> float:
        return sum(r.jev_cost_usd for r in self.records)

    @property
    def total_cost_usd(self) -> float:
        return self.model_cost_usd + self.jev_cost_usd

    @property
    def wall_s(self) -> float:
        return time.monotonic() - self.started_at

    def by_tier(self) -> dict[str, dict[str, float]]:
        totals: dict[str, dict[str, float]] = {}
        for record in self.records:
            bucket = totals.setdefault(
                record.tier, {"steps": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
            )
            bucket["steps"] += 1
            if record.tier == "jev":
                # A decision step spends System One tokens, not model tokens.
                bucket["tokens_in"] += record.jev_tokens_in
                bucket["tokens_out"] += record.jev_tokens_out
                bucket["cost_usd"] += record.jev_cost_usd
            else:
                bucket["tokens_in"] += record.tokens_in
                bucket["tokens_out"] += record.tokens_out
                bucket["cost_usd"] += record.cost_usd
        return totals

    def repriced(self, config: Config) -> dict[str, float]:
        """What the same tokens would have cost on each tier that can run a step.

        Judge-only tiers are excluded, even though they have a price: there is no
        configuration in which this run's step tokens would have been written by the
        judge, so quoting "all-judge" would offer a counterfactual nobody can choose.
        """

        out: dict[str, float] = {}
        for tier in config.executor_tiers:
            total = 0.0
            for record in self.records:
                if record.stub:
                    continue
                total += tier.price.cost(record.tokens_in, record.tokens_out)
            out[tier.name] = total
        return out

    def summary(self, config: Config, stub: bool) -> dict[str, Any]:
        return {
            "steps": len(self.records),
            "wall_s": round(self.wall_s, 2),
            "model_cost_usd": round(self.model_cost_usd, 6),
            "jev_cost_usd": round(self.jev_cost_usd, 6),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "by_tier": self.by_tier(),
            "repriced_usd": {name: round(value, 6) for name, value in self.repriced(config).items()},
            "escalations": sum(r.escalations for r in self.records),
            "retries": sum(r.retries for r in self.records),
            "error_retries": sum(r.error_retries for r in self.records),
            "jev_calls": sum(r.jev_calls for r in self.records),
            "skipped": sum(1 for r in self.records if r.status == "skipped"),
            "stub": stub,
        }

    def render_summary(self, config: Config, stub: bool) -> str:
        summary = self.summary(config, stub)
        lines: list[str] = []
        if stub:
            lines.append("NOTE: this run used stub tiers and/or stub Jev. No real model answered.")
        cur = config.currency
        lines.append(
            f"steps {summary['steps']}  wall {summary['wall_s']}s  "
            f"model {summary['model_cost_usd']:.4f} {cur}  jev {summary['jev_cost_usd']:.4f} {cur}  "
            f"total {summary['total_cost_usd']:.4f} {cur}"
        )
        lines.append("")
        header = f"{'tier':<12}{'steps':>6}{'tok in':>10}{'tok out':>10}{'cost':>12}"
        lines.append(header)
        lines.append("-" * len(header))
        for name, bucket in sorted(self.by_tier().items()):
            lines.append(
                f"{name:<12}{int(bucket['steps']):>6}{int(bucket['tokens_in']):>10}"
                f"{int(bucket['tokens_out']):>10}{bucket['cost_usd']:>12.4f}"
            )
        repriced = summary["repriced_usd"]
        if repriced and any(value > 0 for value in repriced.values()):
            lines.append("")
            lines.append(f"re-pricing the same token counts at each tier's list price ({cur}):")
            for name, value in sorted(repriced.items()):
                lines.append(f"  all-{name}: {value:.4f}")
        lines.append("")
        lines.append(
            f"jev calls {summary['jev_calls']}  escalations {summary['escalations']}  "
            f"same-tier retries {summary['retries']}  error retries {summary['error_retries']}  "
            f"skipped {summary['skipped']}"
        )
        for record in self.records:
            verdict = "n/a" if record.verify_p is None else f"p={record.verify_p:.2f}"
            route = f"{record.routed_by}" if record.routed_by else "-"
            lines.append(
                f"  [{record.step_id}] {record.tier:<10} {record.kind:<9} {verdict:<8} "
                f"router={route:<10} esc={record.escalations} ret={record.retries} {record.status}"
            )
        return "\n".join(lines)
