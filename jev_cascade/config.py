"""Configuration for jev-cascade.

Resolution order: ``--config PATH`` > ``$JEVCASCADE_CONFIG`` > ``./config.toml``
> ``./config.example.toml``.

Secrets are never stored in the config file. Each tier names an environment
variable through ``api_key_env``, and values may interpolate other variables
with ``${VAR}`` or ``${VAR:-default}``.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_NAMES = ("config.toml", "config.example.toml")
DEFAULT_ENV_FILE = ".env"

_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(ValueError):
    """Raised when the configuration is missing or contradictory."""


def expand_vars(value: str) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}`` against the environment."""

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        found = os.environ.get(name)
        if found:
            return found
        return default if default is not None else ""

    return _VAR_RE.sub(replace, value)


def load_env_file(path: str | os.PathLike[str] | None = None) -> int:
    """Load KEY=VALUE lines from a .env file into the environment.

    Existing environment variables always win, so a real shell export is never
    silently overridden by a file. Returns the number of variables set.
    """

    target = Path(path).expanduser() if path else Path.cwd() / DEFAULT_ENV_FILE
    if not target.is_file():
        return 0
    loaded = 0
    for raw in target.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if name.startswith("export "):
            name = name[len("export ") :].strip()
        if not name or name in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[name] = value
        loaded += 1
    return loaded


@dataclass(frozen=True)
class Price:
    """List price per 1M tokens. Zero means "unknown", which is reported as such."""

    input: float = 0.0
    output: float = 0.0
    currency: str = "USD"

    @property
    def known(self) -> bool:
        return self.input > 0.0 or self.output > 0.0

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in / 1_000_000.0) * self.input + (tokens_out / 1_000_000.0) * self.output


@dataclass(frozen=True)
class TierConfig:
    """One model tier. ``kind`` is ``openai`` (any OpenAI-compatible endpoint) or ``mock``."""

    name: str
    model: str
    kind: str = "openai"
    base_url: str = ""
    api_key_env: str = ""
    description: str = ""
    price: Price = field(default_factory=Price)
    max_tokens: int = 2048
    temperature: float = 0.2
    timeout_s: float = 120.0
    judge_only: bool = False
    """A tier that exists to grade, never to run a step.

    Kept out of routing and out of the escalation ladder deliberately. An outside
    judge is worth having precisely because it is a different model family, and a
    different family is what the ladder is not shaped for: escalating a step into
    the judge pays judge prices for work the judge is not being measured on, and
    routing a step to it would let the judge produce one side of the pairs it is
    supposed to be grading.
    """

    def api_key(self) -> str:
        if not self.api_key_env:
            return ""
        return expand_vars(os.environ.get(self.api_key_env, ""))

    @property
    def key_present(self) -> bool:
        return bool(self.api_key())

    @property
    def usable(self) -> bool:
        if self.kind == "mock":
            return True
        return bool(self.base_url and self.model and self.key_present)


@dataclass(frozen=True)
class JevConfig:
    """System One settings. Jev is the router, the verifier, and the executor of decision steps."""

    kind: str = "http"  # "http" is the hosted API, "laya" is local MLX inference
    base_url: str = "https://api.typesafe.ai/v1/systemone"
    model: str = "jev-latest"
    api_key_env: str = "TYPESAFE_API_KEY"
    # Laya only, and off by default: raising these above the checkpoint's own limits is the
    # one measured way to feed it a bigger question set, but it is outside the published
    # validation. Both must be set together.
    max_len: int = 0
    head_max_len: int = 0
    price: Price = field(default_factory=lambda: Price(input=0.042, output=0.0))
    timeout_s: float = 90.0
    max_retries: int = 3
    max_state_chars: int = 12_000
    verify: bool = True
    escalate_below: float = 0.6
    candidates: int = 1
    max_escalations: int = 1
    same_tier_retries: int = 0  # retry on the same tier before paying for the next one up

    def api_key(self) -> str:
        return expand_vars(os.environ.get(self.api_key_env, ""))


@dataclass(frozen=True)
class AgentConfig:
    """Loop limits, planner mode, and the planner tier."""

    planner_tier: str = "cheap"
    planner_mode: str = "llm"  # llm | heuristic
    force_tier: str = ""  # non-empty: bypass Jev routing and pin every generative step here
    max_steps: int = 12
    planner_max_tokens: int = 2048  # reasoning models count their thinking against this
    max_cost_usd: float = 1.0
    budget_guard_ratio: float = 1.0  # above this share of the budget, stop escalating
    error_retries: int = 1  # retry the same tier this many times on a provider error
    task_timeout_s: float = 600.0
    ledger_path: str = ""
    verbose: bool = True


@dataclass(frozen=True)
class EvalConfig:
    """Settings for the split-quality eval: which judge, how many samples, which thresholds."""

    judge_tier: str = "expensive"
    strong_tier: str = ""  # empty: the strongest tier that can actually run a step
    task_set: str = "evals/tasks.toml"
    max_pairs: int = 24
    threshold_step: float = 0.1
    miss_tolerance: float = 0.05
    corrupt_ratio: float = 0.5
    seed: int = 1234
    max_tokens: int = 0  # 0 = use each tier's own cap, so the eval measures the tiers you run
    judge_max_tokens: int = 4096  # the judge reasons over two outputs before answering
    min_judged_pairs: int = 8  # below this, no threshold is recommended, only measured


@dataclass(frozen=True)
class BrowserConfig:
    """Settings for the browser tool, which reads a page and takes one action at a time.

    ``actor`` is the engine that chooses the action. It defaults to the hosted Jev on
    purpose: the local Laya engine scored 1/6 on situated action selection where Jev scored
    5/6 on real pages, so putting a config's classification roles on Laya does not move this
    one. Set it to "laya" to overrule that, knowing the number.
    """

    actor: str = "jev"  # "jev" (the [jev] engine as configured) or "laya" (local, measured weak here)
    max_steps: int = 12
    min_confidence: float = 0.35
    max_controls: int = 25
    node: str = "node"
    headed: bool = False
    playwright: str = ""  # module specifier for the bridge; empty = the bridge's own default
    writer_tier: str = ""  # tier that composes typed text; empty = words come from --text only


@dataclass(frozen=True)
class ComputerConfig:
    """Settings for the desktop tool, which reads the Accessibility tree and acts once per step.

    The same measured rule as the browser applies to ``actor``: action selection stays on
    the hosted Jev engine unless this config says otherwise. ``allowed_apps`` is the one
    guard a desktop needs that a page does not: a comma-separated list of app names the
    loop may act on (matched against the frontmost app's name or bundle id), empty for
    "any". Acting on a desktop is less reversible than clicking a web page.
    """

    actor: str = "jev"
    max_steps: int = 10
    min_confidence: float = 0.35
    max_controls: int = 25
    max_elements: int = 120  # AX-tree walk cap inside the Swift bridge
    swift: str = "swift"
    bridge: str = ""  # path to the Swift source; empty = the bundled tools/macos_bridge.swift
    writer_tier: str = ""
    allowed_apps: str = ""


@dataclass(frozen=True)
class Config:
    tiers: list[TierConfig]
    jev: JevConfig = field(default_factory=JevConfig)
    browser: BrowserConfig = field(default_factory=BrowserConfig)
    computer: ComputerConfig = field(default_factory=ComputerConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    source_path: str = ""

    @property
    def tier_names(self) -> list[str]:
        return [tier.name for tier in self.tiers]

    @property
    def executor_tiers(self) -> list[TierConfig]:
        """The tiers that can run a step, in escalation order.

        Judge-only tiers are not in here. Everything that picks an executor, walks
        the ladder, or quotes a fallback tier goes through this, so a tier marked
        ``judge_only`` cannot be routed to, escalated into, or used as the default.
        """

        return [tier for tier in self.tiers if not tier.judge_only]

    @property
    def executor_names(self) -> list[str]:
        return [tier.name for tier in self.executor_tiers]

    @property
    def strongest_executor(self) -> TierConfig:
        """The top of the escalation ladder: the strongest tier that runs steps."""

        tiers = self.executor_tiers
        if not tiers:
            raise ConfigError("no tier can run a step: every configured tier is judge_only")
        return tiers[-1]

    @property
    def currency(self) -> str:
        """The currency every price table is denominated in, for honest reporting."""

        return self.tiers[0].price.currency or "USD"

    def tier(self, name: str) -> TierConfig:
        for tier in self.tiers:
            if tier.name == name:
                return tier
        raise ConfigError(f"unknown tier {name!r}; configured tiers: {', '.join(self.tier_names)}")

    def stronger_than(self, name: str) -> TierConfig | None:
        """The next tier up the ladder of tiers that run steps, or None at the top.

        Walks ``executor_tiers`` rather than the declared order, so a judge-only
        tier declared at the end can never become the escalation target, and a
        judge-only tier declared in the middle is skipped over rather than landed on.
        """

        names = self.executor_names
        if name not in names:
            return None
        index = names.index(name)
        if index + 1 >= len(names):
            return None
        return self.executor_tiers[index + 1]

    def strong_tier(self, override: str = "") -> TierConfig:
        """The tier the eval treats as "the strong side", and the top of the ladder.

        Defaults to the strongest tier that runs steps, which is deliberately not
        ``tiers[-1]``: a judge-only tier is usually declared last, and taking it as
        the strong side would compare the cheap tier against the judge, then pay
        judge prices to grade a pair that contains the judge's own output.
        """

        if override:
            tier = self.tier(override)
            if tier.judge_only:
                raise ConfigError(
                    f"[eval] strong_tier {override!r} is judge_only, so it cannot be an executor"
                )
            return tier
        return self.strongest_executor

    def planner_tier(self) -> TierConfig:
        if self.agent.planner_tier in self.executor_names:
            return self.tier(self.agent.planner_tier)
        return self.executor_tiers[0]


def _as_dict(data: Any, where: str) -> dict[str, Any]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{where} must be a table, got {type(data).__name__}")
    return data


def _price(raw: Any, where: str) -> Price:
    table = _as_dict(raw, where)
    try:
        return Price(
            input=float(table.get("input", 0.0)),
            output=float(table.get("output", 0.0)),
            currency=str(table.get("currency", "USD")),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where} has non-numeric prices: {exc}") from exc


def _tier(raw: dict[str, Any], index: int) -> TierConfig:
    where = f"[[tiers]] #{index + 1}"
    name = str(raw.get("name", "")).strip()
    if not name:
        raise ConfigError(f"{where} is missing a name")
    kind = str(raw.get("kind", "openai")).strip() or "openai"
    if kind not in ("openai", "anthropic", "mock"):
        raise ConfigError(
            f"{where} ({name}) has unsupported kind {kind!r}; use 'openai', 'anthropic' or 'mock'"
        )
    try:
        max_tokens = int(raw.get("max_tokens", 2048))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where} ({name}) has a non-integer max_tokens") from exc
    judge_only = bool(raw.get("judge_only", False))
    return TierConfig(
        name=name,
        model=expand_vars(str(raw.get("model", ""))),
        kind=kind,
        base_url=expand_vars(str(raw.get("base_url", ""))).rstrip("/"),
        api_key_env=str(raw.get("api_key_env", "")),
        description=expand_vars(str(raw.get("description", ""))),
        price=_price(raw.get("price"), f"{where} ({name})"),
        max_tokens=max_tokens,
        temperature=float(raw.get("temperature", 0.2)),
        timeout_s=float(raw.get("timeout_s", 120.0)),
        judge_only=judge_only,
    )


def config_from_dict(data: dict[str, Any], source_path: str = "") -> Config:
    tier_rows = data.get("tiers") or []
    if not isinstance(tier_rows, list) or not tier_rows:
        raise ConfigError("at least one [[tiers]] entry is required")
    tiers = [_tier(_as_dict(row, f"[[tiers]] #{i + 1}"), i) for i, row in enumerate(tier_rows)]
    names = [tier.name for tier in tiers]
    if len(set(names)) != len(names):
        raise ConfigError(f"duplicate tier names in {names}")

    jraw = _as_dict(data.get("jev"), "[jev]")
    jpc = _as_dict(jraw.get("price"), "[jev] price")
    jev_kind = str(jraw.get("kind", JevConfig.kind)).strip().lower() or JevConfig.kind
    if jev_kind not in ("http", "laya"):
        raise ConfigError(
            f"[jev] kind {jev_kind!r} is not supported; use 'http' (hosted) or 'laya' (local MLX)"
        )
    default_model = "aac6fef/laya-mlx" if jev_kind == "laya" else JevConfig.model
    default_price = 0.0 if jev_kind == "laya" else 0.042
    jev = JevConfig(
        kind=jev_kind,
        base_url=expand_vars(str(jraw.get("base_url", JevConfig.base_url))).rstrip("/"),
        model=str(jraw.get("model", default_model)),
        api_key_env=str(jraw.get("api_key_env", JevConfig.api_key_env)),
        max_len=max(0, int(jraw.get("max_len", 0))),
        head_max_len=max(0, int(jraw.get("head_max_len", 0))),
        price=Price(
            input=float(jpc.get("input", default_price)),
            output=float(jpc.get("output", 0.0)),
            currency=str(jpc.get("currency", "USD")),
        ),
        timeout_s=float(jraw.get("timeout_s", 90.0)),
        max_retries=int(jraw.get("max_retries", 3)),
        max_state_chars=int(jraw.get("max_state_chars", 12_000)),
        verify=bool(jraw.get("verify", True)),
        escalate_below=float(jraw.get("escalate_below", 0.6)),
        candidates=max(1, int(jraw.get("candidates", 1))),
        max_escalations=max(0, int(jraw.get("max_escalations", 1))),
        same_tier_retries=max(0, int(jraw.get("same_tier_retries", 0))),
    )
    if not 0.0 <= jev.escalate_below <= 1.0:
        raise ConfigError("[jev] escalate_below must be between 0 and 1")
    if jev_kind == "laya":
        # The local engine has no key and no bill, so a missing key is not an error and a
        # nonzero price would be a lie in every ledger line.
        if bool(jev.max_len) != bool(jev.head_max_len):
            raise ConfigError(
                "[jev] max_len and head_max_len apply to Laya and must be set together"
            )
        if jev.max_len and (jev.max_len <= jev.head_max_len or jev.head_max_len <= 4):
            raise ConfigError(
                "[jev] needs 4 < head_max_len < max_len; the Laya runtime rejects anything else"
            )

    braw = _as_dict(data.get("browser"), "[browser]")
    browser_actor = str(braw.get("actor", "jev")).strip().lower() or "jev"
    if browser_actor not in ("jev", "laya"):
        raise ConfigError(
            f"[browser] actor {browser_actor!r} is not supported; use 'jev' or 'laya'"
        )
    browser = BrowserConfig(
        actor=browser_actor,
        max_steps=max(1, int(braw.get("max_steps", 12))),
        min_confidence=min(1.0, max(0.0, float(braw.get("min_confidence", 0.35)))),
        max_controls=max(1, int(braw.get("max_controls", 25))),
        node=str(braw.get("node", "node")),
        headed=bool(braw.get("headed", False)),
        playwright=str(braw.get("playwright", "")),
        writer_tier=str(braw.get("writer_tier", "")).strip(),
    )

    craw = _as_dict(data.get("computer"), "[computer]")
    computer_actor = str(craw.get("actor", "jev")).strip().lower() or "jev"
    if computer_actor not in ("jev", "laya"):
        raise ConfigError(
            f"[computer] actor {computer_actor!r} is not supported; use 'jev' or 'laya'"
        )
    computer = ComputerConfig(
        actor=computer_actor,
        max_steps=max(1, int(craw.get("max_steps", 10))),
        min_confidence=min(1.0, max(0.0, float(craw.get("min_confidence", 0.35)))),
        max_controls=max(1, int(craw.get("max_controls", 25))),
        max_elements=max(10, int(craw.get("max_elements", 120))),
        swift=str(craw.get("swift", "swift")),
        bridge=str(craw.get("bridge", "")),
        writer_tier=str(craw.get("writer_tier", "")).strip(),
        allowed_apps=str(craw.get("allowed_apps", "")).strip(),
    )

    araw = _as_dict(data.get("agent"), "[agent]")
    planner_mode = str(araw.get("planner_mode", "llm")).strip().lower() or "llm"
    if planner_mode not in ("llm", "heuristic"):
        raise ConfigError("[agent] planner_mode must be 'llm' or 'heuristic'")
    agent = AgentConfig(
        planner_tier=str(araw.get("planner_tier", "cheap")),
        planner_mode=planner_mode,
        force_tier=str(araw.get("force_tier", "")),
        max_steps=max(1, int(araw.get("max_steps", 12))),
        planner_max_tokens=max(256, int(araw.get("planner_max_tokens", 2048))),
        max_cost_usd=float(araw.get("max_cost_usd", 1.0)),
        budget_guard_ratio=float(araw.get("budget_guard_ratio", 1.0)),
        error_retries=max(0, int(araw.get("error_retries", 1))),
        task_timeout_s=float(araw.get("task_timeout_s", 600.0)),
        ledger_path=expand_vars(str(araw.get("ledger_path", ""))),
        verbose=bool(araw.get("verbose", True)),
    )
    executor_names = [tier.name for tier in tiers if not tier.judge_only]
    if not executor_names:
        raise ConfigError(
            "at least one tier must be able to run a step; every tier in this config is judge_only"
        )
    if tiers[0].judge_only:
        raise ConfigError(
            f"tier {tiers[0].name!r} is judge_only, so it cannot be the first tier: the first tier "
            f"is the cheap default an unrouted step falls back to"
        )
    if agent.planner_tier not in executor_names:
        raise ConfigError(
            f"[agent] planner_tier {agent.planner_tier!r} is not one of the tiers that can run a "
            f"step: {', '.join(executor_names)}"
        )
    if agent.force_tier and agent.force_tier not in executor_names:
        raise ConfigError(
            f"[agent] force_tier {agent.force_tier!r} is not one of the tiers that can run a step: "
            f"{', '.join(executor_names)}"
        )
    for section, writer_tier in (("[browser]", browser.writer_tier), ("[computer]", computer.writer_tier)):
        if writer_tier and writer_tier not in executor_names:
            raise ConfigError(
                f"{section} writer_tier {writer_tier!r} is not one of the tiers that can run a step: "
                f"{', '.join(executor_names)}"
            )

    eraw = _as_dict(data.get("eval"), "[eval]")
    eval_config = EvalConfig(
        # An unset judge defaults to the strongest configured tier, which is usually
        # the strong side of every pair (a model grading its own output, by design,
        # unless a judge-only tier is named here).
        judge_tier=str(eraw.get("judge_tier", "")).strip() or names[-1],
        strong_tier=str(eraw.get("strong_tier", "")).strip(),
        task_set=expand_vars(str(eraw.get("task_set", "evals/tasks.toml"))),
        max_pairs=max(1, int(eraw.get("max_pairs", 24))),
        threshold_step=float(eraw.get("threshold_step", 0.1)),
        miss_tolerance=float(eraw.get("miss_tolerance", 0.05)),
        corrupt_ratio=float(eraw.get("corrupt_ratio", 0.5)),
        seed=int(eraw.get("seed", 1234)),
        max_tokens=max(0, int(eraw.get("max_tokens", 0))),
        judge_max_tokens=max(128, int(eraw.get("judge_max_tokens", 4096))),
        min_judged_pairs=max(1, int(eraw.get("min_judged_pairs", 8))),
    )
    if eval_config.judge_tier not in names:
        raise ConfigError(
            f"[eval] judge_tier {eval_config.judge_tier!r} is not one of the configured tiers: {', '.join(names)}"
        )
    if eval_config.strong_tier and eval_config.strong_tier not in executor_names:
        raise ConfigError(
            f"[eval] strong_tier {eval_config.strong_tier!r} is not one of the tiers that can run a "
            f"step: {', '.join(executor_names)}"
        )
    if not 0.0 <= eval_config.corrupt_ratio < 1.0:
        raise ConfigError("[eval] corrupt_ratio must be in [0, 1)")
    if not 0.0 < eval_config.threshold_step <= 0.5:
        raise ConfigError("[eval] threshold_step must be in (0, 0.5]")
    if agent.budget_guard_ratio <= 0.0:
        raise ConfigError("[agent] budget_guard_ratio must be greater than 0")

    return Config(
        tiers=tiers,
        jev=jev,
        agent=agent,
        eval=eval_config,
        browser=browser,
        computer=computer,
        source_path=source_path,
    )


def find_config(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        return path
    env_path = os.environ.get("JEVCASCADE_CONFIG")
    if env_path:
        path = Path(env_path).expanduser()
        if not path.is_file():
            raise ConfigError(f"JEVCASCADE_CONFIG points at a missing file: {path}")
        return path
    here = Path(__file__).resolve().parent.parent
    for candidate in (Path.cwd(), here):
        for name in DEFAULT_CONFIG_NAMES:
            path = candidate / name
            if path.is_file():
                return path
    return None


def load_config(explicit: str | os.PathLike[str] | None = None, env_file: str | None = None) -> Config:
    if env_file != "":
        load_env_file(env_file)
    path = find_config(explicit)
    if path is None:
        raise ConfigError(
            "no config found; copy config.example.toml to config.toml or pass --config PATH"
        )
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return config_from_dict(data, source_path=str(path))
