"""Command line interface for jev-cascade."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

from .agent import CascadeAgent, result_to_json
from .config import Config, ConfigError, load_config
from .eval import report_to_json, run_eval
from .jev import build_jev
from .planner import build_planner, plan_to_json
from .providers import build_provider

VERSION = "0.1.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jev-cascade",
        description=(
            "A cheap-first agent loop: decompose a task into steps, let TypeSafe System One "
            "(Jev) route and verify each step, execute on the tier that earns it, and escalate "
            "only when a verification gate fails."
        ),
    )
    parser.add_argument("--config", help="path to a TOML config (default: ./config.toml)")
    parser.add_argument("--version", action="version", version=f"jev-cascade {VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="plan and execute a task")
    run.add_argument("task", help="the task text, or '-' to read it from stdin")
    run.add_argument("--dry-run", action="store_true", help="stub tiers and stub Jev, no network")
    run.add_argument("--json", action="store_true", help="emit the whole run as JSON")
    run.add_argument("--quiet", action="store_true", help="suppress the step-by-step trace")
    run.add_argument("--planner", choices=("llm", "heuristic"), help="override the planner mode")
    run.add_argument("--tier", help="force every generative step onto this tier (skips Jev routing)")
    run.add_argument(
        "--max-cost", type=float, help="hard budget for this run, in the config's price currency"
    )
    run.add_argument("--candidates", type=int, help="cheap drafts for Jev to choose from")
    run.add_argument("--no-verify", action="store_true", help="skip Jev verification and escalation")
    run.add_argument("--ledger", help="append run events to this JSONL file")

    plan = sub.add_parser("plan", help="show the decomposition without executing it")
    plan.add_argument("task", help="the task text, or '-' to read it from stdin")
    plan.add_argument("--dry-run", action="store_true", help="use the offline heuristic planner")
    plan.add_argument("--planner", choices=("llm", "heuristic"), help="override the planner mode")

    evaluate = sub.add_parser(
        "eval",
        help="measure the split: pair cheap vs strong outputs per step and sweep the gate threshold",
    )
    evaluate.add_argument("--tasks", help="task set TOML (default: [eval] task_set)")
    evaluate.add_argument("--pairs", type=int, help="stop after this many step pairs")
    evaluate.add_argument(
        "--judge",
        metavar="TIER",
        help="tier used to grade each pair (default: [eval] judge_tier); "
        "point at a judge_only tier of another family to remove self-grading bias",
    )
    evaluate.add_argument(
        "--strong",
        metavar="TIER",
        help="tier treated as the strong side of each pair (default: [eval] strong_tier, "
        "else the strongest tier that can run a step)",
    )
    evaluate.add_argument("--json", action="store_true", help="emit the report as JSON")
    evaluate.add_argument("--out", help="also write the JSON report to this path")
    evaluate.add_argument("--dry-run", action="store_true", help="stub tiers and stub Jev, no network")
    evaluate.add_argument("--quiet", action="store_true", help="suppress per-pair progress lines")

    browse = sub.add_parser(
        "browse",
        help="drive a browser toward a goal, one typed decision per step (cheap computer use)",
        description=(
            "Read the page's own text and controls, ask the decision engine for one action, "
            "take it, repeat. Nothing is clicked or typed unless --act is passed."
        ),
    )
    browse.add_argument("goal", help="what to achieve in the browser")
    browse.add_argument("--url", default="about:blank", help="page to start on")
    browse.add_argument(
        "--text", default="", help="the string to type when the chosen action needs one"
    )
    browse.add_argument(
        "--act", action="store_true", help="actually act; without it, one step is reported and stopped"
    )
    browse.add_argument("--steps", type=int, help="override [browser] max_steps")
    browse.add_argument("--min-confidence", type=float, help="stop below this confidence")
    browse.add_argument("--headed", action="store_true", help="show the browser window")
    browse.add_argument(
        "--writer",
        metavar="TIER",
        help="tier that composes typed text, gated by Jev (default: [browser] writer_tier; "
        "without a writer, the only words typed are the ones passed in --text)",
    )
    browse.add_argument("--json", action="store_true", help="emit the run as JSON")
    browse.add_argument("--dry-run", action="store_true", help="stub the decision engine; no keys")
    browse.add_argument("--ledger", help="append the step trace to this JSONL file")
    browse.add_argument("--quiet", action="store_true", help="suppress the step trace")

    computer = sub.add_parser(
        "computer",
        help="drive the macOS desktop toward a goal, one typed decision per step",
        description=(
            "Read the frontmost app's Accessibility tree, ask the decision engine for one "
            "action, take it, repeat. Nothing is clicked or typed unless --act is passed. "
            "Needs the Accessibility permission; --check-permissions asks macOS to show it."
        ),
    )
    computer.add_argument(
        "goal", nargs="?", default="", help="what to achieve on the desktop (required unless --check-permissions)"
    )
    computer.add_argument("--app", default="", help="app to bring to the front before the first step")
    computer.add_argument(
        "--text", default="", help="the string to type when the chosen action needs one"
    )
    computer.add_argument(
        "--act", action="store_true", help="actually act; without it, one step is reported and stopped"
    )
    computer.add_argument("--steps", type=int, help="override [computer] max_steps")
    computer.add_argument("--min-confidence", type=float, help="stop below this confidence")
    computer.add_argument(
        "--writer",
        metavar="TIER",
        help="tier that composes typed text, gated by Jev (default: [computer] writer_tier)",
    )
    computer.add_argument("--json", action="store_true", help="emit the run as JSON")
    computer.add_argument("--dry-run", action="store_true", help="stub the decision engine; no keys")
    computer.add_argument(
        "--check-permissions",
        action="store_true",
        help="ask macOS about the Accessibility grant (shows the dialog) and exit",
    )
    computer.add_argument("--ledger", help="append the step trace to this JSONL file")
    computer.add_argument("--quiet", action="store_true", help="suppress the step trace")

    sub.add_parser(
        "serve",
        help="speak MCP over stdio so an agent client (ZCode, Claude Code, ...) can call the cascade as tools",
        description=(
            "Runs until the client closes stdin. Register it with your client as "
            "command=python3, args=[-m, jev_cascade, serve]."
        ),
    )
    sub.add_parser("check", help="validate the config and report which keys are present")
    sub.add_parser("selftest", help="run the offline test suite (no network, no keys)")

    demo = sub.add_parser("demo", help="run a bundled dry-run demo with no keys and no network")
    demo.add_argument("--json", action="store_true", help="emit the run as JSON")

    return parser


def _task_text(raw: str) -> str:
    if raw != "-":
        return raw
    data = sys.stdin.read().strip()
    if not data:
        raise SystemExit("no task text on stdin")
    return data


def _apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    agent = config.agent
    changes: dict[str, object] = {}
    if getattr(args, "planner", None):
        changes["planner_mode"] = args.planner
    if getattr(args, "max_cost", None) is not None:
        changes["max_cost_usd"] = args.max_cost
    if getattr(args, "ledger", None):
        changes["ledger_path"] = args.ledger
    if getattr(args, "quiet", False):
        changes["verbose"] = False
    if changes:
        agent = dataclasses.replace(agent, **changes)

    jev = config.jev
    jev_changes: dict[str, object] = {}
    if getattr(args, "candidates", None) is not None:
        jev_changes["candidates"] = max(1, args.candidates)
    if getattr(args, "no_verify", False):
        jev_changes["verify"] = False
    if jev_changes:
        jev = dataclasses.replace(jev, **jev_changes)

    tiers = config.tiers
    if getattr(args, "tier", None):
        # Only tiers that can run a step: pinning every step to a judge-only tier
        # would make the judge produce the very work it is supposed to grade.
        if args.tier not in config.executor_names:
            raise SystemExit(
                f"--tier {args.tier!r} cannot run a step; available: {', '.join(config.executor_names)}"
            )
        agent = dataclasses.replace(agent, force_tier=args.tier)

    return dataclasses.replace(config, tiers=tiers, jev=jev, agent=agent)


def _cmd_run(config: Config, args: argparse.Namespace) -> int:
    task = _task_text(args.task)
    dry_run = bool(args.dry_run or json_hint())
    if not dry_run and config.source_path.endswith("config.example.toml"):
        print(
            "note: running against config.example.toml; copy it to config.toml and set your own "
            "tiers and prices first (or pass --dry-run to exercise the loop offline).",
            file=sys.stderr,
        )
    quiet = args.quiet or args.json
    agent = CascadeAgent(
        config,
        dry_run=dry_run,
        on_event=(lambda message: None) if quiet else (lambda message: print(message, flush=True)),
    )
    result = agent.run(task)

    if args.json:
        print(result_to_json(result, config, stub=agent.stubbed))
    else:
        if result.outputs:
            print("")
            print("outputs")
            print("-------")
            for step_id, text in result.outputs.items():
                print(f"[{step_id}]")
                print(text.rstrip())
                print("")
        if result.ledger is not None:
            print(result.ledger.render_summary(config, stub=agent.stubbed))
        if result.aborted:
            print(f"\naborted: {result.aborted}")
        if result.error:
            print(f"\nerror: {result.error}")
    agent.ledger.close()
    return 2 if result.error else 0


def json_hint() -> bool:
    """Dry runs are opt-in; this hook exists so a future env var can force them."""

    return os.environ.get("JEVCASCADE_DRY_RUN", "").lower() in ("1", "true", "yes")


def _cmd_plan(config: Config, args: argparse.Namespace) -> int:
    task = _task_text(args.task)
    dry_run = bool(args.dry_run or json_hint())
    providers = {tier.name: build_provider(tier, dry_run=dry_run) for tier in config.tiers}
    planner = build_planner(config, providers)
    plan = planner.plan(task, dry_run=dry_run)
    print(plan_to_json(plan))
    return 0


def _cmd_eval(config: Config, args: argparse.Namespace) -> int:
    if getattr(args, "judge", None):
        if args.judge not in config.tier_names:
            raise SystemExit(
                f"--judge {args.judge!r} is not configured; available: {', '.join(config.tier_names)}"
            )
        config = dataclasses.replace(
            config, eval=dataclasses.replace(config.eval, judge_tier=args.judge)
        )
    if getattr(args, "strong", None):
        # strong_tier() validates the name and rejects a judge_only tier, so a --strong
        # that cannot run a step fails here rather than silently comparing the wrong two.
        try:
            config.strong_tier(args.strong)
        except ConfigError as exc:
            raise SystemExit(str(exc)) from exc
        config = dataclasses.replace(
            config, eval=dataclasses.replace(config.eval, strong_tier=args.strong)
        )
    if not args.json and not args.quiet:
        print(
            f"pairing {config.executor_names[0]} against "
            f"{config.strong_tier(config.eval.strong_tier).name}, "
            f"judged by {config.eval.judge_tier}",
            flush=True,
        )
    report = run_eval(
        config,
        task_set=args.tasks,
        dry_run=bool(args.dry_run or json_hint()),
        max_pairs=args.pairs,
        on_event=None if not args.quiet else (lambda _message: None),
    )
    if args.out:
        Path(args.out).expanduser().write_text(report_to_json(report) + "\n")
    if args.json:
        print(report_to_json(report))
    else:
        print("")
        print(report.render(config))
    return 0


def _cmd_browse(config: Config, args: argparse.Namespace) -> int:
    from .browser import BridgeError, run_browser_task  # noqa: PLC0415 (Node side only when used)

    browser = config.browser
    if args.steps is not None:
        browser = dataclasses.replace(browser, max_steps=max(1, args.steps))
    if args.min_confidence is not None:
        browser = dataclasses.replace(browser, min_confidence=args.min_confidence)
    if args.headed:
        browser = dataclasses.replace(browser, headed=True)
    if getattr(args, "writer", None):
        browser = dataclasses.replace(browser, writer_tier=args.writer)
    agent_overrides: dict[str, object] = {}
    if args.ledger:
        agent_overrides["ledger_path"] = args.ledger
    if args.quiet or args.json:
        agent_overrides["verbose"] = False
    agent = dataclasses.replace(config.agent, **agent_overrides) if agent_overrides else config.agent
    config = dataclasses.replace(config, browser=browser, agent=agent)

    stub = bool(args.dry_run or json_hint())
    actor = build_jev(config.jev, dry_run=stub) if stub else None
    if not stub and config.jev.kind == "http" and not config.jev.api_key():
        print(
            f"browse needs a decision engine: set ${config.jev.api_key_env}, or pass --dry-run "
            "to exercise the loop with a scripted stub, or set [jev] kind = \"laya\" for the "
            "local one (not recommended for action selection).",
            file=sys.stderr,
        )
        return 1
    quiet = args.quiet or args.json
    try:
        result = run_browser_task(
            args.goal,
            config,
            actor=actor,
            url=args.url,
            text=args.text,
            act=bool(args.act),
            max_steps=browser.max_steps,
            min_confidence=browser.min_confidence,
            max_controls=browser.max_controls,
            writer_tier=browser.writer_tier,
            on_event=(lambda message: None) if quiet else (lambda message: print(message, flush=True)),
        )
    except (BridgeError, ValueError) as exc:
        print(f"browser error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(browser_result_to_json(result), indent=2))
    else:
        print("")
        print(result.render())
    return 0


def browser_result_to_json(result) -> dict[str, object]:
    return {
        "goal": result.goal,
        "act": result.acted,
        "stop_reason": result.stop_reason,
        "stub": result.stub,
        "jev_cost_usd": round(result.jev_cost_usd, 6),
        "steps": result.steps,
        "final": {
            "url": result.final_state.get("url", ""),
            "title": result.final_state.get("title", ""),
            "lines": (result.final_state.get("lines") or [])[:12],
        },
    }


def _cmd_computer(config: Config, args: argparse.Namespace) -> int:
    from .computer import (  # noqa: PLC0415 (Swift side only when used)
        BridgeError,
        allowed_apps,
        check_permissions,
        run_computer_task,
    )

    cfg = config.computer
    if args.check_permissions:
        verdict = check_permissions(cfg)
        if verdict["trusted"]:
            print("Accessibility permission: granted. The desktop tool can read and act.")
            return 0
        print("Accessibility permission: missing.")
        print(f"  {verdict['hint']}")
        return 1
    if not args.goal.strip():
        raise SystemExit("computer needs a goal (or --check-permissions)")

    if args.steps is not None:
        cfg = dataclasses.replace(cfg, max_steps=max(1, args.steps))
    if args.min_confidence is not None:
        cfg = dataclasses.replace(cfg, min_confidence=args.min_confidence)
    if getattr(args, "writer", None):
        cfg = dataclasses.replace(cfg, writer_tier=args.writer)
    config = dataclasses.replace(config, computer=cfg)

    stub = bool(args.dry_run or json_hint())
    actor = build_jev(config.jev, dry_run=stub) if stub else None
    if not stub and config.jev.kind == "http" and not config.jev.api_key():
        print(
            f"computer needs a decision engine: set ${config.jev.api_key_env}, or pass --dry-run "
            "to exercise the loop with a scripted stub, or set [jev] kind = \"laya\" for the "
            "local one (not recommended for action selection).",
            file=sys.stderr,
        )
        return 1
    quiet = args.quiet or args.json
    try:
        result = run_computer_task(
            args.goal,
            config,
            actor=actor,
            text=args.text,
            act=bool(args.act),
            max_steps=cfg.max_steps,
            min_confidence=cfg.min_confidence,
            max_controls=cfg.max_controls,
            writer_tier=cfg.writer_tier,
            app=args.app,
            allowed=allowed_apps(cfg),
            on_event=(lambda message: None) if quiet else (lambda message: print(message, flush=True)),
        )
    except (BridgeError, ValueError) as exc:
        print(f"computer error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(computer_result_to_json(result), indent=2))
    else:
        print("")
        print(result.render())
    return 0


def computer_result_to_json(result) -> dict[str, object]:
    final = result.final_state or {}
    return {
        "goal": result.goal,
        "act": result.acted,
        "stop_reason": result.stop_reason,
        "stub": result.stub,
        "jev_cost_usd": round(result.jev_cost_usd, 6),
        "steps": result.steps,
        "final": {
            "app": final.get("app", ""),
            "window": final.get("window", ""),
            "lines": (final.get("lines") or [])[:12],
        },
    }


def _cmd_check(config: Config) -> int:
    print(f"config      {config.source_path}")
    print(f"planner     tier={config.agent.planner_tier} mode={config.agent.planner_mode}")
    print(f"jev         {config.jev.model} key=${config.jev.api_key_env} "
          f"({'present' if config.jev.api_key() else 'MISSING'})")
    print(f"jev policy  verify={config.jev.verify} escalate_below={config.jev.escalate_below} "
          f"candidates={config.jev.candidates} retries={config.jev.same_tier_retries}")
    print(f"eval        judge_tier={config.eval.judge_tier} "
          f"strong_tier={config.strong_tier(config.eval.strong_tier).name} "
          f"tasks={config.eval.task_set} tolerance={config.eval.miss_tolerance:.0%}")
    print(f"budget      max_cost_usd={config.agent.max_cost_usd} "
          f"guard_ratio={config.agent.budget_guard_ratio}")
    print(f"limits      max_steps={config.agent.max_steps} max_cost_usd={config.agent.max_cost_usd}")
    print("tiers:")
    for tier in config.tiers:
        state = "ready" if tier.usable else "needs a key" if tier.kind != "mock" else "mock"
        role = "  (judge only)" if tier.judge_only else ""
        print(
            f"  {tier.name:<10} {tier.kind:<7} {tier.model or '-':<28} "
            f"{tier.price.input:g}/{tier.price.output:g} {config.currency} per 1M  {state}{role}"
        )
    print(f"ladder      {' -> '.join(config.executor_names)}")
    return 0


def _cmd_selftest() -> int:
    root = Path(__file__).resolve().parent.parent
    return subprocess.call([sys.executable, "-m", "unittest", "discover", "-s", str(root / "tests"), "-t", str(root), "-v"])


DEMO_TASK = (
    "Classify the severity of this bug report, and then write the one paragraph reply we "
    "send back to the reporter. The report: the export button does nothing when the project "
    "has more than 5000 rows."
)


def _cmd_demo(args: argparse.Namespace) -> int:
    config_path = Path(__file__).resolve().parent.parent / "config.example.toml"
    config = load_config(config_path)
    config = dataclasses.replace(config, agent=dataclasses.replace(config.agent, verbose=False))
    agent = CascadeAgent(
        config,
        dry_run=True,
        on_event=(lambda message: None) if args.json else (lambda message: print(message, flush=True)),
    )
    result = agent.run(DEMO_TASK)
    if args.json:
        print(result_to_json(result, config, stub=True))
    else:
        print("")
        print("outputs")
        print("-------")
        for step_id, text in result.outputs.items():
            print(f"[{step_id}]")
            print(text.rstrip())
            print("")
        print(result.ledger.render_summary(config, stub=True))
    agent.ledger.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "selftest":
        return _cmd_selftest()
    if args.command == "demo":
        return _cmd_demo(args)
    if args.command == "serve":
        from .mcp_server import McpServer, build_tools, serve_stdio  # noqa: PLC0415

        serve_stdio(McpServer(build_tools()))
        return 0

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1

    if args.command == "check":
        return _cmd_check(config)
    if args.command == "eval":
        return _cmd_eval(config, args)
    if args.command == "browse":
        return _cmd_browse(config, args)
    if args.command == "computer":
        return _cmd_computer(config, args)
    config = _apply_overrides(config, args)
    if args.command == "run":
        return _cmd_run(config, args)
    if args.command == "plan":
        return _cmd_plan(config, args)
    parser.error(f"unknown command {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
