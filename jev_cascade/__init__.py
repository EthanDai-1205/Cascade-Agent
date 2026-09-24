"""jev-cascade: a cheap-first agent loop with TypeSafe System One as the router."""

from .agent import CascadeAgent, RunResult, run_task
from .config import Config, ConfigError, load_config
from .jev import JevClient, JevError, StubJev, build_jev
from .ledger import Ledger, StepRecord
from .planner import Plan, Step, parse_plan
from .providers import Completion, MockProvider, OpenAICompatibleProvider, ProviderError

__all__ = [
    "CascadeAgent",
    "Completion",
    "Config",
    "ConfigError",
    "JevClient",
    "JevError",
    "Ledger",
    "MockProvider",
    "OpenAICompatibleProvider",
    "Plan",
    "ProviderError",
    "RunResult",
    "Step",
    "StepRecord",
    "StubJev",
    "build_jev",
    "load_config",
    "parse_plan",
    "run_task",
]

__version__ = "0.1.0"
