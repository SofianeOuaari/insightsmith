"""Configuration from ``~/.insightsmith/config.toml`` and the environment.

The one rule worth stating loudly: ``local_only = true`` is a hard failure, not a
warning. If it is set and any role points at a provider that would send data off
the machine, loading the configuration raises. A privacy switch that merely warns
is not a privacy switch.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from insightsmith.errors import ConfigError

if sys.version_info >= (3, 11):  # pragma: no cover - trivial version shim
    import tomllib
else:  # pragma: no cover - trivial version shim
    import tomli as tomllib

__all__ = ["DEFAULT_CONFIG_PATH", "Budget", "Config", "load_config"]

DEFAULT_CONFIG_PATH: Final = Path.home() / ".insightsmith" / "config.toml"
#: Roles the router knows about. Config may name a subset.
ROLES: Final = ("planner", "coder", "cheap", "vision", "reasoner", "embed")

_DEFAULT_ROLES: Final[dict[str, str]] = {
    "planner": "ollama/qwen3:8b",
    "coder": "ollama/qwen3:8b",
    "cheap": "ollama/qwen3:8b",
    "reasoner": "ollama/qwen3:8b",
}


@dataclass(slots=True)
class Budget:
    max_usd_per_session: float = 0.50
    local_only: bool = False


@dataclass(slots=True)
class Config:
    roles: dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_ROLES))
    budget: Budget = field(default_factory=Budget)
    #: Extra or overriding base URLs for OpenAI-compatible backends.
    base_urls: dict[str, str] = field(default_factory=dict)
    path: Path | None = None

    def model_for(self, role: str) -> str | None:
        return self.roles.get(role)


def load_config(path: Path | None = None, *, environ: dict[str, str] | None = None) -> Config:
    """Load configuration, falling back to defaults when the file is absent.

    Raises:
        ConfigError: if the file is malformed, or if ``local_only`` is set while a
            role points at a provider that would send data off the machine.
    """
    env = os.environ if environ is None else environ
    target = path or Path(env.get("INSIGHTSMITH_CONFIG", DEFAULT_CONFIG_PATH))

    payload: dict[str, Any] = {}
    if target.is_file():
        try:
            payload = tomllib.loads(target.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, OSError) as exc:
            raise ConfigError(f"could not read {target}: {exc}") from exc

    roles = dict(_DEFAULT_ROLES)
    for role, model in (payload.get("roles") or {}).items():
        if not isinstance(model, str):
            raise ConfigError(f"roles.{role} must be a string, got {type(model).__name__}")
        roles[role] = model

    raw_budget = payload.get("budget") or {}
    budget = Budget(
        max_usd_per_session=float(raw_budget.get("max_usd_per_session", 0.50)),
        local_only=bool(raw_budget.get("local_only", False)),
    )
    if env.get("INSIGHTSMITH_LOCAL_ONLY", "").lower() in {"1", "true", "yes"}:
        budget.local_only = True

    config = Config(
        roles=roles,
        budget=budget,
        base_urls=dict(payload.get("base_urls") or {}),
        path=target if target.is_file() else None,
    )
    _enforce_local_only(config)
    return config


def _enforce_local_only(config: Config) -> None:
    """Hard-fail rather than quietly downgrading the guarantee."""
    if not config.budget.local_only:
        return
    # Imported here so config stays importable without the provider layer.
    from insightsmith.llm.registry import is_local_model

    offenders = sorted(
        f"{role} = {model}" for role, model in config.roles.items() if not is_local_model(model)
    )
    if offenders:
        raise ConfigError(
            "local_only is set but these roles use a remote provider: "
            + "; ".join(offenders)
            + ". Point them at a local provider or clear local_only."
        )
