"""Configuration from ``~/.insightsmith/config.toml`` and the environment.

The one rule worth stating loudly: ``local_only = true`` is a hard failure, not a
warning. If it is set and any role points at a provider that would send data off
the machine, loading the configuration raises. A privacy switch that merely warns
is not a privacy switch.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from insightsmith.engine import Engine
from insightsmith.errors import ConfigError

if sys.version_info >= (3, 11):  # pragma: no cover - trivial version shim
    import tomllib
else:  # pragma: no cover - trivial version shim
    import tomli as tomllib

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "Budget",
    "Config",
    "config_path",
    "ensure_config",
    "load_config",
]

DEFAULT_CONFIG_PATH: Final = Path.home() / ".insightsmith" / "config.toml"
#: Roles the router knows about. Config may name a subset.
ROLES: Final = ("planner", "coder", "critic", "cheap", "vision", "reasoner", "embed")

_DEFAULT_ROLES: Final[dict[str, str]] = {
    "planner": "ollama/qwen3:8b",
    "coder": "ollama/qwen3:8b",
    "critic": "ollama/qwen3:8b",
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
    #: Which dataframe API generated code is written against.
    engine: Engine = Engine.POLARS
    #: Extra or overriding base URLs for OpenAI-compatible backends.
    base_urls: dict[str, str] = field(default_factory=dict)
    path: Path | None = None

    def model_for(self, role: str) -> str | None:
        return self.roles.get(role)


#: Written on first run. Commented rather than bare, because the file exists to
#: be edited and a reader should not have to find the docs to know what may go
#: in it. The values are the defaults, so writing it changes nothing by itself.
_TEMPLATE: Final = """\
# insightsmith configuration.
#
# Every role may name a different model. Routing a cheap question and planning an
# analysis are different jobs, and your machine may afford one but not the other.
# `ismith doctor` reports what actually fits; `ismith models` shows what each
# role resolves to now.

[roles]
{roles}

[budget]
# Spend ceiling for one session, counted only for providers that charge.
max_usd_per_session = 0.50

# Set true to refuse any provider that would send data off this machine. It is a
# hard failure, not a warning: configuration will not load if a role points at a
# remote model.
local_only = false

# Which dataframe API generated code is written against: polars, pandas or
# fireducks. Polars is the default and the only one with no platform caveat.
# engine = "polars"
"""


def config_path(path: Path | None = None, *, environ: Mapping[str, str] | None = None) -> Path:
    """Where configuration lives, by the same rules for reading and for writing.

    ``INSIGHTSMITH_CONFIG`` has to win in both, or creating a file and loading it
    can disagree about which file they mean.
    """
    if path is not None:
        return Path(path)
    env = os.environ if environ is None else environ
    return Path(env.get("INSIGHTSMITH_CONFIG", DEFAULT_CONFIG_PATH))


def ensure_config(
    path: Path | None = None,
    *,
    roles: dict[str, str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[Path, bool]:
    """Write a commented default configuration if there is none.

    Returns the path and whether it had to be created. Loading deliberately does
    not do this: a function that reads should not write, and the ``Consultant``
    API has no business creating files in someone's home directory. The CLI calls
    it, because a first run with no file is exactly when a worked example helps.

    Raises:
        ConfigError: if the file cannot be written.
    """
    target = config_path(path, environ=environ)
    if target.exists():
        return target, False

    chosen = roles or _DEFAULT_ROLES
    body = "\n".join(f'{role:<9}= "{model}"' for role, model in chosen.items())
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_TEMPLATE.format(roles=body), encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"could not create {target}: {exc}") from exc
    return target, True


def load_config(path: Path | None = None, *, environ: Mapping[str, str] | None = None) -> Config:
    """Load configuration, falling back to defaults when the file is absent.

    Raises:
        ConfigError: if the file is malformed, or if ``local_only`` is set while a
            role points at a provider that would send data off the machine.
    """
    env = os.environ if environ is None else environ
    target = config_path(path, environ=env)

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

    raw_engine = payload.get("engine", Engine.POLARS.value)
    try:
        engine = Engine(str(raw_engine))
    except ValueError as exc:
        known = ", ".join(sorted(member.value for member in Engine))
        raise ConfigError(f"engine must be one of {known}, got {raw_engine!r}") from exc

    config = Config(
        roles=roles,
        budget=budget,
        engine=engine,
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
