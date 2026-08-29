"""Provider, config and router tests.

Every HTTP exchange goes through ``httpx.MockTransport`` with a recorded response
body, so nothing here touches the network — no live calls in CI, and no vcrpy
dependency either, since httpx ships the mock transport.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from insightsmith.config import load_config
from insightsmith.errors import BudgetError, ConfigError, ProviderError
from insightsmith.llm.base import Capabilities, Message, Usage
from insightsmith.llm.ollama import OllamaProvider
from insightsmith.llm.openai_compat import BACKENDS, OpenAICompatProvider
from insightsmith.llm.registry import build_provider, is_local_model, split_model
from insightsmith.llm.router import Router, Strategy, extract_json

# --------------------------------------------------------------------------- #
# recorded response bodies
# --------------------------------------------------------------------------- #

OPENAI_REPLY: dict[str, Any] = {
    "model": "gpt-4o-mini",
    "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
}

OPENAI_TOOL_REPLY: dict[str, Any] = {
    "model": "gpt-4o-mini",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "respond",
                            "arguments": '{"question": "why", "effort": 2}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 8},
}

# Shape verified against a running Ollama.
OLLAMA_SHOW_TOOLS: dict[str, Any] = {
    "capabilities": ["completion", "tools", "thinking"],
    "details": {"parameter_size": "8.2B", "quantization_level": "Q4_K_M"},
    "model_info": {
        "qwen3.block_count": 36,
        "qwen3.attention.head_count_kv": 8,
        "qwen3.context_length": 40960,
    },
}
OLLAMA_SHOW_NO_TOOLS: dict[str, Any] = {
    "capabilities": ["completion"],
    "model_info": {"llama.context_length": 8192},
}
# qwen3:8b really does report this; verified against a running Ollama.
OLLAMA_SHOW_THINKING: dict[str, Any] = {
    "capabilities": ["completion", "tools", "thinking"],
    "model_info": {"qwen3.context_length": 40960},
}
OLLAMA_TAGS: dict[str, Any] = {"models": [{"name": "qwen3:8b"}, {"name": "mistral:latest"}]}


_OPEN_CLIENTS: list[httpx.Client] = []


@pytest.fixture(autouse=True)
def _close_clients():
    """Close every stub client.

    A leaked httpx.Client surfaces as an unraisable exception during GC, which
    filterwarnings=error then reports against whichever unrelated test happened
    to be running — the same trap the sqlite fixture fell into.
    """
    yield
    while _OPEN_CLIENTS:
        _OPEN_CLIENTS.pop().close()


def _transport(handler) -> httpx.Client:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    _OPEN_CLIENTS.append(client)
    return client


def _always(body: dict[str, Any], status: int = 200) -> httpx.Client:
    return _transport(lambda _: httpx.Response(status, json=body))


# --------------------------------------------------------------------------- #
# model references
# --------------------------------------------------------------------------- #


def test_split_model_splits_on_the_first_slash_only() -> None:
    """openrouter models keep their vendor prefix."""
    assert split_model("ollama/qwen3:8b") == ("ollama", "qwen3:8b")
    assert split_model("openrouter/anthropic/claude-sonnet-4.5") == (
        "openrouter",
        "anthropic/claude-sonnet-4.5",
    )


@pytest.mark.parametrize("bad", ["qwen3:8b", "/model", "ollama/", ""])
def test_split_model_rejects_references_without_a_provider(bad: str) -> None:
    with pytest.raises(ProviderError):
        split_model(bad)


def test_locality_of_a_reference() -> None:
    assert is_local_model("ollama/qwen3:8b")
    assert is_local_model("lmstudio/whatever")
    assert not is_local_model("openai/gpt-4o-mini")
    assert not is_local_model("nonsense")


def test_building_a_remote_provider_without_a_key_fails_clearly() -> None:
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        build_provider("openai", environ={})


def test_building_an_unknown_provider_lists_the_known_ones() -> None:
    with pytest.raises(ProviderError, match="Known:"):
        build_provider("not-a-provider", environ={})


def test_every_backend_declares_a_base_url() -> None:
    for name, (base_url, _) in BACKENDS.items():
        assert base_url.startswith("http"), name


# --------------------------------------------------------------------------- #
# OpenAI-compatible provider
# --------------------------------------------------------------------------- #


def test_openai_compat_chat() -> None:
    provider = OpenAICompatProvider("openai", api_key="k", client=_always(OPENAI_REPLY))
    completion = provider.chat("gpt-4o-mini", [Message(role="user", content="hi")])
    assert completion.text == "hello"
    assert completion.usage.input_tokens == 10


def test_openai_compat_sends_the_key_and_the_right_url() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=OPENAI_REPLY)

    provider = OpenAICompatProvider("groq", api_key="secret", client=_transport(handler))
    provider.chat("llama-3.3-70b", [Message(role="user", content="hi")])
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["auth"] == "Bearer secret"


def test_openai_compat_parses_tool_calls_with_stringified_arguments() -> None:
    provider = OpenAICompatProvider("openai", api_key="k", client=_always(OPENAI_TOOL_REPLY))
    completion = provider.chat("gpt-4o-mini", [Message(role="user", content="hi")])
    assert completion.tool_calls[0].name == "respond"
    assert completion.tool_calls[0].arguments == {"question": "why", "effort": 2}


def test_openai_compat_surfaces_http_errors() -> None:
    provider = OpenAICompatProvider(
        "openai", api_key="k", client=_transport(lambda _: httpx.Response(401, text="nope"))
    )
    with pytest.raises(ProviderError, match="401"):
        provider.chat("gpt-4o-mini", [Message(role="user", content="hi")])


def test_openai_compat_prices_usage_when_costs_are_declared() -> None:
    provider = OpenAICompatProvider("openai", api_key="k", client=_always(OPENAI_REPLY))
    provider.declare(
        "gpt-4o-mini",
        Capabilities(context_window=128_000, cost_in_per_mtok=0.15, cost_out_per_mtok=0.60),
    )
    completion = provider.chat("gpt-4o-mini", [Message(role="user", content="hi")])
    assert completion.usage.cost_usd == pytest.approx((10 * 0.15 + 5 * 0.60) / 1e6)


def test_local_costs_stay_none() -> None:
    assert Usage(10, 5).priced(Capabilities(context_window=4096)).cost_usd is None


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #


def test_ollama_reads_capabilities_rather_than_assuming() -> None:
    provider = OllamaProvider(client=_always(OLLAMA_SHOW_TOOLS))
    capabilities = provider.capabilities("qwen3:8b")
    assert capabilities.tool_calling
    assert capabilities.context_window == 40960


def test_ollama_reports_a_model_without_tool_calling_honestly() -> None:
    provider = OllamaProvider(client=_always(OLLAMA_SHOW_NO_TOOLS))
    assert not provider.capabilities("llama3.2:3b").tool_calling


def test_ollama_caches_show() -> None:
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=OLLAMA_SHOW_TOOLS)

    provider = OllamaProvider(client=_transport(handler))
    provider.capabilities("qwen3:8b")
    provider.capabilities("qwen3:8b")
    assert calls["n"] == 1


def test_ollama_list_models() -> None:
    provider = OllamaProvider(client=_always(OLLAMA_TAGS))
    assert provider.list_models() == ["mistral:latest", "qwen3:8b"]


def test_ollama_says_what_to_do_when_the_server_is_down() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    provider = OllamaProvider(client=_transport(handler))
    with pytest.raises(ProviderError, match="ollama serve"):
        provider.list_models()


def test_ollama_chat_maps_its_own_token_counts() -> None:
    body = {
        "model": "qwen3:8b",
        "message": {"role": "assistant", "content": "hi"},
        "prompt_eval_count": 7,
        "eval_count": 3,
        "done_reason": "stop",
    }
    provider = OllamaProvider(client=_always(body))
    completion = provider.chat("qwen3:8b", [Message(role="user", content="hi")])
    assert (completion.usage.input_tokens, completion.usage.output_tokens) == (7, 3)


def _recording_ollama(show: dict[str, Any]) -> tuple[OllamaProvider, list[dict[str, Any]]]:
    """An Ollama stub that answers /api/show and keeps every chat payload."""
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(200, json=show)
        sent.append(json.loads(request.content))
        return httpx.Response(
            200, json={"model": "m", "message": {"role": "assistant", "content": "ok"}}
        )

    return OllamaProvider(client=_transport(handler)), sent


def test_ollama_states_the_context_it_needs() -> None:
    """Left unsaid, Ollama assumes 2048 and truncates the front of the prompt.

    That silently discards the system prompt and the dataset card — the model
    answers from the tail of the question and nothing in the reply says so.
    """
    provider, sent = _recording_ollama(OLLAMA_SHOW_TOOLS)
    provider.chat("qwen3:8b", [Message(role="user", content="hi")])
    assert sent[0]["options"]["num_ctx"] == 2048


def test_ollama_grows_the_context_to_fit_a_long_prompt() -> None:
    provider, sent = _recording_ollama(OLLAMA_SHOW_TOOLS)
    provider.chat("qwen3:8b", [Message(role="user", content="x" * 30_000)])
    assert sent[0]["options"]["num_ctx"] == 16_384


def test_ollama_never_asks_for_more_context_than_the_model_has() -> None:
    """llama here declares 8192; asking for 32k would fail or thrash, not help."""
    provider, sent = _recording_ollama(OLLAMA_SHOW_NO_TOOLS)
    provider.chat("llama3:8b", [Message(role="user", content="x" * 200_000)])
    assert sent[0]["options"]["num_ctx"] == 8192


def test_a_caller_can_size_the_context_itself() -> None:
    provider, sent = _recording_ollama(OLLAMA_SHOW_TOOLS)
    provider.chat("qwen3:8b", [Message(role="user", content="hi")], num_ctx=4096)
    assert sent[0]["options"]["num_ctx"] == 4096


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def test_config_defaults_without_a_file(tmp_path: Path) -> None:
    config = load_config(tmp_path / "absent.toml", environ={})
    assert config.roles["planner"].startswith("ollama/")
    assert not config.budget.local_only
    assert config.path is None


def test_config_reads_roles_and_budget(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[roles]\nplanner = "openai/gpt-4o-mini"\n\n'
        "[budget]\nmax_usd_per_session = 1.25\nlocal_only = false\n",
        encoding="utf-8",
    )
    config = load_config(path, environ={})
    assert config.roles["planner"] == "openai/gpt-4o-mini"
    assert config.budget.max_usd_per_session == 1.25


def test_local_only_hard_fails_on_a_remote_role(tmp_path: Path) -> None:
    """A privacy switch that merely warned would not be a privacy switch."""
    path = tmp_path / "config.toml"
    path.write_text(
        '[roles]\nplanner = "openai/gpt-4o-mini"\n\n[budget]\nlocal_only = true\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="planner"):
        load_config(path, environ={})


def test_local_only_accepts_local_roles(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[roles]\nplanner = "ollama/qwen3:8b"\n\n[budget]\nlocal_only = true\n',
        encoding="utf-8",
    )
    assert load_config(path, environ={}).budget.local_only


def test_local_only_can_be_forced_from_the_environment(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[roles]\nplanner = "openai/gpt-4o-mini"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path, environ={"INSIGHTSMITH_LOCAL_ONLY": "1"})


def test_malformed_config_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("this is not = = toml", encoding="utf-8")
    with pytest.raises(ConfigError, match="could not read"):
        load_config(path, environ={})


def test_non_string_role_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[roles]\nplanner = 42\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a string"):
        load_config(path, environ={})


# --------------------------------------------------------------------------- #
# JSON salvage
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        'Sure! Here is the answer:\n{"a": 1}\nHope that helps.',
        '  \n {"a": 1}  ',
    ],
)
def test_extract_json_salvages_what_small_models_actually_return(text: str) -> None:
    assert extract_json(text) == {"a": 1}


@pytest.mark.parametrize("text", ["no json here", "", "[1, 2, 3]", "{not json}"])
def test_extract_json_gives_up_honestly(text: str) -> None:
    assert extract_json(text) is None


# --------------------------------------------------------------------------- #
# router
# --------------------------------------------------------------------------- #


def _router(tmp_path: Path, roles: str, provider: Any) -> Router:
    path = tmp_path / "config.toml"
    path.write_text(roles, encoding="utf-8")
    router = Router(config=load_config(path, environ={}))
    router._providers["ollama"] = provider
    router._providers["openai"] = provider
    return router


def test_router_picks_tool_calling_when_available(tmp_path: Path) -> None:
    router = _router(
        tmp_path,
        '[roles]\nplanner = "ollama/qwen3:8b"\n',
        OllamaProvider(client=_always(OLLAMA_SHOW_TOOLS)),
    )
    assert router.route("planner").strategy is Strategy.TOOL_CALLING


def test_router_degrades_when_the_model_cannot_call_tools(tmp_path: Path) -> None:
    """The path that makes small local models usable at all."""
    router = _router(
        tmp_path,
        '[roles]\nplanner = "ollama/llama3.2:3b"\n',
        OllamaProvider(client=_always(OLLAMA_SHOW_NO_TOOLS)),
    )
    assert router.route("planner").strategy is Strategy.JSON_MODE


def test_router_refuses_a_remote_model_under_local_only(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[roles]\nplanner = "ollama/qwen3:8b"\n', encoding="utf-8")
    config = load_config(path, environ={})
    config.budget.local_only = True
    config.roles["planner"] = "openai/gpt-4o-mini"
    router = Router(config=config)
    with pytest.raises(ProviderError, match="local_only"):
        router.route("planner")


def test_router_reports_an_unconfigured_role(tmp_path: Path) -> None:
    router = _router(
        tmp_path,
        '[roles]\nplanner = "ollama/qwen3:8b"\n',
        OllamaProvider(client=_always(OLLAMA_SHOW_TOOLS)),
    )
    with pytest.raises(ProviderError, match="no model configured"):
        router.route("vision")


def test_structured_uses_the_tool_call_arguments(tmp_path: Path) -> None:
    replies = [OLLAMA_SHOW_TOOLS, OPENAI_TOOL_REPLY]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=replies[0] if "show" in str(request.url) else replies[1])

    provider = OpenAICompatProvider("openai", api_key="k", client=_transport(handler))
    path = tmp_path / "config.toml"
    path.write_text('[roles]\nplanner = "openai/gpt-4o-mini"\n', encoding="utf-8")
    router = Router(config=load_config(path, environ={}))
    router._providers["openai"] = provider

    result = router.structured(
        "planner", [Message(role="user", content="go")], schema={"type": "object"}
    )
    assert result == {"question": "why", "effort": 2}


def test_structured_retries_until_the_json_parses(tmp_path: Path) -> None:
    """Feed the failure back rather than giving up on the first bad reply."""
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if "show" in str(request.url):
            return httpx.Response(200, json=OLLAMA_SHOW_NO_TOOLS)
        attempts["n"] += 1
        text = "sorry, I cannot" if attempts["n"] == 1 else '{"ok": true}'
        return httpx.Response(
            200, json={"model": "m", "message": {"role": "assistant", "content": text}}
        )

    router = _router(
        tmp_path,
        '[roles]\nplanner = "ollama/llama3.2:3b"\n',
        OllamaProvider(client=_transport(handler)),
    )
    assert router.structured("planner", [Message(role="user", content="go")]) == {"ok": True}
    assert attempts["n"] == 2


def test_structured_gives_up_after_the_retry_budget(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "show" in str(request.url):
            return httpx.Response(200, json=OLLAMA_SHOW_NO_TOOLS)
        return httpx.Response(
            200, json={"model": "m", "message": {"role": "assistant", "content": "nope"}}
        )

    router = _router(
        tmp_path,
        '[roles]\nplanner = "ollama/llama3.2:3b"\n',
        OllamaProvider(client=_transport(handler)),
    )
    with pytest.raises(ProviderError, match="did not return valid JSON"):
        router.structured("planner", [Message(role="user", content="go")], retries=2)


def test_budget_is_enforced(tmp_path: Path) -> None:
    provider = OpenAICompatProvider("openai", api_key="k", client=_always(OPENAI_REPLY))
    provider.declare(
        "gpt-4o-mini",
        Capabilities(context_window=8192, cost_in_per_mtok=1000.0, cost_out_per_mtok=1000.0),
    )
    path = tmp_path / "config.toml"
    path.write_text(
        '[roles]\nplanner = "openai/gpt-4o-mini"\n\n[budget]\nmax_usd_per_session = 0.001\n',
        encoding="utf-8",
    )
    router = Router(config=load_config(path, environ={}))
    router._providers["openai"] = provider

    with pytest.raises(BudgetError, match="budget"):
        for _ in range(10):
            router.complete("planner", [Message(role="user", content="hi")])


def test_message_wire_shape_round_trips_tool_calls() -> None:
    message = Message(role="assistant", content="", tool_calls=[])
    assert message.as_wire() == {"role": "assistant", "content": ""}
    tool_result = Message(role="tool", content="42", tool_call_id="call_1")
    assert tool_result.as_wire()["tool_call_id"] == "call_1"


def test_recorded_bodies_are_valid_json() -> None:
    """Guards the fixtures themselves against typos."""
    for body in (OPENAI_REPLY, OPENAI_TOOL_REPLY, OLLAMA_SHOW_TOOLS, OLLAMA_TAGS):
        assert json.loads(json.dumps(body)) == body


# --------------------------------------------------------------------------- #
# reasoning models and timeouts
# --------------------------------------------------------------------------- #


def test_a_slow_model_is_not_reported_as_an_absent_server() -> None:
    """A read timeout once said "is `ollama serve` running?" — it was running."""

    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    provider = OllamaProvider(client=_transport(handler), timeout=12.0)
    with pytest.raises(ProviderError) as caught:
        provider.list_models()
    message = str(caught.value)
    assert "no reply within 12s" in message or "sent no reply within 12s" in message
    assert "ollama serve" not in message


def test_a_genuinely_absent_server_still_says_so() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    provider = OllamaProvider(client=_transport(handler))
    with pytest.raises(ProviderError, match="ollama serve"):
        provider.list_models()


def _capture_chat(show: dict[str, Any]) -> tuple[OllamaProvider, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "show" in str(request.url):
            return httpx.Response(200, json=show)
        seen.append(json.loads(request.content))
        return httpx.Response(
            200, json={"model": "m", "message": {"role": "assistant", "content": "{}"}}
        )

    return OllamaProvider(client=_transport(handler)), seen


def test_thinking_is_disabled_when_json_is_wanted() -> None:
    """Chain-of-thought into a JSON slot is latency for no output.

    qwen3:8b exceeded a 300s timeout thinking, and returned the same object in
    0.8s with thinking off.
    """
    provider, seen = _capture_chat(OLLAMA_SHOW_THINKING)
    provider.chat("qwen3:8b", [Message(role="user", content="go")], json_mode=True)
    assert seen[0]["think"] is False
    assert seen[0]["format"] == "json"


def test_thinking_is_disabled_when_tools_are_used() -> None:
    provider, seen = _capture_chat(OLLAMA_SHOW_THINKING)
    provider.chat("qwen3:8b", [Message(role="user", content="go")], tools=[{"type": "function"}])
    assert seen[0]["think"] is False


def test_thinking_is_left_alone_for_free_form_replies() -> None:
    """Only structured output pays for reasoning it cannot use."""
    provider, seen = _capture_chat(OLLAMA_SHOW_THINKING)
    provider.chat("qwen3:8b", [Message(role="user", content="explain")])
    assert "think" not in seen[0]


def test_a_model_without_reasoning_is_sent_no_think_flag() -> None:
    provider, seen = _capture_chat(OLLAMA_SHOW_NO_TOOLS)
    provider.chat("llama3.2:3b", [Message(role="user", content="go")], json_mode=True)
    assert "think" not in seen[0]


def test_the_caller_can_force_thinking_back_on() -> None:
    provider, seen = _capture_chat(OLLAMA_SHOW_THINKING)
    provider.chat("qwen3:8b", [Message(role="user", content="go")], json_mode=True, think=True)
    assert seen[0]["think"] is True


def test_thinks_reads_the_capability_list() -> None:
    assert OllamaProvider(client=_always(OLLAMA_SHOW_THINKING)).thinks("qwen3:8b")
    assert not OllamaProvider(client=_always(OLLAMA_SHOW_NO_TOOLS)).thinks("llama3.2:3b")
