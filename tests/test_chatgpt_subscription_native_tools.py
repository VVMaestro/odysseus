"""Subscription native function calling through the real payload/stream adapters."""

import asyncio
import copy
import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from core import database
from src import agent_loop, chatgpt_subscription, llm_core


BASE = "https://chatgpt.com/backend-api/codex"
MODEL = "gpt-5.6-sol"
TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a workspace file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "mcp__notes__lookup", "description": "Find an MCP note",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}, "strict": False,
    }},
]


def _item(index=0, name="read_file", arguments='{"path":"sample.txt"}'):
    return {"type": "function_call", "id": f"fc_{index}", "call_id": f"call_{index}",
            "name": name, "arguments": arguments}


def _sse(event, **data):
    return "data: " + json.dumps({"type": event, **data})


def _completed(output=None):
    return _sse("response.completed", response={"output": output or [], "usage": {"input_tokens": 12, "output_tokens": 4}})


class _Response:
    def __init__(self, lines, status=200):
        self.lines = lines
        self.status_code = status

    async def aiter_lines(self):
        for line in self.lines:
            yield line

    async def aread(self):
        return b"unavailable"


class _Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    @asynccontextmanager
    async def stream(self, method, url, **kwargs):
        self.requests.append((url, copy.deepcopy(kwargs["json"])))
        yield next(self.responses)


def _patch_transport(monkeypatch, responses):
    client = _Client(responses)
    monkeypatch.setattr(llm_core, "_get_http_client", lambda: client)
    monkeypatch.setattr(llm_core, "_is_host_dead", lambda url: False)
    monkeypatch.setattr(llm_core, "_clear_host_dead", lambda url: None)
    monkeypatch.setattr(llm_core, "note_model_activity", lambda *args, **kwargs: None)
    return client


async def _collect(stream):
    return [chunk async for chunk in stream]


def _events(chunks):
    return [json.loads(line[6:]) for chunk in chunks for line in chunk.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"]


def _stream(messages=None, tools=TOOLS, url=BASE):
    return llm_core.stream_llm(url, MODEL, messages or [{"role": "user", "content": "Read sample.txt"}], tools=tools)


def test_payload_flattens_builtin_and_mcp_schemas_without_mutation():
    before = copy.deepcopy(TOOLS)
    payload = llm_core._build_chatgpt_responses_payload(
        MODEL, [{"role": "system", "content": "Instructions"}, {"role": "user", "content": "Hi"}],
        0.5, 2048, stream=True, tools=TOOLS,
    )
    assert payload["tools"] == [{"type": "function", "strict": False, **tool["function"]} for tool in TOOLS]
    assert TOOLS == before
    assert payload["instructions"] == "Instructions"
    assert payload["store"] is False
    assert "max_output_tokens" not in payload
    assert "max_tokens" not in payload
    assert "temperature" not in payload


def test_explicit_strict_schema_setting_is_preserved():
    tools = [{"type": "function", "function": {"name": "strict_tool", "strict": True,
              "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}}]
    payload = llm_core._build_chatgpt_responses_payload(MODEL, [], 0.5, 2048, tools=tools)
    assert payload["tools"][0]["strict"] is True


@pytest.mark.parametrize("terminal", ["completed", "done", "eof"])
def test_parallel_fragmented_calls_emit_once_with_original_call_ids(monkeypatch, terminal):
    first, second = _item(), _item(2, "mcp__notes__lookup", '{"query":"Привет"}')
    lines = [
        _sse("response.output_item.added", output_index=0, item={**first, "arguments": ""}),
        _sse("response.output_item.added", output_index=1, item={"type": "message", "id": "msg"}),
        _sse("response.output_item.added", output_index=2, item={**second, "arguments": ""}),
        _sse("response.function_call_arguments.delta", output_index=0, item_id="fc_0", delta='{"path":'),
        _sse("response.function_call_arguments.delta", output_index=2, item_id="fc_2", delta='{"query":"'),
        _sse("response.output_text.delta", delta="Reading now"),
        _sse("response.function_call_arguments.delta", output_index=0, item_id="fc_0", delta='"sample.txt"}'),
        _sse("response.function_call_arguments.delta", output_index=2, item_id="fc_2", delta='Привет"}'),
        _sse("response.function_call_arguments.done", output_index=0, item_id="fc_0", arguments=first["arguments"]),
        _sse("response.output_item.done", output_index=2, item=second),
        _sse("response.output_item.done", output_index=0, item=first),
        _sse("response.output_item.done", output_index=0, item=first),
    ]
    if terminal == "completed":
        lines += [_completed([first, {"type": "message"}, second]), "data: [DONE]"]
    elif terminal == "done":
        lines += ["data: [DONE]"]
    client = _patch_transport(monkeypatch, [_Response(lines)])
    chunks = asyncio.run(_collect(_stream()))
    events = _events(chunks)
    calls = [event for event in events if event.get("type") == "tool_calls"]
    assert calls == [{"type": "tool_calls", "calls": [
        {"id": item["call_id"], "name": item["name"], "arguments": item["arguments"]}
        for item in (first, second)
    ]}]
    assert {"delta": "Reading now"} in events
    assert chunks[-1] == "data: [DONE]\n\n"
    assert len(client.requests[0][1]["tools"]) == 2


@pytest.mark.parametrize("source", ["item_done", "completed_output"])
def test_final_snapshot_alone_recovers_tool_call(monkeypatch, source):
    item = _item()
    lines = ([_sse("response.output_item.done", output_index=0, item=item), _completed()]
             if source == "item_done" else [_completed([item])])
    _patch_transport(monkeypatch, [_Response(lines)])
    events = _events(asyncio.run(_collect(_stream())))
    assert next(event["calls"] for event in events if event.get("type") == "tool_calls") == [
        {"id": "call_0", "name": "read_file", "arguments": item["arguments"]}
    ]


def test_argument_events_use_item_identity_without_output_index(monkeypatch):
    item = _item()
    lines = [
        "event: response.output_item.added",
        "data:" + json.dumps({"output_index": 0, "item": {**item, "arguments": ""}}),
        _sse("response.function_call_arguments.delta", item_id="fc_0", delta='{"path":'),
        _sse("response.function_call_arguments.delta", item_id="fc_0", delta='"sample.txt"}'),
        _sse("response.function_call_arguments.done", item_id="fc_0", arguments=item["arguments"]),
        _completed(),
    ]
    _patch_transport(monkeypatch, [_Response(lines)])
    events = _events(asyncio.run(_collect(_stream())))
    assert next(event["calls"] for event in events if event.get("type") == "tool_calls") == [
        {"id": "call_0", "name": "read_file", "arguments": item["arguments"]}
    ]


def test_output_item_id_is_not_substituted_for_missing_call_id(monkeypatch):
    item = _item()
    del item["call_id"]
    _patch_transport(monkeypatch, [_Response([_completed([item])])])
    chunks = asyncio.run(_collect(_stream()))
    assert any("event: error" in chunk for chunk in chunks)
    assert not any(event.get("type") == "tool_calls" for event in _events(chunks))


def test_history_preserves_text_calls_outputs_and_order():
    messages = [
        {"role": "user", "content": "Read files"},
        {"role": "assistant", "content": [{"text": "Checking"}], "tool_calls": [
            {"id": "call_a", "function": {"name": "read_file", "arguments": '{"path":"a"}'}},
            {"id": "call_b", "function": {"name": "mcp__notes__lookup", "arguments": {"query": "b"}}},
        ]},
        {"role": "tool", "tool_call_id": "call_b", "content": [{"text": "MCP note"}]},
        {"role": "tool", "tool_call_id": "call_a", "content": "File contents"},
        {"role": "assistant", "content": "Final answer"},
    ]
    before = copy.deepcopy(messages)
    items = chatgpt_subscription.build_responses_input(messages)
    assert items == [
        {"role": "user", "content": [{"type": "input_text", "text": "Read files"}]},
        {"role": "assistant", "content": [{"type": "output_text", "text": "Checking"}]},
        {"type": "function_call", "call_id": "call_a", "name": "read_file", "arguments": '{"path":"a"}'},
        {"type": "function_call", "call_id": "call_b", "name": "mcp__notes__lookup", "arguments": '{"query": "b"}'},
        {"type": "function_call_output", "call_id": "call_b", "output": "MCP note"},
        {"type": "function_call_output", "call_id": "call_a", "output": "File contents"},
        {"role": "assistant", "content": [{"type": "output_text", "text": "Final answer"}]},
    ]
    assert messages == before


@pytest.mark.parametrize("name,arguments,result", [
    ("read_file", '{"path":"sample.txt"}', "file contents: violet"),
    ("mcp__notes__lookup", '{"query":"sample"}', "MCP note: violet"),
])
def test_two_rounds_replay_native_history_via_agent_result_writer(monkeypatch, name, arguments, result):
    item = _item(name=name, arguments=arguments)
    client = _patch_transport(monkeypatch, [
        _Response([_completed([item])]),
        _Response([_sse("response.output_text.delta", delta="The result says violet."), _completed()]),
    ])
    messages = [{"role": "user", "content": "Read the sample"}]

    async def run():
        events = _events(await _collect(_stream(messages)))
        calls = next(event["calls"] for event in events if event.get("type") == "tool_calls")
        agent_loop._append_tool_results(messages, "", calls, [result], [result], True, 1)
        return await _collect(_stream(messages))

    chunks = asyncio.run(run())
    assert {"delta": "The result says violet."} in _events(chunks)
    assert client.requests[1][1]["input"][-2:] == [
        {"type": "function_call", "call_id": "call_0", "name": name, "arguments": arguments},
        {"type": "function_call_output", "call_id": "call_0", "output": result},
    ]
    assert len(client.requests) == 2


def test_plain_chat_sends_no_tools(monkeypatch):
    client = _patch_transport(monkeypatch, [_Response([
        _sse("response.output_text.delta", delta="Hello"), _completed(),
    ])])
    events = _events(asyncio.run(_collect(_stream(tools=None))))
    assert {"delta": "Hello"} in events
    assert not any(event.get("type") == "tool_calls" for event in events)
    assert "tools" not in client.requests[0][1]


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("name", ["read_file", "mcp__notes__lookup"])
def test_agent_executes_tool_and_continues_in_selected_mode(monkeypatch, tmp_path, native, name):
    engine = create_engine("sqlite:///:memory:")
    database.Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(database, "SessionLocal", sessions)
    with sessions() as db:
        db.add(database.ModelEndpoint(id="sub", name="Subscription", base_url=BASE, supports_tools=native))
        db.commit()
    sample = tmp_path / "sample.txt"
    sample.write_text("violet", encoding="utf-8")
    arguments = json.dumps({"path": str(sample)}) if name == "read_file" else '{"query":"sample"}'
    item = _item(name=name, arguments=arguments)
    text_call = (f"```read_file\n{sample}\n```" if name == "read_file" else
                 '<tool_call><invoke name="mcp__notes__lookup"><parameter name="query">sample</parameter></invoke></tool_call>')
    first_lines = ([_completed([item])] if native else [
        _sse("response.output_text.delta", delta=text_call),
        _completed(),
    ])
    client = _patch_transport(monkeypatch, [
        _Response(first_lines),
        _Response([_sse("response.output_text.delta", delta="The result says violet."), _completed()]),
    ])
    monkeypatch.setattr(agent_loop, "get_setting", lambda key, default=None: default)
    monkeypatch.setattr(agent_loop, "get_mcp_manager", lambda: None)
    monkeypatch.setattr(agent_loop, "blocked_tools_for_owner", lambda owner: set())
    monkeypatch.setattr(agent_loop, "estimate_tokens", lambda *args, **kwargs: 10)
    # Focus on transport/execution; tool selection and prompt construction are
    # independent of the adapter and intentionally kept out of this regression.
    monkeypatch.setattr(agent_loop, "_build_system_prompt", lambda messages, *args, **kwargs: (messages, TOOLS[1:]))
    executed = []

    async def execute(block, **kwargs):
        executed.append(block)
        if block.tool_type == "read_file":
            path = Path(block.content)
            if not path.is_absolute():
                path = tmp_path / path
            assert path.resolve() == sample.resolve()
            return "read_file", {"content": sample.read_text(encoding="utf-8")}
        assert block.tool_type == "mcp__notes__lookup"
        assert json.loads(block.content) == {"query": "sample"}
        return block.tool_type, {"content": "violet"}

    monkeypatch.setattr(agent_loop, "execute_tool_block", execute)
    prompt = "Read the sample.txt file." if name == "read_file" else "Look up the sample using MCP."
    chunks = asyncio.run(_collect(agent_loop.stream_agent_loop(
        BASE, MODEL, [{"role": "user", "content": prompt}],
        max_rounds=3, relevant_tools={name}, workspace=str(tmp_path) if name == "read_file" else None,
        _is_teacher_run=True,
    )))
    assert len(executed) == 1
    assert len(client.requests) == 2
    assert any(event.get("delta") == "The result says violet." for event in _events(chunks))
    first_payload = client.requests[0][1]
    second_input = client.requests[1][1]["input"]
    if native:
        assert name in [tool["name"] for tool in first_payload["tools"]]
        assert any(item.get("type") == "function_call" and item["call_id"] == "call_0" for item in second_input)
        assert any(item.get("type") == "function_call_output" and item["call_id"] == "call_0" and "violet" in item["output"]
                   for item in second_input)
    else:
        assert "tools" not in first_payload
        assert not any(item.get("type") in {"function_call", "function_call_output"} for item in second_input)
        assert "violet" in json.dumps(second_input)


@pytest.mark.parametrize("failure", ["response.failed", "response.incomplete", "eof", "done", "completed"])
def test_incomplete_calls_are_not_executed(monkeypatch, failure):
    lines = [_sse("response.output_item.added", output_index=0, item={**_item(), "arguments": ""}),
             _sse("response.function_call_arguments.delta", output_index=0, item_id="fc_0", delta='{"path":')]
    if failure == "done":
        lines.append("data: [DONE]")
    elif failure == "completed":
        lines.append(_completed())
    elif failure != "eof":
        lines.append(_sse(failure, response={"error": {"message": "failed"}}))
    _patch_transport(monkeypatch, [_Response(lines)])
    chunks = asyncio.run(_collect(_stream()))
    assert any("event: error" in chunk for chunk in chunks)
    assert not any(event.get("type") == "tool_calls" for event in _events(chunks))


@pytest.mark.parametrize("subscription_is_fallback", [True, False])
def test_fallback_selects_each_providers_tool_transport(monkeypatch, subscription_is_fallback):
    subscription = (BASE, MODEL, {})
    chat = ("https://api.openai.com/v1", "gpt-4o", {})
    selected, backup = (chat, subscription) if subscription_is_fallback else (subscription, chat)
    item = _item()
    success = ([_completed([item])] if subscription_is_fallback else [
        "data: " + json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_0", "type": "function",
                          "function": {"name": "read_file", "arguments": item["arguments"]}}]}}]}),
        "data: [DONE]",
    ])
    client = _patch_transport(monkeypatch, [_Response([], 503), _Response(success)])
    chunks = asyncio.run(_collect(llm_core.stream_llm_with_fallback(
        [selected, backup], [{"role": "user", "content": "Read file"}], tools=TOOLS,
        fallback_statuses={503}, fallback_on_empty=False,
    )))
    assert any(event.get("type") == "fallback" for event in _events(chunks))
    assert any(event.get("type") == "tool_calls" for event in _events(chunks))
    for url, payload in client.requests:
        if url.startswith(BASE):
            assert url.endswith("/responses")
            assert "input" in payload and "messages" not in payload
            assert payload["tools"][0]["name"] == "read_file"
        else:
            assert url.endswith("/chat/completions")
            assert "messages" in payload and "input" not in payload
            assert payload["tools"] == TOOLS


@pytest.mark.parametrize("subscription_is_fallback", [True, False])
def test_agent_pins_fallback_and_continues_with_matching_native_history(monkeypatch, tmp_path, subscription_is_fallback):
    subscription = (BASE, MODEL, chatgpt_subscription.chatgpt_headers(None))
    chat = ("https://api.openai.com/v1", "gpt-4o", {})
    primary, backup = (chat, subscription) if subscription_is_fallback else (subscription, chat)
    engine = create_engine("sqlite:///:memory:")
    database.Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(database, "SessionLocal", sessions)
    with sessions() as db:
        for endpoint_id, route, native in (("primary", primary, False), ("backup", backup, True)):
            db.add(database.ModelEndpoint(id=endpoint_id, name=endpoint_id, base_url=route[0], supports_tools=native))
        db.commit()
    item = _item()
    chat_call = "data: " + json.dumps({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "call_0", "type": "function", "function": {"name": "read_file", "arguments": item["arguments"]}}
    ]}}]})
    if subscription_is_fallback:
        call_lines = [_completed([item])]
        answer_lines = [_sse("response.output_text.delta", delta="The file says violet."), _completed()]
    else:
        call_lines = [chat_call, "data: [DONE]"]
        answer_lines = ['data: {"choices":[{"delta":{"content":"The file says violet."}}]}', "data: [DONE]"]
    client = _patch_transport(monkeypatch, [_Response([], 503), _Response(call_lines), _Response(answer_lines)])
    monkeypatch.setattr(agent_loop, "get_setting", lambda key, default=None: default)
    monkeypatch.setattr(agent_loop, "get_mcp_manager", lambda: None)
    monkeypatch.setattr(agent_loop, "blocked_tools_for_owner", lambda owner: set())
    monkeypatch.setattr(agent_loop, "estimate_tokens", lambda *args, **kwargs: 10)
    monkeypatch.setattr(agent_loop, "_build_system_prompt", lambda messages, *args, **kwargs: (messages, []))

    async def compact(_session, _url, _model, messages, *args, **kwargs):
        return list(messages), 4096, False

    monkeypatch.setattr(agent_loop, "maybe_compact", compact)
    executed = []

    async def execute(block, **kwargs):
        executed.append(block)
        return "read_file", {"content": "violet"}

    monkeypatch.setattr(agent_loop, "execute_tool_block", execute)
    chunks = asyncio.run(_collect(agent_loop.stream_agent_loop(
        primary[0], primary[1], [{"role": "user", "content": "Read sample.txt"}], headers=primary[2],
        max_rounds=3, relevant_tools={"read_file"}, workspace=str(tmp_path), fallbacks=[backup],
        fallback_statuses={503}, fallback_on_empty=False, _is_teacher_run=True,
    )))
    assert len(executed) == 1
    assert len(client.requests) == 3
    assert any(event.get("delta") == "The file says violet." for event in _events(chunks))
    assert "tools" not in client.requests[0][1]
    assert client.requests[1][0] == client.requests[2][0]
    assert client.requests[2][1]["model"] == backup[1]
    payload = client.requests[2][1]
    if subscription_is_fallback:
        result = payload["input"][-1]
        assert result["type"] == "function_call_output"
        assert result["call_id"] == "call_0"
        assert "violet" in result["output"]
    else:
        result = payload["messages"][-1]
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_0"
        assert "violet" in result["content"]


@pytest.mark.parametrize("mode", [True, False, None])
def test_subscription_tool_mode_honors_endpoint_override(monkeypatch, mode):
    engine = create_engine("sqlite:///:memory:")
    database.Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(database, "SessionLocal", sessions)
    with sessions() as db:
        db.add(database.ModelEndpoint(id="sub", name="Subscription", base_url=BASE, supports_tools=mode))
        db.commit()
    # An unrecognized catalog slug must still use the Subscription transport.
    assert agent_loop._agent_route_tool_mode(BASE + "/responses", "custom-catalog-slug")[0] is (mode is not False)


def test_legacy_migration_only_updates_subscription_once(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    database.Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(database, "engine", engine)
    with sessions() as db:
        db.add_all([
            database.ProviderAuthSession(id="sub-auth", provider="chatgpt-subscription", base_url=BASE),
            database.ProviderAuthSession(id="other-auth", provider="copilot", base_url="https://api.githubcopilot.com"),
        ])
        db.add_all([
            database.ModelEndpoint(id="legacy", name="Legacy", base_url=BASE, provider_auth_id="sub-auth", supports_tools=False),
            database.ModelEndpoint(id="plain", name="API", base_url=BASE, supports_tools=False),
            database.ModelEndpoint(id="custom", name="Custom", base_url="https://custom.example/v1", provider_auth_id="sub-auth", supports_tools=False),
            database.ModelEndpoint(id="other", name="Other", base_url=BASE, provider_auth_id="other-auth", supports_tools=False),
        ])
        db.commit()
    database._migrate_chatgpt_subscription_native_tools()
    with sessions() as db:
        assert db.get(database.ModelEndpoint, "legacy").supports_tools is True
        for endpoint_id in ("plain", "custom", "other"):
            assert db.get(database.ModelEndpoint, endpoint_id).supports_tools is False
        db.get(database.ModelEndpoint, "legacy").supports_tools = False
        db.commit()
    database._migrate_chatgpt_subscription_native_tools()
    with sessions() as db:
        assert db.get(database.ModelEndpoint, "legacy").supports_tools is False
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM schema_migrations")).scalar_one() == 1
