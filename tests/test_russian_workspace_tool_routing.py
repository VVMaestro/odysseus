"""Russian file/code intent must advertise editing tools to the agent."""

import asyncio

import pytest

from src import agent_loop, tool_index


SNAKE_REQUEST = (
    'В твоей рабочей папке есть игра "Змейка"\n'
    'Надо доработать игру и исправить баг: при старте змейка сразу же '
    'исчезает и больше ничего не происходит'
)
WRITE_TOOLS = {"write_file", "edit_file", "apply_patch"}


@pytest.mark.parametrize("prompt", [
    SNAKE_REQUEST,
    "Исправьте ошибку в игре и сохраните изменения",
    "Нужно починить приложение и добавить тесты",
    "Доработай программу: добавь кнопку перезапуска",
    "Переименуй модуль и обнови функцию",
    "Создай скрипт на Python",
    "Напиши игру Змейка",
    "Отредактируй код приложения",
    "Улучши программу и поменяй обработчик ошибки",
    "Надо протестировать и собрать проект",
    "Измени функцию update в game.js",
    "Fix ошибку в приложении",
    "Игра зависает после длинной последовательности действий. "
    "Сначала работает меню, потом экран перестаёт обновляться. Надо починить.",
    "Fix the bug in the Snake game and save the changes to the file",
])
def test_code_requests_select_file_domain_and_workspace_toolset(prompt):
    intent = agent_loop._classify_agent_request([], prompt)

    assert intent["low_signal"] is False
    assert "files" in intent["domains"]
    assert agent_loop._looks_like_workspace_coding_request(prompt)


@pytest.mark.parametrize("prompt", [
    "Прочитай содержимое файла",
    "Покажи файлы в папке проекта",
    "Найди строку в исходниках",
    "Посмотри структуру репозитория",
    "Проверь каталог с тестами",
])
def test_russian_file_inflections_select_file_domain(prompt):
    intent = agent_loop._classify_agent_request([], prompt)

    assert "files" in intent["domains"]
    assert intent["low_signal"] is False


@pytest.mark.parametrize("prompt", [
    "Привет",
    "Спасибо",
    "Расскажи про правила игры Змейка",
    "Напиши рассказ о змейке",
    "Исправь опечатку в письме",
    "Добавь напоминание купить хлеб",
    "Расскажи про программирование",
    "Изучи программирование",
    "Исправь опечатку: актёр играет злодея",
])
def test_unrelated_russian_requests_do_not_select_file_tools(prompt):
    intent = agent_loop._classify_agent_request([], prompt)

    assert "files" not in intent["domains"]
    assert not agent_loop._looks_like_workspace_coding_request(prompt)


def test_russian_request_does_not_inherit_unrelated_history():
    intent = agent_loop._classify_agent_request([
        {"role": "user", "content": SNAKE_REQUEST},
        {"role": "assistant", "content": "Готово."},
        {"role": "user", "content": "Расскажи про правила игры"},
    ], "Расскажи про правила игры")

    assert "files" not in intent["domains"]
    assert intent["retrieval_query"] == "Расскажи про правила игры"


@pytest.mark.parametrize("retrieval", ["minimal", "failed", "timeout"])
@pytest.mark.parametrize("mode", ["agent", "plan", "edit_disabled"])
def test_snake_request_sends_edit_schemas_and_honors_restrictions(
    monkeypatch, tmp_path, retrieval, mode,
):
    """Exercise actual selection and schema filtering, without model/file I/O."""
    queries = []

    class MinimalIndex:
        def get_tools_for_query(self, query, k):
            queries.append(query)
            return set(tool_index.ALWAYS_AVAILABLE)

    def get_index():
        if retrieval == "failed":
            raise RuntimeError("embedding backend unavailable")
        if retrieval == "timeout":
            raise asyncio.TimeoutError
        return MinimalIndex()

    monkeypatch.setattr(tool_index, "get_tool_index", get_index)
    monkeypatch.setattr(agent_loop, "get_setting", lambda key, default=None: default)
    monkeypatch.setattr(agent_loop, "get_mcp_manager", lambda: None)
    monkeypatch.setattr(agent_loop, "blocked_tools_for_owner", lambda owner: set())
    monkeypatch.setattr(agent_loop, "estimate_tokens", lambda *args, **kwargs: 10)
    # Prompt enrichment is independent of intent/schema selection and can read
    # user-specific skills/preferences. Keep those outside this regression.
    monkeypatch.setattr(agent_loop, "_build_system_prompt", lambda messages, *args, **kwargs: (messages, []))
    from services.memory.skills import SkillsManager
    monkeypatch.setattr(SkillsManager, "load", lambda *args, **kwargs: [])
    sent = []

    async def stream(*args, **kwargs):
        sent.append({schema["function"]["name"] for schema in kwargs.get("tools") or []})
        yield 'data: {"delta": "Готово."}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", stream)

    async def collect():
        return [chunk async for chunk in agent_loop.stream_agent_loop(
            "https://chatgpt.com/backend-api/codex", "gpt-5.6-sol",
            [{"role": "user", "content": SNAKE_REQUEST}],
            workspace=str(tmp_path), plan_mode=mode == "plan",
            disabled_tools={"edit_file"} if mode == "edit_disabled" else set(),
            max_rounds=1, _is_teacher_run=True,
        )]

    asyncio.run(collect())

    assert len(sent) == 1
    assert {"read_file", "grep", "glob", "ls"} <= sent[0]
    if mode == "agent":
        assert WRITE_TOOLS <= sent[0]
    elif mode == "plan":
        assert WRITE_TOOLS.isdisjoint(sent[0])
    else:
        assert "edit_file" not in sent[0]
        assert {"write_file", "apply_patch"} <= sent[0]
    if retrieval == "minimal":
        # The old read-only shortcut skipped retrieval altogether for this
        # exact prompt, leaving only read/search tools in the model context.
        assert queries == [SNAKE_REQUEST]


def test_vague_workspace_request_keeps_readonly_shortcut(monkeypatch, tmp_path):
    retrieval_calls = []

    def unexpected_index():
        retrieval_calls.append(True)
        raise AssertionError("a vague workspace request should skip retrieval")

    monkeypatch.setattr(tool_index, "get_tool_index", unexpected_index)
    monkeypatch.setattr(agent_loop, "get_setting", lambda key, default=None: default)
    monkeypatch.setattr(agent_loop, "get_mcp_manager", lambda: None)
    monkeypatch.setattr(agent_loop, "blocked_tools_for_owner", lambda owner: set())
    monkeypatch.setattr(agent_loop, "estimate_tokens", lambda *args, **kwargs: 10)
    monkeypatch.setattr(agent_loop, "_build_system_prompt", lambda messages, *args, **kwargs: (messages, []))
    sent = []

    async def stream(*args, **kwargs):
        sent.append({schema["function"]["name"] for schema in kwargs.get("tools") or []})
        yield 'data: {"delta": "Готово."}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", stream)

    async def collect():
        return [chunk async for chunk in agent_loop.stream_agent_loop(
            "https://chatgpt.com/backend-api/codex", "gpt-5.6-sol",
            [{"role": "user", "content": "Что тут есть?"}],
            workspace=str(tmp_path), max_rounds=1, _is_teacher_run=True,
        )]

    asyncio.run(collect())

    assert retrieval_calls == []
    assert len(sent) == 1
    assert "read_file" in sent[0]
    assert WRITE_TOOLS.isdisjoint(sent[0])
