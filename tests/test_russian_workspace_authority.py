import json

import pytest

from src.agent_runtime.authority import create_request_authority, ExactOperation, bind_request_authority
from src.tool_capabilities import ToolRunSecurityContext, capabilities_for_action
from src.tool_approvals import ToolApprovalStore
from src.tool_execution import execute_tool_block
from src.tool_policy import ToolPolicy
from src.tool_types import ToolBlock
from src.turn_contract import requested_capabilities

REQUEST = "Проанализируй игру змейка в своём рабочем каталоге и опиши кратко что в ней надо делать"


@pytest.mark.parametrize("prompt", [REQUEST, "Прочитайте файлы проекта", "Проверь игру в рабочей папке"])
def test_russian_workspace_request_admits_file_family(prompt, tmp_path):
    authority = create_request_authority(prompt, owner="alice", session_id="thread", workspace=str(tmp_path))
    assert requested_capabilities(prompt) == {"shell_files"}
    assert authority.permits(ExactOperation.normalize("read_file", '{"path":"game.js"}'))
    assert authority.permits(ExactOperation.normalize("get_workspace", ""))


@pytest.mark.parametrize("prompt", ["Проанализируй правила игры змейка", "Исправь опечатку в письме",
                                      "Не прочитай файлы проекта", "Расскажи о рабочих каталогах"])
def test_unrelated_or_prohibited_russian_requests_do_not_admit_files(prompt):
    assert "shell_files" not in requested_capabilities(prompt)


async def test_russian_request_reads_files_after_task_approval(tmp_path, monkeypatch):
    from src import tool_execution
    monkeypatch.setattr(tool_execution, "_owner_is_admin", lambda owner: True)
    path = tmp_path / "game.js"
    path.write_text("const snake = [];")
    authority = create_request_authority(REQUEST, owner="alice", session_id="thread", workspace=str(tmp_path))
    store = ToolApprovalStore()
    pending = store.create(owner="alice", session_id="thread", origin_run_id="run", tool_name="bash",
        content="pwd", workspace=str(tmp_path), request_authority=authority,
        capabilities=capabilities_for_action("bash", "pwd"), external_untrusted_context_seen=True)
    approval = store.consume(pending.approval_id, owner="alice", session_id="thread", decision="approve_task")
    # The process producer is covered by the container smoke check. Keep only
    # this spawn stubbed; subsequent file reads use the real dispatcher.
    real_implementation = tool_execution._execute_tool_block_impl
    async def implementation(block, **kwargs):
        if block.tool_type == "bash":
            return "bash", {"exit_code": 0, "output": str(tmp_path)}
        return await real_implementation(block, **kwargs)
    monkeypatch.setattr(tool_execution, "_execute_tool_block_impl", implementation)
    security = ToolRunSecurityContext(external_untrusted_context_seen=True)
    with bind_request_authority(authority):
        _, result = await execute_tool_block(ToolBlock("bash", "pwd"), owner="alice", session_id="thread",
            workspace=str(tmp_path), security_context=security, exact_approval=approval)
        assert result["exit_code"] == 0
        security = ToolRunSecurityContext(external_untrusted_context_seen=True,
            approval_gate_bypassed=approval.allow_remaining_actions)
        for tool, content in [("get_workspace", ""), ("read_file", json.dumps({"path":str(path)}))]:
            _, result = await execute_tool_block(ToolBlock(tool, content), owner="alice", session_id="thread",
                workspace=str(tmp_path), security_context=security)
            assert result["exit_code"] == 0, result


def test_russian_workspace_authority_preserves_explicit_denial(tmp_path):
    authority = create_request_authority(REQUEST, owner="alice", workspace=str(tmp_path),
        policy=ToolPolicy(disabled_tools=frozenset({"read_file"})))
    assert not authority.permits(ExactOperation.normalize("read_file", '{"path":"game.js"}'))

