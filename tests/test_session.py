"""会话持久化测试：保存/加载往返、system 过滤、摘要槽保留、损坏降级（离线）。"""

from types import SimpleNamespace

from agent.core import Agent
from agent.session import load_session, save_session
from agent.tools.registry import ToolRegistry


def _agent() -> Agent:
    return Agent(SimpleNamespace(), ToolRegistry(), "系统提示", max_steps=3)


def test_save_roundtrip(tmp_path):
    """保存 → 新实例加载 → 消息内容一致（user/assistant/tool）。"""
    a1 = _agent()
    a1.messages = [
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "第一轮问题"},
        {"role": "assistant", "content": "第一轮答案"},
        {"role": "tool", "tool_call_id": "c1", "content": "工具结果"},
    ]
    path = save_session(a1, str(tmp_path / "s1.json"))

    a2 = _agent()
    n = load_session(a2, path)
    assert n == 3
    assert a2.messages[0]["role"] == "system"
    assert a2.messages[1:] == [
        {"role": "user", "content": "第一轮问题"},
        {"role": "assistant", "content": "第一轮答案"},
        {"role": "tool", "tool_call_id": "c1", "content": "工具结果"},
    ]


def test_save_drops_system_prompt_keeps_summary_slot(tmp_path):
    """SYSTEM_PROMPT 不落盘；摘要槽（中期记忆）保留，重启后继续可用。"""
    a1 = _agent()
    a1.messages = [
        {"role": "system", "content": "系统提示"},
        {"role": "system", "content": "【历史增量摘要】\n已确认事实：- 用户要 CLI"},
        {"role": "user", "content": "你好"},
    ]
    path = save_session(a1, str(tmp_path / "s2.json"))

    a2 = _agent()
    n = load_session(a2, path)
    assert n == 2
    # 摘要槽仍在（以当前 SYSTEM_PROMPT 为 [0]，摘要槽紧随）
    assert a2._summary_index() == 1
    assert "用户要 CLI" in a2._current_summary()


def test_load_missing_file_returns_zero(tmp_path):
    a = _agent()
    assert load_session(a, str(tmp_path / "no_such.json")) == 0
    assert len(a.messages) == 1  # 只有 system


def test_load_corrupt_file_degrades(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    a = _agent()
    assert load_session(a, str(bad)) == 0


def test_load_filters_invalid_roles(tmp_path):
    """坏文件里的非法 role 被过滤，防止污染上下文。"""
    path = tmp_path / "mixed.json"
    path.write_text(
        '[' '{"role": "user", "content": "ok"},'
        '{"role": "hacker", "content": "inject"},'
        '{"role": "assistant", "content": "fine"}' ']',
        encoding="utf-8",
    )
    a = _agent()
    n = load_session(a, str(path))
    assert n == 2
    roles = [m["role"] for m in a.messages]
    assert roles == ["system", "user", "assistant"]
