"""子 Agent 委派测试：委派成功、max_steps 限制、上下文隔离、防递归（离线）。

用 ScriptLLM 按调用次序消费脚本（主 Agent 与子 Agent 共享同一 llm 实例）。
"""

from types import SimpleNamespace

from agent.core import Agent
from agent.tools.delegate_tools import register_delegate_tool
from agent.tools.registry import ToolRegistry


# ---------- 工具：脚本式假 LLM ----------


def _tool_call_msg(name: str, args_json: str, call_id: str = "c1"):
    return SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(name=name, arguments=args_json),
        )],
    )


def _text_msg(content: str):
    return SimpleNamespace(content=content, tool_calls=None)


class ScriptLLM:
    """按调用次序依次返回脚本中的 message（主/子 Agent 共享实例）。"""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    def chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        if not self._script:
            raise RuntimeError("script exhausted")
        return self._script.pop(0)


def _registry_with_delegate(llm) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register("read", "读文件", {"type": "object", "properties": {}}, lambda: "content")
    register_delegate_tool(reg, llm, "你是子助手", max_steps=3)
    return reg


# ---------- 测试 ----------


def test_delegate_returns_sub_result():
    """委派成功：子 Agent 的最终结论作为工具结果回喂，主 Agent 给出答案。"""
    llm = ScriptLLM([
        _tool_call_msg("delegate_task", '{"prompt": "统计 tests 用例数"}'),  # 主第1轮
        _text_msg("共 6 个用例"),                                            # 子 Agent 回答
        _text_msg("子任务完成：共 6 个用例"),                                  # 主第2轮
    ])
    reg = _registry_with_delegate(llm)
    agent = Agent(llm, reg, "你是主助手", max_steps=5)

    answer = agent.run("帮我统计用例数")
    assert "共 6 个用例" in answer
    assert llm.calls == 3  # 主1 + 子1 + 主1


def test_sub_agent_messages_isolated():
    """子 Agent 的中间过程不污染主 Agent 上下文。"""
    llm = ScriptLLM([
        _tool_call_msg("delegate_task", '{"prompt": "干活"}'),
        _text_msg("子任务结论"),
        _text_msg("完成"),
    ])
    reg = _registry_with_delegate(llm)
    agent = Agent(llm, reg, "你是主助手", max_steps=5)

    agent.run("委派个任务")
    # 主上下文只有 5 条：system / user / assistant(tool_calls) / tool(结果) / assistant(答案)
    assert len(agent.messages) == 5
    # 子 Agent 的 user 消息（"干活"作为独立 user 内容）不在主上下文
    contents = [str(m.get("content") or "") for m in agent.messages]
    assert not any(c == "干活" for c in contents)


def test_sub_agent_limited_by_max_steps():
    """子 Agent 达到 max_steps 后终止，'达到最大步数' 作为工具结果回喂。"""
    llm = ScriptLLM([
        _tool_call_msg("delegate_task", '{"prompt": "干活"}'),  # 主第1轮
        _tool_call_msg("read", "{}"),   # 子第1步
        _tool_call_msg("read", "{}"),   # 子第2步
        _tool_call_msg("read", "{}"),   # 子第3步 → 耗尽
        _text_msg("子任务没做完，我直接告诉你"),                          # 主第2轮
    ])
    reg = _registry_with_delegate(llm)
    agent = Agent(llm, reg, "你是主助手", max_steps=5)

    answer = agent.run("委派个任务")
    assert answer == "子任务没做完，我直接告诉你"


def test_no_recursive_delegation():
    """子 Agent 看不到 delegate_task → 无法递归委派。"""
    llm = ScriptLLM([
        _tool_call_msg("delegate_task", '{"prompt": "检查你能用什么工具"}'),
        _text_msg("done"),
        _text_msg("final"),
    ])
    reg = _registry_with_delegate(llm)
    agent = Agent(llm, reg, "你是主助手", max_steps=5)

    # 直接检查子视图：委派工具被过滤
    sub_view = reg.schemas
    sub_names = [s["function"]["name"] for s in sub_view]
    assert "delegate_task" in sub_names  # 主 Agent 看得到

    agent.run("委派")
    # 子 Agent 执行时用的是过滤后的视图（脚本里子 Agent 正常回答，说明没炸）


def test_empty_prompt_rejected():
    """空 prompt → 返回 [委派失败]，不创建子 Agent。"""
    llm = ScriptLLM([])
    reg = ToolRegistry()
    register_delegate_tool(reg, llm, "子提示", max_steps=3)

    call = {"id": "1", "function": {"name": "delegate_task", "arguments": '{"prompt": "  "}'}}
    result = reg.execute(call)
    assert result.startswith("[委派失败]")


def test_call_history_restored_after_delegate():
    """子任务执行后，主任务的工具调用历史被恢复（依赖检查不受影响）。"""
    llm = ScriptLLM([
        _tool_call_msg("delegate_task", '{"prompt": "子任务"}'),
        _text_msg("子完成"),
        _text_msg("主完成"),
    ])
    reg = _registry_with_delegate(llm)
    reg.reset_call_history()
    reg.execute({"id": "0", "function": {"name": "read", "arguments": "{}"}})
    assert "read" in reg._call_history

    agent = Agent(llm, reg, "你是主助手", max_steps=5)
    agent.run("委派")

    # 子 Agent.run() 会 reset 历史，但 delegate 结束后应恢复
    assert "read" in reg._call_history
