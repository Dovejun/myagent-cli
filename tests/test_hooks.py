"""验证点钩子测试：通过/失败/异常/关闭场景（离线，ScriptLLM + Fake 工具）。"""

from types import SimpleNamespace

from agent.core import Agent
from agent.tools.registry import ToolRegistry


def _tool_call_msg(name: str, args_json: str = "{}", call_id: str = "c1"):
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
    def __init__(self, script):
        self._script = list(script)

    def chat(self, messages, tools=None, **kwargs):
        return self._script.pop(0)


def _hook_ok(result):
    return (True, "all checks passed")


def _hook_fail(result):
    return (False, "pytest: 1 failed, 2 passed")


def _registry(extra_tool=None):
    reg = ToolRegistry()
    reg.register("write", "写文件", {"type": "object", "properties": {}}, lambda: "written")
    if extra_tool:
        reg.register(*extra_tool)
    return reg


def _tool_message(agent, index=3):
    """取第 index 条（默认 tool 消息位置）的内容。"""
    return agent.messages[index]["content"]


def test_hook_pass_appends_marker():
    llm = ScriptLLM([_tool_call_msg("write"), _text_msg("done")])
    agent = Agent(llm, _registry(), "sys", max_steps=5,
                  validation_hooks={"write": [_hook_ok]})
    agent.run("干活")
    assert "[验证通过: _hook_ok]" in _tool_message(agent)


def test_hook_fail_appends_output():
    llm = ScriptLLM([_tool_call_msg("write"), _text_msg("done")])
    agent = Agent(llm, _registry(), "sys", max_steps=5,
                  validation_hooks={"write": [_hook_fail]})
    agent.run("干活")
    content = _tool_message(agent)
    assert "[验证失败: _hook_fail]" in content
    assert "1 failed" in content  # 失败输出回喂，模型可读并修复
    assert "written" in content   # 原工具结果保留


def test_no_hooks_result_unchanged():
    """不配钩子 → tool 结果原样（向后兼容，行为不变）。"""
    llm = ScriptLLM([_tool_call_msg("write"), _text_msg("done")])
    agent = Agent(llm, _registry(), "sys", max_steps=5)
    agent.run("干活")
    assert _tool_message(agent) == "written"


def test_tool_failure_skips_hooks():
    """工具本身出错（抛异常）→ 不跑验证钩子（校验无意义）。"""
    llm = ScriptLLM([_tool_call_msg("boom"), _text_msg("done")])
    reg = _registry(extra_tool=("boom", "会失败", {"type": "object", "properties": {}},
                                lambda: (_ for _ in ()).throw(ValueError("boom"))))
    agent = Agent(llm, reg, "sys", max_steps=5,
                  validation_hooks={"boom": [_hook_ok]})
    agent.run("干活")
    content = _tool_message(agent)
    assert "[工具执行出错]" in content
    assert "[验证通过" not in content and "[验证失败" not in content


def test_hook_exception_treated_as_failure():
    """钩子自身抛异常 → 按失败处理，不崩溃。"""
    def bad_hook(result):
        raise RuntimeError("hook boom")

    llm = ScriptLLM([_tool_call_msg("write"), _text_msg("done")])
    agent = Agent(llm, _registry(), "sys", max_steps=5,
                  validation_hooks={"write": [bad_hook]})
    agent.run("干活")
    content = _tool_message(agent)
    assert "[验证失败: bad_hook]" in content
    assert "hook boom" in content


def test_multiple_hooks_run_in_order():
    """多个钩子顺序执行，结果都附加。"""
    llm = ScriptLLM([_tool_call_msg("write"), _text_msg("done")])
    agent = Agent(llm, _registry(), "sys", max_steps=5,
                  validation_hooks={"write": [_hook_ok, _hook_fail]})
    agent.run("干活")
    content = _tool_message(agent)
    assert "[验证通过: _hook_ok]" in content
    assert "[验证失败: _hook_fail]" in content
    assert content.index("验证通过") < content.index("验证失败")


def test_hooks_not_triggered_for_unregistered_tool():
    """只为配置了的工具跑钩子（read 未配置 → 无附加）。"""
    llm = ScriptLLM([_tool_call_msg("read"), _text_msg("done")])
    reg = ToolRegistry()
    reg.register("read", "读", {"type": "object", "properties": {}}, lambda: "content")
    agent = Agent(llm, reg, "sys", max_steps=5,
                  validation_hooks={"write": [_hook_ok]})
    agent.run("干活")
    assert _tool_message(agent) == "content"
