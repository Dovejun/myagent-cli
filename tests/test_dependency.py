"""工具依赖声明测试：前置工具检查、调用历史、任务隔离（离线）。"""

import json

from agent.tools.registry import ToolRegistry


def _call(name: str, args: dict | None = None) -> dict:
    return {
        "id": "call_1",
        "function": {"name": name, "arguments": json.dumps(args or {})},
    }


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register("read", "读文件", {"type": "object", "properties": {}}, lambda: "content")
    reg.register(
        "write", "写文件", {"type": "object", "properties": {}}, lambda: "written",
        depends_on=("read",),
    )
    return reg


def test_dependency_satisfied_after_read():
    """先 read 再 write → write 正常执行。"""
    reg = _registry()
    reg.reset_call_history()
    assert reg.execute(_call("read")) == "content"
    assert reg.execute(_call("write")) == "written"


def test_dependency_not_satisfied():
    """未 read 直接 write → 返回 [前置工具未调用]，不执行。"""
    reg = _registry()
    reg.reset_call_history()
    result = reg.execute(_call("write"))
    assert result.startswith("[前置工具未调用]")
    assert "read" in result


def test_call_history_resets_per_task():
    """上一次任务里 read 过，新任务里再 write 仍需先 read。"""
    reg = _registry()
    reg.reset_call_history()
    reg.execute(_call("read"))
    reg.execute(_call("write"))  # 这次通过
    # 新任务开始：reset 后依赖重新生效
    reg.reset_call_history()
    result = reg.execute(_call("write"))
    assert result.startswith("[前置工具未调用]")


def test_failed_call_not_recorded():
    """工具执行失败（抛异常）不算"已调用"——依赖仍不满足。"""
    reg = ToolRegistry()
    reg.register("flaky", "会失败", {"type": "object", "properties": {}},
                 lambda: (_ for _ in ()).throw(ValueError("boom")))
    reg.register("next", "依赖", {"type": "object", "properties": {}}, lambda: "ok",
                 depends_on=("flaky",))
    reg.reset_call_history()
    flaky_result = reg.execute(_call("flaky"))
    assert "ValueError" in flaky_result  # 失败回喂
    next_result = reg.execute(_call("next"))
    assert next_result.startswith("[前置工具未调用]")


def test_no_dependency_passes_directly():
    """无 depends_on 的工具直接执行。"""
    reg = ToolRegistry()
    reg.register("simple", "简单工具", {"type": "object", "properties": {}}, lambda: "ok")
    reg.reset_call_history()
    assert reg.execute(_call("simple")) == "ok"


def test_multiple_dependencies():
    """多依赖：必须全部调用过才放行。"""
    reg = ToolRegistry()
    reg.register("a", "工具A", {"type": "object", "properties": {}}, lambda: "a")
    reg.register("b", "工具B", {"type": "object", "properties": {}}, lambda: "b")
    reg.register("c", "依赖AB", {"type": "object", "properties": {}}, lambda: "c",
                 depends_on=("a", "b"))
    reg.reset_call_history()
    reg.execute(_call("a"))
    result = reg.execute(_call("c"))  # 缺 b
    assert result.startswith("[前置工具未调用]")
    assert "b" in result
    reg.execute(_call("b"))
    assert reg.execute(_call("c")) == "c"  # 现在都齐了
