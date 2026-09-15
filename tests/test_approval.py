"""工具审批层测试（离线，不依赖 API）。"""

import json

from agent.tools.registry import RISK_CONFIRM, RISK_SAFE, ToolRegistry


def _call(name: str, args: dict | None = None) -> dict:
    return {
        "id": "call_1",
        "function": {"name": name, "arguments": json.dumps(args or {})},
    }


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        "read", "只读工具", {"type": "object", "properties": {}},
        lambda: "content", risk=RISK_SAFE,
    )
    reg.register(
        "write", "写文件工具", {"type": "object", "properties": {}},
        lambda: "written", risk=RISK_CONFIRM,
    )
    return reg


def test_safe_tool_skips_approver():
    """safe 级工具即使有 approver 也不触发审批。"""
    reg = _registry()
    called: list[str] = []

    def approver(name, args):
        called.append(name)
        return True

    assert reg.execute(_call("read"), approver=approver) == "content"
    assert called == []


def test_confirm_approved_executes():
    """confirm 级工具 + approver 返回 True → 正常执行。"""
    reg = _registry()
    result = reg.execute(_call("write"), approver=lambda n, a: True)
    assert result == "written"


def test_confirm_rejected_returns_message():
    """confirm 级工具 + approver 返回 False → [用户拒绝]，不崩溃。"""
    reg = _registry()
    result = reg.execute(_call("write"), approver=lambda n, a: False)
    assert result.startswith("[用户拒绝]")
    assert "write" in result


def test_confirm_without_approver_passes_through():
    """不传 approver（默认 None）→ confirm 工具直通，向后兼容。"""
    reg = _registry()
    assert reg.execute(_call("write")) == "written"


def test_approver_receives_tool_name_and_args():
    """approver 能看到要执行的工具名与参数（审批的依据）。"""
    reg = _registry()
    seen: dict = {}

    def approver(name, args):
        seen["name"] = name
        seen["args"] = args
        return True

    reg.execute(_call("write", {"path": "a.txt"}), approver=approver)
    assert seen["name"] == "write"
    assert seen["args"] == {"path": "a.txt"}


def test_approver_exception_treated_as_reject():
    """approver 自身抛异常 → 按拒绝处理（审批层不崩溃）。"""
    reg = _registry()

    def bad_approver(name, args):
        raise RuntimeError("approver boom")

    result = reg.execute(_call("write"), approver=bad_approver)
    assert result.startswith("[用户拒绝]")
