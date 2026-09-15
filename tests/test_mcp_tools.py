"""MCP 工具接入测试：schema 转换、调用分发、降级（FakeServer，离线）。

用 FakeMCPParams 模拟 StdioServerParameters，
让 _list_tools / _call_tool 返回预设结果，不依赖真实 MCP server。
"""

import asyncio
import json
from types import SimpleNamespace

from agent.tools.mcp_tools import _extract_text_content, register_mcp_tools
from agent.tools.registry import ToolRegistry


class FakeMCPParams:
    """模拟 mcp.StdioServerParameters：携带预设的工具列表与调用结果。"""

    def __init__(self, tools, call_results=None, list_error=None):
        self.tools = tools
        self.call_results = call_results or {}
        self.list_error = list_error


def _fake_tool(name, description, input_schema):
    return SimpleNamespace(
        name=name, description=description, input_schema=input_schema
    )


def _patch_mcp_client(monkeypatch, params):
    """把 mcp_tools 里的 _list_tools / _call_tool 替换成读 params 的假实现。"""

    async def fake_list(p):
        if p.list_error:
            raise p.list_error
        return p.tools

    async def fake_call(p, name, args):
        result = p.call_results.get(name, SimpleNamespace(
            content=[SimpleNamespace(text="(空)")],
            structured_content=None,
            is_error=False,
        ))
        return result

    monkeypatch.setattr("agent.tools.mcp_tools._list_tools", fake_list)
    monkeypatch.setattr("agent.tools.mcp_tools._call_tool", fake_call)
    # asyncio.run 在 fake 里会直接 await 协程（因为 fake_list/fake_call 是协程）
    monkeypatch.setattr("agent.tools.mcp_tools.asyncio.run", lambda coro: asyncio.get_event_loop().run_until_complete(coro))


def test_register_generates_schemas(monkeypatch):
    tools = [
        _fake_tool("search", "搜索文档", {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        })
    ]
    params = FakeMCPParams(tools)
    _patch_mcp_client(monkeypatch, params)

    reg = ToolRegistry()
    names = register_mcp_tools(reg, params)
    assert names == ["search"]
    assert len(reg.schemas) == 1
    schema = reg.schemas[0]
    assert schema["function"]["name"] == "search"
    assert schema["function"]["description"] == "搜索文档"
    assert schema["function"]["parameters"]["required"] == ["q"]


def test_register_multiple_tools(monkeypatch):
    tools = [
        _fake_tool("tool_a", "工具 A", {"type": "object", "properties": {}}),
        _fake_tool("tool_b", "工具 B", {"type": "object", "properties": {}}),
    ]
    params = FakeMCPParams(tools)
    _patch_mcp_client(monkeypatch, params)

    reg = ToolRegistry()
    names = register_mcp_tools(reg, params)
    assert set(names) == {"tool_a", "tool_b"}
    assert len(reg.schemas) == 2


def test_execute_calls_mcp_tool(monkeypatch):
    tools = [_fake_tool("echo", "回显", {"type": "object", "properties": {"msg": {"type": "string"}}})]
    call_results = {
        "echo": SimpleNamespace(
            content=[SimpleNamespace(text="hello world")],
            structured_content=None,
            is_error=False,
        )
    }
    params = FakeMCPParams(tools, call_results)
    _patch_mcp_client(monkeypatch, params)

    reg = ToolRegistry()
    register_mcp_tools(reg, params)

    call = {"id": "1", "function": {"name": "echo", "arguments": '{"msg": "hi"}'}}
    result = reg.execute(call)
    assert result == "hello world"


def test_execute_uses_structured_content(monkeypatch):
    """有 structured_content 时优先用 JSON 文本（模型可读）。"""
    tools = [_fake_tool("lookup", "查询", {"type": "object", "properties": {}})]
    call_results = {
        "lookup": SimpleNamespace(
            content=[SimpleNamespace(text='{"raw": "text"}')],
            structured_content={"title": "Dune", "year": 1965},
            is_error=False,
        )
    }
    params = FakeMCPParams(tools, call_results)
    _patch_mcp_client(monkeypatch, params)

    reg = ToolRegistry()
    register_mcp_tools(reg, params)

    call = {"id": "1", "function": {"name": "lookup", "arguments": "{}"}}
    result = reg.execute(call)
    assert json.loads(result) == {"title": "Dune", "year": 1965}


def test_execute_error_returns_message(monkeypatch):
    """MCP 工具 is_error=True → 返回 [MCP 工具错误] 前缀，不崩溃。"""
    tools = [_fake_tool("boom", "会失败", {"type": "object", "properties": {}})]
    call_results = {
        "boom": SimpleNamespace(
            content=[SimpleNamespace(text="内部错误: 文件不存在")],
            structured_content=None,
            is_error=True,
        )
    }
    params = FakeMCPParams(tools, call_results)
    _patch_mcp_client(monkeypatch, params)

    reg = ToolRegistry()
    register_mcp_tools(reg, params)

    call = {"id": "1", "function": {"name": "boom", "arguments": "{}"}}
    result = reg.execute(call)
    assert result.startswith("[MCP 工具错误]")
    assert "文件不存在" in result


def test_register_degrades_gracefully_on_connection_failure(monkeypatch):
    """MCP server 不可用时降级：返回空列表，不注册任何工具，不阻塞。"""
    params = FakeMCPParams(tools=[], list_error=ConnectionError("server down"))
    _patch_mcp_client(monkeypatch, params)

    reg = ToolRegistry()
    names = register_mcp_tools(reg, params)
    assert names == []
    assert reg.schemas == []


def test_extract_text_content_handles_text_blocks():
    blocks = [
        SimpleNamespace(text="line1"),
        SimpleNamespace(text="line2"),
        SimpleNamespace(image="binary"),  # 非 text，用类名占位
    ]
    result = _extract_text_content(blocks)
    assert "line1" in result
    assert "line2" in result
    # 非 TextContent 用类名占位
    assert "SimpleNamespace" in result


def test_extract_text_content_empty():
    assert _extract_text_content([]) == "(无输出)"
    assert _extract_text_content(None) == "(无输出)"
