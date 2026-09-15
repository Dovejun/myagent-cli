"""工具注册表单元测试（不依赖真实 API，可离线运行）。"""

import json

from agent.tools.registry import ToolRegistry


def test_register_generates_schema():
    reg = ToolRegistry()
    reg.register(
        "add",
        "两个整数相加",
        {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
        lambda a, b: a + b,
    )
    assert len(reg.schemas) == 1
    schema = reg.schemas[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "add"
    assert schema["function"]["parameters"]["required"] == ["a", "b"]


def test_execute_calls_handler_with_parsed_args():
    reg = ToolRegistry()
    reg.register(
        "add",
        "两个整数相加",
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        lambda a, b: a + b,
    )
    call = {
        "id": "call_1",
        "function": {"name": "add", "arguments": json.dumps({"a": 1, "b": 2})},
    }
    assert reg.execute(call) == "3"


def test_execute_bad_json_returns_error_string():
    reg = ToolRegistry()
    reg.register(
        "add",
        "两个整数相加",
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        lambda a, b: a + b,
    )
    call = {"id": "call_1", "function": {"name": "add", "arguments": "{bad json"}}
    result = reg.execute(call)
    assert isinstance(result, str)
    assert "参数解析失败" in result


def test_execute_handler_error_returns_string():
    reg = ToolRegistry()

    def boom() -> None:
        raise ValueError("boom!")

    reg.register("boom", "故意抛错", {"type": "object", "properties": {}}, boom)
    call = {"id": "call_1", "function": {"name": "boom", "arguments": "{}"}}
    result = reg.execute(call)
    assert isinstance(result, str)
    assert "ValueError" in result


def test_execute_unknown_tool_returns_string():
    reg = ToolRegistry()
    call = {
        "id": "call_1",
        "function": {"name": "no_such_tool", "arguments": "{}"},
    }
    result = reg.execute(call)
    assert isinstance(result, str)
    assert "未知工具" in result


def test_non_str_result_gets_json_encoded():
    reg = ToolRegistry()
    reg.register(
        "get_info",
        "返回 dict",
        {"type": "object", "properties": {}},
        lambda: {"name": "测试", "count": 3},
    )
    call = {"id": "call_1", "function": {"name": "get_info", "arguments": "{}"}}
    result = reg.execute(call)
    assert result == '{"name": "测试", "count": 3}'
