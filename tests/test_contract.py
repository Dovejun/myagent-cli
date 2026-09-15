"""工具契约校验测试：匹配 / 多参 / 缺参 / 宽松跳过 / 可关闭（离线）。"""

import pytest

from agent.tools.registry import ToolRegistry, validate_contract

SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "content": {"type": "string"},
    },
    "required": ["path", "content"],
}


def test_matching_contract_registers():
    """schema 与 handler 签名一致 → 注册成功。"""
    reg = ToolRegistry()
    reg.register("write", "写", SCHEMA, lambda path, content: "ok")
    assert len(reg.schemas) == 1


def test_schema_extra_param_raises():
    """schema 声明了 handler 不接受的参数 → 注册时 raise（多参）。"""
    bad_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "extra": {"type": "string"}},
    }
    with pytest.raises(ValueError, match="不接受的参数"):
        ToolRegistry().register("write", "写", bad_schema, lambda path: "ok")


def test_required_param_missing_in_schema_raises():
    """handler 必填参数未在 schema 声明 → 注册时 raise（缺参）。"""
    bad_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
    }
    with pytest.raises(ValueError, match="未在 schema 声明"):
        ToolRegistry().register("write", "写", bad_schema, lambda path, content: "ok")


def test_default_value_param_optional_in_schema():
    """有默认值的参数可以不在 schema 声明（如 run_shell 的 timeout）。"""
    reg = ToolRegistry()
    reg.register(
        "shell", "执行", {"type": "object", "properties": {"command": {"type": "string"}}},
        lambda command, timeout=30: "ok",
    )
    assert len(reg.schemas) == 1


def test_var_kwargs_skips_rule1():
    """handler 有 **kwargs → schema 声明的参数都接受，跳过规则 1。"""
    reg = ToolRegistry()
    reg.register(
        "flex", "灵活", {"type": "object", "properties": {"anything": {"type": "string"}}},
        lambda **kwargs: "ok",
    )
    assert len(reg.schemas) == 1


def test_validate_contracts_disabled():
    """ToolRegistry(validate_contracts=False) → 不校验，坏契约也能注册（向后兼容逃生门）。"""
    reg = ToolRegistry(validate_contracts=False)
    bad_schema = {"type": "object", "properties": {"path": {"type": "string"}, "extra": {"type": "string"}}}
    reg.register("write", "写", bad_schema, lambda path: "ok")
    assert len(reg.schemas) == 1


def test_validate_contract_function_direct():
    """直接调用 validate_contract 函数也可用（供外部工具复用）。"""
    validate_contract("ok", {"type": "object", "properties": {"a": {}}}, lambda a: 1)
    with pytest.raises(ValueError):
        validate_contract("bad", {"type": "object", "properties": {"a": {}, "b": {}}}, lambda a: 1)


def test_existing_tools_still_register():
    """回归确认：项目现有的三个内置工具契约仍然匹配（D5-2 不破坏既有功能）。"""
    from agent.tools.file_tools import read_file, write_file
    from agent.tools.shell_tools import run_shell

    reg = ToolRegistry()
    reg.register("read_file", "读", {"type": "object", "properties": {"path": {"type": "string"}}}, read_file)
    reg.register("write_file", "写", {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}, write_file)
    reg.register("run_shell", "跑", {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}}, run_shell)
    assert len(reg.schemas) == 3
