"""规格一致性检查测试：解析与对比逻辑（离线，用临时文件）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_spec import compare, parse_code_file, parse_spec_md  # noqa: E402

SAMPLE_MD = """# AGENTS.md

> **版本**: v1.0

## 1. constitution（宪法）

1. 不猜测

## 2. spec（工具规格）

| 工具名 | risk | 参数 | 行为契约 | 副作用 |
|---|---|---|---|---|
| `read_file` | safe | path | 读取文件 | 无 |
| `write_file` | confirm | path, content | 写入文件 | 覆盖磁盘 |
| `run_shell` | confirm | command, timeout(可选,默认30) | 执行命令 | 执行命令 |

## 3. plan（开发路线）

| 阶段 | 内容 |
|---|---|
| MVP | ReAct 循环 |
"""

SAMPLE_CODE = '''
from agent.tools.registry import ToolRegistry

registry = ToolRegistry()
registry.register(
    "read_file",
    "读取本地文件内容",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "路径"},
        },
        "required": ["path"],
    },
    read_file,
    risk="safe",
)
registry.register(
    "write_file",
    "写入文件",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    },
    write_file,
    risk="confirm",
)
registry.register(
    "run_shell",
    "执行命令",
    {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer"},
        },
        "required": ["command"],
    },
    run_shell,
    risk="confirm",
)
'''


def _write(tmp_path, md_text, code_text):
    md = tmp_path / "AGENTS.md"
    code = tmp_path / "code.py"
    md.write_text(md_text, encoding="utf-8")
    code.write_text(code_text, encoding="utf-8")
    return md, code


def test_parse_spec_md_extracts_tools(tmp_path):
    md, _ = _write(tmp_path, SAMPLE_MD, "")
    tools = parse_spec_md(md.read_text(encoding="utf-8"))
    assert set(tools) == {"read_file", "write_file", "run_shell"}
    assert tools["read_file"].risk == "safe"
    assert tools["write_file"].params == {"path", "content"}
    # 参数列里的括号说明要被剥离（"timeout(可选,默认30)" → "timeout"）
    assert tools["run_shell"].params == {"command", "timeout"}
    # 变更日志表格 / plan 表格不被误认（版本号 v1.0 不在结果里）
    assert "v1.0" not in tools


def test_parse_code_file_extracts_static_registrations(tmp_path):
    _, code = _write(tmp_path, "", SAMPLE_CODE)
    tools = parse_code_file(code)
    assert set(tools) == {"read_file", "write_file"}
    assert tools["read_file"].risk == "safe"
    assert tools["read_file"].params == {"path"}
    assert tools["write_file"].params == {"path", "content"}


def test_parse_code_fallback_for_dynamic_tool(tmp_path):
    """tool_name 默认参数的动态注册（delegate）用正则兜底，risk/params 未知。"""
    _, code = _write(tmp_path, "", '''
def register_delegate_tool(registry, tool_name: str = "delegate_task"):
    registry.register(tool_name, "委派", {}, handler)
''')
    tools = parse_code_file(code)
    assert "delegate_task" in tools
    assert tools["delegate_task"].risk is None
    assert tools["delegate_task"].params is None


def test_compare_identical_returns_empty(tmp_path):
    md, code = _write(tmp_path, SAMPLE_MD, SAMPLE_CODE)
    diffs = compare(parse_spec_md(md.read_text("utf-8")), parse_code_file(code))
    assert diffs == []


def test_compare_missing_in_code(tmp_path):
    md, code = _write(tmp_path, SAMPLE_MD, SAMPLE_CODE)
    spec = parse_spec_md(md.read_text("utf-8"))
    spec["ghost_tool"] = spec["read_file"]
    spec["ghost_tool"].name = "ghost_tool"
    diffs = compare(spec, parse_code_file(code))
    assert any("文档有但代码未注册" in d and "ghost_tool" in d for d in diffs)


def test_compare_missing_in_spec(tmp_path):
    md, code = _write(tmp_path, SAMPLE_MD, SAMPLE_CODE)
    code_tools = parse_code_file(code)
    code_tools["undeclared"] = code_tools["read_file"]
    code_tools["undeclared"].name = "undeclared"
    diffs = compare(parse_spec_md(md.read_text("utf-8")), code_tools)
    assert any("代码有但文档未登记" in d and "undeclared" in d for d in diffs)


def test_compare_risk_mismatch(tmp_path):
    md, code = _write(tmp_path, SAMPLE_MD, SAMPLE_CODE)
    code_tools = parse_code_file(code)
    code_tools["read_file"].risk = "confirm"  # 代码里改了 risk 但文档没同步
    diffs = compare(parse_spec_md(md.read_text("utf-8")), code_tools)
    assert any("risk 不符" in d and "read_file" in d for d in diffs)


def test_compare_params_mismatch(tmp_path):
    md, code = _write(tmp_path, SAMPLE_MD, SAMPLE_CODE)
    code_tools = parse_code_file(code)
    code_tools["read_file"].params = {"path", "extra"}  # 代码多了一个参数
    diffs = compare(parse_spec_md(md.read_text("utf-8")), code_tools)
    assert any("参数不符" in d and "read_file" in d for d in diffs)


def test_compare_skips_unknown_risk_and_params(tmp_path):
    """动态注册的工具（risk/params 未知）不参与 risk/参数对比。"""
    md, code = _write(tmp_path, SAMPLE_MD, SAMPLE_CODE)
    spec = parse_spec_md(md.read_text("utf-8"))
    spec["delegate_task"] = spec["read_file"]
    spec["delegate_task"].name = "delegate_task"
    code_tools = parse_code_file(code)
    code_tools["delegate_task"] = code_tools["read_file"]
    code_tools["delegate_task"].name = "delegate_task"
    code_tools["delegate_task"].risk = None
    code_tools["delegate_task"].params = None
    diffs = compare(spec, code_tools)
    assert diffs == []
