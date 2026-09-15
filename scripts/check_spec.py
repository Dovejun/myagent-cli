#!/usr/bin/env python3
"""规格一致性检查：校验 AGENTS.md 的 spec 表格与代码实际注册的工具是否对齐。

对应学习清单 D4-3（规格治理）：
- 解析 AGENTS.md §2 spec 表格（工具名 / risk / 参数）
- ast 解析代码里所有 registry.register(...) 调用（工具名 / risk / 参数名）
  - 仅能解析字面量注册；变量名注册（如 delegate 的 tool_name 默认参数）
    用正则兜底提取工具名，risk/参数标记为"未知"不参与对比
- 双向对比：文档有代码没有 / 代码有文档没有 / risk 不符 / 参数不符
- 任何差异 → 退出码 1（CI 可直接用）；一致 → 退出码 0

用法（项目根目录）：
    python scripts/check_spec.py
"""

import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENTS_MD = PROJECT_ROOT / "AGENTS.md"
CODE_FILES = [
    PROJECT_ROOT / "agent_cli.py",
    # D6 起基础工具（read_file/write_file/run_shell）注册在 config.py 的
    # _register_builtin_tools()，知识工具在 agent/tools/ 下，两处都要扫
    PROJECT_ROOT / "config.py",
    *sorted((PROJECT_ROOT / "agent" / "tools").glob("*.py")),
]


@dataclass
class ToolSpec:
    name: str
    risk: str | None = None          # None = 未知，跳过校验
    params: set[str] | None = None   # None = 未知，跳过校验


# ---------------------------------------------------------------------------
# 解析 AGENTS.md 的 §2 spec 表格
# ---------------------------------------------------------------------------

def parse_spec_md(md_text: str) -> dict[str, ToolSpec]:
    """从 AGENTS.md 提取 spec 表格。只认 '## 2. spec' 到下一个 '## ' 之间的表。"""
    m = re.search(r"^## 2\..*?spec.*?$(.*?)(?=^## )", md_text, re.M | re.I | re.S)
    if not m:
        return {}
    section = m.group(1)
    tools: dict[str, ToolSpec] = {}
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        name = cells[0].strip("`")
        if not name or name == "工具名" or set(name) <= {"-", " "}:
            continue  # 跳过表头与分隔行
        risk = cells[1] if cells[1] in ("safe", "confirm") else None
        # 参数列：先去括号里的说明（"timeout(可选,默认30)" → "timeout"），
        # 再按逗号/空白切分，去掉 `*`（必填标记）后取参数名
        param_text = re.sub(r"[（(][^）)]*[）)]", "", cells[2].strip("`"))
        raw_params = re.split(r"[,\s]+", param_text)
        params = {p.lstrip("*") for p in raw_params if p}
        tools[name] = ToolSpec(name=name, risk=risk, params=params)
    return tools


# ---------------------------------------------------------------------------
# ast 解析代码里的 registry.register(...) 调用
# ---------------------------------------------------------------------------

def _extract_from_register_call(node: ast.Call) -> ToolSpec | None:
    """从单个 register(...) 调用提取工具规格；字面量缺失则尽量降级。"""
    if not (isinstance(node.func, ast.Attribute) and node.func.attr == "register"):
        return None
    if not node.args:
        return None
    try:
        name = ast.literal_eval(node.args[0])
        if not isinstance(name, str):
            return None
    except ValueError:
        return None  # name 是变量（如 delegate/mcp 的动态注册），ast 拿不到

    risk: str | None = None
    params: set[str] | None = None
    for kw in node.keywords:
        if kw.arg == "risk":
            try:
                risk = ast.literal_eval(kw.value)
            except ValueError:
                pass
    if len(node.args) >= 3 and isinstance(node.args[2], ast.Dict):
        for k, v in zip(node.args[2].keys, node.args[2].values):
            if k is not None:
                try:
                    if ast.literal_eval(k) == "properties" and isinstance(v, ast.Dict):
                        params = {ast.literal_eval(pk) for pk in v.keys if pk is not None}
                except ValueError:
                    pass
    return ToolSpec(name=name, risk=risk, params=params)


def _extract_default_tool_names(source: str) -> set[str]:
    """正则兜底：提取 `tool_name: str = "xxx"` 这类动态注册的默认工具名。"""
    return set(re.findall(r'tool_name:\s*str\s*=\s*"([a-zA-Z0-9_]+)"', source))


def parse_code_file(path: Path) -> dict[str, ToolSpec]:
    """解析单个代码文件里静态可识别的注册工具。"""
    source = path.read_text(encoding="utf-8")
    tools: dict[str, ToolSpec] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return tools
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            spec = _extract_from_register_call(node)
            if spec:
                tools[spec.name] = spec
    # 动态注册兜底：只补名字，risk/params 未知
    for name in _extract_default_tool_names(source):
        if name not in tools:
            tools[name] = ToolSpec(name=name, risk=None, params=None)
    return tools


def parse_code_files(paths: list[Path]) -> dict[str, ToolSpec]:
    all_tools: dict[str, ToolSpec] = {}
    for p in paths:
        if p.exists():
            all_tools.update(parse_code_file(p))
    return all_tools


# ---------------------------------------------------------------------------
# 对比
# ---------------------------------------------------------------------------

def compare(spec_tools: dict[str, ToolSpec], code_tools: dict[str, ToolSpec]) -> list[str]:
    """返回差异列表（空 = 完全一致）。"""
    diffs: list[str] = []
    for name in sorted(set(spec_tools) - set(code_tools)):
        diffs.append(f"[文档有但代码未注册] {name}")
    for name in sorted(set(code_tools) - set(spec_tools)):
        diffs.append(f"[代码有但文档未登记] {name}")
    for name in sorted(set(spec_tools) & set(code_tools)):
        s, c = spec_tools[name], code_tools[name]
        if s.risk and c.risk and s.risk != c.risk:
            diffs.append(f"[risk 不符] {name}: 文档={s.risk} 代码={c.risk}")
        if s.params is not None and c.params is not None and s.params != c.params:
            diffs.append(
                f"[参数不符] {name}: 文档={sorted(s.params)} 代码={sorted(c.params)}"
            )
    return diffs


def main() -> int:
    spec_tools = parse_spec_md(AGENTS_MD.read_text(encoding="utf-8"))
    code_tools = parse_code_files(CODE_FILES)
    diffs = compare(spec_tools, code_tools)

    print(f"spec 表格工具数: {len(spec_tools)}  代码注册工具数: {len(code_tools)}")
    if diffs:
        print("\n发现规格不一致：")
        for d in diffs:
            print(f"  {d}")
        print("\n修复方式：代码与 AGENTS.md §2 spec 表格对齐后重新运行")
        return 1
    print("✅ 规格与代码一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
