#!/usr/bin/env python3
"""从 harness 骨架生成一个新 Agent CLI 项目（D7 模板脚手架）。

本仓库即模板：agent/ 下全是通用骨架（ReAct 循环 / 三层记忆 / 审计 / 检索 /
审批 / 重试 / MCP / 委派 / 验证钩子），无特定业务痕迹。
scaffold 复制骨架到新目录并替换项目名。

用法：
    python scripts/scaffold.py my-agent --output ~/projects --model deepseek-v4-flash

完成后到新项目里做三处定制即可跑通：
    1. .env：填 API Key / 模型（--model 已写入模板）
    2. 工具：在 config.py 的 _register_builtin_tools 里增删（或接 MCP）
    3. 规格：新工具登记到 AGENTS.md §2 spec 表，跑 check_spec.py 验证
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 复制骨架时排除的路径片段（运行产物 / 个人学习记录 / 密钥）
_EXCLUDE_PARTS = (
    "__pycache__", ".venv", ".git", ".pytest_cache",
    ".pyc", ".jsonl", ".tmp", "memory_store.jsonl",
    "docs/harness-engineer-week1.md",  # 个人一周学习计划
    "docs/week1-notes.md",             # 个人复盘笔记
)
# 需要把 "ai-agent-cli" 替换成新项目名的文本文件
_RENAME_FILES = ("README.md", "AGENTS.md")


def _excluded(rel: str) -> bool:
    # 精确排除根 .env（含真实密钥）；.env.example 模板必须复制
    if rel == ".env":
        return True
    return any(part in rel for part in _EXCLUDE_PARTS)


def scaffold(project_name: str, output_dir: Path) -> Path:
    """把骨架复制到 output_dir/project_name，返回目标目录。"""
    if not project_name.isidentifier():
        raise ValueError(f"项目名必须是合法标识符: {project_name!r}")
    dest = output_dir / project_name
    if dest.exists():
        raise FileExistsError(f"目标目录已存在: {dest}")
    dest.mkdir(parents=True)

    copied = 0
    for src in PROJECT_ROOT.rglob("*"):
        if src.is_dir() or src == PROJECT_ROOT:
            continue
        rel = src.relative_to(PROJECT_ROOT).as_posix()
        if _excluded(rel):
            continue
        target = dest / src.relative_to(PROJECT_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel in _RENAME_FILES:
            # 文本替换项目名后写入
            text = src.read_text(encoding="utf-8").replace("ai-agent-cli", project_name)
            target.write_text(text, encoding="utf-8")
        else:
            shutil.copy2(src, target)
        copied += 1

    print(f"✅ 已生成 {copied} 个文件 → {dest}")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description="生成新的 Agent CLI 项目（基于 harness 骨架）")
    parser.add_argument("name", help="新项目名（合法标识符，如 my-agent）")
    parser.add_argument("--output", default=str(PROJECT_ROOT.parent),
                        help=f"输出目录（默认 {PROJECT_ROOT.parent}）")
    parser.add_argument("--model", default="deepseek-v4-flash",
                        help="写入 .env.example 的默认模型")
    args = parser.parse_args()

    try:
        dest = scaffold(args.name, Path(args.output).expanduser())
    except (ValueError, FileExistsError) as e:
        print(f"✗ {e}")
        return 1

    # 写默认模型到 .env.example
    env = dest / ".env.example"
    env.write_text(
        env.read_text(encoding="utf-8").replace(
            "deepseek-chat", args.model
        ).replace("gpt-4o-mini", args.model),
        encoding="utf-8",
    )
    print("✅ .env.example 默认模型已设为:", args.model)

    print(f"""
下一步（约 5 分钟跑通）：
  1. cd {dest} && python3 -m venv .venv && source .venv/bin/activate
  2. pip install -r requirements.txt && cp .env.example .env  # 填 API Key
  3. 定制工具：编辑 config.py 的 _register_builtin_tools（增删/接 MCP）
  4. 新增工具记得登记 AGENTS.md §2 spec 表，再跑 python scripts/check_spec.py
  5. pytest tests/ -v && python agent_cli.py "你好"
详见 docs/harness-template.md
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
