"""验证器：可在工具执行后自动触发的校验函数（质量闭环的验证点）。

对应学习清单 D5-1：
- 每个验证器签名统一：fn(tool_result: str) -> tuple[bool, str]
  ok       —— 校验是否通过
  output   —— 校验输出（失败时作为 tool 结果的一部分回喂模型，让模型修复）
- 内置验证器：run_pytest（跑测试）、run_lint（跑 ruff，未安装优雅跳过）
- 输出截断：校验输出可能很长，截断防上下文爆炸

用法（Agent 装配时）：
    agent = Agent(..., validation_hooks={
        "write_file": [run_pytest, run_lint],   # 写完代码自动验证
    })
"""

import subprocess
import sys

from .tools.shell_tools import _decode_output

# 单个验证器输出的最大字符数（防止测试失败日志撑爆上下文）
MAX_OUTPUT_CHARS = 4000


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...(输出已截断，共 {len(text)} 字符)"


def _run_cmd(cmd: list[str], timeout: int) -> tuple[bool, str]:
    """执行子进程并返回 (ok, 合并输出)。超时/异常都转成失败结果。"""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=timeout,
            text=False,  # 拿 bytes，复用 shell_tools 的跨平台解码
        )
    except subprocess.TimeoutExpired:
        return False, f"[超时] 命令超过 {timeout} 秒未完成"
    except FileNotFoundError:
        return False, f"[未找到命令] {cmd[0]}（可能未安装）"

    output = (_decode_output(proc.stdout) + "\n" + _decode_output(proc.stderr)).strip()
    return proc.returncode == 0, _truncate(output)


def run_pytest(tool_result: str, path: str = "tests/", timeout: int = 120) -> tuple[bool, str]:
    """验证点：跑 pytest 测试套件（默认 tests/，-x 快速失败，-q 精简输出）。"""
    return _run_cmd([sys.executable, "-m", "pytest", path, "-x", "-q"], timeout)


def run_lint(tool_result: str, path: str = ".", timeout: int = 60) -> tuple[bool, str]:
    """验证点：跑 ruff 静态检查；ruff 未安装时优雅跳过（不阻塞闭环）。"""
    ok, output = _run_cmd([sys.executable, "-m", "ruff", "check", path], timeout)
    if "No module named ruff" in output:
        return True, "[lint 跳过] ruff 未安装"
    return ok, output
