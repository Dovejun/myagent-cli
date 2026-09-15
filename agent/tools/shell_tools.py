"""命令执行工具。

跨平台编码处理：早期 MVP 直接 subprocess.run(text=True, encoding="utf-8")，
在 Windows 上执行 cmd 命令（dir、where 等）输出常为 GBK/cp936，
硬解 UTF-8 会产生乱码，模型读到的工具结果不可信。

当前方案：
1. 不设 text=True，先拿原始 bytes
2. 按平台尝试解码：Windows 依次试 gbk → cp936 → utf-8；Unix 用 utf-8
3. 全部失败则 utf-8 + errors="replace" 兜底
"""

import subprocess
import sys


def _decode_output(data: bytes) -> str:
    if sys.platform == "win32":
        for enc in ("gbk", "cp936", "utf-8"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def run_shell(command: str, timeout: int = 30) -> str:
    """执行 shell 命令并返回 stdout/stderr/退出码。"""
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"[命令超时] 超过 {timeout} 秒"

    out = _decode_output(proc.stdout)
    err = _decode_output(proc.stderr)

    parts: list[str] = []
    if out:
        parts.append(out)
    if err:
        parts.append(f"[stderr]\n{err}")
    if proc.returncode != 0:
        parts.append(f"[退出码] {proc.returncode}")
    return "\n".join(parts) if parts else "(无输出)"
