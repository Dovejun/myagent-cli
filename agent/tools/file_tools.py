"""文件读写工具。"""

import os


def read_file(path: str) -> str:
    """读取本地文件内容。文件不存在或读取出错时返回可读的错误信息，
    让模型知道发生了什么，而不是抛异常打断循环。"""
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        return f"[文件不存在] {path}"
    if os.path.isdir(path):
        return f"[这是一个目录] {path}"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError as e:
        return f"[读取失败] {e}"


def write_file(path: str, content: str) -> str:
    """写入或覆盖本地文件内容，自动创建父目录。"""
    path = os.path.abspath(os.path.expanduser(path))
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as e:
        return f"[写入失败] {e}"
    return f"[已写入] {path}（{len(content)} 字符）"
