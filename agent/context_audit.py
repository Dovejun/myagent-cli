"""上下文审计：把"上下文里有什么"变成可观测、可量化的数据。

对应学习清单 D1-1 / D1-2：
- 分层统计：系统提示 / 任务记忆 / 增量摘要 / 窗口区 各自的条数与大小
- token 估算：优先 tiktoken 精确计数，未安装时降级为字符估算（中文约 2 字符/token）
- 报告渲染：生成可打印的审计文本（verbose 模式用）

用法（Agent 内部）：
    request = agent._build_request_messages()
    report = render_report(audit(request))
    console.print(report)
"""

import io

from rich.console import Console
from rich.table import Table

# 与 core.py 中摘要槽/任务记忆消息的前缀保持一致
SUMMARY_PREFIX = "【历史增量摘要】"
MEMORY_PREFIX = "【任务记忆"


def audit(messages: list[dict], model: str = "gpt-4o") -> dict:
    """按层统计消息列表，返回结构化审计数据。

    分层规则（基于 core.py 的 messages 布局）：
        system_prompt : 第一条 system 消息（常驻，永不裁剪）
        task_memory   : 以【任务记忆 开头的 system 消息（长期记忆）
        summary       : 以【历史增量摘要】开头的 system 消息（中期记忆）
        window        : 其余全部（短期窗口区）

    返回示例：
        {
          "layers": {
            "system_prompt": {"count": 1, "chars": 120, "tokens": 60},
            ...
          },
          "total": {"chars": 3000, "tokens": 1500},
        }
    """
    layers = {
        "system_prompt": {"count": 0, "chars": 0, "tokens": 0},
        "task_memory": {"count": 0, "chars": 0, "tokens": 0},
        "summary": {"count": 0, "chars": 0, "tokens": 0},
        "window": {"count": 0, "chars": 0, "tokens": 0},
    }
    for i, m in enumerate(messages):
        content = str(m.get("content") or m.get("tool_calls") or "")
        if i == 0 and m["role"] == "system":
            layer = "system_prompt"
        elif content.startswith(SUMMARY_PREFIX):
            layer = "summary"
        elif content.startswith(MEMORY_PREFIX):
            layer = "task_memory"
        else:
            layer = "window"
        layers[layer]["count"] += 1
        layers[layer]["chars"] += len(content)
        layers[layer]["tokens"] += count_tokens(content, model=model)

    total = {"chars": 0, "tokens": 0}
    for stat in layers.values():
        total["chars"] += stat["chars"]
        total["tokens"] += stat["tokens"]
    return {"layers": layers, "total": total}


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """token 计数：优先 tiktoken 按模型分词，未安装/模型未知时降级字符估算。

    降级规则：chars // 2（对中文较接近真实 token 数，英文偏保守）。
    """
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:  # noqa: BLE001 —— tiktoken 未安装时优雅降级
        return max(1, len(text) // 2)


# 层名 → 中文展示名
_LAYER_NAMES = {
    "system_prompt": "系统提示",
    "task_memory": "任务记忆",
    "summary": "增量摘要",
    "window": "窗口区",
}


def render_report(result: dict, width: int = 80) -> str:
    """把审计结果渲染成 Rich 表格文本（返回字符串，可直接 console.print）。"""
    table = Table(title="上下文审计（token 估算）", width=width)
    table.add_column("层", style="bold")
    table.add_column("条数", justify="right")
    table.add_column("字符", justify="right")
    table.add_column("token", justify="right")

    for key, name in _LAYER_NAMES.items():
        stat = result["layers"][key]
        if stat["count"] == 0:
            continue
        table.add_row(
            name, str(stat["count"]), str(stat["chars"]), str(stat["tokens"])
        )
    t = result["total"]
    table.add_row(
        "[bold]总计[/bold]", "", str(t["chars"]), str(t["tokens"])
    )

    console = Console(file=io.StringIO(), width=width)
    console.print(table)
    return console.file.getvalue()
