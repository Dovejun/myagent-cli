"""MCP 工具接入：把外部 MCP server 的工具自动注册进 ToolRegistry。

对应学习清单 D3-1：
- 用官方 mcp Python SDK（v2）的 Client，以 stdio 方式启动本地 server 子进程
- 拉取 server 的 tools 列表，把每个 tool 的 input_schema 转成 registry 格式
- 分发时经 client.call_tool(name, args) 执行，结果转成字符串回喂

关键设计：
- MCP SDK 是纯异步的，而 ToolRegistry/Agent 是同步的，
  用 asyncio.run() 在同步调用里桥接异步方法
- 连接是 context manager，不能"连一次用一辈子"，
  每次 execute 时重新建立连接 → 简单但慢；教学版够用
  （生产环境应维持长连接，见模块末"进阶优化"注释）
- schema 转换：MCP tool 的 input_schema 已是 JSON Schema，
  直接套 registry 的 {type:function, function:{name,description,parameters}} 格式

用法（agent_cli.py）：
    from agent.tools.mcp_tools import register_mcp_tools
    from mcp import Client, StdioServerParameters

    params = StdioServerParameters(command="npx", args=["@modelcontextprotocol/server-filesystem", "."])
    register_mcp_tools(registry, params)   # 把 server 的工具都注册进来
"""

import asyncio
import json


def _extract_text_content(content_blocks) -> str:
    """把 MCP call_tool 的 content（ContentBlock 列表）拍平成字符串。

    MCP 的 content 是联合类型（TextContent/ImageContent/...），
    模型只能读文本，所以只提取 TextContent.text，其他类型用 repr 占位。
    """
    parts: list[str] = []
    for block in content_blocks or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
        else:
            parts.append(f"[{type(block).__name__}]")
    return "\n".join(parts) if parts else "(无输出)"


async def _list_tools(params) -> list:
    """异步拉取 MCP server 的工具列表。返回 Tool 对象列表。"""
    from mcp import Client

    async with Client(params) as client:
        result = await client.list_tools()
        return result.tools


async def _call_tool(params, name: str, arguments: dict) -> str:
    """异步调用一个 MCP 工具，返回字符串结果。"""
    from mcp import Client

    async with Client(params) as client:
        result = await client.call_tool(name, arguments)
        if result.is_error:
            # MCP 工具失败不抛异常，而是 is_error=True，把错误文本回喂模型
            return "[MCP 工具错误] " + _extract_text_content(result.content)
        structured = result.structured_content
        if structured is not None:
            # 有结构化输出，转 JSON 字符串（模型可读）
            return json.dumps(structured, ensure_ascii=False)
        # 无结构化输出，用 content 文本
        return _extract_text_content(result.content)


def register_mcp_tools(registry, params, risk="safe") -> list[str]:
    """把一个 MCP server 的所有工具注册进 registry。

    params: mcp.StdioServerParameters（或任何 Client 接受的传输参数）
    risk:   MCP 工具默认 safe（只读语义居多）；如 server 有写操作可传 "confirm"
    返回：成功注册的工具名列表（供 CLI 打印 / 测试断言）

    连接失败时优雅降级：返回空列表，不注册任何工具，不阻塞 Agent 启动
    （降级策略见 docs/fallback-policy.md 场景 B 的精神）。
    """
    try:
        tools = asyncio.run(_list_tools(params))
    except Exception:  # noqa: BLE001 —— MCP server 不可用时降级
        return []

    registered: list[str] = []
    for tool in tools:
        name = tool.name
        description = tool.description or f"MCP tool: {name}"
        parameters = tool.input_schema or {"type": "object", "properties": {}}

        # 闭包绑定 name，避免循环变量延迟求值的老坑
        def _make_handler(tool_name: str):
            def handler(**kwargs):
                return asyncio.run(_call_tool(params, tool_name, kwargs))

            return handler

        registry.register(
            name,
            description,
            parameters,
            _make_handler(name),
            risk=risk,
        )
        registered.append(name)

    return registered


# ──────────────────────────────────────────────────────────────────────
# 进阶优化（生产环境参考，教学版不实现）
# ──────────────────────────────────────────────────────────────────────
# 当前每次 call_tool 都重新建立 stdio 连接（async with Client），
# 简单但慢（每次 fork 子进程 + 协议握手）。
# 生产优化：
#   1. 维持一个长连接的 Client（在独立 event loop 线程里跑）
#   2. execute 时用 run_coroutine_threadsafe 把调用提交到那个 loop
#   3. Agent 关闭时清理连接
# 这样能把 MCP 工具的延迟从 ~200ms 降到 ~5ms。
