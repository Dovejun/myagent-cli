"""CLI 入口（D6 重构后：只做参数解析与交互，装配全在 config.py）。

用法：
  python agent_cli.py "读 README.md 并总结"     # 单次提问
  python agent_cli.py                           # REPL 交互
  python agent_cli.py -v "..."                  # verbose：工具调用 + 上下文审计
  python agent_cli.py --yes "..."               # 跳过工具审批（危险）
  python agent_cli.py --hooks pytest "..."      # 写文件后自动跑 pytest（质量闭环）
  python agent_cli.py --no-knowledge "..."      # 关闭知识检索（默认开启）
  python agent_cli.py --no-memory "..."         # 关闭三层记忆（朴素 ReAct）
  python agent_cli.py --stream "..."            # 流式输出
  python agent_cli.py --web                     # 启动浏览器聊天界面（Web UI）

连接/开关类配置不在命令行，全部下沉到文件（默认即开）：
  - LLM 密钥/模型        → .env（OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL）
  - 知识检索注入（含向量） → 默认开启，落 knowledge_index.json
  - 语义 embedding       → .env：EMBEDDING_MODEL_PATH=本地 HF 模型目录
                        （或 EMBEDDING_BASE_URL+EMBEDDING_MODEL 走 API）
  - MCP servers          → mcp_servers.json（存在即加载）或 .env MCP_CONFIG 指向
  详见 .env.example 与 README「配置」小节。

五方面能力全部由 config.AgentConfig 集中装配（D6），
入口层只负责三件事：解析参数 → 装配 → Rich 交互展示。
"""

import argparse

from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

from config import AgentConfig, build_agent
from agent.session import load_session, save_session


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("prompt", nargs="?", help="单次提问；省略则进入 REPL 模式")
    parser.add_argument("-v", "--verbose", action="store_true", help="打印工具调用过程 + 上下文审计")
    parser.add_argument("--max-steps", type=int, default=15, help="ReAct 循环上限（默认 15）")
    parser.add_argument("--yes", action="store_true", help="跳过危险工具的人工审批")
    parser.add_argument("--retry", type=int, default=3, help="LLM 请求重试次数（默认 3）")
    parser.add_argument("--hooks", default="", help="验证钩子（逗号分隔：pytest,lint），写文件后自动校验；默认关闭")
    parser.add_argument("--no-knowledge", action="store_true", help="关闭知识检索注入（默认开启）")
    parser.add_argument("--no-memory", action="store_true", help="关闭三层记忆（退化为朴素 ReAct）")
    parser.add_argument("--stream", action="store_true", help="流式输出：答案边生成边显示（不渲染 Markdown）")
    parser.add_argument("--session", default="", help="会话文件路径：REPL 启动时恢复历史、退出时保存快照（如 session.json）")
    parser.add_argument("--web", action="store_true", help="启动 Web UI（浏览器聊天界面，替代终端交互）")
    parser.add_argument("--port", type=int, default=8765, help="Web UI 端口（默认 8765）")
    parser.add_argument("--no-browser", action="store_true", help="Web UI 启动时不自动打开浏览器")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Agent CLI（ReAct + Harness）")
    _add_arguments(parser)
    args = parser.parse_args()

    # Web UI 模式：终端交互整段交给 webui/server.py（配置复用同一套）
    if args.web:
        from webui.server import serve

        serve(
            AgentConfig.from_args(args),
            port=args.port,
            open_browser=not args.no_browser,
        )
        return

    console = Console()
    config = AgentConfig.from_args(args)
    agent = build_agent(config)

    # 会话持久化：REPL 启动时恢复历史、退出时保存快照（单次提问不适用）
    if args.session:
        restored = load_session(agent, args.session)
        if restored:
            console.print(f"[dim]已恢复 {restored} 条会话历史（{args.session}）[/]")

    # 流式输出：最终答案边生成边显示（不渲染 Markdown，直接吐字）
    def _run_and_show(prompt: str) -> None:
        if args.stream:
            agent.run(prompt, stream=True, on_token=lambda t: console.print(t, end=""))
            console.print()  # 补换行
        else:
            answer = agent.run(prompt)
            console.print(Markdown(answer))

    def _save_session() -> None:
        if args.session:
            save_session(agent, args.session)

    # 单次提问模式
    if args.prompt:
        try:
            _run_and_show(args.prompt)
        except KeyboardInterrupt:
            console.print("\n[dim]已中断[/]")
        return

    # REPL 模式：处理 Ctrl+C / Ctrl+D，避免异常退出；退出时保存会话
    console.print("[bold cyan]AI Agent CLI[/] 输入问题开始，Ctrl+C 退出")
    while True:
        try:
            user_input = Prompt.ask("you")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]再见[/]")
            break
        if not user_input.strip():
            continue
        try:
            _run_and_show(user_input)
        except KeyboardInterrupt:
            console.print("\n[dim]已中断[/]")
        except Exception as e:  # noqa: BLE001 —— CLI 层兜底，不让用户看到堆栈
            console.print(f"[red]错误: {e}[/]")
    _save_session()


if __name__ == "__main__":
    main()
