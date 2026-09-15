"""集中装配（D6）：五方面能力的全部开关收敛到一个配置对象。

AgentConfig（dataclass）声明所有开关，build_agent(config) 完成装配：
- CLI（agent_cli.py）、测试、嵌入其他程序共用同一入口
- 新增能力 = 加一个字段 + build_agent 里一段装配，不碰调用方
- 装配产物（三层记忆 / 审批 / hooks / 知识检索 / MCP / 委派）
  即 Harness 五方面能力在运行时的完整形态

配置来源分层（CLI 参数最简，连接/开关类配置下沉到文件）：
1. .env        —— LLM 密钥/模型、知识库路径、embedding 源、MCP 配置文件路径
2. mcp_servers.json —— MCP server 列表（command/args），默认文件或 MCP_CONFIG 指定
3. AgentConfig —— 编程级覆盖（测试/嵌入场景），CLI 只传行为开关

默认行为（开箱即用）：三层记忆 ✅ 知识检索注入 ✅（知识+向量落库
knowledge_index.json）语义检索：.env 配了 embedding 源即启用（否则退 BM25）。

五方面能力对应的字段：
    D1 上下文工程 : enable_memory / task_memory_budget / window_rounds
                    compress_threshold / archive_path / knowledge_index
    D2 约束设计   : approval / retry_times
    D3 工具编排   : mcp_config_path（JSON）/ delegate（恒启用）
    D4 规格治理   : （无运行时字段，见 AGENTS.md 与 scripts/check_spec.py）
    D5 质量闭环   : hooks
"""

import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from rich.prompt import Prompt

from agent.core import Agent
from agent.knowledge import KnowledgeStore
from agent.llm import LLMClient
from agent.memory import MemoryCompressor, TaskMemory
from agent.prompts import SYSTEM_PROMPT
from agent.tools.delegate_tools import register_delegate_tool
from agent.tools.file_tools import read_file, write_file
from agent.tools.registry import RISK_CONFIRM, RISK_SAFE, ToolRegistry
from agent.tools.shell_tools import run_shell
from agent.validators import run_lint, run_pytest

load_dotenv()  # 让 .env 在装配阶段生效（与 llm.py 的加载一致，重复调用无害）

# 环境变量名（集中声明，文档与代码同源）
ENV_KNOWLEDGE_INDEX = "KNOWLEDGE_INDEX"
ENV_EMBED_MODEL_PATH = "EMBEDDING_MODEL_PATH"   # 本地 HF 模型目录（优先级最高）
ENV_EMBED_BASE_URL = "EMBEDDING_BASE_URL"       # API 通道端点（DashScope/Ollama 等）
ENV_EMBED_MODEL = "EMBEDDING_MODEL"             # API 通道模型名
ENV_MCP_CONFIG = "MCP_CONFIG"                   # MCP servers JSON 路径

DEFAULT_KNOWLEDGE_INDEX = "knowledge_index.json"
DEFAULT_MCP_CONFIG = "mcp_servers.json"

# 基础工具的 OpenAPI schema（保持与 AGENTS.md §2 spec 一致）
_TOOL_SCHEMAS = {
    "read_file": {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "文件绝对或相对路径"}},
        "required": ["path"],
    },
    "write_file": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件绝对或相对路径"},
            "content": {"type": "string", "description": "要写入的完整内容"},
        },
        "required": ["path", "content"],
    },
    "run_shell": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "要执行的完整命令"},
            "timeout": {"type": "integer", "description": "超时秒数，默认 30"},
        },
        "required": ["command"],
    },
}


@dataclass
class AgentConfig:
    """Agent 全部开关。默认值 = 生产可用的推荐配置。

    知识检索默认开启（knowledge_enabled=True），索引路径默认走 .env /
    knowledge_index.json；embedding 与 MCP 均从 .env / JSON 文件读取，
    编程级可在此覆盖（测试/嵌入场景用）。
    """

    # 编排层
    max_steps: int = 15
    verbose: bool = False

    # D2 约束设计
    approval: bool = True          # False = 跳过工具人工审批（危险）
    retry_times: int = 3           # LLM 请求重试次数

    # D1 三层记忆
    enable_memory: bool = True     # False = 退化为朴素 ReAct（无记忆压缩）
    task_memory_budget: int = 1500  # 长期记忆块字符预算
    window_rounds: int = 10        # 短期记忆：保留最近 N 轮完整消息
    compress_threshold: int = 60000  # 总量兜底阈值（字符）
    archive_path: str = "memory_store.jsonl"  # 原文双写存档

    # 知识检索（默认开启；知识+向量落 knowledge_index.json）
    knowledge_enabled: bool = True
    knowledge_index: str = ""      # 覆盖默认/环境变量指定的索引路径（"关"请用 knowledge_enabled=False）

    # L3 语义检索：embedding 源默认从 .env 读取（EMBEDDING_MODEL_PATH /
    # EMBEDDING_BASE_URL + EMBEDDING_MODEL）；以下字段为编程级覆盖
    embedding_url: str = ""
    embedding_model: str = ""

    # D5 质量闭环验证钩子
    hooks: tuple[str, ...] = ()    # 如 ("pytest", "lint")，挂到 write_file

    # D3 工具编排：MCP servers JSON 路径（默认 .env MCP_CONFIG 或 ./mcp_servers.json）
    mcp_config_path: str = ""

    @classmethod
    def from_args(cls, args) -> "AgentConfig":
        """从 argparse.Namespace 构造（只收 CLI 行为开关，无连接配置）。"""
        return cls(
            max_steps=args.max_steps,
            verbose=args.verbose,
            approval=not args.yes,
            retry_times=args.retry,
            enable_memory=not args.no_memory,
            knowledge_enabled=not args.no_knowledge,
            hooks=tuple(h.strip() for h in args.hooks.split(",") if h.strip()),
        )


# 审批的"肯定"回答集合（宽松匹配：忽略大小写与首尾空白）
# 英文：y / yes / yeah；中文：是 / 确认 / 同意 / 可以 / ok 等
_APPROVE_WORDS = {
    "y", "yes", "yeah", "yep", "ok", "sure",
    "是", "是的", "对", "确认", "同意", "可以", "好", "通过", "没问题",
}


def is_approved(answer: str) -> bool:
    """判断审批回答是否为肯定（y/yes/是/确认/ok 等），否则视为拒绝。

    抽成纯函数便于测试；任何不在白名单的输入（n/no/取消/空/乱按）都算拒绝。
    """
    return answer.strip().lower() in _APPROVE_WORDS


def make_approver(approval: bool):
    """构造工具审批回调。

    approval=False → None（全放行）；True → 交互确认（终端提示 [y/N]）。
    通过条件：输入 y / yes / 是 / 确认 / ok 等（见 is_approved）即放行。
    """
    if not approval:
        return None
    return lambda name, tool_args: is_approved(
        Prompt.ask(
            f"[yellow]批准执行[/] {name}({tool_args})?\n"
            "[yellow]输入 y/yes/是/确认 通过，直接回车或输入其他内容拒绝[/]"
        )
    )


def _build_hooks(hook_names: tuple[str, ...]) -> dict[str, list]:
    """把 hooks 配置转成 {工具名: [验证器]}。目前统一挂到 write_file。"""
    fns = []
    if "pytest" in hook_names:
        fns.append(run_pytest)
    if "lint" in hook_names:
        fns.append(run_lint)
    return {"write_file": fns} if fns else {}


def _register_builtin_tools(registry: ToolRegistry) -> None:
    """注册三个基础工具（risk 分级与 AGENTS.md spec 对齐）。"""
    registry.register(
        "read_file", "读取本地文件内容，返回文件全文",
        _TOOL_SCHEMAS["read_file"], read_file,
        risk=RISK_SAFE,
    )
    registry.register(
        "write_file", "写入或覆盖本地文件内容，会自动创建父目录",
        _TOOL_SCHEMAS["write_file"], write_file,
        risk=RISK_CONFIRM,
    )
    registry.register(
        "run_shell", "在本地执行 shell 命令并返回 stdout/stderr/退出码",
        _TOOL_SCHEMAS["run_shell"], run_shell,
        risk=RISK_CONFIRM,
    )


def _load_mcp_servers(path: str) -> list[dict]:
    """从 JSON 文件读取 MCP server 列表。

    文件格式（顶层对象或数组均可）：
        {"servers": [{"name": "fs", "command": "npx", "args": [...]}, ...]}
        或 [{"name": ..., "command": ..., "args": [...]}, ...]
    文件不存在 / 损坏 / 结构非法 → 返回 []（优雅降级，不阻塞启动）。
    """
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(data, dict):
        data = data.get("servers", [])
    if not isinstance(data, list):
        return []
    servers = []
    for s in data:
        if not isinstance(s, dict) or not s.get("command"):
            continue  # 跳过非法项
        servers.append({
            "name": str(s.get("name") or s["command"]),
            "command": s["command"],
            "args": [str(a) for a in s.get("args", [])],
        })
    return servers


def _connect_mcp_servers(registry: ToolRegistry, servers: list[dict]) -> int:
    """逐个接入 MCP server（stdio），返回注册的工具总数。失败优雅降级。"""
    try:
        from mcp import StdioServerParameters

        from agent.tools.mcp_tools import register_mcp_tools
    except ImportError:
        return 0
    total = 0
    for s in servers:
        params = StdioServerParameters(command=s["command"], args=s["args"])
        try:
            total += len(register_mcp_tools(registry, params))
        except Exception:  # noqa: BLE001 —— 单个 server 失败不影响其余
            continue
    return total


def env_embedding_config() -> dict | None:
    """从 .env 读取 embedding 源配置（无配置返回 None = 知识库退 BM25）。

    优先级：EMBEDDING_MODEL_PATH（本地 HF）> EMBEDDING_BASE_URL + EMBEDDING_MODEL（API）
    """
    model_path = os.getenv(ENV_EMBED_MODEL_PATH, "").strip()
    if model_path:
        return {"kind": "local", "model_path": model_path}
    url = os.getenv(ENV_EMBED_BASE_URL, "").strip()
    model = os.getenv(ENV_EMBED_MODEL, "").strip()
    if url and model:
        return {"kind": "api", "base_url": url, "model": model}
    return None


def _resolve_knowledge_index(config: AgentConfig) -> str:
    """知识库索引路径：编程覆盖 > .env > 默认（知识检索默认开启）。"""
    if config.knowledge_index:
        return config.knowledge_index
    return (os.getenv(ENV_KNOWLEDGE_INDEX, "").strip() or DEFAULT_KNOWLEDGE_INDEX)


def _resolve_embedding(config: AgentConfig) -> dict | None:
    """embedding 源：.env > 编程字段 > 无（退 BM25）。"""
    env_cfg = env_embedding_config()
    if env_cfg:
        return env_cfg
    if config.embedding_url and config.embedding_model:
        return {"kind": "api", "base_url": config.embedding_url, "model": config.embedding_model}
    if config.embedding_model:
        return {"kind": "local", "model_path": config.embedding_model}
    return None


def _inject_embedder(knowledge: KnowledgeStore, emb: dict, verbose: bool) -> None:
    """按配置注入 embedder 到知识库；失败打印指引并退回 BM25（不阻塞启动）。"""
    kind = emb.get("kind")
    if kind == "api":
        from agent.embeddings import make_openai_embedder

        try:
            knowledge.embedder = make_openai_embedder(emb["base_url"], emb["model"])
            if verbose:
                print(f"[knowledge] 语义检索已启用(API): {emb['model']}")
        except ValueError as e:
            print(f"[knowledge] 语义检索未启用: {e}")
    elif kind == "local":
        from agent.embeddings import make_local_embedder

        try:
            knowledge.embedder = make_local_embedder(emb["model_path"])
            if verbose:
                print(f"[knowledge] 语义检索已启用(本地HF): {emb['model_path']}")
        except (ImportError, ValueError, OSError) as e:
            # 依赖缺失 / 模型路径不存在：打印指引，退 BM25
            print(f"[knowledge] 本地语义检索未启用: {e}")


def build_agent(config: AgentConfig, approver=None) -> Agent:
    """按配置装配一个完整 Agent（五方面能力全接入，可单开关关闭）。

    approver：自定义审批回调 approver(name, args) -> bool。
    不传则按 config.approval 生成终端交互审批（CLI 行为）；
    Web UI 等交互式前端传入自己的实现（见 webui/server.py 的审批桥）。
    """
    llm = LLMClient(retry_times=config.retry_times)
    registry = ToolRegistry()

    # 基础工具 + 委派工具（审批继承）
    _register_builtin_tools(registry)
    if approver is None:
        approver = make_approver(config.approval)
    register_delegate_tool(
        registry, llm, SYSTEM_PROMPT,
        max_steps=5, approver=approver,
    )

    # D3 工具编排：MCP 从 JSON 配置文件接入（多 server，失败降级）
    # 路径优先级：AgentConfig.mcp_config_path > .env MCP_CONFIG > ./mcp_servers.json
    # （默认文件存在即自动加载，放好 json 就生效，零参数）
    mcp_path = (
        config.mcp_config_path
        or os.getenv(ENV_MCP_CONFIG)
        or (DEFAULT_MCP_CONFIG if os.path.exists(DEFAULT_MCP_CONFIG) else "")
    )
    if mcp_path:
        servers = _load_mcp_servers(mcp_path)
        if servers and config.verbose:
            names = ", ".join(s["name"] for s in servers)
            print(f"[mcp] 配置 {len(servers)} 个 server: {names}")
        _connect_mcp_servers(registry, servers)

    # D1 三层记忆（可选关闭）
    task_memory = compressor = None
    if config.enable_memory:
        task_memory = TaskMemory(llm, budget=config.task_memory_budget)
        # user 信息由 TaskMemory 提炼，压缩层只保留 system（含摘要槽）
        compressor = MemoryCompressor(llm, keep_roles=("system",))

    # 知识检索（默认开启）：知识+向量落库；注入 embedder 启用语义检索
    knowledge = None
    if config.knowledge_enabled:
        knowledge = KnowledgeStore(_resolve_knowledge_index(config))
        emb = _resolve_embedding(config)
        if emb:
            _inject_embedder(knowledge, emb, config.verbose)
        if config.verbose:
            mode = "语义检索" if knowledge.embedder else "BM25"
            print(f"[knowledge] 已启用({mode}): {knowledge.index_path}")

        from agent.tools.knowledge_tools import register_knowledge_tools

        register_knowledge_tools(registry, knowledge, llm)

    # D5 质量闭环验证钩子（可选，默认空 = 行为不变）
    validation_hooks = _build_hooks(config.hooks)

    return Agent(
        llm,
        registry,
        SYSTEM_PROMPT,
        max_steps=config.max_steps,
        verbose=config.verbose,
        task_memory=task_memory,
        compressor=compressor,
        window_rounds=config.window_rounds,
        archive_path=config.archive_path,
        compress_threshold=config.compress_threshold,
        knowledge=knowledge,
        approver=approver,
        validation_hooks=validation_hooks,
    )
