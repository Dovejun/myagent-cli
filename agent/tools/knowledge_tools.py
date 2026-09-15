"""知识管理工具（L2：让 Agent 自己喂知识/查知识）。

对应 RAG 演进第二层"文档自动处理"，路径 A + 查询能力：
- knowledge_add     ：Agent 直接登记一条知识（如任务复盘经验，路径 B）
- knowledge_ingest  ：读取原始文档 → 分块 → LLM 摘要 → 批量登记（路径 A）
- knowledge_query   ：Agent 任务中途主动检索（区别于 core 的提问注入）

设计决策：
1. risk=safe（免审批）—— 与 write_file 的关键区别：只写【固定的知识库
   JSON 文件】（knowledge_index.json），不触碰用户业务文件，且 remove
   可逆。审批的价值在"破坏性/不可逆"，这里没有；高频写入若每步确认
   会毁掉"Agent 自动学习"的流畅性。（如仍想审批，注册时传 risk=confirm）
2. Agent 生成的知识默认 source=agent / confidence=speculative，
   ingest 的文档块默认 source=document / confidence=pending ——
   与用户主动登记的 confirmed 区分，模型引用时能看到置信度标记
3. 分块无重叠，chunk_size 默认 1500 字符；摘要失败降级为截断原文

用法（config.py 的 build_agent，knowledge 启用时）：
    register_knowledge_tools(registry, knowledge_store, llm)
"""

import re

from ..knowledge import (
    CONF_PENDING,
    CONF_SPECULATIVE,
    SOURCE_AGENT,
    SOURCE_DOCUMENT,
    KnowledgeStore,
)
from .file_tools import read_file

DEFAULT_CHUNK_SIZE = 1500  # 字符/块

# ingest 分块摘要：system 定规则，user 放文本块
_INGEST_SYSTEM = (
    "把用户消息中的文本压缩成 1-2 句要点摘要："
    "保留关键名词、数字、结论与决策，去掉过程细节与客套话。"
    "直接输出摘要文本，不要任何解释、前缀或格式标记。"
)


def register_knowledge_tools(registry, store: KnowledgeStore, llm=None) -> None:
    """注册三个知识管理工具（handler 闭包绑定 store/llm）。

    registry: ToolRegistry；store: KnowledgeStore 实例；
    llm: LLMClient，仅 knowledge_ingest 的分块摘要需要（缺省则 ingest 报错）。
    """
    from ..tools.registry import RISK_SAFE

    # ---------- knowledge_add：登记一条知识（路径 B：复盘/经验） ----------

    def knowledge_add(file_path: str, summary: str, tags: str = "") -> str:
        if not summary.strip():
            return "[登记失败] summary 不能为空"
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        return store.add(
            file_path,
            summary,
            tags=tag_list,
            source=SOURCE_AGENT,
            confidence=CONF_SPECULATIVE,  # Agent 生成 → 推测，待用户确认
        )

    registry.register(
        "knowledge_add",
        "把一条值得长期记住的知识登记进知识库（如刚完成任务的要点、经验、结论）。"
        "登记后后续会话提问可检索到；会自动标记为[推测]，重要条目请用户确认。",
        {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "来源/主题路径，如 'notes/登录模块.md' 或 '经验'"},
                "summary": {"type": "string", "description": "1-2 句要点（保留关键名词/数字/结论）"},
                "tags": {"type": "string", "description": "逗号分隔标签，可选"},
            },
            "required": ["file_path", "summary"],
        },
        knowledge_add,
        risk=RISK_SAFE,
    )

    # ---------- knowledge_query：任务中途主动检索 ----------

    def knowledge_query(query: str, top_k: int = 3) -> str:
        return store.retrieve(query, top_k=top_k) or "[知识库无命中]"

    registry.register(
        "knowledge_query",
        "用 BM25 检索知识库，返回相关条目（含来源与置信度标记）。"
        "当任务需要参考之前的结论/经验/文档要点时使用。",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "要查询的问题或关键词"},
                "top_k": {"type": "integer", "description": "返回条目数，默认 3"},
            },
            "required": ["query"],
        },
        knowledge_query,
        risk=RISK_SAFE,
    )

    # ---------- knowledge_ingest：读文档 → 分块 → 摘要 → 批量登记（路径 A） ----------

    def knowledge_ingest(file_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
        if llm is None:
            return "[摄取失败] 知识库未配置 LLM，无法做分块摘要"
        content = read_file(file_path)
        if content.startswith(("[文件不存在]", "[这是一个目录]", "[读取失败]")):
            return content  # 读不到就把错误回喂，模型自己处理
        chunks = _chunk_text(content, chunk_size)
        if not chunks:
            return f"[摄取] {file_path}: 无有效内容"

        added = 0
        for i, chunk in enumerate(chunks):
            summary = _summarize(llm, chunk) or chunk[:200]
            store.add(
                file_path,
                summary,
                source=SOURCE_DOCUMENT,
                confidence=CONF_PENDING,  # 文档摄取 → 待验证
                chunk_index=i,            # 文档-片段两级结构
            )
            added += 1
        return f"[已摄取] {file_path}: 共 {added} 块已入知识库（source=document）"

    registry.register(
        "knowledge_ingest",
        "读取一个文档，按块切片并生成要点摘要后批量登记进知识库"
        "（文档-片段两级结构，后续提问可按块检索）。适合把长文档/笔记喂给 Agent 记住。",
        {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "要摄取的文档路径"},
                "chunk_size": {"type": "integer", "description": "每块约多少字符，默认 1500"},
            },
            "required": ["file_path"],
        },
        knowledge_ingest,
        risk=RISK_SAFE,
    )


# ---------------------------------------------------------------------------
# 内部工具：分块与摘要
# ---------------------------------------------------------------------------


def _chunk_text(text: str, size: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    """把文本切成约 size 字符的块（无重叠）。

    策略：先按空行分段落，把段落聚合进当前块（不超 size）；
    单段超过 size 时按 size 硬切（长代码/长文兜底）。
    返回非空块列表。
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) <= size:
            buf += p + "\n"
            continue
        if buf:
            chunks.append(buf.strip())
        # 段落本身超 size：硬切，硬切残留再接下一段
        while len(p) > size:
            chunks.append(p[:size].strip())
            p = p[size:]
        buf = p
    if buf:
        chunks.append(buf.strip())
    return [c for c in chunks if c]


def _summarize(llm, chunk: str) -> str:
    """用 LLM 把文本块压成 1-2 句摘要；失败降级返回空串（调用方截断原文兜底）。"""
    try:
        resp = llm.chat(
            [
                {"role": "system", "content": _INGEST_SYSTEM},
                {"role": "user", "content": chunk},
            ]
        )
        return (resp.content or "").strip()
    except Exception:  # noqa: BLE001 —— 摘要失败不阻塞摄取，截断原文兜底
        return ""
