"""知识管理工具测试：add/query/ingest 与分块逻辑（离线，FakeLLM 摘要）。"""

import json
from types import SimpleNamespace

from agent.knowledge import (
    CONF_PENDING,
    CONF_SPECULATIVE,
    SOURCE_AGENT,
    SOURCE_DOCUMENT,
    KnowledgeStore,
)
from agent.tools.knowledge_tools import _chunk_text, register_knowledge_tools
from agent.tools.registry import ToolRegistry


class FakeSummarizeLLM:
    """ingest 摘要用假 LLM：chat 返回固定摘要。"""

    def __init__(self, summary="块要点摘要"):
        self.summary = summary
        self.calls = 0

    def chat(self, messages, tools=None, **kw):
        self.calls += 1
        return SimpleNamespace(content=self.summary, tool_calls=None)


def _registry_with_tools(llm=None, index_path="kb_tool_test.json"):
    store = KnowledgeStore(index_path)
    reg = ToolRegistry()
    register_knowledge_tools(reg, store, llm)
    return reg, store


def _call(name: str, args: dict) -> dict:
    return {
        "id": "c1",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


# ---------- chunk_text ----------


def test_chunk_text_small_single_chunk():
    chunks = _chunk_text("第一段。\n\n第二段。", size=1000)
    assert len(chunks) == 1
    assert "第一段" in chunks[0] and "第二段" in chunks[0]


def test_chunk_text_paragraph_aggregation():
    """小段落聚合到接近 size 再切块（size=9 时约两段一块）。"""
    chunks = _chunk_text("段一内容\n\n段二内容\n\n段三内容", size=9)
    assert len(chunks) == 2  # 前两段一块（8 字符），第三段单独一块
    assert "段一内容" in chunks[0] and "段二内容" in chunks[0]
    assert chunks[1] == "段三内容"


def test_chunk_text_hard_split_long_paragraph():
    """单段超 size → 硬切为多块。"""
    text = "很长的段落" * 50  # 250 字符
    chunks = _chunk_text(text, size=60)
    assert len(chunks) >= 4
    assert "".join(chunks) == text  # 内容不丢失


# ---------- knowledge_add ----------


def test_knowledge_add_marks_agent_speculative(tmp_path):
    reg, store = _registry_with_tools(index_path=str(tmp_path / "kb.json"))
    result = reg.execute(_call("knowledge_add", {"file_path": "经验", "summary": "审批用 y/yes/确认 均可通过", "tags": "审批,交互"}))
    assert "[已登记]" in result
    e = store.entries[0]
    assert e["source"] == SOURCE_AGENT
    assert e["confidence"] == CONF_SPECULATIVE
    assert e["tags"] == ["审批", "交互"]
    # 检索命中（content 可被注入）
    assert store.retrieve("审批 交互") != ""


def test_knowledge_add_empty_summary_rejected(tmp_path):
    reg, _ = _registry_with_tools(index_path=str(tmp_path / "kb.json"))
    result = reg.execute(_call("knowledge_add", {"file_path": "x", "summary": "  "}))
    assert "不能为空" in result


# ---------- knowledge_query ----------


def test_knowledge_query_hit_and_miss(tmp_path):
    reg, store = _registry_with_tools(index_path=str(tmp_path / "kb.json"))
    store.add("notes.md", "MCP 协议用于工具编排", source=SOURCE_DOCUMENT)
    result = reg.execute(_call("knowledge_query", {"query": "MCP 工具"}))
    assert "notes.md" in result
    result2 = reg.execute(_call("knowledge_query", {"query": "量子纠缠理论"}))
    assert "[知识库无命中]" in result2


# ---------- knowledge_ingest ----------


def test_knowledge_ingest_chunks_and_registers(tmp_path):
    fake = FakeSummarizeLLM()
    reg, store = _registry_with_tools(llm=fake, index_path=str(tmp_path / "kb.json"))
    doc = tmp_path / "long_doc.md"
    doc.write_text("第一块内容\n\n第二块内容\n\n第三块内容", encoding="utf-8")

    result = reg.execute(_call("knowledge_ingest", {"file_path": str(doc), "chunk_size": 8}))
    assert "[已摄取]" in result
    assert "3 块" in result
    assert len(store.entries) == 3
    # 文档-片段两级结构：chunk_index 递增、source=document、confidence=pending
    for i, e in enumerate(store.entries):
        assert e["chunk_index"] == i
        assert e["source"] == SOURCE_DOCUMENT
        assert e["confidence"] == CONF_PENDING
        assert e["summary"] == "块要点摘要"  # FakeLLM 摘要
    assert fake.calls == 3  # 每块一次摘要


def test_knowledge_ingest_file_missing_returns_error(tmp_path):
    reg, _ = _registry_with_tools(llm=FakeSummarizeLLM(), index_path=str(tmp_path / "kb.json"))
    result = reg.execute(_call("knowledge_ingest", {"file_path": "/no/such/file.md"}))
    assert result.startswith("[文件不存在]")


def test_knowledge_ingest_without_llm_reports(tmp_path):
    reg, _ = _registry_with_tools(llm=None, index_path=str(tmp_path / "kb.json"))
    doc = tmp_path / "a.md"
    doc.write_text("内容", encoding="utf-8")
    result = reg.execute(_call("knowledge_ingest", {"file_path": str(doc)}))
    assert "[摄取失败]" in result  # 未配 LLM 时明确报错，不静默


def test_ingest_summarize_failure_falls_back_to_truncation(tmp_path):
    """摘要失败（异常）→ 用原文截断兜底，摄取不中断。"""

    class BrokenLLM:
        def chat(self, messages, tools=None, **kw):
            raise RuntimeError("api down")

    reg, store = _registry_with_tools(llm=BrokenLLM(), index_path=str(tmp_path / "kb.json"))
    doc = tmp_path / "b.md"
    doc.write_text("内容A\n\n内容B", encoding="utf-8")
    result = reg.execute(_call("knowledge_ingest", {"file_path": str(doc)}))
    assert "[已摄取]" in result
    assert len(store.entries) == 2  # 用截断原文兜底，不中断
    assert "内容A" in store.entries[0]["summary"]


# ---------- registry schema ----------


def test_three_tools_registered(tmp_path):
    reg, _ = _registry_with_tools(llm=FakeSummarizeLLM(), index_path=str(tmp_path / "kb.json"))
    names = {s["function"]["name"] for s in reg.schemas}
    assert {"knowledge_add", "knowledge_query", "knowledge_ingest"} <= names
