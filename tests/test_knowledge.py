"""知识检索测试：索引增删、BM25 检索、停用词、结构化字段、JSON 持久化（离线）。

覆盖 v2 升级（L1 检索质量 + L4a 数据模型）与 v1 向后兼容。
"""

import json

from agent.knowledge import (
    CONF_CONFIRMED,
    CONF_SPECULATIVE,
    SOURCE_AGENT,
    SOURCE_USER,
    KnowledgeStore,
)


def test_add_and_retrieve_hit(tmp_path):
    store = KnowledgeStore(str(tmp_path / "index.json"))
    store.add("notes.md", "项目要点：CLI 用 Python，支持工具审批")
    hits = store.retrieve("审批怎么实现")
    assert "notes.md" in hits
    assert "审批" in hits


def test_retrieve_miss_returns_empty(tmp_path):
    store = KnowledgeStore(str(tmp_path / "index.json"))
    store.add("notes.md", "项目要点：CLI 用 Python")
    assert store.retrieve("完全不相关的内容xyz") == ""


def test_add_overwrites_same_path(tmp_path):
    store = KnowledgeStore(str(tmp_path / "index.json"))
    store.add("a.md", "旧摘要")
    store.add("a.md", "新摘要")
    assert len(store.entries) == 1
    assert store.entries[0]["summary"] == "新摘要"


def test_remove(tmp_path):
    store = KnowledgeStore(str(tmp_path / "index.json"))
    store.add("a.md", "摘要")
    assert store.remove("a.md") == "[已移除] a.md"
    assert store.list_entries() == []
    assert store.remove("a.md") == "[未找到] a.md"


def test_persistence_across_instances(tmp_path):
    path = str(tmp_path / "index.json")
    store1 = KnowledgeStore(path)
    store1.add("notes.md", "持久化测试内容")
    # 模拟重启：新实例从磁盘读回
    store2 = KnowledgeStore(path)
    assert store2.list_entries() == ["notes.md"]
    assert "持久化" in store2.retrieve("持久化")


def test_corrupt_index_degrades_gracefully(tmp_path):
    path = tmp_path / "index.json"
    path.write_text("{bad json", encoding="utf-8")
    store = KnowledgeStore(str(path))
    assert store.entries == []
    # 还能正常写入（自愈）
    store.add("a.md", "摘要")
    assert store.list_entries() == ["a.md"]


def test_keywords_extraction():
    words = KnowledgeStore._keywords("CLI 审批 approve python")
    assert "CLI" in words
    assert "审批" in words
    assert "python" in words
    # 去重
    assert words.count("审批") == 1


# ---------- v2：停用词 ----------


def test_stopwords_filtered(tmp_path):
    """查询意图词全是停用词 → 不误召无关文档（滑动 bigram 残留噪音对不命中即空）。"""
    store = KnowledgeStore(str(tmp_path / "index.json"))
    store.add("env.md", "配置 Python 环境")
    assert store.retrieve("怎么实现什么") == ""  # 意图词全被滤掉，噪音对不匹配文档


def test_stopwords_do_not_break_intent():
    """'怎么实现' 是停用词，但意图词 '审批' 保留 → 仍命中。"""
    store = KnowledgeStore("unused.json")
    store.entries = [{"file": "n.md", "source_path": "n.md", "summary": "支持工具审批", "tags": []}]
    hits = store.retrieve("审批怎么实现")
    assert "n.md" in hits


def test_stopwords_filtered_from_docs():
    """停用词不参与文档长度与词频（'实现' 高频不再稀释真实关键词）。"""
    words = KnowledgeStore._keywords("审批功能实现方式")
    assert "实现" not in words  # 停用词
    assert "审批" in words or "功能" in words


# ---------- v2：BM25 排序质量 ----------


def test_bm25_ranks_more_relevant_first(tmp_path):
    store = KnowledgeStore(str(tmp_path / "index.json"))
    store.add("a.md", "配置 Python 环境，安装依赖")
    store.add("b.md", "Python 项目的部署与 Python 版本管理")
    hits = store.retrieve("python")
    # 两条都含 python，但 b 出现 3 次 + 主题更贴 → BM25 词频饱和应让 b 排前
    assert hits.index("b.md") < hits.index("a.md")


def test_bm25_ranks_specific_query_above_noise(tmp_path):
    store = KnowledgeStore(str(tmp_path / "index.json"))
    store.add("a.md", "Python 环境配置教程")
    store.add("b.md", "审批流程与权限控制")
    hits = store.retrieve("权限审批")
    assert "b.md" in hits
    assert "a.md" not in hits  # a 不含任何意图词


# ---------- v2：结构化条目 ----------


def test_add_struct_fields(tmp_path):
    store = KnowledgeStore(str(tmp_path / "index.json"))
    store.add("d.md", "部署文档", source="document", confidence="pending")
    e = store.entries[0]
    assert e["source"] == "document"
    assert e["confidence"] == "pending"
    assert e["source_path"] == "d.md"
    assert e["created_at"]  # 非空
    assert e["hit_count"] == 0
    assert e["vector"] is None


def test_chunk_entries_coexist_and_remove_all(tmp_path):
    store = KnowledgeStore(str(tmp_path / "index.json"))
    store.add("doc.md", "第一章内容", chunk_index=0)
    store.add("doc.md", "第二章内容", chunk_index=1)
    store.add("doc.md", "第三章内容", chunk_index=2)
    assert len(store.entries) == 3  # 不同分块共存
    # 同分块覆盖
    store.add("doc.md", "第一章内容 v2", chunk_index=0)
    assert len(store.entries) == 3
    # remove 清除整个文档
    store.remove("doc.md")
    assert store.list_entries() == []


def test_v1_legacy_data_normalized(tmp_path):
    """旧版扁平条目（无新字段）加载时自动补默认，不崩。"""
    path = tmp_path / "index.json"
    path.write_text(json.dumps({
        "entries": [
            {"file": "old.md", "summary": "旧数据", "tags": ["x"]},
        ]
    }, ensure_ascii=False), encoding="utf-8")
    store = KnowledgeStore(str(path))
    e = store.entries[0]
    assert e["source"] == SOURCE_USER        # 默认来源
    assert e["confidence"] == CONF_CONFIRMED  # 默认置信度
    assert e["source_path"] == "old.md"      # 回填
    assert e["hit_count"] == 0
    assert "old.md" in store.retrieve("旧数据")


def test_hit_tracking(tmp_path):
    """检索命中后 hit_count 递增、last_hit_at 记录（为 L4b 热度铺垫）。"""
    store = KnowledgeStore(str(tmp_path / "index.json"))
    store.add("n.md", "登录模块说明")
    store.retrieve("登录模块")
    assert store.entries[0]["hit_count"] == 1
    assert store.entries[0]["last_hit_at"]
    store.retrieve("登录模块")
    assert store.entries[0]["hit_count"] == 2


def test_confidence_label_in_output(tmp_path):
    """检索输出带置信度标记：[确认]/[推测]/[待验证]。"""
    store = KnowledgeStore(str(tmp_path / "index.json"))
    store.add("ok.md", "已验收的事实", confidence=CONF_CONFIRMED)
    store.add("guess.md", "Agent 生成的推测", source=SOURCE_AGENT, confidence=CONF_SPECULATIVE)
    hits = store.retrieve("推测")
    assert "guess.md" in hits
    assert "[推测]" in hits          # 行内带置信度标记
    assert "Agent 生成的推测" in hits

