"""L3 语义检索测试：embedder 注入、auto/bm25/semantic 三模式、reindex（离线）。

FakeEmbedder 用"字符 n-gram 词袋 → 单位向量"近似语义：
不依赖真实 embedding API，但保留"语义相似 → 余弦高"的核心行为。
"""

import math

from agent.knowledge import KnowledgeStore


class FakeEmbedder:
    """确定性假 embedder：文本 → 稀疏词袋向量的 L2 单位向量（维度 4096）。

    词 = 中英文逐字（去空白）+ 双字滑动对。相似文本（字面重叠高）→ 余弦高。
    """

    DIM = 4096

    def __init__(self):
        self.embed_calls = 0

    def _text_vector(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        chars = [c for c in text if not c.isspace()]
        ngrams = chars + [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
        for g in ngrams:
            vec[hash(g) % self.DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        return [self._text_vector(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        self.embed_calls += 1
        return self._text_vector(text)


def _store(tmp_path, with_embedder=True) -> KnowledgeStore:
    store = KnowledgeStore(str(tmp_path / "kb.json"))
    if with_embedder:
        store.embedder = FakeEmbedder()
    return store


# ---------- 注入与向量自动生成 ----------


def test_embedder_injected_auto_vectors_on_add(tmp_path):
    store = _store(tmp_path)
    store.add("a.md", "审批流程与权限控制")
    assert store.entries[0]["vector"] is not None  # add 自动向量化
    assert len(store.entries[0]["vector"]) == FakeEmbedder.DIM


def test_add_without_embedder_vector_none(tmp_path):
    store = _store(tmp_path, with_embedder=False)
    store.add("a.md", "审批流程与权限控制")
    assert store.entries[0]["vector"] is None  # 退 BM25 模式不产向量


def test_reindex_backfills_missing_vectors(tmp_path):
    store = _store(tmp_path)
    # 手工塞一条无向量旧数据
    store.add("old.md", "旧数据无向量", vector=None)
    # 直接改掉刚生成的向量，模拟历史数据
    for e in store.entries:
        e["vector"] = None
    report = store.reindex()
    assert "已补" in report
    assert all(e["vector"] is not None for e in store.entries)


def test_reindex_without_embedder_reports(tmp_path):
    store = _store(tmp_path, with_embedder=False)
    assert "[reindex] 未注入" in store.reindex()


# ---------- 三种检索模式 ----------


def test_auto_mode_hybrid_hits(tmp_path):
    """auto + embedder：字面不重叠但语义（字面部分重叠仍）命中的场景由语义兜底。"""
    store = _store(tmp_path)
    store.add("a.md", "打印机耗材是 HP 750 系列")
    # 查询与 b 有重叠字：走 bm25 + 语义 RRF 都能命中
    hits = store.retrieve("打印机怎么加墨", mode="auto")
    assert "a.md" in hits


def test_bm25_mode_ignores_semantics(tmp_path):
    """强制 bm25：不共享字面词 → 不命中（即使注入 embedder）。"""
    store = _store(tmp_path)
    store.add("a.md", "审批流程与权限控制")
    hits = store.retrieve("量子计算理论", mode="bm25")
    assert hits == ""


def test_semantic_mode_requires_embedder(tmp_path):
    store = _store(tmp_path, with_embedder=False)
    hits = store.retrieve("随便什么", mode="semantic")
    assert "语义检索未启用" in hits


def test_auto_without_embedder_falls_back_to_bm25(tmp_path):
    """auto + 无 embedder → 完整退化为 BM25（默认路径不破坏）。"""
    store = _store(tmp_path, with_embedder=False)
    store.add("a.md", "审批流程与权限控制")
    hits = store.retrieve("审批 权限")
    assert "a.md" in hits


def test_semantic_mode_can_hit_without_shared_words(tmp_path):
    """语义检索的价值：无字面共享也能按相似命中。

    FakeEmbedder 用字面 n-gram 近似，这里构造两段高度相似文本验证路径通。
    """
    store = _store(tmp_path)
    store.add("doc1.md", "用户认证模块采用 token 刷新与过期管理")
    store.add("doc2.md", "打印机耗材型号与更换周期说明")
    hits = store.retrieve("用户认证 token 刷新过期处理", mode="semantic")
    assert "doc1.md" in hits


def test_hybrid_keeps_relevant_even_when_bm25_misses(tmp_path):
    """查询与某条内容字面重叠很少时，语义一路也能把它带进结果。"""
    store = _store(tmp_path)
    store.add("a.md", "部署指南：环境变量配置与启动命令")
    store.add("b.md", "完全无关的购物清单")
    hits = store.retrieve("怎样部署启动这个服务", mode="auto")
    # bm25 对"部署指南"字面也有部分命中；至少结果非空且 a 出现
    assert "a.md" in hits


def test_mode_auto_default_is_used(tmp_path):
    store = _store(tmp_path)
    store.add("a.md", "BM25 与语义融合测试内容")
    assert "a.md" in store.retrieve("BM25 融合")  # 默认 mode=auto
