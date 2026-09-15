"""知识生命周期测试（L4b）：去重合并、冷条目衰减、热度排序（离线）。"""

from datetime import datetime, timedelta

from agent.knowledge import (
    CONF_CONFIRMED,
    CONF_SPECULATIVE,
    SOURCE_AGENT,
    SOURCE_USER,
    KnowledgeStore,
)

_NOW = datetime.now()


def _ago(days: int) -> str:
    return (_NOW - timedelta(days=days)).isoformat(timespec="seconds")


def _store(tmp_path) -> KnowledgeStore:
    return KnowledgeStore(str(tmp_path / "kb.json"))


# ---------- deduplicate ----------


def test_dedup_removes_identical(tmp_path):
    store = _store(tmp_path)
    store.add("a.md", "登录模块使用 token 刷新机制")
    store.add("b.md", "登录模块使用 token 刷新机制")
    report = store.deduplicate()
    assert "[去重]" in report and "b.md" in report
    assert len(store.entries) == 1
    assert store.entries[0]["source_path"] == "a.md"


def test_dedup_confirmed_wins_over_speculative(tmp_path):
    """同内容条目：confirmed 保留，speculative 移除。"""
    store = _store(tmp_path)
    store.add("a.md", "登录模块使用 token 刷新机制", source=SOURCE_USER, confidence=CONF_CONFIRMED)
    store.add("b.md", "登录模块使用 token 刷新机制", source=SOURCE_AGENT, confidence=CONF_SPECULATIVE)
    store.deduplicate()
    assert len(store.entries) == 1
    assert store.entries[0]["confidence"] == CONF_CONFIRMED
    assert store.entries[0]["source_path"] == "a.md"


def test_dedup_high_similarity_above_threshold(tmp_path):
    """措辞略异但高度相似 → 合并。"""
    store = _store(tmp_path)
    store.add("a.md", "登录模块使用 token 刷新机制")
    store.add("b.md", "登录模块采用 token 刷新机制方案")
    report = store.deduplicate()
    assert "合并移除" in report
    assert len(store.entries) == 1


def test_dedup_distinct_untouched(tmp_path):
    """内容无关的条目不被误合并。"""
    store = _store(tmp_path)
    store.add("a.md", "登录模块使用 token 刷新机制")
    store.add("b.md", "打印机耗材是 HP Smart Tank 750")
    report = store.deduplicate()
    assert "无重复" in report
    assert len(store.entries) == 2


def test_dedup_persists(tmp_path):
    """去重结果落盘，重建实例后仍只有一条。"""
    path = str(tmp_path / "kb.json")
    store1 = KnowledgeStore(path)
    store1.add("a.md", "一样的摘要内容")
    store1.add("b.md", "一样的摘要内容")
    store1.deduplicate()
    store2 = KnowledgeStore(path)
    assert len(store2.entries) == 1


# ---------- decay ----------


def test_decay_dry_run_no_mutation(tmp_path):
    store = _store(tmp_path)
    store.add("old.md", "很久没用的知识")
    store.entries[0]["created_at"] = _ago(200)  # 从未命中 + 入库超期
    report = store.decay(days=90, dry_run=True)
    assert "old.md" in report
    assert store.entries[0]["cold"] is False  # 预览不改


def test_decay_marks_never_hit_old_entry(tmp_path):
    store = _store(tmp_path)
    store.add("old.md", "很久没用的知识")
    store.entries[0]["created_at"] = _ago(200)
    report = store.decay(days=90)
    assert "已标记" in report
    assert store.entries[0]["cold"] is True


def test_decay_marks_entry_with_stale_last_hit(tmp_path):
    """命中过但最近一次命中在 90 天前 → 冷。"""
    store = _store(tmp_path)
    store.add("stale.md", "曾经热门的知识")
    store.entries[0]["created_at"] = _ago(300)
    store.entries[0]["last_hit_at"] = _ago(120)
    store.decay(days=90)
    assert store.entries[0]["cold"] is True


def test_decay_fresh_entry_not_cold(tmp_path):
    """今天入库的条目（即使从未命中）不算冷（冷启动保护）。"""
    store = _store(tmp_path)
    store.add("new.md", "刚收录的知识")
    report = store.decay(days=90)
    assert "无冷条目" in report
    assert store.entries[0]["cold"] is False


def test_decay_recent_hit_not_cold(tmp_path):
    """命中过的且最近仍活跃 → 不冷。"""
    store = _store(tmp_path)
    store.add("active.md", "活跃知识")
    store.entries[0]["created_at"] = _ago(300)
    store.entries[0]["last_hit_at"] = _ago(3)
    store.decay(days=90)
    assert store.entries[0]["cold"] is False


# ---------- 热度排序与 [冷] 标记 ----------


def _two_similar_with_different_hotness(tmp_path):
    """两条内容几乎相同但热度悬殊的条目（a 热门、b 从未命中）。"""
    store = _store(tmp_path)
    store.add("hot.md", "MCP 协议用于工具编排与连接")
    store.add("cold.md", "MCP 协议用于工具编排与连接")
    hot = store.entries[0]   # hot.md（先 add）
    cold = store.entries[1]  # cold.md
    hot["hit_count"] = 10
    hot["last_hit_at"] = _ago(1)
    cold["hit_count"] = 0
    cold["last_hit_at"] = None
    cold["created_at"] = _ago(300)
    return store, hot, cold


def test_hot_entry_ranks_first(tmp_path):
    """BM25 相当时，热门条目排前（热度加权只调序）。"""
    store, _, cold = _two_similar_with_different_hotness(tmp_path)
    store.decay(days=90)  # 把 cold 标记冷
    hits = store.retrieve("MCP 工具")
    assert hits.index("hot.md") < hits.index("cold.md")


def test_cold_entry_has_marker(tmp_path):
    store, _, _ = _two_similar_with_different_hotness(tmp_path)
    store.decay(days=90)
    hits = store.retrieve("MCP 工具")
    assert "cold.md" in hits
    assert "[冷]" in hits  # 冷条目输出带标记


def test_hotness_does_not_create_false_hits(tmp_path):
    """BM25=0 的条目不因热度而出现（热度只作用于相关条目排序）。"""
    store = _store(tmp_path)
    store.add("popular.md", "打印机耗材说明")  # 与查询完全无关
    store.entries[0]["hit_count"] = 999
    store.entries[0]["last_hit_at"] = _ago(0)
    assert store.retrieve("量子计算") == ""


def test_retrieve_bumps_hotness(tmp_path):
    """检索命中后 hit_count 增长（feed 进热度模型的数据）。"""
    store = _store(tmp_path)
    store.add("n.md", "BM25 检索与停用词")
    store.retrieve("BM25 检索")
    store.retrieve("BM25 检索")
    assert store.entries[0]["hit_count"] == 2
