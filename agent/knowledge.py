"""知识检索（RAG 演进 v3）：BM25 + 可选语义检索 + 结构化条目 + 生命周期。

演进历程：
- v1（D1-3）：命中关键词计数 + 扁平条目（简化版 RAG）
- v2：L1 BM25（词频饱和 + IDF + 长度归一 + 停用词）+ L4a 结构化 Entry
  （source/confidence/chunk_index/热度字段/vector 预留），旧数据自动兼容
- v3：L4b 生命周期（deduplicate 去重 / decay 冷条目）+ L3 语义检索——
  embedder 可选注入（agent/embeddings.py，DashScope/本地 OpenAI 兼容端点），
  retrieve 三模式：auto（BM25+语义 RRF 融合，无 embedder 退 BM25）/
  bm25 / semantic；add 自动向量化、reindex 批量补向量

设计约束（教学定位）：
- 零新依赖：embedder 走已装的 openai SDK（OpenAI 兼容 /embeddings）
- 不传 embedder 时完整退化回 v2 的 BM25 行为（默认路径不变）
- BM25 在查询时对全量条目在线计分（条目量几十~几百，无需倒排索引）

用法：
    store = KnowledgeStore("knowledge_index.json")
    store.add("notes.md", "项目要点：CLI 用 Python，支持审批", source="user")
    store.retrieve("审批怎么实现")                        # auto：无 embedder=BM25
    store.embedder = make_openai_embedder(base_url, model) # 注入后变混合检索
    store.reindex()                                       # 旧数据补向量
    store.retrieve("审批怎么实现", mode="semantic")        # 强制语义
"""

import json
import math
import os
import re
from datetime import datetime, timedelta
from difflib import SequenceMatcher

DEFAULT_INDEX = "knowledge_index.json"

# 知识来源
SOURCE_USER = "user"          # 用户主动登记
SOURCE_AGENT = "agent"        # Agent 自动喂入（L2 工具）
SOURCE_DOCUMENT = "document"  # 文档批量摄取（L2 ingest）

# 置信度
CONF_CONFIRMED = "confirmed"    # 确认事实（用户确认/验证过）
CONF_SPECULATIVE = "speculative"  # 推测（Agent 生成未验证）
CONF_PENDING = "pending"        # 待验证

# 置信度显示名
_CONF_LABEL = {
    CONF_CONFIRMED: "确认",
    CONF_SPECULATIVE: "推测",
    CONF_PENDING: "待验证",
}

# 去重合并时的置信度优先级（高者保留）
_CONF_RANK = {
    CONF_CONFIRMED: 2,
    CONF_SPECULATIVE: 1,
    CONF_PENDING: 0,
}

# 热度参数（L4b）：命中频率满分数、时间衰减半衰期（天）
_HIT_FULL_POPULARITY = 5      # 命中 5 次即视为高频
_HALF_LIFE_DAYS = 45          # 热度半衰期
_COLD_PENALTY = 0.5           # cold 条目的额外降权系数

# BM25 参数（词频饱和与长度归一）
_K1 = 1.2
_B = 0.75

# 停用词表：查询意图词/虚词不参与评分（中英文）
_STOPWORDS = {
    # 中文（双字为主，单字在 bigram 下本就不产生，仍列出兜底）
    "怎么", "如何", "什么", "哪个", "哪些", "为什么", "是否", "能否",
    "实现", "进行", "一个", "这个", "那个", "一下", "请问", "我们",
    "你们", "他们", "就是", "还是", "不是", "没有", "关于", "相关",
    "比如", "例如", "这样", "那样", "然后", "而且", "但是", "如果",
    "因为", "所以", "时候", "地方", "应该", "能够", "可能", "已经",
    "现在", "这里", "那里", "问题", "方法", "方式", "内容", "信息",
    "结果", "需要", "了解", "介绍", "说说", "讲讲", "告诉",
    # 英文
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to",
    "of", "in", "on", "at", "for", "with", "and", "or", "but", "not",
    "no", "how", "what", "why", "when", "where", "which", "who", "do",
    "does", "did", "can", "could", "should", "would", "please", "you",
    "your", "it", "its", "this", "that", "these", "those", "there",
}

# 旧 JSON 缺失字段的默认值（向后兼容归一化）
_FIELD_DEFAULTS = {
    "source": SOURCE_USER,
    "confidence": CONF_CONFIRMED,
    "source_path": None,   # 由 file 回填
    "chunk_index": None,
    "created_at": None,    # 由当前时间回填
    "last_hit_at": None,
    "hit_count": 0,
    "cold": False,         # L4b：decay() 标记的冷条目（长期未命中）
    "vector": None,
}


class KnowledgeStore:
    """结构化知识索引：add / remove / retrieve（BM25），JSON 持久化。"""

    def __init__(
        self,
        index_path: str = DEFAULT_INDEX,
        embedder=None,  # L3：可选注入，embed(texts) -> list[list[float]]
    ) -> None:
        self.index_path = index_path
        self.embedder = embedder
        self.entries: list[dict] = []
        self._load()

    # ------------------------------------------------------------------
    # 索引管理
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """启动时读取 JSON 并归一化旧条目（缺失字段补默认，兼容 v1 数据）。"""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, encoding="utf-8") as f:
                    data = json.load(f)
                self.entries = [self._normalize(e) for e in data.get("entries", [])]
            except (json.JSONDecodeError, OSError):
                self.entries = []  # 索引损坏时降级为空，不阻塞启动

    @staticmethod
    def _normalize(raw: dict) -> dict:
        """把条目补全为标准结构（旧数据/手写数据缺字段时给默认值）。"""
        entry: dict = {**_FIELD_DEFAULTS, **raw}
        # file 与 source_path 双向兼容
        entry.setdefault("file", entry.get("source_path"))
        entry.setdefault("source_path", entry.get("file"))
        if entry["source_path"] is None:
            entry["source_path"] = entry.get("file", "unknown")
        if entry["created_at"] is None:
            entry["created_at"] = datetime.now().isoformat(timespec="seconds")
        entry["tags"] = list(entry.get("tags") or [])
        if entry["vector"] is not None:
            entry["vector"] = list(entry["vector"])  # 防御手写 dict
        return entry

    def _save(self) -> None:
        """把索引写回 JSON（原子性靠临时文件 + replace）。"""
        os.makedirs(os.path.dirname(os.path.abspath(self.index_path)) or ".", exist_ok=True)
        tmp = self.index_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"entries": self.entries}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.index_path)

    @staticmethod
    def _key(source_path: str, chunk_index: int | None) -> str:
        """条目标识：同一文档的同一分块视为同一条（覆盖更新）。"""
        return f"{source_path}#{chunk_index if chunk_index is not None else 0}"

    def add(
        self,
        file_path: str,
        summary: str,
        tags: list[str] | None = None,
        source: str = SOURCE_USER,
        confidence: str = CONF_CONFIRMED,
        chunk_index: int | None = None,
        vector: list[float] | None = None,
    ) -> str:
        """登记一条知识（L4a 结构化版本，旧调用方式完全兼容）。

        覆盖规则：同 (file_path, chunk_index) 覆盖更新；
        不同 chunk_index 共存（支持文档-片段两级结构，为 L2 分块铺路）。
        未显式传 vector 且已注入 embedder 时自动向量化（失败静默退 BM25）。
        """
        if vector is None and summary and self.embedder is not None:
            try:
                vector = self.embedder.embed_one(summary)
            except Exception:  # noqa: BLE001 —— 向量化失败不阻塞登记
                vector = None
        entry = self._normalize({
            "file": file_path,
            "source_path": file_path,
            "chunk_index": chunk_index,
            "summary": summary,
            "tags": list(tags or []),
            "source": source,
            "confidence": confidence,
            "vector": vector,
        })
        key = self._key(file_path, chunk_index)
        self.entries = [e for e in self.entries if self._key(e["source_path"], e["chunk_index"]) != key]
        self.entries.append(entry)
        self._save()
        return f"[已登记] {file_path}" + (f"#{chunk_index}" if chunk_index is not None else "")

    def reindex(self, batch: int = 16) -> str:
        """给缺向量的条目批量补向量（增量重建，L3）。

        已注入 embedder 时：旧数据/手工数据（vector=None）一次性补齐。
        网络中断时保留已补部分并停止。
        """
        if self.embedder is None:
            return "[reindex] 未注入 embedder（语义检索未启用）"
        missing = [e for e in self.entries if not e.get("vector")]
        if not missing:
            return "[reindex] 无缺向量条目"
        done = 0
        for i in range(0, len(missing), batch):
            group = missing[i : i + batch]
            try:
                vecs = self.embedder.embed([str(e.get("summary") or "") for e in group])
            except Exception as e:  # noqa: BLE001 —— 中断保留已补部分
                if done:
                    self._save()
                return f"[reindex] 中断于第 {done} 条: {type(e).__name__}: {e}"
            for e, v in zip(group, vecs):
                e["vector"] = v
                done += 1
        self._save()
        return f"[reindex] 已补 {done}/{len(missing)} 条向量"

    def remove(self, file_path: str) -> str:
        """按来源路径移除该文档的全部条目（含所有分块）。"""
        before = len(self.entries)
        self.entries = [
            e for e in self.entries
            if e["source_path"] != file_path and e.get("file") != file_path
        ]
        if len(self.entries) < before:
            self._save()
            return f"[已移除] {file_path}"
        return f"[未找到] {file_path}"

    def list_entries(self) -> list[str]:
        """列出所有已登记的来源路径（去重）。"""
        return sorted({e["source_path"] or e["file"] for e in self.entries})

    # ------------------------------------------------------------------
    # 检索（BM25）
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3, mode: str = "auto") -> str:
        """检索知识库，返回拼接的相关条目文本；无命中返回空串。

        mode：
        - "auto"    （默认）已注入 embedder → BM25 + 语义 RRF 融合；
                    未注入 → 退化为纯 BM25（零重依赖兜底）
        - "bm25"    强制关键词检索（词频饱和 + IDF + 长度归一 + 热度）
        - "semantic" 强制语义检索（需 embedder，未注入返回提示）

        命中后更新 last_hit_at / hit_count（L4b 热度数据）。
        """
        if mode == "semantic" and self.embedder is None:
            return "[语义检索未启用] 未注入 embedder（需配置 embedding-url/model）"
        if mode == "bm25":
            return self._retrieve_bm25(query, top_k)
        if self.embedder is None:
            return self._retrieve_bm25(query, top_k)  # auto 无 embedder → 退 BM25
        if mode == "semantic":
            return self._retrieve_semantic(query, top_k)
        return self._retrieve_hybrid(query, top_k)    # auto：BM25 + 语义 RRF

    # --- BM25 检索路径 ---

    def _retrieve_bm25(self, query: str, top_k: int) -> str:
        """纯 BM25 路径（与 v2 行为一致，供 bm25 模式与 auto 兜底）。"""
        query_tokens = self._keywords(query)
        if not query_tokens:
            return ""
        ranked = self._bm25_ranked(query_tokens)
        return self._finish(ranked, top_k)

    def _bm25_ranked(self, query_tokens: list[str]) -> list[tuple[float, dict]]:
        """BM25 打分 + 热度加权 → 正分条目降序列表。"""
        docs: list[tuple[dict, list[str]]] = []
        for e in self.entries:
            terms = self._tokenize((e["summary"] or "") + " " + " ".join(e["tags"]))
            if terms:
                docs.append((e, terms))
        if not docs:
            return []

        n = len(docs)
        avgdl = sum(len(terms) for _, terms in docs) / n
        df: dict[str, int] = {}
        for q in set(query_tokens):
            df[q] = sum(1 for _, terms in docs if q in set(terms))

        scored = []
        for e, terms in docs:
            score = self._bm25(query_tokens, terms, df, n, avgdl)
            if score <= 0:
                continue  # BM25 为 0 不产生命中（热度只排序不制造虚假命中）
            scored.append((score * self._hotness_boost(e), e))
        scored.sort(key=lambda x: -x[0])
        return scored

    def _finish(self, ranked: list[tuple[float, dict]], top_k: int) -> str:
        """截取 top_k → bump 热度 → 渲染。"""
        if not ranked:
            return ""
        hits = [e for _, e in ranked[:top_k]]
        self._bump_hits(hits)
        return self._format_hits(hits)

    # --- 语义/混合检索路径（L3） ---

    def _ensure_vectors(self) -> None:
        """混合检索前补齐缺向量条目（异常静默，语义部分跳过无向量条目）。"""
        missing = [e for e in self.entries if not e.get("vector")]
        if not missing:
            return
        try:
            vecs = self.embedder.embed([str(e.get("summary") or "") for e in missing])
            for e, v in zip(missing, vecs):
                e["vector"] = v
            self._save()
        except Exception:  # noqa: BLE001 —— 补不上就跳过，BM25 兜底
            pass

    def _retrieve_semantic(self, query: str, top_k: int) -> str:
        """纯语义路径：查询向量 → 余弦排序（含热度加权）。"""
        try:
            qv = self.embedder.embed_one(query)
        except Exception as e:  # noqa: BLE001 —— 语义服务挂了降级 BM25
            if self.verbose_debug():
                print(f"[knowledge] 语义检索失败({type(e).__name__})，降级 BM25")
            return self._retrieve_bm25(query, top_k)

        self._ensure_vectors()
        scored = []
        for e in self.entries:
            vec = e.get("vector")
            if not vec:
                continue
            cos = self._cosine(vec, qv)
            if cos > 0:
                scored.append((cos * self._hotness_boost(e), e))
        scored.sort(key=lambda x: -x[0])
        return self._finish(scored, top_k)

    def _retrieve_hybrid(self, query: str, top_k: int) -> str:
        """auto 模式：BM25 + 语义双路打分，RRF 无参融合。

        RRF（Reciprocal Rank Fusion）：score = Σ 1/(60 + rank)，
        免调参、对不同量纲的两种打分天然免疫——教学上优于线性加权。
        """
        query_tokens = self._keywords(query)
        if not query_tokens and not query.strip():
            return ""
        try:
            qv = self.embedder.embed_one(query)
        except Exception as e:  # noqa: BLE001 —— 语义服务挂了降级 BM25
            if self.verbose_debug():
                print(f"[knowledge] 语义检索失败({type(e).__name__})，降级 BM25")
            return self._retrieve_bm25(query, top_k)

        self._ensure_vectors()
        bm25_ranked = self._bm25_ranked(query_tokens) if query_tokens else []

        sem_scored = []
        for e in self.entries:
            vec = e.get("vector")
            if not vec:
                continue
            cos = self._cosine(vec, qv)
            if cos > 0:
                sem_scored.append((cos * self._hotness_boost(e), e))  # 热度语义一致
        sem_scored.sort(key=lambda x: -x[0])

        merged = self._rrf_merge(bm25_ranked, sem_scored, limit=top_k * 4)
        return self._finish(merged, top_k)

    @staticmethod
    def _rrf_merge(
        a: list[tuple[float, dict]],
        b: list[tuple[float, dict]],
        k: int = 60,
        limit: int = 40,
    ) -> list[tuple[float, dict]]:
        """Reciprocal Rank Fusion：按名次贡献 1/(k+rank)，融合后降序。"""
        scores: dict[tuple, float] = {}
        entries: dict[tuple, dict] = {}
        for lst in (a, b):
            for pos, (_, e) in enumerate(lst[:limit]):
                key = (e.get("source_path"), e.get("chunk_index"))
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + pos + 1)
                entries[key] = e
        ranked = sorted(
            ((score, entries[key]) for key, score in scores.items()),
            key=lambda x: -x[0],
        )
        return ranked

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        """余弦相似度（0~1）。维度不齐或零向量返回 0。"""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return max(0.0, dot / (na * nb))

    @staticmethod
    def verbose_debug() -> bool:
        """环境变量 KNOWLEDGE_DEBUG=1 时打印降级日志（避免引入 console 依赖）。"""
        return os.getenv("KNOWLEDGE_DEBUG", "") == "1"

    # --- BM25 核心 ---

    @staticmethod
    def _bm25(
        query_tokens: list[str],
        doc_terms: list[str],
        df: dict[str, int],
        n: int,
        avgdl: float,
    ) -> float:
        """对单个文档算 BM25 分数（词频必须来自保留重复的原始词表）。

        score = Σ_qi IDF(qi) * tf(qi,d)*(k1+1) / (tf + k1*(1-b+b*dl/avgdl))
        IDF(qi) = ln((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)   # 平滑，防负
        """
        tf: dict[str, int] = {}
        for t in doc_terms:
            tf[t] = tf.get(t, 0) + 1
        dl = len(doc_terms)

        total = 0.0
        for q in set(query_tokens):
            f = tf.get(q, 0)
            if f <= 0:
                continue  # 文档没有这个词 → 该词贡献 0
            doc_freq = df.get(q, 0)
            if doc_freq <= 0:
                continue
            idf_log = math.log(((n - doc_freq + 0.5) / (doc_freq + 0.5)) + 1)
            denom = f + _K1 * (1 - _B + _B * dl / avgdl) if avgdl > 0 else f + _K1
            total += idf_log * (f * (_K1 + 1)) / denom
        return total

    # --- 热度（L4b） ---

    def _hotness_boost(self, e: dict) -> float:
        """热度乘子 ∈ [0.55, 1.0]（冷条目再乘 _COLD_PENALTY）。

        组成：命中频率（popularity）+ 时间衰减（recency，半衰期 _HALF_LIFE_DAYS）。
        只影响相关条目的相对排序，不产生虚假命中（bm25=0 时乘了还是 0）。
        """
        hits = int(e.get("hit_count") or 0)
        popularity = min(1.0, hits / _HIT_FULL_POPULARITY)

        last = e.get("last_hit_at")
        if not last:
            recency = 0.0  # 从未命中：时间分最低
        else:
            try:
                age_days = (datetime.now() - datetime.fromisoformat(last)).days
                recency = math.exp(-max(age_days, 0) / _HALF_LIFE_DAYS)
            except ValueError:
                recency = 0.0

        boost = 0.55 + 0.25 * popularity + 0.20 * recency  # ∈ [0.55, 1.0]
        if e.get("cold"):
            boost *= _COLD_PENALTY  # 冷条目再降一半
        return boost

    def _bump_hits(self, hits: list[dict]) -> None:
        """命中条目更新 last_hit_at / hit_count 并落盘（L4b 热度数据）。
        命中过的条目自动"回暖"（冷标记保留，但时间分恢复）。"""
        changed = False
        for e in hits:
            e["hit_count"] = int(e.get("hit_count") or 0) + 1
            e["last_hit_at"] = datetime.now().isoformat(timespec="seconds")
            changed = True
        if changed:
            try:
                self._save()
            except OSError:
                pass  # 只读目录等场景：命中统计失败不影响检索

    @staticmethod
    def _format_hits(hits: list[dict]) -> str:
        """把命中的条目渲染成可注入上下文的文本（含来源/冷标记/置信度）。"""
        lines = []
        for e in hits:
            src = e.get("source_path") or e.get("file")
            conf = _CONF_LABEL.get(e.get("confidence"), e.get("confidence"))
            head = f"- {src}"
            if e.get("chunk_index") is not None:
                head += f"#{e['chunk_index']}"
            if e.get("cold"):
                head += " [冷]"  # 长期未命中的冷条目（decay 标记）
            head += f" [{conf}]"
            lines.append(f"{head}: {e['summary']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 知识生命周期（L4b）：去重 / 衰减
    # ------------------------------------------------------------------

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """两条摘要的文本相似度（difflib 字符级 ratio，0~1）。

        无 embedding 时的退化度量：中文摘要句的顺序敏感相似度
        （词集完全相同但顺序不同 → ratio 高；主题不同 → 低）。
        L3 引入向量后可用余弦替换。
        """
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    def _entry_rank(self, e: dict) -> tuple:
        """去重合并时的保留优先级：置信度高 > 创建时间新 > 摘要长。"""
        conf = _CONF_RANK.get(e.get("confidence"), 0)
        return (conf, str(e.get("created_at") or ""), len(str(e.get("summary") or "")))

    def deduplicate(self, threshold: float = 0.75) -> str:
        """语义去重：摘要相似度 ≥ threshold 的条目合并，保留组内最优。

        保留策略（_entry_rank）：确认事实 > 推测 > 待验证；同级留新的；再留长的。
        贪心近似：遍历时与已保留条目比较（组内保留一个最优），
        非精确聚类——对教学知识库规模足够。

        返回操作报告（含被移除的条目标识）。
        """
        remaining: list[dict] = []
        removed: list[str] = []
        for e in self.entries:
            dup = next(
                (
                    keep for keep in remaining
                    if self._similarity(
                        str(e.get("summary") or ""), str(keep.get("summary") or "")
                    ) >= threshold
                ),
                None,
            )
            if dup is None:
                remaining.append(e)
                continue
            # 发现重复：组内保留更优者
            loser = dup if self._entry_rank(e) > self._entry_rank(dup) else e
            if loser is dup:
                remaining.remove(dup)
                remaining.append(e)
            removed.append(self._label(loser))

        if not removed:
            return "[去重] 无重复条目"
        self.entries = remaining
        self._save()
        return "[去重] 合并移除 " + str(len(removed)) + " 条: " + "、".join(removed)

    @staticmethod
    def _label(e: dict) -> str:
        """条目标识（用于报告）：路径#分块。"""
        src = e.get("source_path") or e.get("file") or "?"
        if e.get("chunk_index") is not None:
            return f"{src}#{e['chunk_index']}"
        return src

    def decay(self, days: int = 90, dry_run: bool = False) -> str:
        """冷条目衰减：超过 days 天未被命中 → 标记 cold=True（软降级，不删除）。

        判定规则：
        - 命中过且距 last_hit_at 已超 days → 冷
        - 从未命中且入库（created_at）已超 days → 冷
        - 新条目（< days）不算冷（给冷启动期）
        冷条目在检索排序中额外降权（_hotness_boost × _COLD_PENALTY）
        并带 [冷] 标记，提醒用户清理或确认。
        dry_run=True 只报告不落盘。
        """
        now = datetime.now()
        cold: list[dict] = []
        for e in self.entries:
            last = e.get("last_hit_at")
            created = e.get("created_at")
            if last:
                try:
                    age_days = (now - datetime.fromisoformat(last)).days
                except ValueError:
                    age_days = 0  # 时间格式异常按新鲜处理，不误伤
                if age_days >= days:
                    cold.append(e)
            elif created:
                try:
                    created_days = (now - datetime.fromisoformat(created)).days
                except ValueError:
                    created_days = 0
                if created_days >= days:  # 从未命中且入库超期
                    cold.append(e)

        names = [self._label(e) for e in cold]
        if dry_run:
            if not names:
                return "[decay 预览] 无冷条目"
            return "[decay 预览] 以下条目将标记为冷: " + "、".join(names)

        changed = False
        for e in cold:
            if not e.get("cold"):
                e["cold"] = True
                changed = True
        if changed:
            self._save()
        if not names:
            return "[decay] 无冷条目"
        return f"[decay] 已标记 {len(names)} 条冷条目（检索将降权并带 [冷]）: " + "、".join(names)

    # ------------------------------------------------------------------
    # 分词
    # ------------------------------------------------------------------

    @staticmethod
    def _cjk_bigrams(text: str):
        """对每段连续中文做【滑动】双字切分（保留重复，含任意位置的双字词）。

        滑动 vs 非重叠：非重叠会丢掉奇数位起词（如"推测"在"生成的推测"
        里被切成 生成/的推，永远匹配不到"推测"）；滑动保证任意位置的
        双字词都出现在词表里，文档侧与查询侧匹配一致性最好。
        代价是产生少量跨词噪音对（如"批怎"），由停用词表 + IDF 压制。
        """
        for seg in re.findall(r"[\u4e00-\u9fff]+", text):
            if len(seg) >= 2:
                for i in range(len(seg) - 1):
                    yield seg[i : i + 2]

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        """分词（保留重复，BM25 词频依赖它）：英文词 + 中文滑动 bigram + 停用过滤。"""
        words = [w.lower() for w in re.findall(r"[a-zA-Z0-9_]+", text)]
        bigrams = list(cls._cjk_bigrams(text))
        return [t for t in words + bigrams if t not in _STOPWORDS and len(t) >= 2]

    @classmethod
    def _keywords(cls, text: str) -> list[str]:
        """查询分词：去重保序（查询里同一词出现几次无意义）。"""
        return list(dict.fromkeys(cls._tokenize(text)))
