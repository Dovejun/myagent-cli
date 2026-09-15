"""Embedding 注入（RAG L3）：两条通道，同一鸭子协议 embed(texts) -> list[list[float]]。

通道一 OpenAIEmbedder：OpenAI 兼容 /embeddings API（远程或网关）
    - DashScope（国产）：https://dashscope.aliyuncs.com/compatible-mode/v1
    - 本地 OpenAI 兼容网关：Ollama /v1、vLLM、Xinference、LM Studio
通道二 LocalHFEmbedder：HuggingFace 本地模型（transformers + torch）
    - 加载本地模型目录（如 ./dir-embed），与参考实现 localpy/agent.py 相同：
      tokenize(max_length=512) → last_hidden_state → attention_mask 加权 mean pooling
      → L2 归一化
    - 全程离线、数据不出机器；依赖 transformers + torch（可选安装，仅此通道需要）

设计原则：
- embedder 是"可选注入"——KnowledgeStore 不传 embedder 时退回 BM25
- 注入对象是鸭子类型：只要实现 embed(texts: list[str]) -> list[list[float]]
- 两种 embedder 都惰性/可注入加载：未用到对应通道时不强依赖其库

用法（embedding 源在 .env 配置，config.py 自动判定通道）：
    EMBEDDING_MODEL_PATH=本地模型目录          → 本地 HF 通道
    EMBEDDING_BASE_URL + EMBEDDING_MODEL=模型名 → API 通道（无 MODEL_PATH 时）
    编程级可直接 make_openai_embedder / make_local_embedder 构造
"""

import os


# ---------------------------------------------------------------------------
# 通道一：OpenAI 兼容 API
# ---------------------------------------------------------------------------


class OpenAIEmbedder:
    """OpenAI 兼容 embeddings 客户端（DashScope / Ollama / vLLM 通用）。

    embed(texts) 返回与输入等长、等维的向量列表（已按 index 排序，防乱序）。
    """

    def __init__(self, base_url: str, model: str, api_key: str | None = None) -> None:
        from openai import OpenAI  # 惰性导入：未配 embedding 时不增加导入链

        self.model = model
        # key 优先级：显式传参 > OPENAI_EMBEDDING_API_KEY > OPENAI_API_KEY
        key = api_key or os.getenv("OPENAI_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        self.client = OpenAI(api_key=key, base_url=base_url)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量向量化，返回按输入顺序排列的向量列表（空输入返回 []）。"""
        if not texts:
            return []
        resp = self.client.embeddings.create(model=self.model, input=list(texts))
        # 保险：不信任服务端返回顺序，按 index 重排
        by_index = {d.index: d.embedding for d in resp.data}
        return [by_index[i] for i in range(len(texts))]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def make_openai_embedder(
    base_url: str, model: str, api_key: str | None = None
) -> OpenAIEmbedder:
    """工厂：构造 OpenAIEmbedder（校验必填参数）。"""
    if not base_url or not model:
        raise ValueError("make_openai_embedder 需要 base_url 与 model")
    return OpenAIEmbedder(base_url=base_url, model=model, api_key=api_key)


# ---------------------------------------------------------------------------
# 通道二：HuggingFace 本地模型（transformers + torch）
# ---------------------------------------------------------------------------


def _mean_pool_l2(token_embeddings, attention_mask):
    """attention_mask 加权平均池化 + L2 归一（与 localpy/agent.py 的 local_embed 相同）。

    输入必须是 torch.Tensor（last_hidden_state 与对应 attention_mask），
    返回同样设备上的归一化向量张量。
    """
    import torch

    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    pooled = sum_embeddings / sum_mask
    return torch.nn.functional.normalize(pooled, p=2, dim=1)


class LocalHFEmbedder:
    """HuggingFace 本地 embedding 模型（离线、数据不出机器）。

    与 API 通道对齐同一协议：embed(texts) -> list[list[float]]（L2 单位向量）。

    model_path：本地模型目录（AutoModel.from_pretrained 兼容，如 ./dir-embed）。
    依赖（仅此通道）：pip install transformers torch

    测试注入：tokenizer/model/torch_ 三个可选参数用于离线单测，
    生产代码不传（None → 真实加载/导入）。
    """

    def __init__(
        self,
        model_path: str,
        max_length: int = 512,
        device: str | None = None,
        *,
        tokenizer=None,
        model=None,
        torch_=None,
    ) -> None:
        self.model_path = model_path
        self.max_length = max_length

        if torch_ is None:
            import torch

            torch_ = torch
        self.torch = torch_

        if tokenizer is None or model is None:  # 真实加载（本地模型目录）
            if not model_path:
                raise ValueError("LocalHFEmbedder 需要 model_path（本地模型目录）")
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as e:
                raise ImportError(
                    "本地 HF embedding 需要安装依赖：pip install transformers torch"
                ) from e
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path).eval()

        self.tokenizer = tokenizer
        self.model = model

        # 设备：显式指定 > cuda 可用 > cpu（Mac 的 mps 交给用户显式指定）
        self.device = device or (
            "cuda" if getattr(torch_, "cuda", None) and torch_.cuda.is_available() else "cpu"
        )
        if self.device != "cpu":
            self.model.to(self.device)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量向量化（mean pooling + L2 归一，对齐参考实现）。空输入返回 []。"""
        if not texts:
            return []
        t = self.torch
        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with t.no_grad():
            outputs = self.model(**inputs)
            pooled = _mean_pool_l2(outputs.last_hidden_state, inputs["attention_mask"])
        return pooled.cpu().tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def make_local_embedder(
    model_path: str, max_length: int = 512, device: str | None = None
) -> LocalHFEmbedder:
    """工厂：构造本地 HF embedder（校验参数）。

    用法：
        make_local_embedder("./dir-embed")            # 本地目录，自动选设备
        make_local_embedder("./dir-embed", device="mps")  # Mac GPU
    """
    if not model_path:
        raise ValueError("make_local_embedder 需要 model_path（本地模型目录）")
    return LocalHFEmbedder(model_path, max_length=max_length, device=device)
