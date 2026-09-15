"""本地 HF Embedder 测试：embed 流程、参数传递、错误路径（离线，Fake 组件注入）。

不依赖真实 transformers/torch：
- tokenizer/model/torch_ 通过构造参数注入 Fake
- _mean_pool_l2 通过 monkeypatch 替换（torch 数学由真实库保证，真机验证）
验证的是"与本地 localpy/agent.py 相同方式"的调用契约：
tokenize(max_length=512, padding, truncation, return_tensors=pt) →
model(**inputs) → last_hidden_state + attention_mask → mean pool + L2 归一
"""

from contextlib import contextmanager

import pytest

import agent.embeddings as emb


# ---------- Fake 组件 ----------


class FakeTensor:
    """支持 .to / .cpu / .tolist 的最小张量替身。"""

    def __init__(self, data=None):
        self._data = data or [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    def to(self, device):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._data


class FakeTorch:
    def __init__(self):
        self.no_grad_calls = 0

    @contextmanager
    def no_grad(self):
        self.no_grad_calls += 1
        yield


class _FakeCudaModule:
    """torch.cuda 替身：is_available 由实例控制。"""

    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append({"texts": texts, **kwargs})
        return {"input_ids": FakeTensor(), "attention_mask": FakeTensor()}


class FakeHFModel:
    def __init__(self):
        self.calls = []
        self.device = None

    def __call__(self, **inputs):
        self.calls.append(inputs)
        return type("Out", (), {"last_hidden_state": FakeTensor()})()

    def to(self, device):
        self.device = device
        return self


def _make_embedder(**kw):
    defaults = dict(
        model_path="./dir-embed",
        device="cpu",
        tokenizer=FakeTokenizer(),
        model=FakeHFModel(),
        torch_=FakeTorch(),
    )
    defaults.update(kw)
    return emb.LocalHFEmbedder(**defaults)


# ---------- 工厂 ----------


def test_make_local_embedder_requires_path():
    with pytest.raises(ValueError):
        emb.make_local_embedder("")


# ---------- embed 流程（协议与参数） ----------


def test_embed_empty_returns_empty(monkeypatch):
    e = _make_embedder()
    assert e.embed([]) == []
    assert e.tokenizer.calls == []  # 空输入不触发模型


def test_embed_tokenizes_with_padding_truncation_max512(monkeypatch):
    """与 localpy/agent.py 一致：padding/truncation/max_length=512/return_tensors=pt。"""
    e = _make_embedder()
    monkeypatch.setattr(emb, "_mean_pool_l2", lambda hidden, mask: FakeTensor([[0.1, 0.2]]))
    vecs = e.embed(["文本一", "文本二"])
    assert e.tokenizer.calls[0]["texts"] == ["文本一", "文本二"]
    assert e.tokenizer.calls[0]["padding"] is True
    assert e.tokenizer.calls[0]["truncation"] is True
    assert e.tokenizer.calls[0]["max_length"] == 512
    assert e.tokenizer.calls[0]["return_tensors"] == "pt"
    assert len(vecs) == 2
    assert vecs[0] == [0.1, 0.2]  # 值来自 pool 结果


def test_embed_passes_hidden_and_mask_to_pool(monkeypatch):
    """pool 收到的是 last_hidden_state 与 attention_mask（mean pooling 输入）。"""
    e = _make_embedder()
    captured = {}

    def fake_pool(hidden, mask):
        captured["hidden"] = hidden
        captured["mask"] = mask
        return FakeTensor()

    monkeypatch.setattr(emb, "_mean_pool_l2", fake_pool)
    e.embed(["一段文本"])
    assert captured["hidden"] is not None
    assert captured["mask"] is not None
    # model 收到的 inputs 与 tokenizer 产出对应
    assert "input_ids" in e.model.calls[0]


def test_embed_uses_no_grad_and_runs_on_cpu_device():
    """推理在 no_grad 下进行；device=cpu 时 inputs.to('cpu')。"""
    e = _make_embedder()
    assert e.device == "cpu"


def test_embed_one_returns_single_vector(monkeypatch):
    e = _make_embedder()
    monkeypatch.setattr(emb, "_mean_pool_l2", lambda h, m: FakeTensor([[0.5, 0.5]]))
    v = e.embed_one("单条")
    assert v == [0.5, 0.5]
    assert len(e.tokenizer.calls) == 1


# ---------- 错误路径 ----------

def test_missing_transformers_raises_clear_error(monkeypatch):
    """未装 transformers → 清晰指引（而非裸 ImportError 堆栈）。"""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "transformers":
            raise ImportError("No module named 'transformers'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="pip install transformers torch"):
        emb.LocalHFEmbedder("./dir-embed")


def test_cuda_auto_selected_when_available(monkeypatch):
    """torch.cuda 可用 → 自动 device=cuda 并把模型搬到设备。"""
    torch_ = FakeTorch()
    torch_.cuda = _FakeCudaModule(available=True)
    e = _make_embedder(torch_=torch_)
    assert e.device == "cuda"
    assert e.model.device == "cuda"


def test_cpu_default_without_cuda():
    """无 cuda → device=cpu，模型不搬。"""
    e = _make_embedder()
    assert e.device == "cpu"
    assert e.model.device is None
