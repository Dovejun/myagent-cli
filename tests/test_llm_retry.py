"""LLM 重试测试：指数退避、可重试/不可重试异常分类（离线，用 FakeClient）。"""

from types import SimpleNamespace

import httpx
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)

from agent.llm import LLMClient, _is_retryable


# ---------- 工具：构造 fake client 与真实 openai 异常 ----------


def _ok(content: str = "ok"):
    """模拟一次成功响应（message 对象）。"""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))]
    )


def _request():
    return httpx.Request("POST", "https://api.example.com/v1/chat/completions")


def _conn_err():
    return APIConnectionError("connection failed", request=_request())


def _timeout_err():
    return APITimeoutError("timed out", request=_request())


def _rate_err():
    return RateLimitError(
        "rate limited", response=httpx.Response(429, request=_request()), body=None
    )


def _server_err():
    return APIStatusError(
        "internal error", response=httpx.Response(500, request=_request()), body=None
    )


def _auth_err():
    return AuthenticationError(
        "bad api key", response=httpx.Response(401, request=_request()), body=None
    )


class _Completions:
    def __init__(self, fake):
        self._fake = fake

    def create(self, **kwargs):
        return self._fake._create(**kwargs)


class _Chat:
    def __init__(self, fake):
        self.completions = _Completions(fake)


class FakeClient:
    """模拟 openai client：按脚本依次消费（成功值或异常）。"""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0
        self.chat = _Chat(self)

    def _create(self, **kwargs):
        self.calls += 1
        if not self._script:
            raise RuntimeError("script exhausted")
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _llm(script) -> LLMClient:
    return LLMClient(client=FakeClient(script), model="test-model")


# ---------- _is_retryable 分类 ----------


def test_is_retryable_classification():
    assert _is_retryable(_conn_err()) is True       # 连接失败：重试
    assert _is_retryable(_timeout_err()) is True    # 超时：重试
    assert _is_retryable(_rate_err()) is True       # 限流 429：重试
    assert _is_retryable(_server_err()) is True     # 5xx：重试
    assert _is_retryable(_auth_err()) is False      # 鉴权失败：不重试


# ---------- chat 重试行为 ----------


def test_retry_success_after_failures(monkeypatch):
    """前 2 次连接失败，第 3 次成功 → 返回结果，共调用 3 次。"""
    monkeypatch.setattr("time.sleep", lambda s: None)  # 不等真实时间
    llm = _llm([_conn_err(), _conn_err(), _ok("done")])
    msg = llm.chat([{"role": "user", "content": "hi"}], retry_base_delay=0)
    assert msg.content == "done"
    assert llm.client.calls == 3


def test_retry_exhausted_raises(monkeypatch):
    """连续失败 3 次 → 抛出原始异常，不再重试。"""
    monkeypatch.setattr("time.sleep", lambda s: None)
    llm = _llm([_conn_err(), _conn_err(), _conn_err()])
    with pytest.raises(APIConnectionError):
        llm.chat([{"role": "user", "content": "hi"}], retry_base_delay=0)
    assert llm.client.calls == 3


def test_non_retryable_raises_immediately():
    """鉴权失败（4xx）不可重试 → 1 次调用即抛。"""
    llm = _llm([_auth_err()])
    with pytest.raises(AuthenticationError):
        llm.chat([{"role": "user", "content": "hi"}])
    assert llm.client.calls == 1


def test_success_no_retry():
    """一次成功 → 只调用 1 次。"""
    llm = _llm([_ok("hi")])
    msg = llm.chat([{"role": "user", "content": "hi"}])
    assert msg.content == "hi"
    assert llm.client.calls == 1


def test_instance_default_retry_times(monkeypatch):
    """不传 retry_times 时用实例默认值（3 次）。"""
    monkeypatch.setattr("time.sleep", lambda s: None)
    llm = _llm([_rate_err(), _rate_err(), _ok("ok")])
    msg = llm.chat([{"role": "user", "content": "hi"}], retry_base_delay=0)
    assert msg.content == "ok"
    assert llm.client.calls == 3
