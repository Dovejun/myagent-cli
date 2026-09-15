"""LLM 流式调用测试：content 增量、tool_calls 分块累积、事件顺序（离线）。

用 FakeStreamClient 模拟 OpenAI 流式 chunk（SimpleNamespace 结构对齐 SDK）。
"""

from types import SimpleNamespace

from agent.llm import LLMClient


# ---------- 工具：构造流式 chunk ----------


def _delta(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tc(index, id=None, name=None, arguments=None):
    """构造一个 tool_calls 分块 delta。"""
    return SimpleNamespace(
        index=index,
        id=id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _chunk(delta):
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class FakeStreamClient:
    """chat.completions.create(..., stream=True) 返回 chunk 迭代器。"""

    def __init__(self, chunks):
        self.chunks = chunks
        self.stream_kwargs = None

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.stream_kwargs = kwargs
        return iter(self.chunks)


def _llm(chunks) -> tuple[LLMClient, FakeStreamClient]:
    client = FakeStreamClient(chunks)
    return LLMClient(client=client, model="test-model"), client


def _collect(gen):
    """消费流式生成器，返回 (事件列表, done_msg_dict)。"""
    events, done = [], None
    for event, payload in gen:
        if event == "content":
            events.append(payload)
        else:
            done = payload
    return events, done


# ---------- 测试 ----------


def test_stream_content_accumulates():
    """纯文本回答：content 分块增量全部产出，done 消息内容完整。"""
    llm, client = _llm([
        _chunk(_delta(content="你好")),
        _chunk(_delta(content="，我是")),
        _chunk(_delta(content="Agent")),
    ])
    parts, done = _collect(llm.chat_stream([{"role": "user", "content": "hi"}]))
    assert parts == ["你好", "，我是", "Agent"]
    assert done["content"] == "你好，我是Agent"
    assert "tool_calls" not in done
    assert client.stream_kwargs["stream"] is True


def test_stream_empty_content_ok():
    """内容为空（纯工具轮）：done.content 为 None。"""
    llm, _ = _llm([
        _chunk(_delta(tool_calls=[_tc(0, id="call_1", name="read_file", arguments='{"path":')])),
        _chunk(_delta(tool_calls=[_tc(0, arguments='"a.txt"})'))),
    ])
    parts, done = _collect(llm.chat_stream([]))
    assert parts == []
    assert done["content"] is None
    calls = done["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["id"] == "call_1"
    assert calls[0]["function"]["name"] == "read_file"
    assert calls[0]["function"]["arguments"] == '{"path": "a.txt"}'


def test_stream_multiple_tool_calls_by_index():
    """一条消息里多个 tool_call：按 index 各自累积 arguments。"""
    llm, _ = _llm([
        _chunk(_delta(tool_calls=[_tc(0, id="c0", name="read_file", arguments='{"path": "a"')])),
        _chunk(_delta(tool_calls=[_tc(1, id="c1", name="run_shell", arguments='{"command": "ls"')])),
        _chunk(_delta(tool_calls=[_tc(0, arguments='.txt"}')])),
        _chunk(_delta(tool_calls=[_tc(1, arguments='"}'))),
    ])
    _, done = _collect(llm.chat_stream([]))
    calls = done["tool_calls"]
    assert len(calls) == 2
    by_id = {c["id"]: c for c in calls}
    assert by_id["c0"]["function"]["arguments"] == '{"path": "a.txt"}'
    assert by_id["c1"]["function"]["arguments"] == '{"command": "ls"}'


def test_stream_skips_usage_chunks():
    """无 choices 的块（如 usage）被跳过，不影响结果。"""
    llm, _ = _llm([
        _chunk(_delta(content="hi")),
        SimpleNamespace(choices=[]),  # usage 块
    ])
    parts, done = _collect(llm.chat_stream([]))
    assert parts == ["hi"]
    assert done["content"] == "hi"
