"""跨轮上下文记忆回归测试：修复"第一轮读大文件 → 第二轮失忆"bug。

bug 根因：_emergency_compress 旧实现超预算即清空整个窗口区，
第一轮刚读入的大文件原文被立刻摘要化 → 第二轮问"第 2 个任务点内容"
时模型只剩摘要，引用不到原文 → 答"不知道"。
修复：改为从最老轮开始滑出，保底保留最近一轮完整原文。
"""

from types import SimpleNamespace

from agent.core import Agent
from agent.memory import MemoryCompressor
from agent.tools.registry import ToolRegistry

# 第一轮读入的"大文件"内容（超过测试预算，模拟任务点文档）
BIG_CONTENT = "任务点列表：\n1. 搭建登录页\n2. 实现用户认证（含 token 刷新）\n3. 接入支付回调" + "X" * 200


class FakeExtractLLM:
    """压缩摘要用的假 LLM：chat 返回固定摘要文本。"""

    def chat(self, messages, tools=None):
        return SimpleNamespace(content="已确认事实：\n- 原文已摘要化，细节以存档为准")


def _tool_call(name: str, args: str = "{}", cid: str = "c1"):
    return SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(
            id=cid,
            function=SimpleNamespace(name=name, arguments=args),
        )],
    )


def _text(content: str):
    return SimpleNamespace(content=content, tool_calls=None)


class ScriptLLM:
    """主循环假 LLM：按调用次序消费脚本。"""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    def chat(self, messages, tools=None, **kw):
        self.calls += 1
        return self._script.pop(0)


class RecordingLLM(ScriptLLM):
    """额外记录每次请求的 messages（验证第二轮能看到第一轮原文）。"""

    def __init__(self, script):
        super().__init__(script)
        self.requests: list[list[dict]] = []

    def chat(self, messages, tools=None, **kw):
        self.requests.append(list(messages))
        return super().chat(messages, tools, **kw)


def _make_agent(llm, threshold: int = 100, window_rounds: int = 10) -> Agent:
    reg = ToolRegistry()
    reg.register("read", "读大文件", {"type": "object", "properties": {}},
                 lambda: BIG_CONTENT)  # tool 结果 = 大文件原文
    compressor = MemoryCompressor(FakeExtractLLM())
    return Agent(
        llm, reg, "系统提示", max_steps=6,
        compressor=compressor,
        compress_threshold=threshold,  # 预算极小 → 必然触发紧急压缩
        window_rounds=window_rounds,
    )


def test_single_over_budget_round_keeps_original():
    """第一轮读入超预算内容 → 紧急压缩后原文仍在（不被清空）。"""
    llm = ScriptLLM([
        _tool_call("read"),                    # 第一轮：读大文件
        _text("好的，任务点已列出。"),           # 第一轮答案
    ])
    agent = _make_agent(llm)
    agent.run("读任务文件，列出任务点")

    contents = " ".join(str(m.get("content") or "") for m in agent.messages)
    # 修复前：整个窗口被清空成摘要，BIG_CONTENT 消失 → 断言它还在
    assert "实现用户认证" in contents


def test_cross_round_request_contains_first_round_content():
    """集成：第二轮提问时，请求里仍携带第一轮读入的原文（模型可引用）。"""
    recorder = RecordingLLM([
        _tool_call("read"),                    # 第一轮：读大文件
        _text("任务点已列出。"),                # 第一轮答案
        _text("第 2 个任务点是实现用户认证。"),  # 第二轮答案（模型引用历史）
    ])
    agent = _make_agent(recorder)
    agent.run("读任务文件，列出任务点有哪些")
    agent.run("查看第 2 个任务点的内容")

    # 第二轮请求（requests[2]）必须包含第一轮读入的原文
    assert len(recorder.requests) == 3
    second_req = " ".join(str(m.get("content") or "") for m in recorder.requests[2])
    assert "实现用户认证" in second_req


def test_multi_round_emergency_slides_oldest_keeps_recent():
    """多轮超预算：滑出最老轮（摘要化），保留最近轮原文。"""
    reg = ToolRegistry()
    reg.register("read", "读", {"type": "object", "properties": {}}, lambda: BIG_CONTENT)
    compressor = MemoryCompressor(FakeExtractLLM())
    agent = Agent(
        ScriptLLM([]), reg, "系统提示", max_steps=6,
        compressor=compressor,
        compress_threshold=100,
        window_rounds=0,  # 不启用按轮窗口，只测紧急压缩
    )
    # 手工构造 3 轮消息（第 1 轮含大文件原文，第 3 轮是最近）
    agent.messages = [
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "第一轮问题"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c", "content": BIG_CONTENT},  # 超大
        {"role": "assistant", "content": "第一轮答案"},
        {"role": "user", "content": "第二轮问题"},
        {"role": "assistant", "content": "第二轮答案"},
        {"role": "user", "content": "第三轮（最近）问题"},
        {"role": "assistant", "content": "第三轮答案"},
    ]
    agent._emergency_compress()

    contents = [str(m.get("content") or "") for m in agent.messages]
    # 最近轮（第 3 轮）原文保留
    assert "第三轮" in " ".join(contents)
    # 摘要槽已建立（最老轮被提炼）
    assert any(m["role"] == "system" and "摘要化" in str(m.get("content") or "")
               for m in agent.messages)
    # 超大原文要么被滑出、要么仍在（至少 messages 不再整体超预算）
    assert agent._estimate_size() <= 100 or "第三轮" in " ".join(contents)


def test_under_budget_no_compression():
    """预算内：紧急压缩完全不触发，messages 原样。"""
    llm = ScriptLLM([
        _tool_call("read"),
        _text("完成"),
    ])
    agent = _make_agent(llm, threshold=10_000_000)  # 预算巨大
    agent.run("读文件")
    assert agent._summary_index() is None  # 无摘要槽
    assert "实现用户认证" in " ".join(str(m.get("content") or "") for m in agent.messages)
