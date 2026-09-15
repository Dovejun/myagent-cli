"""Agent 流式集成测试：最终答案边生成边回调、工具轮不误显示（离线）。

StreamLLM 同时支持 chat（对象）与 chat_stream（事件生成器），
用 rounds 脚本驱动 Agent 循环。
"""

from types import SimpleNamespace

from agent.core import Agent
from agent.tools.registry import ToolRegistry


class StreamLLM:
    """按轮消费脚本。每轮: {content_parts: [...], msg: {content/tool_calls}}。"""

    def __init__(self, rounds):
        self._rounds = list(rounds)
        self.stream_calls = 0
        self.chat_calls = 0

    # --- 流式路径 ---
    def chat_stream(self, messages, tools=None):
        self.stream_calls += 1
        r = self._rounds.pop(0)
        for piece in r.get("content_parts", []):
            yield ("content", piece)
        yield ("done", r["msg"])

    # --- 非流式路径（供对比/混合场景）---
    def chat(self, messages, tools=None, **kw):
        self.chat_calls += 1
        r = self._rounds.pop(0)
        calls = None
        if r["msg"].get("tool_calls"):
            calls = [
                SimpleNamespace(
                    id=c["id"],
                    function=SimpleNamespace(
                        name=c["function"]["name"],
                        arguments=c["function"]["arguments"],
                    ),
                )
                for c in r["msg"]["tool_calls"]
            ]
        return SimpleNamespace(content=r["msg"].get("content"), tool_calls=calls)


def _tool_msg(name, args):
    return {"id": "c1", "function": {"name": name, "arguments": args}}


def _registry():
    reg = ToolRegistry()
    reg.register("read", "读", {"type": "object", "properties": {}}, lambda: "内容")
    return reg


def test_stream_final_answer_callback():
    """纯文本轮：content 逐段回调 on_token，run 返回完整答案。"""
    llm = StreamLLM([
        {"content_parts": ["你好", "，我", "是 Agent"], "msg": {"role": "assistant", "content": "你好，我是 Agent"}},
    ])
    agent = Agent(llm, _registry(), "sys", max_steps=5)

    tokens: list[str] = []
    answer = agent.run("打招呼", stream=True, on_token=tokens.append)
    assert tokens == ["你好", "，我", "是 Agent"]
    assert answer == "你好，我是 Agent"
    assert llm.stream_calls == 1


def test_stream_tool_round_no_content_callback():
    """工具轮流式：content 为空不回调；执行工具后下一轮流式输出最终答案。"""
    llm = StreamLLM([
        # 第一轮：调工具（content 空 → 不打扰用户）
        {"content_parts": [], "msg": {
            "role": "assistant", "content": None,
            "tool_calls": [_tool_msg("read", "{}")],
        }},
        # 第二轮：最终答案
        {"content_parts": ["文件内容是：内容"], "msg": {"role": "assistant", "content": "文件内容是：内容"}},
    ])
    agent = Agent(llm, _registry(), "sys", max_steps=5)

    tokens: list[str] = []
    answer = agent.run("读文件", stream=True, on_token=tokens.append)
    assert tokens == ["文件内容是：内容"]  # 工具轮零打扰
    assert answer == "文件内容是：内容"
    # 主上下文：system / user / assistant(tool) / tool / assistant(答案)
    assert len(agent.messages) == 5


def test_stream_off_ignores_on_token():
    """stream=False 时即使传 on_token 也不回调（行为与旧版一致）。"""
    llm = StreamLLM([
        {"content_parts": ["不应显示"], "msg": {"role": "assistant", "content": "答案"}},
    ])
    agent = Agent(llm, _registry(), "sys", max_steps=5)

    tokens: list[str] = []
    answer = agent.run("问", stream=False, on_token=tokens.append)
    assert tokens == []
    assert answer == "答案"
    assert llm.chat_calls == 1  # 走了非流式路径
