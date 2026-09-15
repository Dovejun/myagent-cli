"""LLM 客户端封装。

职责单一：封装 OpenAI 兼容的 Chat Completions API，
统一处理 messages 与可选的 tools 参数。

为什么单独抽一层？将来换模型、加重试、加流式输出，
只改这一个文件，core.py 的 ReAct 循环完全不用动。

约束设计（D2-2）：chat() 内置指数退避重试。
- 可重试：连接错误 / 限流(429) / 超时 / 5xx 服务端错误
- 不重试：鉴权失败、参数错误等 4xx（重试无意义）
- 默认 3 次，退避间隔 1s/2s/4s；连续失败后抛异常，由外层降级
  （降级策略见 docs/fallback-policy.md 场景 A）
"""

import os
import time

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    RateLimitError,
    OpenAI,
)

load_dotenv()

DEFAULT_RETRY_TIMES = 3
DEFAULT_RETRY_BASE_DELAY = 1.0  # 秒，指数退避基数


def _is_retryable(exc: Exception) -> bool:
    """判断异常是否值得重试。

    可重试：网络连接失败、限流(429)、请求超时、服务端 5xx —— 都是
    "对方暂时不可用"，重试有希望成功。
    不重试：4xx（鉴权失败、参数错误）—— 重试结果一样，纯浪费。
    """
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code >= 500:
        return True
    return False


class LLMClient:
    def __init__(
        self,
        client=None,
        model: str | None = None,
        retry_times: int = DEFAULT_RETRY_TIMES,
    ) -> None:
        """封装 OpenAI 兼容客户端。

        测试可注入 client（FakeClient），避免真实网络调用：
            LLMClient(client=fake, model="test")
        retry_times 为实例级默认重试次数，chat() 可临时覆盖。
        """
        self.retry_times = retry_times
        if client is not None:
            self.client = client
            self.model = model or "test-model"
            return
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def chat(
        self,
        messages: list,
        tools: list | None = None,
        retry_times: int | None = None,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    ):
        """调用模型，带指数退避重试。返回 response.choices[0].message。

        - tools 为 None：普通对话，模型直接回答
        - tools 传入时：同时设置 tool_choice="auto"，让模型自行决定
          是否调用工具、调用哪个工具
        - retry_times：默认用实例值；retry_base_delay 供测试注入 0 秒间隔
        """
        retry_times = retry_times or self.retry_times
        kwargs: dict = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        last_error: Exception | None = None
        for attempt in range(retry_times):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message
            except Exception as e:  # noqa: BLE001 —— 统一判断是否可重试
                last_error = e
                if not _is_retryable(e) or attempt == retry_times - 1:
                    raise  # 不可重试 / 已到上限：抛给上层降级
                time.sleep(retry_base_delay * (2**attempt))  # 指数退避

        raise last_error  # 理论不可达，兜底

    def chat_stream(self, messages: list, tools: list | None = None):
        """流式调用模型：生成器，yield 两种事件。

        ("content", str) —— 最终答案的内容增量（可立即显示给用户）
        ("done",    dict) —— 流结束，携带完整可序列化的 assistant 消息
                             （content + tool_calls，格式与 _msg_to_dict 一致，
                             可直接 append 进 messages 并判断是否有工具调用）

        为什么"结束才给完整消息"：流式时 content 和 tool_calls 是分块交错下发
        的，只有流结束才能确定这一轮到底是"直接回答"还是"要调工具"。
        调用方（Agent 循环）据此决定：有 tool_calls 就执行工具继续循环，
        没有就把已流式显示的内容作为最终答案。

        注意：流式中断（用户 Ctrl+C）需由调用方 close 生成器，
        这里不做重试（已开始吐字的内容重试会重复）。
        """
        kwargs: dict = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        stream = self.client.chat.completions.create(**kwargs, stream=True)

        content_parts: list[str] = []
        # 累积每个 tool_call（key = 分块 index）：OpenAI 流式把
        # id/name/arguments 分成多个 chunk 下发，arguments 是字符串增量
        tool_acc: dict[int, dict] = {}

        for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue  # usage 等无 choices 的块
            delta = chunk.choices[0].delta
            if delta is None:
                continue

            content = getattr(delta, "content", None)
            if content:
                content_parts.append(content)
                yield ("content", content)

            for tc in getattr(delta, "tool_calls", None) or []:
                acc = tool_acc.setdefault(
                    tc.index,
                    {"id": "", "function": {"name": "", "arguments": ""}},
                )
                if getattr(tc, "id", None):
                    acc["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn:
                    if getattr(fn, "name", None):
                        acc["function"]["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        acc["function"]["arguments"] += fn.arguments  # 增量拼接

        # 重组完整 assistant 消息（与 chat() 返回的对象等价，但已是 dict）
        content = "".join(content_parts) or None
        msg_dict: dict = {"role": "assistant", "content": content}
        if tool_acc:
            calls = []
            for idx in sorted(tool_acc):
                acc = tool_acc[idx]
                calls.append(
                    {
                        "id": acc["id"],
                        "type": "function",
                        "function": {
                            "name": acc["function"]["name"],
                            "arguments": acc["function"]["arguments"],
                        },
                    }
                )
            msg_dict["tool_calls"] = calls
        yield ("done", msg_dict)
