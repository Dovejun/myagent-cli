"""Agent 主循环：ReAct（Reason + Act + Observe）+ 三层记忆架构。

============================================================
一、ReAct 循环（tool-calling loop）
============================================================
    用户输入 → 调 LLM → 模型返回 tool_calls？
        ├─ 否 → 这就是最终答案，返回
        └─ 是 → 逐个执行工具，结果以 role=tool 消息回喂 → 回到调 LLM

    核心约定（容易踩坑）：
    1. assistant 消息必须完整序列化（保留 tool_calls 字段），
       否则下一轮 API 请求报错
    2. tool 消息必须带 tool_call_id，与 assistant 的 tool_calls id
       一一对应
    3. max_steps 是保险丝：防止"读文件失败 → 再读同一文件"的死循环

============================================================
二、三层记忆架构（控制 token 消耗 + 保证不丢重要内容）
============================================================
    "LLM 无状态，记忆 = messages 数组" —— 但 messages 随会话无限
    累积会让 token 成本线性爆炸。三层记忆让总消耗收敛到常数：

    长期记忆  TaskMemory：固定预算（如 1500 字符）的任务记忆块，
              每轮 merge 用户消息的【增量信息】并去重 → 大小恒定。
              "永不压缩"的正确解读：信息不丢，不是原文全留。
    中期记忆  增量摘要（本文件维护一条"摘要槽"）：窗口滑出的旧消息
              交给 MemoryCompressor.extract_summary() 提取要点，
              旧摘要作为高优先级输入 → 只增补不稀释。
    短期记忆  滑动窗口：只保留最近 window_rounds 轮的完整消息，
              旧的滑出。滑出前先摘要 + 原文外置存档，不直接丢。

    messages 内部结构：
        [0] system_prompt                  ← 常驻，永不裁剪
        [1] 【历史增量摘要】system 消息    ← 中期记忆（可选）
        [2..] 窗口区消息                    ← 短期记忆（有界）

    每次 API 请求的组装（_build_request_messages）：
        [system_prompt] + [任务记忆] + [历史增量摘要] + [窗口区]

============================================================
三、集成方式（在 agent_cli.py 的 build_agent 里）：
============================================================
    from agent.memory import MemoryCompressor, TaskMemory

    agent = Agent(
        llm, registry, SYSTEM_PROMPT,
        task_memory=TaskMemory(llm, budget=1500),          # 长期记忆
        compressor=MemoryCompressor(llm, keep_roles=("system",)),  # 中期
        window_rounds=10,                                  # 短期窗口
        archive_path="memory_store.jsonl",                 # 原文兜底
    )
"""

import json
from typing import Callable

from rich.console import Console

from .context_audit import audit, render_report
from .knowledge import KnowledgeStore
from .llm import LLMClient
from .memory import MemoryCompressor, TaskMemory
from .tools.registry import ToolRegistry

# 摘要槽 system 消息的前缀，用于在 messages 里定位"中期记忆"那条消息
_SUMMARY_PREFIX = "【历史增量摘要】"


def _msg_to_dict(msg) -> dict:
    """把 OpenAI 返回的 message 对象序列化成可写回 messages 的 dict。

    关键：tool_calls 必须完整保留（id/type/function.name/arguments），
    这是下一轮 API 调用识别"上一轮调过哪些工具"的唯一依据。
    只存 content 会导致下一轮请求 400 报错。
    """
    d: dict = {"role": "assistant", "content": msg.content}
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
    return d


class Agent:
    """命令行 Agent：ReAct 循环 + 可选的三层记忆。

    参数说明：
        llm            : LLMClient，OpenAI 兼容 API 封装
        registry       : ToolRegistry，工具注册表（schema + 执行分发）
        system_prompt  : System Prompt，行为准则，常驻第一条消息
        max_steps      : ReAct 循环上限（保险丝），默认 15
        verbose        : True 时用 Rich 彩色打印每步工具调用与结果
        console        : Rich Console 实例（CLI 层传入，风格统一）

        三层记忆（均为可选，不传则退化为朴素 ReAct）：
        task_memory    : TaskMemory，长期记忆（固定预算任务记忆块）
        compressor     : MemoryCompressor，中期记忆（增量摘要提取）
        window_rounds  : 滑动窗口轮数（短期记忆），0 表示不启用窗口
        archive_path   : 被滑出/被压缩的原文双写存档路径（JSONL）
        compress_threshold : 总大小兜底阈值（字符数），0 表示不启用；
                             即使窗口内消息也超阈值时强制整体压缩一次

        上下文工程（D1，可选）：
        knowledge      : KnowledgeStore，检索注入（RAG 雏形）——
                         用户提问时按关键词检索索引条目，注入本次请求
        audit_model    : 审计用的模型名（tiktoken 计数），默认 gpt-4o

        约束设计（D2，可选）：
        approver       : 工具审批回调 approver(name, args) -> bool。
                         confirm 级工具（如 write_file/run_shell）执行前会先询问；
                         None = 全放行（向后兼容）。降级策略见 docs/fallback-policy.md

        质量闭环（D5，可选）：
        validation_hooks : dict[工具名, list[验证器]]。验证器签名
                         fn(tool_result) -> (ok, output)。指定工具执行成功后
                         自动运行验证，结果附加到 tool 消息回喂：
                         通过 → [验证通过: xx]；失败 → [验证失败: xx] + 输出，
                         模型可读并修复。None = 不启用（默认，行为不变）
    """

    # 观察者事件类型（run(on_event=...) 收到的事件，供 Web UI / 日志消费）：
    #   step        {"step": int, "max_steps": int}          一轮 Reason 开始
    #   tool_start  {"name": str, "args": dict}              工具即将执行
    #   tool_result {"name": str, "output": str,
    #                "denied": bool, "error": bool}          工具执行结束
    #   hooks       {"name": str, "output": str}             验证钩子输出
    #   done        {"content": str, "steps": int,
    #                "stopped": bool}                        任务结束
    #   error       {"message": str}                         未捕获异常
    # 事件是"给观察者的旁路信息"，与 verbose（给终端的打印）互不影响，
    # 因此 Web 模式可以关掉 verbose 仍拿到完整过程。

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        system_prompt: str,
        max_steps: int = 15,
        verbose: bool = False,
        console: Console | None = None,
        task_memory: TaskMemory | None = None,
        compressor: MemoryCompressor | None = None,
        window_rounds: int = 0,
        archive_path: str | None = None,
        compress_threshold: int = 0,
        knowledge: KnowledgeStore | None = None,
        audit_model: str = "gpt-4o",
        approver=None,
        validation_hooks: dict[str, list] | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.console = console or Console()
        self.max_steps = max_steps
        self.verbose = verbose

        # 三层记忆配置
        self.task_memory = task_memory
        self.compressor = compressor
        self.window_rounds = window_rounds
        self.archive_path = archive_path
        self.compress_threshold = compress_threshold

        # 约束设计：工具审批回调（confirm 级工具执行前询问）
        self.approver = approver

        # 质量闭环：验证点钩子（工具执行后自动校验）
        self.validation_hooks = validation_hooks or {}

        # 上下文工程：知识检索注入 + 审计
        self.knowledge = knowledge
        self.audit_model = audit_model
        self._last_user_input = ""  # 记录当前任务的用户提问，供检索注入

        # 观察者回调（Web UI 等）：run(on_event=...) 注入，见 _emit
        self._on_event = None
        self._on_token = None

        # messages 的起点：system_prompt 常驻在 [0]，永不参与窗口裁剪
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # ------------------------------------------------------------------
    # 观察者事件（Web UI / 日志）
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, **payload) -> None:
        """向观察者发送一个结构化事件（事件契约见类 docstring）。

        与 verbose 解耦：只要 run(on_event=...) 传了回调就发，
        因此 Web 模式可以关掉 verbose（不打印终端）仍拿到完整过程。
        观察者是旁路——回调抛异常必须吞掉，不能拖垮 Agent 主流程。
        """
        cb = self._on_event
        if cb is None:
            return
        try:
            cb({"type": event_type, **payload})
        except Exception:  # noqa: BLE001 —— 观察者异常不得影响主流程
            pass

    @staticmethod
    def _parse_args(raw: str) -> dict:
        """把工具参数 JSON 文本解析成 dict（给事件展示用，坏 JSON 不炸）。"""
        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {"_raw": raw}
        return parsed if isinstance(parsed, dict) else {"_raw": parsed}

    # ------------------------------------------------------------------
    # 质量闭环：验证点钩子
    # ------------------------------------------------------------------

    def _run_validation_hooks(self, tool_name: str) -> str:
        """运行指定工具的所有验证器，返回要附加到 tool 结果的文本（空 = 无附加）。"""
        extras: list[str] = []
        for fn in self.validation_hooks.get(tool_name, []):
            label = getattr(fn, "__name__", "hook")
            try:
                ok, output = fn("")
            except Exception as e:  # noqa: BLE001 —— 钩子出错按失败处理，不崩溃
                ok, output = False, f"[钩子执行出错] {type(e).__name__}: {e}"
            if ok:
                extras.append(f"[验证通过: {label}]")
            else:
                extras.append(f"[验证失败: {label}]\n{output}")
        return "\n".join(extras)

    # ------------------------------------------------------------------
    # 消息构造
    # ------------------------------------------------------------------

    def _build_request_messages(self) -> list[dict]:
        """把三层记忆 + 知识检索组装成一次 API 请求的消息序列。

        顺序：system_prompt → 任务记忆（长期）→ 历史增量摘要（中期）
              → 窗口区（短期）→ 检索到的相关资料（D1-3，追加在最末）。
        任务记忆是临时构造的（TaskMemory 独立维护），不混入 self.messages，
        因此窗口裁剪永远不会误伤长期记忆。
        检索注入也是临时的：只在本次请求携带，不污染窗口统计。
        """
        if self.task_memory:
            # messages[0] 是 system_prompt；messages[1:] 含摘要槽与窗口区
            request = [
                self.messages[0],
                self.task_memory.to_system_message(),
                *self.messages[1:],
            ]
        else:
            request = list(self.messages)

        # D1-3 知识检索注入：按当前用户提问检索索引，命中则追加为 system 消息
        if self.knowledge and self._last_user_input:
            hits = self.knowledge.retrieve(self._last_user_input)
            if hits:
                request.append(
                    {"role": "system", "content": "【检索到的相关资料】\n" + hits}
                )
        return request

    # ------------------------------------------------------------------
    # 中期记忆：摘要槽的定位 / 读取 / 写入
    # ------------------------------------------------------------------

    def _summary_index(self) -> int | None:
        """在 messages 中定位摘要槽（【历史增量摘要】system 消息）的索引。"""
        for i, m in enumerate(self.messages):
            if m["role"] == "system" and m["content"].startswith(_SUMMARY_PREFIX):
                return i
        return None

    def _current_summary(self) -> str | None:
        """读取当前摘要槽的纯文本（去掉前缀），没有则返回 None。"""
        si = self._summary_index()
        if si is None:
            return None
        return self.messages[si]["content"][len(_SUMMARY_PREFIX) + 1 :]

    def _set_summary(self, text: str) -> None:
        """把新摘要写入摘要槽：已有则替换，没有则插到 system 之后。"""
        new_msg = {"role": "system", "content": f"{_SUMMARY_PREFIX}\n{text}"}
        si = self._summary_index()
        if si is not None:
            self.messages[si] = new_msg  # 替换旧摘要（增量更新）
        elif text:
            self.messages.insert(1, new_msg)  # 首次建立摘要槽

    # ------------------------------------------------------------------
    # 短期记忆：滑动窗口
    # ------------------------------------------------------------------

    def _trim_window(self) -> None:
        """滑动窗口：超过 window_rounds 轮时，把最旧的轮次滑出。

        滑出前两步保真（信息不丢，只是出上下文）：
        1. 增量摘要：滑出的旧消息 + 现有摘要 → extract_summary() 合并提取，
           结果写回摘要槽（中期记忆）
        2. 原文双写：滑出的完整原文追加写入 archive_path（外置兜底）
        然后才丢弃原文 —— 摘要是指针，磁盘是副本，随时可找回。
        """
        if not self.window_rounds or not self.compressor:
            return

        # 常驻区：system_prompt + 摘要槽（若有）；其余是窗口区
        si = self._summary_index()
        anchored = (si + 1) if si is not None else 1
        rest = self.messages[anchored:]

        # 统计窗口区共有几轮（一条 user 消息 = 一轮的开始）
        user_indexes = [i for i, m in enumerate(rest) if m["role"] == "user"]
        if len(user_indexes) <= self.window_rounds:
            return

        # 第 window_rounds+1 轮起点之前的全部滑出（保证保留完整的 N 轮，
        # 不切断 user→assistant→tool 的消息链条）
        cut = user_indexes[-(self.window_rounds + 1)]
        old, keep = rest[:cut], rest[cut:]

        if self.verbose:
            self.console.print(
                f"[yellow]memory[/] 窗口超限，滑出 {len(old)} 条旧消息"
            )

        # 保真两步：先摘要、再外置，最后才丢弃原文
        summary = self.compressor.extract_summary(old, existing_summary=self._current_summary())
        if self.archive_path:
            self.compressor.archive(old, self.archive_path)
        self.messages = self.messages[:anchored] + keep
        self._set_summary(summary)

    # ------------------------------------------------------------------
    # 总大小兜底（紧急压缩）
    # ------------------------------------------------------------------

    def _estimate_size(self, messages: list[dict] | None = None) -> int:
        """粗略估算消息总字符数（生产可用 tiktoken 数 token）。

        可传入子集（如裁剪候选）估算，默认估算整个 self.messages。
        """
        msgs = self.messages if messages is None else messages
        return sum(
            len(str(m.get("content") or m.get("tool_calls") or ""))
            for m in msgs
        )

    def _emergency_compress(self) -> None:
        """总量兜底（保险丝）：超预算时从【最老轮次】开始滑出，保底保留最近一轮。

        修复（跨轮失忆 bug）：旧实现超预算即清空整个窗口区——
        第一轮刚读入的大文件原文会被立刻摘要化，下一轮模型引用不到
        细节（用户问"第 2 个任务点内容"答"不知道"）。
        新策略：
        1. 以"轮"为单位，从最老轮开始滑出（增量摘要 + 原文外置），
           直到剩余 ≤ 预算
        2. 至少保留最近 1 轮的完整原文——当前任务正在引用的上下文不清空
        3. 单轮内容本身就超预算（如一次读了超大文件）：保留原文不动，
           由模型窗口容纳；彻底放不下的极端情况由 LLM API 拒收兜底
        """
        if not self.compressor or not self.compress_threshold:
            return
        if self._estimate_size() <= self.compress_threshold:
            return

        si = self._summary_index()
        anchored = (si + 1) if si is not None else 1
        rest = self.messages[anchored:]
        user_idx = [i for i, m in enumerate(rest) if m["role"] == "user"]
        if not user_idx:
            return

        # 找最小滑出量：候选"保留起点"是每一轮的 user 位置（除最后一轮外），
        # 逐个尝试，取第一个能让剩余达标的起点；全不达标则只保最近一轮
        last_start = user_idx[-1]  # 最近一轮起点（保底）
        cut = last_start
        for start in user_idx[:-1]:
            candidate = rest[start:]
            if self._estimate_size(self.messages[:anchored] + candidate) <= self.compress_threshold:
                cut = start
                break

        if cut == 0:
            # 单轮（或多轮但全保留都不达标时 start 停在 0 之外）——
            # 具体说：cut==0 意味着"保留起点就是第一轮"= 不滑出，
            # 典型场景是第一轮就读了超大文件：保留原文供当前任务引用
            if self.verbose:
                self.console.print(
                    f"[yellow]memory[/] 最近轮次已超预算({self.compress_threshold}字符)，保留原文供当前任务引用"
                )
            return

        old, keep = rest[:cut], rest[cut:]
        if self.verbose:
            self.console.print(
                f"[yellow]memory[/] 总量超预算，滑出 {len(old)} 条最旧消息（保留最近轮原文）"
            )

        summary = self.compressor.extract_summary(old, existing_summary=self._current_summary())
        if self.archive_path:
            self.compressor.archive(old, self.archive_path)
        self.messages = self.messages[:anchored] + keep
        self._set_summary(summary)

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def _chat_stream_once(self, request: list[dict]) -> dict:
        """流式 Reason：调 LLM 并实时回调 content 增量，返回完整 assistant 消息 dict。

        - content 片段逐段回调 self._on_token（若设置），实现"边生成边显示"
        - 流结束时拿到重组好的消息（content + tool_calls），
          与 _msg_to_dict 输出格式一致，可继续 append / 判断是否调工具
        - 一轮里如果最终是"要调工具"，这轮 content 通常为空 → 不会误显示
        """
        msg_dict: dict = {}
        for event, payload in self.llm.chat_stream(request, tools=self.registry.schemas):
            if event == "content":
                if self._on_token:
                    self._on_token(payload)
            else:  # "done"
                msg_dict = payload
        return msg_dict

    def run(
        self,
        user_input: str,
        stream: bool = False,
        on_token=None,
        on_event: Callable[[dict], None] | None = None,
    ) -> str:
        """执行一次完整任务，返回最终答案文本。

        stream=True 时（流式输出）：
        - 最终答案会边生成边通过 on_token(text_piece) 回调输出
        - 工具轮不产生用户可见内容（content 为空，由 verbose 日志展示过程）
        - 返回值仍是完整答案字符串；CLI 层在流式模式下不应重复打印

        on_event(event: dict)：结构化过程事件回调（Web UI 等观察者用），
        与 verbose 解耦——关掉 verbose 也能拿到完整过程。事件契约见类 docstring。
        回调异常被吞掉，不影响任务执行。

        流程（与三层记忆的关系）：
        ① 长期记忆：user_input 的增量信息先 merge 进 TaskMemory
           （固定预算，重复去重 → 大小恒定，不随会话增长）
        ② 短期记忆：user_input 原文进入窗口区（有界）
        ③ ReAct 循环：每轮组装"三层记忆 + 窗口"请求 LLM；
           有 tool_calls 就执行并回喂；无则返回最终答案
        ④ 每轮末尾：窗口滑动 + 总量兜底，把记忆压力转移给中期层
        """
        # 记录当前用户提问（供 D1-3 知识检索注入使用）
        self._last_user_input = user_input
        # 流式回调（本轮有效，不跨任务残留）
        self._on_token = on_token if stream else None
        # 观察者回调（本轮有效）
        self._on_event = on_event

        # D3-2：清空工具调用历史，依赖检查按任务隔离
        self.registry.reset_call_history()

        # ① 长期记忆：合并增量（用户的目标/约束/偏好进任务记忆块）
        if self.task_memory:
            self.task_memory.merge(user_input)

        # ② 短期记忆：原文进窗口
        self.messages.append({"role": "user", "content": user_input})

        for step in range(1, self.max_steps + 1):
            # 观察者：一轮 Reason 开始
            self._emit("step", step=step, max_steps=self.max_steps)

            # Reason：带着"三层记忆 + 窗口"的全部上下文问模型
            request = self._build_request_messages()

            # D1-1 上下文审计：verbose 模式下打印各层占比（诊断用）
            if self.verbose:
                self.console.print(render_report(audit(request, model=self.audit_model)))

            # Reason：流式 or 一次性调用（结果消息格式统一为 dict）
            if stream:
                msg = self._chat_stream_once(request)
            else:
                msg = _msg_to_dict(self.llm.chat(request, tools=self.registry.schemas))
            self.messages.append(msg)

            # 模型没有发起工具调用 → 这就是最终答案
            if not msg.get("tool_calls"):
                content = msg.get("content") or "（无输出）"
                self._emit("done", content=content, steps=step, stopped=False)
                return content

            # Act + Observe：逐个执行工具，结果以 tool 消息回喂
            if self.verbose:
                self.console.print(f"[yellow]step {step}/{self.max_steps}[/]")

            for call in msg["tool_calls"]:
                name = call["function"]["name"]
                raw_args = call["function"]["arguments"]
                # 观察者：工具即将执行（args 已解析成 dict，便于展示）
                self._emit("tool_start", name=name, args=self._parse_args(raw_args))
                if self.verbose:
                    self.console.print(f"  [cyan]tool[/] {name}({raw_args})")
                # 执行工具；confirm 级工具会先过 approver 审批
                # （审批被拒返回 [用户拒绝]，见 docs/fallback-policy.md 场景 D）
                result = self.registry.execute(call, approver=self.approver)

                # D5-1 质量闭环：验证点钩子。
                # 仅对"工具本身执行成功"的情况运行验证（被拒/出错时验证无意义），
                # 失败的验证输出附加到 tool 结果回喂，模型可读并修复 → 再验证，循环至绿
                if (
                    self.validation_hooks
                    and name in self.validation_hooks
                    and not result.startswith(("[用户拒绝]", "[工具执行出错]", "[未知工具]"))
                ):
                    hook_output = self._run_validation_hooks(name)
                    if hook_output:
                        result = f"{result}\n{hook_output}"
                        self._emit("hooks", name=name, output=hook_output)
                        if self.verbose:
                            self.console.print(f"  [magenta]hooks[/] {hook_output[:200]}")

                # 观察者：工具执行结束（denied/error 让前端能高亮失败）
                self._emit(
                    "tool_result",
                    name=name,
                    output=result,
                    denied=result.startswith("[用户拒绝]"),
                    error=result.startswith(("[工具执行出错]", "[未知工具]", "[前置工具未调用]")),
                )
                if self.verbose:
                    preview = result[:200] + ("..." if len(result) > 200 else "")
                    self.console.print(f"  [green]result[/] {preview}")
                # tool 消息必须带 tool_call_id，与 assistant 消息里的 id 一一对应
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": result,
                    }
                )

            # ④ 记忆压力管理（在下一轮 Reason 之前）：
            #     窗口滑动（短期 → 中期）→ 总量兜底（防单轮撑爆）
            self._trim_window()
            self._emergency_compress()

            # 循环回到 Reason，模型会看到新加入的 tool 结果

        content = "达到最大步数，任务未完成。"
        self._emit("done", content=content, steps=self.max_steps, stopped=True)
        return content
