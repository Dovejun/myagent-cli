"""记忆模块：压缩保真 + 任务记忆（固定预算，控制 token 消耗）。

## 一、保真压缩（MemoryCompressor）
分层保留 + 显式信息提取 + 增量摘要 + 原文外置兜底：
1. 规则分级：system / user 消息永不压缩（用户原话永远重要）
2. 显式提取：不用"自由总结"，而是给 LLM 一份"必答清单"
   （事实/决策/待办/关键数据），让重要内容有强制出口
3. 增量摘要：只处理上次摘要之后的增量消息，旧摘要作为高优先级输入，
   避免"压缩的压缩"导致信息逐级稀释
4. 双写外置：被压缩的完整原文追加写入 JSONL 存档，摘要只是索引

## 二、任务记忆（TaskMemory）—— 解决 token 大量消耗
"system/user 永不压缩"的正确解读是"信息不丢"，不是"原文全留"。
真正让 token 爆炸的是 user 消息随会话无限累积。解法：
- 每轮把 user 消息的【增量信息】merge 进一个固定预算的任务记忆块
  （目标/约束/状态），重复内容去重 → 大小恒定，不随轮数增长
- user 原文进入滑动窗口（有界）或外置存档，而不是常驻上下文
- 三层架构：长期（任务记忆，固定）→ 中期（增量摘要，有上限）
  → 短期（窗口，固定 N 轮）→ 总消耗收敛到常数
"""

import json
import os
from datetime import datetime

EXTRACT_PROMPT = """请阅读下面的对话历史，提取其中【必须长期记住】的信息，严格按以下格式输出（不要输出其他内容）：

已确认事实：
- ...

关键决策：
- ...

进行中的任务：
- ...

关键数据/路径/数值：
- ...

规则：
- 用户明确提出的要求、偏好、约束必须原样保留
- 工具返回的关键数值、文件路径、结论必须保留
- 省略中间过程、失败尝试、重复内容
- 没有某类内容就写"无"
"""

# 永不压缩的角色（L0）
KEEP_ROLES = ("system", "user")


class MemoryCompressor:
    """增量式记忆压缩器。

    用法（在 Agent 循环中，token 超阈值时触发）：
        compressor = MemoryCompressor(llm)
        messages = compressor.compress(messages, archive_path="memory_store.jsonl")
    """

    def __init__(self, llm, keep_roles=KEEP_ROLES, max_archive_bytes: int = 5 * 1024 * 1024):
        self.llm = llm
        self.keep_roles = keep_roles
        self.max_archive_bytes = max_archive_bytes

    # ---------- 对外接口 ----------

    def extract_summary(self, messages: list[dict], existing_summary: str | None = None) -> str:
        """把消息显式提取成记忆摘要（增量：可传入旧摘要作为高优先级输入）。

        供窗口裁剪（core.py 的 _trim_window）复用：
        滑出窗口的旧消息 + 现有摘要 → 合并提取 → 返回新摘要文本。
        摘要保留原有要点，只增补新信息，避免"压缩的压缩"造成信息稀释。
        """
        history = self._flatten(messages)
        prompt: list[dict] = []
        if existing_summary:
            prompt.append(
                {
                    "role": "system",
                    "content": "【已有摘要，必须保留其全部要点】\n" + existing_summary,
                }
            )
        prompt.append({"role": "system", "content": EXTRACT_PROMPT})
        prompt.append({"role": "user", "content": history})
        resp = self.llm.chat(prompt)
        return (resp.content or "").strip()

    def compress(self, messages: list[dict], archive_path: str | None = None) -> list[dict]:
        """压缩超长对话，返回新的 messages（保留层 + 记忆摘要）。

        - 压缩产物是一段结构化记忆文本，以 system 消息注入最前
        - 若 archive_path 提供，被压缩原文会双写存档（原文不丢，只出上下文）
        """
        keep = [m for m in messages if m["role"] in self.keep_roles]
        old = [m for m in messages if m["role"] not in self.keep_roles]
        if not old:
            return messages

        # ① 显式信息提取 pass（不带 tools 的普通对话）
        extracted = self.extract_summary(old) or "已确认事实：\n无"

        # ② 重组：记忆摘要（system 层） + 永久保留层
        memory_msg = {"role": "system", "content": "【之前的对话记忆，必须遵守】\n" + extracted}
        new_messages = [memory_msg] + keep

        # ③ 双写外置：原文不删，落盘兜底
        if archive_path:
            self.archive(old, archive_path)

        return new_messages

    def estimate_size(self, messages: list[dict]) -> int:
        """粗略估算字符数，用于触发判断（生产可用 tiktoken 数 token）。"""
        return sum(
            len(str(m.get("content") or m.get("tool_calls") or "")) for m in messages
        )

    # ---------- 内部工具 ----------

    def _flatten(self, messages: list[dict]) -> str:
        """把消息列表拍平成文本，供提取 prompt 使用。"""
        parts = []
        for m in messages:
            body = m.get("content")
            if body is None and m.get("tool_calls"):
                body = ", ".join(
                    f"{tc['function']['name']}({tc['function']['arguments']})"
                    for tc in m["tool_calls"]
                )
            parts.append(f"[{m['role']}]\n{body}")
        return "\n\n".join(parts)

    def archive(self, messages: list[dict], path: str) -> None:
        """把被压缩的原文追加写入 JSONL（每条一行），控制文件大小。

        公开方法：core.py 的窗口裁剪同样用它做"原文双写兜底"，
        保证被滑出上下文的原始消息始终有磁盘副本可找回。
        """
        records = [
            {"ts": datetime.now().isoformat(timespec="seconds"), "messages": messages}
        ]
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # 简单轮转：超限就改名加 .old（保留最近两份）
        if os.path.getsize(path) > self.max_archive_bytes:
            os.replace(path, path + ".old")


def maybe_compress(
    compressor: MemoryCompressor,
    messages: list[dict],
    threshold: int = 60000,
    archive_path: str | None = None,
) -> list[dict]:
    """触发封装：字符数超阈值才压缩，否则原样返回。

    60000 字符对 128k 窗口模型是保守触发点（约 1/3 窗口），
    可按实际模型窗口调整。
    """
    if compressor.estimate_size(messages) > threshold:
        return compressor.compress(messages, archive_path=archive_path)
    return messages


# ---------------------------------------------------------------------------
# 任务记忆：固定预算的长期记忆，解决 user 消息无限累积导致的 token 消耗
# ---------------------------------------------------------------------------

TASK_MERGE_PROMPT = """请把【新增对话】中【新增的】重要信息合并进【现有任务记忆】。

规则：
- 只保留新增的目标、约束、偏好、需求变更、状态变化
- 与现有记忆重复的内容【不要】输出
- 现有记忆没有的新信息必须保留
- 直接输出【合并后的完整任务记忆】（包含原有内容 + 新增内容）

【现有任务记忆】
{memory}

【新增对话】
{new_content}
"""


class TaskMemory:
    """固定预算的任务记忆：目标 / 约束 / 状态。

    每轮调用 merge() 把 user 消息的增量信息合并进来（LLM 去重），
    记忆块大小恒定（budget 上限），不随会话轮数增长。
    配合滑动窗口使用：user 原文进窗口（有界），要点进这里（恒定）。

    用法（Agent 循环中）：
        tm = TaskMemory(llm, budget=1500)
        tm.merge(user_input)                       # 每轮新消息进来时
        messages = [system] + [tm.to_system_message()] + window_messages
    """

    def __init__(self, llm, budget: int = 1500, initial: str = "") -> None:
        self.llm = llm
        self.budget = budget
        self.content = initial

    def merge(self, new_content: str) -> None:
        """把新消息的增量信息合并进任务记忆（去重，只留新增）。"""
        if not new_content.strip():
            return
        if not self.content:
            # 第一条 user 消息：直接提取任务目标，不追加历史
            self.content = new_content[: self.budget]
            return
        try:
            resp = self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": TASK_MERGE_PROMPT.format(
                            memory=self.content, new_content=new_content
                        ),
                    },
                    {"role": "user", "content": "请合并。"},
                ]
            )
            merged = (resp.content or "").strip()
            if merged:
                self.content = merged[: self.budget]
        except Exception:  # noqa: BLE001 —— merge 失败不影响主流程，保留旧记忆
            pass

    def to_system_message(self) -> dict:
        """输出固定大小的 system 记忆消息，注入 messages 最前。"""
        return {
            "role": "system",
            "content": "【任务记忆，必须遵守】\n" + (self.content or "（暂无）"),
        }

    @property
    def size(self) -> int:
        return len(self.content)


def apply_window(messages: list[dict], keep_roles: tuple = ("system",), n_rounds: int = 10) -> list[dict]:
    """滑动窗口：只保留最近 n_rounds 轮的完整消息，system 层常驻。

    一"轮" = 一条 user 消息及其后续的 assistant/tool 消息。
    被滑出的消息应由调用方先喂给 MemoryCompressor 压缩/外置，再丢弃。
    """
    keep = [m for m in messages if m["role"] in keep_roles]
    rest = [m for m in messages if m["role"] not in keep_roles]
    if len(rest) <= n_rounds * 2:  # 粗略：每轮至少 user + assistant 两条
        return messages
    return keep + rest[-(n_rounds * 2):]

