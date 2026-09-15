"""会话持久化：把 messages 保存到磁盘，重启后接着聊。

解决"进程退出/重启后上下文丢失"（文章第 7 节进阶项）。

边界说明（重要，避免误解）：
- 会话保存的是【当前上下文】——已被压缩的轮次保存时就是摘要，
  细节不会因"保存再读回"而回来；压缩时的原文已外置在
  archive_path 的 JSONL（见 docs/fallback-policy.md 场景 E）
- 跨会话找回更早细节的正确姿势：
  1. 最近轮次 → 本轮修复后保留原文（直接引用）
  2. 更早轮次 → 摘要槽（要点级） + knowledge 检索注入补细节
- 本模块只做"上下文快照"：加载后摘要槽照常工作（前缀识别），
  TaskMemory 从新的 user 消息重新 merge（目标级记忆，可接受）
"""

import json
from pathlib import Path


def save_session(agent, path: str) -> str:
    """把 agent.messages 保存为 JSON（不含 system 消息）。

    为什么丢弃 system：重启时用当前 SYSTEM_PROMPT，
    避免系统提示词版本漂移导致旧会话行为不一致。
    摘要槽（role=system 但以【历史增量摘要】开头）会被一并丢弃吗？
    不会——它也在 messages 里……注意：摘要槽 role 是 system，
    这里同样会被过滤！见 _load 里按前缀重建的逻辑。
    """
    # 保留摘要槽（系统提示词外的 system 消息）
    keep = []
    for m in agent.messages:
        if m["role"] == "system":
            content = str(m.get("content") or "")
            if content.startswith("【历史增量摘要】"):
                keep.append(m)  # 中期记忆：要点级历史
            # 其他 system（SYSTEM_PROMPT / 任务记忆临时注入）不落盘
        else:
            keep.append(m)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(keep, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


def load_session(agent, path: str) -> int:
    """把会话历史加载进 agent（追加在 system_prompt 之后）。

    返回加载的消息条数；文件不存在 / 损坏 / 空 → 返回 0（静默降级）。
    """
    p = Path(path)
    if not p.exists():
        return 0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    if not isinstance(data, list):
        return 0
    # 只接受合法 role，防止坏文件污染上下文
    valid = [m for m in data if m.get("role") in ("user", "assistant", "tool")]
    if not valid:
        return 0
    agent.messages = [agent.messages[0]] + valid
    return len(valid)
