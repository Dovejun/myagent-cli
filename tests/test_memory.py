"""记忆压缩模块测试（用假 LLM，不依赖真实 API，可离线运行）。"""

from agent.memory import (
    MemoryCompressor,
    TaskMemory,
    apply_window,
    maybe_compress,
)


class FakeLLM:
    """模拟 LLMClient：chat 返回带 content 的 message 对象。"""

    def chat(self, messages, tools=None):
        class Msg:
            content = (
                "已确认事实：\n- 用户目标是做 CLI\n"
                "关键决策：\n- 无\n"
                "进行中的任务：\n- 无\n"
                "关键数据/路径/数值：\n- 无"
            )

        return Msg()


def test_estimate_size():
    c = MemoryCompressor(FakeLLM())
    msgs = [{"role": "system", "content": "abc"}, {"role": "user", "content": "def"}]
    assert c.estimate_size(msgs) == 6


def test_flatten_handles_tool_calls():
    c = MemoryCompressor(FakeLLM())
    msgs = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "a"}',
                    }
                }
            ],
        }
    ]
    flat = c._flatten(msgs)
    assert "read_file" in flat
    assert "tool_calls" not in flat


def test_compress_keeps_system_and_user_only():
    c = MemoryCompressor(FakeLLM())
    msgs = [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "帮我读文件"},
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "tool", "tool_call_id": "1", "content": "文件内容..."},
    ]
    out = c.compress(msgs)
    roles = [m["role"] for m in out]
    # 摘要以 system 注入 + 原 system + 原 user 保留，过程消息全部进压缩池
    assert roles.count("system") == 2
    assert roles.count("user") == 1
    assert all(m["role"] in ("system", "user") for m in out)
    assert "已确认事实" in out[0]["content"]


def test_compress_no_old_messages_returns_unchanged():
    c = MemoryCompressor(FakeLLM())
    msgs = [{"role": "system", "content": "你是助手"}, {"role": "user", "content": "hi"}]
    assert c.compress(msgs) == msgs


def test_compress_archives_original(tmp_path):
    c = MemoryCompressor(FakeLLM())
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "tool", "tool_call_id": "1", "content": "原始长内容"},
    ]
    archive = tmp_path / "mem.jsonl"
    out = c.compress(msgs, archive_path=str(archive))
    assert archive.exists()
    content = archive.read_text(encoding="utf-8")
    assert "原始长内容" in content  # 原文兜底存档，未丢失
    assert len(out) < len(msgs)


def test_maybe_compress_below_threshold_returns_same():
    c = MemoryCompressor(FakeLLM())
    msgs = [{"role": "user", "content": "short"}]
    assert maybe_compress(c, msgs, threshold=100) is msgs


def test_maybe_compress_above_threshold_compresses():
    c = MemoryCompressor(FakeLLM())
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "x" * 200},
        {"role": "tool", "tool_call_id": "1", "content": "y" * 200},
    ]
    out = maybe_compress(c, msgs, threshold=100)
    assert out is not msgs
    assert len(out) < len(msgs)


# ---------- extract_summary：窗口滑出的保真提取 ----------


def test_extract_summary_returns_text():
    c = MemoryCompressor(FakeLLM())
    msgs = [{"role": "user", "content": "随便什么内容"}]
    s = c.extract_summary(msgs)
    assert isinstance(s, str)
    assert "已确认事实" in s  # 显式提取：四类必答清单


def test_extract_summary_with_existing_summary():
    c = MemoryCompressor(FakeLLM())
    msgs = [{"role": "tool", "tool_call_id": "1", "content": "结果"}]
    s = c.extract_summary(msgs, existing_summary="旧摘要要点")
    assert isinstance(s, str)
    assert len(s) > 0


# ---------- TaskMemory：固定预算，控制 token 消耗 ----------


class MergeLLM:
    """模拟合并 LLM：返回"现有记忆 + 新增信息"，验证合并与去重逻辑。"""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        sys_content = messages[0]["content"]
        # 从 system prompt 里解析现有记忆与新增内容
        mem = sys_content.split("【现有任务记忆】")[1].split("【新增对话】")[0].strip()
        new = sys_content.split("【新增对话】")[1].strip()
        merged = mem + "\n" + new

        class Msg:
            content = merged

        return Msg()


def test_task_memory_first_merge_sets_initial():
    tm = TaskMemory(MergeLLM(), budget=100)
    tm.merge("帮我做一个 CLI 工具")
    assert "CLI" in tm.content
    assert tm.size <= 100


def test_task_memory_merge_appends_new_info():
    llm = MergeLLM()
    tm = TaskMemory(llm, budget=500)
    tm.merge("目标：做 CLI")
    tm.merge("新增约束：用 Python")
    assert "做 CLI" in tm.content
    assert "用 Python" in tm.content
    assert llm.calls >= 1


def test_task_memory_budget_truncates():
    llm = MergeLLM()
    tm = TaskMemory(llm, budget=50)
    tm.merge("x" * 30)
    tm.merge("y" * 30)
    assert tm.size <= 50


def test_task_memory_to_system_message_role():
    tm = TaskMemory(MergeLLM(), budget=100)
    tm.merge("目标：做 CLI")
    msg = tm.to_system_message()
    assert msg["role"] == "system"
    assert "任务记忆" in msg["content"]


# ---------- 滑动窗口：有界 ----------


def test_apply_window_keeps_system_and_recent():
    msgs = [{"role": "system", "content": "s"}]
    for i in range(20):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    out = apply_window(msgs, keep_roles=("system",), n_rounds=5)
    assert out[0]["role"] == "system"
    assert len(out) <= 1 + 5 * 2
    assert out[-1]["content"] == "a19"  # 最近的保留
    assert all(m["content"] != "u0" for m in out)  # 最早的滑出


def test_apply_window_small_list_unchanged():
    msgs = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    assert apply_window(msgs, n_rounds=10) == msgs
