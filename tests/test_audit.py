"""上下文审计测试：分层统计、token 计数、报告渲染（离线，不依赖 API）。"""

from agent.context_audit import audit, count_tokens, render_report


def _sample_messages():
    """构造一个典型的三层记忆请求消息序列。"""
    return [
        {"role": "system", "content": "你是命令行 AI 助手，行为准则..."},
        {"role": "system", "content": "【任务记忆，必须遵守】\n目标：做 CLI"},
        {"role": "system", "content": "【历史增量摘要】\n已确认事实：- 用 Python"},
        {"role": "user", "content": "读 README 并总结"},
        {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "read_file", "arguments": '{"path": "README.md"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "# ai-agent-cli 项目说明..."},
    ]


def test_audit_layers_counts():
    result = audit(_sample_messages())
    layers = result["layers"]
    assert layers["system_prompt"]["count"] == 1
    assert layers["task_memory"]["count"] == 1
    assert layers["summary"]["count"] == 1
    assert layers["window"]["count"] == 3  # user + assistant + tool


def test_audit_chars_match_content():
    msgs = _sample_messages()
    result = audit(msgs)
    layers = result["layers"]
    assert layers["window"]["chars"] == sum(
        len(str(m.get("content") or m.get("tool_calls") or ""))
        for m in msgs[3:]
    )


def test_audit_total_is_sum_of_layers():
    result = audit(_sample_messages())
    layers_sum = sum(s["chars"] for s in result["layers"].values())
    assert result["total"]["chars"] == layers_sum


def test_audit_empty_messages():
    result = audit([])
    assert result["total"]["chars"] == 0
    assert result["total"]["tokens"] == 0


def test_audit_plain_messages_all_window():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    result = audit(msgs)
    assert result["layers"]["window"]["count"] == 2
    assert result["layers"]["system_prompt"]["count"] == 0


def test_count_tokens_returns_positive_int():
    assert isinstance(count_tokens("你好世界 hello"), int)
    assert count_tokens("") >= 0


def test_count_tokens_fallback_when_text_long():
    # 未装 tiktoken 时也应返回正数（字符估算兜底）
    n = count_tokens("中" * 100)
    assert n >= 1


def test_render_report_contains_layer_names():
    report = render_report(audit(_sample_messages()))
    assert "系统提示" in report
    assert "任务记忆" in report
    assert "增量摘要" in report
    assert "窗口区" in report
    assert "总计" in report
