"""Web UI 测试（离线）：会话广播、审批桥、事件契约、静态服务。

不触真实 LLM：build_agent 被替换成 FakeAgent，
只验证 webui/server.py 自己的编排逻辑（这些才是最容易写错的部分）。
"""

import json
import queue
import threading
import urllib.error
import urllib.request

import pytest

from webui import server as ws


# ----------------------------------------------------------------------
# 测试替身
# ----------------------------------------------------------------------
class FakeAgent:
    """最小 Agent 替身：只记录消息，并可选地回放事件/流式片段。"""

    def __init__(self):
        self.messages = [{"role": "system", "content": "sys"}]
        self.runs: list[str] = []
        self.emit_events = True

    def run(self, user_input, stream=False, on_token=None, on_event=None):
        self.messages.append({"role": "user", "content": user_input})
        self.runs.append(user_input)
        if self.emit_events and on_event:
            on_event({"type": "step", "step": 1, "max_steps": 3})
        if stream and on_token:
            on_token("你")
            on_token("好")
        if self.emit_events and on_event:
            on_event({"type": "done", "content": "你好", "steps": 1, "stopped": False})
        self.messages.append({"role": "assistant", "content": "你好"})
        return "你好"


@pytest.fixture(autouse=True)
def _patch_build_agent(monkeypatch):
    """所有测试都不装配真 Agent（否则会去读 .env / 初始化 OpenAI 客户端）。"""
    monkeypatch.setattr(ws, "build_agent", lambda cfg, approver=None: FakeAgent())


def _session() -> ws.Session:
    return ws.Session("s1", ws.AgentConfig())


class FakeWFile:
    def __init__(self):
        self.buf = b""

    def write(self, data: bytes) -> None:
        self.buf += data

    def flush(self) -> None:
        pass


# ----------------------------------------------------------------------
# 事件广播
# ----------------------------------------------------------------------
def test_broadcast_reaches_all_subscribers():
    s = _session()
    q1, q2 = s.subscribe(), s.subscribe()
    s.broadcast({"type": "token", "text": "x"})
    assert q1.get_nowait()["text"] == "x"
    assert q2.get_nowait()["text"] == "x"


def test_unsubscribe_stops_delivery():
    s = _session()
    q = s.subscribe()
    s.unsubscribe(q)
    s.broadcast({"type": "token", "text": "x"})
    assert q.empty()


def test_broadcast_without_subscriber_does_not_raise():
    _session().broadcast({"type": "token", "text": "x"})   # 不应抛异常


def test_slow_subscriber_does_not_block_broadcast():
    """队列满时丢事件，而不是把 Agent 线程卡死。"""
    s = _session()
    q = s.subscribe()
    for i in range(1200):                     # 超过 maxsize=1000
        s.broadcast({"type": "token", "text": str(i)})
    assert q.qsize() <= 1000                  # 满了就丢，不抛


# ----------------------------------------------------------------------
# 审批桥
# ----------------------------------------------------------------------
def _start_approval(session, name="run_shell", args=None):
    """在后台线程调用 approver（它会阻塞），返回 (结果槽, 线程)。"""
    box: dict = {}
    t = threading.Thread(
        target=lambda: box.update(v=session._approve(name, args or {"command": "ls"})),
        daemon=True,
    )
    t.start()
    return box, t


def test_approval_bridge_approve():
    s = _session()
    q = s.subscribe()
    box, t = _start_approval(s)

    req = q.get(timeout=2)
    assert req["type"] == "approval_required"
    assert req["name"] == "run_shell"
    assert req["args"] == {"command": "ls"}

    assert s.resolve_approval(req["id"], True) is True
    t.join(timeout=2)
    assert box["v"] is True

    resolved = q.get(timeout=2)
    assert resolved["type"] == "approval_resolved"
    assert resolved["approved"] is True and resolved["timeout"] is False


def test_approval_bridge_reject():
    s = _session()
    q = s.subscribe()
    box, t = _start_approval(s)
    req = q.get(timeout=2)
    s.resolve_approval(req["id"], False)
    t.join(timeout=2)
    assert box["v"] is False


def test_approval_timeout_treated_as_reject(monkeypatch):
    monkeypatch.setattr(ws, "APPROVAL_TIMEOUT", 0.15)
    s = _session()
    q = s.subscribe()
    box, t = _start_approval(s)
    req = q.get(timeout=2)
    t.join(timeout=2)
    assert box["v"] is False                       # 超时 = 拒绝
    assert s.resolve_approval(req["id"], True) is False   # 已失效，不能再改
    resolved = q.get(timeout=2)
    assert resolved["type"] == "approval_resolved" and resolved["timeout"] is True


def test_resolve_unknown_approval_returns_false():
    assert _session().resolve_approval("nope", True) is False


def test_reject_all_pending_unblocks_waiters():
    s = _session()
    q = s.subscribe()                          # 先订阅，避免错过广播
    boxes = [_start_approval(s) for _ in range(3)]
    for _ in range(3):
        assert q.get(timeout=2)["type"] == "approval_required"
    assert s.reject_all_pending() == 3
    for box, t in boxes:
        t.join(timeout=2)
        assert box["v"] is False


# ----------------------------------------------------------------------
# 任务执行
# ----------------------------------------------------------------------
def test_start_task_broadcasts_full_event_sequence():
    s = _session()
    q = s.subscribe()
    assert s.start_task("你好") is True

    seen = []
    while True:
        ev = q.get(timeout=3)
        seen.append(ev["type"])
        if ev["type"] == "idle":
            break
    assert seen[0] == "start"
    assert "step" in seen and "token" in seen and "done" in seen
    assert seen[-1] == "idle"


def test_start_task_rejects_concurrent_run():
    s = _session()
    # 手动占住忙标记，模拟"上一轮未结束"
    s._busy = True
    assert s.start_task("再来一次") is False
    s._busy = False


def test_task_error_is_broadcast_not_raised(monkeypatch):
    class Boom(FakeAgent):
        def run(self, *a, **kw):
            raise RuntimeError("模型爆炸了")

    monkeypatch.setattr(ws, "build_agent", lambda cfg, approver=None: Boom())
    s = _session()
    q = s.subscribe()
    s.start_task("x")
    types = []
    while True:
        ev = q.get(timeout=3)
        types.append(ev["type"])
        if ev["type"] == "error":
            assert "模型爆炸了" in ev["message"]
        if ev["type"] == "idle":
            break
    assert "error" in types and s.busy is False


# ----------------------------------------------------------------------
# 历史与重置
# ----------------------------------------------------------------------
def test_history_filters_system_and_tool():
    s = _session()
    s.agent.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "结果"},
        {"role": "assistant", "content": "答案"},
        {"role": "assistant", "content": "   "},
    ]
    assert s.history() == [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "答案"},
    ]


def test_reset_rebuilds_agent_and_broadcasts():
    s = _session()
    old = s.agent
    q = s.subscribe()
    s.reset()
    assert s.agent is not old
    assert q.get_nowait()["type"] == "reset"


def test_manager_returns_same_session_for_same_id():
    m = ws.SessionManager(ws.AgentConfig())
    assert m.get("a") is m.get("a")
    assert m.get("a") is not m.get("b")


# ----------------------------------------------------------------------
# SSE 帧格式
# ----------------------------------------------------------------------
def test_sse_frame_is_valid_chunk():
    h = ws.Handler.__new__(ws.Handler)
    h.wfile = FakeWFile()
    obj = {"type": "token", "text": "你好"}
    assert h._sse_send(obj) is True

    payload = f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")
    expected = b"%X\r\n" % len(payload) + payload + b"\r\n"
    assert h.wfile.buf == expected


def test_sse_close_writes_terminating_chunk():
    h = ws.Handler.__new__(ws.Handler)
    h.wfile = FakeWFile()
    h._sse_close()
    assert h.wfile.buf == b"0\r\n\r\n"


# ----------------------------------------------------------------------
# HTTP 层冒烟（真实 socket）
# ----------------------------------------------------------------------
@pytest.fixture
def live_server(monkeypatch):
    monkeypatch.setattr(ws.Handler, "sessions", ws.SessionManager(ws.AgentConfig()))
    httpd = ws.ThreadingHTTPServer(("127.0.0.1", 0), ws.Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_http_serves_index(live_server):
    with urllib.request.urlopen(f"{live_server}/", timeout=5) as r:
        body = r.read().decode("utf-8")
    assert r.status == 200
    assert "AI Agent" in body and "/static/app.js" in body


def test_http_serves_static_assets(live_server):
    for path, needle in (("/static/app.js", "EventSource"), ("/static/style.css", "--accent")):
        with urllib.request.urlopen(f"{live_server}{path}", timeout=5) as r:
            assert r.status == 200
            assert needle in r.read().decode("utf-8")


def test_http_static_path_traversal_blocked(live_server):
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(f"{live_server}/static/../server.py", timeout=5)
    assert ei.value.code == 404


def test_http_history_empty_without_session(live_server):
    with urllib.request.urlopen(f"{live_server}/api/history", timeout=5) as r:
        assert json.loads(r.read().decode("utf-8")) == {"messages": []}


def test_http_chat_stream_flow(live_server):
    """端到端走一遍 HTTP：POST 发消息 → 常驻 SSE 收到完整事件序列。"""
    sid = "http-test"

    # 1) 建立 SSE 连接（读到 idle 为止）
    req = urllib.request.Request(f"{live_server}/api/stream?session_id={sid}")
    conn = urllib.request.urlopen(req, timeout=5)
    try:
        # 2) 发消息
        payload = json.dumps({"session_id": sid, "message": "你好"}).encode("utf-8")
        post = urllib.request.Request(
            f"{live_server}/api/chat", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(post, timeout=5) as r:
            assert json.loads(r.read().decode("utf-8"))["ok"] is True

        # 3) 从 SSE 里读事件直到 idle
        types = []
        while True:
            line = conn.readline()
            if not line:
                break
            if line.startswith(b"data: "):
                ev = json.loads(line[6:].decode("utf-8"))
                types.append(ev["type"])
                if ev["type"] == "idle":
                    break
        assert "start" in types and "token" in types and "done" in types
    finally:
        conn.close()
