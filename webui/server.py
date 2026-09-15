"""Web UI 服务端（零依赖）：把 Agent 包成浏览器可用的聊天界面。

====================================================================
一、为什么不用 FastAPI / Gradio（技术选型）
====================================================================
项目定位是"零重依赖教学 harness"，所以：
- 只用标准库 http.server（ThreadingHTTPServer），装完 requirements 就能跑
- 前端零框架（原生 HTML/CSS/JS），因为 Agent 需要工具调用卡片、审批弹窗
  这类"非标准聊天"交互，用现成聊天组件反而要跟它的抽象打架
- 传输用 SSE（Server-Sent Events）：服务端→浏览器单向推送足够，
  浏览器→服务端用普通 POST，比 WebSocket 少一层协议负担

====================================================================
二、架构：一条 SSE 长连接 + 两个 POST
====================================================================
    浏览器 ── GET  /api/stream?session_id=X ──▶ SSE 常驻，收所有事件
           ── POST /api/chat ────────────────▶ 后台线程跑 Agent.run()
           ── POST /api/approve ─────────────▶ 回填审批结果

    为什么"事件走常驻 SSE"而不是"chat 响应直接就是 SSE 流"：
    审批答案是异步到达的（用户可能 30 秒后才点），如果它必须从 chat 的
    响应流里出去，就要把两个方向的时序硬绑在一起。常驻通道让
    "命令"与"事件"彻底解耦，将来加"会话被其他标签页改动"这类通知也零成本。

====================================================================
三、审批桥（本项目最有意思的一处）
====================================================================
Agent 是同步阻塞的：approver(name, args) -> bool 必须在返回前拿到答案。
但答案来自另一个 HTTP 连接。解法：

    approver 里 broadcast 一个 approval_required 事件
    → 在 threading.Event 上阻塞等待（带超时兜底）
    → 浏览器弹窗，用户点"批准/拒绝" → POST /api/approve
    → 服务端 resolve() 里 set() 事件 → Agent 线程继续

于是同步 Agent 无缝接入异步 Web，核心循环一行没改。
超时（默认 180s）按拒绝处理，与 CLI 的"直接回车 = 拒绝"语义一致。
"""

import argparse
import json
import mimetypes
import os
import queue
import sys
import threading
import time
import uuid
import webbrowser
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# 允许两种运行方式：python agent_cli.py --web / python webui/server.py
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AgentConfig, build_agent  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
APPROVAL_TIMEOUT = 180.0   # 审批等待秒数，超时按拒绝处理
SSE_HEARTBEAT = 15.0       # SSE 心跳间隔（秒），防中间层掐断空闲连接


# ======================================================================
# 会话
# ======================================================================
class Session:
    """一个浏览器会话 = 一个 Agent 实例 + 一条事件广播通道。

    并发模型：
    - 任务串行：self._busy 保证同一会话同时只跑一个 Agent.run
      （Agent 的 messages 是共享状态，并发跑会互相污染上下文）
    - 事件广播：订阅者（SSE 连接）放进 self._subscribers，逐个投递
    """

    def __init__(self, sid: str, config: AgentConfig) -> None:
        self.id = sid
        self.config = config
        self._busy = False
        self._busy_lock = threading.Lock()
        self._subscribers: list[queue.Queue] = []
        self._sub_lock = threading.Lock()
        # 待审批：{approval_id: {"event": Event, "approved": bool}}
        self._pending: dict[str, dict] = {}
        self._pending_lock = threading.Lock()
        self.agent = self._new_agent()

    def _new_agent(self):
        """按配置装配 Agent。

        - Web 模式永远关掉 verbose（过程信息走 SSE 事件，不往终端打）
        - config.approval=True  → 注入审批桥（浏览器弹窗确认）
        - config.approval=False → 传 None，走"全放行"（对应 --yes）
        """
        return build_agent(
            replace(self.config, verbose=False),
            approver=self._approve if self.config.approval else None,
        )

    # ---------- 事件广播 ----------

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self._sub_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._sub_lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def broadcast(self, event: dict) -> None:
        """向所有订阅者投递事件（队列满则丢弃，不阻塞 Agent 线程）。"""
        with self._sub_lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass  # 订阅者跟不上：丢事件，不能让 Agent 卡住

    # ---------- 审批桥 ----------

    def _approve(self, name: str, args: dict) -> bool:
        """注入给 Agent 的 approver：推事件 → 阻塞等前端 → 返回结果。

        超时或连接断开都返回 False（拒绝），保证 Agent 不会永久挂死。
        """
        aid = uuid.uuid4().hex[:12]
        rec = {"event": threading.Event(), "approved": False}
        with self._pending_lock:
            self._pending[aid] = rec

        self.broadcast({
            "type": "approval_required",
            "id": aid,
            "name": name,
            "args": args,
        })

        got = rec["event"].wait(timeout=APPROVAL_TIMEOUT)

        with self._pending_lock:
            self._pending.pop(aid, None)
        if not got:
            self.broadcast({"type": "approval_resolved", "id": aid, "approved": False,
                            "timeout": True})
            return False

        self.broadcast({"type": "approval_resolved", "id": aid,
                        "approved": bool(rec["approved"]), "timeout": False})
        return bool(rec["approved"])

    def resolve_approval(self, aid: str, approved: bool) -> bool:
        """前端提交审批结果：唤醒阻塞中的 Agent 线程。"""
        with self._pending_lock:
            rec = self._pending.get(aid)
        if rec is None:
            return False  # 超时已清掉 / 重复提交
        rec["approved"] = bool(approved)
        rec["event"].set()
        return True

    def reject_all_pending(self) -> int:
        """会话重置时，把所有挂起的审批按拒绝放行（避免 Agent 线程永久阻塞）。"""
        with self._pending_lock:
            recs = list(self._pending.values())
            self._pending.clear()
        for rec in recs:
            rec["approved"] = False
            rec["event"].set()
        return len(recs)

    # ---------- 任务执行 ----------

    @property
    def busy(self) -> bool:
        return self._busy

    def start_task(self, message: str) -> bool:
        """启动一次 Agent.run（后台线程）。已有任务在跑则拒绝。"""
        with self._busy_lock:
            if self._busy:
                return False
            self._busy = True
        threading.Thread(target=self._run, args=(message,), daemon=True).start()
        return True

    def _run(self, message: str) -> None:
        self.broadcast({"type": "start", "message": message})
        try:
            self.agent.run(
                message,
                stream=True,   # 流式：token 事件实时到达
                on_token=lambda t: self.broadcast({"type": "token", "text": t}),
                on_event=self.broadcast,
            )
        except Exception as e:  # noqa: BLE001 —— 兜底：错误也要让前端看见
            self.broadcast({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            with self._busy_lock:
                self._busy = False
            self.broadcast({"type": "idle"})

    # ---------- 历史 ----------

    def history(self) -> list[dict]:
        """返回可展示的对话历史（过滤 system/tool 与空内容消息）。"""
        out = []
        for m in self.agent.messages:
            role = m.get("role")
            content = m.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                out.append({"role": role, "content": content})
        return out

    def reset(self) -> None:
        """重置会话：放行挂起审批 + 重建 Agent（清空上下文）。"""
        self.reject_all_pending()
        self.agent = self._new_agent()
        self.broadcast({"type": "reset"})


class SessionManager:
    """session_id → Session 的内存表（本地工具，不做持久化与淘汰）。"""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get(self, sid: str) -> Session:
        with self._lock:
            s = self._sessions.get(sid)
            if s is None:
                s = Session(sid, self.config)
                self._sessions[sid] = s
            return s

    def drop(self, sid: str) -> None:
        with self._lock:
            self._sessions.pop(sid, None)


# ======================================================================
# HTTP 处理
# ======================================================================
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"   # 常驻 SSE 与 keep-alive 需要
    server_version = "ai-agent-cli-webui"

    sessions: SessionManager = None      # 由 serve() 注入
    config: AgentConfig = None

    # ---------- 工具方法 ----------

    def log_message(self, fmt: str, *args) -> None:
        """静音默认访问日志（SSE 长连接会刷屏）；只留错误。"""
        if args and str(args[0]).startswith(("4", "5")):
            sys.stderr.write("[webui] %s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, obj, status: int = 200) -> None:
        self._send(status, "application/json; charset=utf-8",
                   json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return {}
        if length <= 0:
            return {}
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    # ---------- SSE ----------

    def _sse_open(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")   # 禁用 nginx 缓冲（若有）
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _chunk(self, payload: bytes) -> bool:
        """写一个 HTTP chunk；连接已断返回 False。"""
        try:
            self.wfile.write(b"%X\r\n" % len(payload) + payload + b"\r\n")
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False

    def _sse_send(self, obj: dict) -> bool:
        data = json.dumps(obj, ensure_ascii=False)
        return self._chunk(f"data: {data}\n\n".encode("utf-8"))

    def _sse_close(self) -> None:
        self._chunk(b"")   # 终止 chunk

    # ---------- 路由 ----------

    def do_GET(self) -> None:  # noqa: N802 —— BaseHTTPRequestHandler 约定
        path = self.path.split("?", 1)[0]
        if path == "/api/stream":
            self._handle_stream()
        elif path == "/api/history":
            self._handle_history()
        elif path in ("/", "/index.html"):
            self._serve_static("index.html")
        elif path == "/favicon.ico":
            self._send(204, "image/x-icon", b"")
        elif path.startswith("/static/"):
            self._serve_static(path[len("/static/"):])
        else:
            self._send(404, "text/plain; charset=utf-8", b"404 Not Found")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        body = self._read_json()
        sid = str(body.get("session_id") or "")

        if path == "/api/chat":
            message = str(body.get("message") or "").strip()
            if not message:
                self._send_json({"ok": False, "error": "消息为空"}, 400)
                return
            session = self.sessions.get(sid)
            if not session.start_task(message):
                self._send_json({"ok": False, "error": "上一轮还在执行中"}, 409)
                return
            self._send_json({"ok": True})

        elif path == "/api/approve":
            session = self.sessions.get(sid)
            ok = session.resolve_approval(
                str(body.get("approval_id") or ""), bool(body.get("approved"))
            )
            self._send_json({"ok": ok, "error": None if ok else "审批已失效或超时"})

        elif path == "/api/reset":
            self.sessions.get(sid).reset()
            self._send_json({"ok": True})

        elif path == "/api/stop":
            # 停止 = 拒绝所有挂起审批，让阻塞中的 Agent 线程尽快解套
            n = self.sessions.get(sid).reject_all_pending()
            self._send_json({"ok": True, "rejected": n})

        else:
            self._send_json({"ok": False, "error": "未知接口"}, 404)

    # ---------- 各接口实现 ----------

    def _handle_stream(self) -> None:
        from urllib.parse import parse_qs, urlparse

        qs = parse_qs(urlparse(self.path).query)
        sid = (qs.get("session_id") or [""])[0] or uuid.uuid4().hex[:12]
        session = self.sessions.get(sid)
        q = session.subscribe()

        self._sse_open()
        if not self._sse_send({"type": "ready", "session_id": sid, "busy": session.busy}):
            session.unsubscribe(q)
            return

        try:
            while True:
                try:
                    event = q.get(timeout=SSE_HEARTBEAT)
                except queue.Empty:
                    if not self._chunk(b": ping\n\n"):   # 心跳注释帧
                        break
                    continue
                if not self._sse_send(event):
                    break
        finally:
            session.unsubscribe(q)
            self._sse_close()

    def _handle_history(self) -> None:
        from urllib.parse import parse_qs, urlparse

        qs = parse_qs(urlparse(self.path).query)
        sid = (qs.get("session_id") or [""])[0]
        if not sid:
            self._send_json({"messages": []})
            return
        self._send_json({"messages": self.sessions.get(sid).history()})

    def _serve_static(self, rel: str) -> None:
        """静态文件服务（带路径穿越防护）。"""
        target = (STATIC_DIR / rel).resolve()
        if not str(target).startswith(str(STATIC_DIR)) or not target.is_file():
            self._send(404, "text/plain; charset=utf-8", b"404 Not Found")
            return
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        self._send(200, ctype, target.read_bytes())


# ======================================================================
# 启动
# ======================================================================
def serve(config: AgentConfig, port: int = 8765, host: str = "127.0.0.1",
          open_browser: bool = True) -> None:
    """启动 Web UI（阻塞直到 Ctrl+C）。"""
    Handler.sessions = SessionManager(config)
    Handler.config = config

    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True

    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}/"
    print(f"[webui] AI Agent Web UI 已启动: {url}")
    print(f"[webui] 审批超时 {int(APPROVAL_TIMEOUT)}s；Ctrl+C 退出")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[webui] 正在关闭…")
    finally:
        httpd.shutdown()
        httpd.server_close()


def _main() -> None:
    """独立运行入口：python webui/server.py --port 8765"""
    parser = argparse.ArgumentParser(description="AI Agent Web UI（零依赖）")
    parser.add_argument("--port", type=int, default=8765, help="监听端口（默认 8765）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--yes", action="store_true", help="跳过工具审批（危险）")
    parser.add_argument("--no-knowledge", action="store_true", help="关闭知识检索")
    args = parser.parse_args()

    serve(
        AgentConfig(approval=not args.yes, knowledge_enabled=not args.no_knowledge),
        port=args.port, host=args.host, open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    os.chdir(_ROOT)   # 保证 .env / knowledge_index.json / mcp_servers.json 落在项目根
    _main()
