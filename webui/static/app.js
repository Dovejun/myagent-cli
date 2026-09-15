/* ===================================================================
   AI Agent Web UI · 前端逻辑（零框架，原生 DOM）
   -------------------------------------------------------------------
   与后端的契约（事件类型见 webui/server.py 与 agent/core.py）：
     ready / start / step / tool_start / tool_result / hooks /
     token / approval_required / approval_resolved / done / error / idle / reset
   =================================================================== */

const $ = (sel) => document.querySelector(sel);
const el = {
  stream:   $('#stream'),
  empty:    $('#empty'),
  input:    $('#input'),
  send:     $('#btn-send'),
  reset:    $('#btn-reset'),
  dot:      $('#conn-dot'),
  badge:    $('#status-badge'),
  hint:     $('#hint'),
  sid:      $('#session-id'),
};

// ------------------------------ 状态 ------------------------------
const state = {
  sessionId: localStorage.getItem('agent_session_id') || randId(),
  es: null,
  busy: false,
  historyLoaded: false,
  wrap: null,          // 消息容器
  turn: null,          // 当前助手回合
  bubble: null,        // 当前流式文本气泡
  pendingApproval: null,
  renderQueued: false,
};

localStorage.setItem('agent_session_id', state.sessionId);
el.sid.textContent = state.sessionId;

function randId() {
  if (window.crypto?.randomUUID) return crypto.randomUUID().replace(/-/g, '').slice(0, 12);
  return Math.random().toString(36).slice(2, 14);
}

// ------------------------------ DOM 工具 ------------------------------
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/** 极简 Markdown → HTML：代码块 / 行内代码 / 粗体 / 列表。先转义再替换，防 XSS。 */
function md(text) {
  const blocks = [];
  let t = esc(text);
  // 代码块提取（避免其中的换行被段落逻辑拆散）
  t = t.replace(/```[a-zA-Z0-9_+-]*\n?([\s\S]*?)```/g, (_, code) => {
    blocks.push(`<pre><code>${code.replace(/\n$/, '')}</code></pre>`);
    return `\u0000${blocks.length - 1}\u0000`;
  });
  t = t.replace(/`([^`\n]+)`/g, '<code>$1</code>');
  t = t.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
  t = t.replace(/^#{1,6}\s+(.+)$/gm, '<strong>$1</strong>');
  t = t.replace(/^\s*[-*]\s+(.+)$/gm, '• $1');   // 列表降级成圆点，避免错误嵌套
  t = t.split(/\n{2,}/).map((p) => {
    const s = p.trim();
    if (!s) return '';
    if (/^\u0000\d+\u0000$/.test(s)) return s;   // 纯代码块，不包 <p>
    return `<p>${s.replace(/\n/g, '<br>')}</p>`;
  }).join('');
  return t.replace(/\u0000(\d+)\u0000/g, (_, i) => blocks[+i]);
}

function ensureWrap() {
  if (!state.wrap) {
    state.wrap = document.createElement('div');
    state.wrap.className = 'wrap';
    el.stream.appendChild(state.wrap);
  }
  return state.wrap;
}

function hideEmpty() {
  if (el.empty) { el.empty.remove(); el.empty = null; }
}

/** 仅当用户已在底部附近时才自动滚动，避免打断向上翻阅。 */
function nearBottom() {
  return el.stream.scrollHeight - el.stream.scrollTop - el.stream.clientHeight < 120;
}
let stickBottom = true;
el.stream.addEventListener('scroll', () => { stickBottom = nearBottom(); });
function scrollDown(force = false) {
  if (force || stickBottom) el.stream.scrollTop = el.stream.scrollHeight;
}

function setStatus(text, dotClass) {
  el.badge.textContent = text;
  if (dotClass) el.dot.className = `dot ${dotClass}`;
}

function setHint(text) { el.hint.textContent = text || ''; }

// ------------------------------ 消息渲染 ------------------------------
function addUserMessage(text) {
  hideEmpty();
  const d = document.createElement('div');
  d.className = 'msg user';
  const b = document.createElement('div');
  b.className = 'bubble';
  b.textContent = text;
  d.appendChild(b);
  ensureWrap().appendChild(d);
  scrollDown(true);
}

function startTurn() {
  hideEmpty();
  const d = document.createElement('div');
  d.className = 'msg agent';
  const t = document.createElement('div');
  t.className = 'turn';
  d.appendChild(t);
  ensureWrap().appendChild(d);
  state.turn = t;
  state.bubble = null;
  scrollDown(true);
}

/** 取当前文本气泡；工具卡片之后需要新建一个（保证顺序正确）。 */
function currentBubble() {
  if (!state.bubble) {
    const b = document.createElement('div');
    b.className = 'bubble';
    const caret = document.createElement('span');
    caret.className = 'caret';
    b.appendChild(caret);
    state.turn.appendChild(b);
    state.bubble = b;
  }
  return state.bubble;
}

function appendToken(text) {
  if (!state.turn) startTurn();
  const b = currentBubble();
  b.dataset.raw = (b.dataset.raw || '') + text;
  scheduleRender(b);
}

function scheduleRender(node) {
  if (node._queued) return;
  node._queued = true;
  requestAnimationFrame(() => {
    node._queued = false;
    const caret = node.querySelector('.caret');
    node.innerHTML = md(node.dataset.raw || '');
    if (caret && !node.dataset.done) node.appendChild(caret);
    scrollDown();
  });
}

function finishTurn(content) {
  if (!state.turn) return;
  if (typeof content === 'string' && content) {
    const b = currentBubble();
    b.dataset.raw = content;
    b.dataset.done = '1';
    b.innerHTML = md(content);
  } else {
    state.turn.querySelectorAll('.caret').forEach((c) => c.remove());
  }
  state.turn.querySelectorAll('.caret').forEach((c) => c.remove());
  state.turn.querySelectorAll('.thinking').forEach((c) => c.remove());
}

function addError(message) {
  hideEmpty();
  const d = document.createElement('div');
  d.className = 'error-box';
  d.textContent = `⚠ ${message}`;
  (state.turn || ensureWrap()).appendChild(d);
  scrollDown(true);
}

// ------------------------------ 工具卡片 ------------------------------
function addToolCard(name, args) {
  const tpl = $('#tpl-tool').content.cloneNode(true);
  const card = tpl.querySelector('.tool');
  card.querySelector('.tool-name').textContent = name;
  card.querySelector('.tool-args').textContent = summarizeArgs(args);
  card.querySelector('.tool-state').textContent = '运行中';
  card.querySelector('.tool-state').className = 'tool-state run';
  card.querySelector('.tool-out').textContent = '执行中…';
  card.querySelector('.tool-head').addEventListener('click', () => card.classList.toggle('open'));
  if (!state.turn) startTurn();
  state.turn.appendChild(card);
  state.bubble = null;               // 后续 token 要开新气泡
  state.turn.querySelectorAll('.caret').forEach((c) => c.remove());
  scrollDown();
  return card;
}

function updateToolCard(card, output, denied, error) {
  if (!card) return;
  const st = card.querySelector('.tool-state');
  const ok = !denied && !error;
  st.textContent = denied ? '已拒绝' : error ? '出错' : '完成';
  st.className = `tool-state ${ok ? 'ok' : 'error'}`;
  card.querySelector('.tool-out').textContent = output || '（无输出）';
  if (!ok) card.classList.add('open');   // 失败自动展开，成功保持折叠
}

function summarizeArgs(args) {
  if (!args || typeof args !== 'object') return '';
  const parts = Object.entries(args).map(([k, v]) => {
    let s = typeof v === 'string' ? v : JSON.stringify(v);
    if (s.length > 42) s = s.slice(0, 42) + '…';
    return `${k}=${s}`;
  });
  const joined = parts.join(', ');
  return joined.length > 90 ? joined.slice(0, 90) + '…' : joined;
}

// ------------------------------ 审批 ------------------------------
function addApproval(id, name, args) {
  const tpl = $('#tpl-approval').content.cloneNode(true);
  const card = tpl.querySelector('.approval');
  card.dataset.id = id;
  card.querySelector('.approval-tool').textContent = name;
  card.querySelector('.approval-args').textContent = JSON.stringify(args, null, 2);
  card.querySelector('.btn.approve').addEventListener('click', () => resolveApproval(id, true));
  card.querySelector('.btn.reject').addEventListener('click', () => resolveApproval(id, false));
  if (!state.turn) startTurn();
  state.turn.appendChild(card);
  state.pendingApproval = { id, card };
  setHint('等待你的批准：Y 批准 / N 拒绝');
  scrollDown(true);
}

async function resolveApproval(id, approved) {
  const card = document.querySelector(`.approval[data-id="${id}"]`);
  if (card) {
    card.querySelectorAll('button').forEach((b) => { b.disabled = true; });
    const r = card.querySelector('.approval-result');
    r.textContent = approved ? '已批准，继续执行…' : '已拒绝，Agent 会据此调整方案';
    r.className = `approval-result ${approved ? 'ok' : 'err'}`;
  }
  state.pendingApproval = null;
  setHint('');
  try {
    await fetch('/api/approve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId, approval_id: id, approved }),
    });
  } catch (e) {
    addError(`审批提交失败：${e}`);
  }
}

// ------------------------------ 事件分发 ------------------------------
function handleEvent(ev) {
  switch (ev.type) {
    case 'ready':
      setStatus('就绪', 'on');
      if (!state.historyLoaded) { state.historyLoaded = true; loadHistory(); }
      break;

    case 'start':
      state.busy = true;
      updateComposer();
      setStatus('思考中', 'busy');
      startTurn();
      addThinking();
      break;

    case 'step':
      setHint(`第 ${ev.step}/${ev.max_steps} 步`);
      removeThinking();
      break;

    case 'tool_start':
      removeThinking();
      state.toolCards = state.toolCards || [];
      state.toolCards.push(addToolCard(ev.name, ev.args));
      setStatus(`调用 ${ev.name}`, 'busy');
      setHint(`工具：${ev.name}`);
      break;

    case 'tool_result': {
      const cards = state.toolCards || [];
      updateToolCard(cards.pop(), ev.output, ev.denied, ev.error);
      setStatus('思考中', 'busy');
      break;
    }

    case 'hooks':
      if (state.turn) {
        const d = document.createElement('div');
        d.className = 'tool';
        d.innerHTML = `<div class="tool-head"><span class="tool-icon">✓</span>
          <span class="tool-name">验证钩子</span><span class="tool-args"></span></div>
          <div class="tool-body" style="display:block"><pre class="tool-out"></pre></div>`;
        d.querySelector('.tool-args').textContent = ev.name;
        d.querySelector('.tool-out').textContent = ev.output;
        state.turn.appendChild(d);
      }
      break;

    case 'token':
      removeThinking();
      appendToken(ev.text);
      break;

    case 'approval_required':
      addApproval(ev.id, ev.name, ev.args);
      break;

    case 'approval_resolved': {
      const card = document.querySelector(`.approval[data-id="${ev.id}"]`);
      if (card) {
        card.classList.add('resolved');
        if (ev.timeout) {
          const r = card.querySelector('.approval-result');
          r.textContent = '审批超时，已按拒绝处理';
          r.className = 'approval-result err';
        }
      }
      if (state.pendingApproval?.id === ev.id) { state.pendingApproval = null; setHint(''); }
      break;
    }

    case 'done':
      finishTurn(ev.content);
      if (ev.stopped) addError('达到最大步数，任务未完成');
      break;

    case 'error':
      removeThinking();
      addError(ev.message);
      break;

    case 'idle':
      state.busy = false;
      updateComposer();
      setStatus('就绪', 'on');
      setHint('');
      state.turn = null;
      state.bubble = null;
      state.toolCards = [];
      break;

    case 'reset':
      resetView();
      break;
  }
}

function addThinking() {
  if (!state.turn || state.turn.querySelector('.thinking')) return;
  const d = document.createElement('div');
  d.className = 'thinking';
  d.innerHTML = '<span class="d"></span><span class="d"></span><span class="d"></span><span>思考中</span>';
  state.turn.appendChild(d);
  scrollDown();
}
function removeThinking() {
  state.turn?.querySelectorAll('.thinking').forEach((n) => n.remove());
}

// ------------------------------ 会话 ------------------------------
async function loadHistory() {
  try {
    const r = await fetch(`/api/history?session_id=${encodeURIComponent(state.sessionId)}`);
    const j = await r.json();
    const msgs = j.messages || [];
    if (!msgs.length) return;
    hideEmpty();
    msgs.forEach((m) => {
      if (m.role === 'user') addUserMessage(m.content);
      else {
        startTurn();
        appendToken(m.content);
        finishTurn(m.content);
      }
    });
    scrollDown(true);
  } catch { /* 历史拉取失败不影响使用 */ }
}

function resetView() {
  if (state.wrap) state.wrap.remove();
  state.wrap = null; state.turn = null; state.bubble = null;
  state.toolCards = []; state.pendingApproval = null;
  if (!el.empty) location.reload();   // 空状态由页面重新渲染
}

async function doReset() {
  try {
    await fetch('/api/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId }),
    });
  } catch (e) { addError(`重置失败：${e}`); }
}

// ------------------------------ 发送 ------------------------------
function updateComposer() {
  el.send.disabled = state.busy;
  el.input.disabled = state.busy;
  el.input.placeholder = state.busy
    ? 'Agent 正在执行…'
    : '说点什么…（Enter 发送，Shift+Enter 换行）';
}

async function send(text) {
  const msg = text.trim();
  if (!msg || state.busy) return;
  addUserMessage(msg);
  el.input.value = '';
  autoGrow();
  state.busy = true;
  updateComposer();
  stickBottom = true;
  scrollDown(true);
  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId, message: msg }),
    });
    const j = await r.json().catch(() => ({}));
    if (!j.ok) {
      addError(j.error || `请求失败（HTTP ${r.status}）`);
      state.busy = false;
      updateComposer();
    }
  } catch (e) {
    addError(`无法连接到服务端：${e}`);
    state.busy = false;
    updateComposer();
  }
}

// ------------------------------ SSE 连接 ------------------------------
function connect() {
  if (state.es) state.es.close();
  const es = new EventSource(`/api/stream?session_id=${encodeURIComponent(state.sessionId)}`);
  state.es = es;
  es.onopen = () => setStatus(state.busy ? '思考中' : '就绪', state.busy ? 'busy' : 'on');
  es.onmessage = (e) => {
    let ev;
    try { ev = JSON.parse(e.data); } catch { return; }
    handleEvent(ev);
  };
  es.onerror = () => setStatus('连接断开，重连中…', 'off');
}

// ------------------------------ 输入框 ------------------------------
function autoGrow() {
  el.input.style.height = 'auto';
  el.input.style.height = Math.min(el.input.scrollHeight, 190) + 'px';
}

el.input.addEventListener('input', autoGrow);
el.input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    send(el.input.value);
  }
});
el.send.addEventListener('click', () => send(el.input.value));
el.reset.addEventListener('click', doReset);

document.querySelectorAll('.chip').forEach((c) =>
  c.addEventListener('click', () => send(c.dataset.prompt)));

// 审批快捷键：输入框聚焦时不拦截（避免打字冲突）
document.addEventListener('keydown', (e) => {
  if (!state.pendingApproval) return;
  if (document.activeElement === el.input) return;
  const k = e.key.toLowerCase();
  if (k === 'y') { e.preventDefault(); resolveApproval(state.pendingApproval.id, true); }
  if (k === 'n') { e.preventDefault(); resolveApproval(state.pendingApproval.id, false); }
});

// 页面隐藏时断开 SSE（省资源），回来重连
document.addEventListener('visibilitychange', () => {
  if (document.hidden) state.es?.close();
  else connect();
});

autoGrow();
connect();
