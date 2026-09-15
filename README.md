# ai-agent-cli

基于 ReAct（Reason + Act + Observe）模式的最小可用 AI Agent，
且是完整可复用的 Harness（三层记忆 / 审批 / MCP / 验证钩子 / 规格治理），
自带终端 CLI 与浏览器 Web UI。约 20 个模块、200+ 离线测试。

## 能力

- OpenAI 兼容 API（OpenAI / DeepSeek / 通义等，改 .env 即可切换）
- 基础工具：`read_file`、`write_file`、`run_shell`（可接 MCP、子 Agent 委派）
- ReAct 多轮推理，带 `max_steps` 保险丝上限
- 三种交互形态：单次提问 / REPL（`--stream` 流式、`--session` 持久化）/ **Web UI（`--web`）**
- Harness 五方面：上下文工程（三层记忆/审计/检索）、约束（审批/重试/降级）、
  工具编排（MCP/依赖/委派）、规格治理（AGENTS.md/check_spec）、质量闭环（验证钩子/契约）
- `-v` verbose 模式打印每步工具调用 + 上下文审计（Rich 彩色输出）
- 跨平台：System Prompt 按 OS 动态提示；Windows 下子进程输出 GBK 解码

## 安装

### 方式 A：安装成全局命令（推荐）

```bash
cd ai-agent-cli
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # 可编辑安装，含全部依赖 + agent-cli 命令

cp .env.example .env               # 填入你的 API Key

agent-cli "你好"                    # 直接敲命令，等价 python agent_cli.py
agent-cli --help                   # 查看全部参数
```

### 方式 B：不安装，直接跑（开发调试用）

```bash
cd ai-agent-cli
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python agent_cli.py "你好"
```

> 两种方式二选一。方式 A 会把 `agent-cli` 放进 venv 的 bin，
> 之后在该 venv 激活状态下任何目录都能直接敲 `agent-cli`。

## 使用

```bash
# 单次提问
python agent_cli.py "读 README.md 并总结"

# 查看工具调用过程
python agent_cli.py -v "当前目录有几个 Python 文件"

# 进入 REPL（Ctrl+C 退出）
python agent_cli.py

# 调整循环上限
python agent_cli.py --max-steps 10 "统计 tests 里的用例数"

# 写文件后自动跑 pytest 校验（质量闭环）
python agent_cli.py --yes --hooks pytest "给 tests 加一个用例并确保通过"

# 关闭三层记忆（朴素 ReAct，省一次 TaskMemory 合并调用）
python agent_cli.py --no-memory "..."

# 流式输出：答案边生成边显示（不渲染 Markdown）
python agent_cli.py --stream "讲一个笑话"
```

> 说明：D6 起默认启用三层记忆——每条新 user 消息会多一次 TaskMemory 增量合并调用
> （每轮 ~1 次额外 LLM 请求），换取"长会话 token 消耗收敛到常数"。
> 单次短问答可用 `--no-memory` 关闭。

## Web UI（浏览器界面）

零依赖的聊天界面（仅用标准库 `http.server` + SSE，不引 FastAPI/Gradio）：

```bash
python agent_cli.py --web                 # 默认 http://localhost:8765，自动打开浏览器
python agent_cli.py --web --port 9000     # 换端口
python agent_cli.py --web --no-browser    # 不自动开浏览器（远程/容器场景）
python webui/server.py --port 8765        # 也可以独立启动
```

界面能力：

- **流式打字**：答案逐字出现（对应 `--stream`），工具轮不打扰
- **工具调用卡片**：每次工具调用展开成卡片（参数 / 输出 / 成功失败），失败自动展开
- **审批弹窗**：`write_file` / `run_shell` 等 confirm 级工具在界面内弹卡片，
  点"批准/拒绝"或按 <kbd>Y</kbd>/<kbd>N</kbd>；超时（180s）按拒绝处理
- **多标签页独立会话**：每个浏览器标签对应一个 Agent 实例（`session_id` 存 localStorage）
- **历史恢复**：刷新页面自动拉回当前会话的对话记录

架构（三分钟看懂 `webui/server.py`）：

```
浏览器 ── GET  /api/stream ──▶ 常驻 SSE，收所有事件（step/tool/token/审批…）
       ── POST /api/chat ───▶ 后台线程跑 Agent.run()，立即返回
       ── POST /api/approve ─▶ 回填审批结果，唤醒阻塞中的 Agent 线程
```

**审批桥**是这里最巧的一处：Agent 的 `approver(name, args) -> bool` 是同步阻塞的，
而审批答案来自另一个 HTTP 连接。做法是 `approver` 里广播事件后在
`threading.Event` 上阻塞等待，前端 POST 回来时 `set()` 唤醒——
同步 Agent 无缝接入异步 Web，核心循环一行没改。

## 测试（无需 API Key，离线可跑）

```bash
pytest tests/ -v
```

## 记忆压缩（可选）

长对话会撑爆上下文窗口，`agent/memory.py` 提供保真压缩：

- **分层保留**：`system` / `user` 消息永不压缩，只压缩 `assistant` / `tool` 过程消息
- **显式提取**：不是自由总结，而是问 LLM "已确认事实 / 决策 / 待办 / 关键数据" 四类必答清单
- **增量摘要**：只处理上次摘要之后的增量，避免"压缩的压缩"造成信息稀释
- **原文双写**：被压缩的完整原文追加写入 JSONL 存档，摘要只是索引

### 控制 token 消耗：三层记忆架构（已集成进 core.py）

"system/user 永不压缩"的正确解读是"信息不丢"，不是"原文全留"。
user 消息随会话无限累积才是 token 爆炸的根源，三层记忆让总消耗收敛到常数：

| 层 | 机制 | 增长特性 |
|---|---|---|
| 长期 | `TaskMemory` 任务记忆块（目标/约束/状态），每轮 merge 增量、重复去重 | **固定预算**（如 1500 字符） |
| 中期 | 摘要槽（`extract_summary` 增量摘要，常驻一条 system 消息） | 缓慢增长，可设上限 |
| 短期 | 滑动窗口，只留最近 N 轮完整消息 | 有界 |

`Agent` 已原生支持，启用方式（`agent_cli.py` 的 `build_agent` 里）：

```python
from agent.memory import MemoryCompressor, TaskMemory

agent = Agent(
    llm, registry, SYSTEM_PROMPT,
    task_memory=TaskMemory(llm, budget=1500),          # 长期：固定预算任务记忆
    compressor=MemoryCompressor(llm, keep_roles=("system",)),  # 中期：增量摘要
    window_rounds=10,                                  # 短期：最近 10 轮
    archive_path="memory_store.jsonl",                 # 原文双写兜底
    compress_threshold=60000,                          # 总量兜底（字符）
)
```

不传这些参数则退化为朴素 ReAct，行为不变。窗口滑出的旧消息会先
`extract_summary` 增量摘要 + 原文落盘，再丢弃——信息不丢，只出上下文。

配套手段：用户粘贴的大段文档用 `write_file` 落盘 + 摘要进上下文，需要时 `read_file` 找回。

## 目录结构

```
ai-agent-cli/
├── AGENTS.md             # 规格体系：constitution → spec → plan → tasks（版本化）
├── agent_cli.py          # 入口：argparse → AgentConfig → 交互（无业务逻辑）
├── config.py             # 集中装配：AgentConfig 全部开关 + build_agent()（D6）
├── webui/                # 浏览器界面：零依赖 http.server + SSE
│   ├── server.py         # 会话管理 / 事件广播 / 审批桥 / 静态服务
│   └── static/           # index.html + app.js + style.css（原生，无框架）
├── agent/
│   ├── core.py           # Agent 类：ReAct 循环 + 三层记忆 + 审计/检索 + 验证钩子
│   ├── llm.py            # OpenAI 兼容 API 封装（含指数退避重试）
│   ├── memory.py         # 记忆：压缩保真 + TaskMemory 固定预算 + 滑动窗口
│   ├── context_audit.py  # 上下文审计：分层统计 + tiktoken 计数 + 报告
│   ├── knowledge.py      # 知识检索：BM25 + 语义(RRF) + 结构化条目 + 生命周期
│   ├── embeddings.py     # embedder 双通道：OpenAI 兼容 API / 本地 HF 模型
│   ├── validators.py     # 质量闭环验证器：run_pytest / run_lint
│   ├── session.py        # 会话持久化：messages 快照保存/恢复
│   ├── prompts.py        # System Prompt（含 OS 动态提示）
│   └── tools/
│       ├── registry.py   # 工具注册表：schema + 分发 + 风险分级 + 依赖 + 契约校验
│       ├── file_tools.py # read_file / write_file
│       ├── shell_tools.py# run_shell（跨平台编码处理）
│       ├── mcp_tools.py  # MCP 接入：外部 server 工具自动注册
│       ├── knowledge_tools.py # 知识管理工具：add/query/ingest（L2 自动喂知识）
│       └── delegate_tools.py # 子 Agent 委派：独立上下文执行子任务
├── docs/
│   ├── demo.md           # 端到端演示：一条命令验证五方面能力（D6）
│   ├── fallback-policy.md# 降级策略清单（A-F 六类场景）
│   ├── harness-engineer-week1.md  # 一周学习计划与进度
│   ├── harness-template.md# 可复用模板指南：5 分钟起新 Agent CLI（D7）
│   └── week1-notes.md    # 复盘方法论：五方面"为什么/不设计会怎样/换方案牺牲什么"（D7）
├── scripts/
│   ├── check_spec.py     # 规格一致性检查：AGENTS.md spec ↔ 代码注册对齐
│   └── scaffold.py       # 模板脚手架：复制骨架生成新项目（D7）
└── tests/
    ├── test_agent.py
    ├── test_audit.py
    ├── test_approval.py
    ├── test_check_spec.py
    ├── test_config.py
    ├── test_contract.py
    ├── test_cross_round.py
    ├── test_delegate.py
    ├── test_dependency.py
    ├── test_embeddings.py
    ├── test_hooks.py
    ├── test_knowledge.py
    ├── test_knowledge_lifecycle.py
    ├── test_knowledge_tools.py
    ├── test_llm_retry.py
    ├── test_llm_stream.py
    ├── test_mcp_tools.py
    ├── test_memory.py
    ├── test_session.py
    ├── test_stream_agent.py
    ├── test_validators.py
    └── test_webui.py
```

## 综合实战（D6）

五方面能力已集中装配进 `config.py`，配置分三层（CLI 参数最简）：

| 层 | 内容 | 说明 |
|---|---|---|
| CLI 参数 | 行为开关 | 见 `agent_cli.py --help`（审批/记忆/流式/会话/钩子） |
| `.env` | LLM、知识库路径、embedding 源、MCP 配置路径 | 默认即"生产可用" |
| 代码 | `AgentConfig` | 测试/嵌入程序可编程覆盖 |

```python
from config import AgentConfig, build_agent

config = AgentConfig(
    max_steps=15, approval=True, retry_times=3,
    enable_memory=True, task_memory_budget=1500, window_rounds=10,
    hooks=("pytest",),            # 写文件后自动跑测试
    knowledge_enabled=True,       # 知识检索默认开
    mcp_config_path="mcp_servers.json",
)
agent = build_agent(config)       # 测试/嵌入程序同此入口
```

端到端演示（含预期流程与组件对照）：见 `docs/demo.md`。

## 配置（.env 与 mcp_servers.json）

开箱即用只需 `.env` 填 LLM 三件套；其余全默认（知识检索默认开启）。

```ini
# .env —— 详见 .env.example
OPENAI_API_KEY=...            # LLM key
OPENAI_BASE_URL=...           # LLM 端点
OPENAI_MODEL=...              # LLM 模型

KNOWLEDGE_INDEX=knowledge_index.json   # 知识库（含向量）路径，默认即此
EMBEDDING_MODEL_PATH=./dir-embed       # 本地 HF embedding（配了才语义检索）
# EMBEDDING_BASE_URL=...     # 或 API 通道（DashScope/Ollama），二选一
# EMBEDDING_MODEL=text-embedding-v3
# MCP_CONFIG=...             # 可选：指向自定义 MCP json
```

MCP servers 用 JSON 配置（**放好即加载，零参数**）——复制
`mcp_servers.json.example` 为 `mcp_servers.json`：

```json
{ "servers": [
  { "name": "filesystem", "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "."] }
] }
```

## 上下文工程（D1）

- **审计**：`agent/context_audit.py` 按层统计上下文（系统提示/任务记忆/增量摘要/窗口区）
  的条数、字符与 token（tiktoken 精确计数，未安装自动降级为字符估算）。
  启用方式：`python agent_cli.py -v "..."`，每轮请求前打印审计表格。
- **知识检索注入（默认开启）**：`agent/knowledge.py` 用 **BM25**（词频饱和 + IDF + 长度归一）
  维护结构化知识库（条目含来源/置信度/分块/热度/向量字段，旧数据自动兼容），
  用户提问时检索相关条目注入本次请求。编程用法：

  ```python
  from agent.knowledge import KnowledgeStore
  store = KnowledgeStore("knowledge_index.json")
  store.add("notes.md", "项目要点：CLI 用 Python，支持审批")
  agent = Agent(..., knowledge=store)  # 问"审批怎么实现"时会自动带出该条目
  ```

  默认注册三个知识管理工具，**Agent 能自己喂知识**：
  `knowledge_add`（登记经验/结论）、`knowledge_ingest`（读文档 → 分块 → LLM 摘要 → 批量入库）、
  `knowledge_query`（任务中途主动检索）。Agent 登记的内容自动标记 `[推测]`、
  文档摄取标记 `[待验证]`，与用户确认的事实区分（可在检索结果中看到）。

  知识生命周期（L4b）：`deduplicate(threshold)` 语义去重（difflib 相似度，保留
  高置信/较新/较完整条目）；`decay(days=90)` 把长期未命中的条目标记为冷条目
  （检索自动降权 + `[冷]` 标记）；检索排序内置热度加权（命中频率 + 时间衰减），
  热门相关条目排前、冷条目沉底——热度只调序，不制造虚假命中。

  语义检索（L3）：**embedding 源在 `.env` 配置即自动启用**（零参数）——两条通道
  同一协议（embedder 只要有 `embed(texts) -> list[list[float]]` 即可接入）：

  ```ini
  # .env —— 本地 HF 模型（默认推荐：离线、数据不出机器；同 localpy/agent.py：
  #          transformers + torch，mean pooling + L2 归一，max_length=512）
  #          依赖：pip install transformers torch
  EMBEDDING_MODEL_PATH=./dir-embed

  # .env —— 或 API 通道（DashScope / Ollama / vLLM，OpenAI 兼容 /embeddings）
  # EMBEDDING_MODEL_PATH 留空时以下两项生效：
  # EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
  # EMBEDDING_MODEL=text-embedding-v3
  ```

  判定规则：**`EMBEDDING_MODEL_PATH` → 本地 HF**；否则
  **`EMBEDDING_BASE_URL` + `EMBEDDING_MODEL` → API**；都没配 → 知识库退回 BM25。
  注：`add()` 时知识连同向量一起落库（knowledge_index.json），旧知识执行
  `store.reindex()` 补齐向量；向量化失败自动降级 BM25，不阻塞。

  注入后检索为 **auto 模式**：BM25 + 语义双路打分、RRF 无参融合（近义表达也命中）；
  `knowledge_query` 工具同样享受语义升级。
  API Key：优先 `OPENAI_EMBEDDING_API_KEY`，否则复用 `OPENAI_API_KEY`。

## 约束设计（D2）

- **工具审批**：工具注册时可声明风险级别——`risk=RISK_SAFE`（只读免审）
  或 `risk=RISK_CONFIRM`（有副作用，执行前人工确认）。默认不传 approver 则全放行：

  ```python
  from agent.tools.registry import RISK_CONFIRM, RISK_SAFE
  registry.register("write_file", ..., write_file, risk=RISK_CONFIRM)
  agent = Agent(..., approver=lambda name, args: input(f"批准 {name}? [y/N]") == "y")
  ```

  CLI 默认开启审批，`--yes` 跳过（危险）。审批被拒返回 `[用户拒绝]` 回喂模型换方案。
- **LLM 重试**：`chat()` 内置指数退避重试（默认 3 次，1s/2s/4s），
  仅重试连接错误/限流(429)/超时/5xx，鉴权等 4xx 直接抛出不浪费重试。
  CLI 用 `--retry N` 调整次数。
- **降级策略**：`docs/fallback-policy.md` 固化六类失败场景的
  "触发条件 + 动作 + 用户可见信息"，核心原则：失败可感知、可恢复、不静默吞掉。

## 工具编排（D3）

- **MCP 接入**：`agent/tools/mcp_tools.py` 用官方 `mcp` SDK 把外部 server 的工具
  自动注册进 registry（schema 自动转换、asyncio.run 桥接同步调用）。
  运行时配置走 JSON（**放好即加载，零参数**），支持多 server：

  ```json
  // mcp_servers.json（默认读取工作目录；或用 .env MCP_CONFIG 指向其他路径）
  { "servers": [
    { "name": "fs", "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."] }
  ] }
  ```

  编程用法（测试/嵌入场景）：
  ```python
  from mcp import StdioServerParameters
  from agent.tools.mcp_tools import register_mcp_tools
  params = StdioServerParameters(command="npx", args=["@modelcontextprotocol/server-filesystem", "."])
  register_mcp_tools(registry, params)  # server 的所有工具进 registry
  ```
  server 不可用时优雅降级（不注册、不崩溃）。

- **工具依赖声明**：`register(..., depends_on=("read_file",))` 规范使用顺序
  （写之前必须读过）；依赖不满足返回 `[前置工具未调用]` 回喂模型补做。
  调用历史按任务隔离（`reset_call_history`），失败调用不记入。

- **子 Agent 委派**：`delegate_task(prompt)` 工具把子任务外包给独立 Agent
  （共享工具能力、独立上下文，只回结论不占主上下文）。
  三个安全设计：过滤委派工具防递归、调用历史备份恢复、审批继承防绕过。
  CLI 已默认注册，子 Agent `max_steps=5` 防失控。

## 规格治理（D4）

- **规格体系**：`AGENTS.md` 四级结构——constitution（5 条不可违反宪法）→
  spec（工具契约表）→ plan（开发路线）→ tasks（任务模板）。
  任何 AI 在本仓库工作前必须读它、遵循它。
- **版本化**：AGENTS.md 头部维护版本号 + 变更日志；规格变更必须
  提交 git 并打 tag（`git tag spec-v1.x`），演化历史可追溯。
- **一致性检查**：`scripts/check_spec.py` 用 ast 解析代码里的工具注册，
  与 spec 表格双向对比（工具名 / risk / 参数），脱节即退出码 1：

  ```bash
  python scripts/check_spec.py    # ✅ 规格与代码一致 / 列出差异
  ```

  新工具接入流程（见 AGENTS.md §4）：实现 handler → 注册 → 登记 spec →
  跑 check_spec → 补测试 → 更新版本号打 tag。

## 质量闭环（D5）

- **验证点钩子**：工具执行成功后自动触发校验，结果附加到 tool 消息回喂——
  通过给一行标记，失败给完整输出，模型可读并修复 → 再验证，循环至绿。
  默认关闭（行为不变），`--hooks pytest,lint` 启用：

  ```bash
  python agent_cli.py --hooks pytest "给 tests 加一个新用例并确保通过"
  ```

  内置验证器（`agent/validators.py`）：`run_pytest`（-x -q 快速失败）、
  `run_lint`（ruff 未安装时优雅跳过）；输出截断 4000 字符防上下文爆炸。
  自定义验证器签名：`fn(tool_result) -> (ok, output)`。
- **契约校验**：`register()` 时用 `inspect.signature` 校验 schema 与 handler
  签名一致——schema 多声明参数 / handler 必填参数未声明，注册即 raise
  （把错误从运行时提前到装配时）。`ToolRegistry(validate_contracts=False)` 可关。
- **回归测试**：`pytest tests/ -v` 覆盖全部 D2-D5 能力（审批/重试/依赖/
  契约/审计/钩子/窗口），全部离线可跑，不依赖真实 API。

## 复盘与模板（D7）

一周学习全部完成（D1-D7，见 `docs/harness-engineer-week1.md`）。

- **方法论笔记**：`docs/week1-notes.md` —— 五方面各一页"为什么这样设计 /
  不设计会怎样 / 换方案牺牲什么"，加贯穿全程的三条元原则。
- **可复用模板**：本仓库即骨架（`agent/` 无业务痕迹），一条命令生成新项目：

  ```bash
  python scripts/scaffold.py my-agent --output ~/projects --model deepseek-v4-flash
  cd ~/projects/my-agent && source .venv/bin/activate
  pip install -r requirements.txt && cp .env.example .env   # 填 Key
  pytest tests/ -q && python agent_cli.py "你好"
  ```

  定制三处即可跑通：模型（.env）/ 工具（config.py）/ 规格（AGENTS.md §2）。
  详见 `docs/harness-template.md`。

## 设计原则

- **单向依赖**：入口层 → 编排层 → 基础设施层，下层不感知上层
- **handler 返回值必须是 str**：模型只能读文本，dict 由 registry 统一转 JSON
- **错误不向上抛**：工具异常转成字符串喂回模型，让模型自己决定下一步
- **tool 消息必须带 tool_call_id**：与 assistant 的 tool_calls id 一一对应

## 下一步可扩展（本文第 7 节留白）

流式输出、MCP 长连接池、会话持久化、沙箱执行
