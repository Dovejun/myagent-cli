# Harness Engineer 一周学习 · Day 1-7 实现清单

> 载体项目：`ai-agent-cli`（本仓库）
> 目标：把 Harness 五方面能力（上下文工程 / 约束设计 / 工具编排 / 规格治理 / 质量闭环）逐一落地为**代码 + 测试 + 文档**，Day 6 整合为生产可用版本。
>
> 使用方式：每个任务 = 目标 / 改动文件 / 实现要点 / 验收标准。按天执行，完成一项打勾。

---

## Day 1 · 上下文工程

> 已具备：三层记忆（TaskMemory / 摘要槽 / 滑动窗口，见 `agent/memory.py`、`agent/core.py`）。
> 今天做：让记忆"可观测、可量化、可检索"。

### 1.1 上下文审计工具（必做）✅ 已完成
- **目标**：每次请求前能回答"上下文里有什么、各层占比多少、还能撑多久"
- **改动文件**：新增 `agent/context_audit.py`
- **实现要点**：
  1. `audit(messages) -> dict`：按层统计（system / 任务记忆 / 增量摘要 / 窗口区）的条数与字符数
  2. 调用 `Agent._build_request_messages()` 后输出审计报告（verbose 模式下 Rich 表格）
  3. 估算 token：`chars // 2`（中文约 1 字 ≈ 0.6-1 token，粗略够用）
- **验收标准**：
  - [x] `python agent_cli.py -v "读 README 并总结"` 能看到分层审计输出（core.py 每轮打印）
  - [x] `tests/test_audit.py`：构造已知消息，断言各层统计数字正确（8 个用例）

### 1.2 tiktoken 精确计数（可选，建议做）✅ 已完成
- **目标**：用官方分词器精确数 token，替代字符估算
- **改动文件**：`requirements.txt` 加 `tiktoken`；`agent/context_audit.py` 增加 `count_tokens(model)` 分支
- **验收标准**：
  - [x] 审计报告显示 token 数与窗口上限的剩余比例（tiktoken 优先，未安装自动降级字符估算）
  - [~] 对同一段中文，tiktoken 与字符估算差异 ≤ 20%（需用户本地验证）

### 1.3 信息检索注入（进阶，RAG 雏形）✅ 已完成
- **目标**：大段历史不常驻上下文，需要时按关键词检索回来
- **改动文件**：新增 `agent/knowledge.py`（简单实现，不引向量库）
- **实现要点**：文档按"文件名 → 首段摘要"建索引存 JSON；`retrieve(keyword)` 返回匹配片段，注入 `_build_request_messages()` 末尾
- **验收标准**：
  - [~] 用 `write_file` 存一份笔记，问 Agent"笔记里提到的 X 是什么"，能检索并答出（需本地联调验证）
  - [x] 索引文件可被 `knowledge.py` 独立增删查（add/remove/retrieve，7 个用例）

---

## Day 2 · 约束设计

> 已具备：max_steps 保险丝、工具异常转字符串回喂。
> 今天做：行为边界 + 人工审批 + 重试降级。

### 2.1 工具审批层（必做）✅ 已完成
- **目标**：危险操作（写文件、跑命令）需人工确认，只读操作免审
- **改动文件**：`agent/tools/registry.py`（register 加 `risk` 参数）、`agent_cli.py`（交互确认）
- **实现要点**：
  1. `register(name, ..., risk="safe" | "confirm")`；`execute()` 对 `confirm` 级先回调 `approver(result_args) -> bool`
  2. `Agent` 加 `approver` 参数（默认 None = 全放行，保持向后兼容）；CLI 传入 `lambda: Prompt.ask("允许执行? [y/N]") == "y"`
  3. CLI 加 `--yes` 参数跳过审批
- **验收标准**：
  - [x] `write_file` / `run_shell` 触发确认，`read_file` 不触发（risk 分级 + approver 回调）
  - [x] 拒绝时返回 `[用户拒绝]`，模型能感知并换方案（不是崩溃）
  - [x] `pytest tests/test_approval.py`：safe 直通 / confirm 通过 / confirm 拒绝 / --yes 跳过（approver=None 直通等价）

### 2.2 LLM 调用重试（必做）✅ 已完成
- **目标**：网络抖动、限流自动重试，指数退避
- **改动文件**：`agent/llm.py`
- **实现要点**：`chat()` 包 3 次重试（间隔 1s/2s/4s），仅对 `APIConnectionError / RateLimitError / Timeout` 重试，`AuthenticationError` 不重试
- **验收标准**：
  - [x] `tests/test_llm_retry.py`：mock 前 2 次抛错、第 3 次成功 → 返回正常
  - [x] 连续失败 3 次后抛出、且外层有可读错误（CLI 层 catch 显示红色错误）

### 2.3 降级策略清单（必做）✅ 已完成
- **目标**：把"出问题怎么办"固化成文档，不再是临场反应
- **改动文件**：新增 `docs/fallback-policy.md`
- **实现要点**：列四类场景及处理——API 连续失败（告知用户并停止，不无限重试）/ 工具连续返回错误（换方案提示）/ max_steps 耗尽（输出已完成的中间结论）/ 用户拒绝审批（记录意图，停止该分支）
- **验收标准**：
  - [x] 文档每类场景都有"触发条件 + 动作 + 用户可见信息"（A-F 六类 + 决策速查表）
  - [x] core.py 里对应分支注释指向该文档

---

## Day 3 · 工具编排

> 已具备：注册表 schema + 分发、3 个基础工具。
> 今天做：外部工具接入（MCP）+ 工具间依赖 + 子任务委派。

### 3.1 MCP 接入（必做）✅ 已完成
- **目标**：让 Agent 能用任意 MCP server 的工具（如 filesystem、git）
- **改动文件**：新增 `agent/tools/mcp_tools.py`；`requirements.txt` 加 `mcp`、`httpx`
- **实现要点**：
  1. 用官方 `mcp` Python SDK（v2）以 stdio 方式连接本地 MCP server（如 `npx @modelcontextprotocol/server-filesystem .`）
  2. 拉取 server 的 tools 列表，把每个 tool 的 input_schema 转成 registry 格式自动注册
  3. 分发时经 `client.call_tool(name, args)` 执行（asyncio.run 桥接同步 registry）
- **验收标准**：
  - [~] 接入 filesystem server 后，Agent 能 `list_directory` / `read_file`（MCP 版）（需本地联调）
  - [x] 断开 MCP 时注册表优雅降级（不注册该组工具，不崩溃）
  - [x] `tests/test_mcp_tools.py`：用一个 fake server 验证 schema 转换与调用（9 用例）

### 3.2 工具依赖声明（必做）✅ 已完成
- **目标**：规范工具使用顺序（如"写之前必须读过"）
- **改动文件**：`agent/tools/registry.py`（depends_on + _call_history + reset_call_history）、`agent/core.py`（run() 开头 reset）
- **实现要点**：`register(..., depends_on=("read_file",))`；`execute()` 前检查依赖是否在本任务中调用过（记录调用历史），未满足返回提示；失败调用不记入历史
- **验收标准**：
  - [x] 未 read 直接 write → 返回 `[前置工具未调用] read_file`，模型感知后补读
  - [x] `tests/test_dependency.py` 覆盖依赖满足 / 不满足 / 任务隔离 / 失败不算调用 / 多依赖（6 用例）

### 3.3 子 Agent 委派（进阶）✅ 已完成
- **目标**：把子任务外包给独立 Agent（并发/隔离）
- **改动文件**：新增 `agent/tools/delegate_tools.py`；`agent_cli.py` 注册
- **实现要点**：注册 `delegate_task(prompt)` 工具：内部 new 一个 `Agent`（共享 registry、独立 messages）执行子任务并返回结果字符串；`max_steps=5` 小值防失控
- **三个安全设计**：① `_SubRegistryView` 过滤委派工具防递归委派 ② 调用历史备份/恢复，子任务不影响主任务依赖状态 ③ 子 Agent 继承 approver，confirm 工具不被绕过审批
- **验收标准**：
  - [x] 主 Agent 委派子任务，子 Agent 完成并返回结果（ScriptLLM 脚本验证）
  - [x] 子 Agent 的中间过程不污染主 Agent 上下文（只回结果，主上下文恒 5 条消息）

---

## Day 4 · 规格治理

> 已具备：prompts.py 的 SYSTEM_PROMPT（行为准则雏形）。
> 今天做：constitution → spec → plan → tasks 完整规格体系。

### 4.1 编写 AGENTS.md 规格体系（必做）✅ 已完成
- **目标**：把"Agent 该如何在本仓库工作"写成可被另一 AI 读取遵循的文档
- **改动文件**：项目根新增 `AGENTS.md`
- **实现要点**（四级结构）：
  1. **constitution**（宪法）：5 条不可违反原则（不猜测 / tool_call_id 配对 / handler 只返回 str / 错误不向上抛 / 危险操作必须审批）
  2. **spec**（规格）：4 个工具的行为契约表（工具名/risk/参数/行为契约/副作用）
  3. **plan**（计划）：MVP → 记忆 → D1~D7 开发路线与状态
  4. **tasks**（任务模板）：新任务标准格式 + 新工具接入五步流程
- **验收标准**：
  - [~] 把 AGENTS.md 交给另一个 LLM 会话，让它按 spec 实现一个新工具（如 `list_dir`），产物符合预期（需双 LLM 会话人工验证）
  - [x] 文档头部含版本号（v1.0）与变更日志

### 4.2 规格版本化（必做）✅ 已完成
- **目标**：规格演化可追溯
- **实现要点**：AGENTS.md 头部维护 `版本: v1.x` + `变更日志`表；规格修改必须提交 git 并打 tag（`git tag spec-v1.2`）
- **验收标准**：
  - [~] `git log --oneline AGENTS.md` 能看出规格演化历史（需用户首次提交 + 打 tag 后生效）

### 4.3 规格一致性检查（进阶）✅ 已完成
- **目标**：代码里的工具注册与 AGENTS.md 的 spec 脱节时报警
- **改动文件**：新增 `scripts/check_spec.py`、`tests/test_check_spec.py`
- **实现要点**：解析 AGENTS.md §2 spec 表格（工具名/risk/参数）→ ast 解析代码里 `registry.register(...)` 调用（动态注册用 `tool_name: str = "..."` 正则兜底，risk/参数标记未知跳过对比）→ 双向对比四类差异
- **验收标准**：
  - [x] 故意新增一个未文档化工具，脚本退出码非 0 并列出差异（`[代码有但文档未登记]`）

---

## Day 5 · 质量闭环

> 已具备：pytest 测试套件（registry / memory 离线用例）。
> 今天做：把验证从"最后测一次"变成"每一步都验证"。

### 5.1 验证点钩子（必做）
- **目标**：工具执行后可自动触发校验（如改完代码自动跑测试）
- **改动文件**：`agent/core.py`（`Agent` 加 `validation_hooks: dict[str, list[callable]]`）、新增 `agent/validators.py`
- **实现要点**：
  1. `validators.py` 提供 `run_pytest(path)` / `run_lint(path)` 等，返回 `(ok, output)`
  2. 循环里执行完指定工具后调用对应 hooks；失败时把校验输出作为 tool 结果回喂，让模型修复
- **验收标准**：
  - [x] Agent 修改文件后自动跑 `pytest tests/`，失败 → 模型读输出并修复 → 再测，循环至绿（hooks 结果附加到 tool 消息）
  - [x] hooks 可配置（`--hooks pytest,lint`），默认关闭不改变现有行为（仅工具成功时触发，被拒/出错跳过）

### 5.2 工具契约校验（必做）✅ 已完成
- **目标**：schema 与 handler 签名不一致在注册时就暴露
- **改动文件**：`agent/tools/registry.py`（`validate_contract` + `ToolRegistry(validate_contracts=True)`）
- **实现要点**：`register()` 里用 `inspect.signature` 对比 `parameters.properties` 的键与函数参数，不一致则 raise/警告
- **验收标准**：
  - [x] 注册 `lambda path: ...` 但 schema 写 `{"path": ..., "extra": ...}` → 注册报错
  - [x] `tests/test_contract.py` 覆盖匹配 / 缺参 / 多参三种情况（+ 默认值参数可选 / **kwargs 宽松 / 可关闭 / 既有工具回归，共 8 用例）

### 5.3 回归测试扩展（必做）✅ 已完成
- **目标**：把 Day 2-5 的新能力全部纳入离线测试
- **改动文件**：`tests/` 新增 test_hooks.py（7 用例）、test_validators.py（7 用例）、test_contract.py（8 用例）
- **验收标准**：
  - [~] `pytest tests/ -v` 全绿，覆盖：审批、重试、依赖、契约、审计、窗口边界（新增用例已就位，全量运行由用户本地验证）

---

## Day 6 · 综合实战 ✅ 已完成

- **目标**：五方面能力整合为一个可演示的"生产可用" Agent
- **改动文件**：新增 `config.py`（AgentConfig 集中装配 + build_agent）；`agent_cli.py` 重构为纯入口（args → config → build）
- **实现要点**：
  1. 把全部开关收敛到一个配置对象（`AgentConfig` dataclass：approval/retry/hooks/enable_memory/task_memory_budget/window_rounds/compress_threshold/archive_path/knowledge_index/mcp_command），CLI 参数驱动（`from_args`）
  2. 端到端演示场景：Agent 接受"给项目加一个 `list_dir` 工具并补测试"→ 自动 read 现有工具 → write 新工具 → 跑 pytest 验证 → 汇报
- **验收标准**：
  - [~] 一条命令跑通上述演示场景，期间审批、验证钩子、三层记忆全部生效（`docs/demo.md` 命令由用户本地执行）
  - [x] 写 `docs/demo.md`：演示步骤 + 预期输出 + 每步对应哪个 harness 组件（含 3 个进阶演示：三层记忆/MCP/知识检索）

> D6 后默认行为变化：三层记忆默认启用（此前未启用）——每条 user 消息多一次 TaskMemory.merge 调用。
> CLI 新增 `--no-memory` 参数（后续简化：知识检索默认开启、MCP 改 JSON 配置、
> embedding 源下沉 .env，见 README「配置」）。

---

## Day 7 · 复盘发布 ✅ 已完成（一周全部完成 🎉）

- **产出清单**：
  1. `docs/week1-notes.md`：五方面各 1 页方法论（为什么这样设计 / 不设计会怎样 / 换方案牺牲什么）
  2. 把 ai-agent-cli 整理为可复用 **harness 模板**：保留骨架（core/llm/registry/memory/validators），清空业务痕迹
  3. 5 分钟演示脚本：从空仓库用模板起一个新 Agent CLI 的流程
- **验收标准**：
  - [~] 新仓库克隆模板后，改 3 处（名字/模型/工具）即可跑通（`python scripts/scaffold.py <name>` 由用户本地执行验证）
  - [x] 能对任何人讲清：六大核心组件各自解决什么问题，五方面能力如何嵌入（week1-notes.md §0-5）

---

## 附录：验收总览表

| 天 | 必做任务 | 产出文件 | 验收命令 |
|---|---|---|---|
| D1 ✅ | 1.1 审计 / 1.2 tiktoken / 1.3 检索 | `context_audit.py` + `knowledge.py` | `pytest tests/test_audit.py tests/test_knowledge.py` |
| D2 ✅ | 2.1 审批 / 2.2 重试 / 2.3 清单 | `fallback-policy.md` | `pytest tests/test_approval.py tests/test_llm_retry.py` |
| D3 ✅ | 3.1 MCP / 3.2 依赖 / 3.3 委派 | `mcp_tools.py` + `delegate_tools.py` | `pytest tests/test_mcp_tools.py tests/test_dependency.py tests/test_delegate.py` |
| D4 ✅ | 4.1 AGENTS.md / 4.2 版本化 / 4.3 检查 | `AGENTS.md` + `scripts/check_spec.py` | `python scripts/check_spec.py` + `pytest tests/test_check_spec.py` |
| D5 ✅ | 5.1 钩子 / 5.2 契约 / 5.3 回归 | `validators.py` | `pytest tests/ -v` 全绿 |
| D6 ✅ | 整合演示 | `config.py` + `docs/demo.md` | 端到端跑通演示场景（用户本地） |
| D7 ✅ | 模板 + 笔记 | `docs/week1-notes.md` + `scripts/scaffold.py` + `docs/harness-template.md` | `python scripts/scaffold.py <name>`（用户本地） |
