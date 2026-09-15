# AGENTS.md — ai-agent-cli 规格体系

> **版本**: v1.4
> **变更日志**:
>
> | 版本 | 日期 | 变更 |
> |---|---|---|
> | v1.4 | 2026-09-15 | E5：Web UI（webui/ 零依赖 http.server + SSE + 审批桥）；plan 补录 E1-E4 并修正 D5 状态 |
> | v1.3 | 2026-09-03 | spec 新增 knowledge_add/knowledge_query/knowledge_ingest（L2 知识管理工具，注册工具 7 个） |
> | v1.2 | 2026-09-03 | plan 更新：D7 完成（week1-notes.md + scaffold.py + harness-template.md）；一周学习全部完成 |
> | v1.1 | 2026-09-03 | plan 更新：D6 完成（config.py 集中装配 + docs/demo.md）；D7 待做 |
> | v1.0 | 2026-09-02 | 初版：constitution / spec / plan / tasks 四级体系建立，覆盖 6 个注册工具 |

> 本文档是本仓库的"行为规格"：任何 AI（或人）在此仓库工作时必须遵循。
> 规格变更必须修改本文件、更新版本号与变更日志、提交 git 并打 tag（`git tag spec-v1.x`）。
> 一致性由 `python scripts/check_spec.py` 校验（工具名 / risk / 参数与代码对齐）。

---

## 1. constitution（宪法）

不可违反的原则，任何任务、任何层级都不得突破：

1. **不猜测**：拿不到的信息必须用工具去查，查不到就如实告知用户，禁止编造文件内容、命令输出或 API 行为。
2. **工具结果必须回喂**：assistant 的 tool_calls 必须有对应 role=tool 消息（`tool_call_id` 一一对应），丢弃即违规。
3. **handler 只返回 str**：工具函数的返回值必须是字符串；dict/list 由 registry 统一转 JSON，handler 内不做序列化假设。
4. **错误不向上抛**：工具层异常一律转成 `[工具执行出错] ...` 字符串回喂模型；LLM 层可重试错误在 llm.py 内消化，其余抛给 CLI 呈现。
5. **危险操作必须审批**：`risk=confirm` 的工具（写文件、执行命令）未经 approver 批准不得执行；审批被拒返回 `[用户拒绝]`，不得绕过（包括通过子 Agent 委派绕过）。

---

## 2. spec（工具规格）

当前注册的全部工具契约。**新增/修改工具必须同步本表**，`scripts/check_spec.py` 会校验。

| 工具名 | risk | 参数 | 行为契约 | 副作用 |
|---|---|---|---|---|
| `read_file` | safe | path | 读取本地文件全文；文件不存在返回 `[文件不存在] <路径>`；utf-8 + errors=replace 解码 | 无 |
| `write_file` | confirm | path, content | 写入/覆盖文件，自动创建父目录；成功返回 `[已写入] <路径>（N 字符）` | 创建/覆盖磁盘文件 |
| `run_shell` | confirm | command, timeout(可选,默认30) | 执行 shell 命令，返回 stdout/stderr/退出码；Windows GBK 解码；超时返回 `[命令超时]` | 执行任意命令 |
| `delegate_task` | safe | prompt | 把自包含子任务交给子 Agent（max_steps=5，独立上下文，看不到主对话）；返回子 Agent 最终结论或 `[委派失败]`；子 Agent 继承审批、无法再委派 | 通过子 Agent 间接产生 |
| `knowledge_add` | safe | file_path, summary, tags(可选) | 登记一条知识入知识库（source=agent，confidence=speculative）；同 (file_path, chunk_index) 覆盖；空 summary 拒绝 | 写 knowledge_index.json |
| `knowledge_query` | safe | query, top_k(可选,默认3) | BM25 检索知识库，返回条目文本（含来源/置信度标记）或 `[知识库无命中]` | 无（只读） |
| `knowledge_ingest` | safe | file_path, chunk_size(可选,默认1500) | 读文档 → 段落聚合/超长硬切分块 → LLM 摘要 → 批量登记（source=document, confidence=pending, chunk_index 递增）；未配 LLM 返回 `[摄取失败]` | 写 knowledge_index.json |

> 注：knowledge_* 三个工具仅当配置了 knowledge_index 时才注册（config.py）。
> 只写固定的知识库 JSON（可 remove 恢复），非破坏性 → risk=safe 免审批。

**spec 约定**：
- 参数列 = JSON Schema `properties` 的键（顺序无关），必填参数以 `*` 标注
- 依赖关系：`write_file` 建议依赖 `read_file`（改前先读，当前由 System Prompt 约定，未强制 depends_on）
- 审批：confirm 级工具执行前必须经 approver 确认（见 constitution 第 5 条）

---

## 3. plan（开发路线）

| 阶段 | 内容 | 状态 |
|---|---|---|
| MVP | ReAct 循环 + 3 基础工具 + REPL | ✅ 完成 |
| 记忆 | 三层记忆（TaskMemory / 增量摘要 / 滑动窗口） | ✅ 完成 |
| D1 | 上下文工程（审计 / tiktoken / 检索注入） | ✅ 完成 |
| D2 | 约束设计（审批 / 重试 / 降级策略） | ✅ 完成 |
| D3 | 工具编排（MCP / 依赖 / 子 Agent 委派） | ✅ 完成 |
| D4 | 规格治理（本文件 + 版本化 + 一致性检查） | ✅ 完成 |
| D5 | 质量闭环（验证点钩子 / 契约校验 / 回归） | ✅ 完成 |
| D6 | 综合实战（config 集中装配 + 端到端演示） | ✅ 完成 |
| D7 | 复盘发布（方法论笔记 + harness 模板） | ✅ 完成 |
| E1 | 流式输出（llm.chat_stream + run on_token） | ✅ 完成 |
| E2 | 会话持久化（session.py 快照保存/恢复） | ✅ 完成 |
| E3 | RAG 四层演进（BM25 / 结构化 / 自动喂 / 生命周期 / 语义） | ✅ 完成 |
| E4 | 配置下沉（CLI 瘦身：.env + mcp_servers.json） | ✅ 完成 |
| E5 | Web UI（零依赖 http.server + SSE + 审批桥） | ✅ 完成 |

---

## 4. tasks（任务模板）

在此仓库新增任务时，按此格式登记（可写入 plan 对应阶段的任务清单）：

```markdown
### <任务编号> <任务名>（必做/可选）
- **目标**：一句话说清做什么
- **改动文件**：新增/修改哪些文件
- **实现要点**：1-3 条关键设计决策
- **验收标准**：
  - [ ] 可执行的验证方式（pytest 用例 / 命令 / 手工步骤）
```

**新工具接入流程**（保证规格与代码一致）：
1. 在 `agent/tools/` 实现 handler（返回 str，异常不外抛）
2. 在 `agent_cli.py` 的 `build_agent()` 注册（确定 risk 级别）
3. 在本文件 §2 spec 表格加一行（工具名/risk/参数/契约/副作用）
4. 运行 `python scripts/check_spec.py` 确认一致
5. 补 pytest 用例；提交时更新版本号与变更日志并打 tag
