# 端到端演示（D6）

> 目标：一条命令让 Agent 完成"新增 `list_dir` 工具 + 补测试 + 自动验证"的完整任务，
> 期间验证 Harness 五方面能力全部生效。演示由用户本地执行（需要有效 API Key）。

---

## 演示场景

**任务**：给 `agent/tools/file_tools.py` 加一个 `list_dir(path)` 工具（返回目录下的文件名列表），
并在 `tests/test_file_tools.py` 补测试，确保 pytest 全部通过。

**单条命令**：

```bash
cd ai-agent-cli && source .venv/bin/activate
python agent_cli.py -v --hooks pytest --yes \
  "给 agent/tools/file_tools.py 加一个 list_dir(path) 工具：返回目录下文件名列表（每行一个）。然后在 tests/test_file_tools.py 补上对应测试。写完务必确保 pytest tests/ 全部通过（会自动校验，失败就修复再测）。"
```

> `--yes` 跳过人工审批（演示自动化流程用）；`--hooks pytest` 是关键——写完代码自动跑测试。
> 想体验审批交互就去掉 `--yes`。

---

## 预期流程与组件对照

| 步骤 | Agent 动作 | 使用的 Harness 组件 | 你看到的输出（-v） |
|---|---|---|---|
| 1 | 理解任务，规划"先读现有文件" | ④ ReAct 循环 + 提示词管理 | `step 1/15` + 上下文审计表格 |
| 2 | `read_file("agent/tools/file_tools.py")` 读现有实现风格 | ⑥ 工具系统（safe 免审批） | `tool read_file(...)` |
| 3 | `read_file("tests/test_file_tools.py")` 看现有测试写法 | ⑥ 工具系统 | `tool read_file(...)` |
| 4 | `write_file(...)` 写入 `list_dir` 实现 | ⑥ 工具系统 + D2 审批（--yes 跳过） | `tool write_file(...)` |
| 5 | 自动触发 `pytest tests/` | **D5 质量闭环钩子** | `hooks [验证通过/失败: run_pytest]` |
| 6 | （若失败）读 pytest 输出 → 修复 → 再写 → 再测 | ④ ReAct 循环 + D5 闭环 | 循环重复 4-5 直到绿 |
| 7 | 汇报结果 | 入口层 Markdown 渲染 | 最终总结 |

**全程静默生效的能力**（不一定打印，但都在工作）：
- D1 上下文工程：每条 user 消息增量 merge 进任务记忆；窗口超 10 轮自动
  增量摘要 + 原文落盘 `memory_store.jsonl`；`-v` 每轮打印分层审计
- D2 约束设计：confirm 工具审批链（`--yes` 跳过）、LLM 重试（网络抖动自动退避）
- D3 工具编排：工具依赖检查（write 前建议 read）、子 Agent 委派可用
- D4 规格治理：新增工具后应跑 `python scripts/check_spec.py` 确认是否需登记 spec

---

## 演示后的验证清单

```bash
# 1. 新工具真的可用了（shell 级验证）
python -c "from agent.tools.file_tools import list_dir; print(list_dir('.'))"

# 2. 质量闭环的成果：测试全绿
pytest tests/ -v

# 3. 规格一致性：若新增工具进了 agent_cli 注册表，应同步 AGENTS.md spec 并复检
python scripts/check_spec.py

# 4. 记忆落盘产物（演示过程滑出的原文）
ls -la memory_store.jsonl
```

---

## 进阶演示（可选项）

### 体验三层记忆（对话中途"记得"前文）
```bash
python agent_cli.py --yes   # REPL 模式
> 读 README.md 并总结前 3 条要点          # 触发 read_file
> 刚才总结的第 2 条是什么？                # 不读文件也能答出（短期窗口记忆）
> 我们项目用了哪些记忆压缩机制？          # 跨轮次（task memory / 摘要）
```

### 体验 MCP 接入（需 Node/npx）
复制 `mcp_servers.json.example` 为 `mcp_servers.json`（存在即自动加载），然后：
```bash
python agent_cli.py -v "用 MCP 的 list_directory 看看当前目录结构"
```

### 体验知识检索注入（默认开启）
```bash
# 登记一条知识（写入默认 knowledge_index.json，Agent 提问时自动带出）
python -c "
from agent.knowledge import KnowledgeStore
KnowledgeStore('knowledge_index.json').add('决策记录.md', '技术栈锁定 Python 3.13 + openai SDK；模型默认 deepseek')
"
python agent_cli.py "我们的技术栈选型依据是什么"
```
> .env 配好 `EMBEDDING_MODEL_PATH`（或 API 源）后即为语义检索（BM25+向量 RRF）。
