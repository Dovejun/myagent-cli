# Harness 模板指南（D7）

> 本仓库就是一张**可复用的 Agent CLI 骨架**：`agent/` 下全是通用能力
> （ReAct 循环 / 三层记忆 / 上下文审计 / 检索注入 / 工具审批 / LLM 重试 /
> MCP 接入 / 子 Agent 委派 / 验证钩子 / 契约校验），没有任何特定业务代码。
> 用 `scripts/scaffold.py` 复制一份，改三处即可跑通一个新 Agent CLI。

---

## 5 分钟从模板起一个新项目

```bash
# 0. 脚手架：复制骨架（自动替换 README/AGENTS.md 里的项目名）
python scripts/scaffold.py my-agent --output ~/projects --model deepseek-v4-flash
cd ~/projects/my-agent

# 1. 环境
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # 填入你的 API Key

# 2. 冒烟测试：模板自带 80+ 离线用例，先确认骨架没坏
pytest tests/ -q

# 3. 跑起来
python agent_cli.py "你好，介绍一下你自己能做什么"
```

---

## 定制三处（骨架 → 你的 Agent）

### ① 模型与身份（2 分钟）
`.env`：填 `OPENAI_API_KEY`；改 `OPENAI_MODEL`（或 scaffold 时用 `--model` 写好）。
`agent/prompts.py`：改 `SYSTEM_PROMPT` 的身份与工具使用规则。

### ② 工具集（你的核心差异）
编辑 `config.py` 的 `_register_builtin_tools()`：
- 增：在 `agent/tools/` 写 handler（返回 str，异常不外抛），
  按 `_TOOL_SCHEMAS` 格式补 schema，`risk=RISK_SAFE/CONFIRM` 分级
- 删：去掉不需要的内置工具
- 接外部能力：写 `mcp_servers.json`（存在即自动加载，多 server 零代码）
  或在代码里 `register_mcp_tools(registry, params)` 逐个接入

### ③ 规格与测试（收尾 1 分钟）
- 新工具登记进 `AGENTS.md` §2 spec 表 → `python scripts/check_spec.py` 确认一致
- `tests/test_*` 是模板示例，可整体改造为你的业务测试
  （写测试用 FakeLLM / ScriptLLM 模式，不依赖真实 API，见各 test 文件头注释）

---

## 模板结构速查

| 目录/文件 | 用途 | 新项目通常怎么改 |
|---|---|---|
| `agent/core.py` | ReAct 循环 + 三层记忆 | 不用改 |
| `agent/llm.py` | OpenAI 兼容封装 + 重试 | 不用改 |
| `agent/memory.py` | 记忆压缩 + TaskMemory | 不用改 |
| `agent/context_audit.py` | 上下文审计 | 不用改 |
| `agent/knowledge.py` | 知识检索：BM25 + 语义融合 + 生命周期 | 不用改 |
| `agent/embeddings.py` | embedder 双通道（API / 本地 HF） | 不用改 |
| `agent/session.py` | 会话快照保存/恢复 | 不用改 |
| `agent/validators.py` | 验证点（pytest/lint） | 可加你的验证器 |
| `agent/tools/` | 工具系统 | **主要改动点** |
| `agent/tools/registry.py` | 注册/分发/风险/依赖/契约 | 不用改 |
| `agent/tools/mcp_tools.py` | MCP 接入 | 不用改 |
| `agent/tools/delegate_tools.py` | 子 Agent 委派 | 不用改 |
| `config.py` | 集中装配 | 改工具注册、开关默认值 |
| `agent_cli.py` | CLI 入口 | 通常不用改 |
| `webui/` | 浏览器界面（零依赖 http.server + SSE） | 通常不用改；改 `static/` 换皮 |
| `AGENTS.md` | 规格体系 | 同步 spec 表 |
| `docs/fallback-policy.md` | 降级策略 | 保留，按需扩展 |
| `tests/` | 离线测试 | 改造为你的业务测试 |

---

## 验收：这就是一个"能用的 Agent CLI"

- [ ] `python agent_cli.py "..."` 能回答（有 API Key）
- [ ] `python agent_cli.py -v "读 README 并总结"` 能看到工具调用 + 审计表格
- [ ] 危险操作有审批：写文件/跑命令前问 y/N（`--yes` 跳过）
- [ ] `python scripts/check_spec.py` 输出 ✅ 规格与代码一致
- [ ] `pytest tests/ -q` 全绿（骨架自带用例）
- [ ] 长对话不爆窗口：三层记忆默认启用（`--no-memory` 可关）
- [ ] `python agent_cli.py --web` 打开浏览器界面，能流式对话、工具卡片可展开、审批可点

---

## 进阶：能力开关速查（config.AgentConfig）

```python
AgentConfig(
    max_steps=15,            # ReAct 保险丝
    approval=True,           # 工具审批（False 危险）
    retry_times=3,           # LLM 重试
    enable_memory=True,      # 三层记忆
    task_memory_budget=1500, # 长期记忆预算
    window_rounds=10,        # 短期窗口
    compress_threshold=60000,# 总量兜底
    archive_path="memory_store.jsonl",
    knowledge_enabled=True,  # 知识检索（默认开；索引路径走 .env/默认文件）
    hooks=(),                # 验证钩子 ("pytest", "lint")
    mcp_config_path="",      # MCP servers JSON（空 = 走 .env MCP_CONFIG/默认文件）
)
```

连接类配置（embedding 源、MCP server 列表、知识库路径）统一在 `.env` 与
`mcp_servers.json`，见 `.env.example` 与 README「配置」小节——CLI 保持最简。
