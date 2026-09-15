"""工具注册表。

每个工具由三部分组成：
1. schema —— 符合 OpenAI Function Calling 格式的函数描述，传给 LLM
2. handler —— 真正执行的 Python 函数
3. name  —— 两者之间的关联键

核心约定：handler 返回值必须是 str。
模型只能读文本；返回 dict 时由本注册表统一 json.dumps 转成字符串。

错误处理也在这里：任何异常都转成字符串喂回模型，
让模型自己决定下一步（重试、换参数、或者如实告诉用户失败）。

约束设计（D2-1）：每个工具可声明 risk 级别。
- risk="safe"    只读/无副作用操作，直接执行（如 read_file）
- risk="confirm" 有副作用/危险操作，执行前需人工审批（如 write_file、run_shell）
审批通过 approver 回调实现：Agent 注入 `approver(name, args) -> bool`，
返回 False 时返回 "[用户拒绝]" 字符串，模型能感知并换方案（不崩溃）。

质量闭环（D5-2）：注册时校验 schema 与 handler 签名一致（契约校验）。
- schema properties 声明的参数，handler 必须能接受（否则模型传参会炸）
- handler 必填参数（无默认值），schema 必须声明（否则模型不知道要传）
- handler 有 **kwargs 或无法内省时跳过校验（宽松）
不一致在注册时就 raise —— 把错误从"运行时才发现"提前到"装配时暴露"。
"""

import inspect
import json
from typing import Callable

# 工具风险级别
RISK_SAFE = "safe"        # 只读/无副作用，直接执行
RISK_CONFIRM = "confirm"  # 有副作用，需人工审批

# approver 回调签名：approver(tool_name: str, args: dict) -> bool
Approver = Callable[[str, dict], bool] | None


def validate_contract(name: str, parameters: dict, handler: Callable) -> None:
    """校验 schema 参数与 handler 签名一致（D5-2 契约校验）。

    规则：
    1. schema properties 的键 ⊆ handler 可接受的参数（有 **kwargs 视为全接受）
    2. handler 必填参数（无默认值）⊆ schema properties
    不一致抛 ValueError，在注册时暴露（而非运行时模型调用才炸）。
    无法内省（内置函数等）或 handler 有 **kwargs 时宽松跳过。
    """
    try:
        sig = inspect.signature(handler)
    except (TypeError, ValueError):
        return  # 无法内省，宽松跳过

    params = sig.parameters.values()
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    named = {
        n for n, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    required = {
        n for n, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and p.default is inspect.Parameter.empty
    }

    props: set[str] = set()
    if isinstance(parameters, dict):
        props = set((parameters.get("properties") or {}).keys())

    # 规则 1：schema 声明了 handler 不接受的参数（多参）
    if not has_var_kw:
        unknown = props - named
        if unknown:
            raise ValueError(
                f"工具 '{name}' 契约不符: schema 声明了 handler 不接受的参数 {sorted(unknown)}"
            )
    # 规则 2：handler 必填参数未在 schema 声明（缺参）
    missing = required - props
    if missing:
        raise ValueError(
            f"工具 '{name}' 契约不符: handler 必填参数未在 schema 声明 {sorted(missing)}"
        )


class ToolRegistry:
    def __init__(self, validate_contracts: bool = True) -> None:
        # D5-2：注册时是否做契约校验（默认开启，fail fast）
        self.validate_contracts = validate_contracts

        self._handlers: dict[str, Callable] = {}
        self._schemas: list[dict] = []
        self._risks: dict[str, str] = {}
        self._depends_on: dict[str, tuple[str, ...]] = {}
        # D3-2：本任务内已成功调用过的工具名集合（用于依赖检查）
        # 每次 Agent.run() 开始时由 core.py 调用 reset_call_history() 清空
        self._call_history: set[str] = set()

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        handler: Callable,
        risk: str = RISK_SAFE,
        depends_on: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        """注册一个工具：存 handler + 生成 OpenAI 格式 schema。

        risk 决定是否需要人工审批（见模块 docstring）。
        depends_on 声明前置工具：本任务内必须先调用过这些工具才能执行本工具
        （D3-2 工具编排，如 write_file 依赖 read_file）。
        validate_contracts 开启时（D5-2），schema 与 handler 签名不一致直接 raise。
        """
        if self.validate_contracts:
            validate_contract(name, parameters, handler)
        self._handlers[name] = handler
        self._risks[name] = risk
        self._depends_on[name] = tuple(depends_on or ())
        self._schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )

    def reset_call_history(self) -> None:
        """清空调用历史。Agent.run() 开始时调用，保证依赖检查按任务隔离。"""
        self._call_history.clear()

    @property
    def schemas(self) -> list[dict]:
        """返回所有 schema，调用时整体传给 LLM。"""
        return self._schemas

    def execute(self, tool_call: dict, approver: Approver = None) -> str:
        """执行一次工具调用。tool_call 形如：
        {
            "id": "...",
            "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}
        }

        approver 为 None 时所有工具直通（保持向后兼容，默认不审批）。
        """
        name = tool_call["function"]["name"]

        # 1. 解析模型生成的参数 JSON（模型可能生成坏 JSON，要容错）
        try:
            args = json.loads(tool_call["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            return f"[工具调用参数解析失败] raw={tool_call['function']['arguments']!r}"

        # 2. 找不到 handler（防御性检查）
        handler = self._handlers.get(name)
        if handler is None:
            return f"[未知工具: {name}]"

        # 2.5 工具编排（D3-2）：检查前置工具依赖
        deps = self._depends_on.get(name, ())
        if deps:
            missing = [d for d in deps if d not in self._call_history]
            if missing:
                return f"[前置工具未调用] {name} 需要先调用: {', '.join(missing)}"

        # 3. 约束设计：confirm 级工具先过人工审批（approver 存在时）
        if self._risks.get(name) == RISK_CONFIRM and approver is not None:
            try:
                allowed = approver(name, args)
            except Exception:  # noqa: BLE001 —— 审批本身出错按拒绝处理
                allowed = False
            if not allowed:
                return f"[用户拒绝] {name}({json.dumps(args, ensure_ascii=False)})"

        # 4. 执行，任何异常都转为字符串，不向上抛
        try:
            result = handler(**args)
        except Exception as e:  # noqa: BLE001 —— 工具错误要喂回模型
            return f"[工具执行出错] {type(e).__name__}: {e}"

        # 5. 强制 str：dict/list 转 JSON 文本，避免模型读到 repr 乱码
        if isinstance(result, str):
            self._call_history.add(name)  # 成功才记录（失败不算调用过）
            return result
        self._call_history.add(name)
        return json.dumps(result, ensure_ascii=False)
