"""子 Agent 委派：把子任务外包给一个独立 Agent 执行，只把结果回喂主 Agent。

对应学习清单 D3-3：
- 注册 delegate_task(prompt) 工具，模型可主动把子任务委派出去
- 子 Agent 共享工具注册表（同一批工具能力），但拥有独立 messages
  → 子任务的中间过程（读文件、跑命令等）不污染主 Agent 上下文
- max_steps 用小值（默认 5），防止子任务失控烧 token

三个安全设计：
1. 防递归委派：子 Agent 看到的工具列表里【没有】delegate_task 本身
   （_SubRegistryView 过滤），杜绝"子 Agent 再委派孙 Agent"的无限递归
2. 调用历史隔离：子任务执行前备份 registry 的调用历史，结束后恢复
   ——否则子 Agent.run() 里的 reset_call_history 会清掉主任务的依赖状态
3. 审批继承：子 Agent 继承主 Agent 的 approver，
   子任务里的 confirm 级工具同样要过人工审批（否则审批被绕过）

用法（agent_cli.py 的 build_agent）：
    register_delegate_tool(registry, llm, SYSTEM_PROMPT,
                           max_steps=5, approver=approver)
"""

from ..core import Agent

DEFAULT_SUB_MAX_STEPS = 5


class _SubRegistryView:
    """registry 的受限只读视图：隐藏委派工具本身，防止递归委派。

    core.Agent 只用到 registry 的三个接口（鸭子类型即可）：
      schemas / execute(tool_call, approver) / reset_call_history()
    """

    def __init__(self, base, exclude: str) -> None:
        self._base = base
        self._exclude = exclude

    @property
    def schemas(self) -> list[dict]:
        return [
            s for s in self._base.schemas
            if s["function"]["name"] != self._exclude
        ]

    def execute(self, tool_call: dict, approver=None) -> str:
        # 子任务的成功调用也计入共享调用历史（主任务依赖检查可见）
        return self._base.execute(tool_call, approver=approver)

    def reset_call_history(self) -> None:
        # 委托给 base；handler 里已做备份/恢复，主任务历史不受影响
        self._base.reset_call_history()


def register_delegate_tool(
    registry,
    llm,
    system_prompt: str,
    max_steps: int = DEFAULT_SUB_MAX_STEPS,
    approver=None,
    risk: str = "safe",
    tool_name: str = "delegate_task",
) -> None:
    """注册 delegate_task 工具：模型可把子任务委派给独立子 Agent。"""

    def delegate_task(prompt: str) -> str:
        if not prompt.strip():
            return "[委派失败] prompt 不能为空"

        # 备份调用历史：子 Agent.run() 会 reset，结束后恢复主任务状态
        saved_history = set(registry._call_history)
        try:
            sub_agent = Agent(
                llm,
                _SubRegistryView(registry, exclude=tool_name),  # 防递归
                system_prompt,
                max_steps=max_steps,      # 小步数，防失控
                approver=approver,        # 审批继承，防绕过
            )
            return sub_agent.run(prompt)
        except Exception as e:  # noqa: BLE001 —— 委派失败回喂模型，不崩溃
            return f"[委派失败] {type(e).__name__}: {e}"
        finally:
            registry._call_history.clear()
            registry._call_history.update(saved_history)

    registry.register(
        tool_name,
        "把一个独立的子任务委派给子 Agent 执行，返回子 Agent 的最终结论。"
        "适合可独立完成的小任务（统计、检索、验证）；子任务的中间过程不会占用本对话上下文。",
        {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "交给子 Agent 的完整任务描述，要自包含（子 Agent 看不到当前对话）",
                },
            },
            "required": ["prompt"],
        },
        delegate_task,
        risk=risk,
    )
