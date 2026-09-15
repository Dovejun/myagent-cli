"""config 审批交互测试：肯定/否定词判定、approver 构造（离线）。"""

from config import is_approved, make_approver


def test_approved_words():
    """y / yes / 是 / 确认 / ok 等肯定表达 → True（含大小写与空白变体）。"""
    for word in ["y", "Y", "yes", "YES", " Yes ", "yeah", "ok", "OK",
                 "是", "是的", "对", "确认", "同意", "可以", "好", "通过", "没问题"]:
        assert is_approved(word), f"应通过审批: {word!r}"


def test_rejected_words():
    """否定 / 空 / 乱输入 → False。"""
    for word in ["n", "N", "no", "No", "取消", "否", "不要", "退出",
                 "", "   ", "abc", "yesss", "随便"]:
        assert not is_approved(word), f"应拒绝: {word!r}"


def test_make_approver_disabled_returns_none():
    """approval=False → 返回 None（全放行，无审批）。"""
    assert make_approver(False) is None


def test_make_approver_enabled_returns_callable():
    """approval=True → 返回可调用对象（真实输入判定在 CLI 交互层）。"""
    fn = make_approver(True)
    assert callable(fn)
