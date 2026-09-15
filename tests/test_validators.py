"""验证器测试：run_pytest / run_lint / 输出截断 / 超时（mock subprocess，离线）。"""

import subprocess
from types import SimpleNamespace

from agent import validators


def _fake_proc(returncode=0, stdout=b"", stderr=b""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_run_pytest_success(monkeypatch):
    monkeypatch.setattr(
        validators.subprocess, "run",
        lambda *a, **kw: _fake_proc(0, b"3 passed in 0.5s"),
    )
    ok, output = validators.run_pytest("x")
    assert ok is True
    assert "3 passed" in output


def test_run_pytest_failure(monkeypatch):
    monkeypatch.setattr(
        validators.subprocess, "run",
        lambda *a, **kw: _fake_proc(1, b"1 failed, 2 passed"),
    )
    ok, output = validators.run_pytest("x")
    assert ok is False
    assert "1 failed" in output


def test_run_pytest_timeout(monkeypatch):
    def fake_run(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="pytest", timeout=120)

    monkeypatch.setattr(validators.subprocess, "run", fake_run)
    ok, output = validators.run_pytest("x")
    assert ok is False
    assert "[超时]" in output


def test_run_pytest_command_not_found(monkeypatch):
    def fake_run(*a, **kw):
        raise FileNotFoundError("python")

    monkeypatch.setattr(validators.subprocess, "run", fake_run)
    ok, output = validators.run_pytest("x")
    assert ok is False
    assert "[未找到命令]" in output


def test_output_truncated(monkeypatch):
    long_output = ("x" * 100 + "\n") * 100  # 10000+ 字符
    monkeypatch.setattr(
        validators.subprocess, "run",
        lambda *a, **kw: _fake_proc(1, long_output.encode()),
    )
    ok, output = validators.run_pytest("x")
    assert len(output) <= validators.MAX_OUTPUT_CHARS + 100
    assert "已截断" in output


def test_run_lint_ruff_missing_skips(monkeypatch):
    """ruff 未安装 → 优雅跳过（ok=True），不阻塞质量闭环。"""
    monkeypatch.setattr(
        validators.subprocess, "run",
        lambda *a, **kw: _fake_proc(1, b"", b"No module named ruff"),
    )
    ok, output = validators.run_lint("x")
    assert ok is True
    assert "[lint 跳过]" in output


def test_run_lint_failure(monkeypatch):
    monkeypatch.setattr(
        validators.subprocess, "run",
        lambda *a, **kw: _fake_proc(1, b"F401 unused import"),
    )
    ok, output = validators.run_lint("x")
    assert ok is False
    assert "F401" in output
