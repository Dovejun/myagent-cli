"""配置解析测试：env 配置读取、MCP JSON 加载、知识/embedding 解析（离线）。"""

import json

import config as cfgmod
from config import AgentConfig


# ---------- env_embedding_config ----------


def test_env_embedding_local_path(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", "./dir-embed")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://x/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-v3")
    # MODEL_PATH 优先级最高
    assert cfgmod.env_embedding_config() == {
        "kind": "local", "model_path": "./dir-embed",
    }


def test_env_embedding_api_fallback(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL_PATH", raising=False)
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-v3")
    assert cfgmod.env_embedding_config()["kind"] == "api"


def test_env_embedding_none_without_source(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL_PATH", raising=False)
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    assert cfgmod.env_embedding_config() is None


# ---------- _resolve_knowledge_index ----------


def test_knowledge_index_priority(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_INDEX", "env_kb.json")
    # 编程字段 > env > 默认
    assert cfgmod._resolve_knowledge_index(AgentConfig()) == "env_kb.json"
    assert cfgmod._resolve_knowledge_index(AgentConfig(knowledge_index="code_kb.json")) == "code_kb.json"
    monkeypatch.delenv("KNOWLEDGE_INDEX")
    assert cfgmod._resolve_knowledge_index(AgentConfig()) == "knowledge_index.json"


# ---------- MCP JSON 加载 ----------


def test_load_mcp_servers_dict_format(tmp_path):
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps({
        "servers": [
            {"name": "fs", "command": "npx", "args": ["-y", "server-filesystem", "."]},
            {"name": "db", "command": "python", "args": ["server.py"]},
        ]
    }), encoding="utf-8")
    servers = cfgmod._load_mcp_servers(str(p))
    assert len(servers) == 2
    assert servers[0]["name"] == "fs"
    assert servers[0]["args"] == ["-y", "server-filesystem", "."]


def test_load_mcp_servers_array_format(tmp_path):
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps([
        {"name": "db", "command": "python", "args": ["server.py"]},
    ]), encoding="utf-8")
    assert len(cfgmod._load_mcp_servers(str(p))) == 1


def test_load_mcp_servers_missing_or_bad(tmp_path):
    assert cfgmod._load_mcp_servers(str(tmp_path / "no.json")) == []
    bad = tmp_path / "bad.json"
    bad.write_text("{broken", encoding="utf-8")
    assert cfgmod._load_mcp_servers(str(bad)) == []
    # 非法项被跳过，合法项保留
    mixed = tmp_path / "mixed.json"
    mixed.write_text(json.dumps([
        {"command": "python", "args": ["a.py"]},   # 缺 name → 用 command 兜底
        "not-a-dict",
        {"name": "ok2", "command": "ls", "args": []},
    ]), encoding="utf-8")
    servers = cfgmod._load_mcp_servers(str(mixed))
    assert len(servers) == 2
    assert servers[0]["name"] == "python"


# ---------- AgentConfig 默认行为（知识默认开启） ----------


def test_knowledge_enabled_by_default():
    assert AgentConfig().knowledge_enabled is True
    assert AgentConfig().knowledge_index == ""  # 空 = 走 env/默认文件


def test_from_args_maps_no_knowledge():
    args = type("A", (), {
        "max_steps": 15, "verbose": False, "yes": False, "retry": 3,
        "no_memory": False, "no_knowledge": True, "hooks": "",
    })()
    cfg = AgentConfig.from_args(args)
    assert cfg.knowledge_enabled is False
    assert cfg.approval is True


def test_from_args_ignores_config_style_args():
    """旧式连接参数不存在时 from_args 不报错（无 knowledge/embedding/mcp 字段）。"""
    args = type("A", (), {
        "max_steps": 15, "verbose": False, "yes": True, "retry": 1,
        "no_memory": True, "no_knowledge": False, "hooks": "pytest",
    })()
    cfg = AgentConfig.from_args(args)
    assert cfg.approval is False
    assert cfg.enable_memory is False
    assert cfg.hooks == ("pytest",)
