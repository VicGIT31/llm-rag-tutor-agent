"""Configuration: config.json values, overridable from the command line."""

import argparse
import json
from dataclasses import dataclass, field, fields
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("config.json")


@dataclass
class Config:
    model: str = "qwen3-coder:30b"
    host: str = "http://localhost:11434"
    num_ctx: int = 8192
    response_reserve: float = 0.2
    think: bool | None = False
    keep_alive: str | int | None = "30m"  # how long Ollama keeps the models loaded (its default: 5 min)
    db_path: str = "data/sensai.db"
    user: str = "default"
    system_prompt: str = "You are Sensai, a friendly language tutor."
    # M2: compression of old turns
    compression: bool = True
    compress_trigger: float = 0.8
    compress_keep: float = 0.4
    summary_share: float = 0.1
    # M3: structured learner memory
    memory: bool = True
    memory_share: float = 0.1
    # R1/R2: RAG (documents are indexed locally, embeddings come from Ollama)
    rag: bool = True
    rag_sources: list[str] = field(default_factory=lambda: ["docs/lecons"])
    rag_db_path: str = "data/rag.db"
    rag_embed_model: str = "bge-m3"  # multilingual; `ollama pull bge-m3`
    rag_embed_gpu: bool = False  # keep the embedder on the CPU so the GPU memory goes to the chat model
    rag_share: float = 0.25  # share of the token budget reserved for retrieved chunks
    rag_chunk_chars: int = 700
    rag_auto_index: bool = True  # sync the index at startup (only changed files are embedded)
    rag_top_k: int = 4  # chunks injected at most
    rag_candidates: int = 8  # chunks sent to the reranker
    rag_hybrid: bool = True  # fuse BM25 with the cosine ranking (off = dense search only)
    rag_min_cosine: float = 0.30  # gate 1; model- and corpus-dependent, calibrate with /rag search
    rag_rerank: bool = True  # one extra model call per message that has candidates
    rag_min_rerank: float = 0.5  # gate 2, on the 0-1 rerank score
    rag_profile_filters: dict[str, str] = field(default_factory=lambda: {"langue_cible": "langue"})
    # A1: ReAct agent (/agent)
    agent_max_steps: int = 8
    agent_max_parse_errors: int = 2
    agent_observation_chars: int = 2000
    # T4: file access within permissions (paths relative to the working directory)
    file_roots: dict[str, str] = field(default_factory=lambda: {"workspace": "rw", "docs": "ro"})
    file_allowed_ext: list[str] = field(default_factory=lambda: [".md", ".txt", ".csv", ".json", ".py"])
    file_denied_ext: list[str] = field(
        default_factory=lambda: [".env", ".pem", ".key", ".db", ".sqlite", ".sqlite3", ".pyc"]
    )
    file_max_bytes: int = 100_000
    file_allow_symlinks: bool = False
    # T2: sandboxed code execution
    sandbox_backend: str = "subprocess"  # or "docker" (stronger isolation, needs a running daemon)
    sandbox_runtimes: list[str] = field(default_factory=lambda: ["python", "python-unittest"])
    sandbox_timeout: float = 10.0
    sandbox_max_output: int = 8000
    sandbox_memory_mb: int = 256
    sandbox_docker_image: str = "python:3.12-alpine"
    # A3: human-in-the-loop ("write" and "delete" are always confirmed)
    confirm_actions: list[str] = field(default_factory=lambda: ["write", "delete", "execute"])
    confirm_timeout: float = 300.0  # seconds without an answer = refusal (0 = wait forever)
    # EV3: structured logging & monitoring
    log_path: str = "data/logs/sensai.jsonl"
    log_content_chars: int = 120  # longest text excerpt kept in a log event (0 = lengths only)
    log_max_bytes: int = 5_000_000  # rotate the log file past this size


class ConfigError(Exception):
    pass


def _load_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: invalid JSON ({exc})") from exc
    known = {f.name for f in fields(Config)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{path}: unknown keys {sorted(unknown)}")
    return data


def load_config(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(prog="sensai", description="Local language tutor on Ollama")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model", help="Ollama model tag (see `ollama list`)")
    parser.add_argument("--host", help="Ollama base URL")
    parser.add_argument("--num-ctx", type=int, dest="num_ctx", help="context window size in tokens")
    parser.add_argument("--db", dest="db_path", help="SQLite database path")
    parser.add_argument("--user", help="learner profile name")
    args = parser.parse_args(argv)

    values = _load_file(args.config)
    for key in ("model", "host", "num_ctx", "db_path", "user"):
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    config = Config(**values)
    for key in ("response_reserve", "compress_trigger", "compress_keep", "summary_share", "memory_share"):
        if not 0 < getattr(config, key) < 1:
            raise ConfigError(f"{key} must be between 0 and 1")
    if config.compress_keep >= config.compress_trigger:
        raise ConfigError("compress_keep must be lower than compress_trigger")
    if config.agent_max_steps < 1 or config.agent_max_parse_errors < 0 or config.agent_observation_chars < 100:
        raise ConfigError("agent_max_steps >= 1, agent_max_parse_errors >= 0, agent_observation_chars >= 100")
    if not isinstance(config.file_roots, dict) or any(m not in ("ro", "rw") for m in config.file_roots.values()):
        raise ConfigError('file_roots must map directories to "ro" or "rw"')
    if config.file_max_bytes <= 0:
        raise ConfigError("file_max_bytes must be positive")
    if config.sandbox_backend not in ("subprocess", "docker"):
        raise ConfigError('sandbox_backend must be "subprocess" or "docker"')
    if config.sandbox_timeout <= 0 or config.sandbox_max_output <= 0 or config.sandbox_memory_mb <= 0:
        raise ConfigError("sandbox_timeout, sandbox_max_output and sandbox_memory_mb must be positive")
    unknown_levels = set(config.confirm_actions) - {"read", "write", "delete", "execute"}
    if unknown_levels:
        raise ConfigError(f"confirm_actions: unknown levels {sorted(unknown_levels)}")
    if config.confirm_timeout < 0:
        raise ConfigError("confirm_timeout must be positive (0 = no timeout)")
    if config.log_content_chars < 0 or config.log_max_bytes < 0:
        raise ConfigError("log_content_chars and log_max_bytes must be positive")
    if not 0 < config.rag_share < 1:
        raise ConfigError("rag_share must be between 0 and 1")
    if config.rag_top_k < 1 or config.rag_candidates < config.rag_top_k or config.rag_chunk_chars < 100:
        raise ConfigError("rag_top_k >= 1, rag_candidates >= rag_top_k, rag_chunk_chars >= 100")
    if not 0 <= config.rag_min_cosine < 1 or not 0 <= config.rag_min_rerank <= 1:
        raise ConfigError("rag_min_cosine must be in [0, 1) and rag_min_rerank in [0, 1]")
    if not isinstance(config.rag_profile_filters, dict):
        raise ConfigError("rag_profile_filters must map profile keys to metadata keys")
    if config.summary_share + config.memory_share > 0.5:
        raise ConfigError("summary_share + memory_share must stay under 0.5 to leave room for history and RAG")
    return config
