# Sensai

A local language tutor in your terminal or in your browser (**Duolingoy**, its web interface), built on a model served by [Ollama](https://ollama.com) and modeled on Duolingo. It runs short exercises, corrects your mistakes and remembers your progress. In **agent mode** it can also prepare your study material: it reads your lessons, writes revision sheets and Python quizzes, runs them in a sandbox to check them, and always asks for your confirmation before changing anything.

No LLM framework is used: every model call goes straight to Ollama's HTTP API (`/api/chat`).

## Features

| ID | Feature | Where |
|---|---|---|
| B1 | CLI chatbot with streamed answers, conversation history and error handling | `sensai/cli.py`, `sensai/ollama_client.py` |
| M1 | Persistent sessions and learner profile (SQLite) | `sensai/memory/store.py` |
| M2 | Token budget and compression of old turns | `sensai/context.py`, `sensai/memory/compressor.py` |
| M3 | Structured learner memory: vocabulary with spaced repetition, recurring mistakes | `sensai/memory/learner.py`, `sensai/memory/extractor.py` |
| **A1** | **ReAct reasoning loop**: analyse → action → observation → adjustment | `sensai/agent/react.py`, `sensai/agent/tools.py` |
| **T2** | **Sandboxed code execution**: timeout, output cap, process isolation | `sensai/agent/sandbox.py` |
| **T4** | **File access with permissions**: authorised folders, read-only/read-write, symlinks, extensions, sizes | `sensai/agent/permissions.py` |
| **A3** | **Human-in-the-loop**: confirmation before any write, deletion or execution | `sensai/agent/approval.py` |
| **EV3** | **Structured logs, secret masking and metrics** | `sensai/observability.py`, `sensai/monitoring.py` |
| **R1** | **Basic RAG**: ingest lessons, chunk, embed (Ollama `/api/embed`), SQLite vector store, inject top chunks | `sensai/rag/` |
| **R2** | **Advanced RAG**: metadata filters, hybrid search (cosine + BM25), model reranking, per-chunk relevance gates | `sensai/rag/pipeline.py`, `sensai/rag/filters.py` |
| **X1** | **Web UI (Duolingoy)**: streamed chat, learner progress, live context usage and sources, agent mode with confirmations in the browser, model switching | `sensai/web/` |

Detailed documentation:
- [docs/memoire_contexte.md](docs/memoire_contexte.md): memory and context (B1, M1, M2, M3), with **user stories**;
- [docs/rag.md](docs/rag.md): RAG pipeline, metadata, design choices, limits and **user stories** (R1, R2);
- [docs/agent_securite_observabilite.md](docs/agent_securite_observabilite.md): agent, sandbox, permissions, confirmations and monitoring (A1, T2, T4, A3, EV3), with architecture, design choices, security limits, **user stories** and the keynote demo;
- [docs/web.md](docs/web.md): Duolingoy, the web interface (X1): architecture, API, security, design choices and **user stories**;
- [docs/demo.md](docs/demo.md): the keynote scenario, one story in three acts covering every feature, with measured timings.

## Installation

Requirements: **Python ≥ 3.10** and a running Ollama server.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
ollama pull qwen3-coder:30b   # or any other model; check with `ollama list`
ollama pull bge-m3            # embedding model for the RAG (multilingual); without it the RAG is disabled, the chat still works
```

GPU or CPU is Ollama's choice, made automatically: it puts as much of the model as fits on the GPU when its build supports the card, and the rest (or all of it) on the CPU. On Arch Linux the `ollama` package is CPU-only; add `ollama-vulkan` (small, works with the NVIDIA, AMD and Intel drivers) or `ollama-cuda` (NVIDIA, needs CUDA), then restart Ollama. At startup Sensai loads the model and says where it went, e.g. `Modèle qwen3-coder:30b chargé : 32 % GPU / 68 % CPU`; the web interface shows it in its header. The small RAG embedding model stays on the CPU (`rag_embed_gpu: false`): on a GPU that the chat model already fills, Ollama would otherwise unload one model to load the other at every message. Measured on an RTX 3060 Ti (8 GB, Vulkan) with `qwen3-coder:30b`: 32 % of the model on the GPU makes an answer about 1.8 times faster than on the CPU alone (14.9 s instead of 27.1 s for a 2,000-token prompt and a 150-token answer).

## Usage

```bash
.venv/bin/python -m sensai                          # model and settings from config.json
.venv/bin/python -m sensai --model ministral-3:14b  # pick a different model without touching the code
.venv/bin/python -m sensai --user camille --db data/camille.db
```

In the browser, with **Duolingoy** (same options as the CLI, plus `--port`; see [docs/web.md](docs/web.md)):

```bash
.venv/bin/python -m sensai.web                      # then open http://127.0.0.1:8000
.venv/bin/python -m sensai.web --port 8080 --user camille
```

It shows the streamed conversation, the learner's progress (vocabulary, Leitner boxes, recurring mistakes), how the context window was filled and which lesson chunks were used, and runs agent tasks with confirmation cards. It listens on `127.0.0.1` only and works offline; the API is documented at `/api/docs`.

Chat commands (full list with `/help`):

| Command | Purpose |
|---|---|
| `/profile set langue_cible japonais` | fill in the learner profile |
| `/vocab`, `/mistakes`, `/summary`, `/context` | learner memory and context |
| `/agent <task>` | agent mode (A1): tools, sandbox, confirmations |
| `/rag`, `/rag search <question>` | RAG state; per-chunk scores and rejection reasons (use it to tune thresholds) |
| `/rag filter niveau=A1 date>=2026-09-01` | metadata filters; `/rag filter clear` removes them; `/rag index` re-syncs the index |
| `/trace` | steps of the last agent task (public summaries, actions, results) |
| `/tools` | agent tools, allowed folders, sandbox limits |
| `/metrics` | metrics: latency, tokens, errors, tools, confirmations, sandbox |

Example agent session:

```text
> /agent Write a Python quiz from docs/lecons/animaux.md, test it, then save it to workspace/quiz.py
[step 1] I read the lesson to get the vocabulary
           → read_file(path=docs/lecons/animaux.md) ✓ (1645 ms)
┌────────────────────────────────────────────────────────────
│ ⚠  Confirmation required
│ Action     : run_python — Run a Python program in the sandbox
│ ...
Confirm? [y/N] y
[step 2] I test the quiz
           → run_python(code=import random⏎…) ✓ (11839 ms)
           sandbox : error, code 1 — EOFError: EOF when reading a line
[step 3] I replace input() with a simulation mode so it can be tested
...
```

The metrics are also available outside the chat:

```bash
.venv/bin/python -m sensai.monitoring            # dashboard
.venv/bin/python -m sensai.monitoring --json     # raw metrics
tail -f data/logs/sensai.jsonl                   # live events (secrets masked)
```

## Configuration

Everything lives in `config.json` (CLI arguments take precedence over it). Main keys for agent mode:

| Key | Default | Meaning |
|---|---|---|
| `agent_max_steps` | 8 | maximum number of ReAct iterations |
| `file_roots` | `{"workspace": "rw", "docs": "ro"}` | folders the agent may access, and how |
| `file_allowed_ext` / `file_denied_ext` | `.md .txt .csv .json .py` / `.env .pem .key .db…` | allowed and denied extensions |
| `file_max_bytes` | 100000 | maximum size for a read or a write |
| `file_allow_symlinks` | false | symbolic links (when true, only towards an authorised folder) |
| `sandbox_backend` | `subprocess` | `docker` for real isolation (needs the daemon) |
| `sandbox_runtimes` | `python`, `python-unittest` | runtimes the sandbox may use |
| `sandbox_timeout` / `sandbox_max_output` | 10 s / 8000 | limits for one execution |
| `confirm_actions` | `write`, `delete`, `execute` | actions that need confirmation (`write` and `delete` always do) |
| `confirm_timeout` | 300 | after this many seconds with no answer, the action is refused |
| `rag_sources` / `rag_embed_model` | `["docs/lecons"]` / `bge-m3` | indexed files or folders, embedding model (R1) |
| `rag_top_k` / `rag_candidates` / `rag_share` | 4 / 8 / 0.25 | chunks injected, chunks reranked, share of the token budget (R1/R2) |
| `rag_min_cosine` / `rag_rerank` / `rag_min_rerank` | 0.30 / true / 0.5 | relevance gates; **calibrate with `/rag search`** (R2) |
| `log_path` / `log_content_chars` | `data/logs/sensai.jsonl` / 120 | event log and longest text excerpt it keeps |

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
SENSAI_DOCKER_TESTS=1 .venv/bin/python -m unittest tests.test_sandbox   # + Docker backend, if the daemon is running
```

The tests use no network and no model: the ReAct loop, the compression (M2) and the memory extraction (M3) are driven by a scripted fake model, while the permission layer, the sandbox, the confirmations, the SQLite memory and the web server (a real local uvicorn, queried over HTTP) are the real ones. Benchmarks that need Ollama live in `benchmarks/` (M2, M3); their results are in [docs/memoire_contexte.md](docs/memoire_contexte.md#13-tests-et-mesures).

## User stories

At least two per feature, with acceptance criteria:
- B1, M1, M2, M3: [docs/memoire_contexte.md](docs/memoire_contexte.md#12-user-stories);
- R1, R2: [docs/rag.md](docs/rag.md#6-user-stories);
- X1: [docs/web.md](docs/web.md#8-user-stories);
- A1, T2, T4, A3, EV3: [docs/agent_securite_observabilite.md](docs/agent_securite_observabilite.md#9-user-stories).
