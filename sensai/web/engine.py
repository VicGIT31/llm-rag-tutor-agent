"""X1: the engine behind the DuolingoGo web UI.

The web UI drives the very same `Chat` as the terminal: same memory, RAG,
compression and agent. Everything that object touches lives on ONE worker
thread, because a SQLite connection can only be used by the thread that
opened it, and a learner holds one conversation at a time anyway. HTTP
handlers hand jobs to that thread and read back a stream of events.

Two operations deliberately bypass the job queue, because the worker may be
blocked waiting for them: stopping a generation, and answering a confirmation
request from the agent (A3).
"""

import itertools
import queue
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from dataclasses import asdict
from typing import TypeVar

from .. import features
from ..agent.approval import ApprovalRequest, Decision
from ..agent.react import Step
from ..cli import Chat, build_agent, build_rag, load_model
from ..compare import compare
from ..config import Config
from ..memory.store import MemoryStore, last_days, streak
from ..observability import EventLogger
from ..ollama_client import OllamaClient, OllamaError, Placement

T = TypeVar("T")
Emit = Callable[[dict], None]


class Cancelled(Exception):
    """The learner pressed « Stop » during an agent task."""


class WebApprover:
    """A3 in the browser: a confirmation request becomes an event, the worker waits for the answer.

    No answer before `timeout` seconds (0 = wait forever), a closed page or
    « Stop » all mean « no », exactly like the terminal."""

    def __init__(self, timeout: float):
        self.timeout = timeout
        self.emit: Emit = lambda event: None
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._pending: dict[int, tuple[threading.Event, list[bool]]] = {}

    def __call__(self, req: ApprovalRequest) -> Decision:
        request_id = next(self._ids)
        answered, answer = threading.Event(), []
        with self._lock:
            self._pending[request_id] = (answered, answer)
        self.emit({"type": "confirm", "id": request_id, "timeout": self.timeout, **asdict(req)})
        try:
            if not answered.wait(self.timeout or None):
                self.emit({"type": "confirm_closed", "id": request_id})
                return Decision(False, "timeout")
        finally:
            with self._lock:
                self._pending.pop(request_id, None)
        return Decision(True, "approved") if answer[0] else Decision(False, "refused")

    def answer(self, request_id: int, approved: bool) -> bool:
        """Called from an HTTP thread; False if the request is unknown or already closed."""
        with self._lock:
            pending = self._pending.get(request_id)
        if pending is None or pending[0].is_set():
            return False
        pending[1].append(approved)
        pending[0].set()
        return True

    def refuse_all(self) -> None:
        with self._lock:
            pending = list(self._pending.values())
        for answered, answer in pending:
            if not answered.is_set():
                answer.append(False)
                answered.set()


class Engine:
    def __init__(self, config: Config, client: OllamaClient | None = None, logger: EventLogger | None = None):
        """Build the chat on the worker thread; raises what the terminal would report at startup."""
        self.config = config
        self.approver = WebApprover(config.confirm_timeout)
        self.placement: Placement | None = None  # where Ollama put the model (GPU / CPU)
        self._jobs: queue.Queue[Callable[[], None] | None] = queue.Queue()
        self._active: threading.Event | None = None  # cancel flag of the job being run
        ready: Future = Future()
        self._thread = threading.Thread(
            target=self._work, args=(client, logger, ready), name="duolingogo-engine", daemon=True
        )
        self._thread.start()
        ready.result()

    # --- worker thread -----------------------------------------------------

    def _work(self, client: OllamaClient | None, logger: EventLogger | None, ready: Future) -> None:
        try:
            self.chat = self._build(client, logger)
        except BaseException as exc:
            ready.set_exception(exc)
            return
        ready.set_result(None)
        while (job := self._jobs.get()) is not None:
            job()

    def _build(self, client: OllamaClient | None, logger: EventLogger | None) -> Chat:
        config = self.config
        logger = logger or EventLogger(config.log_path, config.log_content_chars, config.log_max_bytes)
        if client is None:
            client = OllamaClient(
                config.host, config.model, config.num_ctx, config.think, logger=logger, keep_alive=config.keep_alive
            )
            client.check()
            self.placement = load_model(client)
        store = MemoryStore(config.db_path)
        chat = Chat(config, store, client, logger, rag=build_rag(config, client, logger, store))
        try:
            chat.agent = build_agent(config, client, logger, chat.context.budget)
        except (ValueError, OSError) as exc:
            print(f"Mode agent désactivé : {exc}")
        else:
            chat.agent.tools.gate.approver = self.approver
        return chat

    def call(self, fn: Callable[[Chat], T]) -> T:
        """Run fn(chat) on the worker thread and return its result (or raise its exception)."""
        future: Future = Future()

        def job() -> None:
            try:
                future.set_result(fn(self.chat))
            except BaseException as exc:
                future.set_exception(exc)

        self._jobs.put(job)
        return future.result()

    def stream(self, work: Callable[[Chat, Emit, threading.Event], None]) -> Iterator[dict]:
        """Run work(chat, emit, cancel) on the worker thread and yield its events as they come."""
        events: queue.Queue[dict | None] = queue.Queue()
        cancel = threading.Event()

        def job() -> None:
            chat = self.chat
            notify = chat.notify
            chat.notify = lambda message: events.put({"type": "notice", "text": message})
            self.approver.emit = events.put
            self._active = chat.cancel = cancel
            try:
                work(chat, events.put, cancel)
            except OllamaError as exc:
                # « Stop » cuts the connection to Ollama: that is a stop, not an error
                events.put({"type": "stopped"} if cancel.is_set() else {"type": "error", "message": str(exc)})
            except Exception as exc:  # report it, never let it kill the worker
                events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            finally:
                self._active = chat.cancel = None
                self.approver.emit = lambda event: None
                chat.notify = notify
                events.put(None)

        self._jobs.put(job)
        finished = False
        try:
            while (event := events.get()) is not None:
                yield event
            finished = True
        finally:
            if not finished:  # the page went away: stop the job and refuse what it waits for
                cancel.set()
                if self._active is cancel:  # still queued: nothing of ours to cut
                    self.chat.client.abort_streams()
                self.approver.refuse_all()

    def stop(self) -> None:
        """« Stop »: end the current generation or agent task, refusing any pending confirmation."""
        active = self._active
        if active is not None:
            active.set()
            self.chat.client.abort_streams()  # even before the first word (prompt evaluation)
        self.approver.refuse_all()

    def close(self) -> None:
        def shutdown(chat: Chat) -> None:
            chat.close()
            chat.store.close()

        self.call(shutdown)
        self._jobs.put(None)
        self._thread.join(timeout=10)

    # --- what the web UI asks for -------------------------------------------

    def chat_events(self, text: str) -> Iterator[dict]:
        """B1 + M1-M3 + R1/R2: one tutor turn, streamed."""

        def work(chat: Chat, emit: Emit, cancel: threading.Event) -> None:
            turn = chat.stream(text)
            for piece in turn:
                emit({"type": "token", "text": piece})
                if cancel.is_set():  # the part already shown is kept, like in the terminal
                    turn.close()
                    break
            self._refresh_placement(chat)
            emit({"type": "done", "interrupted": chat.interrupted, "placement": self.placement_info(), **turn_report(chat)})
            chat.apply_pending()  # M3: memorize the exchange now, so the progress panel follows
            emit({"type": "memory", **memory_snapshot(chat)})

        return self.stream(work)

    def agent_events(self, task: str) -> Iterator[dict]:
        """A1 + T2 + T4 + A3: one agent task, its steps and confirmation requests streamed."""

        def work(chat: Chat, emit: Emit, cancel: threading.Event) -> None:
            if chat.agent is None:
                emit({"type": "error", "message": "Mode agent indisponible."})
                return

            def on_step(step: Step) -> None:
                emit({"type": "step", **step_event(step)})
                if cancel.is_set():
                    raise Cancelled

            previous, chat.agent.on_step = chat.agent.on_step, on_step
            try:
                result = chat.agent_turn(task)
            except Cancelled:
                emit({"type": "stopped"})
                return
            finally:
                chat.agent.on_step = previous
            emit({
                "type": "done", "outcome": result.outcome, "answer": result.answer,
                "steps": len(result.steps), "duration_ms": result.duration_ms,
            })

        return self.stream(work)

    def compare_events(self, text: str, feature: str, judge: bool) -> Iterator[dict]:
        """The same message with and without one feature, both answers streamed (nothing is stored)."""

        def work(chat: Chat, emit: Emit, cancel: threading.Event) -> None:
            run = compare(chat, text, feature, judge)
            for event in run:
                emit(event)
                if cancel.is_set():
                    run.close()  # puts the switches and the side panel state back
                    emit({"type": "stopped"})
                    return

        return self.stream(work)

    def set_model(self, model: str) -> str:
        """Switch every model call (chat, memory, compression, rerank, agent) to `model`."""

        def switch(chat: Chat) -> str:
            previous = chat.client.model
            chat.client.model = model
            try:
                chat.client.check()
            except OllamaError:
                chat.client.model = previous
                raise
            chat.config.model = model
            self.placement = load_model(chat.client)
            return model

        return self.call(switch)

    def placement_info(self) -> dict | None:
        """For the header badge: share of the model on the GPU, and a sentence saying so."""
        placement = self.placement
        if placement is None or not placement.size:
            return None
        return {"gpu_share": round(placement.gpu_share, 3), "label": placement.describe()}

    def _refresh_placement(self, chat: Chat) -> None:
        """Ollama may have unloaded the model after a pause and loaded it again elsewhere."""
        try:
            self.placement = chat.client.placement()
        except OllamaError:
            pass


# --- snapshots sent to the browser (built on the worker thread) --------------


def state_snapshot(chat: Chat) -> dict:
    user = chat.config.user
    saved = chat.store.get_summary(chat.session_id)
    return {
        "app": "DuolingoGo",
        "model": chat.client.model,
        "user": user,
        "session": chat.session_id,
        "profile": chat.store.get_profile(user),
        "messages": [
            {"id": m.id, "role": m.role, "content": m.content, "meta": m.meta}
            for m in chat.store.load_messages(chat.session_id)
        ],
        "summarized_until": saved[1] if saved else 0,  # messages up to this id live in the summary (M2)
        "summary": chat.summary.render() if chat.summary and not chat.summary.is_empty() else None,
        "features": {
            "memory": chat.switches["memory"],
            "compression": chat.switches["compression"],
            "rag": chat.context.retriever is not None,
            "agent": chat.agent is not None,
        },
        "switches": features_snapshot(chat),
        "num_ctx": chat.config.num_ctx,
        "budget": chat.context.budget,
        "last_turn": turn_report(chat) if chat.last_report is not None else None,  # survives a page reload
    }


def features_snapshot(chat: Chat) -> list[dict]:
    return features.describe(chat.switches, chat.feature_available())


def users_snapshot(chat: Chat) -> list[dict]:
    rows = []
    for u in chat.store.list_users():
        rows.append({
            "name": u.name, "created_at": u.created_at, "sessions": u.sessions, "messages": u.messages,
            "last_active": u.last_active, "language": chat.store.get_profile(u.name).get("langue_cible", ""),
            "words": chat.learner.totals(u.name)["words"], "current": u.name == chat.config.user,
        })
    return rows


def history_snapshot(chat: Chat, days: int = 14) -> dict:
    """The learner's history: activity per day, streak, totals, and the sessions that have messages."""
    user = chat.config.user
    activity = chat.store.activity(user)
    sessions = [s for s in chat.store.list_sessions(user, limit=200) if s.message_count or s.id == chat.session_id]
    totals = chat.learner.totals(user)
    return {
        "user": user,
        "streak": streak(set(activity)),
        "active_days": len(activity),
        "first_day": min(activity) if activity else None,
        "messages": sum(activity.values()),
        "sessions_count": sum(1 for s in sessions if s.message_count),
        "words": totals["words"],
        "mastered": totals["mastered"],
        "days": last_days(activity, days),
        "sessions": [
            {
                "id": s.id, "title": s.title, "updated_at": s.updated_at,
                "messages": s.message_count, "current": s.id == chat.session_id,
            }
            for s in sessions
        ],
    }


def memory_snapshot(chat: Chat) -> dict:
    user, language = chat.config.user, chat.language
    words = chat.learner.words(user, language)
    mistakes = chat.learner.mistakes(user, language, limit=50)
    total, mastered = chat.learner.vocab_stats(user, language)
    now = time.time()
    return {
        "language": language,
        "stats": {
            "words": total,
            "mastered": mastered,
            "due": sum(w.next_review <= now for w in words),
            "mistakes": len(mistakes),
        },
        "words": [
            {
                "word": w.word, "translation": w.translation, "box": w.box,
                "correct": w.correct, "wrong": w.wrong, "due": w.next_review <= now,
            }
            for w in words
        ],
        "mistakes": [
            {"pattern": m.pattern, "correction": m.correction, "example": m.example, "count": m.count}
            for m in mistakes
        ],
    }


def turn_report(chat: Chat) -> dict:
    """What went into the prompt (M2 budget), what Ollama counted, which lesson chunks were used (R1/R2)."""
    report = chat.last_report
    if report is None:
        return {"context": None, "sources": []}
    ctx = chat.context
    return {
        "context": {
            **report.to_dict(chat.config.num_ctx),
            "caps": {
                "memory": ctx.memory_cap,
                "summary": ctx.summary_cap,
                "rag": int(ctx.budget * ctx.rag_share) if ctx.retriever is not None else 0,
            },
        },
        "off": chat.last_off,
        "ollama": {
            "prompt_tokens": chat.last_stats.prompt_eval_count,
            "output_tokens": chat.last_stats.eval_count,
            "duration_ms": round(chat.last_stats.total_duration_ms),
        },
        "sources": chat.last_sources,
        "check": chat.last_check.to_dict() if chat.last_check and chat.last_check.checked else None,
    }


def step_event(step: Step) -> dict:
    obs = step.observation
    sandbox = None
    if "outcome" in obs:
        output = (obs.get("stdout") or obs.get("stderr") or "").strip().splitlines()
        sandbox = {"outcome": obs["outcome"], "exit_code": obs.get("exit_code"), "last_line": output[-1][:200] if output else ""}
    return {
        "index": step.index,
        "summary": step.summary,
        "action": step.action,
        "args": step.args,
        "status": step.status,
        "duration_ms": round(step.duration_ms),
        "error": obs.get("error") if step.status not in ("ok", "final") else None,
        "sandbox": sandbox,
    }
