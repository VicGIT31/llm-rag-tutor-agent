"""X1: the Duolingoy web API (FastAPI) and its static front end.

Local only, by design: the server listens on 127.0.0.1, rejects any other Host
header (DNS rebinding) and only accepts JSON bodies, which a page from another
site cannot send without a CORS preflight that is never granted. So a web page
open in the learner's browser cannot drive the agent, which writes files and
runs code.

Streams (a tutor answer, an agent task) are Server-Sent Events: one JSON object
per event, typed by its `type` field.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..cli import Chat
from ..monitoring import compute_metrics, load_events
from ..ollama_client import OllamaError
from .engine import Engine, memory_snapshot, state_snapshot

STATIC = Path(__file__).parent / "static"
LOCAL_HOSTS = ["127.0.0.1", "localhost"]


class ChatIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class TaskIn(BaseModel):
    task: str = Field(min_length=1, max_length=2000)


class ConfirmIn(BaseModel):
    id: int
    approved: bool


class ProfileIn(BaseModel):
    key: str = Field(min_length=1, max_length=40, pattern=r"^[\w\- ]+$")
    value: str = Field(min_length=1, max_length=200)


class ModelIn(BaseModel):
    model: str = Field(min_length=1, max_length=200)


def sse(events: Iterator[dict]) -> StreamingResponse:
    def frames() -> Iterator[str]:
        try:
            for event in events:
                yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            events.close()  # page closed mid-stream: the engine stops the job

    return StreamingResponse(frames(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})


def not_blank(text: str) -> str:
    text = text.strip()
    if not text:
        raise HTTPException(422, "message vide")
    return text


def create_app(engine: Engine, allowed_hosts: list[str] | None = None) -> FastAPI:
    app = FastAPI(
        title="Duolingoy",
        description="Le tuteur de langues Sensai dans le navigateur, sur un modèle local Ollama.",
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts or LOCAL_HOSTS)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    # --- conversation -----------------------------------------------------------

    @app.get("/api/state", summary="Session, profil, messages, fonctionnalités actives, emplacement du modèle")
    def state() -> dict:
        return {**engine.call(state_snapshot), "placement": engine.placement_info()}

    @app.post("/api/chat", summary="Envoyer un message au tuteur (réponse en flux SSE)")
    def chat(body: ChatIn) -> StreamingResponse:
        return sse(engine.chat_events(not_blank(body.text)))

    @app.post("/api/agent", summary="Confier une tâche à l'agent (étapes et confirmations en flux SSE)")
    def agent(body: TaskIn) -> StreamingResponse:
        return sse(engine.agent_events(not_blank(body.task)))

    @app.post("/api/stop", summary="Arrêter la réponse ou la tâche en cours")
    def stop() -> dict:
        engine.stop()
        return {"ok": True}

    @app.post("/api/confirm", summary="Répondre à une demande de confirmation de l'agent")
    def confirm(body: ConfirmIn) -> dict:
        if not engine.approver.answer(body.id, body.approved):
            raise HTTPException(409, "demande expirée ou déjà traitée")
        return {"ok": True}

    # --- learner memory (M1, M3) ------------------------------------------------

    @app.get("/api/memory", summary="Vocabulaire (Leitner) et erreurs récurrentes")
    def memory() -> dict:
        def snapshot(chat: Chat) -> dict:
            chat.apply_pending()
            return memory_snapshot(chat)

        return engine.call(snapshot)

    @app.delete("/api/vocab/{word}", summary="Oublier un mot")
    def forget_word(word: str) -> dict:
        def forget(chat: Chat) -> dict:
            if not chat.learner.delete_word(chat.config.user, chat.language, word):
                raise HTTPException(404, "mot absent")
            return memory_snapshot(chat)

        return engine.call(forget)

    @app.put("/api/profile", summary="Définir une information du profil")
    def set_profile(body: ProfileIn) -> dict:
        key = "_".join(body.key.lower().split())

        def update(chat: Chat) -> dict:
            chat.store.set_profile(chat.config.user, key, body.value.strip())
            return chat.store.get_profile(chat.config.user)

        return {"profile": engine.call(update)}

    @app.delete("/api/profile/{key}", summary="Supprimer une information du profil")
    def delete_profile(key: str) -> dict:
        def delete(chat: Chat) -> dict:
            if not chat.store.delete_profile_key(chat.config.user, key):
                raise HTTPException(404, "clé absente")
            return chat.store.get_profile(chat.config.user)

        return {"profile": engine.call(delete)}

    # --- sessions and context (M1, M2) ------------------------------------------

    @app.get("/api/sessions", summary="Sessions de l'apprenant")
    def sessions() -> dict:
        def listing(chat: Chat) -> list[dict]:
            return [
                {
                    "id": s.id, "title": s.title, "updated_at": s.updated_at,
                    "messages": s.message_count, "current": s.id == chat.session_id,
                }
                for s in chat.store.list_sessions(chat.config.user)
            ]

        return {"sessions": engine.call(listing)}

    @app.post("/api/sessions", summary="Commencer une nouvelle session")
    def new_session() -> dict:
        def fresh(chat: Chat) -> dict:
            chat.new_session()
            return state_snapshot(chat)

        return engine.call(fresh)

    @app.post("/api/sessions/{session_id}/load", summary="Reprendre une session")
    def load_session(session_id: int) -> dict:
        def load(chat: Chat) -> dict:
            if not chat.load_session(session_id):
                raise HTTPException(404, f"session #{session_id} introuvable")
            return state_snapshot(chat)

        return engine.call(load)

    @app.post("/api/compress", summary="Résumer maintenant les anciens échanges")
    def compress_now() -> dict:
        def compress(chat: Chat) -> dict:
            notices: list[str] = []
            notify, chat.notify = chat.notify, notices.append
            try:
                compressed = chat.compress_now()
            finally:
                chat.notify = notify
            return {"compressed": compressed, "notices": notices, "state": state_snapshot(chat)}

        return engine.call(compress)

    # --- configuration and monitoring (EV3) -------------------------------------

    @app.get("/api/models", summary="Modèles Ollama installés")
    def models() -> dict:
        config = engine.config
        try:
            resp = requests.get(f"{config.host.rstrip('/')}/api/tags", timeout=5)
            resp.raise_for_status()
            names = sorted(m["name"] for m in resp.json().get("models", []))
        except (requests.RequestException, ValueError, KeyError) as exc:
            raise HTTPException(502, f"Ollama injoignable : {exc}") from exc
        embedders = {config.rag_embed_model, f"{config.rag_embed_model}:latest"}
        return {"current": config.model, "models": [n for n in names if n not in embedders]}

    @app.put("/api/model", summary="Changer de modèle sans redémarrer")
    def set_model(body: ModelIn) -> dict:
        try:
            return {"model": engine.set_model(body.model), "placement": engine.placement_info()}
        except OllamaError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/metrics", summary="Métriques du journal d'événements (latence, tokens, erreurs…)")
    def metrics() -> dict:
        return compute_metrics(load_events(engine.config.log_path))

    return app
