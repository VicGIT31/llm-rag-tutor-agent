"""R1 + R2: the retrieval pipeline behind ContextBuilder's `retriever` hook.

    query ──► metadata filters (explicit + profile)            R2
          ──► dense score (cosine) + lexical score (BM25)      R1 / R2
          ──► gate 1: drop chunks under rag_min_cosine         R2  (near-misses)
          ──► fusion (RRF) → top rag_candidates
          ──► rerank: the chat model scores each candidate 0-10 in ONE call   R2
          ──► gate 2: drop chunks under rag_min_rerank         R2  (near-misses)
          ──► top rag_top_k that fit in the token budget → text chunks

Failure policy: RAG is an enhancement, never a reason to lose a chat turn.
If embedding fails the turn goes on without documents; if the rerank call
fails or returns garbage, the fusion order is kept (cosine gate still applies).
"""

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ..config import Config
from ..memory.tokens import MESSAGE_OVERHEAD, estimate_tokens
from ..observability import EventLogger, NullLogger
from ..ollama_client import ChatStats, OllamaClient, OllamaError
from .chunker import Chunk, chunk_markdown, load_documents
from .embedder import OllamaEmbedder
from .filters import Filter, matches_all, parse_filter
from .lexical import bm25_scores, tokenize
from .store import RagStore, StoredChunk

RRF_K = 60  # standard reciprocal-rank-fusion constant
RERANK_CHARS = 500  # characters of each candidate shown to the reranker

RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "score": {"type": "integer"}},
                "required": ["id", "score"],
            },
        }
    },
    "required": ["scores"],
}

RERANK_SYSTEM = (
    "Tu évalues la pertinence de passages de cours pour répondre à un message d'apprenant. "
    "Pour chaque passage, donne une note entière de 0 (sans rapport) à 10 (contient directement "
    "ce qu'il faut pour répondre). Un passage qui parle du même thème sans aider à répondre vaut "
    "au plus 3. Réponds uniquement en JSON : {\"scores\": [{\"id\": 0, \"score\": 7}, ...]} avec un "
    "élément par passage."
)


@dataclass
class Hit:
    """One scored chunk. `reason` says why a chunk was dropped (empty when kept)."""

    chunk: StoredChunk
    cosine: float
    bm25: float
    rrf: float = 0.0
    rerank: float | None = None  # 0-1, None when the reranker did not run or failed
    kept: bool = False
    reason: str = ""


@dataclass
class SyncReport:
    files: int = 0
    indexed: int = 0
    unchanged: int = 0
    removed: int = 0
    chunks: int = 0
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        text = (
            f"{self.files} document(s), {self.chunks} morceaux en base — "
            f"{self.indexed} (ré)indexé(s), {self.unchanged} inchangé(s), {self.removed} supprimé(s)"
        )
        return text + "".join(f"\n  ⚠ {w}" for w in self.warnings)


def format_chunk(chunk: StoredChunk) -> str:
    return f"[{chunk.source}]\n{chunk.text}"


class Rag:
    def __init__(
        self,
        config: Config,
        client: OllamaClient,
        logger: EventLogger | None = None,
        profile_fn: Callable[[], dict[str, str]] | None = None,
        embedder: OllamaEmbedder | None = None,
        store: RagStore | None = None,
    ):
        self.config = config
        self.client = client
        self.logger = logger or NullLogger()
        self.profile_fn = profile_fn
        self.embedder = embedder or OllamaEmbedder(config.host, config.rag_embed_model, gpu=config.rag_embed_gpu)
        self.store = store or RagStore(config.rag_db_path)
        self.filters: list[Filter] = []  # explicit, set with /rag filter
        self.last_query = ""
        self.last_hits: list[Hit] = []
        self.last_status = ""
        self._chunks: list[StoredChunk] | None = None
        self._tokens: dict[int, list[str]] = {}
        self._warned = False

    # ------------------------------------------------------------------ indexing

    def check(self) -> None:
        self.embedder.check()

    def sync(self) -> SyncReport:
        """Bring the index in line with the source files; only new or changed files are embedded."""
        cfg = self.config
        docs, load = load_documents(cfg.rag_sources)
        report = SyncReport(files=len(docs))
        report.warnings += [f"dossier introuvable : {s}" for s in load.missing_sources]
        report.warnings += [f"ignoré — {s}" for s in load.skipped]
        known = self.store.state()
        for doc in docs:
            fingerprint = f"{doc.fingerprint}|{cfg.rag_embed_model}|{cfg.rag_chunk_chars}"
            if known.get(doc.source) == fingerprint:
                report.unchanged += 1
                continue
            chunks: list[Chunk] = chunk_markdown(doc.body, doc.meta["title"], cfg.rag_chunk_chars)
            vectors = self.embedder.embed([c.text for c in chunks]) if chunks else []
            self.store.replace_document(doc, fingerprint, chunks, vectors)
            report.indexed += 1
        report.removed = self.store.delete_except({d.source for d in docs})
        report.chunks = self.store.stats()["chunks"]
        self._chunks, self._tokens = None, {}
        self.logger.log("rag_sync", **{k: getattr(report, k) for k in ("files", "indexed", "unchanged", "removed", "chunks")})
        return report

    def _load(self) -> list[StoredChunk]:
        if self._chunks is None:
            self._chunks = self.store.load_chunks()
        return self._chunks

    # ------------------------------------------------------------------ filters

    def set_filters(self, exprs: list[str]) -> None:
        self.filters = [parse_filter(e) for e in exprs]  # raises ValueError before changing anything

    def active_filters(self) -> list[Filter]:
        """Explicit filters + soft filters derived from the learner profile (config.rag_profile_filters)."""
        active = list(self.filters)
        explicit = {f.key for f in active}
        profile = self.profile_fn() if self.profile_fn else {}
        for profile_key, meta_key in self.config.rag_profile_filters.items():
            value = profile.get(profile_key)
            if value and meta_key not in explicit:
                active.append(Filter(meta_key, "=", (value.strip().casefold(),), soft=True))
        return active

    # ------------------------------------------------------------------ retrieval

    def search(self, query: str, max_tokens: int | None = None, interaction_id: str | None = None) -> list[Hit]:
        """Run the whole pipeline; returns every scored chunk (kept ones first)."""
        cfg = self.config
        start = time.perf_counter()
        filters = self.active_filters()
        chunks = self._load()
        pool = [c for c in chunks if matches_all(c.meta, filters)]
        rerank_status = "off"
        hits: list[Hit] = []

        if pool:
            qvec = self.embedder.embed([query])[0]
            lexical = bm25_scores(tokenize(query), [self._doc_tokens(c) for c in pool])
            hits = [Hit(c, sum(a * b for a, b in zip(qvec, c.vec)), s) for c, s in zip(pool, lexical)]

            by_dense = sorted(hits, key=lambda h: -h.cosine)
            by_lexical = [h for h in sorted(hits, key=lambda h: -h.bm25) if h.bm25 > 0]
            for ranking in (by_dense, by_lexical):
                for rank, h in enumerate(ranking):
                    h.rrf += 1 / (RRF_K + rank + 1)

            for h in hits:
                if h.cosine < cfg.rag_min_cosine:
                    h.reason = f"cosinus {h.cosine:.2f} < {cfg.rag_min_cosine:.2f}"
            alive = sorted((h for h in hits if not h.reason), key=lambda h: -h.rrf)
            for h in alive[cfg.rag_candidates:]:
                h.reason = f"hors des {cfg.rag_candidates} meilleurs candidats"
            alive = alive[: cfg.rag_candidates]

            if cfg.rag_rerank and alive:
                rerank_status = self._rerank(query, alive, interaction_id)
                for h in alive:
                    if h.rerank is not None and h.rerank < cfg.rag_min_rerank:
                        h.reason = f"rerank {h.rerank * 10:.0f}/10 < {cfg.rag_min_rerank * 10:.0f}/10"
            alive = [h for h in alive if not h.reason]
            alive.sort(key=lambda h: (-(h.rerank or 0.0), -h.rrf))

            budget = None if max_tokens is None else max(0, max_tokens - MESSAGE_OVERHEAD)
            used = kept = 0
            for h in alive:
                cost = estimate_tokens(format_chunk(h.chunk)) + 2  # + separator
                if kept >= cfg.rag_top_k:
                    h.reason = f"au-delà de rag_top_k={cfg.rag_top_k}"
                elif budget is not None and used + cost > budget:
                    h.reason = "ne tient pas dans le budget de tokens"
                else:
                    h.kept, used, kept = True, used + cost, kept + 1

        hits.sort(key=lambda h: (not h.kept, -(h.rerank or 0.0), -h.rrf if h.kept else -h.cosine))
        self.last_query, self.last_hits, self.last_status = query, hits, "ok"
        self.logger.log(
            "rag_retrieve",
            interaction_id,
            status="ok",
            duration_ms=round((time.perf_counter() - start) * 1000, 1),
            query_chars=len(query),
            indexed=len(chunks),
            after_filters=len(pool),
            kept=sum(h.kept for h in hits),
            rerank=rerank_status,
            filters=[str(f) for f in filters],
        )
        return hits

    def __call__(self, query: str, max_tokens: int) -> list[str]:
        """The `retriever` contract of ContextBuilder: never raises on a model/server problem."""
        try:
            hits = self.search(query, max_tokens)
        except OllamaError as exc:
            self.last_query, self.last_hits, self.last_status = query, [], f"erreur : {exc}"
            self.logger.log("rag_retrieve", status="error", error=f"{type(exc).__name__}: {exc}")
            if not self._warned:
                print(f"[RAG indisponible pour ce message : {exc}]", file=sys.stderr)
                self._warned = True
            return []
        self._warned = False
        return [format_chunk(h.chunk) for h in hits if h.kept]

    def _doc_tokens(self, chunk: StoredChunk) -> list[str]:
        if chunk.id not in self._tokens:
            self._tokens[chunk.id] = tokenize(chunk.text)
        return self._tokens[chunk.id]

    def _rerank(self, query: str, hits: list[Hit], interaction_id: str | None) -> str:
        """One model call scores all candidates. Sets `h.rerank`; returns "ok" or "failed"."""
        passages = "\n\n".join(f"[{i}] {h.chunk.text[:RERANK_CHARS]}" for i, h in enumerate(hits))
        messages = [
            {"role": "system", "content": RERANK_SYSTEM},
            {"role": "user", "content": f"Message de l'apprenant :\n{query}\n\nPassages :\n{passages}"},
        ]
        try:
            raw = self.client.chat(
                messages,
                ChatStats(),
                schema=RERANK_SCHEMA,
                options={"temperature": 0},
                purpose="rerank",
                interaction_id=interaction_id,
            )
            scores = {int(s["id"]): float(s["score"]) for s in json.loads(raw)["scores"]}
            if set(scores) != set(range(len(hits))):
                raise ValueError("incomplete scores")
        except (OllamaError, ValueError, KeyError, TypeError):
            return "failed"
        for i, h in enumerate(hits):
            h.rerank = max(0.0, min(10.0, scores[i])) / 10
        return "ok"

    # ------------------------------------------------------------------ display (/rag)

    def describe(self) -> str:
        cfg = self.config
        stats = self.store.stats()
        filters = ", ".join(str(f) for f in self.active_filters()) or "aucun"
        lines = [
            f"RAG : {stats['documents']} document(s), {stats['chunks']} morceaux — embeddings {cfg.rag_embed_model}",
            f"Filtres actifs : {filters}",
            f"Seuils : cosinus ≥ {cfg.rag_min_cosine}, "
            + (f"rerank ≥ {cfg.rag_min_rerank * 10:.0f}/10" if cfg.rag_rerank else "rerank désactivé")
            + f"; {cfg.rag_candidates} candidats → {cfg.rag_top_k} morceaux max",
        ]
        if self.last_query:
            kept = sum(h.kept for h in self.last_hits)
            lines.append(f"Dernière recherche : {kept} morceau(x) retenu(s) sur {len(self.last_hits)} — {self.last_status}")
        return "\n".join(lines)

    def render_hits(self, limit: int = 12) -> str:
        if not self.last_hits:
            return "Aucun morceau à examiner (index vide ou tous exclus par les filtres)."
        lines = []
        for h in self.last_hits[:limit]:
            rerank = f"{h.rerank * 10:>2.0f}/10" if h.rerank is not None else "  —  "
            title = h.chunk.text.splitlines()[0][:48]
            status = "✓" if h.kept else "✗"
            why = "" if h.kept else f"  (écarté : {h.reason})"
            lines.append(
                f" {status} cos {h.cosine:.2f}  bm25 {h.bm25:>5.2f}  rerank {rerank}  "
                f"{h.chunk.source}#{h.chunk.ord} « {title} »{why}"
            )
        if len(self.last_hits) > limit:
            lines.append(f"   … {len(self.last_hits) - limit} autres morceaux écartés")
        return "\n".join(lines)
