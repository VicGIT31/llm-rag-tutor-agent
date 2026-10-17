"""A/B comparison: the same message answered with and without one feature.

Shows what a feature brings on the learner's own question, side by side:
the two prompts differ only by that feature (the history, profile, memory
and lessons are otherwise the same), and each side reports what went into
its prompt and which lesson chunks were used.

Nothing is stored: no message, no memory extraction, no compression, and the
state shown by /context and the side panel (last report, last RAG hits, the
switches themselves) is put back afterwards, even if the comparison is stopped.

Then, optionally, the model judges both answers against what the feature
contributed (lesson excerpts, learner memory, profile, earlier turns or their
summary): a score per answer and a one-line verdict. The judge sees the
answers in a random order, to cancel its preference for the first one.
"""

import json
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass

from . import features
from .memory.store import Message
from .memory.tokens import estimate_message
from .observability import new_interaction_id
from .ollama_client import ChatStats, OllamaError

JUDGE_CHARS = 2500  # of each answer, and of the reference, shown to the judge

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        **{
            key: {
                "type": "object",
                "properties": {"score": {"type": "integer"}, "raison": {"type": "string"}},
                "required": ["score", "raison"],
            }
            for key in ("reponse_1", "reponse_2")
        },
        "verdict": {"type": "string"},
    },
    "required": ["reponse_1", "reponse_2", "verdict"],
}

JUDGE_SYSTEM = (
    "Tu évalues deux réponses d'un tuteur de langues au même message d'apprenant. On te donne aussi des "
    "INFORMATIONS DE RÉFÉRENCE : ce que le tuteur pouvait savoir (extraits des leçons du cours, mémoire de "
    "l'apprenant, profil, échanges précédents). Pour chaque réponse, donne une note entière de 0 à 10 : "
    "10 = s'appuie correctement sur les informations utiles et ne les contredit jamais ; 5 = juste mais "
    "générique, n'utilise pas les informations alors qu'elles aidaient ; 0 = contredit les informations ou "
    "invente des faits sur le cours ou l'apprenant. Juge le fond, pas le style ni la longueur. Raison : une "
    "phrase courte. Verdict : une phrase qui dit ce que la meilleure réponse apporte de concret, sans citer "
    "les numéros des réponses. "
    "Réponds uniquement en JSON."
)

# The judge only runs where the feature contributes information an answer can be checked against;
# the R2 switches change *which* chunks are retrieved, so their sides are compared on the sources.
JUDGED = {"history", "profile", "memory", "compression", "rag"}


@dataclass
class Side:
    label: str
    answer: str
    reference: str  # what the feature put in this side's prompt


def kept_sources(chat) -> list[dict]:
    """Lesson chunks injected in the prompt just built (empty when the RAG is off)."""
    if chat.context.retriever is None or not chat.rag.last_query:
        return []
    return [
        {
            "source": h.chunk.source, "part": h.chunk.ord, "title": h.chunk.text.splitlines()[0],
            "cosine": round(h.cosine, 2), "bm25": round(h.bm25, 2),
            "rerank": None if h.rerank is None else round(h.rerank * 10),
        }
        for h in chat.rag.last_hits
        if h.kept
    ]


def contribution(chat, feature_id: str, memory: str | None, history: list[Message]) -> str:
    """What `feature_id` added to the prompt just built: the reference the judge checks answers against."""
    if feature_id == "rag":
        if chat.context.retriever is None:
            return ""
        return "\n\n".join(f"[{h.chunk.source}]\n{h.chunk.text}" for h in chat.rag.last_hits if h.kept)
    if feature_id == "memory":
        return memory or ""
    if feature_id == "profile":
        return "\n".join(f"- {k}: {v}" for k, v in chat.prompt_profile().items())
    if feature_id == "compression":
        return chat.summary.render() if chat.summary and not chat.summary.is_empty() else ""
    if feature_id == "history":
        return "\n".join(f"{m.role}: {m.content}" for m in history[:-1])[-JUDGE_CHARS:]
    return ""


def compare(chat, text: str, feature_id: str, judge: bool = True) -> Iterator[dict]:
    """Yield events: side, token…, side_done (twice: « on » then « off »), then verdict or notice, then end."""
    feature = features.check(feature_id)
    if not chat.feature_available()[feature_id]:
        raise features.FeatureError(f"{feature.label} : indisponible (RAG désactivé au démarrage)")
    text = text.strip()
    if not text:
        raise ValueError("message vide")
    chat.apply_pending()  # the memory block must be the same on both sides

    interaction_id = new_interaction_id()
    history = chat.history + [Message("user", text, estimate_message(text))]
    switches = dict(chat.switches)
    saved = (chat.last_report, chat.last_memory, chat.last_stats)
    rag_saved = (chat.rag.last_query, chat.rag.last_hits, chat.rag.last_status) if chat.rag else None
    sides: dict[str, Side] = {}
    sources: dict[str, list[dict]] = {}
    try:
        for side, enabled in (("on", True), ("off", False)):
            chat.switches = dict(switches)
            if feature.requires:
                chat.switches[feature.requires] = True  # comparing rerank on/off needs the RAG on both sides
            chat.switches[feature_id] = enabled
            chat._apply_switches()
            messages, report, memory = chat.build_prompt(text, history)
            sources[side] = kept_sources(chat)
            label = f"{'Avec' if enabled else 'Sans'} {feature.label[0].lower()}{feature.label[1:]}"
            reference = contribution(chat, feature_id, memory, history) if enabled else ""
            yield {"type": "side", "side": side, "label": label}

            stats, parts = ChatStats(), []
            start = time.perf_counter()
            pieces = chat.client.chat_stream(messages, stats, purpose="compare", interaction_id=interaction_id)
            try:
                for piece in pieces:
                    parts.append(piece)
                    yield {"type": "token", "side": side, "text": piece}
            finally:
                pieces.close()
            sides[side] = Side(label, "".join(parts), reference)
            yield {
                "type": "side_done", "side": side, "label": label,
                "context": report.to_dict(chat.config.num_ctx),
                "sources": sources[side],
                "ollama": {"prompt_tokens": stats.prompt_eval_count, "output_tokens": stats.eval_count},
                "duration_ms": round((time.perf_counter() - start) * 1000),
            }
    finally:
        chat.switches = switches
        chat._apply_switches()
        chat.last_report, chat.last_memory, chat.last_stats = saved
        if rag_saved is not None:
            chat.rag.last_query, chat.rag.last_hits, chat.rag.last_status = rag_saved

    key = lambda s: (s["source"], s["part"])  # noqa: E731
    on_keys, off_keys = {key(s) for s in sources["on"]}, {key(s) for s in sources["off"]}
    diff = {
        "only_on": [s for s in sources["on"] if key(s) not in off_keys],
        "only_off": [s for s in sources["off"] if key(s) not in on_keys],
    }
    verdict = None
    if judge and feature_id in JUDGED:
        reference = sides["on"].reference
        if not reference.strip():
            yield {"type": "notice", "text": f"Pas d'arbitrage : « {feature.label} » n'a rien ajouté au prompt pour ce message."}
        else:
            try:
                verdict = judge_answers(chat.client, text, reference, sides["on"], sides["off"], interaction_id)
            except (OllamaError, ValueError, KeyError, TypeError) as exc:
                yield {"type": "notice", "text": f"Arbitrage impossible : {exc}"}
            else:
                yield {"type": "verdict", **verdict}
    chat.logger.log(
        "compare", interaction_id, feature=feature_id,
        sources_on=len(sources["on"]), sources_off=len(sources["off"]),
        score_on=verdict["on"]["score"] if verdict else None,
        score_off=verdict["off"]["score"] if verdict else None,
    )
    yield {"type": "end", "feature": feature_id, "sources_diff": diff}


def judge_answers(client, text: str, reference: str, on: Side, off: Side, interaction_id: str | None = None) -> dict:
    """One model call scores both answers 0-10 against `reference`; raises on a malformed reply."""
    flipped = random.random() < 0.5
    first, second = (off, on) if flipped else (on, off)
    prompt = (
        f"INFORMATIONS DE RÉFÉRENCE :\n{reference[:JUDGE_CHARS]}\n\n"
        f"MESSAGE DE L'APPRENANT :\n{text}\n\n"
        f"RÉPONSE 1 :\n{first.answer[:JUDGE_CHARS]}\n\n"
        f"RÉPONSE 2 :\n{second.answer[:JUDGE_CHARS]}"
    )
    raw = client.chat(
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
        ChatStats(),
        schema=JUDGE_SCHEMA,
        options={"temperature": 0},
        purpose="judge",
        interaction_id=interaction_id,
    )
    data = json.loads(raw)
    scores = {}
    for slot, side in (("reponse_1", first), ("reponse_2", second)):
        score = max(0, min(10, int(data[slot]["score"])))
        scores["on" if side is on else "off"] = {"score": score, "reason": str(data[slot]["raison"]).strip()}
    return {
        "on": scores["on"], "off": scores["off"], "on_label": on.label, "off_label": off.label,
        "verdict": str(data["verdict"]).strip(),
    }
