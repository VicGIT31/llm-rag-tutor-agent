"""R1/R2 retrieval benchmark: which chunks reach the prompt, without generating any answer.

rag_ab.py measures answers; this script measures the retrieval itself, so that
each stage can be judged on its own job and many more queries fit in the time:

  cours        the 16 course questions of rag_ab.py
  mot rare     a rare word of a lesson (« hatsuka », « roppiki », « B204 »): BM25's job
  paraphrase   no word in common with the lesson (« l'examen parlé » for « oral final »)
  relance      a short follow-up (« Et l'acompte ? »), searched alone, then with the
               learner's previous message (context.retrieval_query)
  hors sujet   « Bonjour ! »: nothing should be injected

Variants: dense (cosine only), hybrid (+ BM25), full (+ rerank of 8 candidates)
and full4 (rerank of the 4 best candidates only, half the rerank prompt). Each
variant runs as a separate pass over all the queries: interleaved on the same
query, full4's rerank prompt would start exactly like full's (same instruction,
message and first passages), Ollama would reuse its cache, and full4 would look
ten times faster than it is.

Metrics: recall (a chunk of an expected document is injected), precision (share
of injected documents that are expected), injection on off-topic messages, time.

Usage (needs a running Ollama with the chat and embedding models):
  .venv/bin/python -m benchmarks.rag_retrieval --json data/rag_retrieval.json
"""

import argparse
import json
import time
from dataclasses import replace

from benchmarks.rag_ab import QUESTIONS
from sensai.config import load_config
from sensai.context import RAG_HEADER_TOKENS, retrieval_query
from sensai.memory.store import Message
from sensai.ollama_client import OllamaClient
from sensai.rag import Rag

RARE = [
    ("Que veut dire hatsuka ?", ["jours_dates.md"]),
    ("On dit sanbon ou sanhon pour trois crayons ?", ["compteurs.md", "interro_vendredi.md"]),
    ("C'est quoi roppiki ?", ["compteurs.md"]),
    ("Que veut dire kokonoka ?", ["jours_dates.md"]),
    ("Que signifie tsumetai ?", ["meteo_saisons.md"]),
    ("Quand est-ce qu'on dit juseyo ?", ["coreen_restaurant.md"]),
    ("Comment dit-on 10 000 : ichiman ?", ["achats.md"]),
    ("Qu'est-ce que le Kinkaku-ji ?", ["voyage_kyoto.md"]),
    ("C'est quoi gochisousama ?", ["restaurant.md"]),
    ("Où se trouve la salle A112 ?", ["planning_semestre.md"]),
]
PARAPHRASE = [
    ("Combien d'argent faut-il prévoir pour partir au Japon avec la classe ?", ["voyage_kyoto.md"]),
    ("Quand est-ce qu'on passe l'examen parlé ?", ["oral_final.md", "planning_semestre.md"]),
    ("Comment je demande la note à la fin du repas au resto ?", ["restaurant.md"]),
    ("Comment dire que je suis en train de regarder la télé ?", ["forme_te.md"]),
    ("Pourquoi on n'appelle pas son propre père comme le père d'un ami ?", ["famille.md"]),
    ("Comment dire qu'il fait froid aujourd'hui ?", ["meteo_saisons.md"]),
]
FOLLOW_UPS = [
    ("Combien coûte le voyage à Kyoto ?", "Et l'acompte ?", ["voyage_kyoto.md"]),
    ("Quand a lieu l'interro n°2 de japonais ?", "Et le rattrapage ?", ["interro_vendredi.md"]),
    ("Comment dit-on 3 heures en japonais ?", "Et 4 heures ?", ["heure.md"]),
    ("Quelle est la moyenne de la classe à l'interro n°1 ?", "Et la meilleure note ?", ["corrige_interro1.md"]),
    ("Comment compter des stylos en japonais ?", "Et des billets ?", ["compteurs.md"]),
    ("Comment on dit merci en coréen ?", "Et bonjour ?", ["coreen_salutations.md"]),
]
OFF_TOPIC = ["Bonjour !", "Merci, à demain !", "Quel temps fera-t-il demain à Paris ?",
             "Peux-tu m'aider à écrire un mail à mon patron ?", "Raconte-moi une blague."]

VARIANTS = {
    "dense": {"rag_hybrid": False, "rag_rerank": False},
    "hybrid": {"rag_hybrid": True, "rag_rerank": False},
    "full": {"rag_hybrid": True, "rag_rerank": True, "rag_candidates": 8},
    "full4": {"rag_hybrid": True, "rag_rerank": True, "rag_candidates": 4},
}


def queries() -> list[dict]:
    out = [{"kind": "cours", "query": q, "expected": e} for q, _, control, e in QUESTIONS if not control]
    out += [{"kind": "mot rare", "query": q, "expected": e} for q, e in RARE]
    out += [{"kind": "paraphrase", "query": q, "expected": e} for q, e in PARAPHRASE]
    for previous, current, expected in FOLLOW_UPS:
        history = [Message("user", previous), Message("assistant", "…"), Message("user", current)]
        out.append({"kind": "relance (seule)", "query": current, "expected": expected})
        out.append({"kind": "relance (élargie)", "query": retrieval_query(history, 6), "expected": expected})
    out += [{"kind": "hors sujet", "query": q, "expected": []} for q in OFF_TOPIC]
    return out


def rerank_status(rag: Rag, hits) -> str:
    """off, none (no candidate to score), ok, or failed (the fusion order was kept)."""
    if not rag.rerank:
        return "off"
    if not any(not h.reason.startswith(("cosinus", "hors des")) for h in hits):
        return "none"
    return "ok" if any(h.rerank is not None for h in hits) else "failed"


def doc(hit) -> str:
    return hit.chunk.source.rsplit("/", 1)[-1]


def summarize(rows: list[dict]) -> dict:
    stats: dict = {}
    for variant in VARIANTS:
        for kind in dict.fromkeys(r["kind"] for r in rows):
            sel = [r[variant] | {"expected": r["expected"]} for r in rows if r["kind"] == kind]
            injected = [s for s in sel if s["kept"]]
            docs = [d for s in sel for d in set(s["kept_docs"])]
            good = [d for s in sel for d in set(s["kept_docs"]) if d in s["expected"]]
            stats[f"{variant}|{kind}"] = {
                "n": len(sel),
                "recall": sum(bool(set(s["kept_docs"]) & set(s["expected"])) for s in sel) if sel[0]["expected"] else None,
                "precision": round(len(good) / len(docs), 3) if docs and sel[0]["expected"] else None,
                "injected": len(injected),
                "avg_kept": round(sum(s["kept"] for s in sel) / len(sel), 2),
                "avg_seconds": round(sum(s["seconds"] for s in sel) / len(sel), 2),
            }
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="chat model used by the reranker (default: config.json)")
    p.add_argument("--json", help="write raw results to this file")
    args = p.parse_args()
    base = replace(load_config(["--model", args.model] if args.model else []), rag_profile_filters={})
    client = OllamaClient(base.host, base.model, base.num_ctx, base.think, keep_alive=base.keep_alive)
    client.check()
    rags = {name: Rag(replace(base, **cfg), client) for name, cfg in VARIANTS.items()}
    for rag in rags.values():
        rag.sync()
    budget = int(base.num_ctx * (1 - base.response_reserve) * base.rag_share) - RAG_HEADER_TOKENS
    rags["full"].search("Bonjour", budget)  # load both models before measuring

    rows = [dict(item) for item in queries()]
    for name, rag in rags.items():
        print(f"[{name}]", flush=True)
        for row in rows:
            start = time.perf_counter()
            hits = rag.search(row["query"], budget)
            row[name] = {
                "seconds": round(time.perf_counter() - start, 2),
                "kept": sum(h.kept for h in hits),
                "kept_docs": [doc(h) for h in hits if h.kept],
                "rerank": rerank_status(rag, hits),
                "hits": [
                    {"doc": doc(h), "ord": h.chunk.ord, "cosine": round(h.cosine, 3), "bm25": round(h.bm25, 2),
                     "rrf": round(h.rrf, 5), "rerank": None if h.rerank is None else round(h.rerank * 10),
                     "kept": h.kept, "reason": h.reason}
                    for h in hits if h.kept or not h.reason.startswith("cosinus")
                ],
            }
    for row in rows:
        mark = " ".join(f"{n}:{'✔' if set(row[n]['kept_docs']) & set(row['expected']) else ('∅' if not row[n]['kept'] else '✘')}"
                        for n in rags) if row["expected"] else " ".join(f"{n}:{row[n]['kept']}" for n in rags)
        print(f"[{row['kind']}] {row['query'].replace(chr(10), ' / ')[:70]:<70} {mark}")
    for rag in rags.values():
        rag.store.close()

    stats = summarize(rows)
    print(f"\nModèle {base.model}, embeddings {base.rag_embed_model}, {len(rows)} requêtes\n")
    print("| Variante | Type | Rappel | Précision des documents | Injection | Morceaux injectés | Temps |")
    print("|---|---|---|---|---|---|---|")
    for key, s in stats.items():
        variant, kind = key.split("|")
        recall = f"{s['recall']}/{s['n']}" if s["recall"] is not None else "—"
        prec = f"{100 * s['precision']:.0f} %" if s["precision"] is not None else "—"
        print(f"| {variant} | {kind} | {recall} | {prec} | {s['injected']}/{s['n']} | {s['avg_kept']} | {s['avg_seconds']} s |")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "stats": stats}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
