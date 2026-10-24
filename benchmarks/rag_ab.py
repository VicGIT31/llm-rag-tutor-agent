"""R1/R2 benchmark: what does the RAG change in the answers?

Asks the same questions under several retrieval variants and checks each
answer for the facts it must contain. Most questions are about the course
itself (docs/lecons: the quiz date and room, the grading, the trip to Kyoto,
the homework, the model sentences of a lesson): a model cannot guess them.
Control questions are general knowledge, to check that the RAG does not make
easy answers worse.

  off      no RAG: the model answers from its general knowledge
  dense    R1 only: cosine ranking, no BM25, no rerank
  hybrid   R1 + BM25 fusion, no rerank
  full     R1 + R2: hybrid search and model rerank (the default configuration)
  stuff    no retrieval: the WHOLE corpus is put in the prompt (long-context baseline)

`stuff` answers the obvious question « why not give the model every lesson? »:
it shows what that costs in tokens and time, and what is lost once the corpus
no longer fits in num_ctx.

Each question runs alone (no history, no memory, no profile), at temperature 0.
An answer is correct if, for every keyword group, it contains one of the group's
keywords. Retrieval is scored separately: a course question is « found » when a
chunk of one of its expected documents is among the chunks kept for the prompt.

Usage (needs a running Ollama with the chat and embedding models):
  .venv/bin/python -m benchmarks.rag_ab --model qwen3-coder:30b
  .venv/bin/python -m benchmarks.rag_ab --variants off,dense,hybrid,full,stuff --json data/rag_ab.json
"""

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, replace

from sensai.config import load_config
from sensai.context import ContextBuilder
from sensai.memory.store import Message
from sensai.ollama_client import ChatStats, OllamaClient, OllamaError
from sensai.rag import Rag
from sensai.rag.grounding import check_answer
from sensai.rag.pipeline import format_chunk

SYSTEM = (
    "Tu es Sensai, un tuteur de langues. Réponds brièvement, en français. "
    "Si tu ne connais pas une information sur le cours de l'apprenant, dis-le."
)

# (question, keyword groups: every group needs one hit, control question?, expected documents)
QUESTIONS = [
    ("Quel jour, à quelle heure et dans quelle salle a lieu l'interro n°2 de japonais ?",
     [["b204"], ["14 h", "14h"]], False, ["interro_vendredi.md"]),
    ("Sur quelles leçons porte l'interro n°2 de japonais ?",
     [["animaux", "leçon 3"], ["compteur", "compteurs", "leçon 4"]], False, ["interro_vendredi.md"]),
    ("Les particules wa et ga tombent-elles à l'interro n°2 ?",
     [["novembre", "interro n°3", "interro 3"]], False, ["interro_vendredi.md"]),
    ("À l'interro n°2, peut-on répondre en rōmaji ?", [["moitié"]], False, ["interro_vendredi.md"]),
    ("Combien de points vaut la partie kana de l'interro n°2 ?",
     [["4 points", "4/20", "quatre points"]], False, ["interro_vendredi.md"]),
    ("Quand les absents justifiés repassent-ils l'interro n°2 ?", [["20 octobre"]], False, ["interro_vendredi.md"]),
    ("Quelle est la phrase modèle de la leçon 3 sur les animaux ?", [["neko ga suki"]], False, ["animaux.md"]),
    ("Dans ma leçon de coréen, comment se présente Camille ?", [["카미유", "kamiyu"]], False, ["coreen_salutations.md"]),
    ("Quand a lieu l'écrit final de japonais et combien de temps dure-t-il ?",
     [["29 janvier"], ["2 heures", "2 h", "2h", "deux heures"]], False, ["planning_semestre.md"]),
    ("Comment la note des interros de japonais est-elle calculée ?",
     [["plus basse", "moins bonne", "trois meilleures", "3 meilleures"]], False, ["planning_semestre.md"]),
    ("Quelle a été la moyenne de la classe à l'interro n°1 de japonais ?",
     [["13,4", "13.4"]], False, ["corrige_interro1.md"]),
    ("Quelle erreur a été la plus fréquente à l'interro n°1 de japonais ?",
     [["dare wa", "dare ga"]], False, ["corrige_interro1.md"]),
    ("Combien coûte le voyage à Kyoto et avant quelle date faut-il verser l'acompte ?",
     [["1 450", "1450", "1.450"], ["15 décembre"]], False, ["voyage_kyoto.md"]),
    ("Quel jour part-on pour le voyage à Kyoto ?", [["5 avril"]], False, ["voyage_kyoto.md"]),
    ("Pour quand faut-il rendre les devoirs des vacances de japonais ?",
     [["10 novembre"]], False, ["devoirs_vacances.md"]),
    ("Sur combien de points la prononciation est-elle notée à l'oral final de japonais ?",
     [["4 points", "sur 4", "4/20", "quatre points"]], False, ["oral_final.md"]),
    ("Comment dit-on « chien » en japonais ?", [["inu", "いぬ", "犬"]], True, ["animaux.md"]),
    ("Que veut dire « arigatou gozaimasu » ?", [["merci"]], True, ["voyage.md"]),
    ("Comment dit-on « mardi » en japonais ?", [["kayoubi", "kayōbi", "火曜日", "かようび"]], True, ["jours_dates.md"]),
    ("En coréen, quelle série de nombres utilise-t-on pour dire l'heure ?",
     [["natif", "natifs", "native", "natives", "hana", "한 시", "세 시"]], True, ["coreen_nombres.md"]),
]

VARIANTS = {
    "off": None,
    "dense": {"rag_hybrid": False, "rag_rerank": False},
    "hybrid": {"rag_hybrid": True, "rag_rerank": False},
    "full": {"rag_hybrid": True, "rag_rerank": True},
    "stuff": "everything",
}


class Everything:
    """Long-context baseline: every chunk of the corpus, whatever the question and the budget."""

    def __init__(self, rag: Rag):
        self.texts = [format_chunk(c) for c in rag.store.load_chunks()]

    def __call__(self, query: str, max_tokens: int) -> list[str]:
        return self.texts


def correct(answer: str, groups: list[list[str]]) -> bool:
    text = re.sub(r"[  ]", " ", answer.lower())  # « 1 450 € » is often written with a narrow space
    return all(any(re.search(rf"(?<!\w){re.escape(k.lower())}(?!\w)", text) for k in group) for group in groups)


@dataclass
class Result:
    variant: str
    course_ok: int
    course_total: int
    control_ok: int
    control_total: int
    found: int | None  # course questions whose expected document reached the prompt (None: no retrieval)
    avg_prompt_tokens: float  # estimated by ContextBuilder: what the prompt really contains
    avg_ollama_tokens: float  # prompt_eval_count: what Ollama read (lower when it reuses a cached prefix)
    avg_seconds: float
    errors: int
    facts_checked: int  # facts of the answers looked up in the injected chunks (grounding.py)
    facts_unsupported: int
    details: list[dict]


def run(variant: str, base, client: OllamaClient) -> Result:
    print(f"[{variant}]")
    settings = VARIANTS[variant]
    rag = retriever = None
    if settings is not None:
        rag = Rag(replace(base, **(settings if isinstance(settings, dict) else {})), client)
        rag.sync()
        retriever = Everything(rag) if settings == "everything" else rag
    builder = ContextBuilder(SYSTEM, base.num_ctx, base.response_reserve, retriever=retriever, rag_share=base.rag_share)
    details, tokens, read, seconds = [], [], [], []
    for i, (question, groups, control, expected) in enumerate(QUESTIONS):
        start = time.perf_counter()  # retrieval (embedding, rerank call) included
        messages, report = builder.build({}, [Message("user", question, id=i + 1)])
        stats = ChatStats()
        try:
            answer, error = client.chat(messages, stats, options={"temperature": 0}, purpose="benchmark"), None
        except OllamaError as exc:  # e.g. a prompt Ollama refuses because it exceeds num_ctx
            answer, error = "", str(exc)
        seconds.append(time.perf_counter() - start)
        tokens.append(report.used)
        read.append(stats.prompt_eval_count)
        if retriever is rag and rag is not None:
            sources = sorted({h.chunk.source.rsplit("/", 1)[-1] for h in rag.last_hits if h.kept})
        else:
            sources = ["(tout le corpus)"] if retriever else []
        found = bool(set(expected) & set(sources)) if retriever is rag and rag is not None else None
        ok = correct(answer, groups)
        check = None
        if retriever is rag and rag is not None and any(h.kept for h in rag.last_hits):
            check = check_answer(answer, [h.chunk.text for h in rag.last_hits if h.kept], question).to_dict()
        details.append({"question": question, "control": control, "ok": ok, "found": found, "sources": sources,
                        "error": error, "check": check, "answer": answer.strip()})
        mark = "✔" if ok else "✘"
        print(f"  {mark} {question}  {', '.join(sources)}" + (f"  [erreur : {error}]" if error else ""))
    if rag:
        rag.store.close()
    course = [d for d in details if not d["control"]]
    controls = [d for d in details if d["control"]]
    found = sum(bool(d["found"]) for d in course) if retriever is rag and rag is not None else None
    return Result(
        variant, sum(d["ok"] for d in course), len(course), sum(d["ok"] for d in controls), len(controls), found,
        sum(tokens) / len(tokens), sum(read) / len(read), sum(seconds) / len(seconds),
        sum(1 for d in details if d["error"]),
        sum(d["check"]["checked"] for d in details if d["check"]),
        sum(len(d["check"]["unsupported"]) for d in details if d["check"]),
        details,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="chat model (default: config.json)")
    p.add_argument("--variants", default="off,full", help=f"comma-separated, among {','.join(VARIANTS)}")
    p.add_argument("--json", help="write raw results to this file")
    args = p.parse_args()

    argv = ["--model", args.model] if args.model else []
    base = replace(load_config(argv), rag_profile_filters={})  # the Korean lessons must stay reachable
    client = OllamaClient(base.host, base.model, base.num_ctx, base.think, keep_alive=base.keep_alive)
    client.check()
    results = [run(v, base, client) for v in args.variants.split(",")]

    print(f"\nModèle {base.model}, {len(QUESTIONS)} questions, embeddings {base.rag_embed_model}, num_ctx {base.num_ctx}\n")
    print("| Variante | Questions sur le cours | Contrôle | Bonne leçon dans le prompt | "
          "Tokens de prompt (moy.) | Tokens lus par Ollama (moy.) | Temps par question | Faits absents des leçons |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        found = f"{r.found}/{r.course_total}" if r.found is not None else "—"
        errors = f" ({r.errors} erreur(s))" if r.errors else ""
        print(f"| {r.variant} | {r.course_ok}/{r.course_total} ({100 * r.course_ok / r.course_total:.0f} %){errors} | "
              f"{r.control_ok}/{r.control_total} | {found} | {r.avg_prompt_tokens:.0f} | "
              f"{r.avg_ollama_tokens:.0f} | {r.avg_seconds:.1f} s | "
              f"{f'{r.facts_unsupported}/{r.facts_checked}' if r.facts_checked else '—'} |")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
