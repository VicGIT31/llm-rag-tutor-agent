"""R1/R2 benchmark: what does the RAG change in the answers?

Asks the same questions under several retrieval variants and checks each
answer for the facts it must contain. Most questions are about the course
itself (docs/lecons: the quiz date and room, the grading, the model sentences
of a lesson): a model cannot guess them. Two control questions are general
knowledge, to check that the RAG does not make easy answers worse.

  off      no RAG: the model answers from its general knowledge
  dense    R1 only: cosine ranking, no BM25, no rerank
  hybrid   R1 + BM25 fusion, no rerank
  full     R1 + R2: hybrid search and model rerank (the default configuration)

Each question runs alone (no history, no memory, no profile), at temperature 0.
An answer is correct if, for every keyword group, it contains one of the group's keywords.

Usage (needs a running Ollama with the chat and embedding models):
  .venv/bin/python -m benchmarks.rag_ab --model qwen3-coder:30b
  .venv/bin/python -m benchmarks.rag_ab --variants off,dense,hybrid,full --json data/rag_ab.json
"""

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, replace

from sensai.config import load_config
from sensai.context import ContextBuilder
from sensai.memory.store import Message
from sensai.ollama_client import ChatStats, OllamaClient
from sensai.rag import Rag

SYSTEM = (
    "Tu es Sensai, un tuteur de langues. Réponds brièvement, en français. "
    "Si tu ne connais pas une information sur le cours de l'apprenant, dis-le."
)

# (question, keyword groups: every group needs one hit, control question?)
QUESTIONS = [
    ("Quel jour, à quelle heure et dans quelle salle a lieu l'interro n°2 de japonais ?", [["b204"], ["14 h", "14h"]], False),
    ("Sur quelles leçons porte l'interro n°2 de japonais ?", [["animaux"], ["compteur"]], False),
    ("Les particules wa et ga tombent-elles à l'interro n°2 ?", [["novembre", "interro n°3", "interro 3"]], False),
    ("À l'interro n°2, peut-on répondre en rōmaji ?", [["moitié"]], False),
    ("Combien de points vaut la partie kana de l'interro n°2 ?", [["4 points", "4/20", "quatre points"]], False),
    ("Quand les absents justifiés repassent-ils l'interro n°2 ?", [["20 octobre"]], False),
    ("Quelle est la phrase modèle de la leçon 3 sur les animaux ?", [["neko ga suki"]], False),
    ("Dans ma leçon de coréen, comment se présente Camille ?", [["카미유", "kamiyu"]], False),
    ("Comment dit-on « chien » en japonais ?", [["inu", "いぬ", "犬"]], True),
    ("Que veut dire « arigatou gozaimasu » ?", [["merci"]], True),
]

VARIANTS = {
    "off": None,
    "dense": {"rag_hybrid": False, "rag_rerank": False},
    "hybrid": {"rag_hybrid": True, "rag_rerank": False},
    "full": {"rag_hybrid": True, "rag_rerank": True},
}


def correct(answer: str, groups: list[list[str]]) -> bool:
    text = answer.lower()
    return all(any(re.search(rf"(?<!\w){re.escape(k.lower())}(?!\w)", text) for k in group) for group in groups)


@dataclass
class Result:
    variant: str
    course_ok: int
    course_total: int
    control_ok: int
    control_total: int
    avg_prompt_tokens: float
    avg_seconds: float
    details: list[dict]


def run(variant: str, base, client: OllamaClient) -> Result:
    print(f"[{variant}]")
    settings = VARIANTS[variant]
    rag = None
    if settings is not None:
        rag = Rag(replace(base, **settings), client)
        rag.sync()
    builder = ContextBuilder(SYSTEM, base.num_ctx, base.response_reserve, retriever=rag, rag_share=base.rag_share)
    details, tokens, seconds = [], [], []
    for i, (question, groups, control) in enumerate(QUESTIONS):
        start = time.perf_counter()  # retrieval (embedding, rerank call) included
        messages, report = builder.build({}, [Message("user", question, id=i + 1)])
        stats = ChatStats()
        answer = client.chat(messages, stats, options={"temperature": 0}, purpose="benchmark")
        seconds.append(time.perf_counter() - start)
        tokens.append(stats.prompt_eval_count or report.used)
        sources = sorted({h.chunk.source.rsplit("/", 1)[-1] for h in rag.last_hits if h.kept}) if rag else []
        ok = correct(answer, groups)
        details.append({"question": question, "control": control, "ok": ok, "sources": sources,
                        "answer": answer.strip()[:300]})
        print(f"  {'✔' if ok else '✘'} {question}  {', '.join(sources)}")
    if rag:
        rag.store.close()
    course = [d for d in details if not d["control"]]
    controls = [d for d in details if d["control"]]
    return Result(
        variant, sum(d["ok"] for d in course), len(course), sum(d["ok"] for d in controls), len(controls),
        sum(tokens) / len(tokens), sum(seconds) / len(seconds), details,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="chat model (default: config.json)")
    p.add_argument("--variants", default="off,full", help=f"comma-separated, among {','.join(VARIANTS)}")
    p.add_argument("--json", help="write raw results to this file")
    args = p.parse_args()

    argv = ["--model", args.model] if args.model else []
    base = replace(load_config(argv), rag_profile_filters={})  # the Korean lesson must stay reachable
    client = OllamaClient(base.host, base.model, base.num_ctx, base.think)
    client.check()
    results = [run(v, base, client) for v in args.variants.split(",")]

    print(f"\nModèle {base.model}, {len(QUESTIONS)} questions, embeddings {base.rag_embed_model}\n")
    print("| Variante | Questions sur le cours | Questions de contrôle | Tokens de prompt (moy.) | Temps par question |")
    print("|---|---|---|---|---|")
    for r in results:
        print(f"| {r.variant} | {r.course_ok}/{r.course_total} ({100 * r.course_ok / r.course_total:.0f} %) | "
              f"{r.control_ok}/{r.control_total} | {r.avg_prompt_tokens:.0f} | {r.avg_seconds:.1f} s |")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
