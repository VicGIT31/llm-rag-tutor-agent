"""Check, before a demo, that everything the keynote relies on works on this machine.

    python -m sensai.preflight            # checks only (the RAG index is brought up to date, like at startup)
    python -m sensai.preflight --setup    # also creates the demo learners (camille, hugo)

A local model is slow and the demo runs live: a model left on the CPU, an
index out of date or a learner without a profile is a preparation failure,
not a technical one. Each check prints ✓, ⚠ (works, but read the advice)
or ✗ (fix it before going on stage), and the exit code is 1 if any check failed.
Loading the model and answering one message also warms Ollama up.
"""

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

from .agent.sandbox import Sandbox
from .config import Config, ConfigError, load_config
from .memory.store import MemoryStore
from .ollama_client import ChatStats, OllamaClient, OllamaError
from .rag import Rag

# The questions of the demo (docs/demo.md) and the document each one must retrieve.
DEMO_SEARCHES = [
    ("Qu'est-ce qui tombe à l'interro de vendredi ?", "interro_vendredi.md"),
    ("Combien coûte le voyage à Kyoto et avant quand verser l'acompte ?", "voyage_kyoto.md"),
    ("Comment on compte des billets de train ?", "compteurs.md"),
    ("Quelle est la différence entre wa et ga ?", "particules_wa_ga.md"),
]
DEMO_LEARNERS = {
    "camille": {"langue_cible": "japonais", "niveau": "A1"},
    "hugo": {"langue_cible": "coréen", "niveau": "A1"},
}
SLOW_ANSWER_S = 30  # beyond this, a live demo needs its recordings ready
LOW_GPU_SHARE = 0.2  # qwen3-coder:30b gets about a third of its layers on a free 8 GB card


@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail
    detail: str
    advice: str = ""

    def line(self) -> str:
        mark = {"ok": "✓", "warn": "⚠", "fail": "✗"}[self.status]
        text = f" {mark} {self.name} — {self.detail}"
        return text + (f"\n     → {self.advice}" if self.advice else "")


def check_models(client: OllamaClient, config: Config) -> list[Check]:
    try:
        client.check()
    except OllamaError as exc:
        return [Check("Ollama", "fail", str(exc), "lancer `ollama serve`, puis `ollama pull` du modèle manquant")]
    checks = [Check("Ollama", "ok", f"joignable, modèle {client.model} présent")]
    try:
        placement = client.load()
    except OllamaError as exc:
        return checks + [Check("Chargement du modèle", "fail", str(exc))]
    if not placement.vram:
        checks.append(Check(
            "Placement", "warn", placement.describe(),
            'sans GPU, une réponse prend 30 à 45 s : mettre "rag_rerank": false et "rag_min_cosine": 0.50, '
            "ou installer ollama-vulkan / ollama-cuda",
        ))
    elif placement.gpu_share < LOW_GPU_SHARE:
        # Ollama places a model once, when it loads it: if another program held the card at that moment,
        # the model keeps few layers on the GPU even after the card is freed.
        checks.append(Check(
            "Placement", "warn", placement.describe(),
            "peu de couches sur le GPU : fermer les applications qui occupent la carte graphique (jeux, navigateur), "
            "puis redémarrer Ollama pour qu'il recharge le modèle",
        ))
    else:
        checks.append(Check("Placement", "ok", placement.describe()))
    if config.keep_alive is None:
        checks.append(Check("keep_alive", "warn", "défaut d'Ollama (5 min)",
                            'mettre "keep_alive": "30m" : sinon le modèle se décharge entre deux actes'))
    return checks


def check_rag(rag: Rag | None, searches: list[tuple[str, str]] = DEMO_SEARCHES) -> list[Check]:
    if rag is None:
        return [Check("RAG", "warn", "désactivé dans la configuration", "l'acte 2 de la démo en a besoin")]
    try:
        rag.check()
        report = rag.sync()
    except OllamaError as exc:
        return [Check("RAG", "fail", str(exc), f"`ollama pull {rag.config.rag_embed_model}`")]
    checks = [Check("Index RAG", "warn" if report.warnings else "ok", report.describe().splitlines()[0],
                    "; ".join(report.warnings))]
    for question, expected in searches:
        start = time.perf_counter()
        try:
            hits = rag.search(question)
        except OllamaError as exc:
            checks.append(Check(f"Recherche « {question} »", "fail", str(exc)))
            continue
        seconds = time.perf_counter() - start
        kept = [h for h in hits if h.kept]
        docs = sorted({h.chunk.source.rsplit("/", 1)[-1] for h in kept})
        found = expected in docs
        checks.append(Check(
            f"Recherche « {question} »", "ok" if found else "fail",
            f"{', '.join(docs) or 'aucun morceau'} en {seconds:.1f} s",
            "" if found else f"{expected} attendu : relancer `/rag search` et revoir les seuils",
        ))
    return checks


def check_answer_time(client: OllamaClient) -> Check:
    """One short answer: warms the model up and tells how long the audience will wait."""
    start = time.perf_counter()
    try:
        client.chat([{"role": "user", "content": "Dis bonjour en japonais, en un mot."}], ChatStats(), purpose="preflight")
    except OllamaError as exc:
        return Check("Réponse du modèle", "fail", str(exc))
    seconds = time.perf_counter() - start
    if seconds > SLOW_ANSWER_S:
        return Check("Réponse du modèle", "warn", f"{seconds:.0f} s pour un mot",
                     "préparer les enregistrements de secours de chaque acte")
    return Check("Réponse du modèle", "ok", f"{seconds:.1f} s pour une réponse courte (modèle préchauffé)")


def check_sandbox(config: Config) -> Check:
    try:
        sandbox = Sandbox(config.sandbox_runtimes, config.sandbox_timeout, config.sandbox_max_output,
                          config.sandbox_memory_mb, config.sandbox_backend, config.sandbox_docker_image)
        result = sandbox.run("python", "print('ok')")
    except Exception as exc:  # any failure here means act 3 will fail on stage
        return Check("Sandbox", "fail", f"{type(exc).__name__}: {exc}")
    if not result.ok:
        return Check("Sandbox", "fail", f"{result.outcome}: {result.stderr[-200:]}")
    return Check("Sandbox", "ok", f"backend {config.sandbox_backend}, {result.duration_ms:.0f} ms")


def check_learners(store: MemoryStore, setup: bool) -> list[Check]:
    checks = []
    for name, wanted in DEMO_LEARNERS.items():
        profile = store.get_profile(name)
        missing = {k: v for k, v in wanted.items() if profile.get(k) != v}
        if missing and setup:
            store.add_user(name)
            for key, value in missing.items():
                store.set_profile(name, key, value)
            checks.append(Check(f"Apprenant {name}", "ok", "créé / complété : " + ", ".join(f"{k}={v}" for k, v in missing.items())))
        elif missing:
            checks.append(Check(f"Apprenant {name}", "warn", "profil incomplet pour la démo",
                                "`python -m sensai.preflight --setup` le crée"))
        else:
            checks.append(Check(f"Apprenant {name}", "ok", ", ".join(f"{k}={v}" for k, v in profile.items())))
    return checks


def run(config: Config, setup: bool = False, out: Callable[[str], None] = print,
        client: OllamaClient | None = None, rag: Rag | None = None) -> list[Check]:
    client = client or OllamaClient(config.host, config.model, config.num_ctx, config.think, keep_alive=config.keep_alive)
    checks: list[Check] = []

    def add(new: list[Check]) -> None:
        for c in new:
            out(c.line())
        checks.extend(new)

    out(f"Sensai — vérification avant démo (modèle {config.model})")
    store = MemoryStore(config.db_path)
    own_rag = rag is None and config.rag
    try:
        add(check_learners(store, setup))  # first: camille's profile sets the language filter of the searches
        models = check_models(client, config)
        add(models)
        if all(c.status != "fail" for c in models):
            add([check_answer_time(client)])
            if own_rag:
                rag = Rag(config, client, profile_fn=lambda: store.get_profile("camille"))
            add(check_rag(rag))
        add([check_sandbox(config)])
    finally:
        if own_rag and rag is not None:
            rag.store.close()
        store.close()
    failed = [c for c in checks if c.status == "fail"]
    warned = [c for c in checks if c.status == "warn"]
    out("")
    out("Prêt pour la démo." if not failed and not warned else
        f"{len(failed)} problème(s) bloquant(s), {len(warned)} avertissement(s) : voir les → ci-dessus.")
    return checks


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    setup = "--setup" in argv
    try:
        config = load_config([a for a in argv if a != "--setup"])
    except ConfigError as exc:
        print(f" ✗ Configuration — {exc}")
        return 2
    return 1 if any(c.status == "fail" for c in run(config, setup)) else 0


if __name__ == "__main__":
    sys.exit(main())
