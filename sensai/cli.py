"""Interactive CLI loop."""

import re
import sqlite3
import sys
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime

from . import features
from .agent.approval import ApprovalGate, ConsoleApprover
from .agent.permissions import FileGuard
from .agent.react import AgentResult, ReActAgent, Step
from .agent.sandbox import Sandbox
from .agent.tools import ToolRegistry
from .compare import compare, kept_sources
from .config import Config, ConfigError, load_config
from .context import ContextBuilder, ContextReport
from .memory.compressor import CompressionError, Compressor, Summary, compress
from .memory.extractor import ExtractionError, MemoryExtractor
from .memory.learner import LearnerMemory
from .memory.store import MemoryStore, Message
from .memory.tokens import estimate_message
from .monitoring import compute_metrics, load_events, render_metrics
from .observability import EventLogger, new_interaction_id
from .ollama_client import ChatStats, ModelNotFound, OllamaClient, OllamaError, OllamaUnavailable, Placement
from .rag import Rag

HELP = """Commandes :
  /help                      cette aide
  /new                       nouvelle session
  /sessions                  lister les sessions
  /load <id>                 reprendre une session
  /profile                   afficher le profil
  /profile set <clé> <val>   définir une info (ex. /profile set langue_cible japonais)
  /profile del <clé>         supprimer une info
  /context                   détail du dernier contexte envoyé
  /summary                   afficher le résumé des anciens échanges
  /compress                  forcer la compression des anciens échanges
  /vocab                     vocabulaire appris (révision espacée)
  /vocab del <mot>           oublier un mot
  /mistakes                  erreurs récurrentes
  /memory                    bloc mémoire injecté au dernier message
  /agent <tâche>             mode agent : l'agent lit/écrit dans le workspace et exécute du code
                             en sandbox, étape par étape (confirmation avant toute modification)
  /rag                       état du RAG : index, filtres actifs, seuils, dernière recherche
  /rag search <question>     recherche avec le score de chaque morceau (cosinus, BM25, rerank) et la raison
                             de chaque rejet ; sert à régler les seuils
  /rag filter <k=v> ...      filtrer par métadonnées : niveau=A1  langue=japonais  date>=2026-09-01
                             (a,b = l'un ou l'autre) ; /rag filter clear pour retirer les filtres
  /rag index                 resynchroniser l'index (seuls les fichiers modifiés sont recalculés)
  /users                     lister les apprenants
  /user <nom>                passer à un autre apprenant (créé s'il n'existe pas) ; reprend sa dernière session
  /features                  fonctionnalités activables (historique, profil, mémoire, résumé, RAG…)
  /feature <id> on|off       couper ou rétablir une fonctionnalité ; /feature reset pour revenir à config.json
  /compare <id> <question>   poser la même question avec et sans la fonctionnalité, côte à côte
                             (rien n'est enregistré)
  /trace                     étapes de la dernière tâche agent
  /tools                     outils de l'agent, permissions et limites
  /metrics                   métriques du système (latence, tokens, erreurs, outils…)
  /quit                      quitter"""


USER_NAME = re.compile(r"^[\w-]{1,40}$")


class Chat:
    def __init__(
        self,
        config: Config,
        store: MemoryStore,
        client: OllamaClient,
        logger: EventLogger | None = None,
        agent: ReActAgent | None = None,
        rag: Rag | None = None,
    ):
        self.config = config
        self.logger = logger or client.logger
        self.agent = agent
        self.rag = rag
        self.last_agent: AgentResult | None = None
        self.store = store
        self.client = client
        self.context = ContextBuilder(
            config.system_prompt,
            config.num_ctx,
            config.response_reserve,
            retriever=rag,
            rag_share=config.rag_share,
            summary_share=config.summary_share,
            memory_share=config.memory_share,
        )
        self.compressor = Compressor(client, config.compress_trigger, config.compress_keep)
        # Runtime switches (features.py): each one gates a source of context at every turn.
        self.switches = features.defaults(config, rag is not None)
        self._apply_switches()
        store.add_user(config.user)
        self.session_id = store.create_session(config.user)
        self.history: list[Message] = []  # unsummarized turns only
        self.summary: Summary | None = None
        self.compress_backoff = 0  # turns to wait after a failed compression
        self.learner = LearnerMemory(store.conn)
        self.extractor = MemoryExtractor(client)
        # Extraction runs in the background while the learner reads the answer;
        # only the model call happens there, results are applied on this thread.
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending: Future | None = None
        self.last_memory: str | None = None
        self.last_report = None
        self.last_stats = ChatStats()  # real token counts of the last answer, from Ollama
        self.interrupted = False  # was the last answer cut short?
        self.last_sources: list[dict] = []  # lesson chunks used by the last answer
        self.last_off: list[str] = []  # features switched off for the last answer
        # Side messages ([mémoire : …], [compression …]): stderr in the terminal, events in the web UI.
        self.notify: Callable[[str], None] = lambda message: print(message, file=sys.stderr)

    @property
    def language(self) -> str:
        return self.store.get_profile(self.config.user).get("langue_cible", "")

    # --- feature switches (features.py) -----------------------------------

    def feature_available(self) -> dict[str, bool]:
        return {f.id: self.rag is not None or f.group != "RAG" for f in features.FEATURES}

    def set_feature(self, feature_id: str, enabled: bool) -> None:
        feature = features.check(feature_id)
        if enabled and not self.feature_available()[feature_id]:
            raise features.FeatureError(f"{feature.label} : indisponible (RAG désactivé au démarrage)")
        self.switches[feature_id] = enabled
        self._apply_switches()
        self.logger.log("feature_switch", feature=feature_id, enabled=enabled)

    def reset_features(self) -> None:
        self.switches = features.defaults(self.config, self.rag is not None)
        self._apply_switches()
        self.logger.log("feature_switch", feature="*", enabled=None)

    def _apply_switches(self) -> None:
        sw = self.switches
        self.context.retriever = self.rag if sw["rag"] else None  # its share of the budget goes back to history
        if self.rag is not None:
            self.rag.hybrid, self.rag.rerank, self.rag.profile_filters = sw["rag_hybrid"], sw["rag_rerank"], sw["rag_filters"]

    def prompt_profile(self) -> dict[str, str]:
        return self.store.get_profile(self.config.user) if self.switches["profile"] else {}

    # --- learners (M1) ------------------------------------------------------

    def switch_user(self, name: str) -> bool:
        """Become learner `name` (created if new) and resume its latest session; True if created."""
        if not USER_NAME.match(name):
            raise ValueError("nom d'apprenant : 1 à 40 lettres, chiffres, - ou _")
        self.apply_pending()  # the last exchange still belongs to the previous learner
        created = self.store.add_user(name)
        self.config.user = name
        latest = self.store.latest_session(name)
        if latest is None or not self.load_session(latest):
            self.new_session()
        self.last_report, self.last_memory, self.last_sources, self.last_off = None, None, [], []
        self.logger.log("user_switch", user=name, created=created)
        return created

    def delete_user(self, name: str) -> bool:
        """Forget a learner entirely; never the current one."""
        if name == self.config.user:
            raise ValueError("impossible de supprimer l'apprenant en cours : change d'apprenant d'abord")
        self.learner.delete_user(name)
        return self.store.delete_user(name)

    def close(self) -> None:
        self.apply_pending()
        self.executor.shutdown()
        if self.rag is not None:
            self.rag.store.close()

    def new_session(self) -> None:
        self.session_id = self.store.create_session(self.config.user)
        self.history = []
        self.summary = None
        self.compress_backoff = 0

    def load_session(self, session_id: int) -> bool:
        """M1: resume a session of this learner: its summary, then only the messages after it."""
        if not self.store.session_exists(self.config.user, session_id):
            return False
        self.session_id = session_id
        self.compress_backoff = 0
        saved = self.store.get_summary(session_id)
        self.summary = Summary.from_json(saved[0]) if saved else None
        self.history = self.store.load_messages(session_id, after_id=saved[1] if saved else 0)
        return True

    def compress_now(self) -> bool:
        """M2: summarize every turn but the last one, even far from the threshold."""
        return self._compress(force=True)

    def handle_command(self, line: str) -> bool:
        """Return False to quit."""
        parts = line.split(maxsplit=3)
        cmd = parts[0]
        user = self.config.user
        if cmd in ("/quit", "/exit"):
            return False
        if cmd == "/help":
            print(HELP)
        elif cmd == "/new":
            self.new_session()
            print(f"Nouvelle session #{self.session_id}")
        elif cmd == "/sessions":
            for s in self.store.list_sessions(user):
                when = datetime.fromtimestamp(s.updated_at).strftime("%Y-%m-%d %H:%M")
                mark = "*" if s.id == self.session_id else " "
                print(f"{mark} #{s.id:<4} {when}  {s.message_count:>3} msg  {s.title or '(vide)'}")
        elif cmd == "/load":
            if len(parts) < 2 or not parts[1].isdigit():
                print("Usage : /load <id>")
            elif not self.load_session(int(parts[1])):
                print(f"Session #{parts[1]} introuvable")
            else:
                total = self.store.count_messages(self.session_id)
                print(
                    f"Session #{self.session_id} rechargée ({total} messages, "
                    f"dont {total - len(self.history)} résumés)"
                )
        elif cmd == "/profile":
            self._profile(parts[1:])
        elif cmd == "/context":
            self._print_report()
        elif cmd == "/summary":
            print(self.summary.render() if self.summary else "Aucun résumé pour cette session.")
        elif cmd == "/compress":
            if not self.compress_now():
                print("Rien à compresser (il faut au moins 2 messages en dehors des plus récents).")
        elif cmd == "/vocab":
            self.apply_pending()
            if len(parts) >= 3 and parts[1] == "del":
                word = line.split(maxsplit=2)[2]
                print("Oublié" if self.learner.delete_word(user, self.language, word) else "Mot absent")
            else:
                words = self.learner.words(user, self.language)
                if not words:
                    print("Aucun mot enregistré pour l'instant.")
                for w in words:
                    when = datetime.fromtimestamp(w.next_review).strftime("%Y-%m-%d")
                    print(f"  [boîte {w.box}] {w.line()}  — révision {when}")
        elif cmd == "/mistakes":
            self.apply_pending()
            mistakes = self.learner.mistakes(user, self.language, limit=50)
            if not mistakes:
                print("Aucune erreur enregistrée.")
            for m in mistakes:
                print(f"  {m.line()}")
        elif cmd == "/memory":
            print(self.last_memory or "Aucun bloc mémoire injecté au dernier message.")
        elif cmd == "/agent":
            task = line[len("/agent"):].strip()
            if not task:
                print("Usage : /agent <tâche>   ex. /agent crée une fiche de révision à partir de docs/lecon.md")
            else:
                self.run_agent(task)
        elif cmd == "/rag":
            self._rag(line)
        elif cmd == "/users":
            for u in self.store.list_users():
                mark = "*" if u.name == user else " "
                when = datetime.fromtimestamp(u.last_active).strftime("%Y-%m-%d %H:%M") if u.last_active else "jamais"
                print(f"{mark} {u.name:<20} {u.sessions:>3} sessions  {u.messages:>4} msg  dernière activité : {when}")
        elif cmd == "/user":
            if len(parts) != 2:
                print("Usage : /user <nom>")
            else:
                try:
                    created = self.switch_user(parts[1])
                except ValueError as exc:
                    print(exc)
                else:
                    total = self.store.count_messages(self.session_id)
                    print(f"Apprenant « {parts[1]} »{' créé' if created else ''} — session #{self.session_id} ({total} messages)")
        elif cmd in ("/features", "/feature"):
            self._features(parts[1:])
        elif cmd == "/compare":
            self._compare(line)
        elif cmd == "/trace":
            print(self.last_agent.trace() if self.last_agent else "Aucune tâche agent pour l'instant.")
        elif cmd == "/tools":
            if self.agent is None:
                print("Mode agent indisponible.")
            else:
                print(self.agent.tools.describe())
                print(self.agent.tools.guard.describe())
                print(self.agent.tools.sandbox.describe())
                levels = ", ".join(sorted(self.agent.tools.gate.levels))
                print(f"Confirmation demandée pour : {levels}. Étapes max : {self.agent.max_steps}.")
        elif cmd == "/metrics":
            print(render_metrics(compute_metrics(load_events(self.config.log_path))))
        else:
            print(f"Commande inconnue : {cmd} (voir /help)")
        return True

    def _profile(self, args: list[str]) -> None:
        user = self.config.user
        if not args:
            profile = self.store.get_profile(user)
            if not profile:
                print("Profil vide. Ex. : /profile set langue_cible espagnol")
            for k, v in profile.items():
                print(f"  {k}: {v}")
        elif args[0] == "set" and len(args) == 3:
            self.store.set_profile(user, args[1], args[2])
            print(f"{args[1]} = {args[2]}")
        elif args[0] == "del" and len(args) == 2:
            print("Supprimé" if self.store.delete_profile_key(user, args[1]) else "Clé absente")
        else:
            print("Usage : /profile | /profile set <clé> <valeur> | /profile del <clé>")

    def _features(self, args: list[str]) -> None:
        if args == ["reset"]:
            self.reset_features()
        elif len(args) == 2 and args[1] in ("on", "off"):
            try:
                self.set_feature(args[0], args[1] == "on")
            except ValueError as exc:
                print(exc)
                return
        elif args:
            print("Usage : /features | /feature <id> on|off | /feature reset")
            return
        available = self.feature_available()
        for f in features.FEATURES:
            state = "on " if self.switches[f.id] else "off"
            if not available[f.id]:
                state = "n/a"
            print(f"  [{state}] {f.id:<12} {f.code}  {f.label} — {f.does}")

    def _compare(self, line: str) -> None:
        args = line.split(maxsplit=2)[1:]
        if len(args) < 2:
            print("Usage : /compare <id> <question>   ex. /compare rag Qu'est-ce qui tombe à l'interro de vendredi ?")
            return
        try:
            for event in compare(self, args[1], args[0]):
                kind = event["type"]
                if kind == "side":
                    print(f"\n── {event['label']} " + "─" * 30)
                elif kind == "token":
                    print(event["text"], end="", flush=True)
                elif kind == "side_done":
                    srcs = ", ".join(f"{s['source'].split('/')[-1]}#{s['part']}" for s in event["sources"]) or "aucune leçon"
                    print(f"\n[{event['context']['used']} tokens de prompt — {srcs} — {event['duration_ms'] / 1000:.1f} s]")
                elif kind == "verdict":
                    print(f"\n── Arbitrage du modèle ──\n{event['on_label']} : {event['on']['score']}/10 — {event['on']['reason']}")
                    print(f"{event['off_label']} : {event['off']['score']}/10 — {event['off']['reason']}")
                    print(event["verdict"])
                elif kind == "notice":
                    print(event["text"])
        except KeyboardInterrupt:
            print("\n[comparaison interrompue]")
        except (ValueError, OllamaError) as exc:
            print(f"Erreur : {exc}")

    def _rag(self, line: str) -> None:
        if self.rag is None:
            print("RAG indisponible (désactivé dans config.json ou erreur au démarrage).")
            return
        args = line.split(maxsplit=2)[1:]
        sub = args[0] if args else ""
        rest = args[1] if len(args) > 1 else ""
        if not sub:
            print(self.rag.describe())
        elif sub == "search":
            if not rest:
                print("Usage : /rag search <question>")
                return
            budget = int(self.context.budget * self.context.rag_share)
            try:
                self.rag.search(rest, budget)
            except OllamaError as exc:
                print(f"Erreur : {exc}", file=sys.stderr)
                return
            print(self.rag.render_hits())
        elif sub == "filter":
            if rest.strip() in ("clear", ""):
                self.rag.filters = []
                print("Filtres retirés." if rest.strip() else "Usage : /rag filter <clé=valeur> ... | /rag filter clear")
                return
            try:
                self.rag.set_filters(rest.split())
            except ValueError as exc:
                print(exc)
                return
            print("Filtres actifs : " + ", ".join(str(f) for f in self.rag.active_filters()))
        elif sub == "index":
            try:
                print(self.rag.sync().describe())
            except OllamaError as exc:
                print(f"Erreur : {exc}", file=sys.stderr)
        else:
            print("Usage : /rag | /rag search <question> | /rag filter <clé=valeur>... | /rag filter clear | /rag index")

    def _print_report(self) -> None:
        r = self.last_report
        if r is None:
            print("Aucun message envoyé pour l'instant.")
            return
        off = features.disabled(self.switches, self.feature_available())
        if off:
            print("Fonctionnalités coupées : " + ", ".join(features.BY_ID[i].label for i in off))
        print(
            f"Budget {r.used}/{r.budget} tokens (estimés) — système+profil {r.system_tokens}, "
            f"résumé {r.summary_tokens}/{self.context.summary_cap}, RAG {r.rag_tokens}, "
            f"historique {r.history_tokens}/{self.context.history_budget(self.prompt_profile())} "
            f"({r.history_kept} messages gardés, {r.history_dropped} écartés)"
        )
        print(f"Mémoire de l'apprenant {r.memory_tokens}/{self.context.memory_cap}")
        if self.rag is not None and self.rag.last_query:
            kept = sum(h.kept for h in self.rag.last_hits)
            print(f"RAG {r.rag_tokens}/{int(self.context.budget * self.context.rag_share)} tokens : {kept} morceau(x) retenu(s) — détail avec /rag")
        o = self.learner.op_stats(self.config.user)
        if o["total"]:
            print(
                f"Opérations mémoire : {o['applied']}/{o['total']} appliquées, "
                f"{o['failed_extractions']} extractions échouées"
            )
        s = self.store.compression_stats(self.session_id)
        if s["passes"]:
            print(
                f"Compressions : {s['ok']}/{s['passes']} réussies, {s['messages']} messages résumés, "
                f"~{s['saved']} tokens économisés, {s['avg_ms']:.0f} ms en moyenne"
            )

    def _compress(self, force: bool = False) -> bool:
        """Summarize old turns if needed; return True if something was compressed."""
        available = self.context.history_budget(self.prompt_profile())
        done = False
        while chunk := self.compressor.select(self.history, available, force=force and not done):
            self.notify(f"[compression de {len(chunk)} anciens messages…]")
            try:
                result = compress(self.compressor, chunk, self.summary, self.context.summary_cap)
            except CompressionError as exc:
                self.store.log_compression(self.session_id, len(chunk), 0, 0, 0, error=str(exc))
                self.compress_backoff = 3
                self.notify(f"[compression échouée : {exc} — les plus vieux messages seront écartés]")
                return done
            self.summary = result.summary
            self.store.set_summary(self.session_id, result.summary.to_json(), result.covered_until, result.tokens_after)
            self.store.log_compression(
                self.session_id, result.messages_in, result.tokens_before, result.tokens_after, result.duration_ms
            )
            self.history = self.history[len(chunk):]
            done = True
            self.notify(
                f"[{result.tokens_before} → {result.tokens_after} tokens en {result.duration_ms / 1000:.1f} s"
                + (f", {result.dropped_items} éléments anciens retirés du résumé" if result.dropped_items else "")
                + "]"
            )
        return done

    def apply_pending(self) -> None:
        """Wait for the background extraction of the previous exchange and apply it."""
        if self.pending is None:
            return
        future, self.pending = self.pending, None
        user = self.config.user
        try:
            ops = future.result()
        except ExtractionError as exc:
            self.learner.log_failure(user, self.session_id, str(exc))
            self.notify(f"[mémoire : extraction échouée ({exc})]")
            return
        if not ops:
            return
        report = self.learner.apply(user, self.language, ops, self.store.set_profile, self.session_id)
        self.notify(f"[mémoire : {report.short()}]")

    def _start_extraction(self, user_text: str, answer: str) -> None:
        user = self.config.user
        language = self.language
        profile = self.store.get_profile(user)
        exchange = f"{user_text}\n{answer}"
        known_words = self.learner.mentioned_words(user, language, exchange, limit=20)
        known_mistakes = self.learner.mistakes(user, language, limit=10)
        self.pending = self.executor.submit(
            self.extractor.extract, user_text, answer, profile, known_words, known_mistakes
        )

    def run_agent(self, task: str) -> AgentResult | None:
        """CLI: run an agent task and print its outcome."""
        if self.agent is None:
            print("Mode agent indisponible.")
            return None
        print(f"[agent — {self.agent.max_steps} étapes max, Ctrl-C pour interrompre]")
        try:
            result = self.agent_turn(task)
        except KeyboardInterrupt:
            print("\n[agent interrompu]")
            return None
        label = {"done": "terminé", "max_steps": "limite d'étapes atteinte", "error": "arrêté sur erreur"}[result.outcome]
        print(f"\n[agent {label} — {len(result.steps)} étapes, {result.duration_ms / 1000:.1f} s, /trace pour le détail]")
        print(result.answer)
        return result

    def agent_turn(self, task: str) -> AgentResult:
        """A1: run the ReAct loop on `task`; the exchange is kept in the chat history."""
        self.apply_pending()
        profile = self.store.get_profile(self.config.user)
        context = ("Profil de l'apprenant : " + ", ".join(f"{k} = {v}" for k, v in profile.items())) if profile else None
        result = self.agent.run(task, context)
        self.last_agent = result
        for role, content in (("user", f"[tâche agent] {task}"), ("assistant", result.answer)):
            msg = Message(role, content, estimate_message(content))
            self.history.append(msg)
            self.store.add_message(self.session_id, msg)
        return result

    def send(self, text: str) -> None:
        """CLI: one chat turn, printed as it is generated (Ctrl+C keeps the part already received)."""
        turn = self.stream(text)
        try:
            for piece in turn:
                print(piece, end="", flush=True)
        except KeyboardInterrupt:
            turn.close()
        finally:
            print()
        if self.interrupted:
            print("[génération interrompue]")

    def stream(self, text: str) -> Iterator[str]:
        """B1: one chat turn; yields the answer piece by piece as Ollama generates it.

        Closing the generator stops the generation: the part already received is
        kept and memorized like a complete answer."""
        interaction_id = new_interaction_id()
        with self.logger.timed("chat_turn", interaction_id, session_id=self.session_id, input_chars=len(text)) as ev:
            messages = self._prepare(text)
            self.interrupted = False
            self.last_stats = stats = ChatStats()
            parts: list[str] = []
            pieces = self.client.chat_stream(messages, stats, interaction_id=interaction_id)
            try:
                for piece in pieces:
                    parts.append(piece)
                    yield piece
            except (GeneratorExit, KeyboardInterrupt):
                self.interrupted = True
                ev["status"] = "interrupted"
            finally:
                pieces.close()
            self._finish(text, "".join(parts), stats, ev)

    def _prepare(self, text: str) -> list[dict]:
        """Store the learner's message, compress old turns if needed, build the prompt within the budget."""
        self.apply_pending()
        user_msg = Message("user", text, estimate_message(text))
        self.history.append(user_msg)
        self.store.add_message(self.session_id, user_msg)

        if self.switches["compression"] and self.switches["history"]:
            if self.compress_backoff:
                self.compress_backoff -= 1
            else:
                self._compress()
        messages, self.last_report, self.last_memory = self.build_prompt(text, self.history)
        self.last_sources = kept_sources(self)
        self.last_off = features.disabled(self.switches, self.feature_available())
        if self.last_report.history_dropped:
            self.notify(f"[{self.last_report.history_dropped} anciens messages hors contexte]")
        return messages

    def build_prompt(self, text: str, history: list[Message]) -> tuple[list[dict], ContextReport, str | None]:
        """The prompt for `text` (the last message of `history`), with the switches as they are now.

        Reads the database and runs the retrieval, but stores nothing: compare.py
        calls it twice for the same message."""
        sw = self.switches
        summary = None
        if sw["history"] and sw["compression"] and self.summary and not self.summary.is_empty():
            summary = self.summary.render()
        memory = (
            self.learner.render(self.config.user, self.language, text, self.context.memory_cap)
            if sw["memory"]
            else None
        )
        turns = history if sw["history"] else history[-1:]
        messages, report = self.context.build(self.prompt_profile(), turns, summary, memory)
        return messages, report, memory

    def turn_meta(self) -> dict | None:
        """Stored with an answer: what was switched off, which lesson chunks were used."""
        meta: dict = {}
        if self.last_off:
            meta["off"] = self.last_off
        if self.last_sources:
            meta["sources"] = [{"source": s["source"], "part": s["part"]} for s in self.last_sources]
        return meta or None

    def _finish(self, text: str, answer: str, stats: ChatStats, ev: dict) -> None:
        """Keep the answer, even a partial one, and start the memory extraction in the background."""
        ev.update(
            context_tokens=self.last_report.used,
            history_dropped=self.last_report.history_dropped,
            output_chars=len(answer),
        )
        if answer:
            msg = Message("assistant", answer, stats.eval_count or estimate_message(answer), meta=self.turn_meta())
            self.history.append(msg)
            self.store.add_message(self.session_id, msg)
            if self.switches["memory"]:
                self._start_extraction(text, answer)


def print_step(step: Step) -> None:
    print(step.line(), flush=True)
    error = step.observation.get("error")
    if error and step.status != "ok":
        print(f"           {error}")
    elif "outcome" in step.observation:
        out = (step.observation.get("stdout") or step.observation.get("stderr") or "").strip().splitlines()
        tail = f" — {out[-1][:100]}" if out else ""
        print(f"           sandbox : {step.observation['outcome']}, code {step.observation.get('exit_code')}{tail}")


def build_agent(config: Config, client: OllamaClient, logger: EventLogger, budget: int) -> ReActAgent:
    """Wire the ReAct loop to its tools: permissions (T4), sandbox (T2), confirmation (A3), logs (EV3)."""
    guard = FileGuard(
        config.file_roots,
        allowed_ext=config.file_allowed_ext,
        denied_ext=config.file_denied_ext,
        max_bytes=config.file_max_bytes,
        allow_symlinks=config.file_allow_symlinks,
        logger=logger,
    )
    sandbox = Sandbox(
        runtimes=config.sandbox_runtimes,
        timeout=config.sandbox_timeout,
        max_output=config.sandbox_max_output,
        memory_mb=config.sandbox_memory_mb,
        backend=config.sandbox_backend,
        docker_image=config.sandbox_docker_image,
        logger=logger,
    )
    gate = ApprovalGate(ConsoleApprover(timeout=config.confirm_timeout), config.confirm_actions, logger)
    return ReActAgent(
        client,
        ToolRegistry(guard, sandbox, gate, logger),
        logger,
        max_steps=config.agent_max_steps,
        max_parse_errors=config.agent_max_parse_errors,
        observation_chars=config.agent_observation_chars,
        token_budget=budget,
        on_step=print_step,
    )


def build_rag(config: Config, client: OllamaClient, logger: EventLogger, store: MemoryStore) -> Rag | None:
    """R1/R2. RAG is optional: any problem disables it instead of blocking the chat."""
    if not config.rag:
        return None
    try:
        rag = Rag(config, client, logger, profile_fn=lambda: store.get_profile(config.user))
        rag.check()
        if config.rag_auto_index:
            print("Indexation des documents…")
            print(f"RAG : {rag.sync().describe()}")
    except (OllamaError, OSError, sqlite3.Error) as exc:
        print(f"RAG désactivé : {exc}", file=sys.stderr)
        return None
    return rag


def load_model(client: OllamaClient) -> Placement | None:
    """Load the model at startup rather than at the first message, and say where it went.

    GPU or CPU is Ollama's choice: it uses the GPU whenever its build supports the card,
    for as much of the model as fits, and the CPU for the rest."""
    print("Chargement du modèle…", flush=True)
    try:
        placement = client.load()
    except OllamaError as exc:
        print(f"[chargement anticipé impossible ({exc}) : le modèle sera chargé au premier message]", file=sys.stderr)
        return None
    print(f"Modèle {client.model} chargé : {placement.describe()}")
    return placement


def run(config: Config) -> int:
    logger = EventLogger(config.log_path, config.log_content_chars, config.log_max_bytes)
    client = OllamaClient(config.host, config.model, config.num_ctx, config.think, logger=logger, keep_alive=config.keep_alive)
    try:
        client.check()
    except OllamaError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    load_model(client)

    store = MemoryStore(config.db_path)
    chat = Chat(config, store, client, logger, rag=build_rag(config, client, logger, store))
    try:
        chat.agent = build_agent(config, client, logger, chat.context.budget)
    except (ValueError, OSError) as exc:
        print(f"Mode agent désactivé : {exc}", file=sys.stderr)
    profile = store.get_profile(config.user)
    print(f"Sensai — modèle {config.model}, apprenant « {config.user} », session #{chat.session_id}")
    if not profile:
        print("Astuce : renseigne ton profil, ex. /profile set langue_cible japonais")
    print("/help pour les commandes.\n")

    try:
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line.startswith("/"):
                if not chat.handle_command(line):
                    break
                continue
            try:
                chat.send(line)
            except ModelNotFound as exc:
                print(f"Erreur : {exc}", file=sys.stderr)
            except OllamaUnavailable as exc:
                print(f"Erreur : {exc} — réessaie quand Ollama est relancé.", file=sys.stderr)
            except OllamaError as exc:
                print(f"Erreur du modèle : {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        print()
    finally:
        chat.close()
        store.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        config = load_config(argv)
    except ConfigError as exc:
        print(f"Erreur de configuration : {exc}", file=sys.stderr)
        return 2
    return run(config)
