"""Feature switches, A/B comparison with its judge, and learners with their history.

No model: a scripted fake client answers the chat, the memory extraction and the
judge; the RAG runs on the hashing embedder of test_rag.
"""

import json
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

from sensai import features
from sensai.cli import Chat
from sensai.compare import compare
from sensai.config import Config
from sensai.memory.store import MemoryStore, last_days, streak
from sensai.observability import EventLogger
from sensai.ollama_client import OllamaUnavailable
from tests.test_rag import make_rag


class FakeClient:
    """Answers per purpose; the streamed answer says which prompt it got, so sides can be told apart."""

    model = "fake"

    def __init__(self, judge=None):
        self.judge = judge  # reply of the judge, or an exception
        self.streamed: list[list[dict]] = []
        self.purposes: list[str] = []
        self.judge_prompts: list[str] = []
        self.extractions = 0

    def chat_stream(self, messages, stats, purpose="chat", interaction_id=None):
        self.streamed.append(messages)
        self.purposes.append(purpose)
        stats.eval_count = 2
        grounded = "## Documents de référence" in messages[0]["content"]
        yield "Selon ta leçon, " if grounded else "De mémoire, "
        yield "sanbon."

    def chat(self, messages, stats, schema=None, options=None, purpose="chat", interaction_id=None):
        self.purposes.append(purpose)
        if purpose == "memory_extraction":
            self.extractions += 1
            return json.dumps({"operations": [{"op": "add_word", "word": "neko", "translation": "chat"}]})
        if purpose == "judge":
            self.judge_prompts.append(messages[-1]["content"])
            if isinstance(self.judge, Exception):
                raise self.judge
            return self.judge
        if purpose == "rerank":
            n = messages[-1]["content"].count("\n[")
            return json.dumps({"scores": [{"id": i, "score": 8} for i in range(n)]})
        return json.dumps({"apprenant": [], "vocabulaire": [], "erreurs": [], "exercices": [], "resume": "r", "en_cours": ""})


def judge_reply(first: int, second: int) -> str:
    return json.dumps({
        "reponse_1": {"score": first, "raison": "première"},
        "reponse_2": {"score": second, "raison": "deuxième"},
        "verdict": "La réponse appuyée sur la leçon donne le bon compteur.",
    })


class ChatCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.client = FakeClient()

    def tearDown(self):
        self.chat.close()
        self.store.close()
        self.tmp.cleanup()

    def make_chat(self, with_rag=False, **overrides):
        values = dict(
            db_path=str(self.base / "chat.db"), log_path=str(self.base / "events.jsonl"),
            user="camille", rag=with_rag, num_ctx=4096, system_prompt="Tu es Sensai.",
        )
        values.update(overrides)
        self.config = Config(**values)
        self.store = MemoryStore(self.config.db_path)
        rag = None
        if with_rag:
            rag = make_rag(self.tmp.name, client=self.client, profile={"langue_cible": "japonais"})
            rag.sync()
        self.chat = Chat(self.config, self.store, self.client, EventLogger(self.config.log_path), rag=rag)
        self.chat.notify = lambda message: None
        self.store.set_profile("camille", "langue_cible", "japonais")
        return self.chat

    def say(self, text):
        return "".join(self.chat.stream(text))

    def system(self, index=-1) -> str:
        return self.client.streamed[index][0]["content"]


class SwitchTest(ChatCase):
    def test_defaults_come_from_the_config(self):
        chat = self.make_chat(memory=False, rag_rerank=False)
        sw = chat.switches
        self.assertEqual((sw["history"], sw["profile"], sw["memory"], sw["rag"], sw["rag_rerank"]), (True, True, False, False, False))
        self.assertFalse(chat.feature_available()["rag"])
        with self.assertRaises(features.FeatureError):
            chat.set_feature("rag", True)  # no RAG at startup: cannot be switched on
        with self.assertRaises(features.FeatureError):
            chat.set_feature("telepathie", False)
        chat.set_feature("rag", False)  # switching off what is off is harmless

    def test_history_off_sends_only_the_current_message(self):
        chat = self.make_chat()
        self.say("Je m'appelle Camille.")
        chat.set_feature("history", False)
        self.say("Comment je m'appelle ?")
        sent = self.client.streamed[-1]
        self.assertEqual([m["content"] for m in sent[1:]], ["Comment je m'appelle ?"])
        self.assertEqual(len(chat.history), 4)  # still stored: only the prompt changes
        chat.set_feature("history", True)
        self.say("Et maintenant ?")
        self.assertEqual(len(self.client.streamed[-1]), 1 + 5)

    def test_profile_off_hides_the_profile_from_the_prompt(self):
        chat = self.make_chat()
        self.say("bonjour")
        self.assertIn("langue_cible: japonais", self.system())
        chat.set_feature("profile", False)
        self.say("bonjour")
        self.assertNotIn("Profil de l'apprenant", self.system())

    def test_memory_off_neither_injects_nor_extracts(self):
        chat = self.make_chat()
        self.say("neko ?")
        chat.apply_pending()
        self.assertEqual(self.client.extractions, 1)
        self.say("et neko encore ?")
        self.assertIn("Mémoire de l'apprenant", self.system())
        chat.apply_pending()
        chat.set_feature("memory", False)
        self.say("neko ?")
        chat.apply_pending()
        self.assertNotIn("Mémoire de l'apprenant", self.system())
        self.assertEqual(self.client.extractions, 2)

    def test_compression_off_leaves_the_summary_out(self):
        chat = self.make_chat()
        self.say("un")
        self.say("deux")
        self.assertTrue(chat.compress_now())
        self.say("trois")
        self.assertIn("Résumé des échanges précédents", self.system())
        chat.set_feature("compression", False)
        self.say("quatre")
        self.assertNotIn("Résumé des échanges précédents", self.system())

    def test_rag_off_gives_its_budget_back_to_the_history(self):
        chat = self.make_chat(with_rag=True)
        with_rag = chat.context.history_budget({})
        self.say("que veut dire neko en japonais ?")
        self.assertIn("## Documents de référence", self.system())
        self.assertTrue(chat.last_sources)
        chat.set_feature("rag", False)
        self.assertGreater(chat.context.history_budget({}), with_rag)
        self.say("que veut dire neko en japonais ?")
        self.assertNotIn("## Documents de référence", self.system())
        self.assertEqual((chat.last_sources, chat.last_report.rag_tokens), ([], 0))

    def test_rag_sub_switches_reach_the_pipeline(self):
        chat = self.make_chat(with_rag=True)
        for feature_id, attr in (("rag_hybrid", "hybrid"), ("rag_rerank", "rerank"), ("rag_filters", "profile_filters")):
            chat.set_feature(feature_id, False)
            self.assertFalse(getattr(chat.rag, attr))
        chat.rag.search("neko", 2000)
        self.assertTrue(all(h.bm25 == 0 for h in chat.rag.last_hits))
        chat.reset_features()
        self.assertTrue(chat.rag.hybrid)
        self.assertEqual(chat.switches, features.defaults(chat.config, True))

    def test_answers_remember_what_was_off_and_which_lessons_were_used(self):
        chat = self.make_chat(with_rag=True)
        self.say("Comment dit-on chat, neko ?")
        chat.set_feature("rag", False)
        chat.set_feature("rag_rerank", False)  # implied by « rag » off: not listed
        chat.set_feature("memory", False)
        self.say("Comment dit-on chat, neko ?")
        answers = [m for m in self.store.load_messages(chat.session_id) if m.role == "assistant"]
        self.assertEqual(answers[0].meta["sources"][0]["source"].rsplit("/", 1)[-1], "animaux.md")
        self.assertNotIn("off", answers[0].meta)
        self.assertEqual(answers[1].meta, {"off": ["memory", "rag"]})

    def test_feature_commands(self):
        chat = self.make_chat()
        with mock.patch("builtins.print") as printed:
            chat.handle_command("/feature memory off")
            chat.handle_command("/feature rag on")
            chat.handle_command("/feature reset")
        out = "\n".join(" ".join(map(str, c.args)) for c in printed.call_args_list)
        self.assertIn("[off] memory", out)
        self.assertIn("indisponible", out)
        self.assertIn("[n/a] rag", out)
        self.assertTrue(chat.switches["memory"])


class CompareTest(ChatCase):
    def run_compare(self, feature, text="que veut dire neko en japonais ?", judge=True):
        return list(compare(self.chat, text, feature, judge))

    def test_rag_on_and_off_side_by_side_and_nothing_is_stored(self):
        chat = self.make_chat(with_rag=True)
        self.say("bonjour")
        chat.apply_pending()
        before = (self.store.count_messages(chat.session_id), len(chat.history), dict(chat.switches), chat.last_report)
        self.client.judge = judge_reply(9, 3)
        with mock.patch("sensai.compare.random.random", return_value=0.9):  # on first, off second
            events = self.run_compare("rag")
        types = [e["type"] for e in events]
        self.assertEqual(types[0], "side")
        self.assertEqual(types[-2:], ["verdict", "end"])
        done = {e["side"]: e for e in events if e["type"] == "side_done"}
        self.assertEqual(done["on"]["label"], "Avec leçons (RAG)")
        self.assertGreater(done["on"]["context"]["rag"], 0)
        self.assertEqual((done["off"]["context"]["rag"], done["off"]["sources"]), (0, []))
        tokens = {side: "".join(e["text"] for e in events if e["type"] == "token" and e["side"] == side) for side in ("on", "off")}
        self.assertEqual(tokens, {"on": "Selon ta leçon, sanbon.", "off": "De mémoire, sanbon."})
        verdict = events[-2]
        self.assertEqual((verdict["on"]["score"], verdict["off"]["score"]), (9, 3))
        self.assertIn("animaux.md", self.client.judge_prompts[0])  # the lessons are the reference
        self.assertEqual(len(events[-1]["sources_diff"]["only_on"]), len(done["on"]["sources"]))
        # nothing stored, nothing extracted, nothing changed
        after = (self.store.count_messages(chat.session_id), len(chat.history), dict(chat.switches), chat.last_report)
        self.assertEqual(after, before)
        self.assertEqual(self.client.purposes.count("memory_extraction"), 1)
        self.assertEqual(self.client.purposes.count("compare"), 2)

    def test_the_judge_sees_the_answers_in_random_order(self):
        self.make_chat(with_rag=True)
        self.client.judge = judge_reply(2, 8)  # « réponse 1 » is the side without RAG when flipped
        with mock.patch("sensai.compare.random.random", return_value=0.1):
            events = self.run_compare("rag")
        prompt = self.client.judge_prompts[0]
        self.assertLess(prompt.index("De mémoire"), prompt.index("Selon ta leçon"))
        verdict = next(e for e in events if e["type"] == "verdict")
        self.assertEqual((verdict["on"]["score"], verdict["off"]["score"]), (8, 2))

    def test_no_judge_without_reference_or_when_it_fails(self):
        chat = self.make_chat()
        events = self.run_compare("history", "Comment je m'appelle ?")  # first message: no earlier turn
        self.assertNotIn("verdict", [e["type"] for e in events])
        self.assertIn("n'a rien ajouté", next(e["text"] for e in events if e["type"] == "notice"))
        self.say("Je m'appelle Camille.")
        self.client.judge = OllamaUnavailable("down")
        events = self.run_compare("history", "Comment je m'appelle ?")
        self.assertIn("Arbitrage impossible", next(e["text"] for e in events if e["type"] == "notice"))
        self.client.judge = "pas du json"
        events = self.run_compare("history", "Comment je m'appelle ?")
        self.assertEqual(events[-1]["type"], "end")
        self.assertEqual(len(self.client.judge_prompts), 2)
        events = self.run_compare("profile", judge=False)
        self.assertNotIn("notice", [e["type"] for e in events])
        self.assertEqual(chat.switches, features.defaults(chat.config, False))

    def test_sub_switch_comparison_keeps_the_rag_on_both_sides(self):
        chat = self.make_chat(with_rag=True)
        chat.set_feature("rag", False)
        events = self.run_compare("rag_filters", "kimchi neko kippu légume")
        done = {e["side"]: e for e in events if e["type"] == "side_done"}
        self.assertTrue(done["on"]["sources"] and done["off"]["sources"])
        names_off = {s["source"].rsplit("/", 1)[-1] for s in done["off"]["sources"]}
        self.assertIn("cuisine.md", names_off)  # the Korean lesson slips in without the profile filter
        self.assertNotIn("verdict", [e["type"] for e in events])  # judged on the sources, not by the model
        self.assertFalse(chat.switches["rag"])  # put back as it was

    def test_stopping_mid_way_puts_everything_back(self):
        chat = self.make_chat(with_rag=True)
        run = compare(chat, "neko ?", "rag")
        for event in run:
            if event["type"] == "token":
                break
        self.assertTrue(chat.switches["rag"])
        run.close()
        self.assertEqual(chat.switches, features.defaults(chat.config, True))
        self.assertIs(chat.context.retriever, chat.rag)

    def test_compare_command_prints_both_sides_and_the_verdict(self):
        chat = self.make_chat(with_rag=True)
        self.client.judge = judge_reply(9, 2)
        with mock.patch("builtins.print") as printed, mock.patch("sensai.compare.random.random", return_value=0.9):
            chat.handle_command("/compare rag que veut dire neko en japonais ?")
            chat.handle_command("/compare rag")
            chat.handle_command("/compare inconnue question ?")
        out = "".join(" ".join(map(str, c.args)) + c.kwargs.get("end", "\n") for c in printed.call_args_list)
        self.assertIn("── Avec leçons (RAG)", out)
        self.assertIn("Selon ta leçon, sanbon.", out)
        self.assertIn("De mémoire, sanbon.", out)
        self.assertIn("animaux.md#0", out)
        self.assertIn("Avec leçons (RAG) : 9/10", out)
        self.assertIn("Usage : /compare", out)
        self.assertIn("fonctionnalité inconnue", out)
        self.assertEqual(self.store.count_messages(chat.session_id), 0)

    def test_bad_requests(self):
        chat = self.make_chat()
        with self.assertRaises(features.FeatureError):
            self.run_compare("rag")  # RAG unavailable
        with self.assertRaises(features.FeatureError):
            self.run_compare("inconnue")
        with self.assertRaises(ValueError):
            list(compare(chat, "   ", "memory"))


class LearnerTest(ChatCase):
    def test_switching_learners_isolates_profile_memory_and_sessions(self):
        chat = self.make_chat()
        self.say("neko ?")
        camille_session = chat.session_id
        self.assertTrue(chat.switch_user("hugo"))  # created; camille's pending extraction applied first
        self.assertEqual(chat.learner.totals("camille")["words"], 1)
        self.assertEqual((chat.config.user, chat.history, chat.last_report), ("hugo", [], None))
        self.assertNotEqual(chat.session_id, camille_session)
        self.say("annyeong ?")
        self.assertNotIn("langue_cible", self.system())  # hugo has no profile
        self.assertFalse(chat.switch_user("camille"))  # exists: latest session resumed
        self.assertEqual(chat.session_id, camille_session)
        self.assertEqual([m.content for m in chat.history], ["neko ?", "De mémoire, sanbon."])
        names = [u.name for u in self.store.list_users()]
        self.assertEqual(sorted(names), ["camille", "hugo"])
        with self.assertRaises(ValueError):
            chat.switch_user("../etc")

    def test_deleting_a_learner_removes_everything_but_never_the_current_one(self):
        chat = self.make_chat()
        chat.switch_user("hugo")
        self.say("neko ?")
        chat.switch_user("camille")
        self.assertEqual(chat.learner.totals("hugo")["words"], 1)
        with self.assertRaises(ValueError):
            chat.delete_user("camille")
        self.assertTrue(chat.delete_user("hugo"))
        self.assertFalse(self.store.user_exists("hugo"))
        self.assertIsNone(self.store.latest_session("hugo"))
        self.assertEqual(chat.learner.totals("hugo")["words"], 0)
        self.assertFalse(chat.delete_user("hugo"))

    def test_user_commands(self):
        chat = self.make_chat()
        with mock.patch("builtins.print") as printed:
            chat.handle_command("/user hugo")
            chat.handle_command("/users")
            chat.handle_command("/user a b")
        out = "\n".join(" ".join(map(str, c.args)) for c in printed.call_args_list)
        self.assertIn("Apprenant « hugo » créé", out)
        self.assertIn("* hugo", out)
        self.assertIn("Usage : /user <nom>", out)


class StoreHistoryTest(unittest.TestCase):
    def test_an_old_database_gets_its_learners_and_message_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "old.db")
            conn = sqlite3.connect(path)
            conn.executescript("""
                CREATE TABLE sessions (id INTEGER PRIMARY KEY, user TEXT NOT NULL, title TEXT,
                                       created_at REAL NOT NULL, updated_at REAL NOT NULL);
                CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL, role TEXT NOT NULL,
                                       content TEXT NOT NULL, tokens INTEGER, created_at REAL NOT NULL);
                CREATE TABLE profile (user TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
                                      updated_at REAL NOT NULL, PRIMARY KEY (user, key));
                INSERT INTO sessions VALUES (1, 'camille', 'bonjour', 100, 200);
                INSERT INTO messages VALUES (1, 1, 'user', 'bonjour', 2, 150);
                INSERT INTO profile VALUES ('lea', 'niveau', 'A1', 50);
            """)
            conn.close()
            store = MemoryStore(path)
            users = {u.name: u for u in store.list_users()}
            self.assertEqual(set(users), {"camille", "lea"})
            self.assertEqual((users["camille"].sessions, users["camille"].messages), (1, 1))
            self.assertIsNone(store.load_messages(1)[0].meta)
            store.close()

    def test_benchmark_scoring_needs_every_keyword_group(self):
        from benchmarks.rag_ab import correct

        groups = [["b204"], ["14 h", "14h"]]
        self.assertTrue(correct("Vendredi à 14 h, salle B204.", groups))
        self.assertFalse(correct("Vendredi, salle B204.", groups))
        self.assertFalse(correct("Salle B2045 à 14h", groups))  # whole words only

    def test_streak_and_last_days(self):
        today = date(2026, 10, 15)
        self.assertEqual(streak({"2026-10-15", "2026-10-14", "2026-10-12"}, today), 2)
        self.assertEqual(streak({"2026-10-14", "2026-10-13"}, today), 2)  # today not over yet
        self.assertEqual(streak({"2026-10-12"}, today), 0)
        days = last_days({"2026-10-14": 3}, 3, today)
        self.assertEqual(days, [{"day": "2026-10-13", "messages": 0}, {"day": "2026-10-14", "messages": 3},
                                {"day": "2026-10-15", "messages": 0}])


if __name__ == "__main__":
    unittest.main()
