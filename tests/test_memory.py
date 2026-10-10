"""B1/M1/M2/M3: token estimation, SQLite persistence, context budget, compression,
learner memory (Leitner, mistakes, validated operations) and the chat loop wiring.

No network and no model: OllamaClient is replaced by a scripted fake client
(summaries for compression, operations for memory extraction, a fixed streamed answer).
"""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from sensai.cli import Chat
from sensai.config import Config
from sensai.context import ContextBuilder
from sensai.memory.compressor import CompressionError, Compressor, Summary, compress
from sensai.memory.extractor import ExtractionError, MemoryExtractor
from sensai.memory.learner import DAY, LEITNER_DAYS, MAX_OPS_PER_TURN, LearnerMemory, Mistake, Word
from sensai.memory.store import MemoryStore, Message
from sensai.memory.tokens import MESSAGE_OVERHEAD, estimate_message, estimate_tokens
from sensai.observability import EventLogger
from sensai.ollama_client import OllamaUnavailable


def summary_json(**fields) -> str:
    data = {"apprenant": [], "vocabulaire": [], "erreurs": [], "exercices": [], "resume": "", "en_cours": ""}
    data.update(fields)
    return json.dumps(data, ensure_ascii=False)


def ops_json(*ops) -> str:
    return json.dumps({"operations": list(ops)}, ensure_ascii=False)


class FakeClient:
    """Stands in for OllamaClient: replies are scripted per purpose."""

    model = "fake"

    def __init__(self, summaries=None, extractions=None, answer=("Très ", "bien !")):
        self.summaries = list(summaries or [])
        self.extractions = list(extractions or [])
        self.answer = answer
        self.calls = []
        self.streamed = []

    def chat(self, messages, stats, schema=None, options=None, purpose="chat", interaction_id=None):
        self.calls.append({"messages": messages, "schema": schema, "options": options, "purpose": purpose})
        queue = self.summaries if purpose == "compression" else self.extractions
        reply = queue.pop(0) if queue else (summary_json(resume="…") if purpose == "compression" else ops_json())
        if isinstance(reply, Exception):
            raise reply
        return reply

    def chat_stream(self, messages, stats, purpose="chat", interaction_id=None):
        self.streamed.append(messages)
        stats.eval_count = 7
        yield from self.answer


def msg(role: str, content: str, id: int | None = None) -> Message:
    return Message(role, content, estimate_message(content), id)


class TmpDirTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()


# --------------------------------------------------------------------------- tokens


class TokenEstimateTest(unittest.TestCase):
    def test_latin_text_uses_chars_per_token(self):
        self.assertEqual(estimate_tokens("a" * 35), 11)  # 35 / 3.5 + 1

    def test_cjk_counts_one_token_per_character(self):
        self.assertEqual(estimate_tokens("日本語"), 4)
        self.assertEqual(estimate_tokens("ねこがすき"), 6)
        # 100 kana cost far more than 100 latin letters
        self.assertGreater(estimate_tokens("ね" * 100), 3 * estimate_tokens("a" * 100))

    def test_message_adds_template_overhead(self):
        self.assertEqual(estimate_message("日本語"), 8)
        self.assertEqual(estimate_message("x") - estimate_tokens("x"), MESSAGE_OVERHEAD)


# --------------------------------------------------------------------------- M1


class MemoryStoreTest(TmpDirTestCase):
    def setUp(self):
        super().setUp()
        self.path = str(self.base / "sub" / "sensai.db")
        self.store = MemoryStore(self.path)

    def tearDown(self):
        self.store.close()
        super().tearDown()

    def test_messages_title_and_reload(self):
        sid = self.store.create_session("camille")
        first = msg("user", "Je veux apprendre les animaux en japonais " * 3)
        self.store.add_message(sid, first)
        self.store.add_message(sid, msg("assistant", "D'accord !"))
        self.store.add_message(sid, msg("user", "neko ?"))
        self.assertIsNotNone(first.id)
        self.assertEqual(self.store.count_messages(sid), 3)
        info = self.store.list_sessions("camille")[0]
        self.assertEqual(info.title, first.content[:60])  # first user message, truncated
        self.assertEqual(info.message_count, 3)
        later = self.store.load_messages(sid, after_id=first.id)
        self.assertEqual([m.content for m in later], ["D'accord !", "neko ?"])

    def test_sessions_are_per_user(self):
        a = self.store.create_session("camille")
        b = self.store.create_session("hugo")
        self.assertTrue(self.store.session_exists("camille", a))
        self.assertFalse(self.store.session_exists("camille", b))
        self.assertEqual([s.id for s in self.store.list_sessions("hugo")], [b])
        self.assertEqual(self.store.latest_session("camille"), a)
        self.assertIsNone(self.store.latest_session("nobody"))

    def test_profile_persists_across_restarts(self):
        self.store.set_profile("camille", "langue_cible", "japonais")
        self.store.set_profile("camille", "niveau", "A1")
        self.store.set_profile("camille", "niveau", "A2")  # update, not duplicate
        self.store.set_profile("hugo", "langue_cible", "coréen")
        self.store.close()
        self.store = MemoryStore(self.path)
        self.assertEqual(self.store.get_profile("camille"), {"langue_cible": "japonais", "niveau": "A2"})
        self.assertTrue(self.store.delete_profile_key("camille", "niveau"))
        self.assertFalse(self.store.delete_profile_key("camille", "niveau"))
        self.assertEqual(self.store.get_profile("camille"), {"langue_cible": "japonais"})
        self.assertEqual(self.store.get_profile("hugo"), {"langue_cible": "coréen"})

    def test_summary_upsert_and_compression_stats(self):
        sid = self.store.create_session("camille")
        self.assertIsNone(self.store.get_summary(sid))
        self.store.set_summary(sid, "{}", 3, 10)
        self.store.set_summary(sid, '{"resume": "x"}', 7, 12)
        self.assertEqual(self.store.get_summary(sid), ('{"resume": "x"}', 7))
        self.store.log_compression(sid, 6, 300, 80, 1200.0)
        self.store.log_compression(sid, 4, 0, 0, 0, error="boom")
        s = self.store.compression_stats(sid)
        self.assertEqual((s["passes"], s["ok"], s["messages"], s["saved"]), (2, 1, 6, 220))
        self.assertAlmostEqual(s["avg_ms"], 1200.0)


# --------------------------------------------------------------------------- context budget


class ContextBuilderTest(unittest.TestCase):
    def test_budget_and_profile_injection(self):
        cb = ContextBuilder("Tu es un tuteur.", num_ctx=1000, response_reserve=0.2)
        self.assertEqual(cb.budget, 800)
        self.assertEqual((cb.summary_cap, cb.memory_cap), (80, 80))
        messages, report = cb.build({"langue_cible": "japonais"}, [msg("user", "bonjour")])
        self.assertIn("## Profil de l'apprenant\n- langue_cible: japonais", messages[0]["content"])
        self.assertEqual(messages[1], {"role": "user", "content": "bonjour"})
        self.assertEqual(report.used, report.system_tokens + report.history_tokens)

    def test_sections_order_memory_summary_rag(self):
        cb = ContextBuilder("SYS", 4000, 0.2, retriever=lambda q, n: ["[lecon.md]\nneko = chat"])
        messages, report = cb.build({}, [msg("user", "neko ?")], summary="RESUME", memory="MEMOIRE")
        system = messages[0]["content"]
        positions = [system.index(s) for s in ("SYS", "MEMOIRE", "RESUME", "## Documents de référence")]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(report.sections, ["memory", "summary", "rag"])
        self.assertGreater(report.rag_tokens, 0)

    def test_retriever_receives_last_message_and_its_share(self):
        seen = []
        cb = ContextBuilder("SYS", 1000, 0.2, retriever=lambda q, n: seen.append((q, n)) or [], rag_share=0.25)
        _, report = cb.build({}, [msg("user", "vieux"), msg("assistant", "a"), msg("user", "et inu ?")])
        self.assertEqual(seen, [("et inu ?", 200)])
        self.assertEqual(report.rag_tokens, 0)
        self.assertNotIn("rag", report.sections)

    def test_history_budget_reserves_rag_share_only_with_retriever(self):
        without = ContextBuilder("SYS", 1000, 0.2)
        with_rag = ContextBuilder("SYS", 1000, 0.2, retriever=lambda q, n: [], rag_share=0.25)
        self.assertEqual(without.history_budget({}) - with_rag.history_budget({}), 200)

    def test_oldest_turns_dropped_newest_kept(self):
        cb = ContextBuilder("SYS", 200, 0.2)  # budget 160
        history = [msg("user" if i % 2 == 0 else "assistant", f"message {i} " + "x" * 120) for i in range(6)]
        messages, report = cb.build({}, history)
        self.assertGreater(report.history_dropped, 0)
        self.assertEqual(report.history_kept + report.history_dropped, 6)
        self.assertEqual(messages[-1]["content"], history[-1].content)
        self.assertLessEqual(report.used, cb.budget)
        kept = [m["content"] for m in messages[1:]]
        self.assertEqual(kept, [m.content for m in history[-len(kept):]])  # a contiguous recent tail

    def test_current_message_always_kept_even_over_budget(self):
        cb = ContextBuilder("SYS", 100, 0.2)
        huge = msg("user", "y" * 2000)
        messages, report = cb.build({}, [msg("user", "avant"), huge])
        self.assertEqual(messages[-1]["content"], huge.content)
        self.assertEqual((report.history_kept, report.history_dropped), (1, 1))

    def test_real_token_count_preferred_over_estimate(self):
        cb = ContextBuilder("SYS", 1000, 0.2)
        _, report = cb.build({}, [Message("assistant", "court", tokens=50), msg("user", "ok")])
        self.assertEqual(report.history_tokens, 50 + estimate_message("ok"))


# --------------------------------------------------------------------------- M2


class SummaryTest(unittest.TestCase):
    def test_from_json_cleans_and_validates(self):
        s = Summary.from_json(json.dumps({"vocabulaire": ["inu = chien", "  ", ""], "erreurs": "wa/ga", "resume": " ok "}))
        self.assertEqual(s.lists["vocabulaire"], ["inu = chien"])
        self.assertEqual(s.lists["erreurs"], ["wa/ga"])  # a scalar becomes a one-item list
        self.assertEqual(s.resume, "ok")
        with self.assertRaises(CompressionError):
            Summary.from_json("pas du json")
        with self.assertRaises(CompressionError):
            Summary.from_json("[1, 2]")

    def test_merge_appends_lists_dedups_and_replaces_text(self):
        old = Summary.from_json(summary_json(vocabulaire=["inu = chien"], resume="avant", en_cours="QCM"))
        new = Summary.from_json(summary_json(vocabulaire=["Inu  =  chien", "neko = chat"], resume="", en_cours=""))
        merged = old.merged_with(new)
        self.assertEqual(merged.lists["vocabulaire"], ["inu = chien", "neko = chat"])
        self.assertEqual(merged.resume, "avant")  # kept when the newer pass has none
        self.assertEqual(merged.en_cours, "")  # en_cours always reflects the latest pass
        self.assertEqual(old.lists["vocabulaire"], ["inu = chien"])  # no mutation

    def test_trim_drops_oldest_exercises_first(self):
        s = Summary.from_json(summary_json(
            apprenant=["voyage à Kyoto en avril"],
            vocabulaire=[f"mot{i} = traduction {i}" for i in range(10)],
            exercices=[f"exercice {i} réussi" for i in range(10)],
            resume="Séance sur les animaux.",
        ))
        cap = s.tokens() - 30
        dropped = s.trim_to(cap)
        self.assertGreater(dropped, 0)
        self.assertLessEqual(s.tokens(), cap)
        self.assertEqual(len(s.lists["vocabulaire"]), 10)  # exercises went first
        self.assertNotIn("exercice 0 réussi", s.lists["exercices"])
        self.assertEqual(s.lists["apprenant"], ["voyage à Kyoto en avril"])

    def test_trim_cuts_narrative_before_learner_facts(self):
        s = Summary.from_json(summary_json(
            apprenant=["voyage à Kyoto en avril", "son chat s'appelle Mochi"],
            vocabulaire=["inu = chien"],
            resume="La séance a porté sur des exercices de traduction très variés. " * 6,
            en_cours="exercice 12",
        ))
        s.trim_to(60)
        self.assertLessEqual(s.tokens(), 60)
        self.assertEqual(s.lists["apprenant"], ["voyage à Kyoto en avril", "son chat s'appelle Mochi"])
        self.assertLess(len(s.resume), 60)

    def test_trim_drops_learner_facts_only_as_last_resort(self):
        s = Summary.from_json(summary_json(apprenant=[f"fait numéro {i}" for i in range(20)], resume="R " * 50))
        s.trim_to(40)
        self.assertLessEqual(s.tokens(), 40)
        self.assertEqual(s.resume, "")
        self.assertIn("fait numéro 19", s.lists["apprenant"])  # the newest facts survive
        self.assertNotIn("fait numéro 0", s.lists["apprenant"])

    def test_speaker_prefixes_are_stripped(self):
        s = Summary.from_json(summary_json(apprenant=["Apprenant : je vais à Kyoto", "tuteur: propose un QCM"]))
        self.assertEqual(s.lists["apprenant"], ["je vais à Kyoto", "propose un QCM"])

    def test_render_sections(self):
        s = Summary.from_json(summary_json(vocabulaire=["inu = chien"], erreurs=["wa/ga"], resume="R", en_cours="QCM"))
        text = s.render()
        for part in ("## Résumé des échanges précédents", "### Vocabulaire vu\n- inu = chien",
                     "### Erreurs à surveiller", "### En cours\nQCM"):
            self.assertIn(part, text)
        self.assertTrue(Summary().is_empty())
        self.assertFalse(s.is_empty())


class CompressorTest(unittest.TestCase):
    def history(self, n: int, chars: int = 70) -> list[Message]:
        return [msg("user" if i % 2 == 0 else "assistant", f"{i} " + "x" * chars, id=i + 1) for i in range(n)]

    def test_select_nothing_below_trigger(self):
        c = Compressor(FakeClient(), trigger=0.8, keep=0.4)
        history = self.history(4)
        total = sum(m.tokens for m in history)
        self.assertEqual(c.select(history, available=total * 2), [])

    def test_select_oldest_keeps_recent_and_last(self):
        c = Compressor(FakeClient(), trigger=0.8, keep=0.4)
        history = self.history(12)
        available = int(sum(m.tokens for m in history) / 0.9)  # 90 % > trigger
        chunk = c.select(history, available)
        self.assertGreaterEqual(len(chunk), 2)
        self.assertEqual(chunk, history[: len(chunk)])  # oldest first, contiguous
        self.assertNotIn(history[-1], chunk)
        recent = sum(m.tokens for m in history[len(chunk):])
        self.assertLessEqual(sum(m.tokens for m in chunk), 0.5 * available)  # one pass <= chunk share
        self.assertGreater(recent, 0)

    def test_select_needs_two_messages_and_force(self):
        c = Compressor(FakeClient(), trigger=0.8, keep=0.4)
        self.assertEqual(c.select(self.history(1), available=1, force=True), [])
        history = self.history(6)
        self.assertEqual(c.select(history, available=10_000), [])
        self.assertGreaterEqual(len(c.select(history, available=200, force=True)), 2)

    def test_forced_compression_keeps_only_the_last_message(self):
        c = Compressor(FakeClient(), trigger=0.8, keep=0.4)
        history = self.history(6)
        self.assertEqual(c.select(history, available=10_000), [])  # far under the trigger
        self.assertEqual(c.select(history, available=10_000, force=True), history[:-1])

    def test_compress_merges_and_keeps_facts_across_passes(self):
        client = FakeClient(summaries=[
            summary_json(apprenant=["voyage à Kyoto"], vocabulaire=["inu = chien"], resume="Passe 1"),
            summary_json(vocabulaire=["neko = chat"], erreurs=["confond wa et ga"], resume="Passe 2", en_cours="QCM"),
        ])
        c = Compressor(client, 0.8, 0.4)
        h = self.history(8)
        first = compress(c, h[:4], None, 500)
        second = compress(c, h[4:6], first.summary, 500)
        s = second.summary
        self.assertEqual(s.lists["apprenant"], ["voyage à Kyoto"])
        self.assertEqual(s.lists["vocabulaire"], ["inu = chien", "neko = chat"])
        self.assertEqual((s.resume, s.en_cours), ("Passe 2", "QCM"))
        self.assertEqual((first.covered_until, second.covered_until), (4, 6))
        self.assertEqual(second.messages_in, 2)
        self.assertLess(first.tokens_after, first.tokens_before)
        # second pass only sends the new chunk, with the previous resume as context
        prompt = client.calls[1]["messages"][1]["content"]
        self.assertIn("Passe 1", prompt)
        self.assertNotIn(h[0].content, prompt)
        self.assertEqual(client.calls[0]["options"], {"temperature": 0})
        self.assertIsNotNone(client.calls[0]["schema"])

    def test_compress_respects_summary_cap(self):
        client = FakeClient(summaries=[summary_json(exercices=[f"exercice numéro {i}" for i in range(40)], resume="R")])
        result = compress(Compressor(client, 0.8, 0.4), self.history(4), None, 60)
        self.assertLessEqual(result.tokens_after, 60)
        self.assertGreater(result.dropped_items, 0)

    def test_compress_errors(self):
        for reply in (OllamaUnavailable("down"), "{pas du json"):
            with self.subTest(reply=reply), self.assertRaises(CompressionError):
                compress(Compressor(FakeClient(summaries=[reply]), 0.8, 0.4), self.history(4), None, 500)


# --------------------------------------------------------------------------- M3


class LearnerMemoryTest(TmpDirTestCase):
    def setUp(self):
        super().setUp()
        self.store = MemoryStore(str(self.base / "m.db"))
        self.mem = LearnerMemory(self.store.conn)
        self.profile = {}

    def tearDown(self):
        self.store.close()
        super().tearDown()

    def set_profile(self, user, key, value):
        self.profile[key] = value

    def apply(self, *ops, lang="japonais"):
        return self.mem.apply("camille", lang, list(ops), self.set_profile, session_id=1)

    def box(self, word, lang="japonais"):
        return {w.word: w for w in self.mem.words("camille", lang)}[word]

    def test_words_are_normalized(self):
        self.mem.upsert_word("camille", "japonais", "Inu", "chien")
        self.mem.upsert_word("camille", "japonais", " inu ", "le chien")
        words = self.mem.words("camille", "japonais")
        self.assertEqual([(w.word, w.translation) for w in words], [("inu", "le chien")])

    def test_leitner_steps(self):
        now = time.time()
        self.mem.review_word("camille", "japonais", "neko", True, "chat")  # created then promoted
        w = self.box("neko")
        self.assertEqual((w.box, w.correct, w.wrong), (1, 1, 0))
        self.assertAlmostEqual(w.next_review, now + LEITNER_DAYS[1] * DAY, delta=5)
        for _ in range(10):
            self.mem.review_word("camille", "japonais", "neko", True)
        self.assertEqual(self.box("neko").box, len(LEITNER_DAYS) - 1)  # capped at the last box
        self.mem.review_word("camille", "japonais", "neko", False)
        w = self.box("neko")
        self.assertEqual((w.box, w.wrong), (0, 1))
        self.assertLessEqual(w.next_review, time.time())  # due right away
        self.assertEqual([d.word for d in self.mem.due_words("camille", "japonais")], ["neko"])
        with self.assertRaises(ValueError):
            self.mem.review_word("camille", "japonais", "tori", True)  # unknown, no translation

    def test_mastered_count(self):
        self.mem.upsert_word("camille", "japonais", "inu", "chien")
        for _ in range(4):
            self.mem.review_word("camille", "japonais", "neko", True, "chat")
        self.assertEqual(self.mem.vocab_stats("camille", "japonais"), (2, 1))

    def test_languages_and_users_are_separate(self):
        self.mem.upsert_word("camille", "japonais", "inu", "chien")
        self.mem.upsert_word("camille", "coréen", "gae", "chien")
        self.mem.upsert_word("hugo", "japonais", "neko", "chat")
        self.assertEqual([w.word for w in self.mem.words("camille", "japonais")], ["inu"])
        self.assertEqual([w.word for w in self.mem.words("hugo", "japonais")], ["neko"])

    def test_mistakes_counted_not_duplicated(self):
        self.mem.add_mistake("camille", "japonais", "Confond wa et ga", "wa = thème", "watashi ga Camille")
        self.mem.add_mistake("camille", "japonais", "confond  wa et ga", "wa = thème, ga = sujet")
        self.mem.add_mistake("camille", "japonais", "oublie desu", "desu en fin de phrase")
        top = self.mem.mistakes("camille", "japonais")
        self.assertEqual([(m.pattern, m.count) for m in top], [("confond wa et ga", 2), ("oublie desu", 1)])
        self.assertEqual(top[0].example, "watashi ga Camille")  # kept when the new one has none
        self.assertTrue(self.mem.resolve_mistake("camille", "japonais", "Oublie desu"))
        self.assertEqual(len(self.mem.mistakes("camille", "japonais")), 1)

    def test_apply_valid_operations(self):
        self.mem.upsert_word("camille", "japonais", "tori", "oiseau")
        report = self.apply(
            {"op": "add_word", "word": "inu", "translation": "chien"},
            {"op": "review_word", "word": "neko", "correct": False, "translation": "chat"},
            {"op": "add_mistake", "pattern": "confond wa et ga", "correction": "wa = thème"},
            {"op": "set_profile", "key": "Langue Maternelle", "value": " français "},
            {"op": "delete_word", "word": "tori"},
        )
        self.assertEqual(report.rejected, [])
        self.assertEqual(len(report.applied), 5)
        self.assertEqual(sorted(w.word for w in self.mem.words("camille", "japonais")), ["inu", "neko"])
        self.assertEqual(self.profile, {"langue_maternelle": "français"})
        self.assertIn("1× add_word", report.short())

    def test_apply_rejects_invalid_operations_without_blocking_others(self):
        report = self.apply(
            {"op": "teleport"},
            {"op": "add_word", "word": "inu"},
            {"op": "add_word", "word": "x" * 201, "translation": "long"},
            {"op": "add_word", "word": 5, "translation": "cinq"},
            {"op": "add_word", "word": "go", "translation": ["cinq"]},
            {"op": "review_word", "word": "neko", "correct": "oui", "translation": "chat"},
            {"op": "review_word", "word": "tori", "correct": True},
            {"op": "delete_word", "word": "absent"},
            {"op": "resolve_mistake", "pattern": "absente"},
            {"op": "add_word", "word": "uma", "translation": "cheval"},
        )
        self.assertEqual(report.applied, ["add_word"])
        self.assertEqual(len(report.rejected), 9)
        self.assertEqual([w.word for w in self.mem.words("camille", "japonais")], ["uma"])
        stats = self.mem.op_stats("camille")
        self.assertEqual((stats["total"], stats["applied"]), (10, 1))
        rows = self.store.conn.execute("SELECT op, error FROM memory_ops WHERE applied = 0").fetchall()
        self.assertTrue(all(r["error"] for r in rows))

    def test_apply_rejects_sentences_and_explanations(self):
        report = self.apply(
            {"op": "add_word", "word": "Neko ga suki desu.", "translation": "J'aime les chats."},
            {"op": "add_word", "word": "ga",
             "translation": "particule qui marque le sujet dans une question ou quand on veut mettre l'accent sur le sujet"},
            {"op": "review_word", "word": "arigatou gozaimasu", "correct": True, "translation": "merci beaucoup"},
        )
        self.assertEqual(report.applied, ["review_word"])  # a short expression is fine
        self.assertEqual(len(report.rejected), 2)

    def test_apply_caps_operations_per_turn(self):
        ops = [{"op": "add_word", "word": f"mot{i}", "translation": "t"} for i in range(MAX_OPS_PER_TURN + 3)]
        report = self.apply(*ops)
        self.assertEqual(len(report.applied), MAX_OPS_PER_TURN)
        self.assertEqual(len(report.rejected), 3)

    def test_failed_extraction_is_counted(self):
        self.mem.log_failure("camille", 1, "invalid JSON")
        self.assertEqual(self.mem.op_stats("camille")["failed_extractions"], 1)

    def test_mentions_whole_words_and_cjk(self):
        self.mem.upsert_word("camille", "japonais", "inu", "chien")
        self.mem.upsert_word("camille", "japonais", "ねこ", "chat")
        self.assertEqual(self.mem.mentioned_words("camille", "japonais", "c'est inutile"), [])
        found = self.mem.mentioned_words("camille", "japonais", "Inu ! Et ねこがすき")
        self.assertEqual(sorted(w.word for w in found), ["inu", "ねこ"])

    def test_render_priorities_and_cap(self):
        self.assertIsNone(self.mem.render("camille", "japonais", "bonjour", 500))
        self.mem.upsert_word("camille", "japonais", "inu", "chien")
        self.mem.review_word("camille", "japonais", "neko", False, "chat")
        self.mem.add_mistake("camille", "japonais", "confond wa et ga", "wa = thème")
        text = self.mem.render("camille", "japonais", "inu desu", 500)
        self.assertIn("## Mémoire de l'apprenant (japonais)", text)
        self.assertIn("### Mots présents dans le message\n- inu = chien", text)
        self.assertIn("### Mots à réviser", text)
        self.assertIn("neko = chat", text)
        self.assertIn("### Erreurs fréquentes\n- confond wa et ga", text)
        self.assertEqual(text.count("inu = chien"), 1)  # not repeated in the review section
        self.assertTrue(text.endswith("surveille les erreurs fréquentes."))
        for cap in (60, 80, 120):
            small = self.mem.render("camille", "japonais", "inu desu", cap)
            self.assertLessEqual(estimate_message(small), cap)
        self.assertIsNone(self.mem.render("camille", "japonais", "inu", 10))


class MemoryExtractorTest(unittest.TestCase):
    def test_prompt_contains_known_items_and_returns_ops(self):
        client = FakeClient(extractions=[ops_json({"op": "add_word", "word": "inu", "translation": "chien"}, "parasite")])
        ops = MemoryExtractor(client).extract(
            "inu ?", "inu = chien", {"langue_cible": "japonais"},
            [Word("neko", "chat", 1, 1, 0, 0.0)], [Mistake("confond wa et ga", "wa = thème", None, 2)],
        )
        self.assertEqual(ops, [{"op": "add_word", "word": "inu", "translation": "chien"}])
        call = client.calls[0]
        self.assertEqual(call["purpose"], "memory_extraction")
        prompt = call["messages"][1]["content"]
        for part in ("langue_cible = japonais", "neko = chat", "confond wa et ga", "Apprenant : inu ?"):
            self.assertIn(part, prompt)

    def test_set_profile_restating_the_profile_is_dropped(self):
        client = FakeClient(extractions=[ops_json(
            {"op": "set_profile", "key": "Langue cible", "value": "japonais "},
            {"op": "set_profile", "key": "niveau", "value": "débutant"},
            {"op": "set_profile", "key": "langue_cible", "value": "coréen"},
        )])
        ops = MemoryExtractor(client).extract("a", "b", {"langue_cible": "japonais"}, [], [])
        self.assertEqual([(o["key"], o["value"]) for o in ops], [("niveau", "débutant"), ("langue_cible", "coréen")])

    def test_failures_raise_extraction_error(self):
        for reply in (OllamaUnavailable("down"), "{oops", '{"ops": []}', '{"operations": "add_word"}'):
            with self.subTest(reply=reply), self.assertRaises(ExtractionError):
                MemoryExtractor(FakeClient(extractions=[reply])).extract("a", "b", {}, [], [])


# --------------------------------------------------------------------------- chat loop wiring


class ChatMemoryTest(TmpDirTestCase):
    def make_chat(self, client, **overrides):
        values = dict(
            db_path=str(self.base / "chat.db"), log_path=str(self.base / "events.jsonl"),
            user="camille", rag=False, num_ctx=600, system_prompt="Tu es Sensai.",
        )
        values.update(overrides)
        self.config = Config(**values)
        self.store = MemoryStore(self.config.db_path)
        self.client = client
        return Chat(self.config, self.store, client, EventLogger(self.config.log_path))

    def tearDown(self):
        self.store.close()
        super().tearDown()

    def run_cmd(self, chat, *lines) -> str:
        with mock.patch("builtins.print") as printed:
            for line in lines:
                chat.handle_command(line)
        return "\n".join(" ".join(str(a) for a in c.args) for c in printed.call_args_list)

    def talk(self, chat, n, text="Voici une phrase d'exercice assez longue pour remplir le budget du contexte. "):
        with mock.patch("builtins.print"):
            for i in range(n):
                chat.send(f"{i} {text * 2}")

    def test_long_session_is_compressed_and_reloaded(self):
        client = FakeClient(summaries=[summary_json(apprenant=["voyage à Kyoto"], resume="Début de séance")] * 5)
        chat = self.make_chat(client, memory=False)
        self.talk(chat, 8)
        self.assertIsNotNone(chat.summary)
        self.assertIn("voyage à Kyoto", chat.summary.render())
        self.assertEqual(chat.last_report.history_dropped, 0)
        self.assertIn("## Résumé des échanges précédents", client.streamed[-1][0]["content"])
        stats = self.store.compression_stats(chat.session_id)
        self.assertGreaterEqual(stats["ok"], 1)
        total = self.store.count_messages(chat.session_id)
        self.assertEqual(total, 16)  # raw history is never deleted
        live = [m.content for m in chat.history]

        sid = chat.session_id
        out = self.run_cmd(chat, "/new", f"/load {sid}")
        self.assertIn(f"dont {total - len(live)} résumés", out)
        self.assertEqual([m.content for m in chat.history], live)
        self.assertIn("voyage à Kyoto", chat.summary.render())
        chat.close()

    def test_failed_compression_falls_back_and_backs_off(self):
        client = FakeClient(summaries=[OllamaUnavailable("down")] * 20)
        chat = self.make_chat(client, memory=False)
        self.talk(chat, 14)
        self.assertIsNone(chat.summary)
        stats = self.store.compression_stats(chat.session_id)
        self.assertEqual(stats["ok"], 0)
        # one try, then 3 turns of back-off before the next one
        self.assertTrue(2 <= stats["passes"] <= 14 // 4 + 1, stats["passes"])
        self.assertGreater(chat.last_report.history_dropped, 0)  # fallback: oldest turns dropped
        self.assertLessEqual(chat.last_report.used, chat.context.budget)
        chat.close()

    def test_extraction_applied_on_next_message(self):
        client = FakeClient(extractions=[
            ops_json({"op": "add_word", "word": "inu", "translation": "chien"},
                     {"op": "add_mistake", "pattern": "confond wa et ga", "correction": "wa = thème"}),
            OllamaUnavailable("down"),
        ])
        chat = self.make_chat(client, num_ctx=4096)
        self.store.set_profile("camille", "langue_cible", "japonais")
        self.talk(chat, 1, "inu ?")
        self.talk(chat, 1, "et inu au pluriel ?")
        self.assertEqual([w.word for w in chat.learner.words("camille", "japonais")], ["inu"])
        self.assertIn("inu = chien", chat.last_memory)  # injected in the second turn
        self.assertIn("Mémoire de l'apprenant", client.streamed[-1][0]["content"])
        chat.close()  # applies the pending (failed) extraction
        stats = chat.learner.op_stats("camille")
        self.assertEqual((stats["applied"], stats["failed_extractions"]), (2, 1))

    def test_stopping_the_stream_keeps_the_partial_answer(self):
        chat = self.make_chat(FakeClient(answer=("Neko ", "= chat", ", et aussi…")), memory=False)
        turn = chat.stream("neko ?")
        self.assertEqual(next(turn), "Neko ")
        turn.close()  # « Stop » in the web UI
        self.assertTrue(chat.interrupted)
        self.assertEqual([m.content for m in chat.history], ["neko ?", "Neko "])
        self.assertEqual(self.store.count_messages(chat.session_id), 2)
        chat.close()

    def test_profile_and_session_commands(self):
        chat = self.make_chat(FakeClient(), memory=False)
        out = self.run_cmd(chat, "/profile set objectif voyager au Japon", "/profile")
        self.assertIn("objectif: voyager au Japon", out)
        self.talk(chat, 1, "bonjour")
        self.assertIn("- objectif: voyager au Japon", self.client.streamed[-1][0]["content"])
        other = self.store.create_session("hugo")
        out = self.run_cmd(chat, f"/load {other}", "/load abc")
        self.assertIn(f"Session #{other} introuvable", out)
        self.assertIn("Usage : /load <id>", out)
        out = self.run_cmd(chat, "/sessions")
        self.assertIn(f"* #{chat.session_id}", out)
        self.assertIn("bonjour", out)
        self.assertNotIn(f"#{other} ", out)  # hugo's session is not listed
        chat.close()


if __name__ == "__main__":
    unittest.main()
