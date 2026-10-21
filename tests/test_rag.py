"""R1 + R2: chunking, hybrid search, metadata filters, thresholds, rerank and fallbacks.

No network and no model: embeddings come from a deterministic hashing embedder
(shared words => similar vectors) and the reranker is a scripted fake client.
"""

import json
import tempfile
import unittest
import zlib
from unittest import mock
from array import array
from pathlib import Path

from sensai.config import Config
from sensai.context import ContextBuilder
from sensai.memory.store import Message
from sensai.memory.tokens import estimate_tokens
from sensai.ollama_client import OllamaUnavailable
from sensai.rag import Rag, parse_filter
from sensai.rag.chunker import chunk_markdown, load_documents, parse_front_matter
from sensai.rag.embedder import OllamaEmbedder, normalize
from sensai.rag.pipeline import parse_scores, rerank_schema
from sensai.rag.lexical import bm25_scores, tokenize
from sensai.rag.store import RagStore

DIM = 256


class FakeEmbedder:
    """Bag-of-words hashing: texts that share words get a high cosine, others ~0."""

    def __init__(self):
        self.calls = 0
        self.texts = 0
        self.fail = False

    def check(self):
        return DIM

    def embed(self, texts):
        if self.fail:
            raise OllamaUnavailable("embedder down")
        self.calls += 1
        self.texts += len(texts)
        out = []
        for text in texts:
            vec = [0.0] * DIM
            for tok in tokenize(text):
                vec[zlib.crc32(tok.encode()) % DIM] += 1.0
            out.append(normalize(vec))
        return out


class FakeClient:
    """Stands in for OllamaClient.chat; replies are scripted."""

    model = "fake"

    def __init__(self, scores=None, raw=None, error=None, compact=False):
        self.compact = compact  # reply [7, 2, …] instead of [{"id": 0, "score": 7}, …]
        self.scores = scores or {}
        self.raw = raw
        self.error = error
        self.calls = 0
        self.last_messages = None

    def chat(self, messages, stats, schema=None, options=None, purpose="chat", interaction_id=None):
        self.calls += 1
        self.last_messages = messages
        if self.error:
            raise self.error
        if self.raw is not None:
            return self.raw
        n = messages[-1]["content"].count("\n[")
        self.last_schema = schema
        if self.compact:
            return json.dumps({"scores": [self.scores.get(i, 8) for i in range(n)]})
        return json.dumps({"scores": [{"id": i, "score": self.scores.get(i, 8)} for i in range(n)]})


ANIMAUX = """---
langue: japonais
niveau: A1
---
# Leçon 3 — Les animaux (japonais, niveau A1)

| Japonais | Kana | Français |
|---|---|---|
| neko | ねこ | chat |
| inu | いぬ | chien |

Phrase modèle : « Neko ga suki desu » = J'aime les chats.
"""

CUISINE = """---
langue: coréen
niveau: B1
date: 2025-01-15
---
# Leçon 9 — La cuisine (coréen, niveau B1)

kimchi = légume fermenté épicé. bibimbap = riz mélangé avec des légumes.
"""

SANS_META = """# Voyager au Japon

Pour acheter un billet de train il faut dire kippu o kudasai au guichet.
"""


def make_rag(tmp: str, client=None, embedder=None, profile=None, **cfg) -> Rag:
    base = Path(tmp)
    (base / "lecons").mkdir(exist_ok=True)
    for name, text in (("animaux.md", ANIMAUX), ("cuisine.md", CUISINE), ("voyage.md", SANS_META)):
        (base / "lecons" / name).write_text(text, encoding="utf-8")
    settings = {
        "rag_sources": [str(base / "lecons")],
        "rag_db_path": ":memory:",
        "rag_min_cosine": 0.15,
        "rag_rerank": False,
        "rag_profile_filters": {},
        **cfg,
    }
    config = Config(**settings)
    return Rag(
        config, client or FakeClient(), profile_fn=(lambda: profile) if profile is not None else None,
        embedder=embedder or FakeEmbedder(), store=RagStore(":memory:"),
    )


class ChunkerTests(unittest.TestCase):
    def test_front_matter(self):
        meta, body = parse_front_matter(ANIMAUX)
        self.assertEqual(meta, {"langue": "japonais", "niveau": "A1"})
        self.assertTrue(body.startswith("# Leçon 3"))
        self.assertEqual(parse_front_matter("# pas de front matter")[0], {})

    def test_chunks_keep_breadcrumb_and_tables_together(self):
        _, body = parse_front_matter(ANIMAUX)
        chunks = chunk_markdown(body, "animaux")
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].text.startswith("Leçon 3 — Les animaux (japonais, niveau A1)\n"))
        self.assertIn("| neko | ねこ | chat |", chunks[0].text)

    def test_nested_headings_and_size_limit(self):
        body = "# Cours\n## Verbes\n" + "\n\n".join(f"Paragraphe {i} " + "x" * 150 for i in range(8))
        chunks = chunk_markdown(body, "t", max_chars=400)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(c.text.startswith("Cours > Verbes\n") for c in chunks))
        self.assertTrue(all(len(c.text) <= 400 + len("Cours > Verbes\n") for c in chunks))

    def test_very_long_line_is_sliced(self):
        chunks = chunk_markdown("a" * 2000, "t", max_chars=500, overlap=50)
        self.assertGreaterEqual(len(chunks), 4)

    def test_no_heading_uses_title(self):
        self.assertTrue(chunk_markdown("juste du texte", "Mon titre")[0].text.startswith("Mon titre\n"))

    def test_load_documents_metadata_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.md").write_text(SANS_META, encoding="utf-8")
            (Path(tmp) / "ignore.pdf").write_bytes(b"%PDF")
            (Path(tmp) / "big.md").write_text("x" * 600_000, encoding="utf-8")
            docs, report = load_documents([tmp, str(Path(tmp) / "absent")])
        self.assertEqual(len(docs), 1)
        meta = docs[0].meta
        self.assertEqual(meta["title"], "Voyager au Japon")
        self.assertRegex(meta["date"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertEqual(len(report.skipped), 1)
        self.assertEqual(len(report.missing_sources), 1)


class LexicalTests(unittest.TestCase):
    def test_tokenize_latin_and_cjk(self):
        self.assertEqual(tokenize("Leçon Écrite"), ["lecon", "ecrite"])
        self.assertEqual(tokenize("ねこが好き"), ["ねこ", "こが", "が好", "好き"])
        self.assertEqual(tokenize("猫 cat"), ["猫", "cat"])

    def test_dakuten_is_kept(self):
        self.assertNotEqual(tokenize("がが"), tokenize("かか"))

    def test_bm25_prefers_matching_document(self):
        docs = [tokenize("le chat dort"), tokenize("le chien mange"), tokenize("la pluie tombe")]
        scores = bm25_scores(tokenize("chat"), docs)
        self.assertGreater(scores[0], 0)
        self.assertEqual(scores[1:], [0.0, 0.0])


class FilterTests(unittest.TestCase):
    def test_equality_or_and_ranges(self):
        meta = {"niveau": "A2", "date": "2026-09-30"}
        self.assertTrue(parse_filter("niveau=a2").matches(meta))
        self.assertTrue(parse_filter("niveau=A1,A2").matches(meta))
        self.assertFalse(parse_filter("niveau=B1").matches(meta))
        self.assertTrue(parse_filter("date>=2026-09-01").matches(meta))
        self.assertFalse(parse_filter("date<=2026-09-01").matches(meta))
        self.assertTrue(parse_filter("niveau<=B1").matches(meta))

    def test_strict_vs_soft_on_missing_key(self):
        self.assertFalse(parse_filter("langue=japonais").matches({}))
        soft = parse_filter("langue=japonais")
        self.assertTrue(type(soft)(soft.key, soft.op, soft.values, soft=True).matches({}))

    def test_invalid_filters(self):
        for bad in ("niveau", "=A1", "niveau=", "niveau==", "date>=1,2"):
            with self.assertRaises(ValueError):
                parse_filter(bad)


class IndexingTests(unittest.TestCase):
    def test_incremental_sync(self):
        with tempfile.TemporaryDirectory() as tmp:
            emb = FakeEmbedder()
            rag = make_rag(tmp, embedder=emb)
            first = rag.sync()
            self.assertEqual((first.files, first.indexed, first.unchanged), (3, 3, 0))
            calls = emb.calls
            again = rag.sync()
            self.assertEqual((again.indexed, again.unchanged), (0, 3))
            self.assertEqual(emb.calls, calls)  # nothing re-embedded

            (Path(tmp) / "lecons" / "cuisine.md").write_text(CUISINE + "\nramyeon = nouilles.\n", encoding="utf-8")
            (Path(tmp) / "lecons" / "voyage.md").unlink()
            changed = rag.sync()
            self.assertEqual((changed.indexed, changed.unchanged, changed.removed), (1, 1, 1))
            self.assertEqual(rag.store.stats()["documents"], 2)

    def test_changing_model_reindexes_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            rag = make_rag(tmp)
            rag.sync()
            rag.config.rag_embed_model = "autre-modele"
            self.assertEqual(rag.sync().indexed, 3)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def rag(self, **kw):
        rag = make_rag(self.tmp.name, **kw)
        rag.sync()
        return rag

    def test_finds_the_relevant_lesson(self):
        rag = self.rag()
        out = rag("comment dit-on chat en japonais : neko ?", 2000)
        self.assertTrue(out)
        self.assertIn("animaux.md", out[0])

    def test_off_topic_message_returns_nothing(self):
        rag = self.rag()
        rag.config.rag_min_cosine = 0.3
        self.assertEqual(rag("quel temps fera-t-il demain à Marseille ?", 2000), [])
        reasons = [h.reason for h in rag.last_hits]
        self.assertTrue(all(r.startswith("cosinus") for r in reasons))

    def test_metadata_filter_excludes_other_documents(self):
        rag = self.rag()
        rag.set_filters(["niveau=B1"])
        hits = rag.search("kimchi neko kippu légume", 2000)
        self.assertEqual({h.chunk.source.rsplit("/", 1)[-1] for h in hits}, {"cuisine.md"})
        rag.set_filters(["date>=2026-01-01"])  # cuisine.md is dated 2025
        sources = {h.chunk.source.rsplit("/", 1)[-1] for h in rag.search("kimchi neko kippu", 2000)}
        self.assertNotIn("cuisine.md", sources)
        rag.set_filters(["auteur=personne"])  # strict: no document has this key
        self.assertEqual(rag.search("neko", 2000), [])

    def test_profile_filter_is_soft(self):
        rag = self.rag(profile={"langue_cible": "japonais"})
        rag.config.rag_profile_filters = {"langue_cible": "langue"}
        sources = {h.chunk.source.rsplit("/", 1)[-1] for h in rag.search("kimchi neko kippu légume", 2000)}
        self.assertEqual(sources, {"animaux.md", "voyage.md"})  # coréen excluded, no-metadata kept
        rag.set_filters(["langue=coréen"])  # an explicit filter overrides the profile one
        sources = {h.chunk.source.rsplit("/", 1)[-1] for h in rag.search("kimchi neko", 2000)}
        self.assertEqual(sources, {"cuisine.md"})

    def test_token_budget_is_respected(self):
        rag = self.rag()
        full = rag("neko kimchi kippu japonais chat", 5000)
        self.assertGreater(len(full), 1)
        budget = estimate_tokens(full[0]) + 10
        small = rag("neko kimchi kippu japonais chat", budget)
        self.assertLess(len(small), len(full))
        self.assertLessEqual(sum(estimate_tokens(c) for c in small), budget)

    def test_top_k(self):
        rag = self.rag(rag_top_k=1, rag_candidates=3)
        self.assertEqual(len(rag("neko kimchi kippu", 5000)), 1)

    def test_embedding_failure_does_not_raise(self):
        emb = FakeEmbedder()
        rag = make_rag(self.tmp.name, embedder=emb)
        rag.sync()
        emb.fail = True
        self.assertEqual(rag("neko", 2000), [])
        self.assertIn("erreur", rag.last_status)

    def test_empty_index(self):
        rag = Rag(Config(rag_db_path=":memory:", rag_rerank=False), FakeClient(), embedder=FakeEmbedder(), store=RagStore(":memory:"))
        self.assertEqual(rag("neko", 2000), [])


class RerankTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def rag(self, client):
        rag = make_rag(self.tmp.name, client=client, rag_rerank=True, rag_min_rerank=0.5)
        rag.sync()
        return rag

    def test_rerank_drops_near_misses_and_reorders(self):
        # candidates are ordered by fusion score; give the first one a low score
        client = FakeClient(scores={0: 2, 1: 9})
        rag = self.rag(client)
        rag.search("neko kimchi kippu", 5000)
        ranked = [h for h in rag.last_hits if h.rerank is not None]
        self.assertGreaterEqual(len(ranked), 2)
        low = [h for h in ranked if h.rerank < 0.5]
        self.assertTrue(low and all(not h.kept and h.reason.startswith("rerank") for h in low))
        kept = [h for h in rag.last_hits if h.kept]
        self.assertTrue(all(h.rerank >= 0.5 for h in kept))
        self.assertEqual(client.calls, 1)  # one call for all candidates

    def test_compact_scores_and_schema_sized_to_the_candidates(self):
        client = FakeClient(scores={0: 2, 1: 9}, compact=True)
        rag = self.rag(client)
        rag.search("neko kimchi kippu", 5000)
        ranked = [h for h in rag.last_hits if h.rerank is not None]
        self.assertGreaterEqual(len(ranked), 2)
        scores = client.last_schema["properties"]["scores"]
        self.assertEqual((scores["minItems"], scores["maxItems"]), (len(ranked), len(ranked)))
        self.assertTrue(any(not h.kept and h.reason.startswith("rerank") for h in ranked))

    def test_no_candidates_means_no_rerank_call(self):
        client = FakeClient()
        rag = self.rag(client)
        rag.config.rag_min_cosine = 0.99
        self.assertEqual(rag("quel temps demain", 2000), [])
        self.assertEqual(client.calls, 0)

    def test_bad_json_falls_back_to_fusion_order(self):
        for client in (FakeClient(raw="pas du json"), FakeClient(raw='{"scores": [{"id": 7, "score": 9}]}'),
                       FakeClient(error=OllamaUnavailable("down"))):
            rag = self.rag(client)
            out = rag("comment dit-on chat en japonais : neko ?", 2000)
            self.assertTrue(out)  # fallback keeps the cosine-gated candidates
            self.assertTrue(all(h.rerank is None for h in rag.last_hits))


class LexicalRescueTests(unittest.TestCase):
    """Gate 1 is the cosine, but a chunk at the top of BM25 passes anyway (a rare word the embedding dilutes)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def search(self, **cfg):
        rag = make_rag(self.tmp.name, rag_rerank=False, rag_min_cosine=0.95, **cfg)
        rag.sync()
        rag.search("kimchi", 5000)
        return rag, {h.chunk.source.rsplit("/", 1)[-1]: h for h in rag.last_hits}

    def test_top_bm25_chunk_is_rescued_under_the_cosine_gate(self):
        rag, hits = self.search(rag_min_bm25=0.5)
        self.assertLess(hits["cuisine.md"].cosine, 0.95)
        self.assertTrue(hits["cuisine.md"].rescued and hits["cuisine.md"].kept)
        self.assertIn("repêché par BM25", rag.render_hits())
        self.assertFalse(hits["animaux.md"].kept)  # no exact-word match: the cosine gate still applies

    def test_no_rescue_when_off_or_without_bm25(self):
        for cfg in ({"rag_min_bm25": 0.0}, {"rag_min_bm25": 0.5, "rag_hybrid": False}, {"rag_min_bm25": 99.0}):
            with self.subTest(**cfg):
                _, hits = self.search(**cfg)
                self.assertFalse(hits["cuisine.md"].kept or hits["cuisine.md"].rescued)

    def test_function_words_do_not_count(self):
        self.assertEqual(tokenize("Que veut dire hatsuka ?"), ["hatsuka"])
        self.assertEqual(tokenize("C'est quoi le compteur pour les stylos ?"), ["compteur", "stylos"])
        # Japanese particles and words a learner may ask to translate are kept
        self.assertEqual(tokenize("watashi wa, neko ga, oui, non, moi"), ["watashi", "wa", "neko", "ga", "oui", "non", "moi"])


class RerankFormatTests(unittest.TestCase):
    def test_compact_and_legacy_forms(self):
        self.assertEqual(parse_scores('{"scores": [7, 0, 10]}', 3), [7.0, 0.0, 10.0])
        legacy = '{"scores": [{"id": 1, "score": 4}, {"id": 0, "score": 9}]}'
        self.assertEqual(parse_scores(legacy, 2), [9.0, 4.0])

    def test_wrong_count_or_type_is_refused(self):
        for raw in ('{"scores": [7, 0]}', '{"scores": [7, "x", 1]}', '{"scores": [true, 1, 2]}',
                    '{"scores": [{"id": 0, "score": 1}]}', '{"notes": [1, 2, 3]}', "[1, 2, 3]"):
            with self.subTest(raw=raw), self.assertRaises((ValueError, KeyError, TypeError)):
                parse_scores(raw, 3)

    def test_schema_fixes_the_number_of_scores(self):
        scores = rerank_schema(4)["properties"]["scores"]
        self.assertEqual((scores["minItems"], scores["maxItems"], scores["items"]["type"]), (4, 4, "integer"))


class IntegrationTests(unittest.TestCase):
    def test_context_builder_injects_chunks_within_share(self):
        with tempfile.TemporaryDirectory() as tmp:
            rag = make_rag(tmp)
            rag.sync()
            builder = ContextBuilder("Tu es Sensai.", 8192, 0.2, retriever=rag, rag_share=0.25)
            history = [Message("user", "que veut dire neko en japonais ?", 10)]
            messages, report = builder.build({}, history)
            system = messages[0]["content"]
            self.assertIn("## Documents de référence", system)
            self.assertIn("animaux.md", system)
            self.assertIn("rag", report.sections)
            self.assertLessEqual(report.rag_tokens, int(builder.budget * 0.25) + 4)



class EmbedderRequestTest(unittest.TestCase):
    """The real embedder's request (HTTP mocked): by default the embedding model stays off the GPU."""

    def embed(self, embedder):
        reply = mock.Mock(status_code=200)
        reply.json.return_value = {"embeddings": [[3.0, 4.0]]}
        with mock.patch("requests.post", return_value=reply) as post:
            vectors = embedder.embed(["neko"])
        return post.call_args.kwargs["json"], vectors

    def test_cpu_by_default_so_the_chat_model_keeps_the_gpu(self):
        body, vectors = self.embed(OllamaEmbedder("http://x", "bge-m3"))
        self.assertEqual(body["options"], {"num_gpu": 0})
        self.assertAlmostEqual(vectors[0][0], 0.6, places=5)  # still normalized

    def test_gpu_left_to_ollama_when_enabled(self):
        body, _ = self.embed(OllamaEmbedder("http://x", "bge-m3", gpu=True))
        self.assertNotIn("options", body)

    def test_keep_alive_is_sent_when_configured(self):
        body, _ = self.embed(OllamaEmbedder("http://x", "bge-m3", keep_alive="30m"))
        self.assertEqual(body["keep_alive"], "30m")
        body, _ = self.embed(OllamaEmbedder("http://x", "bge-m3"))
        self.assertNotIn("keep_alive", body)  # Ollama's own default


class ChatPayloadTest(unittest.TestCase):
    def test_keep_alive_and_num_ctx_in_every_chat_request(self):
        from sensai.ollama_client import OllamaClient
        payload = OllamaClient("http://x", "m", 8192, False, keep_alive="30m")._payload([], stream=True)
        self.assertEqual((payload["keep_alive"], payload["options"]["num_ctx"]), ("30m", 8192))
        self.assertNotIn("keep_alive", OllamaClient("http://x", "m", 8192)._payload([], stream=True))

    def test_config_default_keeps_models_loaded_for_a_session(self):
        self.assertEqual(Config().keep_alive, "30m")


if __name__ == "__main__":
    unittest.main()
