"""Pre-demo checks (python -m sensai.preflight), with a fake Ollama and the hashing embedder of test_rag."""

import tempfile
import unittest
from pathlib import Path

from sensai import preflight
from sensai.config import Config
from sensai.memory.store import MemoryStore
from sensai.ollama_client import ModelNotFound, OllamaUnavailable, Placement
from tests.test_rag import make_rag


class FakeClient:
    model = "fake:7b"

    def __init__(self, vram=4, down=False, missing=False):
        self.vram, self.down, self.missing = vram, down, missing
        self.chats = 0

    def check(self):
        if self.down:
            raise OllamaUnavailable("Ollama unreachable at http://localhost:11434")
        if self.missing:
            raise ModelNotFound("model 'fake:7b' not found; pull it with `ollama pull fake:7b`")

    def load(self):
        return Placement(10, self.vram)

    def chat(self, messages, stats, schema=None, options=None, purpose="chat", interaction_id=None):
        self.chats += 1
        if purpose == "rerank":
            n = messages[-1]["content"].count("\n[")
            return '{"scores": [%s]}' % ", ".join(["8"] * n)
        return "Konnichiwa"


class PreflightTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = Config(db_path=str(Path(self.tmp.name) / "demo.db"), rag=False, log_path=str(Path(self.tmp.name) / "e.jsonl"))
        self.lines: list[str] = []

    def run_checks(self, client, setup=False, rag=None):
        return {c.name: c for c in preflight.run(self.config, setup, self.lines.append, client=client, rag=rag)}

    def test_everything_ready_after_setup(self):
        checks = self.run_checks(FakeClient(), setup=True)
        self.assertEqual(checks["Placement"].status, "ok")
        self.assertEqual(checks["Réponse du modèle"].status, "ok")
        self.assertEqual(checks["Sandbox"].status, "ok")
        self.assertEqual(checks["Apprenant camille"].status, "ok")
        store = MemoryStore(self.config.db_path)
        self.addCleanup(store.close)
        self.assertEqual(store.get_profile("hugo"), {"langue_cible": "coréen", "niveau": "A1"})
        self.assertEqual(checks["RAG"].status, "warn")  # disabled in this config: act 2 needs it
        self.assertIn("avertissement", self.lines[-1])

    def test_without_setup_nothing_is_written(self):
        checks = self.run_checks(FakeClient())
        self.assertEqual(checks["Apprenant camille"].status, "warn")
        self.assertIn("--setup", checks["Apprenant camille"].advice)
        store = MemoryStore(self.config.db_path)
        self.addCleanup(store.close)
        self.assertEqual(store.get_profile("camille"), {})

    def test_ollama_down_or_model_missing_fails_with_the_fix(self):
        for client, hint in ((FakeClient(down=True), "ollama serve"), (FakeClient(missing=True), "ollama pull")):
            checks = self.run_checks(client)
            self.assertEqual(checks["Ollama"].status, "fail")
            self.assertIn(hint, checks["Ollama"].advice + checks["Ollama"].detail)
            self.assertNotIn("Réponse du modèle", checks)  # no point asking a model that is not there
            self.assertEqual(client.chats, 0)

    def test_few_layers_on_the_gpu_is_a_warning(self):
        check = self.run_checks(FakeClient(vram=1))["Placement"]  # 10 % of the model on the GPU
        self.assertEqual(check.status, "warn")
        self.assertIn("redémarrer Ollama", check.advice)

    def test_cpu_only_advises_the_cpu_settings(self):
        check = self.run_checks(FakeClient(vram=0))["Placement"]
        self.assertEqual(check.status, "warn")
        self.assertIn('"rag_rerank": false', check.advice)

    def test_demo_searches_must_retrieve_their_document(self):
        rag = make_rag(self.tmp.name, client=FakeClient())
        checks = preflight.check_rag(rag, [("que veut dire neko en japonais ?", "animaux.md"),
                                           ("que veut dire neko en japonais ?", "interro_vendredi.md")])
        self.assertEqual([c.status for c in checks], ["ok", "ok", "fail"])
        self.assertIn("animaux.md", checks[1].detail)
        self.assertIn("interro_vendredi.md attendu", checks[2].advice)
        rag.store.close()

    def test_check_lines_are_readable(self):
        line = preflight.Check("Sandbox", "fail", "timeout", "installer Docker").line()
        self.assertEqual(line, " ✗ Sandbox — timeout\n     → installer Docker")


if __name__ == "__main__":
    unittest.main()
