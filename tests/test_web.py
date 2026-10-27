"""X1: the Duolingoy web UI — engine (one worker thread, streamed events, stop,
confirmations from the browser) and HTTP API (served by a real local uvicorn).

No model: a scripted fake client stands in for Ollama.
"""

import http.server
import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

import requests
import uvicorn

from sensai.config import Config
from sensai.monitoring import load_events
from sensai.observability import EventLogger
from sensai.ollama_client import ChatStats, ModelNotFound, OllamaClient, OllamaUnavailable, Placement
from sensai.web.app import create_app
from sensai.web.engine import Engine


def react(action, summary="étape", answer="", **args):
    return json.dumps({"step_summary": summary, "action": action, "args": args, "answer": answer})


class FakeClient:
    """Stands in for OllamaClient: streamed answer, memory operations, scripted agent steps."""

    def __init__(self, answer=("Neko ", "= chat", ".")):
        self.model = "fake"
        self.answer = list(answer)
        self.steps: list[str] = []
        self.release: threading.Event | None = None  # when set, the stream waits after its first piece
        self.silent = False  # the stream says nothing until abort_streams(), like a long prompt evaluation
        self.cut = threading.Event()

    def check(self):
        if self.model.startswith("absent"):
            raise ModelNotFound(f"model '{self.model}' not found")

    def load(self):
        return self.placement()

    def placement(self):
        return Placement(1000, 400)  # Ollama put 40 % of the model on the GPU

    def abort_streams(self):
        self.cut.set()

    def chat_stream(self, messages, stats, purpose="chat", interaction_id=None):
        if self.silent:
            self.cut.wait(5)
            raise OllamaUnavailable("connection lost during generation")
        for i, piece in enumerate(self.answer):
            if i == 1 and self.release is not None:
                self.release.wait(5)
            yield piece
        stats.eval_count = len(self.answer)

    def chat(self, messages, stats, schema=None, options=None, purpose="chat", interaction_id=None):
        if purpose == "memory_extraction":
            return json.dumps({"operations": [{"op": "add_word", "word": "neko", "translation": "chat"}]})
        if purpose == "react_step":
            return self.steps.pop(0)
        return json.dumps({"apprenant": [], "vocabulaire": [], "erreurs": [], "exercices": [], "resume": "r", "en_cours": ""})


class WebTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name).resolve()
        self.workspace = self.base / "workspace"
        self.client = FakeClient()
        config = Config(
            db_path=str(self.base / "sensai.db"),
            log_path=str(self.base / "events.jsonl"),
            user="camille",
            rag=False,
            file_roots={str(self.workspace): "rw"},
            confirm_timeout=5,
        )
        self.engine = Engine(config, client=self.client, logger=EventLogger(config.log_path))
        self.engine.call(lambda chat: chat.store.set_profile("camille", "langue_cible", "japonais"))

    def tearDown(self):
        self.engine.close()
        self.tmp.cleanup()


class EngineTest(WebTestCase):
    def test_chat_turn_streams_tokens_then_report_and_memory(self):
        events = list(self.engine.chat_events("Comment dit-on chat ?"))
        types = [e["type"] for e in events]
        self.assertEqual(types[:3], ["token"] * 3)
        self.assertEqual("".join(e["text"] for e in events if e["type"] == "token"), "Neko = chat.")
        done = next(e for e in events if e["type"] == "done")
        self.assertFalse(done["interrupted"])
        self.assertEqual(done["context"]["num_ctx"], 8192)
        self.assertGreater(done["context"]["history"], 0)
        self.assertEqual(done["ollama"]["output_tokens"], 3)
        self.assertEqual(done["placement"], {"gpu_share": 0.4, "label": "40 % GPU / 60 % CPU"})
        self.assertIn("mémoire : 1× add_word", [e.get("text") for e in events if e["type"] == "notice"][0])
        memory = events[-1]
        self.assertEqual(memory["type"], "memory")
        self.assertEqual([w["word"] for w in memory["words"]], ["neko"])
        self.assertEqual(memory["stats"]["due"], 1)

    def test_stop_keeps_the_part_already_shown(self):
        self.client.release = threading.Event()
        tokens = []
        for event in self.engine.chat_events("neko ?"):
            if event["type"] == "token":
                tokens.append(event["text"])
                if len(tokens) == 1:
                    self.engine.stop()
                    self.client.release.set()
            if event["type"] == "done":
                self.assertTrue(event["interrupted"])
        self.assertEqual(tokens, ["Neko ", "= chat"])
        history = self.engine.call(lambda chat: [m.content for m in chat.history])
        self.assertEqual(history, ["neko ?", "Neko = chat"])

    def test_stop_before_the_first_word(self):
        self.client.silent = True
        threading.Timer(0.2, self.engine.stop).start()
        start = time.perf_counter()
        events = list(self.engine.chat_events("neko ?"))
        self.assertLess(time.perf_counter() - start, 3)  # not the 5 s the stream would have waited
        self.assertTrue(self.client.cut.is_set())
        done = next(e for e in events if e["type"] == "done")
        self.assertTrue(done["interrupted"])
        self.assertNotIn("error", [e["type"] for e in events])
        self.assertEqual(self.engine.call(lambda chat: chat.history), [])  # nothing answered: forgotten

    def test_agent_asks_the_browser_before_writing(self):
        target = str(self.workspace / "fiche.md")
        self.client.steps = [
            react("write_file", "J'écris la fiche", path=target, content="neko = chat"),
            react("final_answer", "Terminé", answer="Fiche écrite."),
        ]
        events = []
        for event in self.engine.agent_events("Écris une fiche"):
            events.append(event)
            if event["type"] == "confirm":
                self.assertEqual(event["action"], "write_file")
                self.assertIn("+neko = chat", event["summary"])
                self.assertTrue(self.engine.approver.answer(event["id"], True))
                self.assertFalse(self.engine.approver.answer(event["id"], True))  # answered once only
        self.assertEqual([e["type"] for e in events], ["confirm", "step", "step", "done"])
        self.assertEqual(events[-1]["answer"], "Fiche écrite.")
        self.assertEqual((self.workspace / "fiche.md").read_text(), "neko = chat")

    def test_stop_during_a_confirmation_refuses_and_ends_the_task(self):
        target = str(self.workspace / "fiche.md")
        self.client.steps = [react("write_file", path=target, content="x"), react("final_answer", answer="jamais lu")]
        types = []
        for event in self.engine.agent_events("Écris une fiche"):
            types.append(event["type"])
            if event["type"] == "confirm":
                self.engine.stop()
        self.assertEqual(types, ["confirm", "step", "stopped"])
        self.assertFalse((self.workspace / "fiche.md").exists())

    def test_closing_the_page_refuses_the_pending_confirmation(self):
        target = str(self.workspace / "fiche.md")
        self.client.steps = [react("write_file", path=target, content="x"), react("final_answer", answer="fin")]
        stream = self.engine.agent_events("Écris une fiche")
        self.assertEqual(next(stream)["type"], "confirm")
        stream.close()  # the browser tab is closed: the request is refused and the task stopped
        self.assertIsNone(self.engine.call(lambda chat: chat.last_agent))
        self.assertFalse((self.workspace / "fiche.md").exists())
        decisions = [e["decision"] for e in load_events(self.engine.config.log_path) if e["event"] == "confirmation"]
        self.assertEqual(decisions, ["refused"])

    def test_failures_are_events_and_the_worker_survives(self):
        self.client.steps = ["pas du json"] * 5
        events = list(self.engine.agent_events("tâche"))
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["outcome"], "error")

        def boom(*args, **kwargs):
            raise RuntimeError("panne")
            yield  # pragma: no cover

        self.client.chat_stream = boom
        self.assertEqual([e["type"] for e in self.engine.chat_events("bonjour")], ["error"])
        self.assertEqual(self.engine.call(lambda chat: chat.config.user), "camille")

    def test_model_switch_is_checked_and_reverted_on_failure(self):
        self.assertEqual(self.engine.set_model("autre:7b"), "autre:7b")
        with self.assertRaises(ModelNotFound):
            self.engine.set_model("absent:1b")
        self.assertEqual(self.engine.call(lambda chat: chat.client.model), "autre:7b")


class ApiTest(WebTestCase):
    def setUp(self):
        super().setUp()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.engine.config.host = "http://127.0.0.1:9"  # no Ollama behind /api/models
        server = uvicorn.Server(uvicorn.Config(create_app(self.engine), host="127.0.0.1", port=port, log_level="error"))
        self.server = server
        self.thread = threading.Thread(target=server.run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)

    def tearDown(self):
        self.server.should_exit = True
        self.thread.join(timeout=10)
        super().tearDown()

    def events(self, path, body):
        with requests.post(self.url + path, json=body, stream=True, timeout=10) as resp:
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.headers["content-type"].split(";")[0], "text/event-stream")
            return [json.loads(line[5:]) for line in resp.iter_lines(decode_unicode=True) if line.startswith("data:")]

    def test_page_and_assets(self):
        page = requests.get(self.url + "/", timeout=5)
        self.assertEqual(page.status_code, 200)
        self.assertIn("<title>Duolingoy</title>", page.text)
        for asset in ("app.js", "style.css", "mascot.svg"):
            self.assertEqual(requests.get(f"{self.url}/static/{asset}", timeout=5).status_code, 200)
        self.assertEqual(requests.get(self.url + "/api/docs", timeout=5).status_code, 200)

    def test_chat_over_sse_then_state_and_memory(self):
        types = [e["type"] for e in self.events("/api/chat", {"text": "neko ?"})]
        self.assertEqual((types[0], types[-1]), ("token", "memory"))
        self.assertIn("done", types)
        state = requests.get(self.url + "/api/state", timeout=5).json()
        self.assertEqual(state["app"], "Duolingoy")
        self.assertEqual([m["content"] for m in state["messages"]], ["neko ?", "Neko = chat."])
        self.assertEqual(state["features"], {"memory": True, "compression": True, "rag": False, "agent": True})
        memory = requests.get(self.url + "/api/memory", timeout=5).json()
        self.assertEqual(memory["stats"]["words"], 1)
        self.assertEqual(requests.delete(self.url + "/api/vocab/neko", timeout=5).json()["stats"]["words"], 0)
        self.assertEqual(requests.delete(self.url + "/api/vocab/neko", timeout=5).status_code, 404)

    def test_profile_sessions_and_compression(self):
        resp = requests.put(self.url + "/api/profile", json={"key": "Langue Maternelle", "value": " français "}, timeout=5)
        self.assertEqual(resp.json()["profile"]["langue_maternelle"], "français")
        self.assertEqual(requests.delete(self.url + "/api/profile/langue_maternelle", timeout=5).status_code, 200)
        self.assertEqual(requests.delete(self.url + "/api/profile/langue_maternelle", timeout=5).status_code, 404)
        self.assertEqual(requests.put(self.url + "/api/profile", json={"key": "<b>", "value": "x"}, timeout=5).status_code, 422)

        first = requests.get(self.url + "/api/state", timeout=5).json()["session"]
        self.events("/api/chat", {"text": "un"})
        self.events("/api/chat", {"text": "deux"})
        compressed = requests.post(self.url + "/api/compress", timeout=10).json()
        self.assertTrue(compressed["compressed"])
        self.assertGreater(compressed["state"]["summarized_until"], 0)
        fresh = requests.post(self.url + "/api/sessions", timeout=5).json()
        self.assertNotEqual(fresh["session"], first)
        self.assertEqual(fresh["messages"], [])
        sessions = requests.get(self.url + "/api/sessions", timeout=5).json()["sessions"]
        self.assertEqual([s["current"] for s in sessions if s["id"] == fresh["session"]], [True])
        reloaded = requests.post(f"{self.url}/api/sessions/{first}/load", timeout=5).json()
        self.assertEqual(len(reloaded["messages"]), 4)
        self.assertEqual(requests.post(self.url + "/api/sessions/999/load", timeout=5).status_code, 404)

    def test_agent_confirmation_through_http(self):
        target = str(self.workspace / "quiz.py")
        self.client.steps = [react("write_file", path=target, content="print(1)"), react("final_answer", answer="ok")]
        events = []
        with requests.post(self.url + "/api/agent", json={"task": "écris"}, stream=True, timeout=10) as resp:
            for line in resp.iter_lines(decode_unicode=True):
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[5:])
                events.append(event["type"])
                if event["type"] == "confirm":
                    answer = requests.post(self.url + "/api/confirm", json={"id": event["id"], "approved": False}, timeout=5)
                    self.assertEqual(answer.status_code, 200)
        self.assertEqual(events, ["confirm", "step", "step", "done"])
        self.assertFalse(Path(target).exists())
        self.assertEqual(requests.post(self.url + "/api/confirm", json={"id": 999, "approved": True}, timeout=5).status_code, 409)

    def test_cross_site_requests_are_rejected(self):
        # A page from another site can only send "simple" requests (no JSON content type, no preflight)...
        resp = requests.post(self.url + "/api/agent", data=json.dumps({"task": "efface tout"}), headers={"Content-Type": "text/plain"}, timeout=5)
        self.assertEqual(resp.status_code, 422)
        form = requests.post(self.url + "/api/chat", data={"text": "x"}, timeout=5)
        self.assertEqual(form.status_code, 422)
        # ...and a rebinding domain that resolves to 127.0.0.1 still sends its own Host header.
        rebound = requests.get(self.url + "/api/state", headers={"Host": "evil.example:8000"}, timeout=5)
        self.assertEqual(rebound.status_code, 400)
        self.assertNotIn("access-control-allow-origin", {k.lower() for k in resp.headers})

    def test_learners_and_their_history(self):
        self.events("/api/chat", {"text": "neko ?"})
        users = requests.get(self.url + "/api/users", timeout=5).json()["users"]
        self.assertEqual([(u["name"], u["current"], u["words"]) for u in users], [("camille", True, 1)])
        history = requests.get(self.url + "/api/history", timeout=5).json()
        self.assertEqual((history["messages"], history["streak"], history["sessions_count"]), (1, 1, 1))
        self.assertEqual(len(history["days"]), 14)
        self.assertEqual(history["days"][-1]["messages"], 1)
        session = history["sessions"][0]["id"]

        created = requests.post(self.url + "/api/users", json={"name": "hugo"}, timeout=5).json()
        self.assertEqual((created["user"], created["messages"], created["profile"]), ("hugo", [], {}))
        self.assertEqual(requests.post(self.url + "/api/users", json={"name": "hugo"}, timeout=5).status_code, 409)
        self.assertEqual(requests.post(self.url + "/api/users", json={"name": "../x"}, timeout=5).status_code, 422)
        self.assertEqual(requests.get(f"{self.url}/api/sessions/{session}", timeout=5).status_code, 404)  # camille's
        self.assertEqual(requests.delete(self.url + "/api/users/hugo", timeout=5).status_code, 409)  # current

        back = requests.post(self.url + "/api/users/camille/select", timeout=5).json()
        self.assertEqual((back["session"], len(back["messages"])), (session, 2))
        read = requests.get(f"{self.url}/api/sessions/{session}", timeout=5).json()
        self.assertEqual([m["role"] for m in read["messages"]], ["user", "assistant"])
        self.assertEqual(requests.post(self.url + "/api/users/nobody/select", timeout=5).status_code, 404)
        remaining = requests.delete(self.url + "/api/users/hugo", timeout=5).json()["users"]
        self.assertEqual([u["name"] for u in remaining], ["camille"])

    def test_feature_switches_and_comparison(self):
        switches = {f["id"]: f for f in requests.get(self.url + "/api/features", timeout=5).json()["features"]}
        self.assertTrue(switches["memory"]["enabled"])
        self.assertFalse(switches["rag"]["available"])
        off = requests.put(self.url + "/api/features/memory", json={"enabled": False}, timeout=5).json()["features"]
        self.assertFalse(next(f for f in off if f["id"] == "memory")["enabled"])
        self.assertEqual(requests.put(self.url + "/api/features/rag", json={"enabled": True}, timeout=5).status_code, 409)
        self.assertEqual(requests.put(self.url + "/api/features/nope", json={"enabled": True}, timeout=5).status_code, 404)

        events = self.events("/api/chat", {"text": "neko ?"})
        done = next(e for e in events if e["type"] == "done")
        self.assertEqual(done["off"], ["memory"])
        state = requests.get(self.url + "/api/state", timeout=5).json()
        self.assertEqual(state["messages"][-1]["meta"], {"off": ["memory"]})
        self.assertFalse(state["features"]["memory"])
        requests.post(self.url + "/api/features/reset", timeout=5)

        self.client.answer = ["Tu ", "es Camille."]
        events = self.events("/api/compare", {"text": "Comment je m'appelle ?", "feature": "history", "judge": False})
        self.assertEqual([e["side"] for e in events if e["type"] == "side"], ["on", "off"])
        self.assertEqual(events[-1]["type"], "end")
        self.assertEqual(len(requests.get(self.url + "/api/state", timeout=5).json()["messages"]), 2)  # nothing stored
        self.assertEqual(requests.post(self.url + "/api/compare", json={"text": "x", "feature": "rag"}, timeout=5).status_code, 409)
        self.assertEqual(requests.post(self.url + "/api/compare", json={"text": "x", "feature": "nope"}, timeout=5).status_code, 404)

    def test_input_validation_models_and_metrics(self):
        self.assertEqual(requests.post(self.url + "/api/chat", json={"text": "   "}, timeout=5).status_code, 422)
        self.assertEqual(requests.post(self.url + "/api/chat", json={"text": "x" * 4001}, timeout=5).status_code, 422)
        self.assertEqual(requests.get(self.url + "/api/models", timeout=10).status_code, 502)
        self.assertEqual(requests.put(self.url + "/api/model", json={"model": "absent:1b"}, timeout=5).status_code, 400)
        switched = requests.put(self.url + "/api/model", json={"model": "autre:7b"}, timeout=5).json()
        self.assertEqual(switched, {"model": "autre:7b", "placement": {"gpu_share": 0.4, "label": "40 % GPU / 60 % CPU"}})
        self.assertEqual(requests.get(self.url + "/api/state", timeout=5).json()["placement"]["label"], "40 % GPU / 60 % CPU")
        metrics = requests.get(self.url + "/api/metrics", timeout=5).json()
        self.assertIn("model_calls", metrics)


if __name__ == "__main__":
    unittest.main()


class SilentOllama(http.server.BaseHTTPRequestHandler):
    """Accepts a chat request, then says nothing for 5 s (a long prompt evaluation)."""

    protocol_version = "HTTP/1.1"

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        time.sleep(5)
        try:
            self.wfile.write(b"0\r\n\r\n")
        except OSError:
            pass

    def log_message(self, *args):
        pass


class AbortStreamTest(unittest.TestCase):
    def test_abort_wakes_a_stream_waiting_for_its_first_word(self):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), SilentOllama)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        client = OllamaClient(f"http://127.0.0.1:{server.server_port}", "fake", 2048)
        stream = client.chat_stream([{"role": "user", "content": "neko ?"}], ChatStats())
        threading.Timer(0.3, client.abort_streams).start()
        start = time.perf_counter()
        with self.assertRaises(OllamaUnavailable):
            list(stream)
        self.assertLess(time.perf_counter() - start, 3)
        self.assertEqual(client._streams, set())
