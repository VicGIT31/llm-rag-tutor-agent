"""R1: embeddings through Ollama's HTTP API (POST /api/embed), no framework."""

import math
from array import array

import requests

from ..ollama_client import ModelNotFound, OllamaError, OllamaUnavailable


def normalize(vec: list[float]) -> array:
    """L2-normalized float32 vector: a plain dot product is then the cosine similarity."""
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return array("f", (x / norm for x in vec))


class OllamaEmbedder:
    def __init__(self, host: str, model: str, timeout: float = 120, batch_size: int = 16, gpu: bool = False):
        """gpu=False keeps the embedding model on the CPU. It is small and fast there, and
        on a GPU that the chat model already fills, Ollama would unload one model to load
        the other at every message."""
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size
        self.gpu = gpu

    def check(self) -> int:
        """Fail early if the embedding model is missing; returns the vector size."""
        return len(self.embed(["test"])[0])

    def embed(self, texts: list[str]) -> list[array]:
        out: list[array] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._embed_batch(texts[i:i + self.batch_size]))
        return out

    def _embed_batch(self, batch: list[str]) -> list[array]:
        try:
            payload: dict = {"model": self.model, "input": batch}
            if not self.gpu:
                payload["options"] = {"num_gpu": 0}  # no layer on the GPU
            resp = requests.post(f"{self.host}/api/embed", json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            raise OllamaUnavailable(f"connection to Ollama failed: {exc}") from exc
        if resp.status_code != 200:
            try:
                message = resp.json().get("error", resp.text)
            except ValueError:
                message = resp.text
            if resp.status_code == 404:
                raise ModelNotFound(
                    f"embedding model '{self.model}' not found; pull it with `ollama pull {self.model}`"
                )
            raise OllamaError(f"HTTP {resp.status_code}: {message}")
        try:
            vectors = resp.json()["embeddings"]
        except (ValueError, KeyError) as exc:
            raise OllamaError(f"unreadable embedding response: {exc}") from exc
        if len(vectors) != len(batch) or not all(vectors):
            raise OllamaError("Ollama returned a wrong number of embeddings")
        return [normalize(v) for v in vectors]
