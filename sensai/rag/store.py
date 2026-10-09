"""R1: local vector store in SQLite (standard library only).

Embeddings are float32 BLOBs. Search is a brute-force dot product in Python:
fine for a few thousand chunks (a course, a documentation folder), not for
millions (see docs/rag.md for the limit). The index is a cache of the source
files: deleting the database file is always safe, it is rebuilt at startup.
"""

import json
import sqlite3
import time
from array import array
from dataclasses import dataclass
from pathlib import Path

from .chunker import Chunk, Document

SCHEMA = """
CREATE TABLE IF NOT EXISTS rag_docs (
    source TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,   -- sha256 of the file | embedding model | chunk size
    meta TEXT NOT NULL,          -- JSON metadata, shared by all chunks of the document
    indexed_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS rag_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    ord INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_source ON rag_chunks(source);
"""


@dataclass
class StoredChunk:
    id: int
    source: str
    ord: int
    text: str
    meta: dict[str, str]
    vec: array


class RagStore:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def state(self) -> dict[str, str]:
        """source -> fingerprint, for incremental indexing."""
        return dict(self.conn.execute("SELECT source, fingerprint FROM rag_docs"))

    def replace_document(self, doc: Document, fingerprint: str, chunks: list[Chunk], vectors: list[array]) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM rag_chunks WHERE source = ?", (doc.source,))
            self.conn.execute(
                "INSERT OR REPLACE INTO rag_docs(source, fingerprint, meta, indexed_at) VALUES (?, ?, ?, ?)",
                (doc.source, fingerprint, json.dumps(doc.meta, ensure_ascii=False), time.time()),
            )
            self.conn.executemany(
                "INSERT INTO rag_chunks(source, ord, text, embedding) VALUES (?, ?, ?, ?)",
                [(doc.source, c.ord, c.text, v.tobytes()) for c, v in zip(chunks, vectors)],
            )

    def delete_except(self, keep: set[str]) -> int:
        """Drop documents that no longer exist on disk; returns how many were removed."""
        gone = [s for s in self.state() if s not in keep]
        with self.conn:
            for source in gone:
                self.conn.execute("DELETE FROM rag_chunks WHERE source = ?", (source,))
                self.conn.execute("DELETE FROM rag_docs WHERE source = ?", (source,))
        return len(gone)

    def load_chunks(self) -> list[StoredChunk]:
        metas = {s: json.loads(m) for s, m in self.conn.execute("SELECT source, meta FROM rag_docs")}
        chunks: list[StoredChunk] = []
        rows = self.conn.execute("SELECT id, source, ord, text, embedding FROM rag_chunks ORDER BY source, ord")
        for chunk_id, source, ord_, text, blob in rows:
            vec = array("f")
            vec.frombytes(blob)
            chunks.append(StoredChunk(chunk_id, source, ord_, text, metas.get(source, {}), vec))
        return chunks

    def stats(self) -> dict[str, int]:
        docs = self.conn.execute("SELECT COUNT(*) FROM rag_docs").fetchone()[0]
        chunks = self.conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
        return {"documents": docs, "chunks": chunks}
