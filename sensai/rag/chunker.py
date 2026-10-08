"""R1: loading documents and cutting them into chunks.

Metadata comes from an optional front matter block at the top of a file:

    ---
    langue: japonais
    niveau: A1
    auteur: Camille
    date: 2026-09-30
    ---

Missing fields are filled from the file itself (`source`, `title`, `date` =
modification day, `folder` = parent directory name). Keys are lowercased.
Use ISO dates (YYYY-MM-DD): date filters compare strings.
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

FRONT_MATTER = re.compile(r"\A\ufeff?---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.S)
HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@dataclass
class Document:
    source: str
    body: str
    meta: dict[str, str]
    fingerprint: str  # sha256 of the raw file: change detection for incremental indexing


@dataclass
class Chunk:
    ord: int
    text: str


@dataclass
class LoadReport:
    skipped: list[str] = field(default_factory=list)  # "path: reason"
    missing_sources: list[str] = field(default_factory=list)


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    m = FRONT_MATTER.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split(":", 1)
        key, value = key.strip().lower(), value.strip().strip("'\"")
        if key and value:
            meta[key] = value
    return meta, text[m.end():]


def _first_heading(body: str) -> str | None:
    for line in body.splitlines():
        m = HEADING.match(line)
        if m:
            return m.group(2)
    return None


def load_documents(
    sources: list[str], exts: tuple[str, ...] = (".md", ".txt"), max_bytes: int = 500_000
) -> tuple[list[Document], LoadReport]:
    """Read every indexable file under `sources` (files or directories). Symlinks are ignored."""
    report = LoadReport()
    files: list[Path] = []
    for src in sources:
        path = Path(src)
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.is_file()))
        elif path.is_file():
            files.append(path)
        else:
            report.missing_sources.append(src)

    docs: list[Document] = []
    seen: set[str] = set()
    for path in files:
        name = path.as_posix()
        if name in seen or path.suffix.lower() not in exts:
            continue
        seen.add(name)
        if path.is_symlink():
            report.skipped.append(f"{name}: lien symbolique")
            continue
        try:
            if path.stat().st_size > max_bytes:
                report.skipped.append(f"{name}: plus de {max_bytes} octets")
                continue
            raw = path.read_text(encoding="utf-8")
            mtime = path.stat().st_mtime
        except (OSError, UnicodeDecodeError) as exc:
            report.skipped.append(f"{name}: {exc}")
            continue
        meta, body = parse_front_matter(raw)
        meta.setdefault("title", _first_heading(body) or path.stem)
        meta.setdefault("date", datetime.fromtimestamp(mtime).strftime("%Y-%m-%d"))
        meta.setdefault("folder", path.parent.name)
        meta["source"] = name
        docs.append(Document(name, body, meta, hashlib.sha256(raw.encode("utf-8")).hexdigest()))
    return docs, report


def _split_long(paragraph: str, max_chars: int, overlap: int) -> list[str]:
    """Break a paragraph longer than max_chars: by lines first, then hard slices with overlap."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    pieces: list[str] = []
    current = ""
    for line in paragraph.splitlines():
        if len(line) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            step = max(1, max_chars - overlap)
            pieces.extend(line[i:i + max_chars] for i in range(0, len(line), step))
        elif current and len(current) + len(line) + 1 > max_chars:
            pieces.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        pieces.append(current)
    return pieces


def chunk_markdown(body: str, title: str, max_chars: int = 700, overlap: int = 80) -> list[Chunk]:
    """Cut a markdown body into chunks that follow its structure.

    Each chunk starts with its heading breadcrumb (or the document title when there
    is none): a small header is enough for the embedding to know what the passage is
    about, e.g. "Leçon 3 — Les animaux (japonais, niveau A1)". Tables and short
    paragraphs stay together; only oversized paragraphs are split.
    """
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        buf.clear()
        if text:
            sections.append((" > ".join(h for _, h in stack) or title, text))

    for line in body.splitlines():
        m = HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2)))
        else:
            buf.append(line)
    flush()

    chunks: list[Chunk] = []
    for breadcrumb, text in sections:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        current = ""
        packed: list[str] = []
        for paragraph in paragraphs:
            for piece in _split_long(paragraph, max_chars, overlap):
                if current and len(current) + len(piece) + 2 > max_chars:
                    packed.append(current)
                    current = piece
                else:
                    current = f"{current}\n\n{piece}" if current else piece
        if current:
            packed.append(current)
        for text_part in packed:
            chunks.append(Chunk(len(chunks), f"{breadcrumb}\n{text_part}"))
    return chunks
