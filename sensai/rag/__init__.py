"""R1 + R2: retrieval-augmented generation (see docs/rag.md)."""

from .filters import Filter, parse_filter
from .pipeline import Hit, Rag, SyncReport

__all__ = ["Filter", "Hit", "Rag", "SyncReport", "parse_filter"]
