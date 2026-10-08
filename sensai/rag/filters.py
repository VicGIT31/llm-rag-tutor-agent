"""R2: metadata filters, applied BEFORE scoring so excluded chunks cost nothing.

Syntax (same as the /rag filter command):
    niveau=A1            equality, case-insensitive
    niveau=A1,A2         one of several values
    date>=2026-09-01     ISO dates compare correctly as strings
    niveau<=B1           works for A1 < A2 < B1 < ... too

A filter is "strict" by default: a document that lacks the key is excluded.
A "soft" filter (derived from the learner profile) lets such documents pass,
so general material without metadata is still found.
"""

import re
from dataclasses import dataclass

_EXPR = re.compile(r"^\s*([\w\-]+)\s*(>=|<=|=)\s*([^=<>\s].*?)\s*$")


@dataclass(frozen=True)
class Filter:
    key: str
    op: str
    values: tuple[str, ...]  # casefolded
    soft: bool = False

    def __str__(self) -> str:
        return f"{self.key}{self.op}{','.join(self.values)}" + (" (profil)" if self.soft else "")

    def matches(self, meta: dict[str, str]) -> bool:
        have = meta.get(self.key)
        if have is None:
            return self.soft
        have = have.casefold()
        if self.op == "=":
            return have in self.values
        if self.op == ">=":
            return have >= self.values[0]
        return have <= self.values[0]


def parse_filter(expr: str) -> Filter:
    m = _EXPR.match(expr)
    if not m:
        raise ValueError(f"filtre invalide : {expr!r} (attendu clé=valeur, clé>=valeur ou clé<=valeur)")
    key, op, raw = m.groups()
    values = tuple(v.strip().casefold() for v in raw.split(",") if v.strip())
    if not values:
        raise ValueError(f"filtre invalide : {expr!r} (valeur vide)")
    if op != "=" and len(values) != 1:
        raise ValueError(f"filtre invalide : {expr!r} ({op} n'accepte qu'une valeur)")
    return Filter(key.lower(), op, values)


def matches_all(meta: dict[str, str], filters: list[Filter]) -> bool:
    return all(f.matches(meta) for f in filters)
