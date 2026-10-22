"""Checks a grounded answer against the lesson chunks it was given (R2).

When lesson chunks were put in the prompt, the code looks in the answer for
the facts a learner acts on and a model easily invents: dates, times, rooms,
prices, scores, durations, percentages, years, decimal figures. Each one must
appear in the injected chunks or in the learner's own message; otherwise it
is reported as « absent de tes leçons ». No model call: it takes a millisecond
and cannot itself hallucinate.

What it catches: a fact that is in none of the sources (an invented date, a
room that does not exist, « noté sur 10 points » when the grid says 4).
What it does not catch: a real fact used in the wrong place (a date copied
from another line of the planning), or an error in Japanese vocabulary.
"""

import re
import unicodedata
from dataclasses import dataclass, field

MONTHS = "janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|octobre|novembre|decembre"

# Applied to the answer and to the sources alike, so that « 14 h », « 14h » and « 14h00 »,
# or « 1 450 € » and « 1450 euros », become the same text before comparing.
_NORMALIZE: list[tuple[re.Pattern, str]] = [
    (re.compile(r"[   ]"), " "),
    (re.compile(r"(?<=\d) (?=\d{3}(?!\d))"), ""),  # 1 450 -> 1450
    (re.compile(r"(?<=\d),(?=\d)"), "."),  # 13,4 -> 13.4
    (re.compile(r"\b1er\b"), "1"),
    (re.compile(r"(\d) ?(?:heures?|h)(?: ?(\d{2}))?\b"), lambda m: f"{m[1]}h{m[2] or ''}"),
    (re.compile(r"(\d)h00\b"), r"\1h"),
    (re.compile(r"(\d) ?(?:minutes?|min)\b"), r"\1 min"),
    (re.compile(r"(\d) ?(?:points?|pts?)\b"), r"\1 pt"),
    (re.compile(r"(\d) ?(?:euros?|€)"), r"\1 €"),
    (re.compile(r"(\d) ?(?:yens?|円)"), r"\1 yen"),
    (re.compile(r"(\d) ?(?:wons?|원)"), r"\1 won"),
    (re.compile(r"(\d) ?/ ?20\b"), r"\1/20"),
    (re.compile(r"(\d) ?%"), r"\1 %"),
    (re.compile(r"[ \t]+"), " "),
]

# (kind, pattern) of the facts worth checking, on normalized text.
FACTS: list[tuple[str, re.Pattern]] = [
    ("date", re.compile(rf"\b\d{{1,2}} (?:{MONTHS})\b")),
    ("heure", re.compile(r"\b\d{1,2}h(?:\d{2})?\b")),
    ("durée", re.compile(r"\b\d+ min\b")),
    ("salle", re.compile(r"\b[a-z]\d{3}\b")),
    ("prix", re.compile(r"\b\d+(?:\.\d+)? (?:€|yen|won)")),
    ("note", re.compile(r"\b\d+(?:\.\d+)?/20\b")),
    ("points", re.compile(r"\b\d+(?:\.\d+)? pt\b")),
    ("pourcentage", re.compile(r"\b\d+(?:\.\d+)? %")),
    ("année", re.compile(r"\b20\d{2}\b")),
    ("nombre décimal", re.compile(r"(?<![\d./])\d+\.\d+(?!\d|\.\d|/| pt| €| yen| won| %)")),
]
TABLE_NUMBER = re.compile(r"\|\s*(\d+(?:\.\d+)?)\s*(?=\|)")  # a grid cell: | Kana | 4 | …


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    for pattern, repl in _NORMALIZE:
        text = pattern.sub(repl, text)
    return text


def facts(text: str) -> list[tuple[str, str]]:
    """(kind, fact) found in `text`, in order, without duplicates."""
    norm = normalize(text)
    found: dict[str, str] = {}
    spans: list[tuple[int, int]] = []
    for kind, pattern in FACTS:
        for m in pattern.finditer(norm):
            if any(a <= m.start() < b for a, b in spans):
                continue  # « 13.4/20 » is one score, not also a decimal number
            spans.append(m.span())
            found.setdefault(m.group().replace(" ", ""), kind)
    return [(kind, key) for key, kind in found.items()]


@dataclass
class GroundingCheck:
    checked: list[str] = field(default_factory=list)  # facts of the answer that were looked up
    unsupported: list[str] = field(default_factory=list)  # ... and found in no source

    def to_dict(self) -> dict:
        return {"checked": len(self.checked), "unsupported": self.unsupported}

    def describe(self) -> str:
        if self.unsupported:
            return (f"{len(self.unsupported)} fait(s) sur {len(self.checked)} absent(s) des leçons utilisées : "
                    + ", ".join(f"« {f} »" for f in self.unsupported))
        return f"{len(self.checked)} fait(s) vérifié(s) dans les leçons utilisées"


def check_answer(answer: str, sources: list[str], question: str = "") -> GroundingCheck:
    """Look up every checkable fact of `answer` in the lesson `sources` (and the learner's `question`)."""
    reference = "\n".join([*sources, question])
    known = {key for _, key in facts(reference)}
    norm_ref = normalize(reference)
    cells = set(TABLE_NUMBER.findall(norm_ref))
    check = GroundingCheck()
    for kind, key in facts(answer):
        check.checked.append(key)
        supported = key in known
        if not supported and kind == "points":
            # a grading grid gives the points as bare numbers in a table: | Kana | 4 |
            supported = key.removesuffix("pt") in cells
        if not supported:
            check.unsupported.append(key)
    return check
