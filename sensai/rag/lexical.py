"""R2: lexical side of the hybrid search (BM25, standard library only).

The embedding catches paraphrases; BM25 catches exact words (a rare vocabulary
item, a grammar term). Tokenization has to work for a language tutor:
- Latin words are lowercased and stripped of accents ("leçon" matches "lecon");
- runs of Japanese/Chinese characters have no spaces, so they are cut into
  character bigrams (a lone character stays as it is). Accents are NOT stripped
  there: that would remove the dakuten ("が" would become "か").
"""

import math
import re
import unicodedata
from collections import Counter

_CJK = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff"
_TOKEN = re.compile(rf"[{_CJK}]+|(?:(?![{_CJK}])[^\W_])+")
_IS_CJK = re.compile(rf"[{_CJK}]+")


def _strip_accents(word: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", word) if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN.findall(text.lower()):
        if _IS_CJK.fullmatch(raw):
            if len(raw) == 1:
                tokens.append(raw)
            else:
                tokens.extend(raw[i:i + 2] for i in range(len(raw) - 1))
        else:
            tokens.append(_strip_accents(raw))
    return tokens


def bm25_scores(query: list[str], docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """BM25 score of every document for the query tokens (0.0 when nothing matches)."""
    if not docs or not query:
        return [0.0] * len(docs)
    n = len(docs)
    avg_len = sum(len(d) for d in docs) / n or 1.0
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(d))
    scores: list[float] = []
    for d in docs:
        tf = Counter(d)
        norm = k1 * (1 - b + b * len(d) / avg_len)
        score = 0.0
        for term in set(query):
            f = tf.get(term, 0)
            if f:
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * f * (k1 + 1) / (f + norm)
        scores.append(score)
    return scores
