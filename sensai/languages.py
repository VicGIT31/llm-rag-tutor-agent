"""One name per target language, whatever the learner (or the memory extractor) wrote.

The profile's langue_cible is free text: « Japonais », « japanese », « nihongo »
or « 日本語 » all mean the lessons tagged `langue: japonais`. Without a single
name, the RAG language filter excluded every lesson (they are all tagged) and
the learner memory split its vocabulary between « Japonais » and « japonais ».
"""

import unicodedata


def fold(text: str) -> str:
    """Casefolded, without accents: « Coréen » and « coreen » are the same text."""
    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))


# Canonical name (as written in the lessons' metadata) -> other ways to name it.
ALIASES = {
    "japonais": ["japanese", "japones", "nihongo", "日本語", "jp", "ja", "jap"],
    "coréen": ["korean", "coreano", "hangugeo", "hangukeo", "한국어", "ko", "kr"],
    "chinois": ["chinese", "mandarin", "zhongwen", "中文", "汉语", "zh"],
    "anglais": ["english", "en"],
    "espagnol": ["spanish", "espanol", "es"],
    "allemand": ["german", "deutsch", "de"],
    "italien": ["italian", "italiano", "it"],
    "portugais": ["portuguese", "portugues", "pt"],
    "français": ["french", "fr"],
}
_CANONICAL = {fold(alias): name for name, aliases in ALIASES.items() for alias in [name, *aliases]}


def canonical_language(value: str) -> str:
    """The canonical name of a known language; any other one lowercased, spaces collapsed."""
    plain = " ".join(value.lower().split())
    return _CANONICAL.get(fold(plain), plain)
