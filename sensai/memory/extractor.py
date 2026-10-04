"""Asks the model which memory operations an exchange calls for (M3).

After each exchange, the model reads the last learner message and tutor
answer, plus what is already known, and returns a list of operations
(add_word, review_word, add_mistake...) as JSON constrained by a schema.
learner.LearnerMemory.apply() then validates and applies them.

This runs as a separate call rather than as tool calls inside the tutor's
answer: the tutor stays focused on teaching and keeps streaming, and a
small model is far more reliable at one narrow extraction task than at
deciding mid-answer when to call a tool.
"""

import json

from ..ollama_client import ChatStats, OllamaClient, OllamaError
from .learner import OPS, Mistake, Word, norm

SCHEMA = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": OPS},
                    "word": {"type": "string"},
                    "translation": {"type": "string"},
                    "correct": {"type": "boolean"},
                    "pattern": {"type": "string"},
                    "correction": {"type": "string"},
                    "example": {"type": "string"},
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["op"],
            },
        }
    },
    "required": ["operations"],
}

SYSTEM_PROMPT = """Tu tiens à jour la mémoire d'un apprenant en langues. On te donne un échange entre l'apprenant et son tuteur. Tu réponds uniquement avec du JSON : {"operations": [...]}.

Opérations possibles :
- add_word (word, translation) : un mot ou une expression courte (3 mots au plus) de la langue cible a été enseigné ou utilisé. word = le mot dans l'écriture qu'utilise l'apprenant (en romaji s'il écrit en romaji), sans ponctuation ; translation = sa traduction en quelques mots, pas une explication. Pas de nom propre, pas de phrase entière.
- review_word (word, correct, translation) : l'apprenant a utilisé ce mot ou a donné son sens (exercice ou question). correct = true s'il avait juste, false sinon. Un mot mal traduit est un review_word avec correct = false, pas un add_mistake.
- delete_word (word) : l'apprenant demande d'oublier ce mot, ou il a été enregistré par erreur.
- add_mistake (pattern, correction, example) : l'apprenant a fait une erreur de grammaire ou de construction qui peut se répéter. pattern = description courte et générale réutilisable (ex. « confond wa et ga »), correction = la règle en une phrase, example = la phrase fautive de l'apprenant, recopiée telle quelle.
- resolve_mistake (pattern) : l'apprenant montre qu'il maîtrise maintenant une erreur déjà connue.
- set_profile (key, value) : l'apprenant donne une information durable sur lui (langue_cible, langue_maternelle, niveau, objectif…) qui est absente du profil ou différente de celui-ci.

Règles :
- N'invente rien : ne note que ce qui apparaît dans l'échange.
- Un mot nouveau donné par le tuteur, même au détour d'une correction, est un add_word.
- Pour une erreur ou un mot déjà connu, réutilise exactement le même texte que dans la liste fournie.
- Si rien n'est à retenir, renvoie {"operations": []}."""


class ExtractionError(Exception):
    pass


class MemoryExtractor:
    def __init__(self, client: OllamaClient):
        self.client = client

    def extract(
        self,
        user_message: str,
        assistant_message: str,
        profile: dict[str, str],
        known_words: list[Word],
        known_mistakes: list[Mistake],
    ) -> list[dict]:
        parts = []
        if profile:
            parts.append("Profil : " + ", ".join(f"{k} = {v}" for k, v in profile.items()))
        if known_words:
            parts.append("Mots déjà connus : " + "; ".join(f"{w.word} = {w.translation}" for w in known_words))
        if known_mistakes:
            parts.append("Erreurs déjà connues : " + "; ".join(m.pattern for m in known_mistakes))
        parts.append(f"Apprenant : {user_message}\nTuteur : {assistant_message}")
        try:
            raw = self.client.chat(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n\n".join(parts)}],
                ChatStats(),
                schema=SCHEMA,
                options={"temperature": 0},
                purpose="memory_extraction",
            )
        except OllamaError as exc:
            raise ExtractionError(str(exc)) from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ExtractionError(f"invalid JSON from model: {exc}") from exc
        ops = data.get("operations") if isinstance(data, dict) else None
        if not isinstance(ops, list):
            raise ExtractionError("missing 'operations' list")
        return [op for op in ops if isinstance(op, dict) and not _repeats_profile(op, profile)]


def _repeats_profile(op: dict, profile: dict[str, str]) -> bool:
    """A set_profile that restates the current profile is not a change: drop it."""
    key, value = op.get("key"), op.get("value")
    if op.get("op") != "set_profile" or not isinstance(key, str) or not isinstance(value, str):
        return False
    return profile.get(norm(key).replace(" ", "_")) == value.strip()
