"""Feature switches: turn a feature off at runtime to show what it brings.

Each switch gates one source of context (or one retrieval stage) that the chat
reads at every turn, so the very next message, or a side-by-side comparison
(compare.py), shows the difference without restarting anything.

The switches start from config.json and live in memory only: a demo can cut
the RAG for a few messages, then « Rétablir » puts the configuration back.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Feature:
    id: str
    code: str  # id in the feature catalogue
    label: str
    group: str
    does: str  # what it brings
    when_off: str  # what to look at once it is cut
    requires: str | None = None  # another switch this one depends on


FEATURES = [
    Feature(
        "history", "B1", "Historique de la conversation", "Conversation",
        "Le modèle reçoit les échanges précédents de la session (et leur résumé).",
        "Le tuteur ne voit plus que le message en cours : demande-lui « qu'est-ce que je viens de te dire ? ».",
    ),
    Feature(
        "profile", "M1", "Profil de l'apprenant", "Conversation",
        "Langue cible, niveau, objectif… sont ajoutés au rôle du tuteur.",
        "Le tuteur ne sait plus quelle langue tu apprends, ni à quel niveau.",
    ),
    Feature(
        "memory", "M3", "Mémoire de l'apprenant", "Mémoire",
        "Mots vus (révision espacée) et erreurs récurrentes sont injectés, puis mis à jour après chaque réponse.",
        "Les exercices ne réutilisent plus tes mots à réviser, et rien de nouveau n'est mémorisé.",
    ),
    Feature(
        "compression", "M2", "Résumé des anciens échanges", "Mémoire",
        "Quand la fenêtre se remplit, les vieux échanges sont résumés au lieu d'être coupés.",
        "Les plus vieux messages sortent du contexte sans laisser de trace.",
    ),
    Feature(
        "rag", "R1", "Leçons (RAG)", "RAG",
        "Les passages de tes leçons les plus proches du message sont ajoutés au prompt.",
        "Le tuteur répond de culture générale : il ignore le contenu, les exemples et les consignes de tes leçons.",
    ),
    Feature(
        "rag_hybrid", "R2", "Recherche hybride (BM25)", "RAG",
        "Le score par mots-clés (BM25) s'ajoute au score sémantique (cosinus) pour classer les passages.",
        "Seul le cosinus classe les passages : un mot rare (« sanbon », « kippu ») pèse moins lourd.",
        requires="rag",
    ),
    Feature(
        "rag_rerank", "R2", "Reclassement par le modèle", "RAG",
        "Le modèle note chaque passage candidat de 0 à 10 et écarte ceux qui n'aident pas à répondre.",
        "Les passages « du même thème » mais inutiles passent : plus de tokens, réponse moins ciblée.",
        requires="rag",
    ),
    Feature(
        "rag_filters", "R2", "Filtre par langue du profil", "RAG",
        "Seules les leçons de ta langue cible (métadonnée « langue ») sont cherchées.",
        "Une leçon d'une autre langue peut se glisser dans le prompt (ex. le coréen pour un apprenant de japonais).",
        requires="rag",
    ),
]

BY_ID = {f.id: f for f in FEATURES}


class FeatureError(ValueError):
    pass


def defaults(config, rag_available: bool) -> dict[str, bool]:
    """Initial state of every switch, from config.json."""
    return {
        "history": True,
        "profile": True,
        "memory": config.memory,
        "compression": config.compression,
        "rag": rag_available,
        "rag_hybrid": config.rag_hybrid,
        "rag_rerank": config.rag_rerank,
        "rag_filters": bool(config.rag_profile_filters),
    }


def check(feature_id: str) -> Feature:
    feature = BY_ID.get(feature_id)
    if feature is None:
        raise FeatureError(f"fonctionnalité inconnue : {feature_id} (connues : {', '.join(BY_ID)})")
    return feature


def describe(switches: dict[str, bool], available: dict[str, bool]) -> list[dict]:
    """For the web UI and /features: every switch with its state and explanation."""
    return [{**asdict(f), "enabled": switches[f.id], "available": available[f.id]} for f in FEATURES]


def disabled(switches: dict[str, bool], available: dict[str, bool]) -> list[str]:
    """Ids of the switches turned off by choice; unavailable features and the sub-switches
    of a cut feature are implied, not listed."""
    off = [f.id for f in FEATURES if not switches[f.id] and available[f.id]]
    return [i for i in off if not (BY_ID[i].requires and not switches[BY_ID[i].requires])]
