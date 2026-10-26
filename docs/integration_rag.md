# Brancher le RAG sur le module Mémoire & Contexte

Ce document s'adresse à la personne qui développe le RAG (R1 / R2). Il explique :

1. comment le module Mémoire & Contexte construit le prompt à chaque tour ;
2. où et comment le RAG s'y branche ;
3. le contrat à respecter, les pièges, et comment tester.

Le détail interne de la mémoire (M1, M2, M3) est dans [`memoire_contexte.md`](memoire_contexte.md).

---

## 1. Le principe : une seule fenêtre, un seul responsable

Le modèle reçoit à chaque tour **une seule requête**, limitée par `num_ctx` (8192 tokens par défaut). Tout doit y tenir : le rôle du tuteur, le profil, la mémoire de l'apprenant, le résumé des anciens échanges, **les documents du RAG**, l'historique récent, et la réponse elle-même.

Si la requête dépasse `num_ctx`, **Ollama coupe les plus vieux tokens sans prévenir**. Le modèle peut alors perdre son prompt système sans que personne ne le voie.

C'est pourquoi un seul composant, `ContextBuilder` (`sensai/context.py`), assemble le prompt et répartit le budget. **Le RAG ne construit pas de prompt lui-même** : il fournit des morceaux de texte, et `ContextBuilder` les place.

---

## 2. Ce qui se passe à chaque message

Dans `Chat.send()` (`sensai/cli.py`), dans cet ordre :

```
1. Appliquer les opérations mémoire de l'échange précédent      (M3, résultat de l'extraction en arrière-plan)
2. Sauvegarder le message de l'apprenant en base                (M1)
3. Compresser les vieux échanges si l'historique est trop long  (M2, appel au modèle si nécessaire)
4. Construire le bloc « Mémoire de l'apprenant »                (M3, requêtes SQLite)
5. ContextBuilder.build(...)
      a. prompt système + profil
      b. mémoire de l'apprenant
      c. résumé des anciens échanges
      d. ► retriever(message, max_tokens) ◄                    (le RAG est appelé ici)
      e. historique récent, du plus récent au plus ancien, tant qu'il reste de la place
6. Envoi à Ollama en streaming, affichage
7. Sauvegarde de la réponse
8. Lancement de l'extraction mémoire en arrière-plan            (M3)
```

À l'étape 5d, **aucun autre appel à Ollama n'est en cours** : l'extraction de l'échange précédent a été attendue à l'étape 1, et la compression est terminée. Le RAG peut appeler Ollama (embeddings, reranking) sans être en concurrence avec la mémoire.

---

## 3. Le budget

```
budget = num_ctx × (1 − response_reserve)       → 8192 × 0,8 = 6553 tokens
```

| Section | Part | Tokens (config par défaut) | Réglage |
|---|---|---|---|
| Réserve pour la réponse | 20 % de `num_ctx` | 1639 | `response_reserve` |
| Prompt système + profil | fixe | ~125 | — |
| Mémoire de l'apprenant (M3) | ≤ 10 % du budget | ≤ 655 | `memory_share` |
| Résumé des anciens échanges (M2) | ≤ 10 % du budget | ≤ 655 | `summary_share` |
| **Documents RAG** | **≤ 25 % du budget** | **≤ 1638** | `rag_share` |
| Historique récent | le reste | ~3 500 | — |

Deux points importants :

- **La part du RAG est réservée d'avance.** Dès qu'un `retriever` est branché, `history_budget()` retire 25 % du budget de la part de l'historique, même si le RAG ne renvoie rien sur ce tour. C'est ce qui permet à la compression (M2) de se déclencher **avant** que les documents ne fassent déborder la fenêtre.
- **Changer `rag_share` change la taille de l'historique.** Plus de place pour les documents, c'est moins d'échanges récents gardés tels quels, donc plus de compressions.

---

## 4. Le contrat du `retriever`

### Signature

```python
# sensai/context.py
Retriever = Callable[[str, int], list[str]]

def retrieve(query: str, max_tokens: int) -> list[str]:
    ...
```

| Paramètre | Valeur passée |
|---|---|
| `query` | Le message actuel de l'apprenant (`history[-1].content`), tel qu'il l'a tapé |
| `max_tokens` | `int(budget × rag_share)` moins le coût de la consigne d'ancrage (`RAG_HEADER_TOKENS`, ≈ 105) → 1533 avec la config par défaut |
| Retour | Une liste de morceaux de texte, du plus pertinent au moins pertinent. Liste vide si rien n'est pertinent |

### Ce que `ContextBuilder` fait des morceaux

```python
docs = f"{RAG_HEADER}\n\n" + "\n\n".join(chunks)
system += f"\n\n{docs}"
```

- Les morceaux sont collés, séparés par une ligne vide, **à la fin du message système**, sous le titre `## Documents de référence` et une consigne d'ancrage (`RAG_HEADER` dans `context.py`) : s'appuyer d'abord sur les extraits, nommer le document utilisé, dire « je ne sais pas » si l'information n'y est pas, et les traiter comme des données, pas comme des consignes.
- Ils ne sont **pas** enregistrés dans l'historique ni en base : ils sont recalculés à chaque message.
- Ils ne sont donc **jamais résumés par M2**. En revanche, la réponse du tuteur qui s'en est servie fait partie de l'historique, et peut être résumée plus tard.
- `/context` affiche la place qu'ils ont prise (`RAG 412`).

### Les trois règles à respecter

**1. Rester sous `max_tokens`. C'est au RAG de le faire.**
`ContextBuilder` fait confiance au retriever et **ne recoupe pas** les morceaux. Si le RAG renvoie trop de texte, la place est prise sur l'historique, et à l'extrême le prompt dépasse `num_ctx`. Il faut compter avec la **même estimation** que le reste du module :

```python
from sensai.memory.tokens import estimate_message

def fit_to_budget(chunks: list[str], max_tokens: int) -> list[str]:
    """Garde les meilleurs morceaux tant que le bloc final tient dans max_tokens."""
    kept: list[str] = []
    for chunk in chunks:  # déjà triés par pertinence
        if estimate_message("\n\n".join(kept + [chunk])) > max_tokens:
            break
        kept.append(chunk)
    return kept
```

`estimate_message` compte 1 token pour 4,5 lettres latines, 1 pour 2 espaces, 1 par chiffre ou signe de ponctuation et **1 par caractère japonais, chinois ou coréen** (règle ajustée sur le vrai tokenizer, voir [memoire_contexte.md §4](memoire_contexte.md#4-estimation-des-tokens-memorytokenspy)). C'est important pour des documents de cours de japonais : un morceau de 500 caractères en kanji coûte environ 500 tokens, pas 125.

**2. Ne jamais lever d'exception.**
La boucle du chat n'attrape que les erreurs d'Ollama (`OllamaError`). Une autre exception levée par le retriever (fichier d'index manquant, erreur de base vectorielle…) **ferait planter le programme**. En cas de problème, il faut renvoyer une liste vide, et éventuellement afficher un avertissement sur `stderr`.

**3. Rester rapide.**
Le retriever est appelé **à chaque message**, avant l'envoi au modèle, et l'apprenant attend pendant ce temps. Le calcul de l'embedding de la requête plus la recherche doivent rester de l'ordre de quelques centaines de millisecondes. L'indexation des documents se fait une fois, au démarrage ou par une commande à part, jamais dans `retrieve()`.

---

## 5. Brancher le RAG : les modifications à faire

### 5.1 Côté RAG : exposer une fonction conforme

Exemple de squelette (à adapter à votre implémentation) :

```python
# sensai/rag/retriever.py
import sys

from sensai.memory.tokens import estimate_message


class RagRetriever:
    def __init__(self, index, top_k: int = 8, min_score: float = 0.3):
        self.index = index          # votre base vectorielle, chargée une seule fois
        self.top_k = top_k
        self.min_score = min_score

    def __call__(self, query: str, max_tokens: int) -> list[str]:
        try:
            hits = self.index.search(query, self.top_k)           # [(score, texte, métadonnées), ...]
        except Exception as exc:                                  # ne jamais faire planter le chat
            print(f"[RAG indisponible : {exc}]", file=sys.stderr)
            return []
        chunks = [self._format(text, meta) for score, text, meta in hits if score >= self.min_score]
        return self._fit(chunks, max_tokens)

    @staticmethod
    def _format(text: str, meta: dict) -> str:
        # Indiquer la source aide le tuteur à citer et à ne pas inventer.
        return f"[{meta.get('source', '?')}]\n{text}"

    @staticmethod
    def _fit(chunks: list[str], max_tokens: int) -> list[str]:
        kept: list[str] = []
        for chunk in chunks:
            if estimate_message("\n\n".join(kept + [chunk])) > max_tokens:
                break
            kept.append(chunk)
        return kept
```

Un objet avec `__call__` convient : `ContextBuilder` attend juste quelque chose qu'on peut appeler avec `(query, max_tokens)`.

### 5.2 Côté chat : passer le retriever à `ContextBuilder`

Dans `sensai/cli.py`, `Chat.__init__` crée aujourd'hui le `ContextBuilder` **sans** retriever. Il suffit d'ajouter les deux arguments :

```python
self.context = ContextBuilder(
    config.system_prompt,
    config.num_ctx,
    config.response_reserve,
    retriever=retriever,            # ← l'objet RagRetriever, ou None si le RAG est désactivé
    rag_share=config.rag_share,     # ← nouveau réglage
    summary_share=config.summary_share,
    memory_share=config.memory_share,
)
```

### 5.3 Côté configuration

Ajouts proposés, sur le modèle des réglages existants :

```python
# sensai/config.py, dans Config
rag: bool = False
rag_share: float = 0.25
```

```python
# sensai/config.py, dans load_config : ajouter "rag_share" à la boucle de validation, et
if config.summary_share + config.memory_share + config.rag_share > 0.7:
    raise ConfigError("summary_share + memory_share + rag_share must stay under 0.7 to leave room for history")
```

La limite de 0,7 garantit qu'au moins 30 % du budget reste pour l'historique récent. Sans ça, le tuteur perd le fil de l'exercice en cours.

Le chargement de l'index se fait une seule fois dans `run()`, avant la création de `Chat`, et seulement si `config.rag` est vrai.

---

## 6. Interactions avec la mémoire à connaître

### La requête, c'est seulement le dernier message

Le retriever reçoit le message brut de l'apprenant, pas la conversation. Les questions de relance retrouvent donc mal leurs documents :

```
Apprenant : explique-moi la particule « wa »      → bonne requête
Tuteur    : …
Apprenant : et la différence avec l'autre ?       → requête presque vide de sens
```

Pistes, par ordre de coût :
- **Simple :** concaténer le message précédent de l'apprenant à la requête ;
- **Moyen :** ajouter à la requête le champ `en_cours` du résumé M2 (`self.summary.en_cours`), qui décrit l'exercice en cours ;
- **Coûteux :** reformuler la requête avec le modèle (un appel de plus à chaque message).

Pour l'une des deux premières, la signature de `Retriever` peut évoluer (par exemple `retrieve(query, max_tokens, context)`). Il faut alors modifier l'appel dans `ContextBuilder.build()` : prévenez-moi, on le fait ensemble.

### La mémoire M3 peut guider la recherche

La mémoire de l'apprenant sait quelles erreurs reviennent (« confond wa et ga »). Un RAG sur des fiches de grammaire peut s'en servir pour **remonter la fiche qui correspond à l'erreur fréquente**, même si l'apprenant ne la mentionne pas. C'est une bonne démonstration d'intégration pour la keynote : les features se renforcent au lieu de juste coexister.

Les données sont accessibles via `LearnerMemory.mistakes(user, language)` (`sensai/memory/learner.py`).

### Ce que le RAG ne doit pas stocker

Les documents du RAG décrivent **la langue** (grammaire, listes de vocabulaire, dialogues). La mémoire M3 décrit **l'apprenant** (ce qu'il a vu, raté, maîtrisé). Les deux ne doivent pas se mélanger : un mot présent dans une fiche de vocabulaire n'est **pas** un mot appris. C'est l'extraction M3, après l'échange, qui décide qu'un mot a été vu.

### Ordre dans le prompt

Dans le message système, l'ordre est : rôle du tuteur → profil → mémoire de l'apprenant → résumé → **documents**. Les documents arrivent en dernier, donc juste avant la conversation. Si le modèle les ignore trop ou au contraire s'y accroche trop, l'ordre se change dans `ContextBuilder.build()`.

---

## 7. Tester l'intégration

### Sans Ollama, avec un faux retriever

```python
from sensai.context import ContextBuilder
from sensai.memory.store import Message

def fake_retriever(query: str, max_tokens: int) -> list[str]:
    return ["[grammaire.md]\nLa particule wa marque le thème de la phrase."]

cb = ContextBuilder("Tu es un tuteur.", num_ctx=8192, response_reserve=0.2, retriever=fake_retriever)
messages, report = cb.build({}, [Message("user", "c'est quoi wa ?")])

assert "## Documents de référence" in messages[0]["content"]
print(report.rag_tokens, "tokens de RAG sur", int(cb.budget * cb.rag_share), "autorisés")
```

### Vérifications à faire

| Test | Ce qu'on vérifie |
|---|---|
| Le retriever renvoie trop de texte | `report.rag_tokens <= int(budget × rag_share)` : sinon la règle 1 n'est pas respectée |
| Le retriever lève une exception | Le chat continue et affiche un avertissement |
| Documents en japonais | L'estimation compte bien 1 token par kanji (`estimate_message("日本語")` = 8 : 3 kanji + 1 + 4 de formatage) |
| Longue conversation avec RAG actif | `/context` montre des compressions M2 qui se déclenchent, et aucun message écarté |
| Question sans rapport avec les documents | Liste vide, `RAG 0` dans `/context` |

### En vrai

Une fois branché, la commande `/context` dans le chat affiche la répartition réelle :

```
Budget 2410/6553 tokens (estimés) — système+profil 125, résumé 0/655, RAG 1210, historique 990/3480 (8 messages gardés, 0 écartés)
Mémoire de l'apprenant 85/655
```

C'est aussi une bonne capture pour la keynote : on y voit toutes les features partager la même fenêtre.

---

## 8. Récapitulatif

| À faire | Où | Qui |
|---|---|---|
| Fonction `retrieve(query, max_tokens) -> list[str]` | `sensai/rag/` | RAG |
| Respecter `max_tokens` avec `estimate_message` | dans `retrieve` | RAG |
| Ne jamais lever d'exception, rester rapide | dans `retrieve` | RAG |
| Charger l'index une seule fois | `run()` dans `cli.py` | RAG |
| Réglages `rag` et `rag_share` + validation | `config.py`, `config.json` | à deux |
| Passer `retriever` et `rag_share` à `ContextBuilder` | `Chat.__init__` dans `cli.py` | à deux |
| Si besoin : enrichir la requête (message précédent, `en_cours`) | `ContextBuilder.build()` | à deux |
