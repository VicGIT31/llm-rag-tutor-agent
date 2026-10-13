# RAG : R1 (Basic) + R2 (Advanced)

Le RAG donne au tuteur des connaissances qu'il n'a pas : les leçons du dossier `docs/lecons/`. Il est branché sur `ContextBuilder` par la fonction `retriever` décrite dans [memoire_contexte.md §9](memoire_contexte.md#9-interface-avec-le-rag). Aucun framework LLM : embeddings et rerank passent par l'API HTTP d'Ollama, le stockage est du SQLite standard.

R2 « étend » R1 (le catalogue le dit) : les deux sont donc dans le même pipeline, `sensai/rag/`.

## 1. Pipeline

```
message ─► filtres de métadonnées (explicites + profil)                    R2
        ─► cosinus (embeddings) + BM25 (mots exacts) sur les morceaux restants   R1 / R2
        ─► seuil 1 : cosinus < rag_min_cosine → écarté                      R2
        ─► fusion RRF → rag_candidates meilleurs
        ─► rerank : le modèle de chat note chaque candidat de 0 à 10 (1 appel)  R2
        ─► seuil 2 : note < rag_min_rerank → écarté                         R2
        ─► au plus rag_top_k morceaux, dans le budget de tokens → contexte
```

| Fichier | Rôle |
|---|---|
| `chunker.py` | front matter, métadonnées, découpage markdown (titre de section répété dans chaque morceau) |
| `embedder.py` | `POST /api/embed`, vecteurs normalisés |
| `store.py` | SQLite : documents, morceaux, embeddings en BLOB |
| `lexical.py` | tokenisation (accents, bigrammes pour le japonais) + BM25 |
| `filters.py` | `niveau=A1`, `niveau=A1,A2`, `date>=2026-09-01` |
| `pipeline.py` | indexation incrémentale, recherche, rerank, affichage |

## 2. Métadonnées

Dans un fichier de `docs/lecons/`, un bloc optionnel en tête :

```
---
langue: japonais
niveau: A1
auteur: Camille
date: 2026-09-30
---
```

Ajoutés automatiquement : `source`, `title` (premier titre), `folder` (dossier parent), `date` (date de **modification du fichier**, à défaut). Ce défaut n'est fiable que sur une machine où les fichiers n'ont pas été recopiés : un `git clone` remet toutes les dates au jour du clone. Pour filtrer par date, écrire `date:` dans le front matter. Les dates doivent être en ISO (`AAAA-MM-JJ`), la comparaison est faite sur du texte.

Filtre **strict** (`/rag filter`) : un document sans la clé est exclu. Filtre **souple** (déduit du profil, `rag_profile_filters`, par défaut `langue_cible → langue`) : un document sans la clé passe, pour que le matériel général reste trouvable. Un filtre explicite sur la même clé remplace le filtre du profil.

## 3. Choix et alternatives écartées

| Décision | Alternative écartée | Pourquoi |
|---|---|---|
| Recherche hybride (cosinus + BM25, fusion RRF) | cosinus seul | un mot rare ou un terme de grammaire exact est mal capté par l'embedding ; BM25 le rattrape |
| Bigrammes pour le japonais | découpage par espaces | le japonais n'a pas d'espaces, BM25 ne verrait qu'un seul « mot » par phrase |
| Rerank par le modèle de chat, un seul appel pour tous les candidats | un appel par candidat ; cross-encoder dédié | un appel par candidat multiplie la latence ; un cross-encoder demande une dépendance ou un modèle de plus, hors du cadre « Ollama uniquement » |
| Rerank facultatif avec repli (`rag_rerank`) | rerank obligatoire | si l'appel échoue ou renvoie un JSON invalide, l'ordre de fusion est conservé et le tour continue |
| Deux seuils (cosinus puis rerank) | un seul | le cosinus est bon marché et évite d'appeler le modèle pour rien ; le rerank affine ce qui reste |
| Filtres appliqués avant le calcul des scores | filtrage après | les morceaux exclus ne coûtent rien |
| SQLite + produit scalaire en Python | base vectorielle (Chroma, FAISS…) | pas de dépendance ; suffisant pour un cours |
| Index jetable (cache des fichiers sources) | index = source de vérité | supprimer `data/rag.db` est toujours sans risque |

## 4. Configuration

| Clé | Défaut | Rôle |
|---|---|---|
| `rag` | `true` | active le RAG ; s'il ne démarre pas (modèle d'embedding absent…), il est désactivé et le chat continue |
| `rag_sources` | `["docs/lecons"]` | fichiers ou dossiers (`.md`, `.txt`, 500 Ko max, liens symboliques ignorés) |
| `rag_embed_model` | `bge-m3` | modèle d'embedding (`ollama pull bge-m3`) ; en changer réindexe tout |
| `rag_embed_gpu` | `false` | l'embedding tourne sur le CPU (16 ms par question) : si le modèle de chat remplit déjà le GPU, Ollama déchargerait sinon un modèle pour charger l'autre à chaque message |
| `rag_share` | 0.25 | part du budget de tokens réservée aux morceaux |
| `rag_chunk_chars` | 700 | taille maximale d'un morceau |
| `rag_top_k` / `rag_candidates` | 4 / 8 | morceaux injectés au plus / morceaux envoyés au rerank |
| `rag_min_cosine` | 0.30 (0.45 dans `config.json`) | seuil 1, calibré pour bge-m3 et ce corpus (voir §7) |
| `rag_rerank` / `rag_min_rerank` | `true` / 0.5 | rerank activé / seuil 2 (note ≥ 5/10) |
| `rag_profile_filters` | `{"langue_cible": "langue"}` | clé du profil → clé de métadonnée (filtre souple) |
| `rag_auto_index` | `true` | resynchronise l'index au démarrage |

Commandes : `/rag`, `/rag search <question>`, `/rag filter <k=v>…`, `/rag filter clear`, `/rag index`. Événements loggés (EV3) : `rag_sync`, `rag_retrieve` (compteurs et durées, jamais le contenu) ; l'appel de rerank apparaît aussi comme `model_call` avec `purpose=rerank`.

## 5. Limites

- **Seuils calibrés sur un petit corpus.** 0.45 et 5/10 viennent de 14 questions sur 5 leçons (§7). Le cosinus dépend du modèle d'embedding et du corpus : à recalibrer avec `/rag search` si l'un des deux change.
- **Le rerank coûte cher sans GPU** : environ 28 s par message sur CPU seul avec `qwen3-coder:30b` (11 candidats). Sans GPU, `rag_rerank: false` avec `rag_min_cosine: 0.50` reste correct sur ce corpus (§7).
- **Le rerank juge mal les messages très courts** : à « Bonjour ! », il donne 10/10 aux phrases de voyage et même à la liste des nombres. C'est le seuil cosinus qui protège ces messages.
- **Le rerank est un petit modèle de chat qui juge**, pas un cross-encoder entraîné pour ça : notes bruitées possibles, et un appel de modèle de plus par message qui a des candidats.
- **La requête est le dernier message seul.** « Et le chien ? » est cherché tel quel, sans l'historique ; pas de réécriture de requête.
- **Une recherche par message**, y compris « bonjour » : ce sont les seuils qui évitent l'injection de bruit.
- **Produit scalaire en Python pur** : correct jusqu'à quelques milliers de morceaux, pas au-delà.
- **Les morceaux sont insérés dans le message système.** Avec un corpus non fiable, un document pourrait contenir des consignes destinées au modèle ; ici le corpus est celui du tuteur.
- Découpage des phrases japonaises en bigrammes : heuristique, sans analyseur morphologique.

## 6. User stories

**R1-a — Réviser avec son cours**
En tant qu'apprenant de japonais, je veux poser une question sur du vocabulaire vu en cours, afin que le tuteur réponde à partir de ma leçon et non de mémoire.
Critères d'acceptation :
- après l'indexation, `/rag search que veut dire neko ?` fait apparaître `docs/lecons/animaux.md` parmi les morceaux retenus ;
- le morceau figure sous « ## Documents de référence » dans le contexte, et `/context` affiche ses tokens (au plus `rag_share` du budget) ;
- si le modèle d'embedding est absent, le chat démarre quand même et le message d'erreur donne la commande `ollama pull`.

**R1-b — Ajouter une leçon**
En tant qu'enseignant, je veux déposer ou modifier une leçon dans `docs/lecons/`, afin qu'elle soit utilisée sans tout réindexer à la main.
Critères d'acceptation :
- au démarrage ou avec `/rag index`, seul le fichier nouveau ou modifié est recalculé ;
- un fichier supprimé disparaît de l'index ;
- changer `rag_embed_model` ou `rag_chunk_chars` réindexe tout.

**R2-a — Rester dans sa langue et son niveau**
En tant qu'apprenant de japonais, je veux ne recevoir que des leçons de japonais, afin de ne pas mélanger avec le coréen vu par un autre apprenant.
Critères d'acceptation :
- avec `/profile set langue_cible japonais`, aucun morceau `langue: coréen` n'est retenu, et les documents sans `langue` restent éligibles ;
- `/rag filter niveau<=A2` ajoute un filtre, `/rag filter clear` le retire ;
- un filtre invalide (`niveau==`) est refusé avec un message, sans toucher aux filtres en place.

**R2-b — Ne pas polluer le contexte**
En tant qu'apprenant, je veux qu'une question hors sujet (« quel temps demain ? ») n'ajoute aucun document, afin de garder la place pour l'historique et d'éviter une réponse qui cite un cours sans rapport.
Critères d'acceptation :
- si tous les morceaux sont sous les seuils, aucun morceau n'est injecté et `rag_tokens` vaut 0 ;
- `/rag search` indique pour chaque morceau écarté la raison (cosinus, rerank, `rag_top_k`, budget) ;
- si le rerank échoue (JSON invalide, serveur coupé), la réponse arrive quand même, avec l'ordre de fusion.

**R2-c — Régler la pertinence**
En tant que développeur du tuteur, je veux voir les scores de chaque morceau pour une question donnée, afin de régler les seuils avec des mesures et non au hasard.
Critères d'acceptation :
- `/rag search <question>` affiche cosinus, BM25 et note de rerank par morceau ;
- les seuils se changent dans `config.json` sans toucher au code.

## 7. Tests

`tests/test_rag.py` (26 tests, sans réseau ni modèle) : découpage, front matter, tokenisation japonaise, filtres, indexation incrémentale, seuils, budget, rerank (notes, JSON invalide, serveur coupé), intégration avec `ContextBuilder`. Les embeddings y sont simulés par un hachage de mots, le rerank par un faux client scripté. **Ce qui n'est pas couvert par ces tests : la qualité du retrieval avec un vrai modèle d'embedding** ; elle se mesure avec `/rag search` sur le vrai corpus.

### Calibration avec les vrais modèles (5 octobre 2026)

Corpus : les 5 leçons de `docs/lecons/` (12 morceaux ; la leçon de coréen est écartée par le filtre de profil `langue=japonais`). Modèles : `bge-m3` pour les embeddings, `qwen3-coder:30b` pour le rerank, sur CPU seul. 14 questions : 10 dont on connaît la ou les leçons pertinentes, 4 hors sujet (« Bonjour ! », « Quel temps fera-t-il demain ? », « Peux-tu m'aider à écrire un mail à mon patron ? », « Merci, à demain ! »). Les scores (cosinus, BM25, rerank) ont été relevés pour tous les morceaux avec les deux seuils à 0, puis les seuils ont été rejoués hors ligne (`rag_top_k` = 4, `rag_candidates` = 8).

| Cosinus ≥ | Rerank ≥ | Questions pertinentes servies | Précision des morceaux injectés | Questions hors sujet avec injection |
|---|---|---|---|---|
| 0.30 | — (sans rerank) | 10/10 | 70 % | 4/4 |
| 0.50 | — (sans rerank) | 10/10 | 83 % | 0/4 |
| 0.55 | — (sans rerank) | 8/10 | 85 % | 0/4 |
| 0.30 | 5/10 | 10/10 | 100 % | 1/4 |
| **0.45** | **5/10** | **10/10** | **100 %** | **1/4** |
| 0.50 | 5/10 | 9/10 | 100 % | 0/4 |

Constats :
- **Avec bge-m3, le seuil de 0.30 ne filtre rien** : n'importe quel morceau du même domaine obtient 0.35 à 0.56, le bon morceau 0.55 à 0.64.
- **Le cosinus seul ne suffit pas** : pour « Pourquoi on dit watashi wa Camille et pas watashi ga ? », un morceau pertinent n'a que 0.35 de cosinus, mais 8.1 en BM25 et 10/10 au rerank. Et pour « Comment on dit chien en japonais ? », des morceaux sans rapport atteignent 0.56. La recherche hybride et le rerank servent donc vraiment.
- **Le rerank sépare très bien** les questions de fond : pour chacune des 10 questions pertinentes, le meilleur morceau de la bonne leçon obtient 10/10, et aucun morceau d'une autre leçon ne dépasse 4/10. Il juge mal les messages très courts (voir §5).
- **Le rerank a renvoyé une réponse inutilisable 1 fois sur 14** (« Merci, à demain ! ») : le repli prévu (ordre de fusion, seuil cosinus seul) a fonctionné.
- **Coût** : l'embedding de la question prend environ 20 ms, le rerank environ 28 s sur CPU.

Réglage retenu dans `config.json` : `rag_min_cosine` = 0.45, `rag_min_rerank` = 0.5. Le seul défaut restant : « Bonjour ! » reçoit le morceau des phrases de voyage. Sans GPU, le repli conseillé est `rag_rerank: false` avec `rag_min_cosine: 0.50`, qui donne 10/10, 83 % et 0/4.
