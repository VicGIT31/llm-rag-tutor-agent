# RAG : R1 (Basic) + R2 (Advanced)

Le RAG donne au tuteur des connaissances qu'il n'a pas : les leçons du dossier `docs/lecons/`. Il est branché sur `ContextBuilder` par la fonction `retriever` décrite dans [memoire_contexte.md §9](memoire_contexte.md#9-interface-avec-le-rag). Aucun framework LLM : embeddings et rerank passent par l'API HTTP d'Ollama, le stockage est du SQLite standard.

R2 « étend » R1 (le catalogue le dit) : les deux sont donc dans le même pipeline, `sensai/rag/`.

## 1. Pipeline

```
message ─► requête : le message, ou la relance courte + le message précédent     R2
        ─► filtres de métadonnées (explicites + profil)                    R2
        ─► cosinus (embeddings) + BM25 (mots exacts) sur les morceaux restants   R1 / R2
        ─► seuil 1 : cosinus < rag_min_cosine → écarté                      R2
        ─► fusion RRF → rag_candidates meilleurs
        ─► rerank : le modèle de chat note chaque candidat de 0 à 10 (1 appel, une note par passage)  R2
        ─► seuil 2 : note < rag_min_rerank → écarté                         R2
        ─► au plus rag_top_k morceaux, dans le budget de tokens → contexte, sous une consigne d'ancrage
réponse ─► vérification : dates, heures, salles, prix, notes retrouvés dans les morceaux ?   R2
```

| Fichier | Rôle |
|---|---|
| `chunker.py` | front matter, métadonnées, découpage markdown (titre de section répété dans chaque morceau) |
| `embedder.py` | `POST /api/embed`, vecteurs normalisés |
| `store.py` | SQLite : documents, morceaux, embeddings en BLOB |
| `lexical.py` | tokenisation (accents, bigrammes pour le japonais) + BM25 |
| `filters.py` | `niveau=A1`, `niveau=A1,A2`, `date>=2026-09-01` |
| `pipeline.py` | indexation incrémentale, recherche, rerank, affichage |
| `grounding.py` | vérification des faits d'une réponse contre les morceaux injectés |
| `../context.py` | requête de relance (`retrieval_query`), consigne d'ancrage, part du budget |

## 1 bis. Relances, rerank compact et vérification des faits (ajouts du 25 octobre)

**Relances.** « Et l'acompte ? » ne veut rien dire seul : cherché tel quel, il ramène n'importe quoi. Un message de moins de `rag_followup_words` mots (6) est donc cherché avec le message précédent de l'apprenant (`context.retrieval_query`). Pas d'appel au modèle pour reformuler : la concaténation suffit à l'embedding et à BM25, et les seuils protègent du bruit. `/context` affiche la requête quand elle a été élargie. Contrepartie : « Merci ! » après une question sur l'interro ramène encore l'interro (les seuils ne peuvent pas l'écarter, la requête est pertinente).

**Rerank compact.** Le modèle répond `{"scores": [7, 0, 10, …]}`, une note par passage dans l'ordre, au lieu de `{"id": 0, "score": 7}` pour chacun. Le schéma JSON fixe le nombre de notes (`minItems` = `maxItems`). Le modèle écrit environ 2 tokens par passage au lieu de 12 : sur trois recherches mesurées dans les mêmes conditions, 17 à 23 s au lieu de 23 à 33 s, avec les mêmes morceaux retenus. L'ancien format reste accepté à la lecture.

**Vérification des faits.** Après une réponse appuyée sur des leçons, `grounding.check_answer` relève les faits sur lesquels l'apprenant agit et qu'un modèle invente facilement (dates, heures, salles, prix, notes sur 20, points, durées, pourcentages, années, nombres décimaux), les normalise (« 14 h » = « 14h00 », « 1 450 € » = « 1450 euros », « 13,4 » = « 13.4 ») et cherche chacun dans les morceaux injectés et dans le message de l'apprenant. Un fait absent est signalé : ligne `[vérification : …]` dans le terminal, badge « ⚠ absent de tes leçons » dans le navigateur (« ✓ N faits vérifiés » sinon), champ `check` du message enregistré, compteurs dans `/metrics`. Pour des points donnés dans une grille (`| Kana | 4 |`), le nombre seul suffit. Aucun appel au modèle : la vérification ne peut pas halluciner elle-même et prend une milliseconde. Elle attrape un fait inventé, pas un fait réel mal placé (une date recopiée depuis une autre ligne du planning), ni une erreur de vocabulaire japonais.

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
| `rag_top_k` / `rag_candidates` | 4 / 8 (4 dans `config.json`) | morceaux injectés au plus / morceaux envoyés au rerank ; sur le benchmark de recherche, 4 candidats donnent le même rappel que 8 et une meilleure précision, avec un prompt de rerank deux fois plus court (§9) |
| `rag_min_cosine` | 0.30 (0.45 dans `config.json`) | seuil 1, calibré pour bge-m3 et ce corpus (voir §7) |
| `rag_min_bm25` | 0 (3.0 dans `config.json`) | repêchage au seuil 1 : un morceau en tête du classement BM25 (≥ ce score et ≥ 80 % du meilleur) passe malgré un cosinus faible ; 3.0 est le score d'un mot présent dans au plus 3 morceaux sur 76 |
| `rag_rerank` / `rag_min_rerank` | `true` / 0.5 | rerank activé / seuil 2 (note ≥ 5/10) |
| `rag_profile_filters` | `{"langue_cible": "langue"}` | clé du profil → clé de métadonnée (filtre souple) |
| `rag_auto_index` | `true` | resynchronise l'index au démarrage |
| `rag_followup_words` | 6 | un message plus court est cherché avec le message précédent de l'apprenant (0 = jamais) |
| `keep_alive` | `30m` | durée pendant laquelle Ollama garde les modèles chargés (chat et embedding) ; son défaut de 5 min les décharge pendant que l'apprenant réfléchit |

Commandes : `/rag`, `/rag search <question>`, `/rag filter <k=v>…`, `/rag filter clear`, `/rag index`. Événements loggés (EV3) : `rag_sync`, `rag_retrieve` (compteurs et durées, jamais le contenu) ; l'appel de rerank apparaît aussi comme `model_call` avec `purpose=rerank`.

## 5. Limites

- **Seuils calibrés sur un petit corpus.** 0.45 et 5/10 viennent de 14 questions sur 5 leçons (§7) ; une sonde sur le corpus étendu à 23 documents montre qu'ils tiennent (§7, fin). Le cosinus dépend du modèle d'embedding et du corpus : à recalibrer avec `/rag search` si l'un des deux change.
- **Le rerank coûte cher sans GPU** : environ 28 s par message sur CPU seul avec `qwen3-coder:30b` (11 candidats). Sans GPU, `rag_rerank: false` avec `rag_min_cosine: 0.50` reste correct sur ce corpus (§7).
- **Le rerank juge mal les messages très courts** : à « Bonjour ! », il donne 10/10 aux phrases de voyage et même à la liste des nombres. C'est le seuil cosinus qui protège ces messages.
- **Le rerank est un petit modèle de chat qui juge**, pas un cross-encoder entraîné pour ça : notes bruitées possibles, et un appel de modèle de plus par message qui a des candidats.
- **Relances : une concaténation, pas une reformulation.** Une relance courte est cherchée avec le message précédent ; une relance longue mais elliptique (« Et pour celle de novembre, ça tombe sur quoi ? ») est cherchée seule.
- **La vérification des faits ne voit que les faits chiffrés** ; elle ne juge ni le vocabulaire japonais ni un fait réel mal attribué.
- **Une recherche par message**, y compris « bonjour » : ce sont les seuils qui évitent l'injection de bruit.
- **Produit scalaire en Python pur** : correct jusqu'à quelques milliers de morceaux, pas au-delà.
- **Les morceaux sont insérés dans le message système.** Avec un corpus non fiable, un document pourrait contenir des consignes destinées au modèle. La consigne d'ancrage placée au-dessus des morceaux (`RAG_HEADER`, `context.py`) demande de les traiter comme des données et de nommer le document utilisé : cela atténue le risque sans l'éliminer. Ici le corpus est celui du tuteur.
- Découpage des phrases japonaises en bigrammes : heuristique, sans analyseur morphologique.

## 6. User stories

**R1-a — Réviser avec son cours**
En tant qu'apprenant de japonais, je veux poser une question sur du vocabulaire vu en cours, afin que le tuteur réponde à partir de ma leçon et non de mémoire.
Critères d'acceptation :
- après l'indexation, `/rag search que veut dire neko ?` fait apparaître `docs/lecons/animaux.md` parmi les morceaux retenus ;
- le morceau figure sous « ## Documents de référence » dans le contexte, avec la consigne d'ancrage, et `/context` affiche ses tokens (au plus `rag_share` du budget, consigne comprise) ;
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

**R1-c — Relancer sans tout répéter**
En tant qu'apprenante qui vient de demander le prix du voyage, je veux pouvoir enchaîner par « Et l'acompte ? », afin d'avoir la réponse sans reformuler toute ma question.
Critères d'acceptation :
- un message de moins de `rag_followup_words` mots est cherché avec mon message précédent ; `/context` affiche la requête élargie ;
- un message long, ou le premier de la session, est cherché seul ;
- `rag_followup_words: 0` désactive le comportement sans toucher au code.

**R2-d — Savoir quand vérifier une réponse**
En tant qu'apprenante qui prépare son interro, je veux être prévenue quand une date, une heure, une salle ou une note donnée par le tuteur ne figure pas dans mes leçons, afin de ne pas me présenter le mauvais jour.
Critères d'acceptation :
- après une réponse appuyée sur des leçons, chaque date, heure, salle, prix, note ou durée est cherché dans les extraits utilisés et dans mon message ;
- un fait absent est signalé (« ⚠ absent de tes leçons : 29janvier » dans le navigateur, ligne `[vérification : …]` dans le terminal) ; sinon « ✓ N faits vérifiés » ;
- le résultat est enregistré avec la réponse (relecture d'une ancienne séance) et compté dans `/metrics` ;
- sans leçon dans le prompt, rien n'est vérifié ni affiché : il n'y a rien à quoi comparer.

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

### Sur le corpus étendu (25 octobre 2026)

Le corpus est passé à 23 documents (76 morceaux, environ 8 600 tokens, plus que la fenêtre de 8 192) : 14 leçons de japonais, 3 de coréen, et 6 documents de classe (consignes de l'interro n°2, planning, corrigé de l'interro n°1, devoirs, voyage, oral final). Une sonde avec les réglages de `config.json` (repêchage BM25 à 3.0, 4 candidats), profil japonais :

| Message | Morceaux : indexés → après filtres → après cosinus → candidats → après rerank → injectés |
|---|---|
| « Qu'est-ce qui tombe à l'interro de vendredi ? » | 76 → 71 → 8 → 4 → 1 → 1 (`interro_vendredi.md`, partie « Programme », 10/10) ; sans rerank, 4 morceaux dont le corrigé de l'interro n°1, hors sujet |
| « Que veut dire hatsuka ? » | 76 → 71 → 1 → 1 → 1 → 1 (`jours_dates.md`, cosinus 0,41 mais BM25 3,8 : **repêché**, 10/10) |
| « Où se trouve la salle A112 ? » | 76 → 71 → 1 → 1 → 1 → 1 (`planning_semestre.md`, cosinus 0,40, BM25 9,5 : **repêché**, 10/10) |
| « Comment on demande l'addition au restaurant ? » | 76 → 71 → 9 → 8 → 3 → 3 (`restaurant.md` 10/10, `voyage.md` 7/10 ; mesure avec 8 candidats) |
| « Bonjour ! Comment vas-tu ? » | 76 → 71 → 2 → 2 → 0 → 0 |

Sans le filtre de langue, `coreen_restaurant.md` arrive en tête pour l'addition, et `coreen_nombres.md` obtient 10/10 au rerank pour « Comment dit-on 3 heures ? » : le filtre de métadonnées est la seule protection contre ces passages. Les seuils de 0.45 et 5/10 tiennent donc sur ce corpus.

## 8. Mesurer l'utilité : benchmark avec / sans RAG

`benchmarks/rag_ab.py` compare cinq variantes sur 16 questions de cours et 4 de contrôle : sans RAG 0/16, avec RAG 15 à 16/16 (le bon document dans le prompt 16 fois sur 16), et **tout le corpus dans le prompt 10/16**, avec des réponses fausses et assurées dues à la troncature. Le rerank fait passer la précision des documents injectés de 41 % à 76 % et réduit le prompt de 43 %, pour environ 11 s de plus par question. Détail, conditions et limites : [labo.md §4](labo.md#4-mesurer-lintérêt-du-rag--benchmarksrag_abpy).

## 9. Mesurer chaque étage : benchmark de recherche

`benchmarks/rag_retrieval.py` évalue la recherche seule (sans générer) sur 49 requêtes de cinq types : cours, mot rare, paraphrase, relance, hors sujet. Il a révélé un défaut de conception : **le seuil cosinus était appliqué avant BM25**, si bien qu'un morceau contenant exactement le mot rare (« salle A112 » : meilleur score BM25, cosinus 0,40) était écarté avant la fusion, et BM25 était en plus noyé par les mots de la question (« que veut dire »). Avec les mots outils ignorés et le repêchage (`rag_min_bm25` = 3.0), les questions à mot rare servies passent de **6/10 à 10/10**, sans perte sur les autres types. Les relances élargies au message précédent passent de 5/6 à 6/6. Le rerank reste ce qui protège des messages hors sujet (injection pour 2 messages sur 5 au lieu de 5 sur 5). Avec 4 candidats au lieu de 8, le rappel est le même, la précision égale ou meilleure, et la recherche passe de 17,3 s à 11,4 s (mesuré cache vide). Détail et tableau complet : [labo.md §4](labo.md#4-mesurer-lintérêt-du-rag--benchmarksrag_abpy).

