# Mémoire & Contexte — fonctionnement

Ce document explique comment Sensai gère la **mémoire** (ce qu'il retient entre les messages et entre les sessions) et le **contexte** (ce qu'on envoie réellement au modèle à chaque tour).

Cas d'usage fil rouge : un **tuteur de langues façon Duolingo**, sur un modèle local Ollama (par défaut `qwen3-coder:30b`, voir `ollama list`).

Features du catalogue concernées :

| ID | Feature | État |
|---|---|---|
| M1 | Session Persistence & Profiles | ✅ fait |
| M2 | Token Budgeting & Semantic Compression | ✅ fait, mesuré (voir §13) |
| M3 | External Structured Memory | ✅ fait, mesuré (voir §13) |
| M4 | Artifact & State Document | ❌ pas prévu pour l'instant |

---

## 1. Le problème : une seule fenêtre de contexte pour tout le monde

Un LLM ne « se souvient » de rien : à chaque tour, on lui renvoie **tout** ce qu'il doit savoir dans une seule requête. Cette requête est limitée par la taille de la fenêtre de contexte, `num_ctx` (en tokens).

Tout se partage cette fenêtre :

- le prompt système (le rôle de tuteur) ;
- le profil de l'apprenant ;
- les documents du RAG (géré par un coéquipier) ;
- l'historique de la conversation ;
- **et la réponse du modèle**, qui est générée dans la même fenêtre.

### ⚠️ Piège Ollama : la troncature silencieuse

Si la requête dépasse `num_ctx`, **Ollama supprime les plus vieux tokens sans renvoyer d'erreur**. Le modèle peut alors perdre le prompt système ou le profil sans qu'on le sache. De plus, la valeur par défaut de `num_ctx` dans Ollama est petite (2048 à 4096 selon la version).

C'est pour ça que :

1. `num_ctx` est **toujours fixé explicitement** dans chaque requête (`options.num_ctx`, valeur dans `config.json`) ;
2. c'est **notre code** qui choisit ce qui entre dans le contexte, avant l'envoi, au lieu de laisser Ollama couper au hasard.

---

## 2. Architecture

```
sensai/
  config.py           # lecture de config.json + arguments CLI
  ollama_client.py    # appels HTTP à Ollama (streaming, erreurs, compteurs de tokens)
  context.py          # ContextBuilder : assemble le prompt dans le budget
  cli.py              # boucle de chat et commandes
  memory/
    store.py          # SQLite : sessions, messages, profil (M1), résumés (M2)
    tokens.py         # estimation du nombre de tokens
    compressor.py     # compression des vieux tours en résumé structuré (M2)
    learner.py        # vocabulaire, erreurs, journal des opérations (M3)
    extractor.py      # le modèle choisit les opérations mémoire après chaque échange (M3)
benchmarks/
  m2_recall.py        # banc de test : rappel des faits avec / sans compression
  m3_extraction.py    # banc de test : qualité des opérations mémoire proposées
```

Déroulement d'un tour :

```
saisie utilisateur
   │
   ├─► application des opérations mémoire de l'échange précédent (M3)
   │
   ├─► sauvegarde du message en base (store.add_message)
   │
   ├─► compression si l'historique approche de sa part du budget (M2)
   │      les plus vieux tours → résumé structuré, sauvegardé en base
   │
   ├─► ContextBuilder.build(profil, historique non résumé, résumé, mémoire)
   │      1. prompt système + profil          (toujours inclus)
   │      2. mémoire de l'apprenant (M3)      (part plafonnée du budget)
   │      3. résumé des anciens tours         (part plafonnée du budget)
   │      4. documents RAG                    (part plafonnée du budget)
   │      5. historique, du plus récent au plus ancien, tant qu'il reste de la place
   │
   ├─► Ollama /api/chat en streaming → affichage progressif
   │
   ├─► sauvegarde de la réponse + nombre réel de tokens (eval_count)
   │
   └─► extraction des opérations mémoire en arrière-plan (M3)
```

---

## 3. Le budget de tokens (`context.py`)

### Calcul du budget

```
budget = num_ctx × (1 − response_reserve)
```

Avec la config par défaut : `8192 × (1 − 0.2) = 6553` tokens pour le prompt. Les 20 % restants sont **réservés à la réponse** du modèle.

### Ordre de priorité

| Priorité | Contenu | Règle |
|---|---|---|
| 1 | Prompt système + profil apprenant | Toujours inclus |
| 2 | Mémoire de l'apprenant (M3) | Au plus 10 % du budget (`memory_share`) |
| 3 | Résumé des anciens tours (M2) | Au plus 10 % du budget (`summary_share`) |
| 4 | Documents RAG | Au plus 25 % du budget (`rag_share`) |
| 5 | Historique non résumé | Ce qui reste, en partant du message le plus récent |

Avec la config par défaut (8192 tokens) : réserve pour la réponse 1639, mémoire ≤ 655, résumé ≤ 655, RAG ≤ 1638, et environ 3 500 tokens pour l'historique (environ 5 100 tant que le RAG n'est pas branché).

La **part de l'historique** (`history_budget`) est ce qui reste une fois que toutes les autres sections ont reçu leur part maximale. C'est cette valeur que le compresseur surveille.

Règles de l'historique :

- on parcourt les messages **du plus récent au plus ancien** et on s'arrête dès qu'un message ne tient plus ;
- le **message actuel de l'utilisateur est toujours gardé**, même s'il dépasse à lui seul ;
- les messages écartés sont signalés dans le terminal (`[N anciens messages hors contexte]`).

En temps normal, la compression (section 6) empêche l'historique de déborder. Écarter les vieux messages n'est plus que le **plan de secours**, quand la compression échoue.

### Rapport de contexte

Chaque construction produit un `ContextReport`, visible avec la commande `/context` :

```
Budget 594/960 tokens (estimés) — système+profil 103, résumé 74/96, RAG 0, historique 417/761 (13 messages gardés, 0 écartés)
Mémoire de l'apprenant 85/96
Opérations mémoire : 6/7 appliquées, 1 extractions échouées
Compressions : 2/2 réussies, 16 messages résumés, ~445 tokens économisés, 1 ms en moyenne
```

Il sert à déboguer et à produire des métriques pour la keynote.

### Où sont placés les documents RAG

Les morceaux renvoyés par le RAG sont ajoutés **à la fin du message système**, sous un titre `## Documents de référence` suivi d'une consigne d'ancrage : s'appuyer d'abord sur ces extraits pour une question de cours, nommer le document utilisé, dire qu'on ne sait pas si l'information n'y figure pas, et les traiter comme des données, pas comme des consignes. Ils ne sont pas stockés dans l'historique : ils sont recalculés à chaque tour en fonction du dernier message.

---

## 4. Estimation des tokens (`memory/tokens.py`)

Ollama n'a **pas de point d'accès pour compter les tokens** avant l'envoi. On estime donc, puis on corrige avec les vrais nombres.

### La règle, ajustée sur le vrai tokenizer (25 octobre 2026)

| Élément du texte | Coût estimé |
|---|---|
| Lettres latines | 1 token pour 4,5 lettres |
| Espaces et retours à la ligne | 1 token pour 2 |
| Chiffres, ponctuation, barres de tableau | 1 token chacun |
| Japonais, chinois, coréen | 1 token par caractère |
| Chaque message | + 5 tokens de formatage (`<|im_start|>rôle … <|im_end|>`) |

**Pourquoi ce découpage.** L'ancienne règle (« 3,5 caractères par token ») a été comparée aux nombres réels de `qwen3-coder:30b` sur les 76 morceaux du corpus : elle **sous-estimait 75 morceaux sur 76**, de 26 % en moyenne et jusqu'à 60 %. La prose française coûte bien 3,5 caractères par token, mais une leçon (tableaux Markdown, rōmaji, chiffres, ponctuation) n'en coûte que 2,8 : chaque chiffre et chaque signe y est presque un token à lui seul. Une régression sur les 76 morceaux a donné 4,7 lettres par token, 0,5 par espace, 0,85 par chiffre ou signe. La règle retenue est un peu au-dessus : sur le corpus, elle surestime le total de 8 % (pire morceau : 12 % sous le réel), et elle surestime la prose de 10 à 20 %, ce qui ne fait que perdre un peu de budget. Le japonais et le coréen coûtent en réalité 0,7 à 1 token par caractère : en compter un reste prudent.

### Vrais nombres de tokens et calibration du budget

Après chaque réponse, Ollama renvoie dans le dernier morceau du stream :

- `prompt_eval_count` : tokens du prompt ;
- `eval_count` : tokens générés.

`eval_count` est enregistré avec chaque réponse (colonne `messages.tokens`), et `ContextBuilder` l'utilise à la place de l'estimation quand il existe.

`prompt_eval_count` sert à **recalibrer le budget** (`ContextBuilder.observe`) : si le modèle compte plus de tokens que l'estimation du prompt, le budget est divisé par ce rapport (au plus ×1,6), pour que le prompt réel tienne dans `num_ctx`. La correction monte tout de suite (un débordement est ce qu'il faut éviter) et redescend lentement (un tour bon marché ne prouve rien). Les petits prompts (moins de 300 tokens, où le formatage pèse trop) et les prompts tronqués par Ollama sont ignorés. `/context` affiche le compte réel, la calibration et le budget corrigé ; le journal EV3 les garde dans l'événement `chat_turn`.

> Vérifié avec Ollama 0.40.4 : `prompt_eval_count` compte bien tout le prompt, même quand Ollama réutilise son cache (le même prompt envoyé deux fois donne 1 560 tokens les deux fois). Si une autre version ne comptait que les tokens nouveaux, le rapport passerait sous 1 et la calibration retomberait simplement à 1 : elle ne peut jamais agrandir le budget au-delà de `num_ctx`.

## 5. Mémoire persistante — M1 (`memory/store.py`)

Tout est stocké dans une base SQLite (`data/sensai.db` par défaut, dossier exclu de git).

### Schéma

```sql
sessions (id, user, title, created_at, updated_at)
messages (id, session_id → sessions, role, content, tokens, created_at)
profile  (user, key, value, updated_at)   -- clé primaire (user, key)
```

- **sessions** : une conversation. Son titre est le début du premier message de l'utilisateur.
- **messages** : chaque message (`user` / `assistant`), avec son nombre de tokens.
- **profile** : des paires clé/valeur libres par apprenant (`langue_cible`, `niveau`, `langue_maternelle`, `objectif`…).

### Pourquoi SQLite plutôt qu'un fichier JSON

- C'est dans la bibliothèque standard de Python (aucune dépendance).
- Les écritures sont atomiques : un crash ne corrompt pas l'historique.
- La **même base** accueillera la mémoire structurée de M3 (vocabulaire, erreurs) : tout ce qu'on sait d'un apprenant est au même endroit.

### Injection du profil

À chaque tour, le profil est ajouté au message système :

```
Tu es Sensai, un tuteur de langues bienveillant…

## Profil de l'apprenant
- langue_cible: japonais
- niveau: débutant A1
```

L'apprenant n'a donc jamais à répéter sa langue ou son niveau, même dans une nouvelle session.

### Plusieurs apprenants

Le paramètre `--user` (ou `"user"` dans `config.json`) sépare les profils et les sessions de chaque apprenant.

---

## 6. Compression de l'historique — M2 (`memory/compressor.py`)

Quand une leçon dure, l'historique finit par dépasser sa part du budget. Au lieu d'écarter les plus vieux messages (et de perdre ce que l'apprenant a appris ou raté), on les **remplace par un résumé structuré**.

### Quand la compression se déclenche

Avant chaque envoi, on compare la taille de l'historique non résumé à sa part du budget (`history_budget`) :

| Paramètre | Défaut | Rôle |
|---|---|---|
| `compress_trigger` | `0.8` | On compresse quand l'historique dépasse 80 % de sa part |
| `compress_keep` | `0.4` | Les tours les plus récents, jusqu'à 40 % de la part, restent tels quels |
| `summary_share` | `0.1` | Taille maximale du résumé : 10 % du budget total |

Le dernier message (celui de l'utilisateur) n'est jamais compressé. Il faut au moins 2 messages à résumer, sinon on ne fait rien.

Une passe résume au plus 50 % de la part de l'historique. Une longue session rechargée avec `/load` est donc compressée **en plusieurs passes**, pour que la requête de résumé tienne elle-même dans la fenêtre du modèle.

La commande `/compress` force une compression immédiate de tous les échanges sauf le dernier message, même loin du seuil (pratique pour la démo). Elle ne gardait d'abord intacts que les échanges tenant dans 40 % de la part de l'historique : avec 8192 tokens, une courte session n'avait jamais rien à compresser.

### Le format du résumé

Le modèle produit du **JSON imposé par un schéma** (paramètre `format` d'Ollama, `temperature` à 0). Les champs sont pensés pour l'apprentissage d'une langue :

| Champ | Type | Contenu | Fusion |
|---|---|---|---|
| `apprenant` | liste | Préférences, objectifs, infos personnelles | Ajout à la suite |
| `vocabulaire` | liste | « mot = traduction » | Ajout à la suite |
| `erreurs` | liste | « erreur → correction (règle) » | Ajout à la suite |
| `exercices` | liste | Exercices faits et résultat | Ajout à la suite |
| `resume` | texte | 2 à 3 phrases sur l'extrait | Remplacé |
| `en_cours` | texte | Exercice non terminé, prochaine étape | Remplacé |

Le résumé est injecté dans le message système, en Markdown :

```
## Résumé des échanges précédents
Camille a travaillé les particules…
### Apprenant
- veut voyager à Kyoto en avril
### Vocabulaire vu
- inu = chien
### Erreurs à surveiller
- confond wa et ga
### En cours
QCM sur les couleurs
```

### Choix techniques

**Le modèle ne résume que le nouveau morceau. C'est le code qui fusionne.**
L'approche naïve consiste à envoyer « ancien résumé + nouveaux messages » et à demander un résumé complet réécrit. Avec un petit modèle local, chaque réécriture risque de perdre des faits, et les pertes s'accumulent de passe en passe. Ici :
- le modèle reçoit seulement les messages à résumer, plus le texte `resume` précédent comme contexte ;
- le code **ajoute** les nouveaux éléments aux listes existantes, en supprimant les doublons (sans tenir compte des majuscules ni des espaces).

Un fait déjà résumé ne peut donc disparaître que par une règle explicite et prévisible (voir ci-dessous).

**Le JSON plutôt que du texte libre.**
- On peut vérifier et fusionner le résultat champ par champ.
- Les champs obligent le modèle à chercher ce qui compte pour un apprenant (mots, erreurs) plutôt qu'à paraphraser la conversation.
- Les mêmes champs serviront de base à M3 (vocabulaire et erreurs en base de données).

**Un plafond strict, appliqué par le code.**
Si le résumé dépasse `summary_share`, on retire le contenu le moins utile d'abord, dans cet ordre :

1. les exercices les plus anciens ;
2. le texte narratif `resume` (raccourci, puis supprimé) : le récit passe avant les faits ;
3. le vocabulaire puis les erreurs les plus anciens (M3 les garde de toute façon en base) ;
4. le texte `en_cours` ;
5. en tout dernier recours, les faits sur l'apprenant les plus anciens.

Pas de nouvel appel au modèle, donc pas de surprise. Cet ordre vient d'une mesure : la première version raccourcissait `resume` en dernier, et le premier benchmark avec le vrai modèle a montré que les faits sur l'apprenant (ville du voyage, nom du chat…) étaient retirés alors qu'un long paragraphe narratif restait intact (voir [§13](#13-tests-et-mesures)).

### Stockage

```sql
summaries    (session_id, content, covered_until, tokens, updated_at)
compressions (id, session_id, messages_in, tokens_before, tokens_after, duration_ms, ok, error, created_at)
```

- `summaries` : un résumé par session (le JSON). `covered_until` est l'id du dernier message résumé. Avec `/load`, on recharge le résumé **et seulement les messages qui suivent**.
- Les messages résumés **restent en base** : on ne perd jamais l'historique brut, il n'est simplement plus envoyé au modèle.
- `compressions` : un journal de chaque passe, réussie ou non. Il alimente les statistiques de `/context` et les métriques de la keynote.

### En cas d'échec

Si Ollama ne répond pas ou renvoie du JSON invalide :
1. l'échec est enregistré dans `compressions` (avec le message d'erreur) et signalé dans le terminal ;
2. `ContextBuilder` revient au plan de secours : les plus vieux messages sont écartés ;
3. on attend 3 tours avant de réessayer, pour ne pas ralentir chaque message si le modèle échoue en boucle.

### Mesurer : le banc de test de rappel

Le catalogue précise que la note de M2 « dépend des métriques ». Le script `benchmarks/m2_recall.py` mesure si la compression conserve vraiment les faits :

1. il place 6 faits personnels au **début** d'une leçon synthétique (ville du voyage, mot préféré, mots confondus, type d'exercice préféré, nom du chat, compteur raté). Ils sont choisis pour **ne pas pouvoir être devinés** : un fait comme « inu = chien » serait retrouvé par la culture générale du modèle même si la mémoire l'avait perdu ;
2. il ajoute N échanges de remplissage pour pousser ces faits hors du contexte ;
3. il pose une question par fait, avec trois stratégies :

| Stratégie | Description |
|---|---|
| `full` | Grande fenêtre, rien n'est retiré (référence haute) |
| `truncate` | Petite fenêtre, les vieux messages sont écartés (sans M2) |
| `compress` | Petite fenêtre, les vieux messages sont résumés (avec M2) |

```bash
.venv/bin/python -m benchmarks.m2_recall --model qwen3-coder:30b --num-ctx 2048 --filler 30 --json results.json
```

Il affiche un tableau Markdown (rappel, tokens de prompt moyens, nombre et durée des compressions), prêt à coller dans la keynote. Un fait est compté comme retrouvé si la réponse contient un des mots-clés attendus, en mot entier (« hon » n'est pas trouvé dans « téléphone »). Résultats : [§13](#13-tests-et-mesures).

---

## 7. Mémoire structurée de l'apprenant — M3 (`memory/learner.py`, `memory/extractor.py`)

Le résumé de M2 garde le fil d'**une** session. M3 garde ce que l'apprenant sait **sur la durée, toutes sessions confondues** : les mots vus, quand les réviser, et les erreurs qui reviennent. C'est la « mémoire de prof » d'un Duolingo.

### Les données

```sql
vocabulary (user, language, word, translation, box, correct, wrong, next_review, created_at, updated_at)
           -- unique (user, language, word)
mistakes   (user, language, pattern, correction, example, count, last_seen)
           -- unique (user, language, pattern)
memory_ops (user, session_id, op, payload, applied, error, created_at)
```

- La langue vient du profil (`langue_cible`). Un apprenant qui étudie deux langues a deux vocabulaires séparés.
- Mots et erreurs sont stockés en minuscules, avec les espaces normalisés : « Inu » et « inu » sont le même mot, et une erreur déjà vue voit son compteur augmenter au lieu d'être dupliquée.
- `memory_ops` journalise **chaque opération proposée par le modèle**, acceptée ou rejetée, avec la raison du rejet. On peut toujours expliquer pourquoi la mémoire contient quelque chose.

### Révision espacée (système de Leitner)

Chaque mot est dans une « boîte » de 0 à 5. La boîte décide quand le mot doit être révisé :

| Boîte | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Prochaine révision | tout de suite | 1 jour | 3 jours | 7 jours | 14 jours | 30 jours |

- Bonne réponse : le mot monte d'une boîte.
- Mauvaise réponse : le mot retombe en boîte 0.
- Un mot est considéré comme **maîtrisé** à partir de la boîte 4.

Leitner plutôt que SM-2 (l'algorithme d'Anki) : il est plus simple à expliquer, il n'a besoin que de « juste / faux » (ce qu'un petit modèle local sait juger de façon fiable), et il suffit pour la démo.

### Écriture : c'est le modèle qui décide, le code qui applique

Après chaque échange, un appel séparé au modèle lit le message de l'apprenant, la réponse du tuteur, le profil, ainsi que les mots et erreurs déjà connus. Il renvoie une liste d'opérations, en JSON imposé par un schéma :

| Opération | Champs | Effet |
|---|---|---|
| `add_word` | word, translation | Ajoute un mot (ou met à jour sa traduction) |
| `review_word` | word, correct, (translation) | Fait monter ou redescendre le mot dans les boîtes de Leitner |
| `delete_word` | word | Oublie un mot |
| `add_mistake` | pattern, correction, (example) | Ajoute une erreur, ou augmente son compteur |
| `resolve_mistake` | pattern | Retire une erreur que l'apprenant maîtrise désormais |
| `set_profile` | key, value | Complète le profil (langue, niveau, objectif…) |

C'est ce qui fait de M3 une mémoire **pilotée par l'agent** : le modèle choisit lui-même de créer, modifier ou supprimer des entrées. Avant toute écriture, le code vérifie chaque opération :

- type d'opération connu, champs obligatoires présents, champs texte qui sont bien des chaînes de moins de 200 caractères, `correct` bien booléen ;
- pour `add_word` et `review_word` : un mot ou une expression de 3 mots au plus (pas une phrase), et une traduction de 60 caractères au plus (pas une explication) ;
- `review_word` sur un mot inconnu : accepté seulement si une traduction est fournie (le mot est alors créé) ;
- `delete_word` ou `resolve_mistake` sur quelque chose qui n'existe pas : rejeté ;
- au plus 12 opérations par échange.

Une opération invalide est **rejetée et journalisée**, sans bloquer les autres. Un `set_profile` qui répète exactement le profil actuel est écarté dès l'extraction : ce n'est pas un changement, il n'a pas à apparaître dans `memory_ops`.

Pour que le modèle réutilise exactement le même texte (et que les compteurs augmentent au lieu de créer des doublons), on lui donne les mots connus présents dans l'échange et les 10 erreurs les plus fréquentes.

### Choix techniques

**Un appel séparé plutôt que des appels d'outils pendant la réponse.**
On aurait pu laisser le tuteur appeler des outils (« enregistre ce mot ») en pleine réponse. On ne l'a pas fait :
- le tuteur reste concentré sur l'enseignement, et sa réponse continue de s'afficher progressivement ;
- un petit modèle local est bien plus fiable sur une seule tâche d'extraction étroite que pour décider en pleine réponse quand appeler un outil ;
- le schéma JSON garantit une sortie lisible par le code.

**L'extraction tourne en arrière-plan.**
Elle démarre dès que la réponse est affichée, dans un fil séparé, pendant que l'apprenant lit et tape. Ses opérations sont appliquées au début du message suivant (ou en quittant). Seul l'appel au modèle tourne en arrière-plan : la base SQLite n'est touchée que par le fil principal, ce qui évite les problèmes d'accès concurrent.

**La lecture est faite par le code, pas par le modèle.**
Choisir quoi injecter est déterministe : pas besoin d'un appel au modèle pour « chercher » dans la mémoire.

### Lecture : ce qui est injecté dans le contexte

À chaque message, `LearnerMemory.render()` construit un bloc ajouté au message système. Il est rempli par priorité, ligne par ligne, jusqu'à `memory_share` du budget :

1. **Mots présents dans le message** de l'apprenant (recherche par mot entier, pour que « inu » ne soit pas trouvé dans « inutile » ; par sous-chaîne pour le japonais et le chinois, qui s'écrivent sans espaces) ;
2. **Mots à réviser** (`next_review` dépassé), les plus fragiles d'abord ;
3. **Erreurs les plus fréquentes**.

```
## Mémoire de l'apprenant (japonais)
3 mots vus, 0 maîtrisés.

### Mots présents dans le message
- inu = chien (2 ✔ / 0 ✘)

### Mots à réviser
- neko = chat (0 ✔ / 1 ✘)

### Erreurs fréquentes
- confond wa et ga → wa = thème, ga = sujet (vue 2 fois) — ex. « watashi ga Camille desu »

Réutilise les mots à réviser dans tes exercices et surveille les erreurs fréquentes.
```

La dernière ligne demande au tuteur d'utiliser cette mémoire : c'est ce qui transforme le chatbot en prof qui fait réviser au bon moment.

### Lien avec M2

Le résumé de M2 contient aussi des listes `vocabulaire` et `erreurs`. Il y a un recoupement assumé : le résumé sert au fil de **la session en cours**, M3 à la mémoire **à long terme**. Comme M3 garde tout en base, le résumé peut retirer ses vieux mots et vieilles erreurs quand il manque de place sans que rien ne soit perdu.

### En cas d'échec

Si l'extraction échoue (JSON invalide, Ollama indisponible), l'échec est journalisé (`op = 'extract'`) et signalé dans le terminal, et la conversation continue normalement. Seules les opérations de cet échange sont perdues.

### Mesurer : le banc de test d'extraction

`benchmarks/m3_extraction.py` fait tourner l'extraction sur 11 échanges écrits à la main dont on connaît les bonnes opérations (nouveau mot, bonne et mauvaise réponse, erreur de particule, erreur maîtrisée, infos de profil, demande d'oubli, échanges sans rien à retenir). Les 4 derniers ont été ajoutés après le réglage de la consigne, pour vérifier qu'elle se généralise. Il mesure :

- la **précision** : part des opérations proposées qui étaient attendues ;
- le **rappel** : part des opérations attendues qui ont été proposées ;
- le nombre d'opérations que la validation **rejetterait** ;
- le nombre d'**extractions échouées** (JSON invalide).

```bash
.venv/bin/python -m benchmarks.m3_extraction --model qwen3-coder:30b --json m3.json
```

---

## 8. Utilisation

### Lancer

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
ollama serve                                  # dans un autre terminal
.venv/bin/python -m sensai                    # config.json par défaut
.venv/bin/python -m sensai --model qwen3:4b --num-ctx 16384 --user alice
```

### Commandes

| Commande | Effet |
|---|---|
| `/help` | Liste des commandes |
| `/new` | Nouvelle session |
| `/sessions` | Liste des sessions de l'apprenant (`*` = session active) |
| `/load <id>` | Reprend une session avec tout son historique |
| `/profile` | Affiche le profil |
| `/profile set <clé> <valeur>` | Ajoute ou modifie une info (la valeur peut contenir des espaces) |
| `/profile del <clé>` | Supprime une info |
| `/context` | Répartition du budget lors du dernier envoi + statistiques de compression |
| `/summary` | Affiche le résumé des anciens échanges |
| `/compress` | Force la compression des anciens échanges |
| `/vocab` | Vocabulaire appris, avec la boîte de Leitner et la date de la prochaine révision |
| `/vocab del <mot>` | Oublie un mot |
| `/mistakes` | Erreurs récurrentes, les plus fréquentes d'abord |
| `/memory` | Bloc mémoire injecté au dernier message |
| `/quit` | Quitter (Ctrl+D marche aussi) |

Ctrl+C pendant une réponse coupe la génération. La partie déjà reçue est conservée.

### Configuration (`config.json`)

| Clé | Défaut | Rôle |
|---|---|---|
| `model` | `qwen3-coder:30b` | Tag Ollama du modèle |
| `host` | `http://localhost:11434` | Adresse du serveur Ollama |
| `num_ctx` | `8192` | Taille de la fenêtre de contexte (tokens) |
| `response_reserve` | `0.2` | Part de `num_ctx` réservée à la réponse |
| `think` | `false` | Mode « thinking » de Qwen3 (`false` = plus rapide, `null` = ne pas envoyer le paramètre si le modèle ne le gère pas) |
| `db_path` | `data/sensai.db` | Base SQLite |
| `user` | `default` | Apprenant actif |
| `system_prompt` | tuteur façon Duolingo | Rôle du modèle |
| `compression` | `true` | Active la compression automatique (M2) |
| `compress_trigger` | `0.8` | Seuil de déclenchement (part de l'historique) |
| `compress_keep` | `0.4` | Part de l'historique gardée telle quelle (doit être < `compress_trigger`) |
| `summary_share` | `0.1` | Taille maximale du résumé (part du budget) |
| `memory` | `true` | Active la mémoire structurée (M3) : extraction et injection |
| `memory_share` | `0.1` | Taille maximale du bloc mémoire (part du budget). `summary_share + memory_share` doit rester ≤ 0,5 |

Les arguments `--model`, `--host`, `--num-ctx`, `--db`, `--user` et `--config` remplacent les valeurs du fichier.

### Gestion des erreurs

| Situation | Comportement |
|---|---|
| Ollama arrêté au démarrage | Message clair, sortie avec le code 1 |
| Modèle absent | Message avec la commande `ollama pull` à lancer |
| Connexion perdue pendant une réponse | Message d'erreur, le chat continue |
| Compression échouée (JSON invalide, Ollama indisponible) | Message, vieux messages écartés, nouvel essai 3 tours plus tard |
| Extraction mémoire échouée | Message, échec journalisé, la conversation continue |
| Opération mémoire invalide | Rejetée et journalisée, les autres opérations sont appliquées |
| Entrée vide | Ignorée |
| `config.json` invalide ou clé inconnue | Message d'erreur, sortie avec le code 2 |

---

## 9. Interface avec le RAG

Le RAG (géré par un coéquipier) se branche sur `ContextBuilder` via une seule fonction :

```python
def retrieve(query: str, max_tokens: int) -> list[str]:
    """Renvoie des morceaux de documents dont le total tient dans max_tokens."""

ContextBuilder(system_prompt, num_ctx, response_reserve, retriever=retrieve, rag_share=0.25)
```

Le contrat :

- `query` est le dernier message de l'utilisateur ;
- `max_tokens` = `budget × rag_share`, moins le coût de la consigne d'ancrage (`RAG_HEADER`, ≈ 105 tokens) placée au-dessus des morceaux : **c'est au RAG de respecter cette limite** (il peut utiliser `estimate_tokens` de `memory/tokens.py` pour compter de la même façon). La section entière (consigne + morceaux) tient ainsi dans sa part ;
- renvoyer une liste vide si rien n'est pertinent : aucune place n'est alors consommée.

Implémentation : [rag.md](rag.md) (R1 + R2).

---

## 10. Prochaines étapes

- **Élargir les mesures** (voir [§13](#13-tests-et-mesures)) : plus de cas pour M3, un benchmark M2 avec M3 actif et la fenêtre réelle de 8192 tokens, et le même protocole sur un second modèle.
- **Régler les parts du budget** maintenant que le RAG est branché : les valeurs actuelles (10 % mémoire, 10 % résumé, 25 % RAG) sont un point de départ.
- **M4 (document d'état)**, si le temps le permet : une fiche de progression de l'apprenant, mise à jour section par section, qui pourrait s'appuyer sur les tables de M3.

---

## 11. Limites actuelles

- Les mesures portent sur peu de cas (6 faits pour M2, 11 échanges pour M3) et un seul modèle : ce sont des indications, pas des garanties.
- La qualité du résumé dépend du modèle : un petit modèle local peut rater un fait, en mal formuler un, ou ranger une action de la séance parmi les faits sur l'apprenant. Le schéma JSON, la fusion par le code et l'ordre de rognage limitent les dégâts sans les supprimer.
- La compression ajoute un appel au modèle sur le tour où elle se déclenche : environ 25 s par passe sur CPU seul avec `qwen3-coder:30b`.
- L'extraction M3 ajoute un appel au modèle à chaque échange. Il tourne en arrière-plan, mais si l'apprenant répond très vite, le message suivant attend la fin de l'extraction.
- Les erreurs sont regroupées par texte exact : si le modèle formule la même erreur de deux façons différentes, elle est comptée deux fois. Lui fournir les erreurs connues limite ce risque sans le supprimer.
- Leitner ne note que « juste / faux » : il ne distingue pas une réponse hésitante d'une réponse immédiate.
- La suppression des doublons est textuelle : « neko = chat » et « neko = le chat » restent deux éléments.
- L'estimation des tokens est approximative. La réserve de 20 % sert de marge de sécurité.
- Le modèle peut modifier le profil (`set_profile`) : une mauvaise interprétation peut écraser une valeur. Chaque changement est journalisé dans `memory_ops`, et `/profile set` permet de corriger.
- **La mémoire hérite des erreurs du tuteur.** L'extraction lit la réponse du tuteur : si celui-ci se trompe (« 1 billet = ippai » au lieu de « ichimai », vu en répétition alors que la bonne leçon était dans le contexte), l'erreur est enregistrée comme vocabulaire. De même, si le tuteur juge à tort une réponse fausse, le mot redescend en boîte 0. `/vocab del` permet de corriger ; une vérification contre les documents du RAG serait la vraie parade.
- Un même mot peut exister en deux écritures (« ねこ » et « neko ») : la consigne demande l'écriture de l'apprenant, mais rien ne fusionne kana et romaji.

---

## 12. User stories

Fil rouge : Camille apprend le japonais avec Sensai, le soir, sur son portable, hors ligne.

### B1 — Chatbot CLI

**US-B1-1.** En tant qu'apprenante qui révise avec un modèle local, je veux voir la réponse du tuteur s'écrire au fur et à mesure, pour ne pas attendre devant un écran vide pendant que le modèle génère.
Critères d'acceptation :
- le texte de la réponse apparaît morceau par morceau (streaming de `/api/chat`), pas en un seul bloc à la fin ;
- Ctrl+C pendant la réponse coupe la génération, la partie déjà reçue est gardée dans l'historique et le chat continue ;
- une question de relance (« et au pluriel ? ») est comprise, car le modèle reçoit les tours précédents de la session.

**US-B1-2.** En tant qu'enseignant qui installe Sensai sur les machines de la classe, je veux choisir le modèle sans toucher au code et obtenir des erreurs compréhensibles, pour dépanner sans lire de traceback.
Critères d'acceptation :
- `--model <tag>` ou la clé `model` de `config.json` change le modèle ; l'argument l'emporte sur le fichier ;
- Ollama arrêté au démarrage → message clair et code de sortie 1 ; modèle absent → message avec la commande `ollama pull <tag>` ;
- une entrée vide est ignorée ; un `config.json` invalide ou une clé inconnue → message d'erreur et code de sortie 2 ;
- une connexion perdue pendant une réponse affiche une erreur, sans quitter le chat.

### M1 — Sessions persistantes et profil

**US-M1-1.** En tant qu'apprenante, je veux que Sensai connaisse ma langue cible et mon niveau à chaque nouvelle session, pour ne pas avoir à me représenter à chaque fois.
Critères d'acceptation :
- après `/profile set langue_cible japonais` puis `/quit` et relance, `/profile` affiche toujours `langue_cible: japonais` ;
- le profil est ajouté au message système à chaque tour (section « Profil de l'apprenant »), sans que l'apprenante le redise ;
- `/profile del niveau` retire l'information, qui n'est plus injectée au tour suivant.

**US-M1-2.** En tant qu'apprenante qui révise en plusieurs fois, je veux reprendre une leçon interrompue la veille, pour continuer là où je m'étais arrêtée.
Critères d'acceptation :
- `/sessions` liste mes sessions, de la plus récente à la plus ancienne, avec la date, le nombre de messages et un titre tiré du premier message ;
- `/load <id>` recharge la session ; si elle avait été compressée, le résumé est rechargé avec **seulement** les messages qui le suivent ;
- `/load` d'un identifiant inexistant ou d'une session d'un autre apprenant affiche « introuvable » et ne change pas la session active.

**US-M1-3.** En tant que parent dont les deux enfants partagent l'ordinateur, je veux que chacun ait sa propre progression, pour que leurs profils et leurs mots ne se mélangent pas.
Critères d'acceptation :
- `--user camille` et `--user hugo` ont chacun leur profil, leurs sessions, leur vocabulaire et leurs erreurs ;
- `/sessions` et `/load` ne voient que les sessions de l'apprenant actif.

### M2 — Budget de tokens et compression

**US-M2-1.** En tant qu'apprenante qui fait une longue séance, je veux que le tuteur se souvienne de ce que j'ai dit au début (mon voyage à Kyoto, les mots déjà vus, mes erreurs), pour que la séance reste cohérente même quand elle dépasse la fenêtre du modèle.
Critères d'acceptation :
- quand l'historique dépasse `compress_trigger` (80 %) de sa part du budget, les plus vieux tours sont résumés (`[compression de N anciens messages…]`), les plus récents restent tels quels ;
- `/summary` montre les faits sur l'apprenant, le vocabulaire, les erreurs et l'exercice en cours, et un fait résumé lors d'une passe est encore là après les passes suivantes ;
- tant que la compression réussit, `/context` affiche 0 message écarté ;
- sur `benchmarks/m2_recall.py`, la stratégie `compress` retrouve plus de faits que `truncate` à fenêtre égale.

**US-M2-2.** En tant que développeur de Sensai, je veux que le prompt envoyé tienne toujours dans `num_ctx` et qu'un échec de compression ne bloque pas la séance, pour éviter la troncature silencieuse d'Ollama.
Critères d'acceptation :
- `num_ctx` est envoyé dans chaque requête, et `/context` montre un total estimé inférieur ou égal au budget, `num_ctx × (1 − response_reserve)` divisé par la calibration (seul le message actuel, s'il est énorme, peut dépasser) ;
- quand Ollama compte plus de tokens que l'estimation, le budget est réduit dès le tour suivant, et `/context` affiche le compte réel et la calibration ;
- le résumé ne dépasse jamais `summary_share` du budget : les éléments les plus anciens sont retirés, sans nouvel appel au modèle ;
- si la compression échoue (JSON invalide, Ollama coupé), un message s'affiche, les plus vieux messages sont écartés, l'échec est compté dans `/context` et un nouvel essai a lieu 3 tours plus tard ;
- les messages résumés restent en base : rien n'est supprimé de l'historique brut.

### M3 — Mémoire structurée de l'apprenant

**US-M3-1.** En tant qu'apprenante de japonais, je veux que Sensai me refasse travailler les mots que j'oublie au bon moment, pour retenir le vocabulaire sur la durée.
Critères d'acceptation :
- un mot enseigné pendant l'échange apparaît dans `/vocab` (boîte 0) sans action de ma part ;
- une bonne réponse fait monter le mot d'une boîte et repousse sa révision (1, 3, 7, 14 puis 30 jours) ; une mauvaise le renvoie en boîte 0 ;
- les mots dont la révision est due apparaissent sous « Mots à réviser » dans `/memory`, les plus fragiles d'abord, et le bloc reste sous `memory_share` du budget.

**US-M3-2.** En tant qu'apprenant, je veux que le tuteur repère les erreurs que je refais, pour qu'il insiste dessus au lieu de corriger chaque fois comme si c'était la première.
Critères d'acceptation :
- une erreur déjà connue voit son compteur augmenter (`/mistakes` : « vue N fois ») au lieu d'être enregistrée en double ;
- les erreurs les plus fréquentes sont injectées dans le contexte, avec la consigne de les surveiller ;
- une erreur que l'apprenant maîtrise désormais (`resolve_mistake`) disparaît de `/mistakes`.

**US-M3-3.** En tant qu'enseignante qui veut faire confiance à l'outil, je veux que ce que le modèle écrit dans la mémoire soit contrôlé et traçable, pour pouvoir expliquer et corriger son contenu.
Critères d'acceptation :
- chaque opération proposée par le modèle est vérifiée avant écriture (type connu, champs présents et de moins de 200 caractères, `correct` booléen, au plus 12 par échange) ;
- une opération invalide est rejetée et journalisée dans `memory_ops` avec sa raison, sans empêcher les autres d'être appliquées ;
- un mot traduit par lui-même (« voyage = voyage »), ou un mot qui ne peut pas s'écrire dans la langue cible (« vol » ou « avril » notés comme japonais : ni kana, ni rōmaji valide), est rejeté : ce n'est pas du vocabulaire ;
- une extraction échouée affiche un message, la conversation continue, et `/context` compte les opérations appliquées et les extractions échouées ;
- `/vocab del <mot>` et `/profile set` permettent de corriger une erreur du modèle.

---

## 13. Tests et mesures

### Tests automatisés

`.venv/bin/python -m unittest tests.test_memory -v` : **47 tests**, sans réseau ni modèle (faux client Ollama scripté : résumés pour la compression, opérations pour l'extraction, réponse fixe en streaming), en moins d'une seconde. Les bases SQLite sont réelles, dans un dossier temporaire.

| Partie | Ce qui est vérifié |
|---|---|
| Estimation des tokens | règle latine, 1 token par caractère japonais/chinois/coréen, surcoût par message |
| M1 — `MemoryStore` | messages et titre de session, rechargement après un id, sessions séparées par apprenant, profil persistant après redémarrage, résumé et statistiques de compression |
| `ContextBuilder` | budget, injection du profil, ordre mémoire → résumé → documents, part réservée au RAG seulement s'il est branché, historique gardé du plus récent au plus ancien, message actuel toujours gardé, vrais nombres de tokens prioritaires |
| M2 — résumé et compression | JSON invalide refusé, fusion sans doublon, rognage (récit avant les faits, faits sur l'apprenant en dernier), préfixes « Apprenant : » retirés, choix des tours à résumer (et `/compress` forcé : tout sauf le dernier message), faits conservés d'une passe à l'autre, plafond respecté, erreurs d'Ollama ou de JSON |
| M3 — mémoire de l'apprenant | normalisation des mots, boîtes de Leitner et dates de révision, mots maîtrisés, séparation par langue et par apprenant, compteur d'erreurs, opérations valides et invalides (dont champs non textuels, phrases entières et explications en guise de traduction), plafond de 12 opérations, recherche en mot entier et en japonais, bloc injecté sous son plafond ; extraction : contenu du prompt, profil répété écarté, échecs |
| Boucle de chat (`Chat`) | longue session compressée puis rechargée avec `/load`, compression en échec → repli et attente de 3 tours, extraction appliquée au message suivant, commandes `/profile`, `/load`, `/sessions` |

Défaut trouvé par ces tests : une opération mémoire dont un champ n'est pas un texte (`"word": 5`) faisait planter le chat (`AttributeError` non attrapée dans la boucle). Elle est maintenant rejetée et journalisée comme toute opération invalide.

### Mesures avec le vrai modèle

Conditions : `qwen3-coder:30b` (modèle MoE de 30B paramètres, dont environ 3B actifs par token), Ollama 0.34.4, **CPU seul** (Ryzen 5 7600X, 32 Go de RAM : le paquet Ollama installé n'utilisait pas le GPU), le 5 octobre 2026. Pour retrouver les réponses détaillées, relancer les scripts avec `--json`.

#### M2 — les faits survivent-ils à la compression ?

`python -m benchmarks.m2_recall --num-ctx 2048 --filler 30` : 72 messages, 6 faits personnels au début, puis 30 échanges de remplissage. Fenêtre de 2048 tokens pour `truncate` et `compress`, ce qui plafonne le résumé à 163 tokens. Fenêtre de 32 768 tokens pour `full`.

| Mesure | full | truncate (sans M2) | compress (avec M2) |
|---|---|---|---|
| 1. Code initial, 3 faits devinables sur 6 | 6/6 | 3/6 (que des faits devinés) | 2/6 (que des faits devinés) |
| 2. Faits non devinables ; rognage et consigne corrigés | 6/6 | **0/6** | **4/6** |
| 3. + contre-exemple dans la définition du champ `apprenant` | — | — | **4/6** (les deux mêmes faits perdus) |

Tokens de prompt moyens à la 2e mesure : 2766 (full), 1643 (truncate), 1134 (compress). Trois passes de compression, 77 s au total, soit environ 25 s par passe sur CPU.

Ce que la première mesure a appris, qu'aucun test avec un faux modèle ne pouvait montrer :

1. **Le benchmark mesurait la culture générale du modèle.** Les 3 faits « retrouvés » par truncate (« inu = chien », la confusion wa/ga, « QCM ») étaient devinés : par exemple, à la question sur son exercice préféré, le modèle répondait « je ne sais pas, mais je peux te proposer un QCM », ce qui était compté comme juste. Les faits ont été remplacés par des faits personnels impossibles à deviner, et les mots-clés sont comparés en mots entiers.
2. **Le rognage sacrifiait les faits au profit du récit.** Dès la 1re passe, le résumé faisait 417 tokens pour un plafond de 163. Le code retirait alors tous les faits sur l'apprenant (Kyoto, Mochi…) et gardait intact le paragraphe narratif `resume`, qui n'était raccourci qu'en dernier recours. L'ordre est maintenant : exercices, récit, vocabulaire, erreurs, étape en cours, et les faits sur l'apprenant en tout dernier.
3. **Le modèle recopiait le contexte.** À chaque passe, il reformulait les faits du résumé précédent sous forme de nouveaux éléments, avec des préfixes « Apprenant : ». Ce n'étaient pas des doublons exacts, donc ils n'étaient pas dédupliqués. La consigne interdit maintenant de répéter le contexte, et le code retire ces préfixes.
4. **Le modèle range des actions de la séance parmi les faits sur l'apprenant** (« demande des exercices sur les thèmes 13 à 19 »), et il lui arrive d'inventer une préférence (« préfère les exercices de traduction »). Ces éléments s'accumulent et poussent les vrais faits hors du résumé à la 3e passe. La définition du champ `apprenant` donne maintenant un contre-exemple.

Aux 2e et 3e mesures, les deux faits perdus sont des difficultés de l'apprenant (« confond migi et hidari », « a raté le compteur hon ») : avec un plafond de 163 tokens, le vocabulaire et les erreurs sont retirés avant les faits personnels. Dans Sensai, M3 les garde en base et les réinjecte (`mistakes`, `vocabulary`) : le benchmark mesure M2 seul, sans M3.

#### M3 — le modèle propose-t-il les bonnes opérations ?

`python -m benchmarks.m3_extraction` : échanges écrits à la main dont on connaît les opérations attendues.

| Mesure | Cas | Précision | Rappel | Opérations rejetées | Extractions échouées | Temps par extraction |
|---|---|---|---|---|---|---|
| 1. Consigne initiale | 7 | 67 % | 75 % | 0 | 0 | 5,0 s |
| 2. Consigne précisée, `set_profile` répété écarté | 7 (les mêmes) | 100 % | 100 % | 0 | 0 | 4,8 s |
| 3. Même consigne, + 4 cas inédits, une erreur déjà connue fournie au modèle | 11 | 100 % | 92 % | 0 | 0 | 4,0 s |

Les deux écarts de la 1re mesure :
- à « mizu ça veut dire feu ? », le modèle notait une *erreur* au lieu d'une mauvaise réponse sur le mot (`review_word` faux) et ne créait pas le mot « hi ». Il inventait aussi l'exemple (« mizu = chien »). La consigne précise maintenant qu'un mot mal traduit est un `review_word` faux, qu'un mot donné par le tuteur dans une correction est un `add_word`, et que l'exemple doit être recopié tel quel ;
- à « je suis débutant et j'apprends pour un voyage à Osaka », il répétait aussi la langue cible et la langue maternelle, déjà dans le profil. Ces `set_profile` sans changement sont maintenant écartés par le code.

La 2e mesure porte sur les cas qui ont servi à ajuster la consigne : elle montre que les corrections marchent, pas que la consigne se généralise. D'où la 3e mesure, avec 4 cas écrits après le réglage et jamais utilisés pour l'ajuster : un mot mal traduit (« inu, c'est chat ? »), une erreur connue enfin maîtrisée, un objectif de profil et une discussion banale. Trois sont parfaits. Le modèle n'a pas proposé `resolve_mistake` quand l'apprenant réussit enfin « watashi wa Camille desu » : il reste prudent, ce qui se défend (une bonne réponse isolée ne prouve pas que l'erreur est maîtrisée). Sur ces 4 cas : 3 opérations attendues sur 4, aucune en trop.

#### En conversation réelle (répétition de la démo)

Les benchmarks portent sur des échanges courts et nets. La répétition de la [démo](demo.md) dans la vraie CLI a montré d'autres défauts de M3 :

| Constat | Avant | Correction | Après |
|---|---|---|---|
| Vocabulaire bruité après 2 échanges : prénom, phrase entière (« neko ga suki desu. »), particules avec une définition en guise de traduction | 7 entrées, dont 4 inutiles | consigne (`add_word` : 3 mots au plus, écriture de l'apprenant, traduction courte, pas de nom propre) + validation (3 mots, 60 caractères) | 3 entrées |
| Le mot enregistré en kana (« ねこ ») n'est pas reconnu quand l'apprenant tape « neko » | toujours | consigne : écriture de l'apprenant | « neko » est bien créé, mais « ねこ » aussi |
| Prénom enregistré comme mot | « camille = Camille » | consigne, puis (25 octobre) validation de la forme du mot | rejeté : « Camille » n'est ni en kana ni en rōmaji valide |
| Erreur du tuteur enregistrée (« ippai = un billet ») | — | aucune (voir §11) | limite connue |
| Mots **français** enregistrés comme vocabulaire japonais, après des réponses d'information en français (essais du 25 octobre : « voyage = voyage », « acompte = acompte », puis, au second essai, « vol = transport aérien », « avril = mois ») | 29 mots parasites sur 35 opérations | validation par le code (`learner.looks_like`) : un mot traduit par lui-même est refusé ; un mot japonais doit être en kana/kanji ou en rōmaji Hepburn valide (syllabes consonne + voyelle, n syllabique, consonne doublée) ; un mot coréen en hangul ou sans f, q, v, x, z ni accents | les 29 parasites rejetés ; gardés : hon, mai, hiki, wa, ga, Kyoto. Sur les 178 mots japonais des tableaux de leçons, un seul refusé : « Camille », un prénom |

Le benchmark M3 reste à 100 % de précision et 92 % de rappel avec la consigne et la validation renforcées : pas de régression. Une règle de plus dans la consigne (« jamais un mot français ») a été essayée contre les mots parasites : elle faisait tomber la précision à 92 % (une opération fausse proposée sur « watashi wa Camille desu »). Elle a été retirée : la validation par le code suffit, sans toucher au comportement du modèle.
