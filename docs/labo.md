# Labo : apprenants, historique, fonctionnalités activables et comparaison avec / sans

Trois ajouts qui servent la même idée : **montrer ce que chaque brique apporte**, sur les questions réelles de l'apprenant, au lieu de l'affirmer.

| Ajout | Ce qu'il permet | Où |
|---|---|---|
| **Apprenants et historique** (M1 étendu) | créer plusieurs profils d'apprenant, passer de l'un à l'autre sans redémarrer, voir son historique : série de jours, activité sur 14 jours, sessions relisibles sans les reprendre | `sensai/memory/store.py`, `Chat.switch_user`, onglet **Profil** |
| **Fonctionnalités activables** | couper à chaud l'historique (B1), le profil (M1), la mémoire (M3), le résumé (M2), le RAG (R1) et ses étages R2 (BM25, rerank, filtre de langue) ; chaque réponse garde la trace de ce qui était coupé | `sensai/features.py`, onglet **Labo**, `/features` |
| **Comparer avec / sans** | poser la même question deux fois, avec puis sans une fonctionnalité, voir les deux réponses côte à côte avec le contenu de chaque prompt, et un arbitrage du modèle | `sensai/compare.py`, mode **Comparer**, `/compare` |

Une leçon a été ajoutée pour la démonstration du RAG : `docs/lecons/interro_vendredi.md`, les consignes de l'interro n°2 (date, salle, programme, barème, liste de mots, consignes du professeur). Le modèle ne peut pas les deviner : c'est le cas d'usage où le RAG est indispensable, et non un simple confort.

---

## 1. Apprenants et historique

Avant, l'apprenant était fixé au démarrage (`--user camille`) et l'historique se limitait à une liste de sessions. Maintenant :

- une table `users` liste les apprenants ; elle est remplie automatiquement à partir des sessions et profils existants (migration sans perte, testée sur une base à l'ancien schéma) ;
- **changer d'apprenant** (`/user hugo`, menu 👤 de l'en-tête, onglet Profil) termine d'abord la mémorisation du dernier échange de l'apprenant précédent, puis reprend la **dernière session** du nouveau (ou en crée une) ; profil, vocabulaire, erreurs, résumé et filtre de langue du RAG suivent ;
- **supprimer un apprenant** efface ses sessions, son profil, son vocabulaire, ses erreurs et le journal de ses opérations mémoire ; l'apprenant en cours ne peut pas être supprimé ;
- **l'historique** : série de jours consécutifs (🔥, à la Duolingo : la série tient tant que le jour n'est pas fini), jours actifs, messages envoyés, mots maîtrisés, un histogramme des messages des 14 derniers jours (valeur au survol et au clavier), et la liste des sessions non vides ; **Lire** ouvre une session en lecture seule, **Reprendre** la recharge ;
- chaque réponse enregistrée porte ses **métadonnées** (colonne `messages.meta`, JSON) : fonctionnalités coupées et leçons utilisées. Elles s'affichent sous la bulle (« ✂ sans Leçons (RAG) », « 📄 animaux.md ») dans la conversation et dans l'aperçu d'une ancienne session. On peut donc relire une séance et savoir dans quelles conditions chaque réponse a été produite.

Les noms d'apprenant sont limités à 1–40 lettres, chiffres, `-` ou `_` (pas de chemin, pas de HTML).

## 2. Fonctionnalités activables

| Interrupteur | Feature | Coupé, il se passe ceci |
|---|---|---|
| `history` | B1 | le modèle ne reçoit que le message en cours (ni les échanges précédents, ni leur résumé) ; ils restent enregistrés |
| `profile` | M1 | le profil (langue cible, niveau…) n'est plus ajouté au rôle du tuteur |
| `memory` | M3 | ni bloc « Mémoire de l'apprenant » dans le prompt, ni extraction après la réponse |
| `compression` | M2 | pas de nouveau résumé, et le résumé existant n'est plus injecté : les vieux messages sortent sans trace |
| `rag` | R1 | aucun passage de leçon ; la part du budget réservée au RAG revient à l'historique |
| `rag_hybrid` | R2 | classement par le cosinus seul (BM25 à zéro) |
| `rag_rerank` | R2 | pas d'appel de reclassement : les candidats passent le seuil cosinus sans tri par le modèle |
| `rag_filters` | R2 | le filtre souple dérivé du profil (`langue_cible` → `langue`) n'est plus appliqué |

Mécanique : `Chat.switches` est lu **à chaque tour** par `Chat.build_prompt()`, et `Chat._apply_switches()` répercute l'état sur les composants (le `retriever` du `ContextBuilder`, les attributs `hybrid`, `rerank`, `profile_filters` du pipeline RAG). Aucun redémarrage, aucun objet reconstruit. Les valeurs initiales viennent de `config.json` (nouvelle clé `rag_hybrid`), et « Rétablir » y revient. Les sous-interrupteurs R2 dépendent de `rag` : coupés avec lui, ils ne sont pas listés comme « coupés » sur les réponses. Si le RAG n'a pas pu démarrer (embedder absent), ses interrupteurs sont marqués indisponibles plutôt que faussement « coupés ».

Côté interface, l'onglet **Labo** liste les interrupteurs par groupe, avec ce que chacun apporte et, une fois coupé, ce qu'il faut observer. Un bandeau au-dessus du champ de saisie rappelle ce qui est coupé (« ✂ Coupé : Leçons (RAG) — Tout rétablir ») pour ne pas l'oublier en pleine démo. La jauge de l'onglet Contexte montre la place libérée.

## 3. Comparer avec / sans

Le mode **Comparer** (ou `/compare rag Qu'est-ce qui tombe à l'interro de vendredi ?`) :

1. construit **deux prompts pour le même message**, l'un avec la fonctionnalité, l'autre sans ; tout le reste est identique (historique, profil, mémoire…). Pour un sous-interrupteur R2, le RAG est forcé à « on » des deux côtés ;
2. génère les deux réponses l'une après l'autre (une seule carte graphique : en parallèle, elles se ralentiraient mutuellement), en flux, dans deux colonnes ;
3. sous chaque réponse : tokens du prompt par section (rôle, mémoire, leçons, historique), durée, et les passages de leçons injectés ; les passages présents **d'un seul côté** sont surlignés (c'est la comparaison utile pour les étages R2) ;
4. si l'arbitrage est demandé et que la fonctionnalité a apporté quelque chose au prompt (passages de leçons, bloc mémoire, profil, échanges précédents, résumé), **le modèle note chaque réponse de 0 à 10** sur sa fidélité à ces informations, avec une raison et un verdict.

**Rien n'est enregistré** : ni message, ni extraction mémoire, ni compression ; les interrupteurs, le dernier rapport de contexte et le dernier résultat RAG affichés dans le panneau sont remis en place, même si l'on appuie sur Stop au milieu. Les appels sont journalisés (`purpose` = `compare`, `judge`, événement `compare`), donc visibles dans `/metrics`.

### L'arbitrage, et ses limites

- Le juge reçoit les **informations de référence** (ce que la fonctionnalité a ajouté), le message, et les deux réponses **dans un ordre tiré au hasard** : un modèle a tendance à préférer la première réponse qu'il lit, l'ordre aléatoire annule ce biais en moyenne. Les notes sont remises dans l'ordre « avec / sans » ensuite.
- Il ne note pas le style ni la longueur, mais l'usage correct des informations : une réponse juste mais générique plafonne à 5, une réponse qui contredit ou invente des faits sur le cours descend vers 0.
- Pas d'arbitrage pour les étages R2 : leur effet est de **changer les passages retenus**, ce que la carte montre directement.
- **Limite honnête** : c'est le même modèle qui répond et qui juge (pas de second modèle sur la machine). Le verdict est un indicateur pour la démonstration, pas une mesure : la mesure, c'est le benchmark ci-dessous, qui vérifie des faits attendus mot à mot.

## 4. Mesurer l'intérêt du RAG : `benchmarks/rag_ab.py`

Dix questions, posées seules (sans historique, mémoire ni profil), à température 0 :
- **8 questions sur le cours** dont la réponse est dans `docs/lecons` et nulle part ailleurs (date et salle de l'interro, programme, barème, consignes, phrase modèle d'une leçon, présentation de la leçon de coréen) ;
- **2 questions de contrôle** de culture générale (« chien » en japonais, « arigatou gozaimasu ») pour vérifier que le RAG ne dégrade pas les réponses faciles.

Une réponse est juste si elle contient, pour chaque groupe de mots-clés attendus, au moins l'un d'eux (mot entier : « B2045 » ne valide pas « B204 »).

```bash
.venv/bin/python -m benchmarks.rag_ab                                      # off vs full
.venv/bin/python -m benchmarks.rag_ab --variants off,dense,hybrid,full --json data/rag_ab.json
```

| Variante | Questions sur le cours | Contrôle | Tokens de prompt (moy.) | Temps par question |
|---|---|---|---|---|
| off (sans RAG) | *à mesurer* | | | |
| full (R1 + R2) | *à mesurer* | | | |

Résultat attendu : sans RAG, les questions sur le cours échouent (le modèle n'a aucun moyen de connaître la salle B204 ni le barème) ; avec, elles passent, au prix de quelques centaines de tokens de prompt et d'un appel de rerank. Les variantes `dense` et `hybrid` isolent l'apport de BM25 et du rerank.

## 5. API ajoutée

| Méthode | Chemin | Rôle |
|---|---|---|
| GET | `/api/users` | apprenants (langue, mots, sessions, dernière activité) |
| POST | `/api/users` | `{name}` : créer un apprenant et passer à lui (409 s'il existe) |
| POST | `/api/users/{nom}/select` | passer à un apprenant, reprendre sa dernière session |
| DELETE | `/api/users/{nom}` | supprimer un apprenant et toutes ses données (409 pour l'apprenant en cours) |
| GET | `/api/history` | série, jours actifs, activité des 14 derniers jours, sessions non vides |
| GET | `/api/sessions/{id}` | lire une session sans la reprendre |
| GET | `/api/features` | interrupteurs : état, disponibilité, explication |
| PUT | `/api/features/{id}` | `{enabled}` : couper ou rétablir (404 inconnu, 409 indisponible) |
| POST | `/api/features/reset` | revenir à `config.json` |
| POST | `/api/compare` | `{text, feature, judge}` : flux SSE `side`, `token`, `side_done`, `verdict`, `notice`, `end` (ou `stopped`) |

`GET /api/state` renvoie en plus `switches`, et chaque message son `meta` ; l'événement `done` d'un tour de conversation renvoie `off` (fonctionnalités coupées pour cette réponse).

## 6. Choix et alternatives écartées

| Choix | Alternative écartée | Pourquoi |
|---|---|---|
| Interrupteurs en mémoire, remis à `config.json` par « Rétablir » | interrupteurs persistés par apprenant | c'est un outil de démonstration et de diagnostic : un réglage oublié ne doit pas survivre à un redémarrage et dégrader le tuteur en silence |
| Trace des fonctions coupées **sur chaque réponse** (`messages.meta`) | rien, ou un journal à part | en relisant une session, on sait pourquoi une réponse était pauvre ; le journal EV3 garde aussi chaque bascule (`feature_switch`) |
| Comparaison sans rien enregistrer | enregistrer les deux réponses dans la session | deux réponses à la même question casseraient l'historique et la mémoire (M3 extrairait deux fois) |
| Les deux réponses l'une après l'autre | en parallèle | Ollama sur une seule carte : en parallèle, chaque réponse serait plus lente et la seconde attendrait de toute façon |
| Juge = même modèle, ordre aléatoire, sortie JSON contrainte | second modèle juge ; pas de juge | pas de place pour un second modèle sur 8 Go de VRAM ; sans juge, le public doit lire deux réponses longues pour voir la différence. Le benchmark à mots-clés reste la mesure objective |
| Leçon de démonstration au contenu propre à la classe | questions sur les leçons existantes (vocabulaire de base) | « neko = chat », le modèle le sait déjà : avec ou sans RAG la réponse est juste, et la démonstration ne prouve rien |
| Table `users` remplie depuis les données existantes | demander de recréer les profils | aucune perte pour les bases déjà utilisées en répétition |

## 7. Tests

`tests/test_features.py` (**22 tests**, sans modèle) et 2 tests HTTP de plus dans `tests/test_web.py` :

| Partie | Ce qui est vérifié |
|---|---|
| Interrupteurs | valeurs initiales depuis la config ; RAG indisponible = non activable ; historique coupé = seul le message en cours est envoyé (mais tout reste enregistré) ; profil, mémoire (injection **et** extraction), résumé, RAG coupés = section absente du prompt ; le budget du RAG revient à l'historique ; les sous-interrupteurs R2 atteignent le pipeline (BM25 à zéro) ; réponses enregistrées avec `off` et `sources` ; commandes `/feature` |
| Comparaison | deux côtés, réponses différentes selon le prompt, **rien d'enregistré ni extrait**, interrupteurs et dernier rapport inchangés ; le juge reçoit les leçons comme référence ; **ordre aléatoire** vérifié et notes remises du bon côté ; pas de juge sans référence, juge en panne ou réponse illisible = simple avertissement ; sous-interrupteur R2 = RAG forcé des deux côtés (la leçon de coréen entre sans le filtre) ; arrêt en plein milieu = tout est remis en place ; requêtes invalides |
| Apprenants | changement d'apprenant : extraction de l'ancien appliquée d'abord, profil et sessions isolés, dernière session reprise ; suppression complète, jamais l'apprenant en cours ; nom invalide refusé ; migration d'une base à l'ancien schéma ; série de jours et 14 derniers jours ; commandes `/user`, `/users` |
| API | apprenants (création, doublon 409, nom invalide 422, sélection, lecture d'une session d'un autre apprenant refusée 404, suppression), historique, interrupteurs (404/409), `off` sur l'événement `done` et dans `meta`, comparaison en SSE sans rien enregistrer |

L'interface a été vérifiée dans un Chrome headless (bureau 1360 px et mobile 390 px en thème sombre) sur un faux modèle : bascule des interrupteurs, réponse étiquetée, carte de comparaison et arbitrage, historique, aperçu de session, changement d'apprenant, sans erreur dans la console.

## 8. Scénario de démonstration (2 minutes)

1. Onglet **Labo**, couper **Leçons (RAG)**. Demander au tuteur : « Qu'est-ce qui tombe à l'interro de vendredi ? ». Il répond de culture générale (hiragana, wa/ga…) : faux, wa/ga est justement reporté. La réponse porte « ✂ sans Leçons (RAG) ».
2. **Tout rétablir**, mode **Comparer**, même question : à gauche la date, la salle B204, le programme et le barème tirés de `interro_vendredi.md` ; à droite l'invention ; l'arbitrage chiffre l'écart.
3. Comparer **Filtre par langue du profil** sur « Comment dit-on « chat » ? » : sans le filtre, la leçon de coréen (고양이) peut entrer dans le prompt, surlignée (à vérifier en répétition avec `/rag search`, cela dépend des scores).
4. Onglet **Profil** : passer à l'apprenant « hugo » (coréen) ; son historique, ses mots et ses leçons suivent ; revenir à « camille », relire la séance d'hier avec **Lire**.

## 9. User stories

**US-LAB-1 — Profils d'apprenant.** En tant que famille qui partage un ordinateur, je veux que chacun ait son profil d'apprenant, pour que le tuteur ne mélange pas nos langues, nos mots et nos erreurs.
Critères d'acceptation :
- « ＋ Nouvel apprenant » (ou `/user <nom>`) crée un profil vide et y bascule sans redémarrer ;
- après le changement, le profil, le vocabulaire, les erreurs, les sessions et le filtre de langue du RAG sont ceux du nouvel apprenant ;
- le dernier échange de l'apprenant précédent est mémorisé chez lui, pas chez le nouveau ;
- supprimer un apprenant efface toutes ses données ; l'apprenant en cours ne peut pas être supprimé ;
- un nom contenant `/`, `<` ou des espaces est refusé.

**US-LAB-2 — Historique.** En tant qu'apprenante, je veux voir mon historique de travail et relire mes anciennes séances, pour mesurer ma régularité et retrouver une explication.
Critères d'acceptation :
- l'onglet Profil affiche la série de jours consécutifs, les jours actifs, les messages envoyés et les mots maîtrisés ;
- un histogramme montre les messages des 14 derniers jours, avec la valeur exacte au survol ou au clavier ;
- « Lire » ouvre une ancienne session en lecture seule, sans changer la session en cours ; « Reprendre » la recharge ;
- chaque réponse relue indique les fonctionnalités coupées et les leçons utilisées à ce moment-là.

**US-LAB-3 — Couper une fonctionnalité.** En tant qu'enseignant qui présente Sensai, je veux couper une brique (RAG, mémoire, historique…) en direct, pour montrer ce qu'elle apporte au lieu de l'affirmer.
Critères d'acceptation :
- l'interrupteur prend effet au message suivant, sans redémarrer ; la jauge de contexte montre la section disparue et la place libérée ;
- un bandeau rappelle ce qui est coupé, et « Tout rétablir » revient à `config.json` ;
- la réponse produite porte « ✂ sans … », y compris après rechargement de la page ;
- une fonctionnalité indisponible (RAG sans embedder) est marquée comme telle et ne peut pas être activée.

**US-LAB-4 — Prouver l'intérêt du RAG.** En tant qu'apprenante qui prépare une interro, je veux voir côte à côte la réponse du tuteur avec et sans mes leçons, pour savoir quand je peux lui faire confiance.
Critères d'acceptation :
- en mode Comparer, la même question produit deux réponses côte à côte, « avec » et « sans » ;
- sous chaque réponse : tokens du prompt par section et leçons injectées ; les leçons présentes d'un seul côté sont surlignées ;
- l'arbitrage note chaque réponse sur 10 avec une raison, en voyant les réponses dans un ordre aléatoire ;
- la comparaison n'ajoute rien à la conversation ni à la mémoire, même interrompue par Stop ;
- `benchmarks/rag_ab.py` chiffre l'écart sur des questions dont la réponse n'existe que dans les leçons, avec des questions de contrôle.

**US-LAB-5 — Régler le RAG avancé.** En tant que développeur de l'équipe, je veux comparer les étages R2 (BM25, rerank, filtre de langue) un par un, pour justifier leur coût en latence.
Critères d'acceptation :
- chaque étage R2 a son interrupteur, et la comparaison garde le RAG actif des deux côtés ;
- la carte montre les passages retenus de chaque côté et surligne ceux qui diffèrent ;
- `benchmarks/rag_ab.py --variants off,dense,hybrid,full` donne la justesse, les tokens et le temps de chaque variante.
