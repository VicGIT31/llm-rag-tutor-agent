# Duolingoy — l'interface web de Sensai (X1)

**Duolingoy** est l'interface web du tuteur : on parle à l'IA locale depuis le navigateur, on voit sa progression, ce que le modèle reçoit, et on pilote l'agent en gardant la main sur ses fichiers. Le code reste celui de Sensai (package `sensai`) : l'interface ne réimplémente rien, elle pilote **le même objet `Chat`** que le terminal (mémoire M1–M3, RAG R1/R2, agent A1/T2/T4/A3, journal EV3).

Fonctionnalité du catalogue : **X1 — Web UI** (« un frontend web découplé au-dessus du backend : saisie, affichage en streaming, usage des tokens et du contexte en direct, changement d'agent, configuration… La logique du backend reste inchangée »).

---

## 1. Lancer

```bash
.venv/bin/pip install -r requirements.txt     # ajoute fastapi et uvicorn
.venv/bin/python -m sensai.web                 # puis ouvrir http://127.0.0.1:8000
.venv/bin/python -m sensai.web --port 8080 --user camille --model qwen3-coder:30b
```

Les options sont celles de `python -m sensai` (`--model`, `--user`, `--config`, `--num-ctx`, `--db`, `--host`), plus `--port`. La documentation de l'API est servie sur `/api/docs`.

Fonctionne **hors ligne** : aucune police, bibliothèque ou image n'est chargée depuis Internet.

## 2. Ce qu'on voit

| Zone | Contenu | Features |
|---|---|---|
| Conversation | réponses en streaming, mises en forme (gras, listes, titres, tableaux, code) ; bouton **Stop** ; suggestions au démarrage ; messages déjà résumés grisés | B1, M2 |
| Mode **Agent** | chaque étape (résumé, action, durée, résultat de la sandbox, refus de permission) ; **carte de confirmation** avec action, fichiers, diff coloré, risques, compte à rebours | A1, T2, T4, A3 |
| Onglet **Progrès** | mots vus / maîtrisés / à réviser, boîtes de Leitner, erreurs récurrentes ; un mot faux s'oublie d'un clic | M3 |
| Onglet **Contexte** | répartition de la fenêtre au dernier message (rôle, mémoire, résumé, leçons, historique, libre, réserve) et tokens comptés par Ollama ; leçons utilisées avec cosinus, BM25 et note de rerank ; résumé de session et bouton **Résumer maintenant** ; métriques | M2, R1, R2, EV3 |
| Onglet **Labo** | interrupteurs des fonctionnalités (B1, M1, M2, M3, R1, R2) et démonstrations prêtes à lancer en mode Comparer ([labo.md](labo.md)) | toutes |
| Mode **Comparer** | même question avec et sans une fonctionnalité, deux colonnes, leçons de chaque côté, arbitrage du modèle | toutes |
| Onglet **Profil** | apprenants (créer, choisir, supprimer) ; historique (série, activité sur 14 jours, sessions à lire ou reprendre) ; profil de l'apprenant (ajout, suppression) | M1 |
| En-tête | changement de modèle sans redémarrer ; badge GPU / CPU (où Ollama a placé le modèle, mis à jour après chaque réponse) ; état (prêt, réfléchit, écrit, attend ta confirmation) | configuration |

Sur téléphone, les onglets passent dans un panneau latéral (bouton en haut à droite) ; `#progres`, `#contexte`, `#labo` ou `#profil` à la fin de l'adresse ouvrent directement un onglet.

## 3. Architecture

```
navigateur (static/index.html, app.js, style.css — sans framework ni compilation)
   │  JSON + Server-Sent Events, uniquement vers 127.0.0.1
   ▼
FastAPI (web/app.py) ── routes HTTP, validation des corps JSON, hôte vérifié
   │  engine.call(...) / engine.chat_events(...) / engine.agent_events(...)
   ▼
Engine (web/engine.py) ── UN thread de travail possède le Chat
   │        ▲ /api/stop et /api/confirm passent à côté de la file (voir plus bas)
   ▼
Chat (cli.py) ── le même que dans le terminal : mémoire, RAG, compression, agent
   ▼
Ollama (HTTP local)
```

- **Un seul thread de travail.** Une connexion SQLite ne peut servir que dans le thread qui l'a ouverte, et un apprenant n'a qu'une conversation à la fois. Toutes les opérations sur le `Chat` passent donc par une file et s'exécutent dans l'ordre, sur ce thread.
- **Deux opérations contournent la file**, parce que le thread de travail peut justement être bloqué en train de les attendre : **Stop** (un drapeau, vérifié à chaque morceau de réponse ou à chaque étape de l'agent) et la **réponse à une confirmation** (A3).
- **Streaming en Server-Sent Events.** Chaque événement est un objet JSON typé par `type` :

| Flux | Événements |
|---|---|
| `POST /api/chat` | `notice` (mémoire, compression…), `token` (morceau de réponse), `done` (interrompue ?, répartition du contexte, tokens Ollama, leçons utilisées), `memory` (vocabulaire et erreurs à jour), `error` |
| `POST /api/agent` | `step`, `confirm` (demande à afficher), `confirm_closed` (délai dépassé), `done`, `stopped`, `error` |

- **Le `Chat` a été découpé, pas réécrit.** Un tour de conversation est maintenant un générateur, `Chat.stream()`, que le terminal affiche et que le web diffuse ; les messages annexes passent par `Chat.notify` (stderr dans le terminal, événements dans le web). La tâche d'agent (`Chat.agent_turn`), les sessions (`new_session`, `load_session`) et la compression forcée (`compress_now`) sont partagées de la même façon. Aucune logique n'est dupliquée.

### Points d'API

| Méthode | Chemin | Rôle |
|---|---|---|
| GET | `/api/state` | session, profil, messages, fonctions actives, dernier contexte |
| POST | `/api/chat` | un message au tuteur (SSE) |
| POST | `/api/agent` | une tâche pour l'agent (SSE) |
| POST | `/api/stop` | arrêter la réponse ou la tâche en cours |
| POST | `/api/confirm` | `{id, approved}` : répondre à une demande de confirmation |
| GET | `/api/memory` | vocabulaire et erreurs |
| DELETE | `/api/vocab/{mot}` | oublier un mot |
| PUT / DELETE | `/api/profile`, `/api/profile/{clé}` | modifier le profil |
| GET / POST | `/api/sessions`, `/api/sessions/{id}/load` | lister, créer, reprendre une session |
| POST | `/api/compress` | résumer maintenant les anciens échanges |
| GET / PUT | `/api/models`, `/api/model` | modèles installés, changer de modèle |
| GET | `/api/metrics` | métriques du journal (EV3) |
| … | `/api/users`, `/api/history`, `/api/features`, `/api/compare`… | apprenants, historique, interrupteurs, comparaison avec / sans : voir [labo.md](labo.md#5-api-ajoutée) |

## 4. Sécurité

L'agent écrit des fichiers et exécute du code : une page web ne doit pas pouvoir le piloter à travers le navigateur de l'apprenant.

| Menace | Parade |
|---|---|
| Accès depuis une autre machine | le serveur n'écoute que sur `127.0.0.1` |
| **CSRF** : un autre site envoie une requête au serveur local | les corps sont du JSON uniquement (`application/json`) : un formulaire ou un `text/plain` est refusé (422), et un JSON cross-site exige un *preflight* CORS jamais accordé (aucun en-tête CORS n'est envoyé) |
| **DNS rebinding** : un domaine malveillant qui se résout en 127.0.0.1 | l'en-tête `Host` doit être `127.0.0.1` ou `localhost` (sinon 400) |
| Une réponse du modèle qui contient du HTML ou du script | tout texte est inséré comme texte ; le rendu Markdown échappe le HTML avant d'ajouter ses propres balises, sans liens ni attributs issus du texte |
| Valider une action dangereuse sans le vouloir | confirmation explicite, « Refuser » a le focus par défaut ; délai dépassé, page fermée ou Stop = refus ; une action refusée n'est jamais redemandée |

Limite : aucune authentification. N'importe quel programme qui tourne sur la machine sous le même utilisateur peut appeler l'API, comme il pourrait lancer la CLI.

## 5. Choix et alternatives écartées

| Choix | Alternative écartée | Pourquoi |
|---|---|---|
| FastAPI + uvicorn | serveur HTTP de la bibliothèque standard | validation des corps, documentation d'API automatique, refus des corps non JSON par défaut ; autorisé explicitement par le sujet |
| Server-Sent Events | WebSocket | le flux va dans un seul sens ; les rares messages du navigateur (Stop, confirmation) sont de simples POST ; SSE passe par `fetch`, sans bibliothèque |
| Un thread de travail propriétaire du `Chat` | ouvrir SQLite avec `check_same_thread=False` + des verrous | aucune modification des modules mémoire et RAG, aucun risque d'accès concurrent ; un apprenant n'a qu'une conversation à la fois |
| JavaScript sans framework ni compilation | React, Vue… | rien à installer ni à compiler, fonctionne hors ligne, lisible par toute l'équipe |
| Rendu Markdown maison, minimal | bibliothèque Markdown + nettoyeur HTML depuis un CDN | hors ligne ; un sous-ensemble suffit (gras, listes, titres, tableaux, code) ; la sécurité se vérifie en quelques lignes |
| Le même `Chat` que le terminal | une logique web séparée | une seule implémentation à tester ; le web hérite de toutes les corrections |

## 6. Tests et mesures

`tests/test_web.py` : **15 tests**, sans modèle (faux client Ollama scripté), dont 8 contre un vrai serveur uvicorn local interrogé en HTTP.

| Partie | Ce qui est vérifié |
|---|---|
| Moteur | ordre des événements d'un tour (morceaux, rapport, mémoire) ; Stop garde la partie affichée, en conversation comme en mémoire ; confirmation aller-retour depuis le « navigateur » (une seule réponse acceptée) ; Stop pendant une confirmation = refus et fin de tâche ; page fermée = refus, journalisé ; une panne devient un événement `error` et le thread survit ; changement de modèle vérifié, annulé en cas d'échec |
| API HTTP | page, fichiers statiques, documentation ; conversation en SSE puis état et mémoire ; profil, sessions, compression ; confirmation de l'agent par HTTP ; **requêtes cross-site refusées** (formulaire, `text/plain`, `Host` étranger, aucun en-tête CORS) ; validation des entrées, modèles, métriques |

Le rendu Markdown a été vérifié à part avec Node : `<script>`, `<img onerror>` et les tentatives d'injection d'attribut ressortent en texte.

**Essai réel** (5 octobre 2026, `qwen3-coder:30b`, CPU seul) :
- message au tuteur : premier morceau de réponse au bout de 43 s (rechargement du modèle compris), réponse complète en 64 s, appuyée sur la bonne leçon (compteurs, rerank 10/10) ; la mémoire s'est mise à jour juste après ;
- tâche d'agent « crée une fiche de révision » : lecture, confirmation depuis le navigateur, écriture, réponse finale, en 3 étapes et 26 s.

## 7. Limites

- **Latence** : sur CPU seul, un message prend 30 à 60 s. L'interface affiche un compteur pendant que le modèle réfléchit, mais seul un GPU règle le problème.
- **Une conversation à la fois** : un deuxième onglet attend que le premier ait fini.
- Le panneau « Contexte » décrit le **dernier message** envoyé ; une tâche d'agent ne le modifie pas.
- Pas de bouton pour interrompre l'agent au milieu d'un appel au modèle : Stop prend effet à la fin de l'étape en cours.
- La carte de l'agent n'est visible que dans l'onglet qui a lancé la tâche ; après rechargement, seuls la demande et la réponse finale restent dans la conversation (`/trace` dans le terminal pour le détail).

## 8. User stories

**US-X1-1.** En tant qu'apprenante peu à l'aise avec le terminal, je veux discuter avec mon tuteur dans le navigateur, pour réviser comme dans une application de langues.
Critères d'acceptation :
- après `python -m sensai.web`, la page `http://127.0.0.1:8000` affiche la réponse au fur et à mesure, mise en forme (gras, listes, tableaux) ;
- « Stop » coupe la génération ; la partie reçue reste dans la conversation et en mémoire ;
- après chaque réponse, l'onglet « Progrès » montre les nouveaux mots et erreurs sans recharger la page ; un mot faux s'oublie d'un clic ;
- recharger la page retrouve la conversation et le profil ; les échanges déjà résumés sont grisés ;
- la touche Entrée qui valide une conversion en kana ou kanji n'envoie pas le message.

**US-X1-2.** En tant qu'enseignant qui présente Sensai, je veux voir ce que le modèle reçoit et d'où viennent ses réponses, pour expliquer le système en direct.
Critères d'acceptation :
- l'onglet « Contexte » montre la part de chaque section dans la fenêtre au dernier message, et les tokens comptés par Ollama ;
- les extraits de leçons utilisés s'affichent avec leurs scores (cosinus, BM25, rerank) ;
- « Résumer maintenant » compresse les anciens échanges et les grise dans la conversation ;
- le modèle se change depuis l'en-tête sans redémarrer ; un modèle absent est refusé avec un message.

**US-X1-3.** En tant qu'apprenante, je veux confier une tâche à l'agent depuis le navigateur et garder la main sur mes fichiers, pour préparer mon matériel sans terminal.
Critères d'acceptation :
- le mode « Agent » affiche chaque étape au fur et à mesure (résumé, action, durée, résultat de la sandbox, refus de permission) ;
- avant une écriture, une suppression ou une exécution, une carte montre l'action, les fichiers, le diff et les risques ; rien n'est fait sans « Confirmer », et « Refuser » a le focus par défaut ;
- sans réponse avant `confirm_timeout`, en fermant la page ou avec « Stop », l'action est refusée ;
- une action refusée n'est pas redemandée.

**US-X1-4.** En tant qu'utilisatrice, je veux qu'un autre site ouvert dans mon navigateur ne puisse pas piloter l'agent, pour ne pas exposer ma machine.
Critères d'acceptation :
- le serveur n'écoute que sur `127.0.0.1` ;
- une requête dont l'en-tête `Host` n'est pas `127.0.0.1` ou `localhost` est refusée (400) ;
- une requête POST sans corps JSON (formulaire, `text/plain`) est refusée (422), et le serveur n'envoie aucun en-tête CORS ;
- une réponse du modèle contenant du HTML s'affiche en texte, sans être interprétée.
