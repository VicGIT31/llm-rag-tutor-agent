# Démo keynote — « Camille prépare son voyage à Kyoto »

Un seul scénario, joué en trois actes, qui montre toutes les features dans le même système plutôt qu'une suite de démos séparées.

> **Camille**, étudiante, part au Japon en avril. Elle révise le japonais le soir, sur son portable, **hors ligne**. Sensai est son tuteur : il se souvient d'elle, s'appuie sur ses cours, et prépare son matériel de révision sans jamais rien faire dans son dos.

| Acte | Ce qu'on montre | Features |
|---|---|---|
| 1. Un tuteur qui se souvient de moi | profil, réponses en streaming, mots et erreurs retenus tout seuls, reprise le lendemain | B1, M1, M3 |
| 2. Un tuteur qui s'appuie sur mes cours | recherche dans les leçons, filtres, budget de contexte partagé, compression | R1, R2, M2 |
| 3. Un agent qui prépare mes révisions, sous contrôle | quiz écrit, testé en sandbox, corrigé, enregistré après confirmation ; accès refusés ; traces et métriques | A1, T2, T4, A3, EV3 |

Le scénario a été répété deux fois dans la vraie CLI, le 5 octobre 2026 (`qwen3-coder:30b`, CPU seul), et ajusté d'après ces répétitions (réponse à l'exercice réellement proposé, message envoyé avant `/context`). Les temps mesurés sont en fin de document.

---

## Avant de monter sur scène

- [ ] **`python -m sensai.preflight --setup`** : vérifie Ollama et les modèles, le placement GPU/CPU (avec le réglage conseillé sans GPU), l'index, que chaque recherche de la démo ramène son document, la sandbox, crée les apprenants camille (japonais A1) et hugo (coréen A1), et préchauffe le modèle. Tout doit être ✓ ; un ✗ se règle avant de monter sur scène.
- [ ] `ollama serve` lancé, modèles présents : `ollama list` doit montrer `qwen3-coder:30b` et `bge-m3`. `keep_alive` vaut 30 min dans `config.json` : le modèle ne se décharge pas pendant qu'on parle entre deux actes (le défaut d'Ollama, 5 min, coûterait un rechargement de 15 à 45 s en pleine démo).
- [ ] Fermer les applications qui occupent la carte graphique (jeux, navigateur avec accélération) : chaque Go de VRAM pris ailleurs retire des couches du modèle au GPU.
- [ ] **Ollama avec le GPU** (sous Arch : paquet `ollama-vulkan` ou `ollama-cuda`, puis redémarrer Ollama). Au lancement, Sensai affiche où Ollama a placé le modèle (« Modèle … chargé : 45 % GPU / 55 % CPU » ; badge dans l'en-tête de Duolingoy) : « CPU seul » veut dire que le GPU n'est pas utilisé. Sur CPU seul, un message de chat prend 30 à 45 s et une tâche d'agent environ 2 minutes. Sans GPU, mettre `"rag_rerank": false` et `"rag_min_cosine": 0.50` dans `config.json` fait gagner une quinzaine de secondes par message (réglage mesuré dans [rag.md §7](rag.md#7-tests)).
- [ ] Partir d'une base propre : `rm -rf data workspace`, puis lancer avec `--user camille`.
- [ ] **Préchauffer le modèle** : une première question hors scène (le chargement prend environ 13 s).
- [ ] Terminal en grande police, fenêtre unique.
- [ ] **Enregistrements de secours** de chaque acte (`asciinema rec acteN.cast`) : un modèle local peut être lent ou se tromper en direct.

---

## Acte 1 — Un tuteur qui se souvient de moi (B1, M1, M3)

```text
.venv/bin/python -m sensai --user camille
> /profile set langue_cible japonais
> /profile set niveau A1
> Bonjour ! Je pars à Kyoto en avril. Comment on dit « chat » en japonais ?
> Watashi ga Camille desu. C'est correct pour me présenter ?
> /vocab
> /mistakes
> /quit
```

À montrer et à dire :
- la réponse **s'écrit au fur et à mesure** (B1). Le tuteur reprend l'exemple de la leçon des animaux (« Neko ga suki desu ») et, pour la présentation, la nuance exacte de la leçon sur wa et ga (« Watashi ga Camille desu » = « C'est moi, Camille ») : c'est déjà le RAG qui travaille ;
- la ligne `[mémoire : …]` : le modèle a décidé seul d'enregistrer le mot et l'erreur « confond wa et ga », et le code a **validé** ces opérations avant de les écrire (M3) ;
- `/vocab` : les mots en boîte 0 de Leitner, à réviser (la commande attend la fin de l'extraction en arrière-plan : quelques secondes) ; `/mistakes` : l'erreur, avec sa règle et l'exemple fautif.

Puis on relance, comme le lendemain :

```text
.venv/bin/python -m sensai --user camille
> /profile
> /sessions
> Fais-moi réviser mes mots avec un petit exercice.
> (répondre à l'exercice que le tuteur propose)
> /memory
```

- le profil et la session de la veille sont là, sans rien redire (M1) ;
- le tuteur **construit son exercice à partir de la mémoire** (« wa » et « ga », « neko ga suki ») ; `/memory` montre exactement ce qui lui a été injecté (M3) ;
- répondre à l'exercice réellement proposé : en répétition, une réplique préparée à l'avance ne correspondait pas à l'exercice, et le mot a été noté faux.

## Acte 2 — Un tuteur qui s'appuie sur mes cours (R1, R2, M2)

```text
> /compare rag Qu'est-ce qui tombe à l'interro de vendredi ?
> /rag search Qu'est-ce qui tombe à l'interro de vendredi ?
> Combien coûte le voyage à Kyoto ?
> Et l'acompte ?
> /rag search Quelle est la différence entre wa et ga ?
> /rag filter niveau=A2
> Comment on compte des billets de train ?
> /rag filter clear
> /compress
> /summary
> Et pour compter des stylos ?
> /context
```

À montrer et à dire :
- **commencer par la preuve d'utilité** : `/compare rag` pose la même question avec et sans les leçons. Avec : la date, la salle B204, le programme (leçons 3 et 4) et le fait que wa et ga ne tombent pas, tirés de `interro_vendredi.md`. Sans : « je ne connais pas les détails de ton interro », ou une invention. Rien de cela n'est devinable : c'est le cas où le RAG est indispensable, et non un confort. Le benchmark le chiffre ([labo.md §4](labo.md#4-mesurer-lintérêt-du-rag--benchmarksrag_abpy)). Dans le navigateur, la même chose se fait avec l'onglet Labo puis le mode Comparer ;
- sous la réponse, le badge **« ✓ N faits vérifiés dans tes leçons »** : la date, l'heure et la salle ont été retrouvées dans les extraits injectés, par le code, sans appel au modèle. S'il affiche « ⚠ absent de tes leçons », c'est le moment de dire que le tuteur s'est trompé et que le système l'a vu ;
- **« Et l'acompte ? »** : trois mots qui ne veulent rien dire seuls. Le RAG les cherche avec le message précédent (`/context` affiche « Recherche élargie au message précédent ») et la réponse donne 450 € avant le 15 décembre. Le voyage montre aussi que le cours ne se limite pas aux leçons : planning, corrigés, devoirs et voyage sont indexés (23 documents, plus que la fenêtre de contexte entière, d'où le besoin de chercher plutôt que de tout envoyer) ;
- `/rag search` : pour chaque morceau de leçon, cosinus, BM25 et note du rerank, **la raison** de chaque rejet (« cosinus 0.40 < 0.45 »), et « ↑ repêché par BM25 » pour un morceau sauvé par un mot rare (essayer `/rag search Que veut dire hatsuka ?`). Les leçons de **coréen** n'apparaissent même pas : le profil de Camille (japonais) filtre les documents (R2) ;
- avec `niveau=A2`, la réponse s'appuie sur la leçon des compteurs (« Kippu o nimai kudasai ») ;
- `/compress` puis `/summary` : les vieux échanges deviennent un résumé structuré (apprenant, vocabulaire, erreurs, en cours). Sur une session aussi courte, le résumé n'est pas plus petit que les messages : il faut le dire, et donner le vrai chiffre, celui du benchmark (4 faits sur 6 retrouvés avec la compression, 0 sur 6 sans) ;
- `/context` **après un nouveau message** (il décrit le dernier envoi) : **une seule fenêtre partagée** entre le rôle, le profil, la mémoire, le résumé, les documents et l'historique, chacun avec sa part du budget. C'est notre code qui décide ce qui entre, pas Ollama qui coupe au hasard (M2).

## Acte 3 — Un agent qui prépare mes révisions, sous contrôle (A1, T2, T4, A3, EV3)

```text
> /agent Écris un quiz Python à partir de docs/lecons/compteurs.md, teste-le, puis enregistre-le dans workspace/quiz.py
> /trace
> /agent Lis le fichier ../config.json
> /agent Supprime workspace/quiz.py          ← répondre n
> /metrics
```

À montrer et à dire (déroulé obtenu à la 2e répétition) :
- chaque étape : résumé public, action, résultat ; la demande de confirmation avec le **diff** avant l'écriture et les risques avant l'exécution (A1, A3) ;
- l'exécution en **sandbox** échoue parce que le quiz attend une saisie (`input()`, pas d'entrée standard en sandbox) : l'agent lit l'erreur, **corrige son code**, relance, obtient le code de sortie 0, et seulement alors annonce que le quiz fonctionne (T2) ;
- `../config.json` : refusé, hors des dossiers autorisés, **sans même demander** ; l'agent explique et s'arrête (T4) ;
- la suppression refusée par Camille : l'agent ne la redemande pas, le fichier est intact (A3) ;
- `/metrics` : latence, tokens, erreurs, outils, confirmations, exécutions en sandbox (EV3).

## Variante : tout dans le navigateur (Duolingoy)

`python -m sensai.web --user camille`, puis `http://127.0.0.1:8000` (voir [web.md](web.md)). Les trois actes se jouent de la même façon, sans taper de commande :

- **Acte 1** : mode « Tuteur ». L'onglet « Progrès » se met à jour seul après chaque réponse (mots, boîtes de Leitner, erreurs), et un mot faux s'oublie d'un clic ;
- **Acte 2** : l'onglet « Contexte » montre la jauge de la fenêtre, les tokens comptés par Ollama et les leçons utilisées avec leurs scores ; « Résumer maintenant » grise les messages résumés ;
- **Acte 3** : mode « Agent ». Les étapes et la carte de confirmation (diff, risques, Refuser / Confirmer, compte à rebours) s'affichent dans la conversation.

Restent dans le terminal : `/rag search`, `/rag filter` et `/trace`. Une bonne combinaison pour la keynote : le navigateur projeté pour l'histoire de Camille, et le terminal pour montrer `/rag search` et `/metrics`.

## Bilan honnête

Chiffres mesurés : [memoire_contexte.md §13](memoire_contexte.md#13-tests-et-mesures) (mémoire), [rag.md §7](rag.md#7-tests) (RAG), [agent_securite_observabilite.md §7](agent_securite_observabilite.md#7-tests-réalisés) (agent).

- **Ce qui marche** : les features se renforcent. La mémoire et le RAG partagent un budget unique ; le profil filtre les documents ; les erreurs retenues guident les exercices ; les confirmations et les permissions encadrent l'agent.
- **Ce que les mesures ont corrigé** : les répétitions avec le vrai modèle ont révélé des défauts que les tests avec un faux modèle ne pouvaient pas montrer (faits sacrifiés par la compression, vocabulaire bruité, fichiers non importables en sandbox, action refusée redemandée, agent qui sortait de sa tâche). Tous ont été corrigés et remesurés.
- **Les limites qui restent** :
  - la latence d'un modèle local sur CPU (30 à 45 s par message) ;
  - le tuteur peut se tromper même avec la bonne leçon dans le contexte (« 1 billet = ippai » au lieu de « ichimai ») et **la mémoire hérite de ses erreurs** ;
  - un même mot peut être enregistré en kana et en romaji ;
  - le rerank juge mal les messages très courts (« Bonjour ! ») ;
  - un corpus de 23 documents écrits pour la démo : assez grand pour dépasser la fenêtre de contexte, mais ce n'est pas encore le cours réel d'une classe.
- **Ce qu'on ferait autrement** : mesurer avec le vrai modèle dès le début, et vérifier les opérations mémoire contre les documents du RAG.

---

## Temps mesurés (répétitions du 5 octobre 2026, CPU seul)

Les commandes locales (`/profile`, `/vocab`, `/context`…) répondent immédiatement ; seuls les appels au modèle prennent du temps.

| Étape | Temps |
|---|---|
| 1er message de la séance (modèle à recharger) | 46 à 47 s |
| Message de chat suivant (rerank + réponse) | 29 à 45 s |
| `/vocab` juste après un message (attend l'extraction en arrière-plan) | 8 à 14 s |
| `/rag search` | 16 s |
| `/compress` | 11 s |
| Agent : quiz écrit, testé, corrigé et validé (6 étapes) | 119 s |
| Agent : lecture refusée par la permission (2 étapes) | 9 s |
| Agent : suppression refusée par Camille (2 étapes) | 7 s |

Sur toute la première répétition, `/metrics` donnait une latence moyenne des appels au modèle de 20 s (p95 : 58 s).

**Avec le GPU** (paquet `ollama-vulkan`, RTX 3060 Ti 8 Go ; Ollama y place 32 % du modèle, le reste sur le CPU) : à prompt égal (2 000 tokens) et réponse égale (150 tokens), 14,9 s au lieu de 27,1 s. La lecture du prompt passe de 103 à 190 tokens/s, l'écriture de 19 à 34 tokens/s, soit environ 1,8 fois plus vite.
