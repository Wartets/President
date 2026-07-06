# Le jeu du Président

Implémentation paramétrable et testable du jeu de cartes Président (variante « Trou du Cul »), pensée dès l'origine comme un environnement de recherche : moteur de règles pur, système d'événements complet pour l'observabilité et l'analyse, agents de complexité croissante (aléatoire à Monte-Carlo), et deux chaînes d'entraînement par renforcement, l'une mono-processus à politique linéaire, l'autre distribuée sur GPU via Ray et Redis. La spécification mathématique complète des règles est disponible dans `documentation/rules.md` ; ce dépôt en constitue l'implémentation de référence, vérifiée par construction (validation stricte de toute configuration à l'instanciation, revalidation systématique de toute action avant application).

## 1. Aperçu du jeu implémenté

Le Président est un jeu de défausse à hiérarchie inversée cyclique : $N \ge 3$ joueurs ($p_0, ..., p_{N-1}$) se débarrassent successivement de combinaisons de cartes de puissance croissante au sein d'un pli, jusqu'à ce que $N-1$ d'entre eux aient vidé leur main. L'ordre de sortie détermine à la fois des points de victoire (voir [`rules.md`, §4](documentation/rules.md#4-rôles-et-points-de-victoire-victorypoints)) et des rôles hiérarchiques (`ROLE_PRESIDENT`, `ROLE_VICE_PRESIDENT`, `ROLE_NEUTRAL`, `ROLE_VICE_SCUM`, `ROLE_SCUM`) qui conditionnent l'échange de cartes de la manche suivante (voir [`rules.md`, §5.2](documentation/rules.md#52-échange-exchange_phase)). Le projet implémente non seulement le socle historique du jeu, mais l'ensemble des variantes avancées couramment jouées (Révolution, Double Révolution, Suites, Jokers, clôture magique généralisée à un rang paramétrable, Saut de Tour, Interception, Putsch, Taxe Aveugle, pénalités de sortie étendues, voir [`rules.md`, §6](documentation/rules.md#6-règles-supplémentaires-et-événements-de-jeu)), chacune activable ou désactivable indépendamment via `core.config.GameConfig`. La spécification formelle intégrale, définitions, algorithme de manche, matrice de compatibilité croisée entre règles avancées, se trouve dans [`documentation/rules.md`](documentation/rules.md#table-des-matières).

## 2. Fonctionnalités

**Moteur de règles**

* Dimensionnement automatique du paquet en fonction du nombre de joueurs (nombre de paquets de 52/54 cartes calculé dynamiquement, ou fixé manuellement).
* Deux sémantiques de passe (`HARD_ONLY` / `ALLOW_SOFT`) et trois barèmes de points de victoire (`LEGACY_STEPPED`, `LINEAR`, `SYMMETRICAL`).
* Combinaisons uniformes et Suites, avec substitution par Joker en tant que carte libre (`Wildcard`), y compris comblement d'intervalle dans une Suite.
* Révolution et Double Révolution, avec verrouillage définitif de l'état de puissance pour la manche.
* Clôture magique généralisée à un rang paramétrable (pas seulement le 2 historique), avec remappage automatique du rang magique effectif sous Révolution.
* Saut de Tour à rang paramétrable, Interception hors-tour par carte jumelle, Putsch (droit d'annulation de l'échange), Taxe Aveugle (sélection aléatoire des cartes cédées), attribution stricte du reste de la distribution à un rôle ciblé, pénalités de sortie (rétrogradation immédiate ou reprise de cartes) avec sous-conditions étendues (sortie sur Joker, sortie déclenchant une révolution, carte suprême).
* Validation stricte de toute configuration à la construction (`GameConfig.__post_init__`) et de toute action avant application (`core.rules_engine.is_action_valid`) : aucune configuration incohérente ni aucune action illégale n'atteint jamais l'état de jeu.

**Agents**

* Dix profils prêts à l'emploi de complexité croissante, détaillés dans [`usage.md`, section 4.2](documentation/usage.md#42-sièges-disponibles---seats) : `random_bot`, `greedy_bot`, `aggressive_bot` (symétrique de `greedy_bot`, maximise la puissance résultante), `rule_based_bot`, `lookahead_bot` (anticipation locale de flexibilité de main), `adaptive_bot` (filtres adaptés à la position relative), `scoring_bot` (score composite pondéré puissance/coût/rareté), `probabilistic_bot` (score de risque par comptage probabiliste des cartes restantes), `mcts_bot` (simulation Monte-Carlo par rollouts), `human_agent` (console interactive). Tous les profils sont centralisés dans `registry.agent_registry`, qui expose la fabrique `build_agent` partagée par l'ensemble des points d'entrée. Voir la hiérarchie de complexité complète en [`architecture.md`, section 5.3](documentation/architecture.md#53-gradation-algorithmique-des-profils-fournis).
* Deux agents entraînables : `agents.rl_agent.RLAgent` (politique linéaire) et `agents.torch_rl_agent.TorchRLAgent` (réseau de neurones `torch`, inférence par lot accélérée GPU).
* Interface d'agent unique et minimale (`agents.interface.AbstractBaseAgent`), permettant d'ajouter un nouveau profil sans toucher au moteur.

**Observabilité et analyse**

* Système d'événements complet (structurels et transactionnels) publié en temps réel sur un bus publish/subscribe, avec empreinte déterministe de l'état à chaque événement.
* Journalisation en mémoire, export JSON Lines, export Parquet (table unique ou écriture incrémentale segmentée pour les campagnes massives), conversion en `DataFrame` Polars.
* Bibliothèque de métriques de recherche prêtes à l'emploi : indice de Gini de la puissance de main initiale, matrice de transition de rôles, taux d'efficacité du Putsch, taux de passe sous-optimal, facteur de dominance de pli, facteur de branchement et entropie de l'espace d'action, taux d'interception manquée, et une douzaine d'autres.
* Tableau de bord console temps réel (débit de simulation, utilisation CPU/mémoire/GPU, distribution glissante des points de victoire).

**Simulation et entraînement**

* Simulations de masse parallélisées sur plusieurs cœurs via Ray, avec export Parquet consolidé.
* Moteur vectorisé `numpy` traitant un lot de manches en lock-step, pour l'entraînement à très haut débit sur le sous-ensemble de règles compatible avec la vectorisation.
* Entraînement REINFORCE mono-processus à politique linéaire.
* Chaîne d'entraînement distribué complète (Rollout Workers Ray producteurs de transitions, tampon de rejeu Redis partagé, Trainer consommateur appliquant les mises à jour de gradient sous précision mixte GPU lorsque disponible), avec synchronisation asynchrone des poids via Redis.
* Évaluation comparative directe de profils hétérogènes, y compris de modèles entraînés chargés par poids, sur taux de victoire et VP cumulé (`research.evaluate_agent`).
* Recherche combinatoire sur le produit cartésien de profils, de nombres de joueurs, de présets de règles et de tailles de partie (`research.run_combinatory`).
* Génération non interactive de l'ensemble des graphiques d'analyse en une seule commande (`research.generate_graphs`).
* Pipeline automatique complet de bout en bout (entraînement, évaluation, graphiques, rapport de synthèse), sans intervention humaine, reprenant automatiquement où il s'était arrêté après toute interruption (`research.run_pipeline`).

## 3. Structure du dépôt

```text
President/
├── README.md              # Ce fichier
├── requirements.txt       # Dépendances Python (pip install -r requirements.txt)
├── LICENSE                # Licence MIT (2026 Wartets)
├── play_game.py           # Point d'entrée interactif en console
├── step_by_step_run.py    # Exécution pas à pas d'une partie pour débogage
├── naming.py              # Gestion des noms de fichiers et chemins
│
├── documentation/
│   ├── architecture.md    # Architecture du framework
│   ├── rules.md                # Règles du jeu, matrice de compatibilité, exemples
│   └── usage.md                # Guide d'utilisation opérationnel
│
├── core/                  # Modèles de données et règles pures, sans effet de bord
│   ├── config.py               # GameConfig, constantes de rôles et d'énumérations
│   ├── models.py               # Card, Hand, Action, ActionType, Suit, Rank
│   ├── math_utils.py           # Fonctions de valeur (f_points, f_power), VP, rôles
│   ├── rules_engine.py         # Construction du paquet, légalité, déclencheurs de règles avancées
│   └── action_masking.py       # Espace d'action discret fixe pour l'entraînement par lots
│
├── engine/                # Orchestration mutable d'une partie
│   ├── state.py                # GameState, TrickState (vue matérialisée mutable)
│   ├── round.py                # run_round, orchestration complète d'une manche
│   ├── game_runner.py          # Game, orchestration multi-manches ; interface tensorielle
│   └── event_bus.py            # EventBus, dispatcher publish/subscribe
│
├── events/                 # Types d'événements immuables
│   ├── base.py                 # Event, compute_state_hash
│   ├── structural.py           # Événements macroscopiques (manche, pli, sortie)
│   └── transactional.py        # Événements de décision individuelle
│
├── agents/                 # Implémentations de joueurs
│   ├── interface.py             # AbstractBaseAgent
│   ├── random_bot.py            # Joue au hasard parmi les actions légales
│   ├── greedy_bot.py            # Joue toujours la carte/combinaison la plus basse suffisante
│   ├── rule_based_bot.py        # Joue selon des règles prédéfinies
│   ├── lookahead_bot.py         # Ajoute une anticipation locale de flexibilité de main à RuleBasedBot
│   ├── adaptive_bot.py          # Ajuste ses filtres de préservation selon la position relative de la main
│   ├── scoring_bot.py           # Évalue chaque option par score composite pondéré (puissance/coût/rareté)
│   ├── mcts_bot.py              # Joue en utilisant Monte Carlo Tree Search
│   ├── human_agent.py           # Interface pour joueur humain
│   ├── rl_agent.py              # Politique linéaire entraînable
│   └── torch_rl_agent.py        # Politique neuronale entraînable, inférence par lot GPU
│
├── analytics/              # Journalisation et métriques de recherche
│   ├── event_logger.py          # EventLogger (JSONL, Parquet, DataFrame Polars)
│   ├── metrics_calc.py          # Bibliothèque de métriques macro/micro/comportementales
│   └── live_monitor.py          # Tableau de bord console temps réel
│
├── training/               # Entraînement par renforcement
│   ├── fast_path.py             # Moteur vectorisé numpy (FastPathEngine)
│   ├── train_rl.py              # Entraînement mono-processus (RLAgent)
│   ├── replay_buffer.py         # RedisReplayBuffer
│   ├── rollout_worker.py        # Acteur Ray producteur de transitions
│   ├── trainer.py               # Processus consommateur, mise à jour de politique torch
│   └── launch_distributed.py    # Point d'entrée conjoint Rollout Workers + Trainer
│
└── research/                # Outils de recherche et d'analyse
    ├── run_simulation.py        # Lanceur de simulations massives parallélisées Ray
    ├── run_combinatory.py       # Recherche combinatoire multi-configurations
    ├── evaluate_agent.py        # Évaluation comparative directe d'agents et de modèles entraînés
    ├── generate_graphs.py       # Génération non interactive de l'ensemble des graphiques d'analyse
    └── run_pipeline.py          # Pipeline automatique complet, reprenable après interruption
```

## 4. Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

Dépendances principales : `numpy`, `numba`, `torch`, `ray`, `redis`, `pyarrow`, `polars`, `rich`, `tqdm`, `psutil`, `pynvml` (optionnel, suivi GPU uniquement). Redis n'est requis que pour la chaîne d'entraînement distribué (section 12 de `documentation/usage.md`) ; toutes les autres fonctionnalités du projet s'exécutent sans aucune dépendance externe à installer ou démarrer séparément.

## 5. Démarrage rapide

Vérifier l'installation avec une partie entièrement automatisée :

```bash
python play_game.py --seats greedy,greedy,greedy,greedy --player-count 4 --rounds 1
```

Jouer soi-même contre trois bots :

```bash
python play_game.py --player-count 4 --seats human,greedy,greedy,greedy --rounds 5
```

Lancer mille parties simulées avec journalisation complète :

```bash
python -m research.run_simulation --games 1000 --player-count 4 --rounds-per-game 10 \
  --agent-profile rule_based_bot --workers 4 --output research_output.parquet --seed 0
```

Lister les profils de siège et les présets de règles disponibles, sans lancer de campagne :

```bash
python -m research.run_simulation --list-profiles
python -m research.run_simulation --list-presets
```

Exécuter l'intégralité du cycle de recherche (entraînement, évaluation, graphiques, rapport) en une seule commande, sans intervention
humaine, avec reprise automatique après interruption :

```bash
python -m research.run_pipeline --player-count 4
```

L'ensemble de ces commandes, leurs options et leur exploitation détaillée sont couvertes exhaustivement dans `documentation/usage.md`.

## 6. Documentation complète

| Document | Contenu |
| :--- | :--- |
| [`documentation/rules.md`](documentation/rules.md#table-des-matières) | Spécification mathématique et formelle intégrale des règles du jeu : définitions, algorithme de manche, matrice de compatibilité croisée entre règles avancées. Référence normative de tout comportement implémenté. Table des matières en tête de document. |
| [`documentation/usage.md`](documentation/usage.md#1-présentation-générale-du-projet) | Guide d'utilisation opérationnel : installation détaillée, toutes les commandes de `play_game.py` avec exemples pour chaque règle, exploitation des événements et des métriques, simulations de masse, entraînement (mono-processus et distribué), installation et administration de Redis, écriture d'un agent, génération non interactive des graphiques, pipeline automatique complet, dépannage. |
| [`documentation/architecture.md`](documentation/architecture.md#1-principes-de-conception-globaux) | Document de conception interne : principes de conception retenus, rationale de chaque choix technique, structure des données, machines à états, diagrammes de séquence, invariants garantis, limites connues et points d'extension. |
| [`documentation/expected_results.md`](documentation/expected_results.md) | Modèle scientifique des résultats statistiquement attendus pour chaque graphique produit par `research.generate_graphs`, destiné à servir de base à des tests de validation de l'implémentation plutôt qu'à une simple inspection visuelle. Chaque point renvoie à la section de [`rules.md`](documentation/rules.md#table-des-matières) qui définit la mécanique testée. |

Ce README couvre la présentation générale et l'entrée en matière ; il ne répète ni les commandes détaillées de `usage.md`, ni les justifications de conception d'`architecture.md`, ni le modèle statistique d'`expected_results.md`.

## 7. Panorama des composants principaux

Le moteur repose sur une séparation stricte entre calcul pur et orchestration mutable : `core` ne contient que des fonctions et structures immuables, sans aucun effet de bord ni dépendance à un état global ; `engine` est seul autorisé à muter un état de jeu (`GameState`), à travers l'unique fonction d'orchestration `engine.round.run_round`, appelée en boucle par `engine.game_runner.Game` pour enchaîner les manches d'une partie.

Chaque transition significative de l'état (distribution, échange, ouverture/clôture de pli, action jouée, déclenchement de règle avancée, sortie de joueur, fin de manche) est publiée sous forme d'événement immuable sur un bus publish/subscribe (`engine.event_bus.EventBus`). Cette observabilité est ce qui permet de collecter n'importe quelle partie jouée, qu'elle soit interactive (`play_game.py`), simulée en masse (`research.run_simulation`) ou exécutée dans le cadre d'un entraînement (`training.train_rl`, `training.rollout_worker`), dans un journal exploitable (`analytics.event_logger.EventLogger`) puis analysable par une bibliothèque de métriques dédiée (`analytics.metrics_calc`), sans jamais modifier le moteur lui-même.

Pour l'entraînement à très haut débit, un second moteur, entièrement indépendant, réimplémente en `numpy` un sous-ensemble volontairement restreint des règles sous forme vectorisée par lot (`training.fast_path.FastPathEngine`) : ce moteur ne publie aucun événement et ne partage aucune instance avec le moteur événementiel, les deux étant conçus pour des usages distincts (fidélité et observabilité complètes pour l'un, débit maximal sur un sous-ensemble de règles pour l'autre).

## 8. Compatibilité et prérequis

* Python 3.9 ou supérieur (le projet utilise `from __future__ import annotations` et des annotations de type génériques modernes).
* Un GPU CUDA est optionnel : il accélère l'entraînement neuronal distribué (précision mixte automatique) mais n'est requis nulle part ailleurs ; tout le projet fonctionne intégralement sur CPU.
* Redis (local via Docker/WSL2/Memurai, ou distant) est requis exclusivement pour la chaîne d'entraînement distribué décrite en section 12 de `documentation/usage.md`.
* Aucune dépendance à une base de données ou à un service réseau n'est nécessaire pour jouer en console, simuler en masse, ou entraîner un agent à politique linéaire mono-processus.

## 9. Reproductibilité

Toute partie est intégralement reproductible à graine (`random_seed`) et configuration (`GameConfig`) identiques : la distribution des mains dérive une graine locale par manche (`f"{random_seed}:{round_index}"`), et chaque agent fourni dérive sa propre graine locale par joueur (`f"{random_seed}:{player_id}[...]"`). Cette garantie ne s'étend pas automatiquement à un agent personnalisé qui utiliserait une source d'aléa non dérivée de cette convention, ni à l'ordre d'arrivée non déterministe des transitions dans un entraînement distribué asynchrone. Voir [`documentation/architecture.md`, section 10](documentation/architecture.md#10-invariants-garantis-et-contrats-inter-modules), pour la liste complète des invariants garantis par le moteur.

## 10. Étendre le projet

* **Ajouter un agent** : hériter de `agents.interface.AbstractBaseAgent` et implémenter ses quatre méthodes abstraites (`choose_action`, `choose_exchange_cards`, `ask_putsch`, `on_interception_opportunity`). Voir [la section 14 de `documentation/usage.md`](documentation/usage.md#14-écrire-son-propre-agent) pour un exemple complet, et l'enregistrer dans le registre centralisé `registry/agent_registry.py` (`HEURISTIC_AGENT_REGISTRY`), automatiquement repris par `play_game.py`, `step_by_step_run.py` et `research/run_simulation.py` sans modification supplémentaire. Nommer le module Python de l'agent exactement comme la clé de profil utilisée dans ce registre, afin que retrouver l'implémentation d'un profil reste immédiat. `agents/scoring_bot.py` (`ScoringBot`) et `agents/probabilistic_bot.py` (`ProbabilisticBot`) constituent deux exemples concrets d'agents à score continu (composite pondéré, puis estimation de risque par comptage probabiliste) plutôt qu'à filtres successifs, voir [`architecture.md`, section 5.3](documentation/architecture.md#53-gradation-algorithmique-des-profils-fournis).
* **Ajouter une règle avancée** : introduire le ou les champs correspondants dans `core.config.GameConfig` (avec validation dans `__post_init__` si une contrainte croisée s'impose), implémenter la logique de légalité/déclenchement dans `core.rules_engine`, puis l'intégrer dans `engine.round.run_round` au point du cycle de manche concerné. Documenter la règle dans `documentation/rules.md` selon le même formalisme que les règles existantes.
* **Ajouter une métrique d'analyse** : toute fonction pure prenant un `analytics.event_logger.EventLogger` (ou une structure dérivée de ses événements) en entrée et retournant une valeur ou une structure numérique s'intègre directement dans `analytics.metrics_calc`, suivant le même schéma que les métriques existantes.
* **Ajouter un graphique d'analyse** : toute fonction prenant un ou plusieurs `pandas.DataFrame` chargés depuis `data/`/`weights/` en entrée et écrivant une figure dans `figures/` s'intègre directement dans `research.generate_graphs`, en suivant le même schéma de garde d'absence de données que les fonctions existantes ; documenter le résultat statistique attendu dans `documentation/expected_results.md`.
* **Étendre le pipeline automatique** : toute étape supplémentaire (entraînement, simulation, analyse) s'intègre dans `research.run_pipeline` via un appel à `_run_step` avec un identifiant d'étape unique, garantissant automatiquement l'idempotence et la reprise après interruption.

## 11. Limites connues

Le moteur vectorisé (`training.fast_path.FastPathEngine`) ne modélise ni les couleurs, ni les Suites, ni l'Interception, ni le Putsch, ni la Taxe Aveugle, ni la pénalité de sortie étendue : il constitue un instrument d'entraînement à haut débit sur un sous-ensemble de règles, non un substitut fonctionnellement complet du moteur événementiel. Le tampon de rejeu Redis échantillonne uniformément, sans priorisation. La liste complète et détaillée des limites connues et des points d'extension identifiés se trouve dans [`documentation/architecture.md`, section 11](documentation/architecture.md#11-limites-connues-et-points-dextension).

## 12. État du projet

L'ensemble des règles décrites dans [`documentation/rules.md`](documentation/rules.md#table-des-matières) est implémenté et validé par construction (aucune configuration incohérente, aucune action illégale ne peut atteindre l'état de jeu). Les deux chaînes d'entraînement (linéaire mono-processus, neuronale distribuée) sont fonctionnelles de bout en bout, de la collecte de transitions jusqu'à la sauvegarde et la réutilisation des poids entraînés. La documentation opérationnelle ([`documentation/usage.md`](documentation/usage.md#1-présentation-générale-du-projet)) et la documentation de conception ([`documentation/architecture.md`](documentation/architecture.md#1-principes-de-conception-globaux)) couvrent l'intégralité des modules listés en section 3, y compris les huit profils d'agents détaillés en [`usage.md`, section 4.2](documentation/usage.md#42-sièges-disponibles---seats).

---
