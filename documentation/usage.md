# GUIDE D'UTILISATION

Ce document décrit intégralement le fonctionnement du projet : installation, prise en main en console, configuration fine des règles, exploitation des événements et des métriques, simulations de masse, entraînement d'agents (politique linéaire et politique neuronale distribuée), écriture d'un agent personnalisé, et dépannage. Chaque section renvoie explicitement aux modules du code source concernés afin qu'aucune commande présentée ici ne repose sur un comportement supposé plutôt qu'implémenté.

## Table des matières

1. Présentation générale du projet
2. Installation
3. Architecture du projet
4. Jouer une partie en console (`play_game.py`)
5. Configurer les règles de la partie (`GameConfig`)
6. Écrire un script Python autour du moteur (`engine.game_runner.Game`)
7. Le système d'événements (`events`, `engine.event_bus`)
8. Journalisation et exploitation des résultats (`analytics.event_logger`, `analytics.metrics_calc`)
9. Simulations de masse parallélisées (`research.run_simulation`)
10. Le moteur vectorisé (`training.fast_path.FastPathEngine`)
11. Entraînement d'un agent à politique linéaire (`training.train_rl`)
12. Entraînement distribué d'une politique neuronale (Ray + Redis)
13. Installer et exploiter Redis en détail
14. Écrire son propre agent
15. Suivi en temps réel d'une campagne (`analytics.live_monitor`)
16. Dépannage

## 1. Présentation générale du projet

Le projet implémente une version paramétrable du jeu de cartes Président, avec :

* un moteur de règles pur et sans effet de bord (`core.rules_engine`, `core.math_utils`, `core.models`, `core.config`) ;
* un moteur d'exécution de manche complète orienté événements (`engine.round.run_round`, `engine.game_runner.Game`) ;
* un moteur d'exécution vectorisé `numpy` pour l'entraînement à haut débit, sans passer par le système d'événements (`training.fast_path.FastPathEngine`) ;
* cinq profils d'agents prêts à l'emploi (`random_bot`, `greedy_bot`, `rule_based_bot`, `mcts_bot`, `human_agent`) plus deux agents entraînables sélectionnables au même titre via leur nom de module (`rl_agent` à politique linéaire, `torch_rl_agent` à politique neuronale), chaque nom de profil correspondant exactement au fichier `agents/<profil>.py` qui le définit ;
* un système d'événements complet (`events.structural`, `events.transactional`, `engine.event_bus.EventBus`) permettant de journaliser, rejouer et analyser n'importe quelle partie ;
* une couche d'analyse (`analytics.event_logger.EventLogger`, `analytics.metrics_calc`) transformant le flux d'événements en métriques exploitables (Gini, entropie, taux de passe sous-optimal, matrice de transition de rôles, etc.) ;
* une chaîne d'entraînement distribué complète (`training.rollout_worker.RolloutWorker`, `training.replay_buffer.RedisReplayBuffer`, `training.trainer.Trainer`, `training.launch_distributed`), reposant sur `ray` pour la parallélisation et `redis` comme tampon de rejeu partagé.

L'intégralité des règles avancées (Révolution, Double Révolution, Suites, Jokers, clôture magique généralisée, Saut de Tour, Interception, Putsch, Taxe Aveugle, pénalités de sortie étendues) est paramétrable via `core.config.GameConfig` et documentée précisément dans `rules.md`. Ce guide se concentre sur l'utilisation opérationnelle du code ; se reporter à `rules.md` pour la spécification mathématique complète de chaque règle.

## 2. Installation

### 2.1. Environnement Python

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

Le fichier `requirements.txt` installe : `numpy`, `polars`, `pyarrow`, `ray`, `torch`, `rich`, `psutil`, `pynvml`, `tqdm`, `numba`, `redis[hiredis]`.

### 2.2. Vérification rapide de l'installation

Pour vérifier que l'ensemble des dépendances est correctement installé et que le moteur fonctionne, exécuter une partie minimale entièrement automatisée :

```bash
python play_game.py --seats greedy,greedy,greedy,greedy --player-count 4 --rounds 1
```

Aucune saisie clavier n'est requise puisque les quatre sièges sont automatisés. La commande doit afficher le résumé de fin de manche et le total cumulé sans lever d'exception.

### 2.3. Remarque sur `pynvml`

`pynvml` est utilisé uniquement par `analytics.live_monitor.LiveMonitor` pour afficher l'utilisation GPU. Son absence, ou l'absence de pilote NVIDIA compatible, n'empêche jamais l'exécution : `LiveMonitor` détecte l'échec d'initialisation et affiche `indisponible` à la place de la métrique GPU (voir `_try_init_nvml` dans `analytics/live_monitor.py`).

### 2.4. Remarque sur `torch`

`torch` est nécessaire uniquement pour la voie d'entraînement neuronale (`agents.torch_rl_agent`, `training.trainer`, `training.rollout_worker`). La voie mono-processus à politique linéaire (`training.train_rl`) et le jeu en console (`play_game.py`) n'en dépendent pas directement, bien que le paquet reste listé comme dépendance globale du projet.

## 3. Architecture du projet

| Répertoire / module | Rôle |
| :--- | :--- |
| `core.config` | `GameConfig`, paramètres immuables de la partie et constantes de rôles/énumérations. |
| `core.models` | Types de données fondamentaux : `Card`, `Hand`, `Action`, `ActionType`, `Suit`, `Rank`. |
| `core.math_utils` | Fonctions pures de valeur (`f_points`, `f_power`), de calcul des points de victoire (`compute_vp`) et d'attribution de rôle (`role_for_rank`). |
| `core.rules_engine` | Cœur pur du moteur de règles : construction/distribution du paquet, validation des combinaisons, déclencheurs de règles avancées. |
| `core.action_masking` | Construction d'un espace d'action discret fixe et de masques booléens, utilisé pour l'entraînement par lots. |
| `engine.state` | `GameState` et `TrickState`, vue matérialisée mutable de l'état courant, consommée par les agents. |
| `engine.round` | `run_round`, orchestration complète d'une manche (distribution, Putsch, échange, plis, clôture). |
| `engine.game_runner` | `Game`, orchestration d'une partie complète sur plusieurs manches ; `vectorized_run` pour le moteur tensoriel. |
| `engine.event_bus` | `EventBus`, dispatcher publish/subscribe. |
| `events.base` | Classe abstraite `Event` et `compute_state_hash`. |
| `events.structural` | Événements macroscopiques : configuration, démarrage, distribution, ouverture/clôture de pli, sortie de joueur, fin de manche. |
| `events.transactional` | Événements de décision individuelle : échange, Putsch, action jouée, interception, déclenchement de règle. |
| `agents.*` | Implémentations concrètes de `agents.interface.AbstractBaseAgent`. |
| `analytics.event_logger` | `EventLogger`, abonné du bus qui accumule les événements et les exporte (JSONL, Parquet, DataFrame Polars). |
| `analytics.metrics_calc` | Bibliothèque de métriques pures consommant un `EventLogger`. |
| `analytics.live_monitor` | Tableau de bord console `rich` pour le suivi temps réel d'une campagne de simulation. |
| `training.fast_path` | `FastPathEngine`, moteur vectorisé `numpy` pour l'entraînement à très haut débit. |
| `training.train_rl` | Boucle d'entraînement REINFORCE mono-processus pour `agents.rl_agent.RLAgent`. |
| `training.replay_buffer` | `RedisReplayBuffer`, tampon de transitions partagé via Redis. |
| `training.rollout_worker` | `RolloutWorker`, acteur Ray produisant des transitions vers le tampon Redis. |
| `training.trainer` | `Trainer`, processus consommant le tampon Redis et mettant à jour une politique neuronale `torch`. |
| `training.launch_distributed` | Point d'entrée démarrant conjointement les Rollout Workers et le Trainer. |
| `research.run_simulation` | Lanceur de simulations massives parallélisées via Ray, avec export Parquet et résumé de métriques. |
| `research.run_combinatory` | Recherche combinatoire sur le produit cartésien de profils d'agents, de nombres de joueurs, de présets de règles et de tailles de partie, y compris des combinaisons hétérogènes de sièges. |
| `research.evaluate_agents` | Évaluation comparative directe de profils hétérogènes (dont des modèles entraînés) par taux de victoire et VP cumulé. |
| `play_game.py` | Point d'entrée interactif en console. |

## 4. Jouer une partie en console (`play_game.py`)

### 4.1. Commande de base

```bash
python play_game.py --player-count 4 --seats human,greedy,greedy,greedy --rounds 5
```

### 4.2. Sièges disponibles (`--seats`)

`--seats` attend une liste de profils séparés par des virgules, de taille strictement égale à `--player-count` (`_build_seat_profiles` lève `ValueError` sinon). Profils disponibles, définis dans `_AGENT_REGISTRY` :

| Profil | Classe | Comportement |
| :--- | :--- | :--- |
| `human_agent` | `agents.human_agent.HumanAgent` | Sollicite chaque décision par saisie clavier, affiche la main, la puissance à dépasser et l'état de la Révolution. |
| `random_bot` | `agents.random_bot.RandomBot` | Choisit uniformément une option légale parmi toutes celles disponibles (uniformes et suites). |
| `greedy_bot` | `agents.greedy_bot.GreedyBot` | Joue systématiquement la combinaison légale de puissance résultante minimale. |
| `rule_based_bot` | `agents.rule_based_bot.RuleBasedBot` | Applique des heuristiques déterministes : évite de déclencher la pénalité de sortie étendue quand une alternative existe, préserve les combinaisons de taille ≥ 4 tant que la main compte plus de 4 cartes, puis choisit la puissance minimale suffisante. |
| `mcts_bot` | `agents.mcts_bot.MCTSBot` | Évalue chaque option candidate par 24 rollouts simulés (paramétrable via le constructeur, `rollout_count`) joués par des `GreedyBot` de référence, et retient l'option au meilleur taux de victoire (rang de sortie ≤ 1) simulé. Coût nettement supérieur aux autres profils : à réserver aux parties courtes ou à l'analyse ponctuelle. |
| `rl_agent` | `agents.rl_agent.RLAgent` | Politique linéaire entraînable. Sans poids fournis via `--weights`, joue avec des poids nuls. |
| `torch_rl_agent` | `agents.torch_rl_agent.TorchRLAgent` | Politique neuronale entraînable (`PolicyNet`). Sans poids fournis via `--weights`, joue avec un réseau initialisé aléatoirement. |

Chaque clé de profil correspond exactement au nom du module Python définissant la classe d'agent (`agents/<clé>.py`), afin qu'ajouter un agent au registre et retrouver son implémentation soit immédiat.

Si `--seats` est omis, le siège 0 reçoit `human_agent` et les suivants `greedy_bot` (comportement par défaut de `_build_seat_profiles`).

Pour charger des poids entraînés sur un ou plusieurs sièges `rl_agent`/`torch_rl_agent`, utiliser `--weights`, liste de chemins séparés par des virgules alignée positionnellement sur `--seats` (une entrée vide pour tout siège n'ayant pas besoin de poids) :

```bash
python play_game.py --player-count 4 --seats torch_rl_agent,rule_based_bot,rule_based_bot,greedy_bot \
  --weights weights/torch_rl_weights_player4_learnRate0p001_rounds10000_20260101.pt,,, --rounds 10
```

Exemple : quatre bots heuristiques s'affrontant sur 20 manches, sans aucune interaction humaine :

```bash
python play_game.py --player-count 4 --seats rule_based,rule_based,greedy,random --rounds 20 --seed 42
```

Exemple : un agent MCTS contre trois agents heuristiques, en 6 joueurs :

```bash
python play_game.py --player-count 6 --seats mcts,rule_based,rule_based,greedy,greedy,random --rounds 10
```

### 4.3. Déroulement d'un tour humain

Pour chaque sollicitation (`HumanAgent.choose_action`), la console affiche :

1. la main courante, formatée par `_format_cards` (exemple : `3S 7H 10D KC Joker#0`) ;
2. la puissance à dépasser si le pli n'est pas vide (`Puissance à dépasser : N`) ;
3. un avertissement si la Révolution est active (`Révolution active.`) ;
4. la liste numérotée des options légales, `0` désignant toujours le passe, avec la mention `(Joker déclaré à N)` pour toute combinaison contenant un Joker.

Saisir le numéro de l'option souhaitée. Toute saisie vide, non numérique, ou hors bornes est interprétée comme un passe conforme à `pass_type` (`ACTION_SOFT_PASS` si `ALLOW_SOFT`, `ACTION_HARD_PASS` sinon).

Pour un échange de cartes (`HumanAgent.choose_exchange_cards`), la main est réaffichée avec un index par carte ; saisir les index des cartes à céder séparés par des espaces. Une sélection incomplète est automatiquement complétée par les cartes de plus faible puissance restantes.

Pour le Putsch (`ask_putsch`) et l'interception (`on_interception_opportunity`), répondre `o` (ou toute chaîne commençant par `o`/`O`) pour accepter, toute autre saisie étant interprétée comme un refus.

### 4.4. Sortie affichée en fin de manche et de partie

`_print_round_summary` affiche, pour chaque manche, le point de victoire et le rôle attribué pour la manche suivante de chaque joueur. En fin de script, le total cumulé sur l'ensemble des manches jouées est affiché pour chaque identifiant de joueur (`game.cumulative_vp`).

## 5. Configurer les règles de la partie (`GameConfig`)

### 5.1. Table complète des options CLI

Toutes les options de `GameConfig` sont exposées en ligne de commande sur `play_game.py`, et peuvent également être passées directement au constructeur `GameConfig(...)` dans un script Python.

| Option CLI | Champ `GameConfig` | Effet |
| :--- | :--- | :--- |
| `--seed N` | `random_seed` | Graine de reproductibilité, réutilisée pour toute la distribution et tout tirage aléatoire de la partie. |
| `--player-count N` | `player_count` | Nombre de joueurs, minimum 3 (`__post_init__` lève `ValueError` sinon). |
| `--first-trick-opener-id N` | `first_trick_opener_id` | Joueur ouvrant le tout premier pli de la partie (manche d'index 0 uniquement ; les manches suivantes sont ouvertes par le `ROLE_SCUM` de la manche précédente, cf. `rules.md` §5.3.1). |
| `--disable-deck-scaling-auto` | `deck_scaling_auto=False` | Désactive le calcul automatique du nombre de paquets $N_D$. Doit être combiné avec `--forced-deck-count`. |
| `--forced-deck-count N` | `forced_deck_count` | Nombre de paquets fixé manuellement, effectif uniquement si `deck_scaling_auto` est faux. |
| `--pass-type {HARD_ONLY,ALLOW_SOFT}` | `pass_type` | Sémantique de passe pour toute la partie. `HARD_ONLY` exclut définitivement un joueur passé du pli en cours ; `ALLOW_SOFT` lui permet de resurenchérir si le tour lui revient dans le même pli. |
| `--vp-distribution {LEGACY_STEPPED,LINEAR,SYMMETRICAL}` | `vp_distribution_type` | Barème de points de victoire (cf. `rules.md` §4). `SYMMETRICAL` est recommandé pour l'entraînement (centré sur zéro, sans rupture de pente). |
| `--disable-jokers` | `use_jokers=False` | Retire les Jokers du paquet. |
| `--disable-magic-two` | `magic_two=False` | Désactive la clôture magique historique sur le 2. |
| `--disable-magic-two-single-clears-all` | `magic_two_single_clears_all=False` | Un 2 seul ne clôture plus un pli de taille supérieure à 1. |
| `--magic-card-enabled` | `magic_card_enabled=True` | Généralise la clôture magique à un rang paramétrable via `--magic-card-rank`. |
| `--magic-card-rank R` | `magic_card_rank` | Rang magique (`3` à `2`, hors `JOKER`). |
| `--disable-magic-single-clears-all` | `magic_single_clears_all=False` | Variante à un rang paramétrable de la règle de clôture par carte unique. |
| `--skip-on-equal` | `skip_on_equal=True` | Force une réponse de puissance strictement égale après une égalité déclarée (cf. `rules.md` §6.3). |
| `--disable-revolution` | `revolution_enabled=False` | Désactive la Révolution. |
| `--double-revolution-enabled` | `double_revolution_enabled=True` | Active la Double Révolution. Nécessite un nombre de paquets effectif ≥ 2, sous peine de `ValueError` au démarrage. |
| `--straights-enabled` | `straights_enabled=True` | Active les combinaisons de type suite. |
| `--skip-turn-enabled` | `skip_turn_enabled=True` | Active le Saut de Tour. |
| `--skip-turn-rank R` | `skip_turn_rank` | Rang déclenchant le Saut de Tour (par défaut `8`). |
| `--interception-enabled` | `interception_enabled=True` | Active l'interception hors-tour. Nécessite également un nombre de paquets effectif ≥ 2. |
| `--disable-interception-closes-trick` | `interception_closes_trick=False` | Une interception réussie reprend simplement l'ordre de jeu à partir de l'intercepteur au lieu de clôturer immédiatement le pli. |
| `--putsch-enabled` | `putsch_enabled=True` | Autorise le rôle `ROLE_SCUM` à invoquer le Putsch, annulant la phase d'échange. |
| `--blind-tax-enabled` | `blind_tax_enabled=True` | Remplace la sélection déterministe des deux cartes cédées par le `ROLE_SCUM` par un tirage aléatoire uniforme. |
| `--strict-remainder-allocation` | `strict_remainder_allocation=True` | Attribue le reste de la distribution modulaire à un rôle ciblé plutôt que de le répartir modulo $N$. |
| `--strict-remainder-role ROLE` | `strict_remainder_role` | Rôle ciblé, parmi `ROLE_PRESIDENT`, `ROLE_VICE_PRESIDENT`, `ROLE_NEUTRAL`, `ROLE_VICE_SCUM`, `ROLE_SCUM`. |
| `--finish-penalty-enabled` | `finish_penalty_enabled=True` | Active la pénalité de sortie. |
| `--finish-penalty-type {PENALTY_INSTANT_SCUM,PENALTY_DRAW_CARDS}` | `finish_penalty_type` | Nature de la pénalité : rétrogradation immédiate au rang `ROLE_SCUM`, ou reprise d'une partie des cartes jouées en main. |
| `--finish-penalty-draw-count N` | `finish_penalty_draw_count` | Nombre de cartes reprises en main si `PENALTY_DRAW_CARDS`, entier strictement positif. |
| `--finish-penalty-extended` | `finish_penalty_extended=True` | Active les sous-conditions étendues de la pénalité (carte suprême, sortie sur Joker, sortie déclenchant une révolution). |
| `--no-finish-on-joker` | `no_finish_on_joker=True` | Sous-condition de la pénalité étendue : sortir sur un Joker déclenche la pénalité. |
| `--no-finish-on-revolution` | `no_finish_on_revolution=True` | Sous-condition de la pénalité étendue : sortir en déclenchant une révolution déclenche la pénalité. |

### 5.2. Contraintes de validation (`GameConfig.__post_init__`)

La construction d'un `GameConfig` invalide lève `ValueError` immédiatement, avant tout démarrage de partie. Les contraintes vérifiées :

* `player_count >= 3` ;
* `double_revolution_enabled` implique `revolution_enabled` ;
* `double_revolution_enabled` ou `interception_enabled` impliquent un nombre de paquets effectif ≥ 2, calculé soit par `deck_scaling_auto` (automatique dès que `player_count >= 5`), soit par `forced_deck_count` explicite ;
* `pass_type` appartient à `{HARD_ONLY, ALLOW_SOFT}` ;
* `vp_distribution_type` appartient à `{LEGACY_STEPPED, LINEAR, SYMMETRICAL}` ;
* `finish_penalty_type` appartient à `{PENALTY_INSTANT_SCUM, PENALTY_DRAW_CARDS}` ;
* `strict_remainder_role` est un rôle valide ;
* `magic_card_rank` et `skip_turn_rank` appartiennent aux rangs faciaux non-Joker ;
* `finish_penalty_draw_count >= 1`.

### 5.3. Exemples de configurations complètes

Configuration "règles historiques strictes", sans Jokers ni suites, barème historique :

```bash
python play_game.py --player-count 4 --seats greedy,greedy,greedy,greedy \
  --disable-jokers --vp-distribution LEGACY_STEPPED --rounds 10
```

Configuration "table complète des règles avancées", nécessitant au moins 5 joueurs pour disposer de 2 paquets automatiquement :

```bash
python play_game.py --player-count 5 --seats rule_based,rule_based,rule_based,greedy,greedy \
  --straights-enabled --skip-turn-enabled --interception-enabled --double-revolution-enabled \
  --putsch-enabled --blind-tax-enabled --finish-penalty-enabled --finish-penalty-extended \
  --no-finish-on-joker --no-finish-on-revolution --rounds 15 --seed 7
```

Configuration avec clôture magique généralisée sur le 10 plutôt que sur le 2, et passe souple :

```bash
python play_game.py --player-count 4 --seats greedy,greedy,greedy,greedy \
  --magic-card-enabled --magic-card-rank 10 --disable-magic-two --pass-type ALLOW_SOFT --rounds 10
```

Configuration avec attribution stricte du reste de la distribution au `ROLE_SCUM` :

```bash
python play_game.py --player-count 5 --seats greedy,greedy,greedy,greedy,greedy \
  --strict-remainder-allocation --strict-remainder-role ROLE_SCUM --rounds 10
```

## 6. Écrire un script Python autour du moteur (`engine.game_runner.Game`)

Au-delà de `play_game.py`, le moteur s'utilise directement en Python. Exemple minimal :

```python
from core.config import GameConfig
from agents.greedy_bot import GreedyBot
from agents.rule_based_bot import RuleBasedBot
from engine.event_bus import EventBus
from engine.game_runner import Game

config = GameConfig(player_count=4, random_seed=123, straights_enabled=True)

agents = {
    0: RuleBasedBot(0, config),
    1: RuleBasedBot(1, config),
    2: GreedyBot(2, config),
    3: GreedyBot(3, config),
}

game = Game(config, agents, event_bus=EventBus(), game_id="script-exemple")

for _ in range(50):
    vp_by_player = game.play_round()
    print(vp_by_player, game.roles)

print("Total cumulé :", game.cumulative_vp)
```

`Game.play_rounds(count)` exécute directement `count` manches et retourne la liste des dictionnaires de points de victoire, une entrée par manche. `Game.play_rounds_vectorized(count)` retourne à la place un tableau `numpy.ndarray` de forme `(count, player_count)`, plus pratique pour l'agrégation statistique immédiate :

```python
vp_tensor = game.play_rounds_vectorized(200)
print("VP moyen par joueur :", vp_tensor.mean(axis=0))
print("Écart type par joueur :", vp_tensor.std(axis=0))
```

## 7. Le système d'événements (`events`, `engine.event_bus`)

### 7.1. Principe

`engine.round.run_round` publie systématiquement, à chaque étape de la manche, un événement typé sur l'`EventBus` fourni. Le bus ne transforme jamais les événements ; il les transmet dans l'ordre d'émission à chaque abonné enregistré via `subscribe`.

Événements structurels (`events.structural`) : `EventGameConfig`, `EventGameStart`, `EventRoundStart` (contient `initial_hands`, la matrice complète des mains initiales, utile pour reconstituer entièrement une manche), `EventTrickStart`, `EventTrickClosed`, `EventHandEmpty`, `EventPlayerFinished`, `EventRoundEnd`.

Événements transactionnels (`events.transactional`) : `EventAskPutsch`, `EventPutschInvoked`, `EventExchangeIntent`, `EventExchange`, `EventActionRequest` (porte `legal_action_count`, le nombre d'options légales disponibles au moment de la sollicitation, base du calcul du facteur de branchement et de l'entropie d'action), `EventActionPlayed` (porte `was_suboptimal`, indicateur clé pour l'analyse comportementale), `EventInterceptionBroadcast`, `EventInterceptionResolved`, `EventRuleTriggered` (couvre `REVOLUTION`, `DOUBLE_REVOLUTION`, `MAGIC_CLOSURE`, `SKIP_TURN`, `INTERCEPTION`, `EQUAL_FORCED`, `FINISH_PENALTY`).

Chaque événement hérite de `events.base.Event` et porte systématiquement `timestamp` (horodatage logique croissant), `game_id`, `round_id`, et `state_hash` (empreinte SHA-256 déterministe de l'état au moment de l'émission, calculée par `compute_state_hash`).

### 7.2. S'abonner directement au bus sans passer par `EventLogger`

Il est possible de traiter les événements à la volée, sans accumulation en mémoire, par exemple pour un affichage console personnalisé ou une alerte en temps réel :

```python
from engine.event_bus import EventBus
from events.transactional import EventRuleTriggered

bus = EventBus()

def on_event(event):
    if isinstance(event, EventRuleTriggered) and event.rule_name in ("REVOLUTION", "DOUBLE_REVOLUTION"):
        print(f"Manche {event.round_id} : bascule de révolution déclenchée par le joueur {event.triggering_player_id}")

bus.subscribe(on_event)
# transmettre bus à Game(config, agents, event_bus=bus, ...)
```

Plusieurs abonnés peuvent être enregistrés simultanément sur le même bus (par exemple un `EventLogger` pour l'archivage et une fonction d'alerte console en parallèle) ; ils sont tous appelés, dans l'ordre d'enregistrement, pour chaque événement publié.

## 8. Journalisation et exploitation des résultats

### 8.1. Collecter les événements avec `EventLogger`

```python
from analytics.event_logger import EventLogger
from engine.event_bus import EventBus
from engine.game_runner import Game
from core.config import GameConfig
from agents.rule_based_bot import RuleBasedBot

config = GameConfig(player_count=4, random_seed=0)
agents = {pid: RuleBasedBot(pid, config) for pid in range(4)}

logger = EventLogger()
bus = EventBus()
bus.subscribe(logger)

game = Game(config, agents, event_bus=bus, game_id="ma-partie")
game.play_rounds(10)
```

`EventLogger` étant lui-même appelable (`__call__`), il s'utilise directement comme fonction abonnée au bus.

### 8.2. Formats d'export

* `logger.to_records()` : liste de dictionnaires plats, un par événement, chaque champ étant rendu sérialisable par `_serialize_value` (tuples convertis en listes, énumérations réduites à leur valeur, sous-dataclasses aplaties récursivement).
* `logger.to_jsonl(path)` : un objet JSON par ligne, format adapté au traitement en flux (`grep`, `jq`, ingestion incrémentale).
* `logger.to_dataframe()` / `logger.to_polars_dataframe()` : `polars.DataFrame` construit à partir de `to_records()`, pour l'analyse interactive.
* `logger.to_parquet(path)` : export Parquet complet en une seule table, toutes les colonnes d'événements confondues (schéma large, une colonne par champ possible sur l'ensemble des types d'événements).
* `logger.flush_to_parquet(path)` / `logger.close()` : export Parquet **segmenté**, à schéma stable (`event_type`, `timestamp`, `game_id`, `round_id`, `state_hash`, `payload`, ce dernier portant les champs spécifiques sérialisés en JSON). Ce mode est celui utilisé par `research.run_simulation`, car il permet de vider le tampon en mémoire par lots successifs (`parquet_buffer_size`) sans jamais charger l'intégralité d'une campagne massive en mémoire simultanément.

Exemple d'export et de relecture avec Polars :

```python
logger.to_parquet("ma_partie.parquet")

import polars as pl
df = pl.read_parquet("ma_partie.parquet")
print(df.filter(pl.col("event_type") == "EventActionPlayed").group_by("player_id").count())
```

Exemple de relecture du format segmenté (schéma `payload` en JSON) :

```python
import polars as pl
import json

df = pl.read_parquet("research_output.parquet")
actions = df.filter(pl.col("event_type") == "EventActionPlayed")
payloads = [json.loads(p) for p in actions["payload"].to_list()]
suboptimal_count = sum(1 for p in payloads if p.get("was_suboptimal"))
print(f"{suboptimal_count} actions sous-optimales sur {len(payloads)}")
```

### 8.3. Filtrer le journal en mémoire

`logger.events_of_type(EventCls)` retourne la sous-liste des événements du type demandé, dans l'ordre chronologique de réception. Toutes les fonctions de `analytics.metrics_calc` reposent sur ce filtrage.

```python
from events.structural import EventPlayerFinished

for event in logger.events_of_type(EventPlayerFinished):
    print(f"Manche {event.round_id} : joueur {event.player_id} sorti au rang {event.rank}, VP = {event.vp_earned:+.2f}")
```

### 8.4. Bibliothèque de métriques (`analytics.metrics_calc`)

Toutes les fonctions suivantes sont pures : elles prennent un `EventLogger` (ou des structures dérivées de ses événements) en entrée et ne provoquent aucun effet de bord.

**Métriques macro (échelle de la partie ou des manches) :**

* `gini_initial_hand_power(hand_powers_by_player)` : indice de Gini (domaine `[0, 1]`) de la puissance de main initiale entre joueurs, à construire soi-même à partir de `EventRoundStart.initial_hands` et de `core.math_utils.f_std` :

```python
from core.math_utils import f_std
from events.structural import EventRoundStart
from analytics.metrics_calc import gini_initial_hand_power

round_start = logger.events_of_type(EventRoundStart)[0]
hand_powers = {
    pid: sum(f_std(c) for c in cards if c.rank.value != "JOKER")
    for pid, cards in round_start.initial_hands.items()
}
print("Gini puissance de main initiale :", gini_initial_hand_power(hand_powers))
```

* `role_transition_matrix(role_sequence)` : matrice de Markov des transitions de rôle d'une manche à la suivante. `role_sequence` est la liste, dans l'ordre chronologique, des `roles_by_player` successifs (récupérables via `game.roles` après chaque `play_round`, ou reconstruits depuis `EventRoundEnd.roles_by_player`).
* `social_mobility_index(role_sequence, role_distance)` : distance moyenne parcourue entre rôles d'une manche à l'autre. `role_distance` doit associer chaque rôle à un rang ordinal, par exemple `{"ROLE_PRESIDENT": 0, "ROLE_VICE_PRESIDENT": 1, "ROLE_NEUTRAL": 2, "ROLE_VICE_SCUM": 3, "ROLE_SCUM": 4}`.
* `putsch_efficiency_rate(logger)` : taux de victoire (rang de sortie nul) du joueur sollicité pour le Putsch, séparément selon qu'il a été invoqué ou non, sous forme de dictionnaire `{"invoked": ..., "not_invoked": ...}`.
* `tax_weight_vp_correlation(logger)` : corrélation de Pearson entre le poids en points (`f_points`) des cartes transférées lors d'un échange et le VP obtenu par le destinataire à l'issue de la même manche.
* `opening_position_rank_correlation(logger)` : corrélation entre l'identifiant du joueur ouvrant le premier pli d'une manche et son rang de sortie final.
* `e_rev_volatility(logger)` : nombre moyen de bascules de Révolution (`REVOLUTION` + `DOUBLE_REVOLUTION`) par manche.
* `skip_turn_coverage(logger, player_count)` : proportion moyenne de joueurs effectivement sautés par les déclenchements de Saut de Tour.

**Métriques micro (échelle de la main et de la carte) :**

* `card_ttl(logger, player_id, min_rank_index)` : nombre moyen d'actions `ACTION_PLAY` écoulées avant que `player_id` ne joue une carte de rang facial supérieur ou égal à `min_rank_index` (utiliser `core.math_utils.rank_facial_index("2")` par exemple pour cibler les cartes de rang 2 ou supérieur).
* `joker_substitution_efficiency(logger, player_id)` : écart moyen entre la puissance maximale intrinsèque du Joker (16) et sa puissance substituée effective ; une valeur nulle signale un usage systématiquement optimal.
* `joker_magic_mimic_rate(logger, config)` : taux de Jokers joués en imitation de la carte magique effective.
* `capture_efficiency_ratio(logger, player_id)` : ratio entre les points capturés en remportant des plis et les points engagés en jouant des cartes.

**Métriques comportementales :**

* `sub_optimal_pass_rate(logger, player_id)` et son complément `action_validity_rate(logger, player_id)`.
* `aggressiveness_index_opening(logger, player_id, average_hand_power_by_round)` : nécessite de fournir en entrée la puissance moyenne de main du joueur au moment de chaque ouverture, à calculer à partir de `initial_hands` et de la progression des mains.
* `trick_dominance_factor(logger, player_id)` : proportion de plis remportés parmi ceux auxquels le joueur a participé.
* `revolution_counter_attack_rate(logger, player_id)` : probabilité de déclencher une Révolution en réponse directe à une Révolution adverse.
* `missed_interception_rate(logger)` : proportion d'opportunités d'interception mathématiquement possibles mais non saisies (reconstruction complète des mains à partir des événements de distribution, d'échange et de pose).

**Complexité et théorie de l'information :**

* `branching_factor_average(logger, player_id=None)` : facteur de branchement moyen (`legal_action_count` moyen), globalement ou pour un joueur donné.
* `action_space_entropy(logger, player_id=None)` : entropie de Shannon (en bits) de la distribution du nombre d'options légales.

**Métriques additionnelles :**

* `combination_size_distribution(logger, player_id=None)` : histogramme des tailles de combinaisons effectivement jouées.
* `trick_length_average(logger)` : nombre moyen d'actions par pli.

Exemple complet de tableau de bord de fin de campagne :

```python
from analytics.metrics_calc import (
    action_space_entropy, branching_factor_average, e_rev_volatility,
    sub_optimal_pass_rate, trick_length_average,
)

for pid in range(config.player_count):
    print(f"Joueur {pid} : taux de passe sous-optimal = {sub_optimal_pass_rate(logger, pid):.3f}")

print("Facteur de branchement moyen :", branching_factor_average(logger))
print("Entropie de l'espace d'action :", action_space_entropy(logger))
print("Volatilité de la révolution :", e_rev_volatility(logger))
print("Longueur moyenne des plis :", trick_length_average(logger))
```

## 9. Simulations de masse parallélisées (`research.run_simulation`)

### 9.1. Commande de base

```bash
python -m research.run_simulation --games 1000 --player-count 4 --rounds-per-game 10 \
  --agent-profile rule_based_bot --workers 4 --output research_output.parquet --seed 0
```

`--agent-profile` accepte `greedy_bot`, `rule_based_bot`, `random_bot`, `mcts_bot`, `rl_agent`, `torch_rl_agent`, appliqué à l'ensemble des sièges de toutes les parties simulées (`_AGENT_REGISTRY`/`_TRAINED_AGENT_PROFILES` de `research/run_simulation.py`), les deux derniers nécessitant `--weights-path` pour charger des poids entraînés (le siège 0 reçoit l'agent entraîné, les sièges suivants reçoivent `rule_based_bot` par défaut). Pour composer une partie hétérogène plutôt qu'un profil unique, utiliser `--seat-profiles` (liste de profils séparés par des virgules, taille `--player-count`) et, le cas échéant, `--seat-weights` (liste de couples `siège:chemin` séparés par des virgules, ciblant les sièges de profil entraînable) :

```bash
python -m research.run_simulation --games 500 --player-count 4 --rounds-per-game 10 \
  --seat-profiles torch_rl_agent,rule_based_bot,greedy_bot,random_bot \
  --seat-weights 0:weights/torch_rl_weights_player4_learnRate0p001_rounds10000_20260101.pt \
  --workers 4 --output research_output.parquet --seed 0
```

Le nombre de parties est réparti aussi équitablement que possible entre `--workers` acteurs Ray (`GameSimulationWorker`), chacun exécutant séquentiellement son lot avec une graine dérivée distincte (`base_seed + offset`) garantissant la reproductibilité de chaque partie individuelle.

Pendant l'exécution, un tableau de bord `rich` (`LiveMonitor`) affiche le débit de parties par seconde, l'utilisation CPU/mémoire/GPU, et un résumé de la distribution des VP observés. Une barre de progression `tqdm` suit en parallèle le nombre de lots d'acteurs Ray achevés.

En fin de campagne, un tableau récapitulatif est affiché (nombre de parties, durée totale, débit, nombre d'événements journalisés, chemin du fichier Parquet), et le fichier `--output` contient l'intégralité des événements de toutes les parties, au format Parquet segmenté décrit en section 8.2.

### 9.2. Exploiter le résultat d'une campagne massive

```python
import polars as pl
import json

df = pl.read_parquet("research_output.parquet")

# Nombre total d'événements par type
print(df.group_by("event_type").count().sort("count", descending=True))

# VP moyen par joueur sur l'ensemble de la campagne
finished = df.filter(pl.col("event_type") == "EventPlayerFinished")
payloads = [json.loads(p) for p in finished["payload"].to_list()]
import statistics
vp_by_player: dict = {}
for p in payloads:
    vp_by_player.setdefault(p["player_id"], []).append(p["vp_earned"])
for pid, vps in sorted(vp_by_player.items()):
    print(f"Joueur {pid} : VP moyen = {statistics.mean(vps):.3f} sur {len(vps)} sorties")
```

Pour appliquer directement les fonctions de `analytics.metrics_calc` à une campagne massive exportée sur disque, il est nécessaire de reconstruire un `EventLogger` en mémoire à partir des enregistrements Parquet (les fonctions de métriques attendent des instances d'événements typées, pas des dictionnaires bruts). Pour des campagnes de taille raisonnable, il est en général plus simple d'agréger directement les métriques par partie au fil de l'exécution, plutôt que de désérialiser après coup l'ensemble des enregistrements Parquet en objets `Event`.

### 9.3. Choisir le nombre de manches et de parties

`--rounds-per-game` fixe le nombre de manches jouées par partie (chaque partie repart d'un `Game` neuf, avec rôles réinitialisés à `None` pour la première manche). Pour étudier la convergence des rôles sur longue série au sein d'une même partie plutôt que sur de nombreuses parties courtes, augmenter `--rounds-per-game` et réduire `--games` :

```bash
python -m research.run_simulation --games 20 --rounds-per-game 500 --player-count 4 \
  --agent-profile rule_based_bot --workers 4 --output long_run.parquet
```

### 9.4. Évaluation comparative d'agents et de modèles entraînés (`research.evaluate_agents`)

`research.evaluate_agents` complète `research.run_simulation` en se concentrant sur la performance comparée de profils hétérogènes (taux de victoire, VP cumulé par profil) plutôt que sur les métriques macro/micro de `analytics.metrics_calc`. Il rejoue directement `engine.game_runner.Game.play_round` (donc via le moteur événementiel complet, toutes règles avancées comprises) plutôt que le moteur vectorisé.

```bash
python -m research.evaluate_agents --seat-profiles torch_rl_agent,rule_based_bot,greedy_bot,random_bot \
  --seat-weights 0:weights/torch_rl_weights_player4_learnRate0p001_rounds10000_20260101.pt \
  --games 200 --rounds-per-game 20 --config-preset full --workers 4 --seed 0 \
  --experiment-name eval_torch_vs_baselines
```

Le fichier CSV produit (nommé selon la convention de `naming.build_research_filename`, extension `.csv`) porte une ligne par joueur et par partie, avec `profile`, `cumulative_vp` et `president_rounds` (nombre de manches terminées au rôle `ROLE_PRESIDENT`), directement exploitable pour comparer des profils fixes, des profils heuristiques, et un ou plusieurs modèles entraînés chargés par `--seat-weights`.

## 10. Le moteur vectorisé (`training.fast_path.FastPathEngine`)

### 10.1. Principe

`FastPathEngine` traite un lot de `B` manches indépendantes en lock-step, sur des tableaux `numpy`, sans passer par le système d'événements ni par les instances `AbstractBaseAgent`. Les mains sont représentées par des vecteurs de comptage par rang (colonnes 0 à 12 pour les rangs standard, colonne 13 pour les Jokers) ; les couleurs ne sont pas représentées. Ce moteur est destiné exclusivement à l'entraînement à très haut débit ou à l'exploration algorithmique ; toute analyse fine par couleur (interception, départage d'égalité par couleur) n'y est pas modélisée.

### 10.2. Utilisation directe avec une politique arbitraire

```python
from core.config import GameConfig
from training.fast_path import FastPathEngine, ACTION_PASS
import numpy as np

config = GameConfig(player_count=4)
engine = FastPathEngine(config, batch_size=1024)
state = engine.reset(base_seed=0)

while not state.done.all():
    mask = engine.legal_action_mask()
    has_option = mask.any(axis=1)
    actions = np.full(state.done.shape[0], ACTION_PASS, dtype=np.int64)
    actions[has_option] = mask[has_option].argmax(axis=1)
    state, reward, done = engine.step(actions)

print("Rangs de sortie finaux :", state.finish_rank)
```

### 10.3. Utilisation via `engine.game_runner.Game.vectorized_run`

`Game.vectorized_run` expose la même mécanique avec une fonction de politique optionnelle, recevant `(state_tensor, legal_mask)` et retournant un tableau `(B,)` d'index d'action :

```python
def policy(state_tensor, legal_mask):
    # politique aléatoire uniforme parmi les actions légales
    import numpy as np
    actions = np.full(legal_mask.shape[0], -1, dtype=np.int64)
    for row in range(legal_mask.shape[0]):
        legal_indices = np.nonzero(legal_mask[row])[0]
        if legal_indices.size > 0:
            actions[row] = np.random.choice(legal_indices)
    return actions

result = game.vectorized_run(batch_size=2048, max_steps=400, base_seed=0, policy=policy)
print(result["reward"].mean(), result["steps"])
```

Le dictionnaire retourné contient `finish_rank` (rangs de sortie par joueur et par partie du lot), `done` (statut de fin), `reward` (VP `SYMMETRICAL` cumulé par partie), et `steps` (nombre de transitions exécutées avant arrêt).

### 10.4. Construire un masque d'action à partir d'options légales déjà calculées

`core.action_masking.build_action_space_index`, `build_action_mask_batch` et `legal_option_for_action` permettent de projeter des options `(cards, declared_power)` déjà générées par `core.rules_engine.generate_uniform_plays`/`generate_sequence_plays` sur un espace d'action discret fixe indexé par `(power, size)`, utile pour entraîner un réseau de neurones consommant directement les sorties du moteur événementiel complet plutôt que celles du `FastPathEngine`.

## 11. Entraînement d'un agent à politique linéaire (`training.train_rl`)

### 11.1. Commande de base

```bash
python -m training.train_rl --rounds 5000 --player-count 4 --learning-rate 0.01 \
  --opponent-pool mixed --seed 0 --output rl_weights.npy
```

`--opponent-pool` accepte `greedy`, `rule_based`, `mixed` (alternance `GreedyBot`/`RuleBasedBot` sur les sièges adverses). L'agent entraîné, `agents.rl_agent.RLAgent`, occupe toujours le siège 0. Sa politique est un vecteur de poids linéaires de dimension `FEATURE_DIM = 5`, appliqué à un vecteur de caractéristiques par option candidate (`_option_features` : puissance normalisée, ratio de taille répété deux fois, indicateur de présence de Joker, biais constant).

L'algorithme implémenté est un REINFORCE simplifié : à chaque manche, les caractéristiques et le score de chaque décision `ACTION_PLAY` du joueur entraîné sont collectées (`ReturnTracker`/`tracker`), puis le gradient de politique est calculé comme la moyenne, sur l'ensemble des décisions de la manche, de l'avantage (VP obtenu moins score moyen de référence) pondérant le vecteur de caractéristiques. `trainee.epsilon` décroît géométriquement (`_EPSILON_DECAY = 0.995`, plancher `_EPSILON_MIN = 0.02`) au fil des manches, réduisant progressivement l'exploration aléatoire.

Toutes les 100 manches, un tableau `rich` affiche le VP moyen glissant sur les 100 dernières manches, l'`epsilon` courant et les poids de politique. En fin d'entraînement, les poids finaux sont sauvegardés au format `numpy` (`np.save`) dans le fichier indiqué par `--output`.

### 11.2. Réutiliser les poids entraînés

```python
import numpy as np
from core.config import GameConfig
from agents.rl_agent import RLAgent

config = GameConfig(player_count=4)
weights = np.load("rl_weights.npy")
trained_agent = RLAgent(player_id=0, config=config, weights=weights, epsilon=0.0)
```

Fixer `epsilon=0.0` désactive toute exploration résiduelle, pour une exploitation pure de la politique apprise, par exemple en confrontation avec `research.run_simulation` (à condition d'enregistrer ce profil dans un registre d'agents personnalisé, `research/run_simulation.py` n'exposant que les quatre profils fixes de `_AGENT_REGISTRY` par défaut).

### 11.3. Évaluer un agent entraîné contre les profils de référence

```python
from engine.event_bus import EventBus
from engine.game_runner import Game
from agents.greedy_bot import GreedyBot
from agents.rule_based_bot import RuleBasedBot

agents = {
    0: trained_agent,
    1: GreedyBot(1, config),
    2: RuleBasedBot(2, config),
    3: GreedyBot(3, config),
}
game = Game(config, agents, event_bus=EventBus(), game_id="evaluation")
vp_tensor = game.play_rounds_vectorized(500)
print("VP moyen de l'agent entraîné :", vp_tensor[:, 0].mean())
```

## 12. Entraînement distribué d'une politique neuronale (Ray + Redis)

### 12.1. Vue d'ensemble de la chaîne

Cette voie sépare deux rôles :

* des **Rollout Workers** (`training.rollout_worker.RolloutWorker`, acteurs Ray) qui simulent des manches complètes via le moteur événementiel (`engine.round.run_round`), avec un agent entraînable `agents.torch_rl_agent.TorchRLAgent` au siège 0, synchronisé périodiquement avec les derniers poids publiés ; les transitions `(features, return_value)` collectées sont poussées par lot dans un `RedisReplayBuffer` partagé ;
* un unique **Trainer** (`training.trainer.Trainer`) qui échantillonne ce tampon, applique des mises à jour de gradient de politique sur un `agents.torch_rl_agent.PolicyNet` (perceptron à deux couches cachées de 32 neurones), sous précision mixte automatique lorsqu'un GPU CUDA est disponible, et republie régulièrement les poids mis à jour sur Redis afin que les Rollout Workers les récupèrent à leur prochaine manche.

Le tampon Redis (`training.replay_buffer.RedisReplayBuffer`) matérialise une file bornée (`capacity`, 200000 par défaut) sous forme de liste Redis, chaque transition étant sérialisée en JSON. `push_batch` insère plusieurs transitions en une seule opération réseau, `sample` échantillonne avec remise par lecture d'index uniformes, `size` retourne la longueur courante de la liste, `ping` vérifie la joignabilité du serveur.

### 12.2. Démarrer Redis

Voir la section 13 pour le détail complet des trois méthodes d'installation (Docker, WSL2, Memurai sous Windows natif) et des commandes de vérification.

### 12.3. Lancer l'entraînement distribué conjoint

```bash
python -m training.launch_distributed --workers 4 --rounds-per-batch 50 --opponent-pool mixed \
  --player-count 4 --redis-host localhost --redis-port 6379 --batch-size 256 --total-steps 10000
```

`training.launch_distributed.launch` vérifie d'abord la joignabilité de Redis via `RedisReplayBuffer.ping()` avant toute autre initialisation ; en cas d'échec, une `RuntimeError` explicite est levée immédiatement, indiquant l'hôte et le port testés, plutôt que de laisser Ray ou un thread échouer silencieusement plus tard. Le cluster Ray local est ensuite initialisé (`ray.init(num_cpus=num_workers, ...)`), le `Trainer` démarre dans un thread dédié, et la boucle principale distribue en continu des lots de manches (`rounds_per_worker_batch` par appel) aux acteurs `RolloutWorker` jusqu'à ce que le thread du `Trainer` se termine (c'est-à-dire jusqu'à ce que `total_steps` étapes de gradient aient été exécutées).

`--opponent-pool` (`greedy`, `rule_based`, `mixed`) détermine les profils adverses simulés par chaque Rollout Worker, résolus par `RolloutWorker._opponent_classes`.

### 12.4. Lancer séparément le Trainer

Utile pour reprendre un entraînement avec des Rollout Workers déjà actifs sur un autre poste, ou pour changer de périphérique de calcul sans redémarrer les workers :

```bash
python -m training.trainer --redis-host localhost --redis-port 6379 --batch-size 256 \
  --total-steps 10000 --learning-rate 0.001 --device cuda
```

`--device` accepte `cuda` ou `cpu` ; en son absence, `resolve_device` détecte automatiquement la disponibilité de CUDA (`agents.torch_rl_agent.resolve_device`). Le `Trainer` attend que le tampon Redis contienne au moins `batch_size` transitions avant de démarrer la première étape de gradient (`while self.buffer.size() < batch_size: time.sleep(1.0)`), republie les poids toutes les 50 étapes (`_PUBLISH_INTERVAL`), et affiche à chaque publication un tableau `rich` (perte courante, taille du tampon, périphérique utilisé).

### 12.5. Récupérer et exploiter les poids finaux entraînés

Les poids sont publiés dans Redis sous la clé `president:policy_weights`, sérialisés par `torch.save`/`io.BytesIO`. Pour les récupérer en dehors du processus `Trainer` :

```python
import io
import torch
from training.replay_buffer import RedisReplayBuffer
from agents.torch_rl_agent import PolicyNet, resolve_device

buffer = RedisReplayBuffer(host="localhost", port=6379)
raw = buffer.client.get("president:policy_weights")
device = resolve_device()
policy = PolicyNet().to(device)
policy.load_state_dict(torch.load(io.BytesIO(raw), map_location=device))
policy.eval()

torch.save(policy.state_dict(), "policy_final.pt")
```

Pour recharger ensuite ces poids dans un `TorchRLAgent` prêt à jouer, sans exploration :

```python
from core.config import GameConfig
from agents.torch_rl_agent import TorchRLAgent

config = GameConfig(player_count=4)
agent = TorchRLAgent(player_id=0, config=config, epsilon=0.0)
agent.load_weights("policy_final.pt")
```

`TorchRLAgent.save_weights(path)` et `TorchRLAgent.load_weights(path)` encapsulent directement cette opération pour un agent déjà instancié.

## 13. Installer et exploiter Redis en détail

### 13.1. Pourquoi Redis est nécessaire

Redis sert exclusivement de tampon de rejeu partagé (`training.replay_buffer.RedisReplayBuffer`) entre les acteurs Ray `RolloutWorker` (producteurs de transitions) et le processus `Trainer` (consommateur), ainsi que de canal de republication des poids de politique entraînés (`president:policy_weights`). Redis n'est requis à aucun autre endroit du projet : `play_game.py`, `research.run_simulation`, `training.train_rl` et le moteur vectorisé `FastPathEngine` fonctionnent sans lui.

### 13.2. Installation, trois options

**Docker (recommandé, toutes plateformes)** :

```bash
docker run -p 6379:6379 --name president-redis -d redis
```

Cette commande démarre un conteneur Redis en arrière-plan (`-d`), expose le port 6379 sur l'hôte, et nomme le conteneur `president-redis` pour le retrouver facilement. Pour l'arrêter puis le relancer sans le recréer :

```bash
docker stop president-redis
docker start president-redis
```

Pour le supprimer définitivement (les données du tampon sont perdues, ce qui est acceptable puisqu'il ne s'agit que d'un tampon d'entraînement transitoire) :

```bash
docker rm -f president-redis
```

**WSL2 (Windows avec sous-système Linux)** :

```bash
wsl sudo apt-get update
wsl sudo apt-get install redis-server
wsl redis-server --daemonize yes
```

Le service tourne alors en arrière-plan à l'intérieur de la distribution WSL2, accessible depuis Windows sur `localhost:6379` grâce au transfert réseau automatique de WSL2.

**Windows natif (sans Docker ni WSL2)** : installer Memurai (serveur compatible protocole Redis pour Windows) et démarrer le service Memurai correspondant depuis le gestionnaire de services Windows ou son installateur.

### 13.3. Vérifier que Redis répond

Avec Docker :

```bash
docker exec -it president-redis redis-cli ping
```

Doit répondre `PONG`. Avec WSL2 :

```bash
wsl redis-cli ping
```

Depuis Python, sans dépendre d'un client `redis-cli` installé séparément :

```python
from training.replay_buffer import RedisReplayBuffer

buffer = RedisReplayBuffer(host="localhost", port=6379)
print("Redis joignable :", buffer.ping())
```

`RedisReplayBuffer.ping()` capture toute exception de connexion (serveur non démarré, hôte ou port incorrect, réseau indisponible) et retourne simplement `False` plutôt que de lever une exception, ce qui permet de le tester sans bloc `try/except` supplémentaire.

### 13.4. Inspecter et administrer le tampon manuellement

```python
from training.replay_buffer import RedisReplayBuffer

buffer = RedisReplayBuffer(host="localhost", port=6379)
print("Transitions actuellement en tampon :", buffer.size())

# Vider intégralement le tampon, par exemple entre deux campagnes d'entraînement distinctes
buffer.clear()

# Échantillonner manuellement un lot pour inspection, sans lancer de Trainer
batch = buffer.sample(batch_size=32)
if batch is not None:
    print("Exemple de transition :", batch[0])
```

Le tampon suit une politique de troncature FIFO : chaque insertion (`push`/`push_batch`) place les nouvelles transitions en tête de liste Redis puis tronque la liste à `capacity` éléments (`LTRIM`), garantissant que le tampon ne dépasse jamais la taille configurée même sur une campagne de très longue durée.

### 13.5. Changer d'hôte, de port ou de base logique Redis

Tous les composants (`RedisReplayBuffer`, `RolloutWorker`, `Trainer`, `training.launch_distributed`) acceptent `--redis-host`/`--redis-port` (ou les paramètres `host`/`port` du constructeur `RedisReplayBuffer`) pour cibler une instance Redis distante plutôt que locale, par exemple un serveur Redis managé accessible sur le réseau du cluster d'entraînement. Le paramètre `db` de `RedisReplayBuffer` permet en outre d'isoler plusieurs tampons indépendants sur la même instance Redis (bases logiques numérotées `0` à `15` par défaut sous Redis), utile pour faire cohabiter plusieurs campagnes d'entraînement distinctes sans collision de clés.

## 14. Écrire son propre agent

Tout agent, humain ou automatisé, doit hériter de `agents.interface.AbstractBaseAgent` et implémenter ses quatre méthodes abstraites. Exemple d'agent minimal jouant toujours la première option légale disponible :

```python
from typing import List, Optional, Tuple

from agents.interface import AbstractBaseAgent
from core.config import GameConfig
from core.models import Action, ActionType, Card, Hand
from core.rules_engine import generate_sequence_plays, generate_uniform_plays
from engine.state import GameState


class FirstOptionBot(AbstractBaseAgent):
    def __init__(self, player_id: int, config: GameConfig) -> None:
        super().__init__(player_id)
        self.config = config

    def _legal_options(self, hand: Hand, game_state: GameState):
        trick = game_state.trick
        required_size = trick.size if trick.size > 0 else None
        min_power = trick.current_power
        options = []
        if not trick.is_sequence:
            options.extend(generate_uniform_plays(hand, game_state.e_rev, required_size, min_power))
        if self.config.straights_enabled and (trick.size == 0 or trick.is_sequence):
            seq_min = trick.sequence_min_power if trick.is_sequence else None
            for cards, joker_map in generate_sequence_plays(hand, game_state.e_rev, required_size, seq_min):
                declared = joker_map[min(joker_map)] if joker_map else None
                options.append((cards, declared))
        return options

    def choose_action(self, game_state: GameState) -> Action:
        hand = game_state.hands[self.player_id]
        options = self._legal_options(hand, game_state)
        if not options:
            action_type = (
                ActionType.ACTION_SOFT_PASS if self.config.pass_type == "ALLOW_SOFT"
                else ActionType.ACTION_HARD_PASS
            )
            return Action(action_type=action_type)
        cards, declared_power = options[0]
        return Action(action_type=ActionType.ACTION_PLAY, cards=cards, declared_power=declared_power)

    def choose_exchange_cards(self, hand: Hand, game_state: GameState, count: int) -> List[Card]:
        return list(hand.cards[:count])

    def ask_putsch(self, hand: Hand) -> bool:
        return False

    def on_interception_opportunity(self, game_state: GameState, played_card: Card) -> Tuple[bool, Optional[Card]]:
        return False, None
```

Points impératifs à respecter, contrôlés par le moteur (`core.rules_engine.is_action_valid`) avant application de toute action retournée par `choose_action` :

* toute combinaison contenant un Joker doit obligatoirement porter un `declared_power` non nul dans l'`Action` retournée ;
* une action illégale n'est jamais appliquée telle quelle : `engine.round.run_round` la remplace silencieusement par un passe conforme à `pass_type` et marque l'événement `EventActionPlayed` correspondant avec `was_suboptimal=True`, ce qui permet de détecter après coup, via `analytics.metrics_calc.sub_optimal_pass_rate`, tout agent produisant fréquemment des propositions invalides ;
* aucune contrainte de pureté n'est imposée à l'implémentation d'un agent : consommer un générateur pseudo-aléatoire interne, mémoriser un état entre les appels, ou effectuer des simulations internes (comme `agents.mcts_bot.MCTSBot`) est parfaitement admis, à condition de ne jamais muter directement l'instance `GameState` qui est transmise en lecture.

Pour bénéficier d'une inférence par lot optimisée (utile notamment avec le moteur vectorisé ou pour l'entraînement distribué), surcharger `get_batch_action(self, game_states)` plutôt que de laisser l'implémentation par défaut de `AbstractBaseAgent` appeler `choose_action` séquentiellement ; voir `agents.rl_agent.RLAgent.get_batch_action` et `agents.torch_rl_agent.TorchRLAgent.get_batch_action` pour deux exemples complets d'inférence matricielle groupée.

Pour intégrer le nouvel agent à `play_game.py`, l'ajouter simplement à `_AGENT_REGISTRY` dans `play_game.py` :

```python
from agents.first_option_bot import FirstOptionBot
_AGENT_REGISTRY["first_option"] = FirstOptionBot
```

## 15. Suivi en temps réel d'une campagne (`analytics.live_monitor.LiveMonitor`)

`LiveMonitor` s'utilise comme gestionnaire de contexte, autour de n'importe quelle boucle de simulation personnalisée :

```python
from analytics.live_monitor import LiveMonitor

with LiveMonitor() as monitor:
    for batch_index in range(100):
        # exécuter un lot de parties ici
        rewards = [0.0] # remplacer par les VP réellement observés
        monitor.record_games(count=10, rewards=rewards)
```

`record_games(count, rewards=None)` incrémente le compteur de parties complétées et, si `rewards` est fourni, alimente un échantillon glissant borné (`max_reward_samples`, 5000 par défaut) utilisé pour afficher la moyenne et l'écart type approché des VP observés. Le tableau affiché comprend systématiquement : parties complétées, parties par seconde, durée écoulée, utilisation CPU (`psutil.cpu_percent()`), mémoire utilisée, et utilisation GPU lorsque `pynvml` détecte au moins un périphérique compatible (`indisponible` sinon).

## 16. Dépannage

**`redis.exceptions.ConnectionError` / `ConnectionRefusedError [WinError 10061]`**
Aucun serveur Redis n'écoute sur l'hôte/port indiqués. Démarrer Redis (section 13.2) avant de relancer `training.launch_distributed` ou `training.trainer`. Depuis la vérification systématique ajoutée dans `training.launch_distributed.launch`, l'absence de Redis est détectée avant le démarrage de Ray et des threads, avec un message d'erreur explicite indiquant l'hôte et le port testés.

**`ValueError: double_revolution_enabled et interception_enabled requièrent un nombre de paquets effectif supérieur ou égal à 2.`**
Augmenter `--player-count` (5 ou plus en dimensionnement automatique, puisque $N_D = \max(1, \lfloor (N-1)/4 \rfloor + 1)$), ou fixer explicitement `--forced-deck-count 2 --disable-deck-scaling-auto`.

**`ValueError: --seats doit contenir exactement N profil(s), M fourni(s).`**
Le nombre de profils séparés par des virgules dans `--seats` ne correspond pas à `--player-count`. Recompter les profils ou ajuster `--player-count`.

**`ValueError: Profil de siège inconnu : 'X'.`**
Le profil demandé n'existe pas dans `_AGENT_REGISTRY` de `play_game.py`. Les profils valides sont `human`, `random`, `greedy`, `rule_based`, `mcts`, sauf ajout manuel d'un agent personnalisé (section 14).

**Le script `play_game.py` boucle sans jamais proposer d'options à un siège humain**
Vérifier que le profil du siège concerné est bien `human` dans `--seats`, et que le nombre de profils correspond exactement à `--player-count`.

**`RuntimeError: État incohérent détecté : pli ... dépasse ... actions sans clôture.`**
Cette garde de sécurité, définie dans `engine.round.run_round` (`_MAX_ACTIONS_PER_TRICK`), signale une configuration de règles produisant un pli qui ne se clôture jamais (typiquement `pass_type` mal choisi conjointement à `skip_on_equal`, cf. `rules.md` §6.3, note de cohérence). Vérifier la cohérence de `--pass-type` avec les autres règles actives, en particulier `skip_on_equal` et `finish_penalty_extended`.

**`ImportError` sur `pynvml` lors de l'utilisation de `LiveMonitor`**
Sans effet bloquant : `_try_init_nvml` capture l'absence du paquet ou l'échec d'initialisation et affiche `indisponible` pour la métrique GPU. Installer `pynvml` (déjà listé dans `requirements.txt`) et disposer d'un pilote NVIDIA compatible pour obtenir la métrique réelle.

**Entraînement distribué qui ne progresse jamais (`Trainer` bloqué en attente)**
`Trainer.run` attend que `buffer.size() >= batch_size` avant la première étape de gradient. Vérifier que des `RolloutWorker` sont effectivement actifs et poussent des transitions (`buffer.size()` doit croître au fil du temps), et que `--batch-size` n'est pas disproportionné par rapport au débit de production de transitions.

**Résultats non reproductibles malgré un `--seed` fixé**
La reproductibilité garantie par `random_seed` couvre la distribution des mains (`core.rules_engine.deal_hands`, dérivée de `f"{random_seed}:{round_index}"`) et les tirages internes déterministes des agents fournis (`random.Random(f"{config.random_seed}:{player_id}")`). Un agent personnalisé utilisant une source d'aléa non dérivée de `config.random_seed`, ou une exécution distribuée avec ordre d'arrivée non déterministe des transitions Redis, ne garantit pas la reproductibilité stricte d'une campagne entière.
