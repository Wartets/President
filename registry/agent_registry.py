"""
Module du registre centralisé des profils d'agents.

Le module rassemble en un unique point l'association entre nom de profil et classe d'agent instanciable, jusqu'ici dupliquée
indépendamment dans `play_game.py`, `step_by_step_run.py` et `research.run_simulation`. Toute addition d'un nouveau profil
d'agent au projet ne nécessite désormais qu'une modification de ce module : les trois points d'entrée mentionnés
s'appuient tous sur `HEURISTIC_AGENT_REGISTRY`/`AUTOMATED_AGENT_REGISTRY` et sur la fabrique `build_agent` exposées ici.

Le module dépend de `agents.interface`, de l'ensemble des modules `agents.*` définissant un profil heuristique, et de
`core.config`.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

from agents.adaptive_bot import AdaptiveBot
from agents.aggressive_bot import AggressiveBot
from agents.greedy_bot import GreedyBot
from agents.human_agent import HumanAgent
from agents.interface import AbstractBaseAgent
from agents.lookahead_bot import LookaheadBot
from agents.mcts_bot import MCTSBot
from agents.probabilistic_bot import ProbabilisticBot
from agents.random_bot import RandomBot
from agents.rule_based_bot import RuleBasedBot
from agents.scoring_bot import ScoringBot
from core.config import GameConfig

# Association complète entre nom de profil et classe d'agent instanciable via `(player_id, config)`, incluant le
# profil interactif `human_agent`. Chaque clé correspond exactement au nom du module Python définissant la classe
# d'agent (`agents/<clé>.py`).
HEURISTIC_AGENT_REGISTRY: Dict[str, Callable[[int, GameConfig], AbstractBaseAgent]] = {
    "human_agent": HumanAgent,
    "random_bot": RandomBot,
    "greedy_bot": GreedyBot,
    "aggressive_bot": AggressiveBot,
    "rule_based_bot": RuleBasedBot,
    "lookahead_bot": LookaheadBot,
    "adaptive_bot": AdaptiveBot,
    "scoring_bot": ScoringBot,
    "probabilistic_bot": ProbabilisticBot,
    "mcts_bot": MCTSBot,
}

# Sous-ensemble de HEURISTIC_AGENT_REGISTRY excluant le profil interactif `human_agent`, utilisé par les campagnes de
# simulation automatisées (`research.run_simulation`) où aucun siège ne doit solliciter une saisie clavier.
AUTOMATED_AGENT_REGISTRY: Dict[str, Callable[[int, GameConfig], AbstractBaseAgent]] = {
    key: value for key, value in HEURISTIC_AGENT_REGISTRY.items() if key != "human_agent"
}

# Profils entraînables dont la construction nécessite le chargement optionnel d'un fichier de poids
# (`agents.rl_agent.RLAgent` ou `agents.torch_rl_agent.TorchRLAgent`).
TRAINED_AGENT_PROFILES: Tuple[str, ...] = ("rl_agent", "torch_rl_agent")

# Ensemble complet des profils utilisables pour un siège interactif (y compris `human_agent`).
ALL_SEAT_PROFILES: Tuple[str, ...] = tuple(HEURISTIC_AGENT_REGISTRY.keys()) + TRAINED_AGENT_PROFILES

# Ensemble complet des profils utilisables pour un siège automatisé (excluant `human_agent`).
ALL_AUTOMATED_PROFILES: Tuple[str, ...] = tuple(AUTOMATED_AGENT_REGISTRY.keys()) + TRAINED_AGENT_PROFILES


def build_agent(
    profile: str,
    player_id: int,
    config: GameConfig,
    weights_path: Optional[str] = None,
) -> AbstractBaseAgent:
    """
    Construit l'agent d'un unique siège à partir de son nom de profil, y compris pour les profils entraînables.

    Paramètre `profile` : nom de profil, clé de `HEURISTIC_AGENT_REGISTRY` ou de `TRAINED_AGENT_PROFILES`.
    Paramètre `player_id` : identifiant du joueur occupant le siège.
    Paramètre `config` : configuration de la partie.
    Paramètre `weights_path` : chemin d'un fichier de poids entraîné pour ce siège précis, ou `None`/chaîne vide si le
    siège ne charge aucun poids ; ignoré pour les profils non entraînables.
    Retourne une instance de `AbstractBaseAgent`. Lève `ValueError` si `profile` ne correspond à aucun profil connu.
    Aucun effet de bord hors chargement disque des poids éventuels.
    """
    if profile == "rl_agent":
        import numpy as np
        from agents.rl_agent import RLAgent

        weights = np.load(weights_path) if weights_path else None
        return RLAgent(player_id, config, weights=weights, epsilon=0.0)

    if profile == "torch_rl_agent":
        from agents.torch_rl_agent import TorchRLAgent

        trained = TorchRLAgent(player_id=player_id, config=config, epsilon=0.0)
        if weights_path:
            trained.load_weights(weights_path)
        return trained

    if profile in HEURISTIC_AGENT_REGISTRY:
        return HEURISTIC_AGENT_REGISTRY[profile](player_id, config)

    raise ValueError(f"Profil d'agent inconnu : '{profile}'.")
