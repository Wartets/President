"""
Module du gestionnaire de partie.

Le module définit `Game`, la classe orchestrant la succession des manches d'une partie complète. La classe ne mute aucun état de jeu directement ;
elle délègue l'exécution de chaque manche à `engine.round.run_round` et accumule les points de victoire retournés au fil des manches. Chaque
démarrage de partie publie `EventGameConfig` et `EventGameStart` sur le bus fourni.

Le module dépend de `core.config`, `agents.interface`, `engine.event_bus`, `engine.round` et `events.structural`.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from agents.interface import AbstractBaseAgent
from core.config import GameConfig
from engine.event_bus import EventBus
from engine.round import run_round
from events.structural import EventGameConfig, EventGameStart


class Game:
    """
    Gestionnaire de l'exécution d'une partie complète, machine à états de haut niveau enchaînant les manches.

    Champ `config` : configuration immuable de la partie.
    Champ `agents` : association entre identifiant de joueur et instance d'agent.
    Champ `event_bus` : bus de diffusion des événements de la partie.
    Champ `game_id` : identifiant unique de la partie.
    Champ `cumulative_vp` : association entre identifiant de joueur et somme des points de victoire accumulés sur l'ensemble des manches jouées.
    Champ `roles` : association entre identifiant de joueur et rôle courant, `None` avant la première manche.
    Champ `round_index` : index de la prochaine manche à exécuter.
    """

    def __init__(
        self,
        config: GameConfig,
        agents: Dict[int, AbstractBaseAgent],
        event_bus: Optional[EventBus] = None,
        game_id: str = "game-0",
    ) -> None:
        self.config = config
        self.agents = agents
        self.event_bus = event_bus if event_bus is not None else EventBus()
        self.game_id = game_id
        self.cumulative_vp: Dict[int, float] = {pid: 0.0 for pid in range(config.player_count)}
        self.roles: Optional[Dict[int, str]] = None
        self.round_index = 0

        self.event_bus.publish(
            EventGameConfig(timestamp=0, game_id=game_id, round_id=-1, state_hash="", config=config)
        )
        self.event_bus.publish(
            EventGameStart(
                timestamp=0,
                game_id=game_id,
                round_id=-1,
                state_hash="",
                config=config,
                player_ids=tuple(range(config.player_count)),
            )
        )

    def play_round(self) -> Dict[int, float]:
        """
        Exécute la manche suivante et met à jour l'état cumulatif de la partie.

        Retourne l'association point de victoire par identifiant de joueur pour la manche qui vient d'être jouée. Effet de bord : incrémente
        `round_index`, remplace `roles` par les rôles attribués pour la manche suivante, et ajoute les points de victoire de la manche à
        `cumulative_vp`. Publie les événements de la manche sur `event_bus`.
        """
        roles, vp_by_player, _finish_order = run_round(
            self.config,
            self.agents,
            self.event_bus,
            self.round_index,
            self.roles,
            self.game_id,
        )
        for pid, vp in vp_by_player.items():
            self.cumulative_vp[pid] = self.cumulative_vp.get(pid, 0.0) + vp
        self.roles = roles
        self.round_index += 1
        return vp_by_player

    def play_rounds(self, count: int) -> List[Dict[int, float]]:
        """
        Exécute plusieurs manches consécutives.

        Paramètre `count` : nombre de manches à exécuter, entier positif.
        Retourne la liste des associations point de victoire par identifiant de joueur, une par manche exécutée, dans l'ordre d'exécution. Mêmes
        effets de bord que `play_round`, répétés `count` fois.
        """
        return [self.play_round() for _ in range(count)]

    def play_rounds_vectorized(self, count: int) -> np.ndarray:
        """
        Exécute plusieurs manches consécutives et restitue les points de victoire sous forme de tenseur.

        Paramètre `count` : nombre de manches à exécuter, entier positif.
        Retourne un tableau `numpy.ndarray` de type `float64` et de forme `(count, player_count)`, où l'élément d'indice `[i, pid]` est le
        point de victoire attribué au joueur `pid` à l'issue de la manche d'index `i`. Destiné aux pipelines d'entraînement consommant des
        tenseurs plutôt que des dictionnaires Python, conformément à la séparation Fast-Path / Slow-Path. Mêmes effets de bord que
        `play_round`, répétés `count` fois.
        """
        n = self.config.player_count
        vp_tensor = np.zeros((count, n), dtype=np.float64)
        for i in range(count):
            vp_by_player = self.play_round()
            for pid, vp in vp_by_player.items():
                vp_tensor[i, pid] = vp
        return vp_tensor
