"""
Module des acteurs de génération de transitions.

Le module définit `RolloutWorker`, un acteur Ray produisant des transitions de politique à partir de manches simulées localement. Les
transitions sont insérées par lot dans `RedisReplayBuffer` afin de réduire le coût réseau.
"""

from __future__ import annotations

import io
from typing import Dict, List, Optional, Type, Any

import numpy as np
import ray
import torch

from agents.greedy_bot import GreedyBot
from agents.interface import AbstractBaseAgent
from agents.rl_agent import _option_features
from agents.rule_based_bot import RuleBasedBot
from agents.torch_rl_agent import TorchRLAgent
from core.config import GameConfig
from engine.event_bus import EventBus
from engine.round import run_round
from training.replay_buffer import RedisReplayBuffer

_WEIGHTS_KEY = "president:policy_weights"


@ray.remote
class RolloutWorker:
    """
    Acteur Ray générant des transitions de politique.

    Champ `config` : configuration utilisée par les manches simulées.
    Champ `buffer` : tampon Redis recevant les transitions.
    Champ `round_index` : compteur local de manches exécutées.
    Champ `roles` : rôles issus de la dernière manche exécutée par l'acteur.
    """

    def __init__(self, config: GameConfig, redis_host: str = "localhost", redis_port: int = 6379) -> None:
        self.config = config
        self.buffer = RedisReplayBuffer(host=redis_host, port=redis_port)
        self.round_index = 0
        self.roles = None

    def _load_latest_weights(self, agent: TorchRLAgent) -> None:
        """
        Charge les derniers poids publiés dans Redis lorsque ceux-ci existent.

        Paramètre `agent` : agent neuronal à synchroniser.
        Retourne `None`. Effet de bord : remplace les poids de `agent.policy` si la clé Redis contient un état valide.
        """
        raw = self.buffer.client.get(_WEIGHTS_KEY)
        if raw is None:
            return
        # Redis clients may return `bytes` or `str` depending on configuration; ensure bytes for BytesIO
        if isinstance(raw, str):
            raw_bytes = raw.encode()
        else:
            raw_bytes = raw  # type: ignore[assignment]
        state_dict = torch.load(io.BytesIO(raw_bytes), map_location=agent.device)
        agent.policy.load_state_dict(state_dict)
        agent.policy.eval()

    def _opponent_classes(self, opponent_pool: str) -> List[Type]:
        """
        Résout les classes d'adversaires utilisées dans une manche.

        Paramètre `opponent_pool` : nom du pool, valeurs `greedy`, `rule_based` ou `mixed`.
        Retourne une liste de classes d'agents de taille `player_count - 1`.
        """
        if opponent_pool == "greedy":
            return [GreedyBot] * (self.config.player_count - 1)
        if opponent_pool == "rule_based":
            return [RuleBasedBot] * (self.config.player_count - 1)
        mixed = [GreedyBot, RuleBasedBot] * self.config.player_count
        return mixed[: self.config.player_count - 1]

    def run_rounds(self, opponent_pool: str, round_count: int) -> int:
        """
        Exécute plusieurs manches et publie les transitions collectées.

        Paramètre `opponent_pool` : profil des adversaires simulés.
        Paramètre `round_count` : nombre de manches à exécuter, entier strictement positif.
        Retourne le nombre de transitions publiées dans le tampon Redis. Effet de bord : simule des manches, synchronise périodiquement les
        poids de la politique et insère des transitions dans Redis.
        """
        published = 0
        opponent_classes = self._opponent_classes(opponent_pool)

        for _ in range(round_count):
            trainee = TorchRLAgent(player_id=0, config=self.config, epsilon=0.1)
            self._load_latest_weights(trainee)
            opponents: Dict[int, AbstractBaseAgent] = {
                pid + 1: opponent_classes[pid](pid + 1, self.config)
                for pid in range(self.config.player_count - 1)
            }
            agents = dict(opponents)
            agents[0] = trainee

            transitions: List[Dict[str, object]] = []
            original_choose = trainee.choose_action

            def _instrumented_choose(game_state):
                hand = game_state.hands[trainee.player_id]
                options = trainee._legal_options(hand, game_state)
                action = original_choose(game_state)
                if options and action.cards:
                    hand_size = hand.size()
                    features = _option_features(action.cards, action.declared_power, hand_size, game_state.e_rev)
                    transitions.append(
                        {
                            "features": features.tolist(),
                            "chosen_score": 0.0,
                            "return_value": 0.0,
                        }
                    )
                return action

            trainee.choose_action = _instrumented_choose  # type: ignore[assignment]
            try:
                roles, vp_by_player, _finish_order = run_round(
                    self.config,
                    agents,
                    EventBus(),
                    self.round_index,
                    self.roles,
                    f"rollout-{self.round_index}",
                )
            finally:
                trainee.choose_action = original_choose  # type: ignore[assignment]

            return_value = float(vp_by_player.get(trainee.player_id, 0.0))
            for transition in transitions:
                transition["return_value"] = return_value

            self.buffer.push_batch(transitions)
            published += len(transitions)
            self.roles = roles
            self.round_index += 1

        return published
