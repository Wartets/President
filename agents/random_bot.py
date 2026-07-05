"""
Module de l'agent aléatoire.

Le module définit `RandomBot`, une implémentation de `AbstractBaseAgent` qui sélectionne uniformément une action parmi l'ensemble des actions légales
disponibles à chaque sollicitation. L'agent ne conserve aucune stratégie ; il constitue une base de référence pour la comparaison de profils d'agents
plus élaborés.

Le module dépend de `agents.interface`, `core.models`, `core.rules_engine` et `core.config`.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

from agents.interface import AbstractBaseAgent
from core.config import GameConfig
from core.models import Action, ActionType, Card, Hand
from core.rules_engine import generate_sequence_plays, generate_uniform_plays
from engine.state import GameState


class RandomBot(AbstractBaseAgent):
    """
    Agent sélectionnant uniformément une action légale.

    Champ `config` : configuration de la partie, utilisée pour déterminer la légalité des suites et le générateur pseudo-aléatoire local.
    Champ `_rng` : générateur pseudo-aléatoire dédié à l'agent.
    """

    def __init__(self, player_id: int, config: GameConfig) -> None:
        super().__init__(player_id)
        self.config = config
        self._rng = random.Random(f"{config.random_seed}:{player_id}")

    def choose_action(self, game_state: GameState) -> Action:
        """
        Sélectionne une action de tour aléatoire parmi les options légales.

        Paramètre `game_state` : vue matérialisée de l'état courant.
        Retourne une instance de `Action`. Si aucune combinaison légale n'est disponible, retourne un passe conforme à `pass_type`. Effet de
        bord : consomme l'état interne du générateur pseudo-aléatoire de l'agent.
        """
        hand = game_state.hands[self.player_id]
        trick = game_state.trick

        required_size = trick.size if trick.size > 0 else None
        min_power = trick.current_power

        options: List[Tuple[Tuple[Card, ...], Optional[int]]] = []
        if not trick.is_sequence:
            options.extend(generate_uniform_plays(hand, game_state.e_rev, required_size, min_power))
        if self.config.straights_enabled and (trick.size == 0 or trick.is_sequence):
            seq_min = trick.sequence_min_power if trick.is_sequence else None
            for cards, joker_map in generate_sequence_plays(hand, game_state.e_rev, required_size, seq_min):
                declared = joker_map[min(joker_map)] if joker_map else None
                options.append((cards, declared))

        if not options:
            action_type = (
                ActionType.ACTION_SOFT_PASS
                if self.config.pass_type == "ALLOW_SOFT"
                else ActionType.ACTION_HARD_PASS
            )
            return Action(action_type=action_type)

        cards, declared_power = self._rng.choice(options)
        return Action(action_type=ActionType.ACTION_PLAY, cards=cards, declared_power=declared_power)

    def choose_exchange_cards(self, hand: Hand, game_state: GameState, count: int) -> List[Card]:
        """
        Sélectionne aléatoirement les cartes cédées lors d'un échange libre.

        Paramètre `hand` : main courante de l'agent.
        Paramètre `game_state` : vue matérialisée de l'état courant.
        Paramètre `count` : nombre de cartes à céder.
        Retourne une liste de `Card` de taille `count`. Effet de bord : consomme l'état interne du générateur pseudo-aléatoire de l'agent.
        """
        return list(self._rng.sample(list(hand.cards), count))

    def ask_putsch(self, hand: Hand) -> bool:
        """
        Décide aléatoirement de l'invocation du Putsch.

        Paramètre `hand` : main courante de l'agent.
        Retourne un booléen tiré uniformément. Effet de bord : consomme l'état interne du générateur pseudo-aléatoire de l'agent.
        """
        return self._rng.random() < 0.5

    def on_interception_opportunity(
        self, game_state: GameState, played_card: Card
    ) -> Tuple[bool, Optional[Card]]:
        """
        Décide aléatoirement de l'interception d'une carte posée.

        Paramètre `game_state` : vue matérialisée de l'état courant.
        Paramètre `played_card` : carte cible de l'interception.
        Retourne un tuple `(decision, card)`. Si une carte jumelle est disponible dans la main de l'agent, la décision est tirée
        uniformément ; sinon la décision est toujours fausse. Effet de bord : consomme l'état interne du générateur pseudo-aléatoire de l'agent.
        """
        hand = game_state.hands[self.player_id]
        twins = [
            c for c in hand.cards
            if not c.is_joker() and c.rank == played_card.rank and c.suit == played_card.suit
        ]
        if not twins:
            return False, None
        if self._rng.random() < 0.5:
            return True, twins[0]
        return False, None
