"""
Module de l'agent glouton.

Le module définit `GreedyBot`, une implémentation de `AbstractBaseAgent` qui joue systématiquement la combinaison légale de puissance la plus basse
suffisante pour dépasser le pli courant, et cède les cartes de puissance la plus faible lors d'un échange libre. L'agent invoque le Putsch dès que la
condition mathématique associée est satisfaite et intercepte systématiquement lorsqu'une carte jumelle est disponible.

Le module dépend de `agents.interface`, `core.models`, `core.rules_engine` et `core.config`.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from agents.interface import AbstractBaseAgent
from core.config import GameConfig
from core.math_utils import f_power
from core.models import Action, ActionType, Card, Hand
from core.rules_engine import generate_sequence_plays, generate_uniform_plays
from engine.state import GameState


class GreedyBot(AbstractBaseAgent):
    """
    Agent jouant toujours la combinaison la plus basse suffisante.

    Champ `config` : configuration de la partie.
    """

    def __init__(self, player_id: int, config: GameConfig) -> None:
        super().__init__(player_id)
        self.config = config

    def choose_action(self, game_state: GameState) -> Action:
        """
        Sélectionne la combinaison légale de puissance minimale disponible.

        Paramètre `game_state` : vue matérialisée de l'état courant. Retourne une instance de `Action`. Si aucune combinaison légale
        n'est disponible, retourne un passe conforme à `pass_type`. Aucun effet de bord.
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

        def resulting_power(option: Tuple[Tuple[Card, ...], Optional[int]]) -> int:
            cards, declared = option
            non_jokers = [c for c in cards if not c.is_joker()]
            if non_jokers:
                return f_power(non_jokers[0], game_state.e_rev)
            return declared if declared is not None else 0

        cards, declared_power = min(options, key=resulting_power)
        return Action(action_type=ActionType.ACTION_PLAY, cards=cards, declared_power=declared_power)

    def choose_exchange_cards(self, hand: Hand, game_state: GameState, count: int) -> List[Card]:
        """
        Sélectionne les cartes de puissance la plus faible lors d'un échange.

        Paramètre `hand` : main courante de l'agent.
        Paramètre `game_state` : vue matérialisée de l'état courant.
        Paramètre `count` : nombre de cartes à céder.
        Retourne une liste de `Card` de taille `count`, triée par puissance croissante. Aucun effet de bord.
        """
        ordered = sorted(hand.cards, key=lambda c: f_power(c, game_state.e_rev))
        return ordered[:count]

    def ask_putsch(self, hand: Hand) -> bool:
        """
        Invoque systématiquement le Putsch lorsque la condition est favorable.

        Paramètre `hand` : main courante de l'agent.
        Retourne un booléen, vrai si au moins quatre cartes de la main partagent une même puissance standard ou si la puissance maximale de
        la main hors révolution est inférieure ou égale à dix. Aucun effet de bord.
        """
        from collections import Counter

        powers = [f_power(c, False) for c in hand.cards if not c.is_joker()]
        if not powers:
            return False
        counts = Counter(powers)
        if any(count >= 4 for count in counts.values()):
            return True
        return max(powers) <= 10

    def on_interception_opportunity(
        self, game_state: GameState, played_card: Card
    ) -> Tuple[bool, Optional[Card]]:
        """
        Intercepte systématiquement lorsqu'une carte jumelle est disponible.

        Paramètre `game_state` : vue matérialisée de l'état courant.
        Paramètre `played_card` : carte cible de l'interception.
        Retourne un tuple `(decision, card)`, la décision étant vraie dès qu'une carte de même rang et de même couleur est présente dans la
        main de l'agent. Aucun effet de bord.
        """
        hand = game_state.hands[self.player_id]
        for card in hand.cards:
            if not card.is_joker() and card.rank == played_card.rank and card.suit == played_card.suit:
                return True, card
        return False, None
