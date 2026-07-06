"""
Module de l'agent agressif.

Le module définit `AggressiveBot`, une implémentation de `AbstractBaseAgent` qui joue systématiquement la combinaison légale de puissance
résultante la plus élevée parmi les options disponibles, à l'inverse de `agents.greedy_bot.GreedyBot` qui minimise cette même puissance.
Cette stratégie vide la main plus rapidement de ses combinaisons de forte puissance mais expose l'agent au risque de ne plus disposer de
combinaison de secours en fin de manche ; l'agent constitue un profil de comparaison utile pour étudier l'effet d'une politique de dépense
maximale plutôt que minimale sur le taux de victoire et le facteur de branchement.

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


class AggressiveBot(AbstractBaseAgent):
    """
    Agent jouant toujours la combinaison de puissance résultante la plus élevée disponible.

    Champ `config` : configuration de la partie.
    """

    def __init__(self, player_id: int, config: GameConfig) -> None:
        super().__init__(player_id)
        self.config = config

    def _legal_options(self, hand: Hand, game_state: GameState) -> List[Tuple[Tuple[Card, ...], Optional[int]]]:
        """
        Rassemble l'ensemble des combinaisons légales disponibles pour la main courante.

        Paramètre `hand` : main considérée.
        Paramètre `game_state` : vue matérialisée de l'état courant.
        Retourne une liste de tuples `(cards, declared_power)`. Aucun effet de bord.
        """
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
        return options

    def _resulting_power(self, option: Tuple[Tuple[Card, ...], Optional[int]], e_rev: bool) -> int:
        """
        Calcule la puissance résultante approchée d'une option de jeu.

        Paramètre `option` : tuple `(cards, declared_power)` candidat.
        Paramètre `e_rev` : état de révolution courant.
        Retourne un entier, puissance de la première carte non Joker de la combinaison, ou `declared_power` si la combinaison n'est composée
        que de Jokers. Aucun effet de bord.
        """
        cards, declared = option
        non_jokers = [c for c in cards if not c.is_joker()]
        if non_jokers:
            return f_power(non_jokers[0], e_rev)
        return declared if declared is not None else 0

    def choose_action(self, game_state: GameState) -> Action:
        """
        Sélectionne la combinaison légale de puissance résultante maximale disponible.

        Paramètre `game_state` : vue matérialisée de l'état courant. Retourne une instance de `Action`. Si aucune combinaison légale
        n'est disponible, retourne un passe conforme à `pass_type`. Aucun effet de bord.
        """
        hand = game_state.hands[self.player_id]
        options = self._legal_options(hand, game_state)

        if not options:
            action_type = (
                ActionType.ACTION_SOFT_PASS
                if self.config.pass_type == "ALLOW_SOFT"
                else ActionType.ACTION_HARD_PASS
            )
            return Action(action_type=action_type)

        cards, declared_power = max(
            options, key=lambda option: self._resulting_power(option, game_state.e_rev)
        )
        return Action(action_type=ActionType.ACTION_PLAY, cards=cards, declared_power=declared_power)

    def choose_exchange_cards(self, hand: Hand, game_state: GameState, count: int) -> List[Card]:
        """
        Sélectionne les cartes de puissance la plus faible lors d'un échange.

        Paramètre `hand` : main courante de l'agent.
        Paramètre `game_state` : vue matérialisée de l'état courant.
        Paramètre `count` : nombre de cartes à céder.
        Retourne une liste de `Card` de taille `count`, triée par puissance croissante, préservant ainsi les cartes fortes que l'agent
        cherche systématiquement à dépenser en priorité pendant la phase de jeu. Aucun effet de bord.
        """
        ordered = sorted(hand.cards, key=lambda c: f_power(c, game_state.e_rev))
        return ordered[:count]

    def ask_putsch(self, hand: Hand) -> bool:
        """
        Invoque systématiquement le Putsch lorsque la condition mathématique standard est satisfaite.

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
