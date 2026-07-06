"""
Module de l'agent à anticipation heuristique sur la structure de main.

Le module définit `LookaheadBot`, une implémentation de `AbstractBaseAgent` qui étend les filtres de préservation de
`agents.rule_based_bot.RuleBasedBot` d'une anticipation supplémentaire : parmi les combinaisons restant candidates après ces filtres,
l'agent estime pour chacune la flexibilité de la main résiduelle qu'elle laisserait derrière elle (nombre de combinaisons légales encore
disponibles une fois la combinaison retirée), et retient celle qui préserve le plus grand nombre d'options futures à puissance résultante
égale. Cette anticipation reste locale à la propre main de l'agent, sans simulation des adversaires, ce qui la rend d'un coût quasi identique
à celui de `RuleBasedBot` tout en réduisant le risque de s'enfermer dans une main sans option légale en fin de manche.

Le module dépend de `agents.interface`, `core.models`, `core.math_utils`, `core.rules_engine`, `core.config` et `engine.state`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from agents.interface import AbstractBaseAgent
from core.config import GameConfig
from core.math_utils import f_power
from core.models import Action, ActionType, Card, Hand
from core.rules_engine import (
    generate_sequence_plays, generate_uniform_plays, matches_finish_penalty,
    triggers_revolution,
)
from engine.state import GameState

# Taille de main en deçà de laquelle la préservation des combinaisons de taille élevée n'est
# plus jugée prioritaire, cohérente avec le seuil utilisé par `RuleBasedBot`.
_ENDGAME_HAND_SIZE = 4

# Taille de combinaison à partir de laquelle une pose est considérée comme une réserve de
# puissance à préserver.
_RESERVE_COMBINATION_SIZE = 4

# Nombre maximal de candidats évalués par anticipation de flexibilité résiduelle, afin de
# borner le coût de la décision lorsque de nombreuses options partagent la puissance minimale.
_MAX_LOOKAHEAD_CANDIDATES = 10


class LookaheadBot(AbstractBaseAgent):
    """
    Agent combinant les filtres de préservation de `RuleBasedBot` à une anticipation locale de flexibilité de main.

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

    def _residual_flexibility(self, hand: Hand, played_cards: Tuple[Card, ...], e_rev: bool) -> int:
        """
        Estime le nombre de combinaisons légales encore disponibles après retrait d'une pose candidate.

        Paramètre `hand` : main avant la pose.
        Paramètre `played_cards` : cartes de la pose candidate, sous-ensemble de `hand`.
        Paramètre `e_rev` : état de révolution courant.
        Retourne un entier positif ou nul, somme du nombre d'options uniformes et, si `straights_enabled`, de suites disponibles dans la
        main résiduelle pour une ouverture de pli libre. Aucun effet de bord.
        """
        residual = hand.without(played_cards)
        count = len(generate_uniform_plays(residual, e_rev, None, None))
        if self.config.straights_enabled:
            count += len(generate_sequence_plays(residual, e_rev, None, None))
        return count

    def choose_action(self, game_state: GameState) -> Action:
        """
        Sélectionne une combinaison légale par filtrage de préservation puis anticipation de flexibilité résiduelle.

        Paramètre `game_state` : vue matérialisée de l'état courant.
        Retourne une instance de `Action`. Applique d'abord les mêmes filtres de préservation que `RuleBasedBot` (évitement de la pénalité
        de sortie étendue, réserve des combinaisons de taille supérieure ou égale à `_RESERVE_COMBINATION_SIZE`), puis, parmi les options de
        puissance résultante minimale à égalité, retient celle laissant la main résiduelle la plus flexible. Retourne un passe conforme à
        `pass_type` si aucune option n'est disponible. Aucun effet de bord.
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

        candidates = options

        if self.config.finish_penalty_extended:
            non_penalizing = [
                option for option in candidates
                if len(option[0]) != hand.size()
                or not matches_finish_penalty(
                    option[0],
                    self.config,
                    game_state.e_rev,
                    triggers_revolution(option[0], self.config, False),
                )
            ]
            if non_penalizing:
                candidates = non_penalizing

        if hand.size() > _ENDGAME_HAND_SIZE:
            conservative = [option for option in candidates if len(option[0]) < _RESERVE_COMBINATION_SIZE]
            if conservative:
                candidates = conservative

        min_power = min(self._resulting_power(option, game_state.e_rev) for option in candidates)
        near_minimal = [
            option for option in candidates
            if self._resulting_power(option, game_state.e_rev) == min_power
        ]

        if len(near_minimal) == 1:
            cards, declared_power = near_minimal[0]
            return Action(action_type=ActionType.ACTION_PLAY, cards=cards, declared_power=declared_power)

        evaluated = near_minimal[:_MAX_LOOKAHEAD_CANDIDATES]
        cards, declared_power = max(
            evaluated,
            key=lambda option: self._residual_flexibility(hand, option[0], game_state.e_rev),
        )
        return Action(action_type=ActionType.ACTION_PLAY, cards=cards, declared_power=declared_power)

    def choose_exchange_cards(self, hand: Hand, game_state: GameState, count: int) -> List[Card]:
        """
        Sélectionne les cartes cédées lors d'un échange libre en préservant les combinaisons.

        Paramètre `hand` : main courante de l'agent.
        Paramètre `game_state` : vue matérialisée de l'état courant.
        Paramètre `count` : nombre de cartes à céder.
        Retourne une liste de `Card` de taille `count`, cédant en priorité les cartes isolées plutôt que celles appartenant à un groupe de
        puissance de taille supérieure ou égale à deux, puis triée par puissance croissante à effectif de groupe égal. Aucun effet de bord.
        """
        power_groups: Dict[int, List[Card]] = {}
        for card in hand.cards:
            if card.is_joker():
                continue
            power_groups.setdefault(f_power(card, game_state.e_rev), []).append(card)

        ordered = sorted(
            (c for c in hand.cards if not c.is_joker()),
            key=lambda c: (len(power_groups[f_power(c, game_state.e_rev)]), f_power(c, game_state.e_rev)),
        )
        jokers = [c for c in hand.cards if c.is_joker()]
        return (ordered + jokers)[:count]

    def ask_putsch(self, hand: Hand) -> bool:
        """
        Invoque le Putsch selon la condition mathématique standard.

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
        Intercepte lorsqu'une carte jumelle est disponible et que la main n'est pas en fin de manche.

        Paramètre `game_state` : vue matérialisée de l'état courant.
        Paramètre `played_card` : carte cible de l'interception.
        Retourne un tuple `(decision, card)`. La décision est vraie dès qu'une carte de même rang et de même couleur est disponible dans la
        main de l'agent et que la taille de la main est strictement supérieure à `_ENDGAME_HAND_SIZE`. Aucun effet de bord.
        """
        hand = game_state.hands[self.player_id]
        if hand.size() <= _ENDGAME_HAND_SIZE:
            return False, None
        for card in hand.cards:
            if not card.is_joker() and card.rank == played_card.rank and card.suit == played_card.suit:
                return True, card
        return False, None
