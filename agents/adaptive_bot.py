"""
Module de l'agent adaptatif au contexte de manche.

Le module définit `AdaptiveBot`, une implémentation de `AbstractBaseAgent` qui ajuste dynamiquement son degré d'agressivité selon la taille
de sa propre main relativement à celle des adversaires encore actifs. Contrairement à `agents.rule_based_bot.RuleBasedBot`, dont les seuils
de préservation sont fixes, `AdaptiveBot` relâche son filtre de réserve dès qu'un retard relatif est détecté (main sensiblement plus grande
que la moyenne des mains adverses actives), et le resserre au contraire lorsqu'il est en position favorable, afin de ne prendre des risques
de dépense de combinaisons de réserve que lorsque la situation l'exige. Le seuil d'invocation du Putsch et l'agressivité d'interception sont
également modulés par ce même indicateur de position relative.

Le module dépend de `agents.interface`, `core.models`, `core.math_utils`, `core.rules_engine`, `core.config` et `engine.state`.
"""

from __future__ import annotations

from collections import Counter
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

# Taille de combinaison à partir de laquelle une pose est considérée comme une réserve de
# puissance, réservée en position favorable ou de parité.
_RESERVE_COMBINATION_SIZE = 4

# Seuil de ratio de position relative (main propre / moyenne des mains adverses actives) en
# deçà duquel l'agent se considère en avance ou à parité, et applique donc son filtre de réserve.
_STANDING_PARITY_THRESHOLD = 1.15

# Seuil de ratio de position relative au-delà duquel l'interception est jugée trop coûteuse.
_INTERCEPTION_STANDING_THRESHOLD = 0.85


class AdaptiveBot(AbstractBaseAgent):
    """
    Agent ajustant ses filtres de préservation selon la position relative de sa main.

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

    def _relative_standing(self, game_state: GameState) -> float:
        """
        Évalue la position relative de la main de l'agent parmi les joueurs encore actifs.

        Paramètre `game_state` : vue matérialisée de l'état courant.
        Retourne un nombre, ratio entre la taille de la main de l'agent et la taille moyenne des mains des autres joueurs encore actifs,
        égal à 1.0 si aucun autre joueur actif n'est disponible pour la comparaison. Une valeur supérieure à 1 indique un retard relatif
        (main plus grande que la moyenne adverse), une valeur inférieure à 1 indique une avance relative. Aucun effet de bord.
        """
        own_size = game_state.hands[self.player_id].size()
        others = [
            hand.size() for pid, hand in game_state.hands.items()
            if pid != self.player_id and not game_state.is_finished.get(pid, False)
        ]
        if not others:
            return 1.0
        average_other = sum(others) / len(others)
        if average_other == 0:
            return 1.0
        return own_size / average_other

    def choose_action(self, game_state: GameState) -> Action:
        """
        Sélectionne une combinaison légale en adaptant les filtres de préservation à la position relative de la main.

        Paramètre `game_state` : vue matérialisée de l'état courant.
        Retourne une instance de `Action`. Le filtre de réserve des combinaisons de taille supérieure ou égale à
        `_RESERVE_COMBINATION_SIZE` n'est appliqué que lorsque la position relative (`_relative_standing`) indique une avance ou une
        parité (ratio inférieur ou égal à `_STANDING_PARITY_THRESHOLD`) ; en situation de retard, l'agent dépense librement ses grandes
        combinaisons pour tenter de rattraper son retard. Retourne un passe conforme à `pass_type` si aucune option n'est disponible.
        Aucun effet de bord.
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

        standing = self._relative_standing(game_state)
        if standing <= _STANDING_PARITY_THRESHOLD:
            conservative = [option for option in candidates if len(option[0]) < _RESERVE_COMBINATION_SIZE]
            if conservative:
                candidates = conservative

        cards, declared_power = min(
            candidates, key=lambda option: self._resulting_power(option, game_state.e_rev)
        )
        return Action(action_type=ActionType.ACTION_PLAY, cards=cards, declared_power=declared_power)

    def choose_exchange_cards(self, hand: Hand, game_state: GameState, count: int) -> List[Card]:
        """
        Sélectionne les cartes cédées lors d'un échange libre en préservant les combinaisons.

        Paramètre `hand` : main courante de l'agent.
        Paramètre `game_state` : vue matérialisée de l'état courant.
        Paramètre `count` : nombre de cartes à céder.
        Retourne une liste de `Card` de taille `count`, cédant en priorité les cartes isolées plutôt que celles appartenant à un groupe de
        puissance de taille supérieure ou égale à deux. Aucun effet de bord.
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
        Invoque le Putsch selon un seuil adapté au nombre de joueurs de la partie.

        Paramètre `hand` : main courante de l'agent.
        Retourne un booléen, vrai si au moins `threshold` cartes de la main partagent une même puissance standard (`threshold` valant 4
        pour une partie de 4 joueurs ou moins, 3 au-delà, la disponibilité de groupes de puissance élevés étant plus fréquente dans les
        parties nombreuses), ou si la puissance maximale de la main hors révolution est inférieure ou égale à dix. Aucun effet de bord.
        """
        powers = [f_power(c, False) for c in hand.cards if not c.is_joker()]
        if not powers:
            return False
        counts = Counter(powers)
        threshold = 4 if self.config.player_count <= 4 else 3
        if any(count >= threshold for count in counts.values()):
            return True
        return max(powers) <= 10

    def on_interception_opportunity(
        self, game_state: GameState, played_card: Card
    ) -> Tuple[bool, Optional[Card]]:
        """
        Intercepte lorsqu'une carte jumelle est disponible et que la position relative ne dissuade pas la dépense.

        Paramètre `game_state` : vue matérialisée de l'état courant.
        Paramètre `played_card` : carte cible de l'interception.
        Retourne un tuple `(decision, card)`. La décision est vraie dès qu'une carte de même rang et de même couleur est disponible dans la
        main de l'agent et que la position relative (`_relative_standing`) est supérieure ou égale à `_INTERCEPTION_STANDING_THRESHOLD`,
        l'agent préservant sa carte jumelle uniquement lorsqu'il est déjà nettement en avance. Aucun effet de bord.
        """
        hand = game_state.hands[self.player_id]
        standing = self._relative_standing(game_state)
        for card in hand.cards:
            if not card.is_joker() and card.rank == played_card.rank and card.suit == played_card.suit:
                return standing >= _INTERCEPTION_STANDING_THRESHOLD, card
        return False, None
