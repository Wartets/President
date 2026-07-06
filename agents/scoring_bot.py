"""
Module de l'agent à score pondéré déterministe.

Le module définit `ScoringBot`, une implémentation de `AbstractBaseAgent` qui évalue chaque option légale par une combinaison pondérée de
trois critères déterministes plutôt qu'un unique critère de puissance minimale : la puissance résultante normalisée, le coût relatif de la
combinaison sur la taille de main restante, et une pénalité de rareté qui décourage de dépenser une carte dont peu d'exemplaires équivalents
restent en main. Contrairement à `agents.rule_based_bot.RuleBasedBot`, qui applique des filtres de préservation successifs puis retombe sur
la puissance minimale, `ScoringBot` calcule un score unique par option et retient l'option de score le plus bas, ce qui produit un compromis
continu entre les mêmes objectifs plutôt qu'une cascade de filtres discrets.

Le module dépend de `agents.interface`, `core.models`, `core.math_utils`, `core.rules_engine` et `core.config`.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

from agents.interface import AbstractBaseAgent
from core.config import GameConfig
from core.math_utils import f_power
from core.models import Action, ActionType, Card, Hand
from core.rules_engine import generate_sequence_plays, generate_uniform_plays
from engine.state import GameState

# Poids relatif du coût de taille de combinaison dans le score composite.
_SIZE_COST_WEIGHT = 0.35

# Poids relatif de la pénalité de rareté (dépense d'une carte dont peu d'exemplaires de puissance équivalente restent en main) dans le score composite.
_SCARCITY_WEIGHT = 0.25


class ScoringBot(AbstractBaseAgent):
    """
    Agent sélectionnant l'option légale de score composite minimal parmi trois critères pondérés.

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

    def _scarcity_penalty(
        self, option: Tuple[Tuple[Card, ...], Optional[int]], hand: Hand, e_rev: bool,
    ) -> float:
        """
        Évalue la rareté de la puissance dépensée par une option candidate.

        Paramètre `option` : tuple `(cards, declared_power)` candidat.
        Paramètre `hand` : main avant la pose.
        Paramètre `e_rev` : état de révolution courant.
        Retourne un nombre, domaine $[0, 1]$, égal à l'inverse du nombre de cartes non Joker de `hand` partageant la puissance résultante de
        `option`, nul si la combinaison ne contient que des Jokers. Une valeur élevée signale la dépense d'une puissance peu représentée dans
        la main, donc difficile à reconstituer pour une future combinaison de réserve. Aucun effet de bord.
        """
        power = self._resulting_power(option, e_rev)
        matching = sum(1 for c in hand.cards if not c.is_joker() and f_power(c, e_rev) == power)
        if matching == 0:
            return 0.0
        return 1.0 / matching

    def _composite_score(
        self, option: Tuple[Tuple[Card, ...], Optional[int]], hand: Hand, e_rev: bool,
    ) -> float:
        """
        Calcule le score composite d'une option candidate.

        Paramètre `option` : tuple `(cards, declared_power)` candidat.
        Paramètre `hand` : main avant la pose.
        Paramètre `e_rev` : état de révolution courant.
        Retourne un nombre, somme de la puissance résultante normalisée sur $[0, 1]$, du coût de taille normalisé par la taille de la main
        pondéré par `_SIZE_COST_WEIGHT`, et de la pénalité de rareté pondérée par `_SCARCITY_WEIGHT`. Un score plus bas est préféré. Aucun
        effet de bord.
        """
        cards, _declared = option
        power = self._resulting_power(option, e_rev)
        hand_size = max(hand.size(), 1)
        size_cost = len(cards) / hand_size
        scarcity = self._scarcity_penalty(option, hand, e_rev)
        return (power / 16.0) + _SIZE_COST_WEIGHT * size_cost + _SCARCITY_WEIGHT * scarcity

    def choose_action(self, game_state: GameState) -> Action:
        """
        Sélectionne l'option légale de score composite minimal.

        Paramètre `game_state` : vue matérialisée de l'état courant.
        Retourne une instance de `Action`. Retourne un passe conforme à `pass_type` si aucune option n'est disponible. Aucun effet de bord.
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

        cards, declared_power = min(
            options, key=lambda option: self._composite_score(option, hand, game_state.e_rev)
        )
        return Action(action_type=ActionType.ACTION_PLAY, cards=cards, declared_power=declared_power)

    def choose_exchange_cards(self, hand: Hand, game_state: GameState, count: int) -> List[Card]:
        """
        Sélectionne les cartes cédées lors d'un échange libre en préservant les groupes de puissance rares.

        Paramètre `hand` : main courante de l'agent.
        Paramètre `game_state` : vue matérialisée de l'état courant.
        Paramètre `count` : nombre de cartes à céder.
        Retourne une liste de `Card` de taille `count`, cédant en priorité les cartes de puissance élevée appartenant à un groupe de taille
        supérieure ou égale à deux, les groupes de taille un (rares) étant cédés en dernier recours. Aucun effet de bord.
        """
        power_groups: Dict[int, List[Card]] = {}
        for card in hand.cards:
            if card.is_joker():
                continue
            power_groups.setdefault(f_power(card, game_state.e_rev), []).append(card)

        ordered = sorted(
            (c for c in hand.cards if not c.is_joker()),
            key=lambda c: (len(power_groups[f_power(c, game_state.e_rev)]) == 1, -f_power(c, game_state.e_rev)),
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
        Intercepte lorsqu'une carte jumelle est disponible et que sa rareté résiduelle le justifie.

        Paramètre `game_state` : vue matérialisée de l'état courant.
        Paramètre `played_card` : carte cible de l'interception.
        Retourne un tuple `(decision, card)`. La décision est vraie dès qu'une carte de même rang et de même couleur est disponible dans la
        main de l'agent et que la main compte au moins trois cartes, l'agent préservant sa carte jumelle en main critique. Aucun effet de
        bord.
        """
        hand = game_state.hands[self.player_id]
        if hand.size() < 3:
            return False, None
        for card in hand.cards:
            if not card.is_joker() and card.rank == played_card.rank and card.suit == played_card.suit:
                return True, card
        return False, None
