"""
Module de l'agent probabiliste par comptage de cartes.

Le module définit `ProbabilisticBot`, une implémentation de `AbstractBaseAgent` qui évalue chaque combinaison légale candidate par une
estimation de risque fondée sur le dénombrement des cartes de puissance strictement supérieure encore susceptibles de se trouver hors de sa
propre main, au regard de la composition totale du paquet dérivée de `core.config.GameConfig` (nombre de paquets effectifs, présence de
Jokers) et de l'état de révolution courant. Contrairement aux agents à filtres déterministes (`agents.rule_based_bot.RuleBasedBot`) ou à
score composite fixe (`agents.scoring_bot.ScoringBot`), qui ne raisonnent que sur la structure de la main propre, `ProbabilisticBot` estime
une fraction de cartes adverses potentiellement menaçantes encore en circulation, produisant un score de risque continu combiné à une prime
de taille encourageant la dépense de grandes combinaisons lorsque le risque estimé reste comparable.

Le module dépend de `agents.interface`, `core.models`, `core.math_utils`, `core.rules_engine` et `core.config`.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Tuple

from agents.interface import AbstractBaseAgent
from core.config import GameConfig
from core.math_utils import f_power
from core.models import Action, ActionType, Card, Hand, RANK_ORDER, Rank, Suit
from core.rules_engine import generate_sequence_plays, generate_uniform_plays, num_decks
from engine.state import GameState

# Poids relatif de la prime de taille (encouragement à se débarrasser de grandes combinaisons à risque comparable) dans le score composite.
_SIZE_BONUS_WEIGHT = 0.30


def _all_power_levels(e_rev: bool) -> List[int]:
    """
    Énumère l'ensemble des valeurs de puissance distinctes atteignables par un rang facial.

    Paramètre `e_rev` : état de révolution utilisé pour le calcul de puissance de chaque rang.
    Retourne une liste d'entiers, une valeur de puissance par rang facial distinct (Joker compris), sans répétition. Aucun effet de bord.
    """
    levels = set()
    for rank_value in RANK_ORDER:
        if rank_value == "JOKER":
            probe = Card(rank=Rank.JOKER, suit=Suit.NONE)
        else:
            probe = Card(rank=Rank(rank_value), suit=Suit.SPADES)
        levels.add(f_power(probe, e_rev))
    return sorted(levels)


class ProbabilisticBot(AbstractBaseAgent):
    """
    Agent évaluant chaque option légale par une estimation de risque fondée sur le comptage des cartes restantes.

    Champ `config` : configuration de la partie, utilisée pour dériver la composition totale du paquet
    (`core.rules_engine.num_decks`, `use_jokers`).
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

    def _total_copies_at_power(self, power: int, e_rev: bool) -> int:
        """
        Dénombre le nombre total d'exemplaires du paquet complet partageant une puissance donnée.

        Paramètre `power` : valeur de puissance considérée, domaine $[3, 16]$.
        Paramètre `e_rev` : état de révolution courant, utilisé pour résoudre le ou les rangs correspondant à `power`.
        Retourne un entier positif ou nul, somme sur l'ensemble des paquets effectifs (`core.rules_engine.num_decks`) du nombre d'exemplaires
        de chaque rang facial dont la puissance dynamique vaut `power`, Jokers compris si `use_jokers` est vrai. Aucun effet de bord.
        """
        decks = num_decks(self.config)
        if power == 16:
            return 2 * decks if self.config.use_jokers else 0
        total = 0
        for rank_value in RANK_ORDER:
            if rank_value == "JOKER":
                continue
            probe = Card(rank=Rank(rank_value), suit=Suit.SPADES)
            if f_power(probe, e_rev) == power:
                total += 4 * decks
        return total

    def _outstanding_same_power_count(self, hand: Hand, power: int, e_rev: bool) -> int:
        """
        Dénombre les cartes partageant une puissance donnée encore hors de la main courante.

        Paramètre `hand` : main courante de l'agent.
        Paramètre `power` : puissance considérée.
        Paramètre `e_rev` : état de révolution courant.
        Retourne un entier positif ou nul, égal au nombre total d'exemplaires du paquet à cette puissance diminué du nombre d'exemplaires
        déjà présents dans `hand`. Aucun effet de bord.
        """
        hand_counts = Counter(f_power(c, e_rev) for c in hand.cards)
        total_at_level = self._total_copies_at_power(power, e_rev)
        return max(0, total_at_level - hand_counts.get(power, 0))

    def _outstanding_higher_power_count(self, hand: Hand, power: int, e_rev: bool) -> int:
        """
        Estime le nombre de cartes de puissance strictement supérieure encore hors de la main courante.

        Paramètre `hand` : main courante de l'agent.
        Paramètre `power` : puissance de référence, seules les puissances strictement supérieures sont comptées.
        Paramètre `e_rev` : état de révolution courant.
        Retourne un entier positif ou nul, somme sur chaque niveau de puissance strictement supérieur à `power` du résultat de
        `_outstanding_same_power_count` à ce niveau. Aucun effet de bord.
        """
        outstanding = 0
        for level in _all_power_levels(e_rev):
            if level <= power:
                continue
            outstanding += self._outstanding_same_power_count(hand, level, e_rev)
        return outstanding

    def _risk_score(self, option: Tuple[Tuple[Card, ...], Optional[int]], hand: Hand, e_rev: bool) -> float:
        """
        Calcule le score de risque composite d'une option candidate.

        Paramètre `option` : tuple `(cards, declared_power)` candidat.
        Paramètre `hand` : main avant la pose.
        Paramètre `e_rev` : état de révolution courant.
        Retourne un nombre, égal à la fraction estimée de cartes hors main de puissance strictement supérieure à la puissance résultante de
        `option` parmi l'ensemble des cartes non détenues par l'agent, diminuée d'une prime proportionnelle à la taille de la combinaison
        relativement à la taille de la main, pondérée par `_SIZE_BONUS_WEIGHT`. Un score plus bas est préféré. Aucun effet de bord.
        """
        power = self._resulting_power(option, e_rev)
        outstanding = self._outstanding_higher_power_count(hand, power, e_rev)
        decks = num_decks(self.config)
        total_deck_cards = decks * (52 + (2 if self.config.use_jokers else 0))
        total_unseen = max(1, total_deck_cards - hand.size())
        risk = outstanding / total_unseen
        size_bonus = len(option[0]) / max(hand.size(), 1)
        return risk - _SIZE_BONUS_WEIGHT * size_bonus

    def choose_action(self, game_state: GameState) -> Action:
        """
        Sélectionne l'option légale de score de risque minimal.

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
            options, key=lambda option: self._risk_score(option, hand, game_state.e_rev)
        )
        return Action(action_type=ActionType.ACTION_PLAY, cards=cards, declared_power=declared_power)

    def choose_exchange_cards(self, hand: Hand, game_state: GameState, count: int) -> List[Card]:
        """
        Sélectionne les cartes cédées lors d'un échange libre en préservant les puissances rares.

        Paramètre `hand` : main courante de l'agent.
        Paramètre `game_state` : vue matérialisée de l'état courant.
        Paramètre `count` : nombre de cartes à céder.
        Retourne une liste de `Card` de taille `count`, cédant en priorité les cartes de puissance faible et abondante, en s'appuyant sur le
        même dénombrement d'exemplaires totaux que `_risk_score`. Aucun effet de bord.
        """
        e_rev = game_state.e_rev
        ordered = sorted(
            (c for c in hand.cards if not c.is_joker()),
            key=lambda c: (f_power(c, e_rev), -self._total_copies_at_power(f_power(c, e_rev), e_rev)),
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
        Intercepte lorsqu'une carte jumelle est disponible et que peu d'exemplaires équivalents restent en circulation.

        Paramètre `game_state` : vue matérialisée de l'état courant.
        Paramètre `played_card` : carte cible de l'interception.
        Retourne un tuple `(decision, card)`. La décision est vraie dès qu'une carte de même rang et de même couleur est disponible dans la
        main de l'agent et que le nombre d'exemplaires de cette puissance encore hors de la main (`_outstanding_same_power_count`) est
        inférieur ou égal à un, situation dans laquelle intercepter ne prive pas l'agent d'une réserve encore reconstituable. Aucun effet de
        bord.
        """
        hand = game_state.hands[self.player_id]
        e_rev = game_state.e_rev
        for card in hand.cards:
            if not card.is_joker() and card.rank == played_card.rank and card.suit == played_card.suit:
                power = f_power(card, e_rev)
                outstanding = self._outstanding_same_power_count(hand, power, e_rev)
                return outstanding <= 1, card
        return False, None
