"""
Module de l'agent à politique linéaire entraînable.

Le module définit `RLAgent`, une implémentation de `AbstractBaseAgent` dont la décision de jeu repose sur une politique linéaire appliquée à
un vecteur de caractéristiques par option légale, plutôt que sur une règle fixe. L'agent expose une méthode de sollicitation par lot
(`get_batch_action`) effectuant une unique multiplication matricielle sur l'ensemble des options candidates de plusieurs états simultanés,
conformément à la stratégie d'inférence par lot documentée pour l'accélération GPU. Les poids de la politique sont mutables et destinés à
être ajustés par une boucle d'entraînement externe ; l'agent n'implémente lui-même aucun algorithme d'apprentissage.

Le module dépend de `agents.interface`, `core.models`, `core.math_utils`, `core.rules_engine`, `core.config`, `engine.state` et de `numpy`.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from agents.interface import AbstractBaseAgent
from core.config import GameConfig
from core.math_utils import f_power
from core.models import Action, ActionType, Card, Hand
from core.rules_engine import generate_sequence_plays, generate_uniform_plays
from engine.state import GameState

# Dimension du vecteur de caractéristiques associé à chaque option de jeu candidate.
FEATURE_DIM = 5


def _option_features(
    cards: Tuple[Card, ...],
    declared_power: Optional[int],
    hand_size_before: int,
    e_rev: bool,
) -> np.ndarray:
    """
    Construit le vecteur de caractéristiques d'une option de jeu candidate.

    Paramètre `cards` : combinaison candidate.
    Paramètre `declared_power` : puissance déclarée pour les Jokers présents, ou `None`.
    Paramètre `hand_size_before` : taille de la main avant la pose, entier strictement positif.
    Paramètre `e_rev` : état de révolution courant.
    Retourne un tableau `numpy.ndarray` de type `float64` et de taille `FEATURE_DIM`, composé dans l'ordre de la puissance résultante
    normalisée sur $[0, 1]$, de la taille de la combinaison normalisée par la taille de la main répétée deux fois, d'un indicateur binaire
    de présence de Joker, et d'un terme constant de biais. Aucun effet de bord.
    """
    non_jokers = [c for c in cards if not c.is_joker()]
    power = f_power(non_jokers[0], e_rev) if non_jokers else (declared_power or 0)
    contains_joker = 1.0 if len(non_jokers) != len(cards) else 0.0
    size = len(cards)
    size_ratio = size / max(hand_size_before, 1)
    return np.array(
        [power / 16.0, size_ratio, size_ratio, contains_joker, 1.0],
        dtype=np.float64,
    )


class RLAgent(AbstractBaseAgent):
    """
    Agent dont la décision de jeu repose sur une politique linéaire entraînable.

    Champ `config` : configuration de la partie.
    Champ `weights` : vecteur `numpy.ndarray` de poids de la politique, taille `FEATURE_DIM`, mutable, ajustable par un processus
    d'entraînement externe.
    Champ `epsilon` : probabilité d'exploration aléatoire lors du choix d'une action, domaine $[0, 1]$.
    Champ `_rng` : générateur pseudo-aléatoire dédié à l'exploration et aux décisions par défaut.
    """

    def __init__(
        self,
        player_id: int,
        config: GameConfig,
        weights: Optional[np.ndarray] = None,
        epsilon: float = 0.1,
    ) -> None:
        super().__init__(player_id)
        self.config = config
        self.weights = weights if weights is not None else np.zeros(FEATURE_DIM, dtype=np.float64)
        self.epsilon = epsilon
        self._rng = random.Random(f"{config.random_seed}:{player_id}:rl")

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

    def _default_pass(self) -> Action:
        """
        Construit l'action de passe conforme à la sémantique active.

        Retourne une instance de `Action` de type `ACTION_SOFT_PASS` si `pass_type` vaut `'ALLOW_SOFT'`, `ACTION_HARD_PASS` sinon. Aucun
        effet de bord.
        """
        action_type = (
            ActionType.ACTION_SOFT_PASS
            if self.config.pass_type == "ALLOW_SOFT"
            else ActionType.ACTION_HARD_PASS
        )
        return Action(action_type=action_type)

    def choose_action(self, game_state: GameState) -> Action:
        """
        Sélectionne une action de tour par évaluation linéaire des options légales.

        Paramètre `game_state` : vue matérialisée de l'état courant.
        Retourne une instance de `Action`. Avec une probabilité `epsilon`, une option légale est choisie uniformément (exploration) ;
        sinon, l'option de score `weights . features` maximal est choisie (exploitation). Retourne un passe conforme à `pass_type` si aucune
        option n'est disponible. Effet de bord : consomme l'état interne du générateur pseudo-aléatoire de l'agent.
        """
        hand = game_state.hands[self.player_id]
        options = self._legal_options(hand, game_state)
        if not options:
            return self._default_pass()

        if self._rng.random() < self.epsilon:
            cards, declared_power = self._rng.choice(options)
            return Action(action_type=ActionType.ACTION_PLAY, cards=cards, declared_power=declared_power)

        hand_size = hand.size()
        feature_matrix = np.stack(
            [_option_features(cards, declared, hand_size, game_state.e_rev) for cards, declared in options]
        )
        scores = feature_matrix @ self.weights
        best_index = int(np.argmax(scores))
        cards, declared_power = options[best_index]
        return Action(action_type=ActionType.ACTION_PLAY, cards=cards, declared_power=declared_power)

    def get_batch_action(self, game_states: List[GameState]) -> List[Action]:
        """
        Sélectionne une action de tour pour un lot d'états simultanés par une unique évaluation matricielle.

        Paramètre `game_states` : liste de vues matérialisées de l'état courant, une par simulation parallèle.
        Retourne une liste de `Action` de taille identique à `game_states`, dans le même ordre. Effet de bord : consomme l'état interne du
        générateur pseudo-aléatoire de l'agent pour chaque état sans option légale ou sujet à exploration. Complexité : une unique
        multiplication matricielle sur l'ensemble des options candidates de tous les états, plutôt qu'une évaluation séquentielle par état.
        """
        per_state_options: List[List[Tuple[Tuple[Card, ...], Optional[int]]]] = []
        all_features: List[np.ndarray] = []
        owner_index: List[int] = []
        start_offsets: List[int] = []

        for state_index, game_state in enumerate(game_states):
            hand = game_state.hands[self.player_id]
            options = self._legal_options(hand, game_state)
            start_offsets.append(len(owner_index))
            per_state_options.append(options)
            hand_size = hand.size()
            for cards, declared in options:
                all_features.append(_option_features(cards, declared, hand_size, game_state.e_rev))
                owner_index.append(state_index)

        scores = np.zeros(0, dtype=np.float64)
        if all_features:
            feature_matrix = np.stack(all_features)
            scores = feature_matrix @ self.weights

        best_score_by_state: Dict[int, float] = {}
        best_flat_index_by_state: Dict[int, int] = {}
        for flat_index, state_index in enumerate(owner_index):
            score = float(scores[flat_index])
            if state_index not in best_score_by_state or score > best_score_by_state[state_index]:
                best_score_by_state[state_index] = score
                best_flat_index_by_state[state_index] = flat_index

        results: List[Action] = []
        for state_index, options in enumerate(per_state_options):
            if not options:
                results.append(self._default_pass())
                continue
            if self._rng.random() < self.epsilon:
                cards, declared_power = self._rng.choice(options)
                results.append(Action(action_type=ActionType.ACTION_PLAY, cards=cards, declared_power=declared_power))
                continue
            flat_index = best_flat_index_by_state[state_index]
            local_offset = flat_index - start_offsets[state_index]
            cards, declared_power = options[local_offset]
            results.append(Action(action_type=ActionType.ACTION_PLAY, cards=cards, declared_power=declared_power))
        return results

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
        Intercepte lorsqu'une carte jumelle est disponible, avec une probabilité d'exploration.

        Paramètre `game_state` : vue matérialisée de l'état courant.
        Paramètre `played_card` : carte cible de l'interception.
        Retourne un tuple `(decision, card)`. Si une carte jumelle est disponible, la décision d'intercepter est prise avec une probabilité
        `1 - epsilon`. Effet de bord : consomme l'état interne du générateur pseudo-aléatoire de l'agent.
        """
        hand = game_state.hands[self.player_id]
        twins = [
            c for c in hand.cards
            if not c.is_joker() and c.rank == played_card.rank and c.suit == played_card.suit
        ]
        if not twins:
            return False, None
        decision = self._rng.random() >= self.epsilon
        return decision, (twins[0] if decision else None)
