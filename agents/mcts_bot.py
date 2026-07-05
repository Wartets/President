"""
Module de l'agent Monte-Carlo par simulation de rollouts.

Le module définit `MCTSBot`, une implémentation de `AbstractBaseAgent` évaluant chaque option légale par un ensemble de simulations de fin
de manche jouées par des agents de référence déterministes, et sélectionnant l'option dont le taux de victoire moyen estimé est le plus
élevé. L'agent ne modifie jamais l'état réel de la partie ; toute simulation opère sur une copie profonde de la vue matérialisée courante.

Le module dépend de `agents.interface`, `agents.greedy_bot`, `core.models`, `core.config`, `core.rules_engine`, `engine.state` et de la
bibliothèque standard `copy` et `random`.
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Tuple

from agents.greedy_bot import GreedyBot
from agents.interface import AbstractBaseAgent
from core.config import GameConfig
from core.math_utils import f_power
from core.models import Action, ActionType, Card, Hand
from core.rules_engine import generate_sequence_plays, generate_uniform_plays
from engine.state import GameState

# Nombre de rollouts simulés par option candidate évaluée par l'agent.
_DEFAULT_ROLLOUT_COUNT = 24

# Nombre maximal de demi-coups simulés par rollout avant arrêt forcé.
_MAX_ROLLOUT_STEPS = 400


class MCTSBot(AbstractBaseAgent):
    """
    Agent sélectionnant l'option légale de taux de victoire simulé maximal.

    Champ `config` : configuration de la partie.
    Champ `rollout_count` : nombre de simulations de rollout effectuées par option candidate, entier strictement positif.
    Champ `_rng` : générateur pseudo-aléatoire dédié à l'agent et à ses rollouts internes.
    """

    def __init__(
        self,
        player_id: int,
        config: GameConfig,
        rollout_count: int = _DEFAULT_ROLLOUT_COUNT,
    ) -> None:
        super().__init__(player_id)
        self.config = config
        self.rollout_count = rollout_count
        self._rng = random.Random(f"{config.random_seed}:{player_id}:mcts")
        self._rollout_agents: Dict[int, AbstractBaseAgent] = {}

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

    def _rollout_agent_for(self, pid: int) -> AbstractBaseAgent:
        """
        Fournit l'agent de référence utilisé pour simuler les autres joueurs durant un rollout.

        Paramètre `pid` : identifiant du joueur simulé.
        Retourne une instance de `GreedyBot` associée à `pid`, mémorisée dans `_rollout_agents` pour éviter toute reconstruction répétée.
        Effet de bord : peuple `_rollout_agents` au premier appel pour `pid`.
        """
        if pid not in self._rollout_agents:
            self._rollout_agents[pid] = GreedyBot(pid, self.config)
        return self._rollout_agents[pid]

    def _simulate_rollout(self, initial_state: GameState, first_action: Action) -> bool:
        """
        Simule la fin d'une manche à partir d'un état donné et d'une première action imposée.

        Paramètre `initial_state` : vue matérialisée de l'état courant, copiée avant simulation.
        Paramètre `first_action` : action imposée au joueur courant pour le premier demi-coup simulé.
        Retourne un booléen, vrai si `self.player_id` figure dans les deux premiers rangs de sortie simulés (`ROLE_PRESIDENT` ou
        `ROLE_VICE_PRESIDENT`) à l'issue du rollout, ou si la limite `_MAX_ROLLOUT_STEPS` est atteinte sans que `self.player_id` ait terminé
        en position défavorable. Effet de bord : consomme l'état interne de `_rng`. N'affecte jamais l'état réel de la partie.
        """
        from core.rules_engine import (
            combination_power, generate_sequence_plays as gen_seq,
            generate_uniform_plays as gen_uni, is_valid_sequence_combination,
            is_valid_uniform_combination,
        )

        state = copy.deepcopy(initial_state)
        n = len(state.hands)
        pending_action: Optional[Action] = first_action
        steps = 0

        while len(state.finish_order) < n - 1 and steps < _MAX_ROLLOUT_STEPS:
            steps += 1
            pid = state.current_player_id
            if state.is_finished.get(pid, False) or not state.is_eligible.get(pid, True):
                candidate = (pid + 1) % n
                for _ in range(n):
                    if not state.is_finished.get(candidate, False):
                        break
                    candidate = (candidate + 1) % n
                state.current_player_id = candidate
                pending_action = None
                continue

            if pending_action is not None:
                action = pending_action
                pending_action = None
            elif pid == self.player_id:
                options = self._legal_options(state.hands[pid], state)
                if not options:
                    action = Action(action_type=ActionType.ACTION_HARD_PASS)
                else:
                    cards, declared = self._rng.choice(options)
                    action = Action(action_type=ActionType.ACTION_PLAY, cards=cards, declared_power=declared)
            else:
                action = self._rollout_agent_for(pid).choose_action(state)

            if action.action_type == ActionType.ACTION_PLAY:
                state.hands[pid] = state.hands[pid].without(action.cards)
                is_seq = state.trick.is_sequence or (
                    self.config.straights_enabled
                    and len(action.cards) >= 3
                    and is_valid_sequence_combination(action.cards, state.e_rev)
                    and not is_valid_uniform_combination(action.cards, state.e_rev, action.declared_power)
                )
                if state.trick.size == 0:
                    state.trick.size = len(action.cards)
                    state.trick.is_sequence = is_seq
                if is_seq:
                    joker_power = action.declared_power if action.declared_power is not None else 0
                    min_power = min(
                        f_power(c, state.e_rev) if not c.is_joker() else joker_power
                        for c in action.cards
                    )
                    state.trick.sequence_min_power = min_power
                else:
                    state.trick.current_power = combination_power(action.cards, state.e_rev, action.declared_power)
                state.trick.last_player_id = pid

                if len(action.cards) >= 4 and self.config.revolution_enabled and not any(c.is_joker() for c in action.cards):
                    state.e_rev = not state.e_rev

                if state.hands[pid].is_empty() and not state.is_finished.get(pid, False):
                    state.is_finished[pid] = True
                    state.finish_order.append(pid)
                    if len(state.finish_order) == n - 1:
                        remaining = [p for p in range(n) if not state.is_finished.get(p, False)]
                        if remaining:
                            state.finish_order.append(remaining[0])
                    state.trick = type(state.trick)()
                    active = [p for p in range(n) if not state.is_finished.get(p, False)]
                    for p in active:
                        state.is_eligible[p] = True
                    state.current_player_id = active[0] if active else pid
                    continue
            else:
                state.is_eligible[pid] = False

            others_done = all(
                not state.is_eligible.get(p, False)
                for p in range(n)
                if not state.is_finished.get(p, False) and p != state.trick.last_player_id
            )
            if others_done and state.trick.last_player_id is not None:
                winner = state.trick.last_player_id
                state.trick = type(state.trick)()
                active = [p for p in range(n) if not state.is_finished.get(p, False)]
                for p in active:
                    state.is_eligible[p] = True
                state.current_player_id = winner if not state.is_finished.get(winner, False) else (active[0] if active else winner)
            else:
                candidate = (pid + 1) % n
                for _ in range(n):
                    if not state.is_finished.get(candidate, False):
                        break
                    candidate = (candidate + 1) % n
                state.current_player_id = candidate

        rank = state.finish_order.index(self.player_id) if self.player_id in state.finish_order else n - 1
        return rank <= 1

    def choose_action(self, game_state: GameState) -> Action:
        """
        Sélectionne l'option légale maximisant le taux de victoire simulé.

        Paramètre `game_state` : vue matérialisée de l'état courant.
        Retourne une instance de `Action`. Retourne un passe conforme à `pass_type` si aucune option n'est disponible. Pour chaque option
        candidate, exécute `rollout_count` simulations via `_simulate_rollout` et retient l'option de score moyen maximal. Effet de bord :
        consomme l'état interne de `_rng`.
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

        best_option = options[0]
        best_score = -1.0
        for cards, declared in options:
            candidate_action = Action(action_type=ActionType.ACTION_PLAY, cards=cards, declared_power=declared)
            wins = sum(
                1 for _ in range(self.rollout_count)
                if self._simulate_rollout(game_state, candidate_action)
            )
            score = wins / self.rollout_count
            if score > best_score:
                best_score = score
                best_option = (cards, declared)

        cards, declared_power = best_option
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
        Intercepte lorsqu'une carte jumelle est disponible.

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
