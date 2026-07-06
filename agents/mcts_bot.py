"""
Module de l'agent Monte-Carlo par simulation de rollouts.

Le module définit `MCTSBot`, une implémentation de `AbstractBaseAgent` évaluant un sous-ensemble borné d'options légales par un budget total
de rollouts réparti entre elles, chaque rollout jouant la fin de manche avec des agents de référence déterministes pour les adversaires, et
sélectionnant l'option dont le score moyen estimé est le plus élevé. L'agent ne modifie jamais l'état réel de la partie : toute simulation
opère sur un clone léger de la vue matérialisée courante, obtenu sans recopie profonde des objets immuables (`Card`, `Hand`), ceux-ci n'étant
jamais mutés en place ailleurs dans le moteur.

Le module dépend de `agents.interface`, `agents.greedy_bot`, `core.models`, `core.config`, `core.rules_engine`, `engine.state` et de la
bibliothèque standard `dataclasses` et `random`.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Dict, List, Optional, Tuple

from agents.greedy_bot import GreedyBot
from agents.interface import AbstractBaseAgent
from core.config import GameConfig
from core.math_utils import f_power
from core.models import Action, ActionType, Card, Hand
from core.rules_engine import (
    combination_power, generate_sequence_plays, generate_uniform_plays,
    is_valid_sequence_combination, is_valid_uniform_combination,
)
from engine.state import GameState

# Nombre total de rollouts répartis entre toutes les options candidates d'une décision.
_DEFAULT_ROLLOUT_BUDGET = 160

# Nombre minimal de rollouts garantis par option candidate, même si le budget global est faible.
_MIN_ROLLOUTS_PER_OPTION = 6

# Nombre maximal de demi-coups simulés par rollout avant arrêt forcé.
_MAX_ROLLOUT_STEPS = 250

# Nombre maximal d'options distinctes réellement évaluées par rollout ; au-delà, seules des
# options représentatives sélectionnées par `_prefilter_candidates` sont conservées.
_MAX_CANDIDATES_EVALUATED = 8


class MCTSBot(AbstractBaseAgent):
    """
    Agent sélectionnant l'option légale de score simulé moyen maximal, sous budget de rollouts borné.

    Champ `config` : configuration de la partie.
    Champ `rollout_budget` : nombre total de rollouts répartis entre les options candidates d'une décision, entier strictement positif.
    Champ `_rng` : générateur pseudo-aléatoire dédié à l'agent et à ses rollouts internes.
    """

    def __init__(
        self,
        player_id: int,
        config: GameConfig,
        rollout_count: Optional[int] = None,
        rollout_budget: int = _DEFAULT_ROLLOUT_BUDGET,
    ) -> None:
        super().__init__(player_id)
        self.config = config
        # `rollout_count`, conservé pour compatibilité ascendante avec les constructions
        # existantes, fixe directement le budget total (par équivalence approximative avec
        # l'ancien nombre de rollouts par option) plutôt que le nombre par option lui-même.
        self.rollout_budget = rollout_count * _MAX_CANDIDATES_EVALUATED if rollout_count else rollout_budget
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

    def _prefilter_candidates(
        self, options: List[Tuple[Tuple[Card, ...], Optional[int]]], hand_size: int, e_rev: bool,
    ) -> List[Tuple[Tuple[Card, ...], Optional[int]]]:
        """
        Réduit l'ensemble d'options réellement évaluées par rollout à un sous-ensemble borné et représentatif.

        Paramètre `options` : ensemble complet des options légales disponibles.
        Paramètre `hand_size` : taille de la main avant la pose.
        Paramètre `e_rev` : état de révolution courant.
        Retourne une liste d'options de taille au plus `_MAX_CANDIDATES_EVALUATED`, incluant systématiquement toute option vidant intégralement
        la main (sortie immédiate), puis un échantillon des options restantes réparti sur l'éventail des puissances résultantes croissantes
        plutôt que limité aux seules plus faibles, afin de conserver une diversité représentative malgré la troncature. Aucun effet de bord.
        """
        if len(options) <= _MAX_CANDIDATES_EVALUATED:
            return options

        finishing = [opt for opt in options if len(opt[0]) == hand_size]
        others = sorted(
            (opt for opt in options if len(opt[0]) != hand_size),
            key=lambda opt: self._resulting_power(opt, e_rev),
        )
        remaining_slots = max(0, _MAX_CANDIDATES_EVALUATED - len(finishing))
        if remaining_slots <= 0 or not others:
            selected_others: List[Tuple[Tuple[Card, ...], Optional[int]]] = []
        elif len(others) <= remaining_slots:
            selected_others = others
        else:
            step = len(others) / remaining_slots
            selected_others = [others[int(i * step)] for i in range(remaining_slots)]
        return finishing + selected_others

    @staticmethod
    def _clone_state(state: GameState) -> GameState:
        """
        Clone une vue matérialisée pour une simulation isolée, sans recopie profonde inutile.

        Paramètre `state` : vue matérialisée source.
        Retourne une nouvelle instance de `GameState` dont les conteneurs mutables (dictionnaires de mains, d'éligibilité et de sortie,
        liste ordonnée de sortie, état de pli) sont dupliqués, tandis que les objets immuables qu'ils référencent (`Hand`, `Card`) restent
        partagés en toute sécurité entre l'original et le clone, ceux-ci n'étant jamais mutés en place. Aucun effet de bord sur `state`.
        """
        return GameState(
            hands=dict(state.hands),
            is_finished=dict(state.is_finished),
            is_eligible=dict(state.is_eligible),
            finish_order=list(state.finish_order),
            e_rev=state.e_rev,
            l_rev=state.l_rev,
            is_equal_forced=state.is_equal_forced,
            current_player_id=state.current_player_id,
            round_index=state.round_index,
            trick=dataclasses.replace(state.trick),
            roles=dict(state.roles),
        )

    def _simulate_rollout(self, initial_state: GameState, first_action: Action) -> float:
        """
        Simule la fin d'une manche à partir d'un état donné et d'une première action imposée.

        Paramètre `initial_state` : vue matérialisée de l'état courant, clonée avant simulation via `_clone_state`.
        Paramètre `first_action` : action imposée au joueur courant pour le premier demi-coup simulé.
        Retourne un score continu, domaine $[0, 1]$, égal à $1 - \\text{rang}/(N-1)$ pour le rang de sortie simulé de `self.player_id`
        (rang $N-1$ par défaut si la limite `_MAX_ROLLOUT_STEPS` est atteinte sans sortie). Pour les demi-coups suivants de `self.player_id`
        au sein du même rollout, une politique gloutonne déterministe est utilisée plutôt qu'un tirage uniforme, réduisant la variance de
        l'estimation à budget de rollouts égal. Effet de bord : consomme l'état interne de `_rng` uniquement via les agents de référence
        adverses. N'affecte jamais l'état réel de la partie.
        """
        state = self._clone_state(initial_state)
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
                    cards, declared = min(options, key=lambda opt: self._resulting_power(opt, state.e_rev))
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
        return max(0.0, 1.0 - rank / max(n - 1, 1))

    def choose_action(self, game_state: GameState) -> Action:
        """
        Sélectionne l'option légale maximisant le score simulé moyen, sous budget de rollouts réparti.

        Paramètre `game_state` : vue matérialisée de l'état courant.
        Retourne une instance de `Action`. Retourne un passe conforme à `pass_type` si aucune option n'est disponible, ou l'unique option
        immédiatement si une seule est légale. Sinon, préfiltre les options candidates (`_prefilter_candidates`), répartit `rollout_budget`
        entre elles, et retient l'option de score moyen simulé maximal. Effet de bord : consomme l'état interne de `_rng`.
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

        if len(options) == 1:
            cards, declared_power = options[0]
            return Action(action_type=ActionType.ACTION_PLAY, cards=cards, declared_power=declared_power)

        candidates = self._prefilter_candidates(options, hand.size(), game_state.e_rev)
        rollouts_per_option = max(_MIN_ROLLOUTS_PER_OPTION, self.rollout_budget // max(len(candidates), 1))

        best_option = candidates[0]
        best_score = -1.0
        for cards, declared in candidates:
            candidate_action = Action(action_type=ActionType.ACTION_PLAY, cards=cards, declared_power=declared)
            total = sum(
                self._simulate_rollout(game_state, candidate_action)
                for _ in range(rollouts_per_option)
            )
            score = total / rollouts_per_option
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
