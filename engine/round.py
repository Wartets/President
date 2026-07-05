"""
Module de la logique d'une manche.

Le module implémente `run_round`, la fonction orchestrant l'intégralité d'une manche : distribution des mains, vérification du Putsch, phase
d'échange, phase de jeu par plis successifs avec gestion de l'interception, de l'égalité forcée, de la révolution, de la clôture magique et du saut de
tour, puis clôture de la manche avec attribution des rôles et des points de victoire. La fonction ne mute aucun état en dehors de l'instance de
`GameState` qui lui est confiée ; chaque transition est également publiée sous forme d'événement sur le bus fourni.

Le module dépend de `core.config`, `core.models`, `core.math_utils`, `core.rules_engine`, `engine.state`, `engine.event_bus`, `events.structural`,
`events.transactional` et `agents.interface`.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from agents.interface import AbstractBaseAgent
from core.config import (
    GameConfig, PENALTY_DRAW_CARDS, PENALTY_INSTANT_SCUM,
    ROLE_PRESIDENT, ROLE_SCUM, ROLE_VICE_PRESIDENT, ROLE_VICE_SCUM,
)
from core.math_utils import compute_vp, f_power, role_for_rank
from core.models import Action, ActionType, Card, Hand
from core.rules_engine import (
    build_deck, can_intercept, combination_power, deal_hands,
    generate_sequence_plays, generate_uniform_plays, is_action_valid,
    is_valid_sequence_combination, is_valid_uniform_combination,
    matches_finish_penalty, max_power_cards, num_decks, random_cards,
    triggers_double_revolution, triggers_magic_closure, triggers_revolution,
    triggers_skip_turn,
)
from engine.event_bus import EventBus
from engine.state import GameState, TrickState
from events.structural import (
    EventHandEmpty, EventPlayerFinished, EventRoundEnd, EventRoundStart,
    EventTrickClosed, EventTrickStart,
)
from events.transactional import (
    EventActionPlayed, EventActionRequest, EventAskPutsch, EventExchange,
    EventExchangeIntent, EventInterceptionBroadcast,
    EventInterceptionResolved, EventPutschInvoked, EventRuleTriggered,
)


def _has_any_legal_play(hand: Hand, e_rev: bool, config: GameConfig) -> bool:
    """
    Détermine si une main dispose d'au moins une combinaison ouvrable.

    Paramètre `hand` : main considérée.
    Paramètre `e_rev` : état de révolution courant.
    Paramètre `config` : configuration de la partie.
    Retourne un booléen, vrai si `generate_uniform_plays` ou, lorsque `straights_enabled` est vrai, `generate_sequence_plays` retourne au
    moins une option pour une ouverture de pli. Aucun effet de bord.
    """
    if generate_uniform_plays(hand, e_rev, None, None):
        return True
    if config.straights_enabled and generate_sequence_plays(hand, e_rev, None, None):
        return True
    return False


def _count_legal_plays(state: GameState, pid: int, config: GameConfig) -> int:
    """
    Dénombre les combinaisons légales distinctes disponibles pour un joueur à l'instant courant.

    Paramètre `state` : vue matérialisée de l'état courant.
    Paramètre `pid` : identifiant du joueur considéré.
    Paramètre `config` : configuration de la partie.
    Retourne un entier positif ou nul, somme du nombre d'options retournées par `generate_uniform_plays` lorsque le pli actif n'est pas une
    suite, et par `generate_sequence_plays` lorsque `straights_enabled` est vrai et que le pli est vide ou déjà engagé en suite. Aucun effet
    de bord.
    """
    hand = state.hands[pid]
    trick = state.trick
    required_size = trick.size if trick.size > 0 else None
    min_power_exclusive = trick.current_power

    count = 0
    if not trick.is_sequence:
        count += len(generate_uniform_plays(hand, state.e_rev, required_size, min_power_exclusive))
    if config.straights_enabled and (trick.size == 0 or trick.is_sequence):
        seq_min = trick.sequence_min_power if trick.is_sequence else None
        count += len(generate_sequence_plays(hand, state.e_rev, required_size, seq_min))
    return count


def _putsch_condition_met(hand: Hand, e_rev: bool) -> bool:
    """
    Évalue la condition mathématique d'éligibilité au Putsch.

    Paramètre `hand` : main du joueur `ROLE_SCUM`.
    Paramètre `e_rev` : état de révolution courant, faux en début de manche.
    Retourne un booléen, vrai s'il existe dans `hand` une combinaison uniforme de taille supérieure ou égale à quatre, ou si la carte de
    puissance maximale de la main a une puissance inférieure ou égale à dix. Aucun effet de bord.
    """
    uniform_options = generate_uniform_plays(hand, e_rev, None, None)
    if any(len(cards) >= 4 for cards, _ in uniform_options):
        return True
    top = max_power_cards(hand, 1, e_rev)
    if not top:
        return False
    return f_power(top[0], e_rev) <= 10


def _validated_exchange_cards(hand: Hand, chosen: List[Card], count: int, e_rev: bool) -> Tuple[Card, ...]:
    """
    Normalise une sélection de cartes d'échange.

    Paramètre `hand` : main source.
    Paramètre `chosen` : cartes retournées par l'agent.
    Paramètre `count` : nombre exact de cartes attendues.
    Paramètre `e_rev` : état de révolution utilisé pour la sélection de repli.
    Retourne un tuple de cartes appartenant à `hand`, de taille `count`. La sélection de repli prend les cartes de plus faible puissance.
    Aucun effet de bord.
    """
    remaining = list(hand.cards)
    normalized: List[Card] = []
    for card in chosen:
        if card in remaining:
            normalized.append(card)
            remaining.remove(card)
        if len(normalized) == count:
            break
    if len(normalized) == count:
        return tuple(normalized)
    ordered = sorted(remaining, key=lambda c: (f_power(c, e_rev), repr(c)))
    return tuple(normalized + ordered[: count - len(normalized)])


def run_round(
    config: GameConfig,
    agents: Dict[int, AbstractBaseAgent],
    event_bus: EventBus,
    round_index: int,
    previous_roles: Optional[Dict[int, str]],
    game_id: str,
) -> Tuple[Dict[int, str], Dict[int, float], List[int]]:
    """
    Exécute une manche complète et retourne les rôles, les points de victoire et l'ordre de sortie qui en résultent.

    Paramètre `config` : configuration immuable de la partie.
    Paramètre `agents` : association entre identifiant de joueur et instance d'agent.
    Paramètre `event_bus` : bus de diffusion des événements de la manche.
    Paramètre `round_index` : index $m$ de la manche courante.
    Paramètre `previous_roles` : association entre identifiant de joueur et rôle attribué à l'issue de la manche précédente, ou `None` pour la
    première manche.
    Paramètre `game_id` : identifiant de la partie en cours.
    Retourne un tuple composé de l'association rôle attribué pour la manche suivante par identifiant de joueur, de l'association point de victoire
    par identifiant de joueur, et de la liste ordonnée des identifiants de joueurs par ordre de sortie. Effet de bord : publie l'intégralité des
    événements structurels et transactionnels de la manche sur `event_bus`.
    """
    n = config.player_count
    tick = [0]

    def next_tick() -> int:
        tick[0] += 1
        return tick[0]

    def emit(event_cls, **kwargs):
        event_bus.publish(
            event_cls(
                timestamp=next_tick(),
                game_id=game_id,
                round_id=round_index,
                state_hash=state.snapshot_key().__repr__(),
                **kwargs,
            )
        )

    # Distribution
    strict_remainder_seat = None
    if config.strict_remainder_allocation and previous_roles:
        for pid, role in previous_roles.items():
            if role == config.strict_remainder_role:
                strict_remainder_seat = pid
                break

    deck = build_deck(config)
    hands = deal_hands(config, deck, round_index, strict_remainder_seat)

    state = GameState(
        hands={pid: hands[pid] for pid in range(n)},
        is_finished={pid: False for pid in range(n)},
        is_eligible={pid: True for pid in range(n)},
        finish_order=[],
        e_rev=False,
        l_rev=False,
        is_equal_forced=False,
        current_player_id=0,
        round_index=round_index,
        trick=TrickState(),
        roles=dict(previous_roles) if previous_roles else {},
    )

    emit(EventRoundStart, initial_hands={pid: hands[pid].cards for pid in range(n)})

    # Vérification du Putsch  
    putsch_invoked = False
    if config.putsch_enabled and round_index > 0 and previous_roles:
        scum_id = next((pid for pid, role in previous_roles.items() if role == ROLE_SCUM), None)
        if scum_id is not None:
            condition_met = _putsch_condition_met(state.hands[scum_id], state.e_rev)
            emit(EventAskPutsch, player_id=scum_id, condition_met=condition_met)
            if condition_met and agents[scum_id].ask_putsch(state.hands[scum_id]):
                putsch_invoked = True
                emit(EventPutschInvoked, player_id=scum_id)

    # Phase d'échange
    if round_index > 0 and previous_roles and not putsch_invoked:
        blind_rng = random.Random(f"{config.random_seed}:{round_index}:blind_tax")
        pairs = [
            (ROLE_SCUM, ROLE_PRESIDENT, 2),
            (ROLE_VICE_SCUM, ROLE_VICE_PRESIDENT, 1),
        ]
        role_to_pid = {role: pid for pid, role in previous_roles.items()}
        for giver_role, receiver_role, count in pairs:
            giver = role_to_pid.get(giver_role)
            receiver = role_to_pid.get(receiver_role)
            if giver is None or receiver is None:
                continue
            if giver_role == ROLE_SCUM and config.blind_tax_enabled:
                cards = random_cards(state.hands[giver], count, blind_rng)
                was_blind = True
            else:
                cards = max_power_cards(state.hands[giver], count, state.e_rev)
                was_blind = False
            state.hands[giver] = state.hands[giver].without(cards)
            state.hands[receiver] = state.hands[receiver].with_added(cards)
            emit(EventExchange, from_player=giver, to_player=receiver, cards=cards, was_blind_tax=was_blind)

        reverse_pairs = [
            (ROLE_PRESIDENT, ROLE_SCUM, 2),
            (ROLE_VICE_PRESIDENT, ROLE_VICE_SCUM, 1),
        ]
        for giver_role, receiver_role, count in reverse_pairs:
            giver = role_to_pid.get(giver_role)
            receiver = role_to_pid.get(receiver_role)
            if giver is None or receiver is None:
                continue
            chosen = agents[giver].choose_exchange_cards(state.hands[giver], state, count)
            cards = _validated_exchange_cards(state.hands[giver], chosen, count, state.e_rev)
            emit(EventExchangeIntent, from_player=giver, to_player=receiver, offered_cards=cards)
            state.hands[giver] = state.hands[giver].without(cards)
            state.hands[receiver] = state.hands[receiver].with_added(cards)
            emit(EventExchange, from_player=giver, to_player=receiver, cards=cards, was_blind_tax=False)

    # Phase de jeu---
    if round_index == 0:
        opener = config.first_trick_opener_id
    elif previous_roles:
        opener = next((pid for pid, role in previous_roles.items() if role == ROLE_SCUM), 0)
    else:
        opener = 0

    trick_index = 0
    forced_scum_ref: List[Optional[int]] = [None]
    instant_scum_players: List[int] = []

    # Limite de transitions d'un pli avant arrêt de sécurité.
    _MAX_ACTIONS_PER_TRICK = max(64, n * 8)

    while len(state.finish_order) < n - 1:
        state.trick = TrickState(trick_index=trick_index)
        state.is_equal_forced = False
        active = state.active_players(n)
        for pid in active:
            state.is_eligible[pid] = True
        state.current_player_id = opener
        emit(EventTrickStart, opener_id=opener, trick_index=trick_index)

        _actions_in_trick = 0

        while not state.trick.is_closed:
            _actions_in_trick += 1
            if _actions_in_trick > _MAX_ACTIONS_PER_TRICK:
                raise RuntimeError(
                    f"État incohérent détecté : pli {trick_index} de la manche {round_index} "
                    f"dépasse {_MAX_ACTIONS_PER_TRICK} actions sans clôture. "
                    f"Partie {game_id} exclue des résultats d'entraînement."
                )
            pid = state.current_player_id
            if state.is_finished[pid] or not state.is_eligible[pid]:
                pid = _advance_player(state, n, pid)
                state.current_player_id = pid
                if _trick_should_close(state, n):
                    state.trick.is_closed = True
                    break
                continue

            emit(
                EventActionRequest,
                player_id=pid,
                trick_index=trick_index,
                legal_action_count=_count_legal_plays(state, pid, config),
            )
            action = agents[pid].choose_action(state)
            was_suboptimal = False

            if action.action_type == ActionType.ACTION_PLAY:
                allow_equal = config.skip_on_equal or state.is_equal_forced
                valid = is_action_valid(
                    action.cards,
                    action.declared_power,
                    state.hands[pid],
                    state.e_rev,
                    state.trick.size if state.trick.size > 0 else None,
                    state.trick.current_power,
                    state.trick.is_sequence,
                    state.trick.sequence_min_power,
                    config.straights_enabled,
                    allow_equal,
                    config=config,
                )
                if state.is_equal_forced and valid:
                    resulting = combination_power(action.cards, state.e_rev, action.declared_power)
                    if resulting != state.trick.current_power:
                        valid = False
                if not valid:
                    action = Action(
                        action_type=(
                            ActionType.ACTION_SOFT_PASS
                            if config.pass_type == "ALLOW_SOFT"
                            else ActionType.ACTION_HARD_PASS
                        )
                    )
                    was_suboptimal = True

            if action.action_type == ActionType.ACTION_PLAY:
                interception_finish = _apply_play(state, config, agents, pid, action, emit)
                emit(
                    EventActionPlayed,
                    player_id=pid,
                    action_type=action.action_type,
                    cards_played=action.cards,
                    resulting_power=state.trick.current_power if not state.trick.is_sequence else None,
                    was_suboptimal=was_suboptimal,
                )
                if state.hands[pid].is_empty() and not state.is_finished[pid]:
                    _finish_player(
                        state,
                        config,
                        pid,
                        action,
                        emit,
                        forced_scum_ref,
                        instant_scum_players,
                        state.trick.last_play_e_rev_before,
                        state.trick.last_play_triggered_revolution,
                    )
                if interception_finish is not None:
                    interceptor_id, interception_action = interception_finish
                    if state.hands[interceptor_id].is_empty() and not state.is_finished[interceptor_id]:
                        _finish_player(
                            state,
                            config,
                            interceptor_id,
                            interception_action,
                            emit,
                            forced_scum_ref,
                            instant_scum_players,
                            state.e_rev,
                            False,
                        )
            else:
                if action.action_type == ActionType.ACTION_HARD_PASS:
                    state.is_eligible[pid] = False
                    state.is_equal_forced = False
                else:
                    if config.pass_type != "ALLOW_SOFT":
                        state.is_eligible[pid] = False
                    was_suboptimal = was_suboptimal or _has_any_legal_play(state.hands[pid], state.e_rev, config)
                if state.trick.last_player_id is not None and pid != state.trick.last_player_id:
                    state.trick.passes_since_last_play += 1
                emit(
                    EventActionPlayed,
                    player_id=pid,
                    action_type=action.action_type,
                    cards_played=(),
                    resulting_power=None,
                    was_suboptimal=was_suboptimal,
                )

            if state.trick.is_closed:
                break
            if _trick_should_close(state, n):
                state.trick.is_closed = True
                break

            advance_from = state.trick.last_player_id if (
                action.action_type == ActionType.ACTION_PLAY and state.trick.last_player_id is not None
            ) else pid
            state.current_player_id = _advance_player(state, n, advance_from)

        winner = state.trick.last_player_id
        if winner is not None:
            emit(EventTrickClosed, winner_id=winner, trick_size=state.trick.size)
            trick_index += 1
            if state.is_finished.get(winner, False):
                opener = _advance_player(state, n, winner)
            else:
                opener = winner
        else:
            opener = _advance_player(state, n, opener)

    # Clôture de manche
    remaining = [pid for pid in range(n) if not state.is_finished.get(pid, False)]
    if remaining:
        last_player = remaining[0]
        state.finish_order.append(last_player)
        last_rank = len(state.finish_order) - 1
        emit(
            EventPlayerFinished,
            player_id=last_player,
            rank=last_rank,
            vp_earned=compute_vp(last_rank, n, config.vp_distribution_type),
        )

    vp_by_player: Dict[int, float] = {}
    roles_by_player: Dict[int, str] = {}
    for k, pid in enumerate(state.finish_order):
        vp = compute_vp(k, n, config.vp_distribution_type)
        vp_by_player[pid] = vp
        roles_by_player[pid] = role_for_rank(k, n)

    for pid in instant_scum_players:
        roles_by_player[pid] = ROLE_SCUM
        vp_by_player[pid] = compute_vp(n - 1, n, config.vp_distribution_type)

    if forced_scum_ref[0] is not None:
        roles_by_player[forced_scum_ref[0]] = ROLE_SCUM

    emit(EventRoundEnd, vp_by_player=vp_by_player, roles_by_player=roles_by_player)
    return roles_by_player, vp_by_player, list(state.finish_order)


def _advance_player(state: GameState, n: int, from_pid: int) -> int:
    """
    Détermine le prochain joueur actif dans l'ordre du tour.

    Paramètre `state` : vue matérialisée de l'état courant.
    Paramètre `n` : nombre total de joueurs.
    Paramètre `from_pid` : identifiant du joueur courant.
    Retourne l'identifiant du prochain joueur pour lequel `is_finished` est faux, en parcourant les sièges dans l'ordre croissant modulo $n$ à
    partir de `from_pid`. Aucun effet de bord.
    """
    candidate = (from_pid + 1) % n
    for _ in range(n):
        if not state.is_finished.get(candidate, False):
            return candidate
        candidate = (candidate + 1) % n
    return from_pid


def _trick_should_close(state: GameState, n: int) -> bool:
    """
    Détermine si le pli en cours doit se clôturer par épuisement des éligibilités.

    Paramètre `state` : vue matérialisée de l'état courant.
    Paramètre `n` : nombre total de joueurs.
    Retourne un booléen, vrai si tous les joueurs actifs autres que le dernier ayant validé `ACTION_PLAY` possèdent une éligibilité fausse.
    Aucun effet de bord.
    """
    if state.trick.last_player_id is None:
        return False
    active = state.active_players(n)
    others = [pid for pid in active if pid != state.trick.last_player_id]
    if not others:
        return True
    if state.trick.passes_since_last_play >= len(others):
        return True
    return all(not state.is_eligible.get(pid, False) for pid in others)


def _apply_play(
    state: GameState,
    config: GameConfig,
    agents,
    pid: int,
    action: Action,
    emit,
) -> Optional[Tuple[int, Action]]:
    """
    Applique une action `ACTION_PLAY` validée à l'état courant.

    Paramètre `state` : vue matérialisée de l'état courant, mutée en place.
    Paramètre `config` : configuration de la partie.
    Paramètre `agents` : association entre identifiant de joueur et instance d'agent, utilisée pour solliciter les opportunités d'interception.
    Paramètre `pid` : identifiant du joueur ayant posé la combinaison.
    Paramètre `action` : action validée à appliquer.
    Paramètre `emit` : fonction de publication d'événement liée à la manche courante.
    Retourne l'action d'interception appliquée lorsqu'elle vide la main d'un intercepteur, ou `None`. Effet de bord : retire les cartes jouées de la main du joueur, met à jour la taille et la puissance du pli, applique les
    déclenchements de révolution, de double révolution, de clôture magique et de saut de tour, et déclenche le broadcast d'interception lorsque
    pertinent.
    """
    is_sequence_play = state.trick.is_sequence or (
        config.straights_enabled
        and len(action.cards) >= 3
        and is_valid_sequence_combination(action.cards, state.e_rev)
        and not is_valid_uniform_combination(action.cards, state.e_rev, action.declared_power)
    )

    previous_power = state.trick.sequence_min_power if is_sequence_play else state.trick.current_power

    state.hands[pid] = state.hands[pid].without(action.cards)

    if state.trick.size == 0:
        state.trick.size = len(action.cards)
        state.trick.is_sequence = is_sequence_play

    if is_sequence_play:
        joker_power = action.declared_power if action.declared_power is not None else 0
        min_power = min(
            f_power(c, state.e_rev) if not c.is_joker() else joker_power
            for c in action.cards
        )
        new_power = min_power
        state.trick.sequence_min_power = min_power
    else:
        new_power = combination_power(action.cards, state.e_rev, action.declared_power)
        state.trick.current_power = new_power

    state.trick.last_player_id = pid
    state.trick.passes_since_last_play = 0
    state.trick.last_play_e_rev_before = state.e_rev
    state.trick.last_play_triggered_revolution = False
    state.is_equal_forced = bool(
        config.skip_on_equal and previous_power is not None and new_power == previous_power
    )
    if state.is_equal_forced:
        emit(EventRuleTriggered, rule_name="EQUAL_FORCED", triggering_player_id=pid)

    e_rev_before_flip = state.e_rev

    if not state.l_rev and triggers_double_revolution(action.cards, config, is_sequence_play):
        state.e_rev = not state.e_rev
        state.l_rev = True
        state.trick.last_play_triggered_revolution = True
        emit(
            EventRuleTriggered,
            rule_name="DOUBLE_REVOLUTION",
            triggering_player_id=pid,
            magnitude=len(action.cards),
        )
    elif not state.l_rev and triggers_revolution(action.cards, config, is_sequence_play):
        state.e_rev = not state.e_rev
        state.trick.last_play_triggered_revolution = True
        emit(
            EventRuleTriggered,
            rule_name="REVOLUTION",
            triggering_player_id=pid,
            magnitude=len(action.cards),
        )

    if triggers_magic_closure(action.cards, config, e_rev_before_flip, action.declared_power, is_sequence_play):
        state.trick.is_closed = True
        emit(EventRuleTriggered, rule_name="MAGIC_CLOSURE", triggering_player_id=pid)

    if config.interception_enabled and len(action.cards) == 1 and not state.trick.is_closed:
        candidates = [
            other for other in state.active_players(len(state.hands))
            if other != pid and other != state.trick.last_player_id
        ]
        emit(EventInterceptionBroadcast, played_card=action.cards[0], eligible_player_ids=tuple(candidates))
        interceptor_id = None
        intercepted_card = None
        for candidate in candidates:
            accepted, twin = agents[candidate].on_interception_opportunity(state, action.cards[0])
            if (
                accepted
                and twin is not None
                and twin in state.hands[candidate].cards
                and can_intercept(action.cards[0], twin, config)
            ):
                interceptor_id = candidate
                intercepted_card = twin
                break
        emit(EventInterceptionResolved, interceptor_id=interceptor_id, intercepted_card=intercepted_card)
        if interceptor_id is not None:
            if intercepted_card is not None:
                state.hands[interceptor_id] = state.hands[interceptor_id].without((intercepted_card,))
                state.trick.last_player_id = interceptor_id
                state.trick.passes_since_last_play = 0
                state.current_player_id = interceptor_id
                emit(EventRuleTriggered, rule_name="INTERCEPTION", triggering_player_id=interceptor_id)
                emit(
                    EventActionPlayed,
                    player_id=interceptor_id,
                    action_type=ActionType.ACTION_PLAY,
                    cards_played=(intercepted_card,),
                    resulting_power=combination_power((intercepted_card,), state.e_rev, None),
                    was_suboptimal=False,
                )
                if config.interception_closes_trick:
                    state.trick.is_closed = True
                return interceptor_id, Action(
                    action_type=ActionType.ACTION_PLAY,
                    cards=(intercepted_card,),
                )

    if not state.trick.is_closed:
        skip_count = triggers_skip_turn(action.cards, config)
        if skip_count > 0:
            emit(
                EventRuleTriggered,
                rule_name="SKIP_TURN",
                triggering_player_id=pid,
                magnitude=skip_count,
            )
            skip_target = state.trick.last_player_id
            for _ in range(skip_count):
                skip_target = _advance_player(state, len(state.hands), skip_target)
                if not state.is_finished.get(skip_target, False):
                    state.is_eligible[skip_target] = False
            state.current_player_id = skip_target

    return None


def _finish_player(
    state: GameState,
    config: GameConfig,
    pid: int,
    action: Action,
    emit,
    forced_scum_ref: List[Optional[int]],
    instant_scum_players: List[int],
    e_rev_before: Optional[bool] = None,
    triggered_revolution: Optional[bool] = None,
) -> None:
    """
    Traite la sortie d'un joueur ayant vidé sa main.

    Paramètre `state` : vue matérialisée de l'état courant, mutée en place.
    Paramètre `config` : configuration de la partie.
    Paramètre `pid` : identifiant du joueur sorti.
    Paramètre `action` : action ayant provoqué la sortie.
    Paramètre `emit` : fonction de publication d'événement liée à la manche courante.
    Paramètre `forced_scum_ref` : liste à un élément, mutée en place, portant l'identifiant du joueur devant être forcé au rôle `ROLE_SCUM` la manche
    suivante lorsque `finish_penalty_extended` est vrai et que la sortie satisfait `matches_finish_penalty`.
    Paramètre `e_rev_before` : état de révolution avant la pose de sortie, `state.e_rev` si `None`.
    Paramètre `triggered_revolution` : indique si la pose de sortie a modifié l'état de révolution, recalculé si `None`.
    Retourne `None`. Effet de bord : marque le joueur comme sorti, l'ajoute à `finish_order`, et publie `EventHandEmpty` et `EventPlayerFinished`. Si
    `finish_penalty_enabled` est vrai, que la sortie satisfait `matches_finish_penalty` et que `finish_penalty_type` vaut
    `PENALTY_DRAW_CARDS`, une partie de la combinaison jouée est réintégrée dans la main du joueur, annulant la sortie pour ce tour. Si
    `finish_penalty_extended` est vrai et que la sortie satisfait `matches_finish_penalty`, `forced_scum_ref` est mis à jour avec `pid`. Si
    `finish_penalty_enabled` est vrai, que la sortie satisfait `matches_finish_penalty` et que `finish_penalty_type` vaut
    `PENALTY_INSTANT_SCUM`, `pid` est ajouté à `instant_scum_players`, son rôle et son point de victoire de la manche courante étant alors
    substitués par ceux du dernier index de sortie.
    """
    if triggered_revolution is None:
        triggered_revolution = (
            triggers_revolution(action.cards, config, state.trick.is_sequence)
            or triggers_double_revolution(action.cards, config, state.trick.is_sequence)
        )
    if e_rev_before is None:
        e_rev_before = state.e_rev
    penalty_condition = matches_finish_penalty(action.cards, config, e_rev_before, triggered_revolution)

    if config.finish_penalty_extended and penalty_condition:
        forced_scum_ref[0] = pid

    if config.finish_penalty_enabled and penalty_condition:
        emit(EventRuleTriggered, rule_name="FINISH_PENALTY", triggering_player_id=pid)

    if config.finish_penalty_enabled and penalty_condition and config.finish_penalty_type == PENALTY_DRAW_CARDS:
        kept = action.cards[: min(config.finish_penalty_draw_count, len(action.cards))]
        state.hands[pid] = state.hands[pid].with_added(kept)
        return

    if config.finish_penalty_enabled and penalty_condition and config.finish_penalty_type == PENALTY_INSTANT_SCUM:
        instant_scum_players.append(pid)

    state.is_finished[pid] = True
    state.is_eligible[pid] = False
    state.finish_order.append(pid)
    rank = len(state.finish_order) - 1
    vp_earned = compute_vp(rank, config.player_count, config.vp_distribution_type)
    emit(EventHandEmpty, player_id=pid)
    emit(
        EventPlayerFinished,
        player_id=pid,
        rank=rank,
        vp_earned=vp_earned,
    )
