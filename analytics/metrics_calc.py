"""
Module de calcul des métriques de recherche.

Le module transforme le flux d'événements collecté par `EventLogger` en métriques multidimensionnelles. Il implémente un sous-ensemble représentatif
des familles de métriques documentées par l'architecture : une métrique macro (matrice de transition de Markov des rôles), une métrique micro
(indice de Gini de la puissance de main initiale), et plusieurs métriques comportementales (taux de passe sous-optimal, indice d'agressivité à
l'ouverture, facteur de dominance de pli, taux de validité des actions). La structure du module permet l'ajout de métriques supplémentaires suivant le
même schéma : une fonction pure prenant en entrée les enregistrements d'`EventLogger` ou une séquence de rôles, et retournant une valeur ou une
structure numérique.

Le module dépend de `analytics.event_logger` pour le type `EventLogger`, de `core.math_utils` pour `f_power`, et de `core.models` pour la reconstruction
des cartes à partir des enregistrements sérialisés.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

from analytics.event_logger import EventLogger
from events.structural import EventRoundStart, EventTrickClosed, EventTrickStart
from events.transactional import EventActionPlayed


def gini_initial_hand_power(hand_powers_by_player: Dict[int, float]) -> float:
    """
    Calcule l'indice de Gini de la puissance de main initiale entre joueurs.

    Paramètre `hand_powers_by_player` : association entre identifiant de joueur et somme des puissances standard de sa main initiale.
    Retourne un nombre, domaine $[0, 1]$, nul si toutes les valeurs sont égales, croissant avec l'inégalité de répartition. Aucun effet de bord.
    """
    values = sorted(hand_powers_by_player.values())
    n = len(values)
    if n == 0 or sum(values) == 0:
        return 0.0
    cumulative = sum((i + 1) * v for i, v in enumerate(values))
    total = sum(values)
    return (2 * cumulative) / (n * total) - (n + 1) / n


def sub_optimal_pass_rate(logger: EventLogger, player_id: int) -> float:
    """
    Calcule le taux de passe sous-optimal d'un joueur (SOPR).

    Paramètre `logger` : journal des événements de la partie ou de la manche considérée.
    Paramètre `player_id` : identifiant du joueur considéré.
    Retourne un nombre, domaine $[0, 1]$, égal au ratio entre le nombre de `EventActionPlayed` de `player_id` marqués `was_suboptimal` et le nombre
    total de `EventActionPlayed` de `player_id`, nul si ce joueur n'a jamais agi. Aucun effet de bord.
    """
    actions = [e for e in logger.events_of_type(EventActionPlayed) if e.player_id == player_id]
    if not actions:
        return 0.0
    suboptimal = sum(1 for e in actions if e.was_suboptimal)
    return suboptimal / len(actions)


def action_validity_rate(logger: EventLogger, player_id: int) -> float:
    """
    Calcule le taux de validité des actions d'un joueur.

    Paramètre `logger` : journal des événements de la partie ou de la manche considérée.
    Paramètre `player_id` : identifiant du joueur considéré.
    Retourne un nombre, domaine $[0, 1]$, égal au complément du taux de passe sous-optimal, une action marquée `was_suboptimal` étant
    interprétée comme la substitution d'une action initialement illégale ou dominée. Aucun effet de bord.
    """
    return 1.0 - sub_optimal_pass_rate(logger, player_id)


def aggressiveness_index_opening(
    logger: EventLogger,
    player_id: int,
    average_hand_power_by_round: Dict[int, float],
) -> float:
    """
    Calcule l'indice d'agressivité à l'ouverture d'un joueur.

    Paramètre `logger` : journal des événements de la partie.
    Paramètre `player_id` : identifiant du joueur considéré.
    Paramètre `average_hand_power_by_round` : association entre index de manche et puissance moyenne de la main du joueur considéré au moment de
    chaque ouverture de pli qu'il a réalisée.
    Retourne un nombre, ratio moyen entre la puissance de la carte ou de la combinaison posée à l'ouverture et la puissance moyenne de la main du
    joueur au même instant, non défini (valeur nulle) si le joueur n'a jamais ouvert de pli. Aucun effet de bord.
    """
    trick_starts = {(e.round_id, e.trick_index): e for e in logger.events_of_type(EventTrickStart)}
    ratios: List[float] = []
    for action in logger.events_of_type(EventActionPlayed):
        key = None
        for (round_id, trick_index), start_event in trick_starts.items():
            if start_event.opener_id == player_id and round_id == action.round_id:
                key = (round_id, trick_index)
        if key is None:
            continue
        if action.player_id != player_id or action.resulting_power is None:
            continue
        avg_power = average_hand_power_by_round.get(action.round_id)
        if not avg_power:
            continue
        ratios.append(action.resulting_power / avg_power)
    if not ratios:
        return 0.0
    return sum(ratios) / len(ratios)


def trick_dominance_factor(logger: EventLogger, player_id: int) -> float:
    """
    Calcule le facteur de dominance de pli d'un joueur.

    Paramètre `logger` : journal des événements de la partie ou de la manche considérée.
    Paramètre `player_id` : identifiant du joueur considéré.
    Retourne un nombre, domaine $[0, 1]$, égal au ratio entre le nombre de `EventTrickClosed` remportés par `player_id` et le nombre de plis
    auxquels `player_id` a participé, ce dernier étant approché par le nombre de `EventActionPlayed` distincts de `player_id` par couple
    manche/pli. Aucun effet de bord.
    """
    closures = logger.events_of_type(EventTrickClosed)
    won = sum(1 for e in closures if e.winner_id == player_id)
    if not closures:
        return 0.0
    participated_keys = set()
    trick_starts = logger.events_of_type(EventTrickStart)
    current_round = None
    current_trick = None
    for event in logger.events:
        if isinstance(event, EventTrickStart):
            current_round, current_trick = event.round_id, event.trick_index
        if isinstance(event, EventActionPlayed) and event.player_id == player_id:
            participated_keys.add((current_round, current_trick))
    if not participated_keys:
        return 0.0
    return won / len(participated_keys)


def role_transition_matrix(role_sequence: Sequence[Dict[int, str]]) -> Dict[Tuple[str, str], float]:
    """
    Calcule la matrice de transition de Markov des rôles d'un round au suivant.

    Paramètre `role_sequence` : séquence des associations rôle par identifiant de joueur, une par manche, dans l'ordre chronologique.
    Retourne une association entre couple de rôles `(role_a, role_b)` et probabilité empirique $P(Role_A \\rightarrow Role_B)$ observée d'une
    manche à la suivante, sur l'ensemble des joueurs et des transitions disponibles. Aucun effet de bord.
    """
    transition_counts: Dict[str, Counter] = defaultdict(Counter)
    for earlier, later in zip(role_sequence, role_sequence[1:]):
        for pid, role_a in earlier.items():
            role_b = later.get(pid)
            if role_b is not None:
                transition_counts[role_a][role_b] += 1

    matrix: Dict[Tuple[str, str], float] = {}
    for role_a, counter in transition_counts.items():
        total = sum(counter.values())
        for role_b, count in counter.items():
            matrix[(role_a, role_b)] = count / total
    return matrix


def social_mobility_index(role_sequence: Sequence[Dict[int, str]], role_distance: Dict[str, int]) -> float:
    """
    Calcule l'indice de mobilité sociale moyen entre manches consécutives.

    Paramètre `role_sequence` : séquence des associations rôle par identifiant de joueur, une par manche, dans l'ordre chronologique.
    Paramètre `role_distance` : association entre rôle et rang ordinal utilisé pour mesurer la distance parcourue entre deux rôles.
    Retourne un nombre, moyenne sur l'ensemble des joueurs et des transitions disponibles de la valeur absolue de la différence de rang
    ordinal entre deux manches consécutives. Aucun effet de bord.
    """
    distances: List[int] = []
    for earlier, later in zip(role_sequence, role_sequence[1:]):
        for pid, role_a in earlier.items():
            role_b = later.get(pid)
            if role_b is not None and role_a in role_distance and role_b in role_distance:
                distances.append(abs(role_distance[role_b] - role_distance[role_a]))
    if not distances:
        return 0.0
    return sum(distances) / len(distances)
