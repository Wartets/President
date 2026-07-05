"""
Module de calcul des métriques de recherche.

Le module transforme le flux d'événements collecté par `EventLogger` en métriques multidimensionnelles. Il implémente les familles de
métriques documentées par l'architecture : métriques macro (matrice de transition de Markov des rôles, mobilité sociale, efficacité du
Putsch, impact de la taxe d'échange), métriques micro (indice de Gini de la puissance de main initiale, Card Time-To-Live, efficacité de
substitution du Joker, ratio de capture de points), métriques comportementales (taux de passe sous-optimal, indice d'agressivité à
l'ouverture, facteur de dominance de pli, taux de contre-révolution, taux d'interception manquée), métriques de complexité et théorie de
l'information (facteur de branchement moyen, entropie de l'espace d'actions), et métriques additionnelles (corrélation position d'ouverture
/ rang final, taux d'usage magique du Joker, distribution des tailles de combinaison, longueur moyenne des plis, volatilité de la
Révolution, couverture du saut de tour). La structure du module permet l'ajout de métriques supplémentaires suivant le même schéma : une
fonction pure prenant en entrée les enregistrements d'`EventLogger` ou une séquence de rôles, et retournant une valeur ou une structure
numérique.

Le module dépend de `analytics.event_logger` pour le type `EventLogger`, de `core.config` pour `GameConfig`, de `core.math_utils` pour
`f_power`, `f_points` et `rank_facial_index`, et de `core.models` pour la reconstruction des cartes à partir des enregistrements sérialisés.
"""

from __future__ import annotations

import math

import numba
import numpy as np

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from analytics.event_logger import EventLogger
from core.config import GameConfig
from core.math_utils import f_points, f_power, rank_facial_index
from core.models import ActionType, Card, Rank, Suit
from events.structural import (
    EventPlayerFinished, EventRoundStart, EventTrickClosed, EventTrickStart,
)
from events.transactional import (
    EventActionPlayed, EventActionRequest, EventAskPutsch, EventExchange,
    EventInterceptionBroadcast, EventInterceptionResolved, EventPutschInvoked,
    EventRuleTriggered,
)


# Métriques macro (échelle de la partie / des rounds)

@numba.njit(cache=True)
def _gini_from_sorted(values: np.ndarray) -> float:
    """
    Calcule l'indice de Gini d'un tableau de valeurs déjà triées par ordre croissant.

    Paramètre `values` : tableau numpy de type `float64`, valeurs triées par ordre croissant, taille quelconque positive ou nulle.
    Retourne un nombre, domaine $[0, 1]$, nul si le tableau est vide, si sa somme est nulle, ou si toutes les valeurs sont égales,
    croissant avec l'inégalité de répartition. Complexité linéaire en la taille du tableau. Aucun effet de bord. Fonction compilée en code
    machine par Numba, sans dépendance à l'interpréteur Python lors de l'exécution.
    """
    n = values.shape[0]
    total = 0.0
    cumulative = 0.0
    for i in range(n):
        total += values[i]
        cumulative += (i + 1) * values[i]
    if n == 0 or total == 0.0:
        return 0.0
    return (2.0 * cumulative) / (n * total) - (n + 1) / n


def gini_initial_hand_power(hand_powers_by_player: Dict[int, float]) -> float:
    """
    Calcule l'indice de Gini de la puissance de main initiale entre joueurs.

    Paramètre `hand_powers_by_player` : association entre identifiant de joueur et somme des puissances standard de sa main initiale.
    Retourne un nombre, domaine $[0, 1]$, nul si toutes les valeurs sont égales, croissant avec l'inégalité de répartition. Délègue le
    calcul à `_gini_from_sorted` après tri et conversion en tableau `numpy.ndarray`. Aucun effet de bord.
    """
    values = np.array(sorted(hand_powers_by_player.values()), dtype=np.float64)
    return float(_gini_from_sorted(values))


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


def putsch_efficiency_rate(logger: EventLogger) -> Dict[str, float]:
    """
    Calcule le taux de victoire du rôle `ROLE_SCUM` selon l'invocation ou non du Putsch.

    Paramètre `logger` : journal des événements de la partie.
    Retourne un dictionnaire à deux clés au plus, `'invoked'` et `'not_invoked'`, chacune associée au taux de victoire (rang de sortie nul)
    du joueur sollicité pour le Putsch (`EventAskPutsch.player_id`) à l'issue de la manche correspondante, respectivement sur les manches où
    le Putsch a été invoqué et sur celles où il ne l'a pas été. Une clé est absente du dictionnaire si aucune manche de la catégorie
    correspondante n'est disponible. Aucun effet de bord.
    """
    invoked_rounds = {e.round_id for e in logger.events_of_type(EventPutschInvoked)}
    scum_by_round: Dict[int, int] = {}
    for event in logger.events_of_type(EventAskPutsch):
        scum_by_round[event.round_id] = event.player_id

    wins: Dict[str, int] = {"invoked": 0, "not_invoked": 0}
    totals: Dict[str, int] = {"invoked": 0, "not_invoked": 0}
    for event in logger.events_of_type(EventPlayerFinished):
        scum_id = scum_by_round.get(event.round_id)
        if scum_id is None or event.player_id != scum_id:
            continue
        category = "invoked" if event.round_id in invoked_rounds else "not_invoked"
        totals[category] += 1
        if event.rank == 0:
            wins[category] += 1

    result: Dict[str, float] = {}
    for category in ("invoked", "not_invoked"):
        if totals[category] > 0:
            result[category] = wins[category] / totals[category]
    return result


@numba.njit(cache=True)
def _pearson_correlation_jit(xs: np.ndarray, ys: np.ndarray) -> float:
    """
    Calcule le coefficient de corrélation de Pearson entre deux tableaux numériques appariés.

    Paramètre `xs` : tableau numpy de type `float64`, première série de valeurs.
    Paramètre `ys` : tableau numpy de type `float64`, seconde série de valeurs, de même taille que `xs`.
    Retourne un nombre, domaine $[-1, 1]$, nul si moins de deux points sont disponibles ou si l'écart type d'une des deux séries est nul.
    Complexité linéaire en la taille des tableaux. Aucun effet de bord. Fonction compilée en code machine par Numba, sans dépendance à
    l'interpréteur Python lors de l'exécution.
    """
    n = xs.shape[0]
    if n < 2:
        return 0.0
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    cov = 0.0
    var_x = 0.0
    var_y = 0.0
    for i in range(n):
        dx = xs[i] - mean_x
        dy = ys[i] - mean_y
        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy
    if var_x == 0.0 or var_y == 0.0:
        return 0.0
    return cov / (var_x * var_y) ** 0.5


def _pearson_correlation(xs: List[float], ys: List[float]) -> float:
    """
    Calcule le coefficient de corrélation de Pearson entre deux séries appariées.

    Paramètre `xs` : première série de valeurs numériques.
    Paramètre `ys` : seconde série de valeurs numériques, de même taille que `xs`.
    Retourne un nombre, domaine $[-1, 1]$, nul si moins de deux points sont disponibles ou si l'écart type d'une des deux séries est nul.
    Délègue le calcul à `_pearson_correlation_jit` après conversion en tableaux `numpy.ndarray`. Aucun effet de bord.
    """
    if len(xs) < 2:
        return 0.0
    return float(
        _pearson_correlation_jit(np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64))
    )


def tax_weight_vp_correlation(logger: EventLogger) -> float:
    """
    Calcule la corrélation entre le poids de la taxe d'échange et le point de victoire du destinataire.

    Paramètre `logger` : journal des événements de la partie.
    Retourne un nombre, domaine $[-1, 1]$, coefficient de corrélation de Pearson entre, pour chaque transfert `EventExchange`, la somme des
    `f_points` des cartes transférées et le point de victoire `vp_earned` obtenu par le joueur destinataire à l'issue de la même manche, tel
    qu'attribué au moment de sa sortie. Retourne zéro si moins de deux points sont disponibles. Aucun effet de bord.
    """
    vp_by_round_player: Dict[Tuple[int, int], float] = {}
    for event in logger.events_of_type(EventPlayerFinished):
        vp_by_round_player[(event.round_id, event.player_id)] = event.vp_earned

    xs: List[float] = []
    ys: List[float] = []
    for event in logger.events_of_type(EventExchange):
        vp = vp_by_round_player.get((event.round_id, event.to_player))
        if vp is None:
            continue
        xs.append(float(sum(f_points(c) for c in event.cards)))
        ys.append(float(vp))
    return _pearson_correlation(xs, ys)


def opening_position_rank_correlation(logger: EventLogger) -> float:
    """
    Calcule la corrélation entre la position d'ouverture d'une manche et le rang final obtenu.

    Paramètre `logger` : journal des événements de la partie.
    Retourne un nombre, domaine $[-1, 1]$, coefficient de corrélation de Pearson entre l'identifiant du joueur ouvrant le premier pli d'une
    manche et son index de sortie $k$ au sein de cette manche, calculé sur l'ensemble des manches du journal. Aucun effet de bord.
    """
    openers: Dict[int, int] = {}
    for event in logger.events_of_type(EventTrickStart):
        if event.trick_index == 0 and event.round_id not in openers:
            openers[event.round_id] = event.opener_id

    xs: List[float] = []
    ys: List[float] = []
    for event in logger.events_of_type(EventPlayerFinished):
        opener = openers.get(event.round_id)
        if opener is None:
            continue
        xs.append(float(opener))
        ys.append(float(event.rank))
    return _pearson_correlation(xs, ys)


def e_rev_volatility(logger: EventLogger) -> float:
    """
    Calcule le nombre moyen de bascules de l'état de Révolution par manche.

    Paramètre `logger` : journal des événements de la partie.
    Retourne un nombre positif ou nul, ratio entre le nombre total d'événements `EventRuleTriggered` de nom `REVOLUTION` ou
    `DOUBLE_REVOLUTION` et le nombre de manches distinctes observées via `EventRoundStart`. Nul si aucune manche n'est disponible. Aucun
    effet de bord.
    """
    flips = [
        e for e in logger.events
        if isinstance(e, EventRuleTriggered) and e.rule_name in ("REVOLUTION", "DOUBLE_REVOLUTION")
    ]
    rounds = {e.round_id for e in logger.events_of_type(EventRoundStart)}
    if not rounds:
        return 0.0
    return len(flips) / len(rounds)


def skip_turn_coverage(logger: EventLogger, player_count: int) -> float:
    """
    Calcule la proportion moyenne de joueurs sautés par les déclenchements du Saut de Tour.

    Paramètre `logger` : journal des événements de la partie.
    Paramètre `player_count` : nombre total de joueurs $N$, entier, domaine $N \\ge 3$.
    Retourne un nombre, domaine $[0, 1]$, moyenne sur l'ensemble des événements `EventRuleTriggered` de nom `SKIP_TURN` du ratio entre le
    nombre de joueurs effectivement sautés (`magnitude`, borné à $N - 1$) et $N - 1$. Nul si aucun déclenchement n'est observé. Aucun effet
    de bord.
    """
    skips = [
        e for e in logger.events
        if isinstance(e, EventRuleTriggered) and e.rule_name == "SKIP_TURN" and e.magnitude is not None
    ]
    if not skips or player_count <= 1:
        return 0.0
    total_ratio = sum(min(e.magnitude or 0, player_count - 1) / (player_count - 1) for e in skips)
    return total_ratio / len(skips)


# Métriques micro (échelle de la main et de la carte)

def card_ttl(logger: EventLogger, player_id: int, min_rank_index: int) -> float:
    """
    Calcule le Time-To-Live moyen d'un rang de carte pour un joueur.

    Paramètre `logger` : journal des événements de la partie.
    Paramètre `player_id` : identifiant du joueur considéré.
    Paramètre `min_rank_index` : index de rang minimal (cf. `rank_facial_index`) à partir duquel une carte est considérée comme haute,
    entier, domaine $[0, 13]$.
    Retourne un nombre positif ou nul, moyenne sur les manches où `player_id` joue au moins une carte de rang facial supérieur ou égal à
    `min_rank_index`, du nombre de ses actions `ACTION_PLAY` écoulées depuis le début de la manche jusqu'à cette pose incluse. Nul si le
    joueur ne joue jamais une telle carte. Aucun effet de bord.
    """
    ttls: List[int] = []
    plays_since_round_start = 0
    found_in_round = False
    for event in logger.events:
        if isinstance(event, EventRoundStart):
            plays_since_round_start = 0
            found_in_round = False
        elif isinstance(event, EventActionPlayed) and event.player_id == player_id:
            if event.action_type == ActionType.ACTION_PLAY and not found_in_round:
                plays_since_round_start += 1
                if any(
                    (not c.is_joker()) and rank_facial_index(c.rank.value) >= min_rank_index
                    for c in event.cards_played
                ):
                    ttls.append(plays_since_round_start)
                    found_in_round = True
    if not ttls:
        return 0.0
    return sum(ttls) / len(ttls)


def joker_substitution_efficiency(logger: EventLogger, player_id: int) -> float:
    """
    Calcule l'écart moyen entre la puissance maximale intrinsèque du Joker et sa puissance substituée effective.

    Paramètre `logger` : journal des événements de la partie ou de la manche considérée.
    Paramètre `player_id` : identifiant du joueur considéré.
    Retourne un nombre positif ou nul, moyenne sur les actions `ACTION_PLAY` de `player_id` contenant au moins un Joker au sein d'une
    combinaison uniforme, de l'écart entre la puissance maximale intrinsèque du Joker (seize) et la puissance résultante substituée
    `resulting_power`. Une valeur nulle indique un usage systématiquement optimal du Joker. Nul si aucune pose de Joker n'est observée.
    Aucun effet de bord.
    """
    gaps: List[int] = []
    for event in logger.events_of_type(EventActionPlayed):
        if event.player_id != player_id or event.action_type != ActionType.ACTION_PLAY:
            continue
        if event.resulting_power is None:
            continue
        if any(c.is_joker() for c in event.cards_played):
            gaps.append(16 - event.resulting_power)
    if not gaps:
        return 0.0
    return sum(gaps) / len(gaps)


def joker_magic_mimic_rate(logger: EventLogger, config: GameConfig) -> float:
    """
    Calcule le taux de simulation de la carte magique par les Jokers joués en `Wildcard`.

    Paramètre `logger` : journal des événements de la partie ou de la manche considérée.
    Paramètre `config` : configuration de la partie, utilisée pour déterminer le rang magique effectif.
    Retourne un nombre, domaine $[0, 1]$, ratio entre le nombre de poses contenant un Joker dont la puissance substituée `resulting_power`
    correspond à la puissance standard ou inversée du rang magique effectif, et le nombre total de poses contenant au moins un Joker. Nul si
    aucun Joker n'a été joué. Aucun effet de bord.
    """
    magic_rank = config.effective_magic_card_rank()
    probe = Card(rank=Rank(magic_rank), suit=Suit.NONE)
    magic_powers = {f_power(probe, False), f_power(probe, True)}

    total_joker_plays = 0
    mimic_plays = 0
    for event in logger.events_of_type(EventActionPlayed):
        if event.action_type != ActionType.ACTION_PLAY or event.resulting_power is None:
            continue
        if not any(c.is_joker() for c in event.cards_played):
            continue
        total_joker_plays += 1
        if event.resulting_power in magic_powers:
            mimic_plays += 1
    if total_joker_plays == 0:
        return 0.0
    return mimic_plays / total_joker_plays


def capture_efficiency_ratio(logger: EventLogger, player_id: int) -> float:
    """
    Calcule le ratio de capture de points d'un joueur.

    Paramètre `logger` : journal des événements de la partie ou de la manche considérée.
    Paramètre `player_id` : identifiant du joueur considéré.
    Retourne un nombre positif ou nul, ratio entre la somme des `f_points` des cartes jouées par l'ensemble des participants aux plis
    remportés par `player_id`, et la somme des `f_points` des cartes jouées par `player_id` lui-même sur l'ensemble des plis. Nul si
    `player_id` n'a jamais joué de carte. Aucun effet de bord.
    """
    current_round = None
    current_trick = None
    trick_plays: Dict[Tuple[Optional[int], Optional[int]], List[Tuple[Card, ...]]] = defaultdict(list)
    spent = 0
    captured = 0
    for event in logger.events:
        if isinstance(event, EventTrickStart):
            current_round, current_trick = event.round_id, event.trick_index
        elif isinstance(event, EventActionPlayed) and event.action_type == ActionType.ACTION_PLAY:
            trick_plays[(current_round, current_trick)].append(event.cards_played)
            if event.player_id == player_id:
                spent += sum(f_points(c) for c in event.cards_played)
        elif isinstance(event, EventTrickClosed):
            plays = trick_plays.pop((current_round, current_trick), [])
            if event.winner_id == player_id:
                for cards in plays:
                    captured += sum(f_points(c) for c in cards)
    if spent == 0:
        return 0.0
    return captured / spent


# Métriques comportementales (profilage des agents)

def sub_optimal_pass_rate(logger: EventLogger, player_id: int) -> float:
    """
    Calcule le taux de passe sous-optimal d'un joueur (SOPR).

    Paramètre `logger` : journal des événements de la partie ou de la manche considérée.
    Paramètre `player_id` : identifiant du joueur considéré.
    Retourne un nombre, domaine $[0, 1]$, égal au ratio entre le nombre de `EventActionPlayed` de `player_id` marqués `was_suboptimal` et le
    nombre total de `EventActionPlayed` de `player_id`, nul si ce joueur n'a jamais agi. Aucun effet de bord.
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


def revolution_counter_attack_rate(logger: EventLogger, player_id: int) -> float:
    """
    Calcule la probabilité qu'un joueur déclenche une Révolution en réponse à une Révolution adverse.

    Paramètre `logger` : journal des événements de la partie ou de la manche considérée.
    Paramètre `player_id` : identifiant du joueur considéré.
    Retourne un nombre, domaine $[0, 1]$, ratio entre le nombre de déclenchements `REVOLUTION` de `player_id` survenant directement après un
    déclenchement `REVOLUTION` par un autre joueur, et le nombre total de déclenchements `REVOLUTION` observés par un autre joueur. Nul si
    aucune opportunité de contre-attaque n'est observée. Aucun effet de bord.
    """
    opportunities = 0
    counters = 0
    awaiting_counter = False
    for event in logger.events:
        if isinstance(event, EventRoundStart):
            awaiting_counter = False
            continue
        if isinstance(event, EventRuleTriggered) and event.rule_name == "REVOLUTION":
            if event.triggering_player_id == player_id:
                if awaiting_counter:
                    counters += 1
                awaiting_counter = False
            else:
                opportunities += 1
                awaiting_counter = True
    if opportunities == 0:
        return 0.0
    return counters / opportunities


def missed_interception_rate(logger: EventLogger) -> float:
    """
    Calcule le taux d'opportunités d'interception mathématiquement possibles et non saisies.

    Paramètre `logger` : journal des événements de la partie ou de la manche considérée.
    Retourne un nombre, domaine $[0, 1]$, ratio entre le nombre de diffusions `EventInterceptionBroadcast` pour lesquelles au moins un joueur
    éligible détient, par reconstruction des mains à partir de `EventRoundStart`, `EventExchange` et `EventActionPlayed`, une carte de même
    rang et de même couleur que la carte diffusée mais dont la résolution `EventInterceptionResolved` associée ne désigne aucun intercepteur,
    et le nombre total de diffusions pour lesquelles une telle carte existait. Nul si aucune opportunité de ce type n'est observée. Aucun
    effet de bord.
    """
    hands: Dict[int, List[Card]] = {}
    total_opportunities = 0
    missed = 0
    pending_available = False
    for event in logger.events:
        if isinstance(event, EventRoundStart):
            hands = {pid: list(cards) for pid, cards in event.initial_hands.items()}
        elif isinstance(event, EventExchange):
            remaining = list(hands.get(event.from_player, []))
            for card in event.cards:
                if card in remaining:
                    remaining.remove(card)
            hands[event.from_player] = remaining
            hands[event.to_player] = hands.get(event.to_player, []) + list(event.cards)
        elif isinstance(event, EventActionPlayed):
            remaining = list(hands.get(event.player_id, []))
            for card in event.cards_played:
                if card in remaining:
                    remaining.remove(card)
            hands[event.player_id] = remaining
        elif isinstance(event, EventInterceptionBroadcast):
            played = event.played_card
            pending_available = played is not None and any(
                any(
                    not c.is_joker() and c.rank == played.rank and c.suit == played.suit
                    for c in hands.get(pid, [])
                )
                for pid in event.eligible_player_ids
            )
            if pending_available:
                total_opportunities += 1
        elif isinstance(event, EventInterceptionResolved):
            if pending_available and event.interceptor_id is None:
                missed += 1
            pending_available = False
    if total_opportunities == 0:
        return 0.0
    return missed / total_opportunities


# Métriques de complexité et théorie de l'information

def branching_factor_average(logger: EventLogger, player_id: Optional[int] = None) -> float:
    """
    Calcule le facteur de branchement moyen à la décision.

    Paramètre `logger` : journal des événements de la partie ou de la manche considérée.
    Paramètre `player_id` : identifiant du joueur considéré, ou `None` pour agréger l'ensemble des joueurs.
    Retourne un nombre positif ou nul, moyenne de `legal_action_count` sur l'ensemble des `EventActionRequest` filtrés. Nul si aucun
    événement ne correspond au filtre. Aucun effet de bord.
    """
    counts = [
        e.legal_action_count for e in logger.events_of_type(EventActionRequest)
        if player_id is None or e.player_id == player_id
    ]
    if not counts:
        return 0.0
    return sum(counts) / len(counts)


def action_space_entropy(logger: EventLogger, player_id: Optional[int] = None) -> float:
    """
    Calcule l'entropie de Shannon de l'espace d'actions à la décision.

    Paramètre `logger` : journal des événements de la partie ou de la manche considérée.
    Paramètre `player_id` : identifiant du joueur considéré, ou `None` pour agréger l'ensemble des joueurs.
    Retourne un nombre positif ou nul, en bits, entropie de Shannon de la distribution empirique des valeurs de `legal_action_count`
    observées sur les `EventActionRequest` filtrés. Nulle si moins de deux valeurs sont observées. Aucun effet de bord.
    """
    counts = [
        e.legal_action_count for e in logger.events_of_type(EventActionRequest)
        if player_id is None or e.player_id == player_id
    ]
    if not counts:
        return 0.0
    frequency = Counter(counts)
    total = len(counts)
    entropy = 0.0
    for freq in frequency.values():
        p = freq / total
        entropy -= p * math.log2(p)
    return entropy


# Métriques additionnelles

def combination_size_distribution(logger: EventLogger, player_id: Optional[int] = None) -> Dict[int, int]:
    """
    Calcule la distribution des tailles de combinaison effectivement jouées.

    Paramètre `logger` : journal des événements de la partie ou de la manche considérée.
    Paramètre `player_id` : identifiant du joueur considéré, ou `None` pour agréger l'ensemble des joueurs.
    Retourne un dictionnaire associant une taille de combinaison à son nombre d'occurrences parmi les actions `ACTION_PLAY` filtrées. Aucun
    effet de bord.
    """
    counter: Counter = Counter()
    for event in logger.events_of_type(EventActionPlayed):
        if event.action_type != ActionType.ACTION_PLAY:
            continue
        if player_id is not None and event.player_id != player_id:
            continue
        counter[len(event.cards_played)] += 1
    return dict(counter)


def trick_length_average(logger: EventLogger) -> float:
    """
    Calcule la longueur moyenne des plis en nombre d'actions.

    Paramètre `logger` : journal des événements de la partie ou de la manche considérée.
    Retourne un nombre positif ou nul, moyenne du nombre d'`EventActionPlayed` observés entre deux `EventTrickStart` consécutifs. Nul si
    aucun pli n'est observé. Aucun effet de bord.
    """
    lengths: List[int] = []
    current_length = 0
    started = False
    for event in logger.events:
        if isinstance(event, EventTrickStart):
            if started:
                lengths.append(current_length)
            current_length = 0
            started = True
        elif isinstance(event, EventActionPlayed):
            current_length += 1
    if started:
        lengths.append(current_length)
    if not lengths:
        return 0.0
    return sum(lengths) / len(lengths)
