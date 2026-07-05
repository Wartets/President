"""
Module d'évaluation pure des règles du jeu.

Le module ne provoque aucun effet de bord, n'effectue aucune entrée-sortie et ne dépend d'aucun état mutable externe. Il expose les fonctions de
construction et de distribution du paquet, de validation des combinaisons (uniformes, suites, Jokers), de calcul de la puissance résultante d'une
combinaison, de détection des déclencheurs de règles avancées (révolution, double révolution, clôture magique, saut de tour, interception) et de
résolution déterministe des égalités de puissance lors de la sélection de cartes maximales. Le module implémente également la matrice de
compatibilité croisée entre règles avancées.

Le module dépend de `core.config` pour les constantes de configuration et de `core.models` pour les types `Card`, `Hand`, `Suit`, `Rank` et de
`core.math_utils` pour les fonctions de puissance. Toute résolution d'égalité dépend uniquement de l'ordre des couleurs défini par `Suit` et,
lorsque fourni, d'un générateur pseudo-aléatoire initialisé par `random_seed`, garantissant la reproductibilité exigée par `GameConfig`.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numba
import numpy as np

from core.config import GameConfig
from core.math_utils import f_power, f_std, rank_facial_index
from core.models import Card, Hand, RANK_ORDER, Rank, Suit


@numba.njit(cache=True)
def _is_uniform_power_array(powers: np.ndarray) -> bool:
    """
    Détermine si un tableau de puissances est uniforme.

    Paramètre `powers` : tableau numpy d'entiers 64 bits, puissances des cartes non Joker d'une combinaison candidate, taille quelconque
    positive ou nulle.
    Retourne un booléen, faux si `powers` est vide, vrai si toutes les valeurs de `powers` sont égales entre elles. Complexité linéaire en la
    taille de `powers`. Aucun effet de bord. Fonction compilée en code machine par Numba, sans dépendance à l'interpréteur Python lors de
    l'exécution.
    """
    if powers.shape[0] == 0:
        return False
    first = powers[0]
    for i in range(1, powers.shape[0]):
        if powers[i] != first:
            return False
    return True


# Construction et distribution du paquet

def num_decks(config: GameConfig) -> int:
    """
    Détermine le nombre de paquets de 52 cartes utilisés en fonction de $N$.

    Paramètre `config` : configuration de la partie, type `GameConfig`.
    Retourne un entier strictement positif. Si `deck_scaling_auto` est faux et `forced_deck_count` est fourni, cette valeur est retournée telle
    quelle. Sinon, la valeur retournée est $\\max(1, \\lfloor(N-1)/4\\rfloor + 1)$. Aucun effet de bord.
    """
    if not config.deck_scaling_auto and config.forced_deck_count is not None:
        return config.forced_deck_count
    return max(1, (config.player_count - 1) // 4 + 1)


def build_deck(config: GameConfig) -> List[Card]:
    """
    Construit la liste ordonnée des cartes composant le paquet global.

    Paramètre `config` : configuration de la partie, type `GameConfig`.
    Retourne une liste de `Card` de taille $N_D \\times (52 + 2 \\times \\text{int}(\\text{use\\_jokers}))$. L'ordre de la liste n'a pas de
    signification ; il est destiné à être mélangé par `deal_hands`. Aucun effet de bord.
    """
    decks = num_decks(config)
    standard_ranks = [r for r in RANK_ORDER if r != "JOKER"]
    cards: List[Card] = []
    for deck_index in range(decks):
        for rank_value in standard_ranks:
            for suit in (Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS):
                cards.append(Card(rank=Rank(rank_value), suit=suit))
        if config.use_jokers:
            cards.append(Card(rank=Rank.JOKER, suit=Suit.NONE, instance_id=deck_index * 2))
            cards.append(Card(rank=Rank.JOKER, suit=Suit.NONE, instance_id=deck_index * 2 + 1))
    return cards


def deal_hands(
    config: GameConfig,
    deck: Sequence[Card],
    round_index: int,
    strict_remainder_target_seat: Optional[int] = None,
) -> List[Hand]:
    """
    Distribue le paquet mélangé entre les $N$ joueurs.

    Paramètre `config` : configuration de la partie, type `GameConfig`.
    Paramètre `deck` : séquence de `Card` à distribuer.
    Paramètre `round_index` : index $m$ de la manche courante, entier positif ou nul, utilisé pour dériver un mélange distinct par manche à
    partir de `random_seed`.
    Paramètre `strict_remainder_target_seat` : index de siège recevant le reste de la division modulaire lorsque `strict_remainder_allocation`
    est vrai, ou `None` si cette règle est inactive ou si le rôle ciblé n'est pas encore déterminé pour cette manche.
    Retourne une liste de `Hand`, de taille $N$, telle que chaque main possède une taille dans $\\{\\lfloor S/N \\rfloor, \\lceil S/N \\rceil\\}$ si
    `strict_remainder_allocation` est faux, ou une taille excédentaire concentrée sur `strict_remainder_target_seat` sinon. Aucun effet de bord
    sur `deck` ; une copie mélangée est utilisée en interne.
    """
    n = config.player_count
    rng = random.Random(f"{config.random_seed}:{round_index}")
    shuffled = list(deck)
    rng.shuffle(shuffled)

    total = len(shuffled)
    base_size = total // n
    remainder = total % n

    hands_cards: List[List[Card]] = [[] for _ in range(n)]

    if remainder != 0 and config.strict_remainder_allocation and strict_remainder_target_seat is not None:
        cursor = 0
        for seat in range(n):
            size = base_size
            if seat == strict_remainder_target_seat:
                size += remainder
            hands_cards[seat] = shuffled[cursor:cursor + size]
            cursor += size
    else:
        for index, card in enumerate(shuffled):
            hands_cards[index % n].append(card)

    return [Hand(cards=tuple(cards)) for cards in hands_cards]


# Résolution des égalités de puissance

_SUIT_TIEBREAK_ORDER = {
    Suit.SPADES: 0,
    Suit.HEARTS: 1,
    Suit.DIAMONDS: 2,
    Suit.CLUBS: 3,
    Suit.NONE: 4,
}


def _tiebreak_key(card: Card) -> Tuple[int, int]:
    """
    Construit la clé de tri déterministe utilisée pour départager deux cartes
    de puissance égale.

    Paramètre `card` : carte considérée, type `Card`.
    Retourne un tuple d'entiers, comparable lexicographiquement, dérivé de l'ordre de couleur défini par `Suit` puis de l'identifiant d'exemplaire.
    Aucun effet de bord.
    """
    return (_SUIT_TIEBREAK_ORDER[card.suit], card.instance_id)


def max_power_cards(hand: Hand, count: int, e_rev: bool) -> Tuple[Card, ...]:
    """
    Sélectionne les cartes de puissance maximale d'une main.

    Paramètre `hand` : main considérée, type `Hand`.
    Paramètre `count` : nombre de cartes à sélectionner, entier, domaine $[0, hand.size()]$.
    Paramètre `e_rev` : état de révolution utilisé pour le calcul de puissance.
    Retourne un tuple de `Card` de taille `count`, trié par puissance décroissante puis par la clé de départage déterministe des couleurs.
    Aucun effet de bord.
    """
    ordered = sorted(
        hand.cards,
        key=lambda c: (-f_power(c, e_rev), _tiebreak_key(c)),
    )
    return tuple(ordered[:count])


def random_cards(hand: Hand, count: int, rng: random.Random) -> Tuple[Card, ...]:
    """
    Sélectionne un sous-ensemble uniforme de cartes d'une main.

    Paramètre `hand` : main considérée, type `Hand`.
    Paramètre `count` : nombre de cartes à sélectionner, entier, domaine $[0, hand.size()]$.
    Paramètre `rng` : générateur pseudo-aléatoire initialisé par l'appelant à partir de `random_seed`, garantissant la reproductibilité.
    Retourne un tuple de `Card` de taille `count`, résultat d'un tirage sans remise. Effet de bord : consomme l'état interne de `rng`.
    """
    population = list(hand.cards)
    return tuple(rng.sample(population, count))


# Validation des combinaisons

def is_valid_uniform_combination(
    cards: Sequence[Card],
    e_rev: bool,
    declared_power: Optional[int],
) -> bool:
    """
    Détermine si une combinaison constitue un ensemble de puissance uniforme.

    Paramètre `cards` : séquence de `Card` posées.
    Paramètre `e_rev` : état de révolution courant.
    Paramètre `declared_power` : puissance déclarée pour les Jokers présents dans `cards`, obligatoire si `cards` contient au moins un Joker.
    Retourne un booléen. La combinaison est valide si toutes les cartes non Joker partagent la même puissance et si, en présence de Jokers, la
    puissance déclarée est fournie. Une combinaison vide est invalide. Aucun effet de bord.
    """
    if not cards:
        return False
    non_jokers = [c for c in cards if not c.is_joker()]
    has_joker = len(non_jokers) != len(cards)
    if has_joker and declared_power is None:
        return False
    if not non_jokers:
        return True
    powers_array = np.array([f_power(c, e_rev) for c in non_jokers], dtype=np.int64)
    if not _is_uniform_power_array(powers_array):
        return False
    if has_joker:
        return declared_power == int(powers_array[0])
    return True


def combination_power(
    cards: Sequence[Card],
    e_rev: bool,
    declared_power: Optional[int],
) -> int:
    """
    Calcule la puissance résultante d'une combinaison valide.

    Paramètre `cards` : séquence de `Card` posées, combinaison supposée valide au sens de `is_valid_uniform_combination` ou
    `is_valid_sequence_combination`.
    Paramètre `e_rev` : état de révolution courant.
    Paramètre `declared_power` : puissance déclarée pour les Jokers présents.
    Retourne un entier, égal à la puissance uniforme partagée par les cartes non Joker, ou à `declared_power` si `cards` ne contient que des Jokers.
    Aucun effet de bord.
    """
    non_jokers = [c for c in cards if not c.is_joker()]
    if non_jokers:
        return f_power(non_jokers[0], e_rev)
    return declared_power if declared_power is not None else 0


@numba.njit(cache=True)
def _is_consecutive_run(powers: np.ndarray) -> bool:
    """
    Détermine si un tableau de puissances triées forme une suite consécutive.

    Paramètre `powers` : tableau numpy d'entiers 64 bits, puissances résolues d'une combinaison candidate à la suite, triées par ordre
    croissant, taille quelconque positive ou nulle.
    Retourne un booléen, vrai si moins de deux éléments sont présents, faux dès que deux valeurs consécutives ne diffèrent pas exactement de
    un. Complexité linéaire en la taille de `powers`. Aucun effet de bord. Fonction compilée en code machine par Numba, sans dépendance à
    l'interpréteur Python lors de l'exécution.
    """
    if powers.shape[0] < 2:
        return True
    for i in range(powers.shape[0] - 1):
        if powers[i + 1] - powers[i] != 1:
            return False
    return True


def is_valid_sequence_combination(
    cards: Sequence[Card],
    e_rev: bool,
    joker_declared_powers: Optional[Dict[int, int]] = None,
) -> bool:
    """
    Détermine si une combinaison constitue une suite valide.

    Paramètre `cards` : séquence de `Card` posées, dans un ordre quelconque.
    Paramètre `e_rev` : état de révolution courant.
    Paramètre `joker_declared_powers` : association entre l'index d'un Joker dans `cards` (après tri par puissance croissante) et sa puissance
    déclarée, permettant de combler un intervalle de la suite.
    Retourne un booléen. La combinaison est valide si sa taille est supérieure ou égale à trois et si, une fois les cartes ordonnées par
    puissance croissante (les Jokers prenant la puissance déclarée), chaque puissance successive dépasse la précédente d'exactement un. Aucun effet
    de bord.
    """
    if len(cards) < 3:
        return False
    joker_declared_powers = joker_declared_powers or {}

    resolved_powers: List[int] = []
    non_joker_sorted = sorted(
        [c for c in cards if not c.is_joker()], key=lambda c: f_power(c, e_rev)
    )
    joker_count = len(cards) - len(non_joker_sorted)

    if joker_count == 0:
        resolved_powers = [f_power(c, e_rev) for c in non_joker_sorted]
    else:
        if len(joker_declared_powers) != joker_count:
            return False
        resolved_powers = sorted(
            [f_power(c, e_rev) for c in non_joker_sorted]
            + list(joker_declared_powers.values())
        )

    if len(set(resolved_powers)) != len(resolved_powers):
        return False

    powers_array = np.array(resolved_powers, dtype=np.int64)
    return _is_consecutive_run(powers_array)


# Déclencheurs de règles avancées (matrice de compatibilité)

def _remapped_magic_rank(config: GameConfig, e_rev: bool) -> str:
    """
    Calcule le rang magique effectif après remappage éventuel par la révolution.

    Paramètre `config` : configuration de la partie.
    Paramètre `e_rev` : état de révolution courant, utilisé pour appliquer la résolution [C] de la matrice de compatibilité.
    Retourne une chaîne, rang facial magique effectif, égal à `config.effective_magic_card_rank()` si `e_rev` est faux, ou à son
    symétrique par rapport à la hiérarchie standard si `e_rev` est vrai. Aucun effet de bord.
    """
    magic_rank = config.effective_magic_card_rank()
    magic_index_std = rank_facial_index(magic_rank)
    if e_rev:
        mirror_index = len(RANK_ORDER) - 2 - magic_index_std
        magic_rank = RANK_ORDER[mirror_index]
    return magic_rank


def triggers_magic_closure(
    cards: Sequence[Card],
    config: GameConfig,
    e_rev: bool,
    declared_power: Optional[int],
    is_sequence: bool,
) -> bool:
    """
    Détermine si une combinaison déclenche la clôture magique immédiate.

    Paramètre `cards` : séquence de `Card` posées.
    Paramètre `config` : configuration de la partie.
    Paramètre `e_rev` : état de révolution courant, utilisé pour le remappage du rang magique conformément à la résolution [C] de la matrice de
    compatibilité.
    Paramètre `declared_power` : puissance déclarée pour les Jokers présents, utilisée pour appliquer la résolution [E] : un Joker déclaré à
    la valeur du rang magique ne déclenche pas la clôture.
    Paramètre `is_sequence` : indique si la combinaison est une suite, utilisé pour appliquer la résolution [G] : une suite contenant une carte
    magique déclenche la clôture sauf si la combinaison peut se limiter à la seule règle de suite selon `magic_single_clears_all`.
    Retourne un booléen. Aucun effet de bord.
    """
    if not config.effective_magic_card_enabled():
        return False

    magic_rank = _remapped_magic_rank(config, e_rev)

    for card in cards:
        if card.is_joker():
            # Résolution [E] : un Joker déclaré à la valeur magique ne
            # déclenche pas la clôture.
            continue
        if card.rank.value == magic_rank:
            return True
    return False


def is_magic_single_clear(card: Card, config: GameConfig, e_rev: bool) -> bool:
    """
    Détermine si une carte unique clôture un pli de taille quelconque par effet de clôture magique.

    Paramètre `card` : carte unique candidate à la clôture.
    Paramètre `config` : configuration de la partie.
    Paramètre `e_rev` : état de révolution courant.
    Retourne un booléen, vrai si une règle de clôture magique est active, si `effective_magic_single_clears_all` est vrai, et si `card`
    n'est pas un Joker et possède le rang magique effectif remappé selon `e_rev`. Une telle carte clôture un pli de taille $X \\ge 1$
    quelconque indépendamment de la puissance courante du pli. Aucun effet de bord.
    """
    if not config.effective_magic_card_enabled():
        return False
    if not config.effective_magic_single_clears_all():
        return False
    if card.is_joker():
        return False
    return card.rank.value == _remapped_magic_rank(config, e_rev)


def triggers_revolution(
    cards: Sequence[Card],
    config: GameConfig,
    is_sequence: bool,
) -> bool:
    """
    Détermine si une combinaison déclenche une révolution standard.

    Paramètre `cards` : séquence de `Card` posées.
    Paramètre `config` : configuration de la partie.
    Paramètre `is_sequence` : indique si la combinaison est une suite, utilisé pour appliquer la résolution [B] : une suite ne déclenche jamais
    de révolution, quelle que soit sa taille.
    Retourne un booléen. La combinaison déclenche une révolution si `revolution_enabled` est vrai, si la combinaison n'est pas une suite, si
    sa taille est supérieure ou égale à quatre, et si elle ne contient aucun Joker, conformément à la résolution [A]. Aucun effet de bord.
    """
    if not config.revolution_enabled:
        return False
    if is_sequence:
        return False
    if any(c.is_joker() for c in cards):
        return False
    return len(cards) >= 4


def triggers_double_revolution(
    cards: Sequence[Card],
    config: GameConfig,
    is_sequence: bool,
) -> bool:
    """
    Détermine si une combinaison déclenche une double révolution verrouillante.

    Paramètre `cards` : séquence de `Card` posées.
    Paramètre `config` : configuration de la partie.
    Paramètre `is_sequence` : indique si la combinaison est une suite.
    Retourne un booléen. La combinaison déclenche une double révolution si `double_revolution_enabled` est vrai, si la combinaison n'est pas une
    suite, si sa taille est supérieure ou égale à huit, et si elle ne contient aucun Joker. Aucun effet de bord.
    """
    if not config.double_revolution_enabled:
        return False
    if is_sequence:
        return False
    if any(c.is_joker() for c in cards):
        return False
    return len(cards) >= 8


def triggers_skip_turn(cards: Sequence[Card], config: GameConfig) -> int:
    """
    Détermine le nombre de joueurs sautés par une combinaison.

    Paramètre `cards` : séquence de `Card` posées.
    Paramètre `config` : configuration de la partie.
    Retourne un entier positif ou nul. La valeur retournée est le nombre de cartes de `cards` dont le rang est égal à `skip_turn_rank` si
    `skip_turn_enabled` est vrai, conformément à la résolution [F] pour les suites. Aucun effet de bord.
    """
    if not config.skip_turn_enabled:
        return 0
    return sum(1 for c in cards if not c.is_joker() and c.rank.value == config.skip_turn_rank)


def can_intercept(
    played_card: Card,
    candidate_card: Card,
    config: GameConfig,
) -> bool:
    """
    Détermine si une carte candidate permet d'intercepter une carte posée.

    Paramètre `played_card` : carte unique dernièrement posée.
    Paramètre `candidate_card` : carte détenue par un joueur hors-tour candidate à l'interception.
    Paramètre `config` : configuration de la partie.
    Retourne un booléen, vrai si `interception_enabled` est vrai et si `candidate_card` possède exactement le même rang et la même couleur que
    `played_card`. Aucun effet de bord.
    """
    if not config.interception_enabled:
        return False
    return (
        candidate_card.rank == played_card.rank
        and candidate_card.suit == played_card.suit
    )


def _group_by_power(hand: Hand, e_rev: bool) -> Tuple[Dict[int, List[Card]], List[Card]]:
    """
    Répartit les cartes d'une main par puissance, en isolant les Jokers.

    Paramètre `hand` : main considérée.
    Paramètre `e_rev` : état de révolution utilisé pour le calcul de puissance.
    Retourne un tuple composé d'une association entre puissance et liste de `Card` non Joker partageant cette puissance, et d'une liste des Jokers de
    la main. Aucun effet de bord.
    """
    groups: Dict[int, List[Card]] = {}
    jokers: List[Card] = []
    for card in hand.cards:
        if card.is_joker():
            jokers.append(card)
        else:
            groups.setdefault(f_power(card, e_rev), []).append(card)
    return groups, jokers


def generate_uniform_plays(
    hand: Hand,
    e_rev: bool,
    required_size: Optional[int],
    min_power_exclusive: Optional[int],
) -> List[Tuple[Tuple[Card, ...], Optional[int]]]:
    """
    Génère les combinaisons uniformes légales disponibles dans une main.

    Paramètre `hand` : main considérée.
    Paramètre `e_rev` : état de révolution courant.
    Paramètre `required_size` : taille $X$ imposée par le pli en cours, ou `None` si le pli est vide et que toute taille est recevable.
    Paramètre `min_power_exclusive` : puissance courante du pli à dépasser strictement, ou `None` si le pli est vide.
    Retourne une liste de tuples `(cards, declared_power)`, chaque élément représentant une combinaison jouable distincte par sa puissance et sa
    taille, les Jokers étant utilisés en complément des cartes naturelles lorsque nécessaire pour atteindre la taille requise. Aucun effet de
    bord.
    """
    groups, jokers = _group_by_power(hand, e_rev)
    num_jokers = len(jokers)
    results: List[Tuple[Tuple[Card, ...], Optional[int]]] = []

    for power, cards in groups.items():
        if min_power_exclusive is not None and power <= min_power_exclusive:
            continue
        max_size = len(cards) + num_jokers
        sizes = [required_size] if required_size is not None else list(range(1, max_size + 1))
        for size in sizes:
            if size is None or size < 1 or size > max_size:
                continue
            jokers_needed = size - len(cards) if size > len(cards) else 0
            if jokers_needed > num_jokers:
                continue
            chosen_natural = tuple(cards[:size]) if size <= len(cards) else tuple(cards)
            chosen_jokers = tuple(jokers[:jokers_needed])
            combo = chosen_natural + chosen_jokers
            declared = power if chosen_jokers else None
            results.append((combo, declared))

    if num_jokers > 0:
        power_floor = min_power_exclusive if min_power_exclusive is not None else 2
        sizes = [required_size] if required_size is not None else list(range(1, num_jokers + 1))
        for size in sizes:
            if size is None or size < 1 or size > num_jokers:
                continue
            chosen_jokers = tuple(jokers[:size])
            for declared_power in range(max(power_floor + 1, 3), 16):
                results.append((chosen_jokers, declared_power))
    return results


def generate_sequence_plays(
    hand: Hand,
    e_rev: bool,
    required_size: Optional[int],
    min_power_exclusive: Optional[int],
) -> List[Tuple[Tuple[Card, ...], Dict[int, int]]]:
    """
    Génère les combinaisons de type suite légales disponibles dans une main.

    Paramètre `hand` : main considérée.
    Paramètre `e_rev` : état de révolution courant.
    Paramètre `required_size` : taille $X$ imposée par la suite active, ou `None` si le pli est vide.
    Paramètre `min_power_exclusive` : puissance minimale de la suite active à dépasser strictement, ou `None` si le pli est vide.
    Retourne une liste de tuples `(cards, joker_declared_powers)`, chaque élément représentant une suite jouable, les Jokers comblant au plus un
    intervalle chacun parmi les puissances présentes. Aucun effet de bord.
    """
    groups, jokers = _group_by_power(hand, e_rev)
    available_powers = sorted(groups.keys())
    if not available_powers:
        num_jokers = len(jokers)
        if num_jokers < 3:
            return []
        results_pure: List[Tuple[Tuple[Card, ...], Dict[int, int]]] = []
        power_floor = min_power_exclusive if min_power_exclusive is not None else 2
        sizes_pure = [required_size] if required_size is not None else list(range(3, num_jokers + 1))
        for size in sizes_pure:
            if size is None or size < 3 or size > num_jokers:
                continue
            highest_start = 15 - size + 1
            for start in range(power_floor + 1, highest_start + 1):
                cards: Tuple[Card, ...] = tuple(jokers[:size])
                joker_map = {position: start + position for position in range(size)}
                results_pure.append((cards, joker_map))
        return results_pure

    lowest, highest = available_powers[0], available_powers[-1]
    full_range = list(range(lowest, highest + 1))
    results: List[Tuple[Tuple[Card, ...], Dict[int, int]]] = []

    sizes = [required_size] if required_size is not None else list(range(3, len(full_range) + 1))
    for size in sizes:
        if size is None or size < 3:
            continue
        for start_index in range(0, len(full_range) - size + 1):
            window = full_range[start_index:start_index + size]
            min_power = window[0]
            if min_power_exclusive is not None and min_power <= min_power_exclusive:
                continue
            missing = [p for p in window if p not in groups]
            if len(missing) > len(jokers):
                continue
            selected_cards: List[Card] = []
            joker_map: Dict[int, int] = {}
            joker_cursor = 0
            for position, power in enumerate(window):
                if power in groups:
                    selected_cards.append(groups[power][0])
                else:
                    selected_cards.append(jokers[joker_cursor])
                    joker_map[position] = power
                    joker_cursor += 1
            cards: Tuple[Card, ...] = tuple(selected_cards)
            results.append((cards, joker_map))
    return results


def is_action_valid(
    cards: Sequence[Card],
    declared_power: Optional[int],
    hand: Hand,
    e_rev: bool,
    required_size: Optional[int],
    min_power_exclusive: Optional[int],
    is_sequence_context: bool,
    sequence_min_power: Optional[int],
    straights_enabled: bool,
    allow_equal_power: bool = False,
    config: Optional[GameConfig] = None,
) -> bool:
    """
    Valide strictement une combinaison retournée par un agent avant son
    application à l'état.

    Paramètre `cards` : combinaison proposée par l'agent.
    Paramètre `declared_power` : puissance déclarée pour les Jokers présents.
    Paramètre `hand` : main courante du joueur, utilisée pour vérifier que `cards` en constitue un sous-ensemble effectif.
    Paramètre `e_rev` : état de révolution courant.
    Paramètre `required_size` : taille imposée par le pli en cours, ou `None` si le pli est vide.
    Paramètre `min_power_exclusive` : puissance à dépasser strictement, ou `None` si le pli est vide.
    Paramètre `is_sequence_context` : indique si le pli en cours est une suite active.
    Paramètre `sequence_min_power` : puissance minimale de la suite active, ou `None`.
    Paramètre `straights_enabled` : indique si les suites sont autorisées par la configuration.
    Paramètre `allow_equal_power` : autorise une puissance résultante égale à `min_power_exclusive` plutôt que strictement supérieure, conformément à
    `GameConfig.skip_on_equal`.
    Paramètre `config` : configuration de la partie, optionnelle. Lorsqu'elle est fournie, autorise la clôture magique à carte unique
    (`is_magic_single_clear`) indépendamment de `required_size`. `None` désactive cette vérification.
    Retourne un booléen, vrai si `cards` constitue un sous-ensemble de `hand`, si sa taille correspond à `required_size` lorsque celui-ci est
    fourni, et si `cards` constitue soit une combinaison uniforme valide de puissance strictement supérieure à `min_power_exclusive`, soit, lorsque
    `straights_enabled` est vrai, une suite valide de puissance minimale strictement supérieure à `sequence_min_power`. Retourne également vrai,
    sans autre vérification, si `cards` est réduite à une unique carte satisfaisant `is_magic_single_clear` selon `config`. Aucun effet de bord.
    """
    if not cards:
        return False
    remaining_hand_cards = list(hand.cards)
    for card in cards:
        if card not in remaining_hand_cards:
            return False
        remaining_hand_cards.remove(card)

    if config is not None and len(cards) == 1 and is_magic_single_clear(cards[0], config, e_rev):
        return True

    if required_size is not None and len(cards) != required_size:
        return False

    if is_sequence_context:
        if not straights_enabled:
            return False
        joker_positions = [i for i, c in enumerate(cards) if c.is_joker()]
        if declared_power is None and joker_positions:
            return False
        joker_map = {}
        if joker_positions and declared_power is not None:
            non_joker_powers = sorted(
                f_power(c, e_rev) for c in cards if not c.is_joker()
            )
            base = min(non_joker_powers) if non_joker_powers else declared_power
            for offset, position in enumerate(joker_positions):
                joker_map[position] = declared_power + offset if len(joker_positions) > 1 else declared_power
        if not is_valid_sequence_combination(cards, e_rev, joker_map or None):
            return False
        min_power = min(
            f_power(c, e_rev) if not c.is_joker() else min(joker_map.values(), default=0)
            for c in cards
        )
        if sequence_min_power is not None:
            if allow_equal_power:
                if min_power < sequence_min_power:
                    return False
            elif min_power <= sequence_min_power:
                return False
        return True

    if not is_valid_uniform_combination(cards, e_rev, declared_power):
        return False
    power = combination_power(cards, e_rev, declared_power)
    if min_power_exclusive is not None:
        if allow_equal_power:
            if power < min_power_exclusive:
                return False
        elif power <= min_power_exclusive:
            return False
    return True


def matches_finish_penalty(
    final_cards: Sequence[Card],
    config: GameConfig,
    e_rev_before: bool,
    triggered_revolution: bool,
) -> bool:
    """
    Détermine si une combinaison de sortie déclenche la pénalité étendue.

    Paramètre `final_cards` : combinaison ayant permis de vider la main du joueur.
    Paramètre `config` : configuration de la partie.
    Paramètre `e_rev_before` : état de révolution immédiatement avant la pose de `final_cards`, utilisé pour déterminer le rang suprême courant.
    Paramètre `triggered_revolution` : indique si `final_cards` a déclenché une révolution au sens de `triggers_revolution`.
    Retourne un booléen, vrai si au moins une des conditions est satisfaite : présence du rang suprême courant, ou, lorsque `finish_penalty_extended`
    est vrai, présence d'un Joker sous `no_finish_on_joker`, ou déclenchement d'une révolution sous `no_finish_on_revolution`. Aucun effet de bord.
    """
    supreme_rank = "3" if e_rev_before else "2"
    if any(not c.is_joker() and c.rank.value == supreme_rank for c in final_cards):
        return True
    if config.no_finish_on_joker and any(c.is_joker() for c in final_cards):
        return True
    if config.no_finish_on_revolution and triggered_revolution:
        return True
    return False
