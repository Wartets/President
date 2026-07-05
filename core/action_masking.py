"""
Module de masquage d'actions vectorisé pour le Fast-Path d'entraînement.

Le module fournit `build_action_mask_batch`, fonction pure convertissant un lot d'options légales par état, telles que retournées par
`generate_uniform_plays` et `generate_sequence_plays`, en un unique masque booléen `numpy.ndarray` exploitable directement en sortie du
`forward pass` d'un réseau de neurones. Le module ne recalcule aucune légalité ; il se contente de projeter des options déjà validées par
`core.rules_engine` sur un espace d'action indexé fixe.

Le module dépend de `core.models` pour le type `Card` et `numpy` pour la représentation tensorielle du masque. Aucun effet de bord global.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.models import Card


def build_action_space_index(max_power: int = 16, max_combo_size: int = 20) -> Dict[Tuple[int, int], int]:
    """
    Construit l'index fixe de l'espace d'action discret uniforme.

    Paramètre `max_power` : puissance maximale représentable, entier, domaine $[3, 16]$, seize couvrant le Joker.
    Paramètre `max_combo_size` : taille maximale de combinaison représentable, entier strictement positif, bornée par $4 \\times N_D$ en
    pratique.
    Retourne un dictionnaire associant un couple `(power, size)` à un index entier unique, dense sur $[0, max\\_power \\times max\\_combo\\_size)$.
    Aucun effet de bord.
    """
    index: Dict[Tuple[int, int], int] = {}
    cursor = 0
    for power in range(3, max_power + 1):
        for size in range(1, max_combo_size + 1):
            index[(power, size)] = cursor
            cursor += 1
    return index


def build_action_mask_batch(
    options_by_state: Sequence[List[Tuple[Tuple[Card, ...], Optional[int]]]],
    e_rev_by_state: Sequence[bool],
    action_index: Dict[Tuple[int, int], int],
) -> np.ndarray:
    """
    Construit le masque booléen d'actions légales pour un lot d'états simultanés.

    Paramètre `options_by_state` : liste de listes d'options `(cards, declared_power)`, une par état du lot, telles que retournées par la
    concaténation de `generate_uniform_plays` et `generate_sequence_plays`.
    Paramètre `e_rev_by_state` : liste de booléens d'état de révolution, de taille identique à `options_by_state`.
    Paramètre `action_index` : index fixe de l'espace d'action, type retourné par `build_action_space_index`.
    Retourne un tableau `numpy.ndarray` de type `bool` et de forme `(len(options_by_state), len(action_index))`, dont l'élément `[i, j]` est
    vrai si l'action d'index `j` correspond à au moins une option légale de l'état `i`. Aucun effet de bord.
    """
    from core.math_utils import f_power

    mask = np.zeros((len(options_by_state), len(action_index)), dtype=bool)
    for state_idx, options in enumerate(options_by_state):
        e_rev = e_rev_by_state[state_idx]
        for cards, declared in options:
            non_jokers = [c for c in cards if not c.is_joker()]
            power = f_power(non_jokers[0], e_rev) if non_jokers else (declared or 0)
            key = (power, len(cards))
            if key in action_index:
                mask[state_idx, action_index[key]] = True
    return mask


def legal_option_for_action(
    options: List[Tuple[Tuple[Card, ...], Optional[int]]],
    action_idx: int,
    e_rev: bool,
    action_index: Dict[Tuple[int, int], int],
) -> Optional[Tuple[Tuple[Card, ...], Optional[int]]]:
    """
    Retrouve l'option de jeu concrète correspondant à un index d'action sélectionné par le réseau.

    Paramètre `options` : liste d'options légales de l'état considéré.
    Paramètre `action_idx` : index d'action sélectionné, entier, domaine $[0, len(action\\_index))$.
    Paramètre `e_rev` : état de révolution courant.
    Paramètre `action_index` : index fixe de l'espace d'action.
    Retourne la première option de `options` dont la projection `(power, size)` correspond à `action_idx`, ou `None` si aucune ne
    correspond. Aucun effet de bord.
    """
    from core.math_utils import f_power

    reverse = {v: k for k, v in action_index.items()}
    target = reverse.get(action_idx)
    if target is None:
        return None
    for cards, declared in options:
        non_jokers = [c for c in cards if not c.is_joker()]
        power = f_power(non_jokers[0], e_rev) if non_jokers else (declared or 0)
        if (power, len(cards)) == target:
            return cards, declared
    return None
