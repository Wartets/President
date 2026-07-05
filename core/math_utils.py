"""
Module des fonctions de valeur mathématique.

Le module fournit les fonctions pures de calcul de la valeur de points de fin de partie $f_{points}$, de la valeur de puissance dynamique $f_{power}$, et
des trois modes de distribution des points de victoire ($VP$). Il fournit également la fonction d'attribution de rôle à partir d'un index de sortie et
la fonction de résolution algébrique de la puissance en révolution, qui exploite la propriété selon laquelle la somme de la puissance standard et de
la puissance inversée d'une carte vaut toujours dix-huit.

Le module dépend de `core.models` pour les types `Rank` et `RANK_INDEX`.
Aucun effet de bord global n'est provoqué par l'import du module.
"""

from __future__ import annotations

from typing import Dict

from core.models import Card, Rank, RANK_INDEX

# Valeur faciale de points, constante quelle que soit la révolution.
_POINTS_TABLE: Dict[str, int] = {
    "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
    "J": 11, "Q": 12, "K": 13, "A": 14, "2": 15, "JOKER": 16,
}

# Médiane de la hiérarchie standard, utilisée pour le miroir de révolution.
_REVOLUTION_SUM = 18


def f_points(card: Card) -> int:
    """
    Calcule la valeur de points de fin de partie d'une carte.

    Paramètre `card` : carte considérée, type `Card`.
    Retourne un entier, domaine $[3, 16]$, constant quel que soit l'état de
    révolution. Aucun effet de bord.
    """
    return _POINTS_TABLE[card.rank.value]


def f_std(card: Card) -> int:
    """
    Calcule la puissance standard d'une carte, hors révolution.

    Paramètre `card` : carte considérée, type `Card`.
    Retourne un entier, domaine $[3, 16]$, identique à `f_points(card)`.
    Aucun effet de bord.
    """
    return _POINTS_TABLE[card.rank.value]


def f_power(card: Card, e_rev: bool) -> int:
    """
    Calcule la puissance dynamique d'une carte selon l'état de révolution.

    Paramètre `card` : carte considérée, type `Card`.
    Paramètre `e_rev` : état booléen de la révolution.
    Retourne un entier, domaine $[3, 16]$. La valeur retournée pour un Joker
    est constante et vaut seize, indépendamment de `e_rev`. Pour toute autre
    carte, la valeur retournée est la puissance standard si `e_rev` est
    faux, ou son symétrique par rapport à la constante dix-huit si `e_rev`
    est vrai. Aucun effet de bord.
    """
    if card.is_joker():
        return 16
    std = f_std(card)
    if not e_rev:
        return std
    return _REVOLUTION_SUM - std


def f_power_rev(card_std_power: int) -> int:
    """
    Calcule la puissance inversée à partir de la puissance standard.

    Paramètre `card_std_power` : puissance standard de la carte, entier,
    domaine $[3, 15]$, un Joker ne devant jamais être transmis à cette
    fonction.
    Retourne un entier, domaine $[3, 15]$, égal au symétrique de
    `card_std_power` par rapport à la constante dix-huit. Aucun effet de
    bord.
    """
    return _REVOLUTION_SUM - card_std_power


def rank_facial_index(rank_value: str) -> int:
    """
    Retourne l'index ordinal d'un rang dans la hiérarchie standard.

    Paramètre `rank_value` : rang facial, chaîne parmi les valeurs de
    `RANK_ORDER`.
    Retourne un entier, domaine $[0, 13]$, croissant avec la puissance
    standard. Aucun effet de bord.
    """
    return RANK_INDEX[rank_value]


def vp_legacy_stepped(k: int, n: int) -> float:
    """
    Calcule le point de victoire du barème historique par rupture.

    Paramètre `k` : index de sortie, entier, domaine $[0, n-1]$.
    Paramètre `n` : nombre de joueurs, entier, domaine $n \\ge 3$.
    Retourne un nombre. La valeur retournée vaut $n$ pour $k=0$, $n-1$ pour
    $k=1$, $n-k$ pour $2 \\le k \\le n-3$, zéro pour $k=n-2$, et moins un
    pour $k=n-1$. La courbe présente une rupture de pente entre le rôle
    neutre et le rôle vice-scum. Aucun effet de bord.
    """
    if k == 0:
        return float(n)
    if k == 1:
        return float(n - 1)
    if k == n - 1:
        return -1.0
    if k == n - 2:
        return 0.0
    return float(n - k)


def vp_linear(k: int, n: int) -> float:
    """
    Calcule le point de victoire du barème linéaire.

    Paramètre `k` : index de sortie, entier, domaine $[0, n-1]$.
    Paramètre `n` : nombre de joueurs, entier, domaine $n \\ge 3$.
    Retourne un nombre, décroissant strictement et sans rupture de pente
    entre les index de sortie consécutifs, égal à $n - 1 - k$. Aucun effet
    de bord.
    """
    return float(n - 1 - k)


def vp_symmetrical(k: int, n: int) -> float:
    """
    Calcule le point de victoire du barème symétrique centré sur zéro.

    Paramètre `k` : index de sortie, entier, domaine $[0, n-1]$.
    Paramètre `n` : nombre de joueurs, entier, domaine $n \\ge 3$.
    Retourne un nombre, égal à $(n-1)/2 - k$, symétrique par rapport à zéro
    pour un ordre de sortie complet. Aucun effet de bord.
    """
    return (n - 1) / 2 - k


_VP_FUNCTIONS = {
    "LEGACY_STEPPED": vp_legacy_stepped,
    "LINEAR": vp_linear,
    "SYMMETRICAL": vp_symmetrical,
}


def compute_vp(k: int, n: int, distribution_type: str) -> float:
    """
    Calcule le point de victoire selon le mode de distribution sélectionné.

    Paramètre `k` : index de sortie, entier, domaine $[0, n-1]$.
    Paramètre `n` : nombre de joueurs, entier, domaine $n \\ge 3$.
    Paramètre `distribution_type` : mode de calcul, chaîne parmi
    `LEGACY_STEPPED`, `LINEAR`, `SYMMETRICAL`.
    Retourne un nombre, résultat de la fonction correspondant au mode
    sélectionné. Lève `KeyError` si `distribution_type` ne correspond à
    aucun mode connu. Aucun effet de bord.
    """
    return _VP_FUNCTIONS[distribution_type](k, n)


def role_for_rank(k: int, n: int) -> str:
    """
    Détermine le rôle attribué à un index de sortie donné.

    Paramètre `k` : index de sortie, entier, domaine $[0, n-1]$.
    Paramètre `n` : nombre de joueurs, entier, domaine $n \\ge 3$.
    Retourne une chaîne parmi `ROLE_PRESIDENT`, `ROLE_VICE_PRESIDENT`,
    `ROLE_NEUTRAL`, `ROLE_VICE_SCUM`, `ROLE_SCUM`. Pour $n=3$, le rôle
    `ROLE_VICE_PRESIDENT` et le rôle `ROLE_VICE_SCUM` sont absents, l'unique
    index intermédiaire recevant `ROLE_NEUTRAL`. Pour $n=4$, le rôle
    `ROLE_NEUTRAL` est absent. Aucun effet de bord.
    """
    from core.config import (
        ROLE_PRESIDENT, ROLE_VICE_PRESIDENT, ROLE_NEUTRAL,
        ROLE_VICE_SCUM, ROLE_SCUM,
    )

    if k == 0:
        return ROLE_PRESIDENT
    if k == n - 1:
        return ROLE_SCUM
    if n == 3:
        return ROLE_NEUTRAL
    if k == 1:
        return ROLE_VICE_PRESIDENT
    if k == n - 2:
        return ROLE_VICE_SCUM
    return ROLE_NEUTRAL
