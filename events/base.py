"""
Module de la classe abstraite d'événement.

Le module définit `Event`, la classe de base immuable dont héritent tous les événements structurels et transactionnels du système. Chaque événement
transporte un horodatage, un identifiant de partie, un identifiant de manche et une empreinte de l'état courant permettant de reconstituer le contexte
exact au moment de son émission.

Aucune dépendance interne n'est requise. Le module ne provoque aucun effet de bord global.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional


def compute_state_hash(state_snapshot: Any) -> str:
    """
    Calcule l'empreinte déterministe d'un instantané d'état.

    Paramètre `state_snapshot` : représentation sérialisable de l'état courant, typiquement une chaîne ou un tuple de valeurs primitives.
    Retourne une chaîne hexadécimale de longueur fixe, résultat d'un condensat SHA-256 de la représentation textuelle de `state_snapshot`.
    Aucun effet de bord.
    """
    payload = repr(state_snapshot).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class Event:
    """
    Classe abstraite de base pour tout événement émis par la partie.

    Un événement est immuable et ne porte que des informations descriptives de ce qui s'est produit ; il ne contient aucune logique de mutation. Les
    sous-classes ajoutent les champs propres à leur nature.

    Champ `timestamp` : horodatage logique d'émission, entier croissant au fil de la partie, indépendant de l'horloge système.
    Champ `game_id` : identifiant de la partie, chaîne.
    Champ `round_id` : identifiant de la manche courante, entier, domaine positif ou nul.
    Champ `state_hash` : empreinte de l'état matérialisé au moment de l'émission, chaîne hexadécimale.
    """

    timestamp: int
    game_id: str
    round_id: int
    state_hash: str = field(default="")
