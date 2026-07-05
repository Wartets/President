"""
Module des événements structurels.

Le module définit les événements décrivant le déroulement macroscopique de la partie : configuration, démarrage, distribution, ouverture et clôture de
pli, sortie de joueur et fin de manche. Chaque type hérite de `Event` et ajoute les champs propres à sa nature.

Le module dépend de `events.base` pour la classe `Event`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

from core.config import GameConfig
from core.models import Card
from events.base import Event


@dataclass(frozen=True)
class EventGameConfig(Event):
    """
    Paramètres complets de la partie.

    Champ `config` : configuration immuable de la partie, type `GameConfig`.
    """

    config: GameConfig = None  # type: ignore[assignment]


@dataclass(frozen=True)
class EventGameStart(Event):
    """
    Démarrage de la partie.

    Champ `config` : configuration immuable de la partie.
    Champ `player_ids` : tuple des identifiants de joueurs participants.
    """

    config: GameConfig = None  # type: ignore[assignment]
    player_ids: Tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EventRoundStart(Event):
    """
    Démarrage d'une manche, incluant la matrice complète des mains initiales.

    Champ `initial_hands` : association entre identifiant de joueur et tuple de `Card` composant sa main de départ, information cachée aux agents
    mais visible pour l'analyse.
    """

    initial_hands: Dict[int, Tuple[Card, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class EventTrickStart(Event):
    """
    Ouverture d'un nouveau pli.

    Champ `opener_id` : identifiant du joueur ouvrant le pli.
    Champ `trick_index` : index du pli au sein de la manche courante.
    """

    opener_id: int = -1
    trick_index: int = 0


@dataclass(frozen=True)
class EventTrickClosed(Event):
    """
    Clôture d'un pli.

    Champ `winner_id` : identifiant du joueur remportant le pli.
    Champ `trick_size` : taille $X$ des combinaisons attendues durant le pli.
    """

    winner_id: int = -1
    trick_size: int = 0


@dataclass(frozen=True)
class EventHandEmpty(Event):
    """
    Vidage complet de la main d'un joueur.

    Champ `player_id` : identifiant du joueur dont la main est désormais vide.
    """

    player_id: int = -1


@dataclass(frozen=True)
class EventPlayerFinished(Event):
    """
    Sortie définitive d'un joueur pour la manche courante.

    Champ `player_id` : identifiant du joueur sorti.
    Champ `rank` : index de sortie $k$ au sein de la manche.
    Champ `vp_earned` : point de victoire attribué pour cette sortie.
    """

    player_id: int = -1
    rank: int = -1
    vp_earned: float = 0.0


@dataclass(frozen=True)
class EventRoundEnd(Event):
    """
    Fin de manche et distribution finale des points de victoire.

    Champ `vp_by_player` : association entre identifiant de joueur et point de victoire attribué pour la manche.
    Champ `roles_by_player` : association entre identifiant de joueur et rôle attribué pour la manche suivante.
    """

    vp_by_player: Dict[int, float] = field(default_factory=dict)
    roles_by_player: Dict[int, str] = field(default_factory=dict)
