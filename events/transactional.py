"""
Module des événements transactionnels.

Le module définit les événements décrivant les décisions individuelles des agents et les déclenchements de règles avancées : échange, sollicitation
d'agent, action jouée, diffusion d'interception et déclenchement de règle. Chaque type hérite de `Event` et ajoute les champs propres à sa nature.

Le module dépend de `events.base` pour la classe `Event` et de `core.models` pour les types `Card` et `ActionType`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from core.models import ActionType, Card
from events.base import Event


@dataclass(frozen=True)
class EventAskPutsch(Event):
    """
    Sollicitation du rôle `ROLE_SCUM` pour l'invocation du Putsch.

    Champ `player_id` : identifiant du joueur sollicité.
    Champ `condition_met` : indique si la condition mathématique $P_{putsch}$ était satisfaite au moment de la sollicitation.
    """

    player_id: int = -1
    condition_met: bool = False


@dataclass(frozen=True)
class EventPutschInvoked(Event):
    """
    Invocation effective du Putsch, annulant la phase d'échange de la manche.

    Champ `player_id` : identifiant du joueur ayant invoqué le Putsch.
    """

    player_id: int = -1


@dataclass(frozen=True)
class EventExchangeIntent(Event):
    """
    Choix libre de cartes exprimé par un agent lors d'un échange.

    Champ `from_player` : identifiant du joueur donnant les cartes.
    Champ `to_player` : identifiant du joueur recevant les cartes.
    Champ `offered_cards` : tuple des cartes choisies par l'agent.
    """

    from_player: int = -1
    to_player: int = -1
    offered_cards: Tuple[Card, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EventExchange(Event):
    """
    Transfert effectif de cartes lors de la phase d'échange.

    Champ `from_player` : identifiant du joueur donnant les cartes.
    Champ `to_player` : identifiant du joueur recevant les cartes.
    Champ `cards` : tuple des cartes transférées.
    Champ `was_blind_tax` : indique si la sélection résulte d'un tirage aléatoire uniforme sous `blind_tax_enabled` plutôt que d'une sélection
    déterministe par puissance maximale.
    """

    from_player: int = -1
    to_player: int = -1
    cards: Tuple[Card, ...] = field(default_factory=tuple)
    was_blind_tax: bool = False


@dataclass(frozen=True)
class EventActionRequest(Event):
    """
    Sollicitation d'une décision auprès d'un agent.

    Champ `player_id` : identifiant du joueur sollicité.
    Champ `trick_index` : index du pli courant au sein de la manche.
    """

    player_id: int = -1
    trick_index: int = 0


@dataclass(frozen=True)
class EventActionPlayed(Event):
    """
    Action validée et appliquée à l'état courant.

    Champ `player_id` : identifiant du joueur ayant agi.
    Champ `action_type` : nature de l'action.
    Champ `cards_played` : tuple des cartes posées, vide pour un passe.
    Champ `resulting_power` : puissance résultante de la combinaison posée, non significative pour un passe.
    Champ `was_suboptimal` : indique si le joueur a passé alors qu'il disposait d'au moins une combinaison valide selon le moteur de règles.
    """

    player_id: int = -1
    action_type: ActionType = ActionType.ACTION_HARD_PASS
    cards_played: Tuple[Card, ...] = field(default_factory=tuple)
    resulting_power: Optional[int] = None
    was_suboptimal: bool = False


@dataclass(frozen=True)
class EventInterceptionBroadcast(Event):
    """
    Diffusion d'une opportunité d'interception à tous les joueurs hors-tour.

    Champ `played_card` : carte unique dernièrement posée, cible de l'interception.
    Champ `eligible_player_ids` : tuple des identifiants de joueurs sollicités pour l'opportunité.
    """

    played_card: Optional[Card] = None
    eligible_player_ids: Tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EventInterceptionResolved(Event):
    """
    Résolution d'une opportunité d'interception.

    Champ `interceptor_id` : identifiant du joueur ayant intercepté, ou `None` si aucune interception n'a eu lieu.
    Champ `intercepted_card` : carte jumelle jouée par l'intercepteur, ou `None`.
    """

    interceptor_id: Optional[int] = None
    intercepted_card: Optional[Card] = None


@dataclass(frozen=True)
class EventRuleTriggered(Event):
    """
    Déclenchement d'une règle avancée ayant muté l'état global.

    Champ `rule_name` : nom de la règle déclenchée, parmi `REVOLUTION`, `DOUBLE_REVOLUTION`, `MAGIC_CLOSURE`, `SKIP_TURN`, `INTERCEPTION`,
    `EQUAL_FORCED`, `FINISH_PENALTY`.
    Champ `triggering_player_id` : identifiant du joueur ayant déclenché la règle.
    """

    rule_name: str = ""
    triggering_player_id: int = -1
