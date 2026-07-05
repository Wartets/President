"""
Module du dispatcher d'événements.

Le module définit `EventBus`, le composant central de diffusion des événements émis par la partie vers l'ensemble des abonnés (traceurs
d'analyse, journaliseurs, observateurs de test). Le bus ne transforme jamais les événements qu'il transmet ; il se contente de les propager dans l'ordre
d'émission à chaque abonné enregistré.

Le module dépend de `events.base` pour le type `Event`. Effet de bord global : aucun, l'état de diffusion est entièrement encapsulé dans
l'instance.
"""

from __future__ import annotations

from typing import Callable, List

from events.base import Event

Subscriber = Callable[[Event], None]


class EventBus:
    """
    Dispatcher publish/subscribe d'événements.

    Le bus maintient une liste ordonnée d'abonnés et leur transmet chaque événement publié dans l'ordre d'enregistrement des abonnements.

    Champ `_subscribers` : liste des fonctions d'écoute enregistrées.
    """

    def __init__(self) -> None:
        self._subscribers: List[Subscriber] = []

    def subscribe(self, subscriber: Subscriber) -> None:
        """
        Enregistre un nouvel abonné.

        Paramètre `subscriber` : fonction acceptant un unique argument de type `Event`, appelée pour chaque événement publié après son
        enregistrement.
        Retourne `None`. Effet de bord : ajoute `subscriber` à la liste interne des abonnés.
        """
        self._subscribers.append(subscriber)

    def publish(self, event: Event) -> None:
        """
        Diffuse un événement à tous les abonnés enregistrés.

        Paramètre `event` : événement à diffuser, type `Event`.
        Retourne `None`. Effet de bord : invoque chaque abonné enregistré avec `event` en argument, dans l'ordre d'enregistrement.
        """
        for subscriber in self._subscribers:
            subscriber(event)
