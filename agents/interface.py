"""
Module de l'interface d'agent.

Le module définit `AbstractBaseAgent`, la classe abstraite dont doit hériter toute implémentation de joueur, humaine ou automatisée. L'interface expose
les quatre points de sollicitation du moteur documentés par l'architecture : le choix d'une action de tour, le choix des cartes lors d'un échange libre,
la décision d'invocation du Putsch et la réponse à une opportunité d'interception.

Le module dépend de `core.models` pour les types `Hand`, `Action` et `Card`, et de `engine.state` pour le type `GameState`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from core.models import Action, Card, Hand
from engine.state import GameState


class AbstractBaseAgent(ABC):
    """
    Classe abstraite obligatoire pour tout agent joueur.

    Toute implémentation d'agent, humaine ou automatisée, hérite de cette classe et fournit une implémentation concrète des quatre méthodes
    déclarées. Le moteur ne suppose aucun comportement interne particulier de l'agent ; il ne considère que les valeurs retournées, validées avant
    application par `rules_engine.py`.

    Champ `player_id` : identifiant du joueur associé à l'agent.
    """

    def __init__(self, player_id: int) -> None:
        self.player_id = player_id

    @abstractmethod
    def choose_action(self, game_state: GameState) -> Action:
        """
        Sollicite le choix d'une action de tour.

        Paramètre `game_state` : vue matérialisée de l'état courant, type `GameState`.
        Retourne une instance de `Action`, devant obligatoirement porter un `declared_power` si `cards` contient un Joker. Aucune contrainte de
        pureté n'est imposée à l'implémentation ; le moteur valide le résultat avant application.
        """
        raise NotImplementedError

    @abstractmethod
    def choose_exchange_cards(
        self, hand: Hand, game_state: GameState, count: int
    ) -> List[Card]:
        """
        Sollicite le choix de cartes à céder lors d'un échange libre.

        Paramètre `hand` : main courante de l'agent, type `Hand`.
        Paramètre `game_state` : vue matérialisée de l'état courant.
        Paramètre `count` : nombre de cartes à céder, entier positif.
        Retourne une liste de `Card` de taille `count`, chaque carte devant
        appartenir à `hand`.
        """
        raise NotImplementedError

    @abstractmethod
    def ask_putsch(self, hand: Hand) -> bool:
        """
        Sollicite la décision d'invocation du Putsch.

        Paramètre `hand` : main courante de l'agent, attribué au rôle `ROLE_SCUM` au moment de la sollicitation.
        Retourne un booléen, vrai si l'agent choisit d'invoquer le Putsch.
        """
        raise NotImplementedError

    @abstractmethod
    def on_interception_opportunity(
        self, game_state: GameState, played_card: Card
    ) -> Tuple[bool, Optional[Card]]:
        """
        Sollicite la réponse à une opportunité d'interception.

        Paramètre `game_state` : vue matérialisée de l'état courant.
        Paramètre `played_card` : carte unique dernièrement posée, cible de l'interception.
        Retourne un tuple composé d'un booléen indiquant si l'agent choisit d'intercepter, et de la carte jumelle utilisée si le booléen est
        vrai, ou `None` sinon.
        """
        raise NotImplementedError
