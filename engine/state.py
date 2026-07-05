"""
Module de la vue matérialisée de l'état courant.

Le module définit `GameState`, la structure mutable représentant l'état courant de la partie à un instant donné. Conformément au principe d'event
sourcing de l'architecture, cette structure est une réduction de la séquence d'événements ; elle ne constitue pas la source de vérité mais une
vue de confort utilisée par le moteur pour évaluer la légalité des actions.

Champs principaux exposés par `GameState` : les mains courantes de chaque joueur, la combinaison et la puissance courantes du pli actif, l'état de
révolution $E_{rev}$ et son verrouillage $L_{rev}$, l'état d'égalité forcée, l'éligibilité de chaque joueur pour le pli courant, l'index du joueur
courant et la liste des joueurs déjà sortis pour la manche.

Le module dépend de `core.models` pour les types `Hand` et `Card`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.models import Card, Hand


@dataclass
class TrickState:
    """
    État courant d'un pli en cours de résolution.

    Champ `size` : taille $X$ attendue des combinaisons répondant au pli, fixée par la première combinaison posée.
    Champ `current_power` : puissance $P_{current}$ de la dernière combinaison valide posée, `None` avant toute pose.
    Champ `last_player_id` : identifiant du dernier joueur ayant validé `ACTION_PLAY`, `None` avant toute pose.
    Champ `is_sequence` : indique si la combinaison active est une suite.
    Champ `sequence_min_power` : puissance minimale de la suite active, utilisée pour la condition de surenchère des suites.
    Champ `is_closed` : indique si le pli est clos.
    Champ `trick_index` : index du pli au sein de la manche courante.
    """

    size: int = 0
    current_power: Optional[int] = None
    last_player_id: Optional[int] = None
    is_sequence: bool = False
    sequence_min_power: Optional[int] = None
    is_closed: bool = False
    trick_index: int = 0


@dataclass
class GameState:
    """
    Vue matérialisée mutable de l'état courant de la partie.

    Champ `hands` : association entre identifiant de joueur et sa main courante.
    Champ `is_finished` : association entre identifiant de joueur et booléen indiquant s'il a déjà vidé sa main pour la manche courante.
    Champ `is_eligible` : association entre identifiant de joueur et booléen d'éligibilité au pli courant.
    Champ `finish_order` : liste ordonnée des identifiants de joueurs déjà sortis pour la manche courante, l'index dans la liste étant l'index de
    sortie $k$.
    Champ `e_rev` : état booléen courant de la révolution.
    Champ `l_rev` : état booléen de verrouillage de la révolution.
    Champ `is_equal_forced` : indique si la contrainte d'égalité forcée est active.
    Champ `current_player_id` : identifiant du joueur devant agir.
    Champ `round_index` : index $m$ de la manche courante.
    Champ `trick` : état du pli en cours, type `TrickState`.
    Champ `roles` : association entre identifiant de joueur et rôle courant.
    """

    hands: Dict[int, Hand] = field(default_factory=dict)
    is_finished: Dict[int, bool] = field(default_factory=dict)
    is_eligible: Dict[int, bool] = field(default_factory=dict)
    finish_order: List[int] = field(default_factory=list)
    e_rev: bool = False
    l_rev: bool = False
    is_equal_forced: bool = False
    current_player_id: int = 0
    round_index: int = 0
    trick: TrickState = field(default_factory=TrickState)
    roles: Dict[int, str] = field(default_factory=dict)

    def __init__(
        self,
        hands: Optional[Dict[int, Hand]] = None,
        is_finished: Optional[Dict[int, bool]] = None,
        is_eligible: Optional[Dict[int, bool]] = None,
        finish_order: Optional[List[int]] = None,
        e_rev: bool = False,
        l_rev: bool = False,
        is_equal_forced: bool = False,
        current_player_id: int = 0,
        round_index: int = 0,
        trick: Optional[TrickState] = None,
        roles: Optional[Dict[int, str]] = None,
    ) -> None:
        self.hands = hands or {}
        self.is_finished = is_finished or {}
        self.is_eligible = is_eligible or {}
        self.finish_order = finish_order or []
        self.e_rev = e_rev
        self.l_rev = l_rev
        self.is_equal_forced = is_equal_forced
        self.current_player_id = current_player_id
        self.round_index = round_index
        self.trick = trick or TrickState()
        self.roles = roles or {}

    def snapshot_key(self) -> Tuple:
        """
        Construit une représentation compacte et déterministe de l'état.

        Retourne un tuple de valeurs primitives dérivé des champs de l'instance, destiné au calcul de l'empreinte `state_hash` des
        événements. Aucun effet de bord.
        """
        hands_key = tuple(
            sorted((pid, tuple(sorted(map(repr, hand.cards)))) for pid, hand in self.hands.items())
        )
        return (
            hands_key,
            tuple(sorted(self.is_finished.items())),
            tuple(sorted(self.is_eligible.items())),
            tuple(self.finish_order),
            self.e_rev,
            self.l_rev,
            self.is_equal_forced,
            self.current_player_id,
            self.round_index,
            self.trick.size,
            self.trick.current_power,
            self.trick.last_player_id,
            self.trick.is_closed,
        )

    def active_players(self, n: int) -> List[int]:
        """
        Retourne les joueurs n'ayant pas encore vidé leur main.

        Paramètre `n` : nombre total de joueurs.
        Retourne une liste d'identifiants de joueurs, dans l'ordre croissant de siège, pour lesquels `is_finished` est faux. Aucun effet de bord.
        """
        return [pid for pid in range(n) if not self.is_finished.get(pid, False)]
