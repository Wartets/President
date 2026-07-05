"""
Module des modèles de données fondamentaux.

Le module définit les objets immuables manipulés par le moteur de règles et par la couche d'événements. Il expose les types `Suit`, `Rank`, `Card`,
`Hand` et `Action`, ainsi que l'énumération `ActionType`. Une carte est définie par un rang facial, une couleur optionnelle et transporte ses
valeurs de puissance et de points par l'intermédiaire du module `math_utils`. Une main est une collection ordonnée et immuable de cartes. Une
action est le vecteur de décision retourné par un agent, comprenant éventuellement une combinaison de cartes et une déclaration de puissance
pour la résolution des Jokers.

Le module dépend de `core.math_utils` pour le calcul des valeurs dynamiques de puissance et de points. Aucun effet de bord global n'est provoqué par
l'import du module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class Suit(Enum):
    """
    Couleur d'une carte.

    Chaque valeur représente une des quatre couleurs standard d'un paquet de cartes à jouer. La valeur `NONE` est réservée aux Jokers, qui ne
    possèdent pas de couleur. L'ordre de déclaration des membres définit l'ordre arbitraire de départage utilisé par le moteur de règles pour
    garantir le déterminisme des résolutions d'égalité de puissance.
    """

    SPADES = "SPADES"
    HEARTS = "HEARTS"
    DIAMONDS = "DIAMONDS"
    CLUBS = "CLUBS"
    NONE = "NONE"


# Ordre total des rangs faciaux, du plus faible au plus fort en puissance
# standard. La position dans la liste sert de base au calcul de f_std.
RANK_ORDER: Tuple[str, ...] = (
    "3", "4", "5", "6", "7", "8", "9", "10",
    "J", "Q", "K", "A", "2", "JOKER",
)

RANK_INDEX = {rank: index for index, rank in enumerate(RANK_ORDER)}


class Rank(Enum):
    """
    Rang facial d'une carte.

    L'énumération couvre l'ensemble ordonné des rangs faciaux utilisés par le jeu, du rang le plus bas (3) au Joker. Chaque valeur correspond à une
    chaîne identique à celle utilisée dans `RANK_ORDER` et dans les champs de configuration référençant un rang (`magic_card_rank`, `skip_turn_rank`).
    """

    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"
    TWO = "2"
    JOKER = "JOKER"


@dataclass(frozen=True, order=False)
class Card:
    """
    Carte à jouer immuable.

    Une carte est définie par son rang facial et sa couleur. Les cartes de rang `JOKER` possèdent la couleur `Suit.NONE` et un identifiant
    d'exemplaire permettant de distinguer plusieurs Jokers issus de paquets différents. Les valeurs de puissance et de points ne sont pas stockées
    sur l'instance ; elles sont calculées par les fonctions du module `math_utils` à partir du rang et de l'état de révolution courant. Deux
    cartes de même rang et de même couleur sont considérées égales.

    Champ `rank` : rang facial de la carte, type `Rank`.
    Champ `suit` : couleur de la carte, type `Suit`, valeur `Suit.NONE` obligatoire pour les cartes de rang `JOKER`.
    Champ `instance_id` : identifiant d'exemplaire, entier, utilisé pour distinguer les copies d'une même carte faciale issues de paquets
    multiples ; ne participe pas à la comparaison d'égalité.
    """

    rank: Rank
    suit: Suit
    instance_id: int = field(default=0, compare=False)

    def __post_init__(self) -> None:
        if self.rank is Rank.JOKER and self.suit is not Suit.NONE:
            object.__setattr__(self, "suit", Suit.NONE)

    def is_joker(self) -> bool:
        """
        Indique si la carte est un Joker.

        Retourne un booléen. Aucun effet de bord.
        """
        return self.rank is Rank.JOKER

    def __repr__(self) -> str:
        if self.is_joker():
            return f"Joker#{self.instance_id}"
        return f"{self.rank.value}{self.suit.value[0]}"


@dataclass(frozen=True)
class Hand:
    """
    Main immuable d'un joueur.

    La main est une collection ordonnée de cartes. Toute opération de retrait ou d'ajout produit une nouvelle instance ; aucune mutation en place n'est
    effectuée, conformément au principe d'event sourcing de l'architecture.

    Champ `cards` : tuple de `Card` composant la main.
    """

    cards: Tuple[Card, ...] = field(default_factory=tuple)

    def size(self) -> int:
        """
        Retourne le nombre de cartes de la main.

        Retourne un entier positif ou nul. Aucun effet de bord.
        """
        return len(self.cards)

    def is_empty(self) -> bool:
        """
        Indique si la main est vide.

        Retourne un booléen équivalent à `size() == 0`. Aucun effet de bord.
        """
        return len(self.cards) == 0

    def without(self, cards_to_remove: Tuple[Card, ...]) -> "Hand":
        """
        Retourne une nouvelle main privée des cartes indiquées.

        Paramètre `cards_to_remove` : tuple de `Card` à retirer, chaque carte devant être un élément présent de la main, contrainte non
        vérifiée par la fonction elle-même.
        Retourne une nouvelle instance de `Hand`, type `Hand`. Aucun effet de bord, aucune mutation de l'instance courante.
        """
        remaining = list(self.cards)
        for card in cards_to_remove:
            remaining.remove(card)
        return Hand(cards=tuple(remaining))

    def with_added(self, cards_to_add: Tuple[Card, ...]) -> "Hand":
        """
        Retourne une nouvelle main augmentée des cartes indiquées.

        Paramètre `cards_to_add` : tuple de `Card` à ajouter.
        Retourne une nouvelle instance de `Hand`, type `Hand`. Aucun effet de bord, aucune mutation de l'instance courante.
        """
        return Hand(cards=self.cards + tuple(cards_to_add))


class ActionType(Enum):
    """
    Nature d'une action de tour.

    La valeur `ACTION_PLAY` désigne la pose d'une combinaison. La valeur `ACTION_SOFT_PASS` désigne un passe conservant l'éligibilité au pli
    courant, légale uniquement sous `pass_type == 'ALLOW_SOFT'`. La valeur `ACTION_HARD_PASS` désigne un passe excluant définitivement le joueur du
    pli courant.
    """

    ACTION_PLAY = "ACTION_PLAY"
    ACTION_SOFT_PASS = "ACTION_SOFT_PASS"
    ACTION_HARD_PASS = "ACTION_HARD_PASS"


@dataclass(frozen=True)
class Action:
    """
    Vecteur de décision émis par un agent.

    Une action porte sa nature, la combinaison de cartes concernée le cas échéant, et une déclaration de puissance obligatoire dès qu'un Joker est
    présent dans la combinaison. Le champ `declared_power` permet également de trancher si un Joker déclaré à la valeur du rang magique doit
    déclencher ou non une clôture immédiate.

    Champ `action_type` : nature de l'action, type `ActionType`.
    Champ `cards` : tuple de `Card` posées, vide pour un passe.
    Champ `declared_power` : entier optionnel, obligatoire et strictement positif dès que `cards` contient un Joker, représente la valeur de
    puissance assignée au Joker pour la résolution de la combinaison.
    """

    action_type: ActionType
    cards: Tuple[Card, ...] = field(default_factory=tuple)
    declared_power: Optional[int] = None

    def contains_joker(self) -> bool:
        """
        Indique si la combinaison posée contient au moins un Joker.

        Retourne un booléen. Aucun effet de bord.
        """
        return any(card.is_joker() for card in self.cards)

    def size(self) -> int:
        """
        Retourne la taille de la combinaison posée.

        Retourne un entier positif ou nul, égal à zéro pour un passe. Aucun effet de bord.
        """
        return len(self.cards)
