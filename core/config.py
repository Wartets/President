"""
Module de configuration de la partie.

Le module définit l'objet de paramètres immuable transmis à l'initialisation d'une partie. Il regroupe l'ensemble des booléens et valeurs numériques
contrôlant l'activation des règles avancées, la topologie du paquet et les modes de calcul des points de victoire. Le module expose un unique type 
principal, `GameConfig`, ainsi que les constantes de chaînes utilisées comme valeurs énumérées pour les champs textuels de configuration.

Aucune dépendance interne n'est requise. Le module ne provoque aucun effet de bord global.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Valeurs autorisées pour GameConfig.pass_type.
PASS_TYPE_HARD_ONLY = "HARD_ONLY"
PASS_TYPE_ALLOW_SOFT = "ALLOW_SOFT"

# Valeurs autorisées pour GameConfig.vp_distribution_type.
VP_DISTRIBUTION_LEGACY_STEPPED = "LEGACY_STEPPED"
VP_DISTRIBUTION_LINEAR = "LINEAR"
VP_DISTRIBUTION_SYMMETRICAL = "SYMMETRICAL"

# Valeurs autorisées pour GameConfig.finish_penalty_type.
PENALTY_INSTANT_SCUM = "PENALTY_INSTANT_SCUM"
PENALTY_DRAW_CARDS = "PENALTY_DRAW_CARDS"

# Rôles attribués en fin de manche selon l'ordre de sortie.
ROLE_PRESIDENT = "ROLE_PRESIDENT"
ROLE_VICE_PRESIDENT = "ROLE_VICE_PRESIDENT"
ROLE_NEUTRAL = "ROLE_NEUTRAL"
ROLE_VICE_SCUM = "ROLE_VICE_SCUM"
ROLE_SCUM = "ROLE_SCUM"


@dataclass(frozen=True)
class GameConfig:
    """
    Paramètres complets et immuables d'une partie.

    La classe regroupe la totalité des paramètres nécessaires à l'instanciation d'une `Game`. Chaque champ correspond à un paramètre
    documenté dans la spécification des règles. L'immutabilité de la classe garantit qu'une configuration ne peut pas être modifiée après le
    démarrage d'une partie, ce qui est une précondition de la reproductibilité exigée par `random_seed`. Aucun champ ne possède de valeur dépendante
    d'un état de jeu ; tous les champs sont indépendants les uns des autres à l'exception des couples explicitement documentés (`magic_two` et
    `magic_card_enabled`, `double_revolution_enabled` et `revolution_enabled`, `interception_enabled` et la contrainte $N_D \\ge 2$).

    Paramètre `random_seed` : graine de génération aléatoire, entier, domaine non borné, utilisée pour tout tirage aléatoire de la partie.
    Paramètre `player_count` : nombre de joueurs $N$, entier, domaine $N \\ge 3$.
    Paramètre `first_trick_opener_id` : identifiant du joueur ouvrant le premier pli de la première manche, entier, domaine $[0, N-1]$.
    Paramètre `deck_scaling_auto` : active le calcul dynamique du nombre de paquets en fonction de `player_count`.
    Paramètre `pass_type` : détermine la sémantique de passe légale pour toute la partie, valeurs `HARD_ONLY` ou `ALLOW_SOFT`.
    Paramètre `vp_distribution_type` : mode de calcul des points de victoire, valeurs `LEGACY_STEPPED`, `LINEAR` ou `SYMMETRICAL`.
    Paramètre `use_jokers` : active l'inclusion de Jokers dans le paquet.
    Paramètre `magic_two` : active la clôture magique historique sur le rang 2.
    Paramètre `magic_two_single_clears_all` : autorise une combinaison de taille 1 contenant un 2 à clôturer un pli de taille quelconque.
    Paramètre `magic_card_enabled` : généralisation de `magic_two` à un rang paramétrable.
    Paramètre `magic_card_rank` : rang défini comme magique.
    Paramètre `magic_single_clears_all` : généralisation de `magic_two_single_clears_all` au rang magique paramétrable.
    Paramètre `skip_on_equal` : active l'obligation de répondre par une
    puissance strictement égale après une égalité déclarée.
    Paramètre `revolution_enabled` : active l'inversion de la hiérarchie des puissances par combinaison de taille $\\ge 4$.
    Paramètre `double_revolution_enabled` : active le verrouillage de l'état de révolution par combinaison de taille $\\ge 8$, nécessite $N_D \\ge 2$.
    Paramètre `straights_enabled` : active les combinaisons de type suite.
    Paramètre `skip_turn_enabled` : active le saut de tour déclenché par un rang paramétrable.
    Paramètre `skip_turn_rank` : rang déclenchant le saut de tour.
    Paramètre `interception_enabled` : active l'interception hors-tour, nécessite $N_D \\ge 2$.
    Paramètre `putsch_enabled` : active le droit d'invocation du Putsch par le rôle `ROLE_SCUM`.
    Paramètre `blind_tax_enabled` : remplace la sélection déterministe des cartes transférées par le rôle `ROLE_SCUM` par une sélection aléatoireuniforme.
    Paramètre `strict_remainder_allocation` : attribue le reste de la distribution modulaire à un rôle ciblé plutôt que de le répartir modulo $N$.
    Paramètre `strict_remainder_role` : rôle ciblé par `strict_remainder_allocation`.
    Paramètre `finish_penalty_enabled` : active la pénalité de sortie.
    Paramètre `finish_penalty_type` : nature de la pénalité de sortie, valeurs `PENALTY_INSTANT_SCUM` ou `PENALTY_DRAW_CARDS`.
    Paramètre `finish_penalty_draw_count` : nombre de cartes piochées si `finish_penalty_type` vaut `PENALTY_DRAW_CARDS`, domaine entier positif.
    Paramètre `finish_penalty_extended` : active les sous-conditions étendues de pénalité de sortie.
    Paramètre `no_finish_on_joker` : sous-condition de `finish_penalty_extended` pénalisant une sortie sur Joker.
    Paramètre `no_finish_on_revolution` : sous-condition de `finish_penalty_extended` pénalisant une sortie déclenchant une révolution.
    """

    random_seed: int = 0
    player_count: int = 4
    first_trick_opener_id: int = 0

    deck_scaling_auto: bool = True
    forced_deck_count: Optional[int] = None

    pass_type: str = PASS_TYPE_HARD_ONLY
    vp_distribution_type: str = VP_DISTRIBUTION_SYMMETRICAL

    use_jokers: bool = True
    magic_two: bool = True
    magic_two_single_clears_all: bool = True
    magic_card_enabled: bool = False
    magic_card_rank: str = "2"
    magic_single_clears_all: bool = True

    skip_on_equal: bool = False

    revolution_enabled: bool = True
    double_revolution_enabled: bool = False

    straights_enabled: bool = False

    skip_turn_enabled: bool = False
    skip_turn_rank: str = "8"

    interception_enabled: bool = False

    putsch_enabled: bool = False
    blind_tax_enabled: bool = False

    strict_remainder_allocation: bool = False
    strict_remainder_role: str = ROLE_SCUM

    finish_penalty_enabled: bool = False
    finish_penalty_type: str = PENALTY_INSTANT_SCUM
    finish_penalty_draw_count: int = 1
    finish_penalty_extended: bool = False
    no_finish_on_joker: bool = False
    no_finish_on_revolution: bool = False

    def __post_init__(self) -> None:
        """
        Valide les contraintes structurelles inter-champs de la configuration.

        Retourne `None`. Lève `ValueError` si l'une des contraintes suivantes n'est pas respectée : `player_count` supérieur ou égal à
        trois ; `double_revolution_enabled` impliquant `revolution_enabled` ; `double_revolution_enabled` ou `interception_enabled`
        impliquant un nombre de paquets effectif supérieur ou égal à deux ; `pass_type`, `vp_distribution_type`, `finish_penalty_type` et
        `strict_remainder_role` appartenant à leur ensemble de valeurs autorisées respectif ; `magic_card_rank` et `skip_turn_rank`
        appartenant à l'ensemble des rangs faciaux non-Joker. Aucun effet de bord hors la levée d'exception.
        """
        valid_ranks = ("3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2")

        if self.player_count < 3:
            raise ValueError("player_count doit être supérieur ou égal à 3.")

        if self.double_revolution_enabled and not self.revolution_enabled:
            raise ValueError("double_revolution_enabled requiert revolution_enabled.")

        if not self.deck_scaling_auto and self.forced_deck_count is not None:
            effective_decks = self.forced_deck_count
        else:
            effective_decks = max(1, (self.player_count - 1) // 4 + 1)
        if (self.double_revolution_enabled or self.interception_enabled) and effective_decks < 2:
            raise ValueError(
                "double_revolution_enabled et interception_enabled requièrent un nombre de paquets effectif supérieur ou égal à 2."
            )

        if self.pass_type not in (PASS_TYPE_HARD_ONLY, PASS_TYPE_ALLOW_SOFT):
            raise ValueError("pass_type doit être HARD_ONLY ou ALLOW_SOFT.")

        if self.vp_distribution_type not in (
            VP_DISTRIBUTION_LEGACY_STEPPED, VP_DISTRIBUTION_LINEAR, VP_DISTRIBUTION_SYMMETRICAL,
        ):
            raise ValueError("vp_distribution_type doit être LEGACY_STEPPED, LINEAR ou SYMMETRICAL.")

        if self.finish_penalty_type not in (PENALTY_INSTANT_SCUM, PENALTY_DRAW_CARDS):
            raise ValueError("finish_penalty_type doit être PENALTY_INSTANT_SCUM ou PENALTY_DRAW_CARDS.")

        if self.strict_remainder_role not in (
            ROLE_PRESIDENT, ROLE_VICE_PRESIDENT, ROLE_NEUTRAL, ROLE_VICE_SCUM, ROLE_SCUM,
        ):
            raise ValueError("strict_remainder_role doit être un rôle valide.")

        if self.magic_card_rank not in valid_ranks:
            raise ValueError("magic_card_rank doit être un rang facial non-Joker.")

        if self.skip_turn_rank not in valid_ranks:
            raise ValueError("skip_turn_rank doit être un rang facial non-Joker.")

    def effective_magic_card_enabled(self) -> bool:
        """
        Indique si une règle de clôture magique quelconque est active.

        Retourne un booléen. La valeur est vraie si `magic_two` ou
        `magic_card_enabled` est vrai. Aucun effet de bord.
        """
        return self.magic_two or self.magic_card_enabled

    def effective_magic_card_rank(self) -> str:
        """
        Retourne le rang effectif considéré comme magique.

        Retourne une chaîne parmi les rangs faciaux. Si `magic_card_enabled` est vrai, la valeur retournée est `magic_card_rank`. Sinon, la valeur
        retournée est `"2"`, correspondant au comportement historique de `magic_two`. Aucun effet de bord.
        """
        if self.magic_card_enabled:
            return self.magic_card_rank
        return "2"

    def effective_magic_single_clears_all(self) -> bool:
        """
        Indique si une combinaison de taille un peut clôturer un pli plus grand.

        Retourne un booléen combinant `magic_two_single_clears_all` et `magic_single_clears_all` selon la règle magique active. Aucun effet
        de bord.
        """
        if self.magic_card_enabled:
            return self.magic_single_clears_all
        return self.magic_two_single_clears_all
