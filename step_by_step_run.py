"""
Module du mode de partie explicite et guidé.

Fournit un point d'entrée en ligne de commande interactif qui guide l'utilisateur pas à pas à travers l'ensemble des choix de configuration
(nombre de joueurs, profils de sièges, règles actives, nombre de manches) puis affiche le déroulement complet de la partie manche par manche,
avec la possibilité d'afficher à tout instant l'intégralité des mains de tous les joueurs, triées par ordre de rang.

Le module dépend de `core.config`, `agents.*`, `engine.event_bus`, `engine.game_runner` et `events.*`.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from agents.adaptive_bot import AdaptiveBot
from agents.greedy_bot import GreedyBot
from agents.human_agent import HumanAgent
from agents.interface import AbstractBaseAgent
from agents.lookahead_bot import LookaheadBot
from agents.mcts_bot import MCTSBot
from agents.random_bot import RandomBot
from agents.rule_based_bot import RuleBasedBot
from core.config import (
    PASS_TYPE_ALLOW_SOFT, PASS_TYPE_HARD_ONLY, PENALTY_DRAW_CARDS,
    PENALTY_INSTANT_SCUM, ROLE_NEUTRAL, ROLE_PRESIDENT, ROLE_SCUM,
    ROLE_VICE_PRESIDENT, ROLE_VICE_SCUM, VP_DISTRIBUTION_LEGACY_STEPPED,
    VP_DISTRIBUTION_LINEAR, VP_DISTRIBUTION_SYMMETRICAL, GameConfig,
)
from engine.event_bus import EventBus
from engine.game_runner import Game
from events.structural import (
    EventPlayerFinished, EventRoundEnd, EventRoundStart, EventTrickClosed,
    EventTrickStart,
)
from events.transactional import EventActionPlayed, EventExchange, EventRuleTriggered

_AGENT_REGISTRY: Dict[str, Callable[[int, GameConfig], AbstractBaseAgent]] = {
    "human_agent": HumanAgent,
    "random_bot": RandomBot,
    "greedy_bot": GreedyBot,
    "rule_based_bot": RuleBasedBot,
    "lookahead_bot": LookaheadBot,
    "adaptive_bot": AdaptiveBot,
    "mcts_bot": MCTSBot,
}

_VALID_ROLES = (ROLE_PRESIDENT, ROLE_VICE_PRESIDENT, ROLE_NEUTRAL, ROLE_VICE_SCUM, ROLE_SCUM)

_RANK_ORDER_FOR_DISPLAY = ("3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2", "JOKER")
_RANK_DISPLAY_INDEX = {rank: index for index, rank in enumerate(_RANK_ORDER_FOR_DISPLAY)}


def _ask_int(prompt: str, default: int, minimum: int = 1) -> int:
    """
    Sollicite un entier par saisie interactive avec valeur par défaut.

    Paramètre `prompt` : texte affiché à l'utilisateur.
    Paramètre `default` : valeur retournée si la saisie est vide ou invalide.
    Paramètre `minimum` : valeur minimale acceptée, en deçà de laquelle `default` est retourné.
    Retourne un entier. Effet de bord : lit sur l'entrée standard.
    """
    raw = input(f"{prompt} [{default}] : ").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= minimum else default


def _ask_bool(prompt: str, default: bool) -> bool:
    """
    Sollicite un booléen par saisie interactive avec valeur par défaut.

    Paramètre `prompt` : texte affiché à l'utilisateur.
    Paramètre `default` : valeur retournée si la saisie est vide.
    Retourne un booléen, vrai si la saisie commence par `'o'`/`'O'`. Effet de bord : lit sur
    l'entrée standard.
    """
    suffix = "O/n" if default else "o/N"
    raw = input(f"{prompt} ({suffix}) : ").strip().lower()
    if not raw:
        return default
    return raw.startswith("o")


def _ask_choice(prompt: str, options: Tuple[str, ...], default: str) -> str:
    """
    Sollicite un choix parmi un ensemble fixe d'options avec valeur par défaut.

    Paramètre `prompt` : texte affiché à l'utilisateur.
    Paramètre `options` : tuple des valeurs acceptées.
    Paramètre `default` : valeur retournée si la saisie est vide ou hors de `options`.
    Retourne une chaîne appartenant à `options`. Effet de bord : lit sur l'entrée standard.
    """
    raw = input(f"{prompt} {options} [{default}] : ").strip()
    if not raw:
        return default
    return raw if raw in options else default


def _ask_seat_profiles(player_count: int) -> List[str]:
    """
    Sollicite le profil de chaque siège par saisie interactive.

    Paramètre `player_count` : nombre de sièges à renseigner.
    Retourne une liste de profils, de taille `player_count`. Effet de bord : lit sur l'entrée
    standard, affiche la liste des profils disponibles.
    """
    profiles: List[str] = []
    print("\n--- Choix des profils de siège ---")
    print(f"Profils disponibles : {', '.join(_AGENT_REGISTRY)}")
    for seat in range(player_count):
        default = "human_agent" if seat == 0 else "greedy_bot"
        raw = input(f"Profil du siège {seat} [{default}] : ").strip()
        profiles.append(raw if raw in _AGENT_REGISTRY else default)
    return profiles


def _wizard_config() -> Tuple[GameConfig, List[str], int, bool]:
    """
    Conduit l'assistant de configuration interactif complet.

    Retourne un tuple `(config, seat_profiles, rounds, reveal_all_hands)`. Effet de bord : conduit
    une série de sollicitations interactives couvrant le nombre de joueurs, les profils de siège,
    l'intégralité des champs de `GameConfig`, le nombre de manches à jouer, et l'activation de
    l'affichage complet des mains.
    """
    print("=== Assistant de configuration de partie ===")
    player_count = _ask_int("Nombre de joueurs", 4, minimum=3)
    seat_profiles = _ask_seat_profiles(player_count)
    rounds = _ask_int("Nombre de manches à jouer", 5, minimum=1)
    reveal_all_hands = _ask_bool("Afficher toutes les mains de tous les joueurs à chaque pli", False)

    print("\n--- Choix des règles actives ---")
    seed = _ask_int("Graine aléatoire", 0, minimum=0)
    first_opener = _ask_int("Identifiant du joueur ouvrant la première manche", 0, minimum=0)
    pass_type = _ask_choice("Type de passe", (PASS_TYPE_HARD_ONLY, PASS_TYPE_ALLOW_SOFT), PASS_TYPE_HARD_ONLY)
    vp_distribution = _ask_choice(
        "Distribution des points de victoire",
        (VP_DISTRIBUTION_LEGACY_STEPPED, VP_DISTRIBUTION_LINEAR, VP_DISTRIBUTION_SYMMETRICAL),
        VP_DISTRIBUTION_SYMMETRICAL,
    )
    use_jokers = _ask_bool("Activer les Jokers", True)
    magic_two = _ask_bool("Activer la clôture magique sur le 2", True)
    magic_card_enabled = _ask_bool("Activer une clôture magique généralisée à un rang paramétrable", False)
    magic_card_rank = "2"
    if magic_card_enabled:
        magic_card_rank = _ask_choice(
            "Rang magique",
            ("3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"),
            "2",
        )
    skip_on_equal = _ask_bool("Activer le forçage par égalité", False)
    revolution_enabled = _ask_bool("Activer la Révolution", True)
    double_revolution_enabled = _ask_bool("Activer la Double Révolution", False)
    straights_enabled = _ask_bool("Activer les Suites", False)
    skip_turn_enabled = _ask_bool("Activer le Saut de Tour", False)
    skip_turn_rank = "8"
    if skip_turn_enabled:
        skip_turn_rank = _ask_choice(
            "Rang de Saut de Tour",
            ("3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"),
            "8",
        )
    interception_enabled = _ask_bool("Activer l'Interception", False)
    putsch_enabled = _ask_bool("Activer le Putsch", False)
    blind_tax_enabled = _ask_bool("Activer la Taxe Aveugle", False)
    strict_remainder_allocation = _ask_bool("Activer l'attribution stricte du reste de la distribution", False)
    strict_remainder_role = ROLE_SCUM
    if strict_remainder_allocation:
        strict_remainder_role = _ask_choice("Rôle ciblé par le reste strict", _VALID_ROLES, ROLE_SCUM)
    finish_penalty_enabled = _ask_bool("Activer la pénalité de sortie", False)
    finish_penalty_type = PENALTY_INSTANT_SCUM
    finish_penalty_draw_count = 1
    if finish_penalty_enabled:
        finish_penalty_type = _ask_choice(
            "Type de pénalité de sortie", (PENALTY_INSTANT_SCUM, PENALTY_DRAW_CARDS), PENALTY_INSTANT_SCUM,
        )
        if finish_penalty_type == PENALTY_DRAW_CARDS:
            finish_penalty_draw_count = _ask_int("Nombre de cartes reprises en main", 1, minimum=1)
    finish_penalty_extended = _ask_bool("Activer les sous-conditions étendues de pénalité", False)
    no_finish_on_joker = False
    no_finish_on_revolution = False
    if finish_penalty_extended:
        no_finish_on_joker = _ask_bool("Pénaliser une sortie sur Joker", False)
        no_finish_on_revolution = _ask_bool("Pénaliser une sortie déclenchant une révolution", False)

    config = GameConfig(
        random_seed=seed,
        player_count=player_count,
        first_trick_opener_id=first_opener,
        pass_type=pass_type,
        vp_distribution_type=vp_distribution,
        use_jokers=use_jokers,
        magic_two=magic_two,
        magic_card_enabled=magic_card_enabled,
        magic_card_rank=magic_card_rank,
        skip_on_equal=skip_on_equal,
        revolution_enabled=revolution_enabled,
        double_revolution_enabled=double_revolution_enabled,
        straights_enabled=straights_enabled,
        skip_turn_enabled=skip_turn_enabled,
        skip_turn_rank=skip_turn_rank,
        interception_enabled=interception_enabled,
        putsch_enabled=putsch_enabled,
        blind_tax_enabled=blind_tax_enabled,
        strict_remainder_allocation=strict_remainder_allocation,
        strict_remainder_role=strict_remainder_role,
        finish_penalty_enabled=finish_penalty_enabled,
        finish_penalty_type=finish_penalty_type,
        finish_penalty_draw_count=finish_penalty_draw_count,
        finish_penalty_extended=finish_penalty_extended,
        no_finish_on_joker=no_finish_on_joker,
        no_finish_on_revolution=no_finish_on_revolution,
    )
    return config, seat_profiles, rounds, reveal_all_hands


def _format_hand(cards) -> str:
    """
    Construit une représentation textuelle triée d'une main.

    Paramètre `cards` : séquence de `Card` à représenter.
    Retourne une chaîne, cartes séparées par des espaces, triées par rang facial croissant, les
    Jokers en position finale. Aucun effet de bord.
    """
    ordered = sorted(cards, key=lambda c: _RANK_DISPLAY_INDEX.get(c.rank.value, len(_RANK_ORDER_FOR_DISPLAY)))
    return " ".join(repr(card) for card in ordered)


class _ExplicitObserver:
    """
    Abonné du bus affichant en clair le déroulement complet de la partie.

    Champ `reveal_all_hands` : indique si l'intégralité des mains doit être affichée à chaque pli.
    Champ `_hands` : reconstruction locale des mains courantes de chaque joueur, alimentée par
    `EventRoundStart`, `EventExchange` et `EventActionPlayed`, à l'image de la reconstruction
    utilisée par `analytics.metrics_calc.missed_interception_rate`.
    """

    def __init__(self, reveal_all_hands: bool) -> None:
        self.reveal_all_hands = reveal_all_hands
        self._hands: Dict[int, List] = {}

    def _print_hands(self) -> None:
        for pid in sorted(self._hands):
            print(f"  Main joueur {pid} : {_format_hand(self._hands[pid])}")

    def __call__(self, event) -> None:
        if isinstance(event, EventRoundStart):
            self._hands = {pid: list(cards) for pid, cards in event.initial_hands.items()}
            print(f"\n=== Manche {event.round_id} : distribution ===")
            if self.reveal_all_hands:
                self._print_hands()
        elif isinstance(event, EventExchange):
            remaining = list(self._hands.get(event.from_player, []))
            for card in event.cards:
                if card in remaining:
                    remaining.remove(card)
            self._hands[event.from_player] = remaining
            self._hands[event.to_player] = self._hands.get(event.to_player, []) + list(event.cards)
        elif isinstance(event, EventActionPlayed):
            remaining = list(self._hands.get(event.player_id, []))
            for card in event.cards_played:
                if card in remaining:
                    remaining.remove(card)
            self._hands[event.player_id] = remaining
            if event.cards_played:
                cards_repr = " ".join(repr(c) for c in event.cards_played)
                print(f"  Joueur {event.player_id} joue : {cards_repr} (puissance {event.resulting_power})")
            else:
                print(f"  Joueur {event.player_id} passe ({event.action_type.value})")
        elif isinstance(event, EventTrickStart):
            print(f"\n-- Pli {event.trick_index} ouvert par le joueur {event.opener_id} --")
            if self.reveal_all_hands:
                self._print_hands()
        elif isinstance(event, EventTrickClosed):
            print(f"-- Pli remporté par le joueur {event.winner_id} --")
        elif isinstance(event, EventRuleTriggered):
            print(f"  [Règle déclenchée] {event.rule_name} par le joueur {event.triggering_player_id}")
        elif isinstance(event, EventPlayerFinished):
            print(f"  Joueur {event.player_id} termine au rang {event.rank} (VP = {event.vp_earned:+.2f})")
        elif isinstance(event, EventRoundEnd):
            print(f"=== Fin de la manche {event.round_id} ===")
            for pid in sorted(event.vp_by_player):
                print(
                    f"  Joueur {pid} : VP = {event.vp_by_player[pid]:+.2f}, "
                    f"rôle suivant = {event.roles_by_player.get(pid, '?')}"
                )


def main() -> None:
    """
    Point d'entrée du mode de partie explicite et guidé.

    Retourne `None`. Effet de bord : conduit un assistant de configuration interactif, exécute la
    partie qui en résulte manche par manche, et affiche l'intégralité du déroulement sur la sortie
    standard, y compris, sur demande, les mains complètes de tous les joueurs à chaque pli.
    """
    config, seat_profiles, rounds, reveal_all_hands = _wizard_config()

    agents: Dict[int, AbstractBaseAgent] = {
        pid: _AGENT_REGISTRY[profile](pid, config)
        for pid, profile in enumerate(seat_profiles)
    }

    bus = EventBus()
    game = Game(config, agents, event_bus=bus, game_id="explicit-game")
    bus.subscribe(_ExplicitObserver(reveal_all_hands))

    for _ in range(rounds):
        game.play_round()

    print("\n=== Total cumulé sur la partie ===")
    for pid in sorted(game.cumulative_vp):
        print(f"  Joueur {pid} : VP total = {game.cumulative_vp[pid]:+.2f}")


if __name__ == "__main__":
    main()
