"""
Module du point d'entrée interactif de partie.

Le module implémente `main`, le point d'entrée en ligne de commande permettant de jouer une partie complète en console, avec un siège
humain optionnel et des sièges automatisés parmi les profils exposés par `_AGENT_REGISTRY`. Le module traduit l'intégralité des paramètres
de `core.config.GameConfig` en options de ligne de commande et restitue, après chaque manche, un résumé des rôles et des points de victoire
attribués.

Le module dépend de `core.config`, `agents.interface`, `agents.human_agent`, `agents.random_bot`, `agents.greedy_bot`, `agents.rule_based_bot`,
`agents.mcts_bot`, `engine.game_runner` et `engine.event_bus`.
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict

from agents.greedy_bot import GreedyBot
from agents.human_agent import HumanAgent
from agents.interface import AbstractBaseAgent
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

# Association entre nom de profil de siège et fabrique d'agent instanciable via `(player_id, config)`.
# Chaque clé correspond exactement au nom du module Python définissant la classe d'agent (`agents/<clé>.py`).
_AGENT_REGISTRY: Dict[str, Callable[[int, GameConfig], AbstractBaseAgent]] = {
    "human_agent": HumanAgent,
    "random_bot": RandomBot,
    "greedy_bot": GreedyBot,
    "rule_based_bot": RuleBasedBot,
    "mcts_bot": MCTSBot,
}

# Profils entraînables nécessitant le chargement optionnel d'un fichier de poids via `--weights`.
_TRAINED_AGENT_PROFILES = ("rl_agent", "torch_rl_agent")

# Rôles valides pour l'option --strict-remainder-role, cohérents avec GameConfig.strict_remainder_role.
_VALID_ROLES = (ROLE_PRESIDENT, ROLE_VICE_PRESIDENT, ROLE_NEUTRAL, ROLE_VICE_SCUM, ROLE_SCUM)


def _build_seat_agent(profile: str, pid: int, config: GameConfig, weights_path: str) -> AbstractBaseAgent:
    """
    Construit l'agent d'un unique siège, y compris pour les profils entraînables.

    Paramètre `profile` : nom de profil, clé de `_AGENT_REGISTRY` ou de `_TRAINED_AGENT_PROFILES`.
    Paramètre `pid` : identifiant du joueur occupant le siège.
    Paramètre `config` : configuration de la partie.
    Paramètre `weights_path` : chemin d'un fichier de poids entraîné, chaîne vide si absent, utilisé
    uniquement pour les profils de `_TRAINED_AGENT_PROFILES`.
    Retourne une instance de `AbstractBaseAgent`. Aucun effet de bord hors chargement disque des poids
    éventuels.
    """
    if profile == "rl_agent":
        import numpy as np
        from agents.rl_agent import RLAgent

        weights = np.load(weights_path) if weights_path else None
        return RLAgent(pid, config, weights=weights, epsilon=0.0)
    if profile == "torch_rl_agent":
        from agents.torch_rl_agent import TorchRLAgent

        trained = TorchRLAgent(player_id=pid, config=config, epsilon=0.0)
        if weights_path:
            trained.load_weights(weights_path)
        return trained
    return _AGENT_REGISTRY[profile](pid, config)


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Construit l'analyseur d'arguments de ligne de commande.

    Retourne une instance de `argparse.ArgumentParser` exposant l'intégralité des champs de `GameConfig` ainsi que les options propres au
    script (`--seats`, `--rounds`). Aucun effet de bord.
    """
    parser = argparse.ArgumentParser(description="Partie interactive du Président en console")

    parser.add_argument("--seats", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--weights", type=str, default=None)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--player-count", type=int, default=4)
    parser.add_argument("--first-trick-opener-id", type=int, default=0)

    parser.add_argument("--disable-deck-scaling-auto", action="store_true")
    parser.add_argument("--forced-deck-count", type=int, default=None)

    parser.add_argument("--pass-type", choices=[PASS_TYPE_HARD_ONLY, PASS_TYPE_ALLOW_SOFT], default=PASS_TYPE_HARD_ONLY)
    parser.add_argument(
        "--vp-distribution",
        choices=[VP_DISTRIBUTION_LEGACY_STEPPED, VP_DISTRIBUTION_LINEAR, VP_DISTRIBUTION_SYMMETRICAL],
        default=VP_DISTRIBUTION_SYMMETRICAL,
    )

    parser.add_argument("--disable-jokers", action="store_true")
    parser.add_argument("--disable-magic-two", action="store_true")
    parser.add_argument("--disable-magic-two-single-clears-all", action="store_true")
    parser.add_argument("--magic-card-enabled", action="store_true")
    parser.add_argument("--magic-card-rank", type=str, default="2")
    parser.add_argument("--disable-magic-single-clears-all", action="store_true")

    parser.add_argument("--skip-on-equal", action="store_true")

    parser.add_argument("--disable-revolution", action="store_true")
    parser.add_argument("--double-revolution-enabled", action="store_true")

    parser.add_argument("--straights-enabled", action="store_true")

    parser.add_argument("--skip-turn-enabled", action="store_true")
    parser.add_argument("--skip-turn-rank", type=str, default="8")

    parser.add_argument("--interception-enabled", action="store_true")
    parser.add_argument("--disable-interception-closes-trick", action="store_true")

    parser.add_argument("--putsch-enabled", action="store_true")
    parser.add_argument("--blind-tax-enabled", action="store_true")

    parser.add_argument("--strict-remainder-allocation", action="store_true")
    parser.add_argument("--strict-remainder-role", choices=_VALID_ROLES, default=ROLE_SCUM)

    parser.add_argument("--finish-penalty-enabled", action="store_true")
    parser.add_argument(
        "--finish-penalty-type",
        choices=[PENALTY_INSTANT_SCUM, PENALTY_DRAW_CARDS],
        default=PENALTY_INSTANT_SCUM,
    )
    parser.add_argument("--finish-penalty-draw-count", type=int, default=1)
    parser.add_argument("--finish-penalty-extended", action="store_true")
    parser.add_argument("--no-finish-on-joker", action="store_true")
    parser.add_argument("--no-finish-on-revolution", action="store_true")

    return parser


def _build_config(args: argparse.Namespace) -> GameConfig:
    """
    Traduit les arguments de ligne de commande en instance de `GameConfig`.

    Paramètre `args` : espace de noms retourné par `argparse.ArgumentParser.parse_args`.
    Retourne une instance de `GameConfig`. Lève `ValueError` si la combinaison de champs ne satisfait pas les contraintes structurelles de
    `GameConfig.__post_init__`. Aucun effet de bord.
    """
    return GameConfig(
        random_seed=args.seed,
        player_count=args.player_count,
        first_trick_opener_id=args.first_trick_opener_id,
        deck_scaling_auto=not args.disable_deck_scaling_auto,
        forced_deck_count=args.forced_deck_count,
        pass_type=args.pass_type,
        vp_distribution_type=args.vp_distribution,
        use_jokers=not args.disable_jokers,
        magic_two=not args.disable_magic_two,
        magic_two_single_clears_all=not args.disable_magic_two_single_clears_all,
        magic_card_enabled=args.magic_card_enabled,
        magic_card_rank=args.magic_card_rank,
        magic_single_clears_all=not args.disable_magic_single_clears_all,
        skip_on_equal=args.skip_on_equal,
        revolution_enabled=not args.disable_revolution,
        double_revolution_enabled=args.double_revolution_enabled,
        straights_enabled=args.straights_enabled,
        skip_turn_enabled=args.skip_turn_enabled,
        skip_turn_rank=args.skip_turn_rank,
        interception_enabled=args.interception_enabled,
        interception_closes_trick=not args.disable_interception_closes_trick,
        putsch_enabled=args.putsch_enabled,
        blind_tax_enabled=args.blind_tax_enabled,
        strict_remainder_allocation=args.strict_remainder_allocation,
        strict_remainder_role=args.strict_remainder_role,
        finish_penalty_enabled=args.finish_penalty_enabled,
        finish_penalty_type=args.finish_penalty_type,
        finish_penalty_draw_count=args.finish_penalty_draw_count,
        finish_penalty_extended=args.finish_penalty_extended,
        no_finish_on_joker=args.no_finish_on_joker,
        no_finish_on_revolution=args.no_finish_on_revolution,
    )


def _build_seat_profiles(seats_arg: str, player_count: int) -> list:
    """
    Détermine la liste ordonnée des profils de siège à instancier.

    Paramètre `seats_arg` : valeur brute de `--seats`, chaîne de profils séparés par des virgules, ou `None` si l'option est omise.
    Paramètre `player_count` : nombre de joueurs $N$, utilisé pour valider la taille de la liste et pour construire le profil par défaut.
    Retourne une liste de chaînes de profils, de taille `player_count`. Si `seats_arg` est `None`, le siège 0 reçoit le profil `'human_agent'`
    et les sièges suivants reçoivent le profil `'greedy_bot'`. Lève `ValueError` si un profil est inconnu de `_AGENT_REGISTRY` et de
    `_TRAINED_AGENT_PROFILES`, ou si le nombre de profils fournis diffère de `player_count`. Aucun effet de bord.
    """
    if seats_arg is None:
        return ["human_agent"] + ["greedy_bot"] * (player_count - 1)

    profiles = [token.strip() for token in seats_arg.split(",")]
    if len(profiles) != player_count:
        raise ValueError(
            f"--seats doit contenir exactement {player_count} profil(s), {len(profiles)} fourni(s)."
        )
    valid_profiles = set(_AGENT_REGISTRY) | set(_TRAINED_AGENT_PROFILES)
    for profile in profiles:
        if profile not in valid_profiles:
            raise ValueError(
                f"Profil de siège inconnu : '{profile}'. Profils disponibles : {', '.join(sorted(valid_profiles))}."
            )
    return profiles


def _print_round_summary(round_index: int, vp_by_player: Dict[int, float], roles_by_player: Dict[int, str]) -> None:
    """
    Affiche le résumé d'une manche achevée.

    Paramètre `round_index` : index $m$ de la manche qui vient de se terminer.
    Paramètre `vp_by_player` : association entre identifiant de joueur et point de victoire attribué pour la manche.
    Paramètre `roles_by_player` : association entre identifiant de joueur et rôle attribué pour la manche suivante.
    Retourne `None`. Effet de bord : écrit sur la sortie standard.
    """
    print(f"\n=== Fin de la manche {round_index} ===")
    for pid in sorted(vp_by_player):
        role = roles_by_player.get(pid, "?")
        print(f"  Joueur {pid} : VP = {vp_by_player[pid]:+.2f}, rôle pour la manche suivante = {role}")


def main() -> None:
    """
    Point d'entrée en ligne de commande d'une partie interactive.

    Retourne `None`. Effet de bord : lit les arguments de la ligne de commande, instancie les agents désignés par `--seats`, exécute
    `--rounds` manches via `engine.game_runner.Game`, et affiche un résumé après chaque manche ainsi que le total cumulé en fin de partie.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    config = _build_config(args)
    seat_profiles = _build_seat_profiles(args.seats, config.player_count)
    seat_weights_list = args.weights.split(",") if args.weights else []

    agents: Dict[int, AbstractBaseAgent] = {
        pid: _build_seat_agent(
            profile, pid, config,
            seat_weights_list[pid].strip() if pid < len(seat_weights_list) else "",
        )
        for pid, profile in enumerate(seat_profiles)
    }

    game = Game(config, agents, event_bus=EventBus(), game_id="play-game-cli")

    for round_index in range(args.rounds):
        vp_by_player = game.play_round()
        roles_by_player = game.roles or {}
        _print_round_summary(round_index, vp_by_player, roles_by_player)

    print("\n=== Total cumulé sur la partie ===")
    for pid in sorted(game.cumulative_vp):
        print(f"  Joueur {pid} : VP total = {game.cumulative_vp[pid]:+.2f}")


if __name__ == "__main__":
    main()
