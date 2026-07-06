"""
Module de recherche combinatoire.

Le module exécute `research.run_simulation.launch_research` sur l'ensemble du produit cartésien d'un choix de profils d'agents, de nombres de
joueurs, de présets de règles et de tailles de partie, puis agrège les métriques résumées de chaque combinaison dans un unique fichier
manifeste, afin de faciliter la comparaison statistique de nombreuses configurations distinctes.

Le module dépend de `research.run_simulation`, de `naming` et de `polars`.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import polars as pl

import naming
from research.run_simulation import _RULE_PRESETS, launch_research


def _parse_int_list(raw: str) -> List[int]:
    """
    Analyse une liste d'entiers séparés par des virgules.

    Paramètre `raw` : chaîne de la forme `"4,5,6"`.
    Retourne une liste d'entiers. Aucun effet de bord.
    """
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def _parse_str_list(raw: str) -> List[str]:
    """
    Analyse une liste de chaînes séparées par des virgules.

    Paramètre `raw` : chaîne de la forme `"greedy,rule_based"`.
    Retourne une liste de chaînes dépourvues d'espaces superflus. Aucun effet de bord.
    """
    return [token.strip() for token in raw.split(",") if token.strip()]


def run_grid(
    experiment_name: str,
    agent_profiles: List[str],
    player_counts: List[int],
    rule_presets: List[str],
    rounds_per_game_values: List[int],
    games_per_combo: int,
    num_workers: int,
    base_seed: int,
) -> str:
    """
    Exécute l'ensemble des combinaisons du produit cartésien et agrège les résumés.

    Paramètre `experiment_name` : nom de la campagne combinatoire, utilisé pour la nomenclature des
    fichiers de chaque combinaison et du manifeste agrégé.
    Paramètre `agent_profiles` : liste de profils d'agents à tester.
    Paramètre `player_counts` : liste de nombres de joueurs à tester.
    Paramètre `rule_presets` : liste de noms de présets de `research.run_simulation._RULE_PRESETS`.
    Paramètre `rounds_per_game_values` : liste de nombres de manches par partie à tester.
    Paramètre `games_per_combo` : nombre de parties simulées pour chaque combinaison individuelle.
    Paramètre `num_workers` : nombre d'acteurs Ray parallèles par combinaison.
    Paramètre `base_seed` : graine de base, chaque combinaison recevant une plage de graines distincte.
    Retourne le chemin du fichier manifeste CSV agrégé. Effet de bord : exécute une campagne de
    simulation par combinaison (voir `launch_research`), écrit un fichier Parquet et un résumé par
    combinaison dans `data/`, puis écrit le manifeste agrégé dans `data/`. Une combinaison dont la
    configuration de règles est structurellement invalide (contrainte de `GameConfig.__post_init__`)
    est ignorée avec un message d'avertissement plutôt que d'interrompre la campagne entière.
    """
    all_manifest_rows: List[Dict[str, Any]] = []
    combo_index = 0
    seed_cursor = base_seed

    combos = itertools.product(agent_profiles, player_counts, rule_presets, rounds_per_game_values)
    for agent_profile, player_count, rule_preset, rounds_per_game in combos:
        overrides = dict(_RULE_PRESETS.get(rule_preset, {}))
        combo_name = f"{experiment_name}_combo{combo_index}"
        try:
            summary_rows = launch_research(
                total_games=games_per_combo,
                player_count=player_count,
                rounds_per_game=rounds_per_game,
                agent_profile=agent_profile,
                num_workers=num_workers,
                output_parquet=None,
                base_seed=seed_cursor,
                experiment_name=combo_name,
                config_overrides=overrides,
                weights_path=None,
                return_summary=True,
            )
        except ValueError as error:
            print(f"Combinaison ignorée ({combo_name}) : {error}")
            combo_index += 1
            seed_cursor += games_per_combo
            continue

        for row in summary_rows or []:
            row["rule_preset"] = rule_preset
            row["combo_name"] = combo_name
            all_manifest_rows.append(row)

        combo_index += 1
        seed_cursor += games_per_combo

    manifest_path = naming.build_grid_manifest_filename(experiment_name)
    pl.DataFrame(all_manifest_rows).write_csv(manifest_path)
    print(f"Manifeste de recherche combinatoire écrit dans {manifest_path}")
    return manifest_path


def main() -> None:
    """
    Point d'entrée en ligne de commande de la recherche combinatoire.

    Retourne `None`. Effet de bord : lit les arguments de la ligne de commande et invoque `run_grid`.
    """
    parser = argparse.ArgumentParser(description="Recherche combinatoire multi-configurations")
    parser.add_argument("--experiment-name", type=str, default="grid")
    parser.add_argument("--agent-profiles", type=str, default="greedy,rule_based,random,mcts")
    parser.add_argument("--player-counts", type=str, default="4,5,6")
    parser.add_argument("--rule-presets", type=str, default="base,straights,full")
    parser.add_argument("--rounds-per-game-values", type=str, default="10,50")
    parser.add_argument("--games-per-combo", type=int, default=50)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_grid(
        experiment_name=args.experiment_name,
        agent_profiles=_parse_str_list(args.agent_profiles),
        player_counts=_parse_int_list(args.player_counts),
        rule_presets=_parse_str_list(args.rule_presets),
        rounds_per_game_values=_parse_int_list(args.rounds_per_game_values),
        games_per_combo=args.games_per_combo,
        num_workers=args.workers,
        base_seed=args.seed,
    )


if __name__ == "__main__":
    main()
