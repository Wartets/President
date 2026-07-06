"""
Module d'évaluation comparative directe d'agents et de modèles entraînés.

Le module exécute des parties complètes via `engine.game_runner.Game` (moteur événementiel complet, toutes règles avancées comprises) pour un
ensemble hétérogène de profils de sièges, et enregistre par joueur et par partie le point de victoire cumulé sur les manches jouées ainsi que
le nombre de manches terminées au rôle `ROLE_PRESIDENT`. Le résultat est exporté en CSV, une ligne par joueur et par partie, afin de comparer
directement des profils fixes, des profils heuristiques, et un ou plusieurs modèles entraînés chargés par poids.

Le module dépend de `ray`, `core.config`, `engine.event_bus`, `engine.game_runner`, `research.run_simulation` et `naming`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, cast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import polars as pl
import ray
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

import naming
from core.config import GameConfig, ROLE_PRESIDENT
from engine.event_bus import EventBus
from engine.game_runner import Game
from research.run_simulation import _RULE_PRESETS, _build_agents


@ray.remote
class EvaluationWorker:
    """
    Acteur Ray exécutant séquentiellement un lot de parties d'évaluation pour une combinaison de sièges fixe.
    """

    def run_batch(
        self,
        base_seed: int,
        seat_profiles: List[str],
        rounds_per_game: int,
        game_count: int,
        seat_weights: Optional[Dict[int, str]],
        config_overrides: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Exécute séquentiellement `game_count` parties et retourne, pour chaque joueur et chaque partie, le point de victoire cumulé et le nombre
        de manches terminées au rôle `ROLE_PRESIDENT`.

        Paramètre `base_seed` : graine de base, chaque partie utilisant `base_seed + offset`.
        Paramètre `seat_profiles` : liste ordonnée de profils, un par siège.
        Paramètre `rounds_per_game` : nombre de manches jouées par partie.
        Paramètre `game_count` : nombre de parties du lot confié à cet acteur.
        Paramètre `seat_weights` : association entre identifiant de siège et chemin de poids entraîné.
        Paramètre `config_overrides` : champs supplémentaires de `GameConfig` appliqués à chaque partie du lot.
        Retourne une liste de dictionnaires, une entrée par joueur et par partie. Effet de bord : aucun hors de l'acteur, chaque partie utilise
        un `Game` local et jetable.
        """
        overrides = config_overrides or {}
        player_count = len(seat_profiles)
        rows: List[Dict[str, Any]] = []
        for offset in range(game_count):
            seed = base_seed + offset
            config = GameConfig(random_seed=seed, player_count=player_count, **overrides)
            agents = _build_agents(seat_profiles[0], config, None, seat_profiles, seat_weights)
            game = Game(config, agents, event_bus=EventBus(), game_id=f"eval-{seed}")

            cumulative_vp = {pid: 0.0 for pid in range(player_count)}
            president_rounds = {pid: 0 for pid in range(player_count)}
            for _ in range(rounds_per_game):
                vp_by_player = game.play_round()
                for pid, vp in vp_by_player.items():
                    cumulative_vp[pid] += vp
                for pid, role in (game.roles or {}).items():
                    if role == ROLE_PRESIDENT:
                        president_rounds[pid] += 1

            for pid in range(player_count):
                rows.append({
                    "seed": seed,
                    "player_id": pid,
                    "profile": seat_profiles[pid],
                    "cumulative_vp": cumulative_vp[pid],
                    "president_rounds": president_rounds[pid],
                    "rounds_per_game": rounds_per_game,
                    "player_count": player_count,
                })
        return rows


def launch_evaluation(
    total_games: int,
    seat_profiles: List[str],
    rounds_per_game: int,
    num_workers: int,
    base_seed: int,
    experiment_name: str = "evaluation",
    config_overrides: Optional[Dict[str, Any]] = None,
    seat_weights: Optional[Dict[int, str]] = None,
    output_csv: Optional[str] = None,
    shutdown_ray: bool = True,
    stop_check: Optional[Callable[[], bool]] = None,
) -> str:
    """
    Orchestre une campagne d'évaluation comparative répartie sur plusieurs acteurs Ray.

    Paramètre `total_games` : nombre total de parties à simuler.
    Paramètre `seat_profiles` : liste ordonnée de profils, un par siège, définissant le nombre de joueurs.
    Paramètre `rounds_per_game` : nombre de manches jouées par partie.
    Paramètre `num_workers` : nombre d'acteurs Ray parallèles.
    Paramètre `base_seed` : graine de base de la campagne.
    Paramètre `experiment_name` : nom de la campagne, utilisé pour la nomenclature du fichier de sortie.
    Paramètre `config_overrides` : champs supplémentaires de `GameConfig` appliqués à toutes les parties.
    Paramètre `seat_weights` : association entre identifiant de siège et chemin de poids entraîné.
    Paramètre `output_csv` : chemin du fichier CSV de destination, nommé automatiquement si `None`.
    Retourne le chemin du fichier CSV écrit. Effet de bord : initialise un cluster Ray local, distribue les parties entre les acteurs,
    écrit le fichier CSV, et affiche un résumé via `rich`.
    """
    console = Console()
    ray.init(num_cpus=num_workers, ignore_reinit_error=True, log_to_driver=False)

    workers = [cast(Any, EvaluationWorker).remote() for _ in range(num_workers)]

    # Répartition en petits lots successifs plutôt qu'en un unique lot par acteur, afin que la progression affichée se mette à jour régulièrement
    # même si un profil de siège coûteux (par exemple mcts_bot) ralentit fortement un sous-ensemble des parties.
    chunk_size = max(1, min(5, total_games))
    chunks: List[int] = []
    remaining_games = total_games
    while remaining_games > 0:
        take = min(chunk_size, remaining_games)
        chunks.append(take)
        remaining_games -= take

    futures = []
    seed_cursor = base_seed
    for chunk_index, count in enumerate(chunks):
        worker = workers[chunk_index % num_workers]
        futures.append(
            worker.run_batch.remote(
                seed_cursor, seat_profiles, rounds_per_game, count, seat_weights, config_overrides,
            )
        )
        seed_cursor += count

    start = time.time()
    all_rows: List[Dict[str, Any]] = []
    pending = list(futures)
    with tqdm(total=len(futures), desc="Évaluation", unit="lot") as bar:
        while pending:
            if stop_check is not None and stop_check():
                for leftover in pending:
                    try:
                        ray.cancel(leftover, force=False)
                    except Exception:
                        pass
                break
            done, pending = ray.wait(pending, num_returns=1, timeout=1.0)
            if not done:
                continue
            all_rows.extend(ray.get(done[0]))
            bar.update(1)
    elapsed = time.time() - start

    if output_csv is None:
        output_csv = naming.build_research_filename(
            experiment_name, len(seat_profiles), seat_profiles[0], total_games, rounds_per_game,
            extension="csv",
        )
    pl.DataFrame(all_rows).write_csv(output_csv)

    table = Table(title=f"Évaluation comparative, {len(seat_profiles)} sièges")
    table.add_column("Métrique")
    table.add_column("Valeur")
    table.add_row("Profils de sièges", ", ".join(seat_profiles))
    table.add_row("Parties simulées", str(total_games))
    table.add_row("Durée totale (s)", f"{elapsed:.2f}")
    table.add_row("Fichier CSV", output_csv)
    console.print(table)

    if shutdown_ray:
        ray.shutdown()
    return output_csv


def main() -> None:
    """
    Point d'entrée en ligne de commande de l'évaluation comparative.

    Retourne `None`. Effet de bord : lit les arguments de la ligne de commande et invoque `launch_evaluation`, ou affiche des informations de
    contrôle (`--list-profiles`, `--list-presets`) sans lancer de campagne.
    """
    parser = argparse.ArgumentParser(description="Évaluation comparative d'agents et de modèles entraînés")
    parser.add_argument("--seat-profiles", type=str, default=None)
    parser.add_argument("--seat-weights", type=str, default=None)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--rounds-per-game", type=int, default=20)
    parser.add_argument("--config-preset", choices=list(_RULE_PRESETS.keys()), default="base")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment-name", type=str, default="evaluation")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--list-profiles", action="store_true",
        help="Affiche la liste des profils de siège disponibles puis quitte sans lancer de campagne.",
    )
    parser.add_argument(
        "--list-presets", action="store_true",
        help="Affiche la liste des présets de règles disponibles puis quitte sans lancer de campagne.",
    )
    args = parser.parse_args()

    if args.list_profiles:
        from research.run_simulation import _ALL_SEAT_PROFILES, _TRAINED_AGENT_PROFILES
        print("Profils de siège disponibles :")
        for profile in _ALL_SEAT_PROFILES:
            trained_note = " (entraînable, nécessite --seat-weights)" if profile in _TRAINED_AGENT_PROFILES else ""
            print(f"  - {profile}{trained_note}")
        return

    if args.list_presets:
        print("Présets de règles disponibles :")
        for preset_name in _RULE_PRESETS:
            print(f"  - {preset_name}")
        return

    if not args.seat_profiles:
        parser.error("--seat-profiles est requis en dehors de --list-profiles/--list-presets.")

    seat_profiles = [token.strip() for token in args.seat_profiles.split(",") if token.strip()]

    seat_weights: Optional[Dict[int, str]] = None
    if args.seat_weights:
        seat_weights = {}
        for token in args.seat_weights.split(","):
            pid_str, _, path = token.partition(":")
            if path:
                seat_weights[int(pid_str.strip())] = path.strip()

    from checkpoint_utils import GracefulKiller

    killer = GracefulKiller()
    launch_evaluation(
        total_games=args.games,
        seat_profiles=seat_profiles,
        rounds_per_game=args.rounds_per_game,
        num_workers=args.workers,
        base_seed=args.seed,
        experiment_name=args.experiment_name,
        config_overrides=_RULE_PRESETS[args.config_preset],
        seat_weights=seat_weights,
        output_csv=args.output,
        stop_check=lambda: killer.should_stop,
    )


if __name__ == "__main__":
    main()
