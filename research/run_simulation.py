"""
Module de lancement de simulations massives parallélisées.

Le module implémente le lanceur de recherche : distribution de $P$ parties indépendantes sur les cœurs
disponibles via `ray`, chaque partie accumulant ses événements dans un `EventLogger` dédié, vidangé périodiquement au format Parquet. Le
module agrège ensuite un sous-ensemble des métriques de `analytics.metrics_calc` sur l'ensemble des parties simulées.

Le module dépend de `ray`, `core.config`, `agents.greedy_bot`, `agents.rule_based_bot`, `agents.random_bot`, `agents.mcts_bot`,
`engine.game_runner`, `analytics.event_logger`, `analytics.metrics_calc` et `rich`/`tqdm` pour le suivi de progression.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import os
import shutil
import time
from typing import Any, Dict, List, Type, cast

import ray
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from agents.greedy_bot import GreedyBot
from agents.interface import AbstractBaseAgent
from agents.mcts_bot import MCTSBot
from agents.random_bot import RandomBot
from agents.rule_based_bot import RuleBasedBot
from analytics.event_logger import EventLogger
from analytics.live_monitor import LiveMonitor
from analytics.metrics_calc import (
    action_space_entropy, branching_factor_average, e_rev_volatility,
    gini_initial_hand_power, role_transition_matrix, trick_length_average,
)
from core.config import GameConfig
from engine.game_runner import Game

_AGENT_REGISTRY: Dict[str, Type[Any]] = {
    "greedy": GreedyBot,
    "rule_based": RuleBasedBot,
    "random": RandomBot,
    "mcts": MCTSBot,
}


@ray.remote
class GameSimulationWorker:
    """
    Acteur Ray encapsulant l'exécution séquentielle d'un lot de parties complètes.

    Champ `agent_profile` : nom du profil d'agent appliqué à l'ensemble des sièges, clé de `_AGENT_REGISTRY`.
    """

    def __init__(self, agent_profile: str) -> None:
        self.agent_profile = agent_profile

    def run_batch(
        self, base_seed: int, player_count: int, rounds_per_game: int, game_count: int
    ) -> List[dict]:
        """
        Exécute séquentiellement `game_count` parties complètes et retourne leurs enregistrements d'événements.

        Paramètre `base_seed` : graine de base, chaque partie du lot utilisant `base_seed + offset` comme graine distincte.
        Paramètre `player_count` : nombre de joueurs $N$ par partie.
        Paramètre `rounds_per_game` : nombre de manches jouées par partie.
        Paramètre `game_count` : nombre de parties du lot confié à cet acteur.
        Retourne une liste de dictionnaires plats, un par événement, agrégés sur l'ensemble des parties du lot. Effet de bord : aucun hors
        de l'acteur, chaque partie utilise un `EventLogger` et un `Game` locaux et jetables.
        """
        agent_cls = _AGENT_REGISTRY[self.agent_profile]
        all_records: List[dict] = []
        for offset in range(game_count):
            seed = base_seed + offset
            config = GameConfig(random_seed=seed, player_count=player_count)
            agents = {pid: agent_cls(pid, config) for pid in range(player_count)}
            logger = EventLogger()
            from engine.event_bus import EventBus

            bus = EventBus()
            bus.subscribe(logger)
            game = Game(config, agents, event_bus=bus, game_id=f"sim-{seed}")
            game.play_rounds(rounds_per_game)
            all_records.extend(logger.to_records())
        return all_records


def launch_research(
    total_games: int,
    player_count: int,
    rounds_per_game: int,
    agent_profile: str,
    num_workers: int,
    output_parquet: str,
    base_seed: int,
) -> None:
    """
    Orchestre le lancement d'une campagne de simulation massive et l'agrégation des métriques résultantes.

    Paramètre `total_games` : nombre total de parties à simuler, entier strictement positif.
    Paramètre `player_count` : nombre de joueurs $N$ par partie.
    Paramètre `rounds_per_game` : nombre de manches jouées par partie.
    Paramètre `agent_profile` : profil d'agent appliqué à tous les sièges, clé de `_AGENT_REGISTRY`.
    Paramètre `num_workers` : nombre d'acteurs Ray parallèles, borné par le nombre de cœurs disponibles.
    Paramètre `output_parquet` : chemin du fichier Parquet de destination pour le journal agrégé.
    Paramètre `base_seed` : graine de base de la campagne, chaque partie recevant une graine dérivée distincte.
    Retourne `None`. Effet de bord : initialise un cluster Ray local, distribue les parties entre les acteurs, écrit le journal agrégé au
    format Parquet, et affiche un résumé de métriques via `rich`.
    """
    console = Console()
    ray.init(num_cpus=num_workers, ignore_reinit_error=True, log_to_driver=False)

    games_per_worker = [total_games // num_workers] * num_workers
    for i in range(total_games % num_workers):
        games_per_worker[i] += 1

    # Ray's .remote returns actor handles at runtime; silence static type checker via cast to Any
    workers = [cast(Any, GameSimulationWorker).remote(agent_profile) for _ in range(num_workers)]
    futures = []
    future_game_counts: Dict[Any, int] = {}
    seed_cursor = base_seed
    for worker, count in zip(workers, games_per_worker):
        if count == 0:
            continue
        future = worker.run_batch.remote(seed_cursor, player_count, rounds_per_game, count)
        futures.append(future)
        future_game_counts[future] = count
        seed_cursor += count

    start = time.time()
    all_records: List[dict] = []
    with tqdm(total=len(futures), desc="Simulating", unit="batch", mininterval=0.5) as bar, LiveMonitor(console=console) as monitor:
        pending = list(futures)
        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            batch_records = ray.get(done[0])
            all_records.extend(batch_records)
            monitor.record_games(future_game_counts.get(done[0], 0))
            bar.update(1)
    elapsed = time.time() - start

    logger = EventLogger()
    logger._parquet_buffer = all_records
    if os.path.isdir(output_parquet):
        shutil.rmtree(output_parquet)
    elif os.path.exists(output_parquet):
        os.remove(output_parquet)
    logger.flush_to_parquet(output_parquet)
    logger.close()

    table = Table(title=f"Campagne de recherche, profil {agent_profile}")
    table.add_column("Métrique")
    table.add_column("Valeur")
    table.add_row("Parties simulées", str(total_games))
    table.add_row("Durée totale (s)", f"{elapsed:.2f}")
    table.add_row("Parties / seconde", f"{total_games / elapsed:.2f}")
    table.add_row("Événements journalisés", str(len(all_records)))
    table.add_row("Fichier Parquet", output_parquet)
    console.print(table)

    ray.shutdown()


def main() -> None:
    """
    Point d'entrée en ligne de commande du lanceur de recherche.

    Retourne `None`. Effet de bord : lit les arguments de la ligne de commande et invoque `launch_research`.
    """
    parser = argparse.ArgumentParser(description="Lanceur de simulations massives parallélisées")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--player-count", type=int, default=4)
    parser.add_argument("--rounds-per-game", type=int, default=10)
    parser.add_argument("--agent-profile", choices=list(_AGENT_REGISTRY.keys()), default="rule_based")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--output", type=str, default="research_output.parquet")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    launch_research(
        args.games, args.player_count, args.rounds_per_game,
        args.agent_profile, args.workers, args.output, args.seed,
    )


if __name__ == "__main__":
    main()
