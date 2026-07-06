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
from typing import Any, Dict, List, Optional, Type, cast

import polars as pl
import ray
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

import naming
from agents.greedy_bot import GreedyBot
from agents.interface import AbstractBaseAgent
from agents.mcts_bot import MCTSBot
from agents.random_bot import RandomBot
from agents.rule_based_bot import RuleBasedBot
from analytics.event_logger import EventLogger
from analytics.live_monitor import LiveMonitor
from analytics.metrics_calc import (
    action_space_entropy, branching_factor_average, e_rev_volatility,
    gini_initial_hand_power, trick_length_average,
)
from core.config import GameConfig
from core.math_utils import f_std
from engine.game_runner import Game
from events.structural import EventRoundStart

_AGENT_REGISTRY: Dict[str, Type[Any]] = {
    "greedy": GreedyBot,
    "rule_based": RuleBasedBot,
    "random": RandomBot,
    "mcts": MCTSBot,
}

# Profils spéciaux dont la construction n'utilise pas directement `_AGENT_REGISTRY`, car ils
# nécessitent le chargement d'un fichier de poids entraîné (`agents.rl_agent.RLAgent` ou
# `agents.torch_rl_agent.TorchRLAgent`), le siège 0 recevant l'agent entraîné et les sièges
# suivants un profil `rule_based` de référence.
_TRAINED_AGENT_PROFILES = ("rl_agent", "torch_rl_agent")

# Présets nommés de configuration de règles, utilisés par `--config-preset` et par la recherche
# combinatoire (`research.grid_search`).
_RULE_PRESETS: Dict[str, Dict[str, Any]] = {
    "base": {},
    "straights": {"straights_enabled": True},
    "interception": {"interception_enabled": True, "double_revolution_enabled": True},
    "full": {
        "straights_enabled": True,
        "skip_turn_enabled": True,
        "interception_enabled": True,
        "double_revolution_enabled": True,
        "putsch_enabled": True,
        "blind_tax_enabled": True,
        "finish_penalty_enabled": True,
        "finish_penalty_extended": True,
        "no_finish_on_joker": True,
        "no_finish_on_revolution": True,
    },
    "magic_rank_10": {"magic_card_enabled": True, "magic_card_rank": "10", "magic_two": False},
    "revolution_off": {"revolution_enabled": False},
    "legacy_vp": {"vp_distribution_type": "LEGACY_STEPPED"},
    "allow_soft_pass": {"pass_type": "ALLOW_SOFT"},
}


def _build_agents(
    agent_profile: str,
    config: GameConfig,
    weights_path: Optional[str],
) -> Dict[int, AbstractBaseAgent]:
    """
    Construit l'association joueur/agent pour une partie donnée.

    Paramètre `agent_profile` : profil demandé, clé de `_AGENT_REGISTRY` ou de `_TRAINED_AGENT_PROFILES`.
    Paramètre `config` : configuration de la partie.
    Paramètre `weights_path` : chemin d'un fichier de poids entraîné, utilisé uniquement pour les profils de
    `_TRAINED_AGENT_PROFILES`, l'agent entraîné occupant alors le siège 0.
    Retourne un dictionnaire complet d'agents, de taille `config.player_count`. Aucun effet de bord hors
    chargement disque des poids éventuels.
    """
    if agent_profile == "rl_agent":
        import numpy as np
        from agents.rl_agent import RLAgent

        weights = np.load(weights_path) if weights_path else None
        agents: Dict[int, AbstractBaseAgent] = {0: RLAgent(0, config, weights=weights, epsilon=0.0)}
        for pid in range(1, config.player_count):
            agents[pid] = RuleBasedBot(pid, config)
        return agents

    if agent_profile == "torch_rl_agent":
        from agents.torch_rl_agent import TorchRLAgent

        trained = TorchRLAgent(player_id=0, config=config, epsilon=0.0)
        if weights_path:
            trained.load_weights(weights_path)
        agents = {0: trained}
        for pid in range(1, config.player_count):
            agents[pid] = RuleBasedBot(pid, config)
        return agents

    agent_cls = _AGENT_REGISTRY[agent_profile]
    return {pid: agent_cls(pid, config) for pid in range(config.player_count)}


@ray.remote
class GameSimulationWorker:
    """
    Acteur Ray encapsulant l'exécution séquentielle d'un lot de parties complètes.

    Champ `agent_profile` : nom du profil d'agent appliqué à l'ensemble des sièges (ou au siège 0 pour un
    profil entraîné), clé de `_AGENT_REGISTRY` ou de `_TRAINED_AGENT_PROFILES`.
    """

    def __init__(self, agent_profile: str) -> None:
        self.agent_profile = agent_profile

    def run_batch(
        self,
        base_seed: int,
        player_count: int,
        rounds_per_game: int,
        game_count: int,
        config_overrides: Optional[Dict[str, Any]] = None,
        weights_path: Optional[str] = None,
    ) -> Dict[str, List[dict]]:
        """
        Exécute séquentiellement `game_count` parties complètes et retourne événements et résumés.

        Paramètre `base_seed` : graine de base, chaque partie du lot utilisant `base_seed + offset` comme graine distincte.
        Paramètre `player_count` : nombre de joueurs $N$ par partie.
        Paramètre `rounds_per_game` : nombre de manches jouées par partie.
        Paramètre `game_count` : nombre de parties du lot confié à cet acteur.
        Paramètre `config_overrides` : champs supplémentaires de `GameConfig` à appliquer à chaque partie du lot.
        Paramètre `weights_path` : chemin d'un fichier de poids entraîné, transmis à `_build_agents` pour les profils
        de `_TRAINED_AGENT_PROFILES`.
        Retourne un dictionnaire `{"records": ..., "summaries": ...}` : `records` est la liste plate des
        enregistrements d'événements agrégés sur l'ensemble des parties du lot, `summaries` une liste de
        dictionnaires de métriques résumées, une entrée par partie. Effet de bord : aucun hors de l'acteur,
        chaque partie utilise un `EventLogger` et un `Game` locaux et jetables.
        """
        overrides = config_overrides or {}
        all_records: List[dict] = []
        summary_rows: List[dict] = []
        for offset in range(game_count):
            seed = base_seed + offset
            config = GameConfig(random_seed=seed, player_count=player_count, **overrides)
            agents = _build_agents(self.agent_profile, config, weights_path)
            logger = EventLogger()
            from engine.event_bus import EventBus

            bus = EventBus()
            bus.subscribe(logger)
            game = Game(config, agents, event_bus=bus, game_id=f"sim-{seed}")
            game.play_rounds(rounds_per_game)

            gini_values: List[float] = []
            for round_start in logger.events_of_type(EventRoundStart):
                hand_powers = {
                    pid: sum(f_std(c) for c in cards if c.rank.value != "JOKER")
                    for pid, cards in round_start.initial_hands.items()
                }
                gini_values.append(gini_initial_hand_power(hand_powers))

            summary_rows.append({
                "seed": seed,
                "player_count": player_count,
                "agent_profile": self.agent_profile,
                "rounds_per_game": rounds_per_game,
                "branching_factor_average": branching_factor_average(logger),
                "action_space_entropy": action_space_entropy(logger),
                "e_rev_volatility": e_rev_volatility(logger),
                "trick_length_average": trick_length_average(logger),
                "gini_initial_hand_power_mean": (
                    sum(gini_values) / len(gini_values) if gini_values else 0.0
                ),
                **{f"config_{key}": value for key, value in overrides.items()},
            })

            all_records.extend(logger.to_records())
        return {"records": all_records, "summaries": summary_rows}


def launch_research(
    total_games: int,
    player_count: int,
    rounds_per_game: int,
    agent_profile: str,
    num_workers: int,
    output_parquet: Optional[str],
    base_seed: int,
    experiment_name: str = "simulation",
    config_overrides: Optional[Dict[str, Any]] = None,
    weights_path: Optional[str] = None,
    return_summary: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """
    Orchestre le lancement d'une campagne de simulation massive et l'agrégation des métriques résultantes.

    Paramètre `total_games` : nombre total de parties à simuler, entier strictement positif.
    Paramètre `player_count` : nombre de joueurs $N$ par partie.
    Paramètre `rounds_per_game` : nombre de manches jouées par partie.
    Paramètre `agent_profile` : profil d'agent appliqué (clé de `_AGENT_REGISTRY` ou de `_TRAINED_AGENT_PROFILES`).
    Paramètre `num_workers` : nombre d'acteurs Ray parallèles, borné par le nombre de cœurs disponibles.
    Paramètre `output_parquet` : chemin du fichier Parquet de destination, nommé automatiquement selon la
    convention du projet et placé dans `data/` si `None`.
    Paramètre `base_seed` : graine de base de la campagne, chaque partie recevant une graine dérivée distincte.
    Paramètre `experiment_name` : nom de la campagne, utilisé pour la nomenclature automatique des fichiers.
    Paramètre `config_overrides` : champs supplémentaires de `GameConfig` appliqués à toutes les parties de la
    campagne, par exemple un préset de `_RULE_PRESETS`.
    Paramètre `weights_path` : chemin d'un fichier de poids entraîné, transmis pour les profils de
    `_TRAINED_AGENT_PROFILES`.
    Paramètre `return_summary` : si vrai, retourne la liste des résumés de métriques par partie plutôt que `None`.
    Retourne la liste des résumés de métriques par partie si `return_summary` est vrai, sinon `None`. Effet de
    bord : initialise un cluster Ray local, distribue les parties entre les acteurs, écrit le journal agrégé et
    le résumé de métriques dans `data/`, et affiche un résumé via `rich`.
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
        future = worker.run_batch.remote(
            seed_cursor, player_count, rounds_per_game, count, config_overrides, weights_path,
        )
        futures.append(future)
        future_game_counts[future] = count
        seed_cursor += count

    start = time.time()
    all_records: List[dict] = []
    all_summaries: List[Dict[str, Any]] = []
    with tqdm(total=len(futures), desc="Simulating", unit="batch", mininterval=0.5) as bar, LiveMonitor(console=console) as monitor:
        pending = list(futures)
        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            batch_result = ray.get(done[0])
            all_records.extend(batch_result["records"])
            all_summaries.extend(batch_result["summaries"])
            monitor.record_games(future_game_counts.get(done[0], 0))
            bar.update(1)
    elapsed = time.time() - start

    if output_parquet is None:
        output_parquet = naming.build_research_filename(
            experiment_name, player_count, agent_profile, total_games, rounds_per_game,
        )

    logger = EventLogger()
    logger._parquet_buffer = all_records
    if os.path.isdir(output_parquet):
        shutil.rmtree(output_parquet)
    elif os.path.exists(output_parquet):
        os.remove(output_parquet)
    logger.flush_to_parquet(output_parquet)
    logger.close()

    summary_path = naming.build_research_filename(
        experiment_name, player_count, agent_profile, total_games, rounds_per_game,
        extension="summary.csv",
    )
    if all_summaries:
        pl.DataFrame(all_summaries).write_csv(summary_path)

    table = Table(title=f"Campagne de recherche, profil {agent_profile}")
    table.add_column("Métrique")
    table.add_column("Valeur")
    table.add_row("Parties simulées", str(total_games))
    table.add_row("Durée totale (s)", f"{elapsed:.2f}")
    table.add_row("Parties / seconde", f"{total_games / elapsed:.2f}")
    table.add_row("Événements journalisés", str(len(all_records)))
    table.add_row("Fichier Parquet", output_parquet)
    table.add_row("Fichier de résumé", summary_path if all_summaries else "non écrit (aucun résumé)")
    console.print(table)

    ray.shutdown()

    if return_summary:
        return all_summaries
    return None


def main() -> None:
    """
    Point d'entrée en ligne de commande du lanceur de recherche.

    Retourne `None`. Effet de bord : lit les arguments de la ligne de commande et invoque `launch_research`.
    """
    parser = argparse.ArgumentParser(description="Lanceur de simulations massives parallélisées")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--player-count", type=int, default=4)
    parser.add_argument("--rounds-per-game", type=int, default=10)
    parser.add_argument(
        "--agent-profile",
        choices=list(_AGENT_REGISTRY.keys()) + list(_TRAINED_AGENT_PROFILES),
        default="rule_based",
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default="simulation")
    parser.add_argument("--config-preset", choices=list(_RULE_PRESETS.keys()), default="base")
    parser.add_argument("--weights-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    launch_research(
        args.games, args.player_count, args.rounds_per_game,
        args.agent_profile, args.workers, args.output, args.seed,
        experiment_name=args.experiment_name,
        config_overrides=_RULE_PRESETS[args.config_preset],
        weights_path=args.weights_path,
    )


if __name__ == "__main__":
    main()
