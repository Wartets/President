"""
Module de lancement de simulations massives parallélisées.

Le module implémente le lanceur de recherche : distribution de $P$ parties indépendantes sur les cœurs disponibles via `ray`, chaque partie
accumulant ses événements dans un `EventLogger` dédié, vidangé périodiquement au format Parquet. Le module agrège ensuite un sous-ensemble
des métriques de `analytics.metrics_calc` sur l'ensemble des parties simulées.

Le module dépend de `ray`, `core.config`, `registry.agent_registry`, `engine.game_runner`, `analytics.event_logger`,
`analytics.metrics_calc` et `rich`/`tqdm` pour le suivi de progression.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import os
import shutil
import time
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import polars as pl
import ray
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

import naming
from agents.interface import AbstractBaseAgent
from analytics.event_logger import EventLogger
from analytics.live_monitor import LiveMonitor
from analytics.metrics_calc import (
    action_space_entropy, branching_factor_average, e_rev_volatility,
    gini_initial_hand_power, trick_length_average,
)
from core.config import GameConfig
from core.math_utils import f_std
from engine.game_runner import Game
from events.structural import EventPlayerFinished, EventRoundStart
from registry.agent_registry import (
    ALL_AUTOMATED_PROFILES, AUTOMATED_AGENT_REGISTRY, TRAINED_AGENT_PROFILES, build_agent,
)

# Registre des profils heuristiques automatisés (exclut `human_agent`), conservé sous ce nom pour compatibilité
# avec les autres modules du projet qui l'importent directement (`research.run_pipeline`, `research.evaluate_agent`).
_AGENT_REGISTRY: Dict[str, Callable[[int, GameConfig], AbstractBaseAgent]] = AUTOMATED_AGENT_REGISTRY

# Profils entraînables dont la construction nécessite le chargement d'un fichier de poids (`agents.rl_agent.RLAgent` ou
# `agents.torch_rl_agent.TorchRLAgent`). Le nom de chaque profil correspond exactement au nom du module Python dans lequel la classe d'agent est 
# définie.
_TRAINED_AGENT_PROFILES = TRAINED_AGENT_PROFILES

# Ensemble complet des profils utilisables pour un siège, qu'ils proviennent de `_AGENT_REGISTRY` ou de `_TRAINED_AGENT_PROFILES`.
_ALL_SEAT_PROFILES = ALL_AUTOMATED_PROFILES

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
    "linear_vp": {"vp_distribution_type": "LINEAR"},
    "allow_soft_pass": {"pass_type": "ALLOW_SOFT"},
    "skip_turn": {"skip_turn_enabled": True},
    "double_revolution": {"double_revolution_enabled": True},
    "putsch_blind_tax": {"putsch_enabled": True, "blind_tax_enabled": True},
    "finish_penalty_instant": {
        "finish_penalty_enabled": True,
        "finish_penalty_extended": True,
        "no_finish_on_joker": True,
        "no_finish_on_revolution": True,
    },
    "finish_penalty_draw": {
        "finish_penalty_enabled": True,
        "finish_penalty_type": "PENALTY_DRAW_CARDS",
        "finish_penalty_draw_count": 2,
    },
    "strict_remainder": {"strict_remainder_allocation": True},
    "straights_skip_turn": {"straights_enabled": True, "skip_turn_enabled": True},
    "skip_on_equal": {"skip_on_equal": True},
}


def _shuffled_seat_assignment(
    profiles: List[str],
    weights_map: Dict[int, str],
    default_weight_path: Optional[str],
    trained_profiles_check_pid: int,
    rng: random.Random,
) -> Tuple[List[str], Dict[int, str]]:
    """
    Permute aléatoirement l'association entre profils de siège et identifiants de joueurs.

    Paramètre `profiles` : liste ordonnée des profils à répartir, taille `player_count`.
    Paramètre `weights_map` : association entre identifiant de siège d'origine et chemin de poids, à réaligner après permutation.
    Paramètre `default_weight_path` : chemin de poids par défaut associé au siège d'origine `trained_profiles_check_pid` (typiquement 0),
    à réaligner sur sa nouvelle position après permutation.
    Paramètre `trained_profiles_check_pid` : identifiant de siège d'origine auquel `default_weight_path` était historiquement rattaché.
    Paramètre `rng` : générateur pseudo-aléatoire dérivé de la graine de la partie, garantissant une permutation reproductible.
    Retourne un tuple `(shuffled_profiles, shuffled_weights_map)` : `shuffled_profiles` est la liste des profils après permutation uniforme
    des positions de sièges, `shuffled_weights_map` réassocie chaque chemin de poids à la nouvelle position occupée par le siège d'origine
    correspondant. Effet de bord : consomme l'état interne de `rng`.
    """
    n = len(profiles)
    order = list(range(n))
    rng.shuffle(order)
    # `order[new_pid] = old_pid` : le joueur new_pid reçoit le profil qui occupait old_pid avant permutation.
    shuffled_profiles = [profiles[old_pid] for old_pid in order]
    old_to_new = {old_pid: new_pid for new_pid, old_pid in enumerate(order)}
    shuffled_weights_map: Dict[int, str] = {}
    for old_pid, path in weights_map.items():
        if old_pid in old_to_new:
            shuffled_weights_map[old_to_new[old_pid]] = path
    if default_weight_path and trained_profiles_check_pid in old_to_new:
        new_default_pid = old_to_new[trained_profiles_check_pid]
        shuffled_weights_map.setdefault(new_default_pid, default_weight_path)
    return shuffled_profiles, shuffled_weights_map


def _build_agents(
    agent_profile: str,
    config: GameConfig,
    weights_path: Optional[str],
    seat_profiles: Optional[List[str]] = None,
    seat_weights: Optional[Dict[int, str]] = None,
    randomize_seats: bool = False,
    seat_rng: Optional[random.Random] = None,
) -> Dict[int, AbstractBaseAgent]:
    """
    Construit l'association joueur/agent pour une partie donnée.

    Paramètre `agent_profile` : profil appliqué à tous les sièges lorsque `seat_profiles` n'est pas fourni, clé de `_AGENT_REGISTRY` ou de
    `_TRAINED_AGENT_PROFILES`.
    Paramètre `config` : configuration de la partie.
    Paramètre `weights_path` : chemin d'un fichier de poids entraîné par défaut, appliqué au siège d'origine 0 lorsque celui-ci est d'un profil
    entraînable et qu'aucune entrée `seat_weights` ne le concerne.
    Paramètre `seat_profiles` : association ordonnée de profils par siège, un profil distinct par identifiant de joueur ; si `None`, `agent_profile`
    est appliqué à l'ensemble des sièges.
    Paramètre `seat_weights` : association entre identifiant de siège et chemin de poids entraîné, prioritaire sur `weights_path` pour le siège
    concerné.
    Paramètre `randomize_seats` : si vrai, permute aléatoirement l'affectation profil/siège avant construction, afin que l'impact de voisinage
    de table ne soit jamais figé sur les mêmes profils au fil d'une campagne statistique ou d'un entraînement.
    Paramètre `seat_rng` : générateur pseudo-aléatoire dédié à la permutation, requis si `randomize_seats` est vrai.
    Retourne un dictionnaire complet d'agents, de taille `config.player_count`, chaque agent construit via
    `registry.agent_registry.build_agent`. Aucun effet de bord hors chargement disque des poids éventuels et, si `randomize_seats` est vrai,
    consommation de l'état interne de `seat_rng`.
    """
    profiles = list(seat_profiles) if seat_profiles is not None else [agent_profile] * config.player_count
    weights_map = dict(seat_weights or {})

    if randomize_seats:
        if seat_rng is None:
            raise ValueError("seat_rng est requis lorsque randomize_seats est vrai.")
        profiles, weights_map = _shuffled_seat_assignment(profiles, weights_map, weights_path, 0, seat_rng)
        weights_path = None  # déjà réintégré dans weights_map par _shuffled_seat_assignment le cas échéant

    agents: Dict[int, AbstractBaseAgent] = {}
    for pid in range(config.player_count):
        profile = profiles[pid] if pid < len(profiles) else agent_profile
        default_weights = weights_path if (profile in _TRAINED_AGENT_PROFILES and pid == 0) else None
        seat_weight_path = weights_map.get(pid, default_weights)
        agents[pid] = build_agent(profile, pid, config, seat_weight_path)
    return agents


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
        seat_profiles: Optional[List[str]] = None,
        seat_weights: Optional[Dict[int, str]] = None,
    ) -> Dict[str, List[dict]]:
        """
        Exécute séquentiellement `game_count` parties complètes et retourne événements et résumés.

        Paramètre `base_seed` : graine de base, chaque partie du lot utilisant `base_seed + offset` comme graine distincte.
        Paramètre `player_count` : nombre de joueurs $N$ par partie.
        Paramètre `rounds_per_game` : nombre de manches jouées par partie.
        Paramètre `game_count` : nombre de parties du lot confié à cet acteur.
        Paramètre `config_overrides` : champs supplémentaires de `GameConfig` à appliquer à chaque partie du lot.
        Paramètre `weights_path` : chemin d'un fichier de poids entraîné par défaut, transmis à `_build_agents` pour les profils de
        `_TRAINED_AGENT_PROFILES`.
        Paramètre `seat_profiles` : association ordonnée de profils par siège, permettant de composer une partie hétérogène plutôt qu'un profil
        unique appliqué à tous les sièges.
        Paramètre `seat_weights` : association entre identifiant de siège et chemin de poids entraîné.
        Retourne un dictionnaire `{"records": ..., "summaries": ...}` : `records` est la liste plate des enregistrements d'événements agrégés sur
        l'ensemble des parties du lot, `summaries` une liste de dictionnaires de métriques résumées, une entrée par partie. Effet de bord : aucun
        hors de l'acteur, chaque partie utilise un `EventLogger` et un `Game` locaux et jetables.
        """
        overrides = config_overrides or {}
        all_records: List[dict] = []
        summary_rows: List[dict] = []
        base_profiles = list(seat_profiles) if seat_profiles is not None else [self.agent_profile] * player_count
        from analytics.metrics_calc import sub_optimal_pass_rate

        for offset in range(game_count):
            seed = base_seed + offset
            config = GameConfig(random_seed=seed, player_count=player_count, **overrides)
            seat_rng = random.Random(f"{seed}:seat_shuffle")
            shuffled_profiles, shuffled_weights = _shuffled_seat_assignment(
                base_profiles, dict(seat_weights or {}), weights_path, 0, seat_rng,
            )
            agents = _build_agents(self.agent_profile, config, None, shuffled_profiles, shuffled_weights)
            logger = EventLogger()
            from engine.event_bus import EventBus

            bus = EventBus()
            bus.subscribe(logger)
            game_id = f"sim-{seed}"
            game = Game(config, agents, event_bus=bus, game_id=game_id)
            game.play_rounds(rounds_per_game)

            gini_values: List[float] = []
            for round_start in logger.events_of_type(EventRoundStart):
                hand_powers: Dict[int, float] = {
                    pid: float(sum(f_std(c) for c in cards if c.rank.value != "JOKER"))
                    for pid, cards in round_start.initial_hands.items()
                }
                gini_values.append(gini_initial_hand_power(hand_powers))

            # Association durable identité d'agent / siège, indépendante du mélange aléatoire des sièges
            # d'une partie à l'autre, afin que toute analyse ultérieure regroupe par profil réel plutôt
            # que par identifiant de joueur (qui change d'agent occupant d'une partie à l'autre).
            player_profiles = {pid: shuffled_profiles[pid] for pid in range(player_count)}
            last_round_id = rounds_per_game - 1
            player_ranks: Dict[int, Optional[int]] = {pid: None for pid in range(player_count)}
            for event in logger.events_of_type(EventPlayerFinished):
                if event.round_id == last_round_id:
                    player_ranks[event.player_id] = event.rank
            player_suboptimal = {pid: sub_optimal_pass_rate(logger, pid) for pid in range(player_count)}
            player_branching = {pid: branching_factor_average(logger, pid) for pid in range(player_count)}
            player_cumulative_vp = {pid: float(game.cumulative_vp.get(pid, 0.0)) for pid in range(player_count)}

            summary_rows.append({
                "seed": seed,
                "game_id": game_id,
                "player_count": player_count,
                "agent_profile": self.agent_profile,
                "seat_profiles": ",".join(base_profiles),
                "player_profiles": json.dumps(player_profiles),
                "player_ranks": json.dumps(player_ranks),
                "player_suboptimal_rate": json.dumps(player_suboptimal),
                "player_branching": json.dumps(player_branching),
                "player_cumulative_vp": json.dumps(player_cumulative_vp),
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
    seat_profiles: Optional[List[str]] = None,
    seat_weights: Optional[Dict[int, str]] = None,
    return_summary: bool = False,
    progress_chunk_size: int = 5,
    shutdown_ray: bool = True,
    stop_check: Optional[Any] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    Orchestre le lancement d'une campagne de simulation massive et l'agrégation des métriques résultantes.

    Paramètre `total_games` : nombre total de parties à simuler, entier strictement positif.
    Paramètre `player_count` : nombre de joueurs $N$ par partie.
    Paramètre `rounds_per_game` : nombre de manches jouées par partie.
    Paramètre `agent_profile` : profil d'agent appliqué (clé de `_AGENT_REGISTRY` ou de `_TRAINED_AGENT_PROFILES`).
    Paramètre `num_workers` : nombre d'acteurs Ray parallèles, borné par le nombre de cœurs disponibles.
    Paramètre `output_parquet` : chemin du fichier Parquet de destination, nommé automatiquement selon la convention du projet et placé dans
    `data/` si `None`.
    Paramètre `base_seed` : graine de base de la campagne, chaque partie recevant une graine dérivée distincte.
    Paramètre `experiment_name` : nom de la campagne, utilisé pour la nomenclature automatique des fichiers.
    Paramètre `config_overrides` : champs supplémentaires de `GameConfig` appliqués à toutes les parties de la campagne, par exemple un préset
    de `_RULE_PRESETS`.
    Paramètre `weights_path` : chemin d'un fichier de poids entraîné, transmis pour les profils de `_TRAINED_AGENT_PROFILES`.
    Paramètre `seat_profiles` : association ordonnée de profils par siège, permettant de composer une campagne de parties hétérogènes plutôt
    qu'un profil unique appliqué à tous les sièges de toutes les parties.
    Paramètre `seat_weights` : association entre identifiant de siège et chemin de poids entraîné, appliquée à chaque partie de la campagne
    pour les sièges de profil entraînable concernés.
    Paramètre `return_summary` : si vrai, retourne la liste des résumés de métriques par partie plutôt que `None`.
    Paramètre `progress_chunk_size` : nombre maximal de parties confiées à un unique appel `run_batch.remote`. Une valeur faible (défaut 5)
    garantit des mises à jour de progression fréquentes même pour un profil d'agent lent (ex : `mcts_bot`), au prix d'un léger surcoût de
    planification Ray ; une valeur élevée réduit ce surcoût mais peut laisser la progression apparente figée pendant toute la durée d'un
    lot si le profil simulé est particulièrement coûteux.
    Paramètre `shutdown_ray` : si faux, ne ferme pas le cluster Ray à la fin de l'appel, permettant d'enchaîner plusieurs campagnes
    successives (par exemple depuis `research.run_pipeline`) sans reconstruire un cluster à chaque fois.
    Paramètre `stop_check` : fonction sans argument consultée régulièrement pendant l'attente des résultats ; si elle retourne vrai, les
    tâches Ray encore en attente sont annulées au mieux et la fonction retourne immédiatement avec les résultats déjà collectés, sans
    perdre le travail déjà accompli.
    Retourne la liste des résumés de métriques par partie si `return_summary` est vrai, sinon `None`. Effet de bord : initialise un cluster Ray
    local, distribue les parties entre les acteurs par petits lots successifs, écrit le journal agrégé et le résumé de métriques dans `data/`,
    et affiche un résumé via `rich`.
    """
    console = Console()
    ray.init(num_cpus=num_workers, ignore_reinit_error=True, log_to_driver=False)

    # Ray's .remote returns actor handles at runtime; silence static type checker via cast to Any
    workers = [cast(Any, GameSimulationWorker).remote(agent_profile) for _ in range(num_workers)]

    chunk_size = max(1, progress_chunk_size)
    chunks: List[int] = []
    remaining_games = total_games
    while remaining_games > 0:
        take = min(chunk_size, remaining_games)
        chunks.append(take)
        remaining_games -= take

    futures = []
    future_game_counts: Dict[Any, int] = {}
    seed_cursor = base_seed
    for chunk_index, count in enumerate(chunks):
        worker = workers[chunk_index % num_workers]
        future = worker.run_batch.remote(
            seed_cursor, player_count, rounds_per_game, count, config_overrides, weights_path,
            seat_profiles, seat_weights,
        )
        futures.append(future)
        future_game_counts[future] = count
        seed_cursor += count

    start = time.time()
    all_records: List[dict] = []
    all_summaries: List[Dict[str, Any]] = []
    interrupted = False
    with tqdm(total=len(futures), desc="Simulating", unit="batch", mininterval=0.5) as bar, LiveMonitor(console=console) as monitor:
        pending = list(futures)
        while pending:
            if stop_check is not None and stop_check():
                interrupted = True
                for leftover in pending:
                    try:
                        ray.cancel(leftover, force=False)
                    except Exception:
                        pass
                break
            done, pending = ray.wait(pending, num_returns=1, timeout=1.0)
            if not done:
                continue
            batch_result = ray.get(done[0])
            all_records.extend(batch_result["records"])
            all_summaries.extend(batch_result["summaries"])
            monitor.record_games(future_game_counts.get(done[0], 0))
            bar.update(1)
    elapsed = time.time() - start
    if interrupted:
        console.print("[bold dark_orange]Arrêt demandé : campagne interrompue proprement, résultats partiels conservés.[/bold dark_orange]")

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

    if shutdown_ray:
        ray.shutdown()

    if return_summary:
        return all_summaries
    return None


def main() -> None:
    """
    Point d'entrée en ligne de commande du lanceur de recherche.

    Retourne `None`. Effet de bord : lit les arguments de la ligne de commande et invoque `launch_research`, ou affiche des informations de
    contrôle (`--list-profiles`, `--list-presets`, `--dry-run`) sans lancer de campagne.
    """
    parser = argparse.ArgumentParser(description="Lanceur de simulations massives parallélisées")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--player-count", type=int, default=4)
    parser.add_argument("--rounds-per-game", type=int, default=10)
    parser.add_argument(
        "--agent-profile",
        choices=list(_ALL_SEAT_PROFILES),
        default="rule_based_bot",
    )
    parser.add_argument("--seat-profiles", type=str, default=None)
    parser.add_argument("--seat-weights", type=str, default=None)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default="simulation")
    parser.add_argument("--config-preset", choices=list(_RULE_PRESETS.keys()), default="base")
    parser.add_argument("--weights-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--list-profiles", action="store_true",
        help="Affiche la liste des profils de siège disponibles puis quitte sans lancer de campagne.",
    )
    parser.add_argument(
        "--list-presets", action="store_true",
        help="Affiche la liste des présets de règles disponibles puis quitte sans lancer de campagne.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Valide la configuration (GameConfig, profils, poids) et affiche le plan d'exécution sans lancer Ray.",
    )
    args = parser.parse_args()

    if args.list_profiles:
        print("Profils de siège disponibles :")
        for profile in _ALL_SEAT_PROFILES:
            trained_note = " (entraînable, nécessite --weights-path ou --seat-weights)" if profile in _TRAINED_AGENT_PROFILES else ""
            print(f"  - {profile}{trained_note}")
        return

    if args.list_presets:
        print("Présets de règles disponibles :")
        for preset_name in _RULE_PRESETS:
            print(f"  - {preset_name}")
        return

    seat_profiles = (
        [token.strip() for token in args.seat_profiles.split(",")] if args.seat_profiles else None
    )
    seat_weights: Optional[Dict[int, str]] = None
    if args.seat_weights:
        seat_weights = {}
        for token in args.seat_weights.split(","):
            pid_str, _, path = token.partition(":")
            if path:
                seat_weights[int(pid_str.strip())] = path.strip()

    if args.dry_run:
        try:
            GameConfig(random_seed=args.seed, player_count=args.player_count, **_RULE_PRESETS[args.config_preset])
        except ValueError as error:
            print(f"Configuration invalide : {error}")
            return
        print("Configuration valide. Plan d'exécution :")
        print(f"  Parties : {args.games}, joueurs : {args.player_count}, manches/partie : {args.rounds_per_game}")
        print(f"  Profil uniforme : {args.agent_profile}" if not seat_profiles else f"  Profils de sièges : {seat_profiles}")
        print(f"  Préset de règles : {args.config_preset}")
        print(f"  Travailleurs Ray : {args.workers}")
        return

    from checkpoint_utils import GracefulKiller

    killer = GracefulKiller()
    launch_research(
        args.games, args.player_count, args.rounds_per_game,
        args.agent_profile, args.workers, args.output, args.seed,
        experiment_name=args.experiment_name,
        config_overrides=_RULE_PRESETS[args.config_preset],
        weights_path=args.weights_path,
        seat_profiles=seat_profiles,
        seat_weights=seat_weights,
        stop_check=lambda: killer.should_stop,
    )


if __name__ == "__main__":
    main()
