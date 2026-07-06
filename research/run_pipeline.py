"""
Module du pipeline automatique complet de recherche.

Le module orchestre, sans aucune intervention humaine, l'intégralité d'une campagne de recherche : entraînement de l'agent à politique
linéaire, balayage de plusieurs taux d'apprentissage pour cet agent, tentative d'entraînement distribué de l'agent à politique neuronale
(ignorée proprement si Redis n'est pas joignable), simulations de référence pour chaque profil heuristique disponible, évaluation
comparative de l'agent entraîné contre l'ensemble de ces profils, génération de l'ensemble des graphiques d'analyse, puis rédaction d'un
rapport de synthèse.

Chaque étape est idempotente et journalisée dans un fichier d'état JSON (`data/pipeline_state.json`) : une exécution interrompue en cours de
route reprend exactement où elle s'était arrêtée lors du prochain lancement, sans recommencer les étapes déjà achevées. Chaque étape
affiche sa propre progression fine (barres de progression, chronométrage, identification colorée) plutôt qu'un simple message de début et
de fin, afin qu'une étape longue reste toujours visiblement active plutôt que silencieuse.

Le module dépend de `core.config`, `naming`, `console_theme`, `training.train_rl`, `training.replay_buffer`, `training.launch_distributed`,
`research.run_simulation`, `research.evaluate_agent` et `research.generate_graphs`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import console_theme
import naming
from core.config import GameConfig

_STATE_PATH = os.path.join("data", "pipeline_state.json")
_console = Console()

# Nombre d'étapes macroscopiques du pipeline, utilisé uniquement pour l'affichage "Étape i/N".
_TOTAL_STEPS = 6

# Profils dont la simulation est notoirement plus coûteuse que la moyenne (recherche par rollouts) ; leur nombre de parties de référence est
# réduit proportionnellement pour éviter qu'ils ne dominent la durée totale du pipeline, sans pour autant les exclure de l'analyse.
_EXPENSIVE_PROFILES = ("mcts_bot",)
_EXPENSIVE_PROFILE_GAME_FRACTION = 0.25

# Taille de lot de parties confiée à chaque tâche Ray, propagée à `research.run_simulation` et `research.evaluate_agent` pour garantir une
# progression affichée fréquemment.
_PROGRESS_CHUNK_SIZE = 5


def _load_state() -> Dict[str, Any]:
    """
    Charge l'état de progression du pipeline depuis le disque.

    Retourne un dictionnaire d'état, vide si le fichier d'état n'existe pas encore. Aucun effet de bord.
    """
    if not os.path.exists(_STATE_PATH):
        return {}
    with open(_STATE_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_state(state: Dict[str, Any]) -> None:
    """
    Sauvegarde l'état de progression du pipeline sur le disque.

    Paramètre `state` : dictionnaire d'état complet à sauvegarder.
    Retourne `None`. Effet de bord : crée `data/` si absent et écrit `data/pipeline_state.json`, écrasant tout contenu existant.
    """
    naming.ensure_dir("data")
    with open(_STATE_PATH, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False)


def _run_step(state: Dict[str, Any], step_name: str, step_fn, step_index: int) -> None:
    """
    Exécute une étape du pipeline si elle n'est pas déjà marquée comme achevée.

    Paramètre `state` : dictionnaire d'état courant, mutable et sauvegardé après chaque étape.
    Paramètre `step_name` : identifiant unique de l'étape, clé dans `state`.
    Paramètre `step_fn` : fonction sans argument exécutant l'étape et retournant un dictionnaire de résultats sérialisable en JSON.
    Paramètre `step_index` : index de l'étape au sein de la séquence totale, utilisé pour l'affichage "Étape i/N".
    Retourne `None`. Effet de bord : exécute `step_fn` si l'étape n'est pas marquée `"status": "done"`, affiche un en-tête coloré avant
    exécution et un message de statut coloré après exécution (succès ou échec), met à jour et sauvegarde `state` après chaque tentative.
    Une étape en échec ne bloque pas les étapes suivantes ; elle est retentée au prochain lancement du pipeline.
    """
    existing = state.get(step_name, {})
    if existing.get("status") == "done":
        _console.print(
            f"[{console_theme.STYLE_MUTED}]Étape {step_index}/{_TOTAL_STEPS} '{step_name}' déjà terminée, ignorée."
            f"[/{console_theme.STYLE_MUTED}]"
        )
        return

    _console.rule(f"[{console_theme.STYLE_STEP}]Étape {step_index}/{_TOTAL_STEPS} — {step_name}[/{console_theme.STYLE_STEP}]")
    start = time.time()
    try:
        result = step_fn() or {}
        elapsed = time.time() - start
        state[step_name] = {"status": "done", "result": result, "elapsed_seconds": elapsed}
        _console.print(console_theme.success_text(f"✓ Étape '{step_name}' terminée en {elapsed:.1f}s."))
    except Exception as error:  # noqa: BLE001 - on journalise et on continue le pipeline
        elapsed = time.time() - start
        state[step_name] = {
            "status": "failed",
            "error": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": elapsed,
        }
        _console.print(console_theme.error_text(f"✗ Étape '{step_name}' en échec après {elapsed:.1f}s : {error}"))
    _save_state(state)


def _heuristic_profiles() -> List[str]:
    """
    Détermine dynamiquement l'ensemble des profils heuristiques disponibles pour les campagnes de référence.

    Retourne la liste des clés de `research.run_simulation._AGENT_REGISTRY`, incluant automatiquement tout nouveau profil d'agent heuristique
    enregistré, sans nécessiter de mise à jour manuelle de cette liste. Aucun effet de bord.
    """
    from research.run_simulation import _AGENT_REGISTRY as heuristic_registry

    return list(heuristic_registry.keys())


def _step_train_linear_agent(player_count: int, total_rounds: int, seed: int) -> Dict[str, Any]:
    """
    Entraîne l'agent à politique linéaire `agents.rl_agent.RLAgent` de bout en bout.

    Paramètre `player_count` : nombre de joueurs de la configuration d'entraînement.
    Paramètre `total_rounds` : nombre de manches d'entraînement.
    Paramètre `seed` : graine de reproductibilité.
    Retourne un dictionnaire portant le chemin des poids sauvegardés et le VP moyen final observé. Effet de bord : écrit un fichier de poids
    `.npy`, ses métadonnées JSON, et un historique CSV dans `weights/`. La progression est affichée par la barre de progression interne de
    `training.train_rl.train`.
    """
    from training.train_rl import train

    _console.print(console_theme.info_text(f"Entraînement de l'agent linéaire sur {total_rounds} manches ({player_count} joueurs)…"))
    config = GameConfig(random_seed=seed, player_count=player_count)
    trainee, running_vp = train(config, total_rounds, opponent_pool="mixed")

    output_path = naming.build_weights_filename(
        model_name="pipeline_rl_weights",
        player_count=player_count,
        learning_rate=0.01,
        rounds=total_rounds,
    )
    np.save(output_path, trainee.weights)
    naming.write_weights_metadata(
        output_path,
        {
            "model_name": "pipeline_rl_weights",
            "player_count": player_count,
            "learning_rate": 0.01,
            "rounds_trained": total_rounds,
            "opponent_pool": "mixed",
            "seed": seed,
        },
    )
    history_path = naming.build_weights_metadata_filename(output_path).replace(".meta.json", ".history.csv")
    with open(history_path, "w", encoding="utf-8") as handle:
        handle.write("round_index,vp\n")
        for index, vp in enumerate(running_vp):
            handle.write(f"{index},{vp}\n")

    tail = running_vp[-max(1, len(running_vp) // 20):]
    return {
        "weights_path": output_path,
        "history_path": history_path,
        "final_vp_mean": float(sum(tail) / len(tail)) if tail else 0.0,
    }


def _step_sweep_learning_rates(
    player_count: int, rounds_per_run: int, seed: int, learning_rates: List[float], seeds_per_lr: int,
) -> Dict[str, Any]:
    """
    Entraîne l'agent linéaire sous plusieurs taux d'apprentissage et plusieurs graines, pour caractériser la sensibilité de la
    convergence à ce hyperparamètre.

    Paramètre `player_count` : nombre de joueurs de la configuration d'entraînement.
    Paramètre `rounds_per_run` : nombre de manches d'entraînement par exécution individuelle du balayage.
    Paramètre `seed` : graine de base, dérivée par graine additionnelle pour chaque répétition.
    Paramètre `learning_rates` : liste des taux d'apprentissage testés.
    Paramètre `seeds_per_lr` : nombre de répétitions indépendantes par taux d'apprentissage, permettant une estimation de variance.
    Retourne un dictionnaire portant le chemin du fichier CSV produit (`data/learning_rate_sweep.csv`, consommé par
    `research.generate_graphs`) et le nombre total d'exécutions réalisées. Effet de bord : écrit ce fichier CSV, affiche une barre de
    progression détaillée pour l'ensemble des exécutions.
    """
    import polars as pl

    from training.train_rl import train

    rows: List[Dict[str, Any]] = []
    total_runs = len(learning_rates) * max(1, seeds_per_lr)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=_console,
    ) as progress:
        task = progress.add_task(
            f"[{console_theme.STYLE_STEP}]Balayage des taux d'apprentissage[/{console_theme.STYLE_STEP}]",
            total=total_runs,
        )
        for lr in learning_rates:
            for seed_offset in range(max(1, seeds_per_lr)):
                config = GameConfig(random_seed=seed + seed_offset, player_count=player_count)
                _trainee, running_vp = train(config, rounds_per_run, learning_rate=lr, opponent_pool="mixed")
                tail = running_vp[-max(1, len(running_vp) // 20):]
                rows.append({
                    "learning_rate": lr,
                    "seed_index": seed_offset,
                    "final_vp_mean": float(sum(tail) / len(tail)) if tail else 0.0,
                    "rounds": rounds_per_run,
                })
                progress.update(
                    task,
                    advance=1,
                    description=(
                        f"[{console_theme.STYLE_STEP}]Balayage — lr={lr:g}, répétition {seed_offset + 1}/{seeds_per_lr}"
                        f"[/{console_theme.STYLE_STEP}]"
                    ),
                )

    naming.ensure_dir("data")
    output_csv = os.path.join("data", "learning_rate_sweep.csv")
    pl.DataFrame(rows).write_csv(output_csv)
    return {"output_csv": output_csv, "total_runs": total_runs}


def _step_attempt_distributed_training(
    player_count: int, total_steps: int, redis_host: str, redis_port: int,
) -> Dict[str, Any]:
    """
    Tente un entraînement distribué de l'agent à politique neuronale si Redis est joignable.

    Paramètre `player_count` : nombre de joueurs de la configuration d'entraînement.
    Paramètre `total_steps` : nombre d'étapes de gradient à exécuter si Redis est disponible.
    Paramètre `redis_host`, `redis_port` : coordonnées du serveur Redis à tester.
    Retourne un dictionnaire indiquant si l'entraînement a été exécuté ou ignoré faute de Redis. Effet de bord : si Redis est joignable,
    démarre un cluster Ray local éphémère et exécute l'entraînement distribué complet.
    """
    from training.launch_distributed import launch
    from training.replay_buffer import RedisReplayBuffer

    probe = RedisReplayBuffer(host=redis_host, port=redis_port)
    if not probe.ping():
        _console.print(
            console_theme.warning_text(
                f"Redis non joignable sur {redis_host}:{redis_port}, entraînement neuronal distribué ignoré."
            )
        )
        return {"executed": False, "reason": "Redis non joignable, entraînement neuronal distribué ignoré."}

    _console.print(console_theme.info_text(f"Redis joignable, lancement de l'entraînement distribué ({total_steps} étapes)…"))
    launch(
        num_workers=max(1, os.cpu_count() or 1),
        rounds_per_worker_batch=20,
        opponent_pool="mixed",
        player_count=player_count,
        redis_host=redis_host,
        redis_port=redis_port,
        batch_size=64,
        total_steps=total_steps,
        model_name="pipeline_torch_rl_weights",
    )
    return {"executed": True}


def _step_simulate_baselines(
    player_count: int, games_per_profile: int, rounds_per_game: int, seed: int, profiles: List[str],
) -> Dict[str, Any]:
    """
    Simule une campagne de référence pour chaque profil heuristique disponible.

    Paramètre `player_count` : nombre de joueurs par partie simulée.
    Paramètre `games_per_profile` : nombre de parties simulées par profil, réduit automatiquement pour les profils coûteux
    (`_EXPENSIVE_PROFILES`).
    Paramètre `rounds_per_game` : nombre de manches par partie.
    Paramètre `seed` : graine de base de la campagne.
    Paramètre `profiles` : liste des profils heuristiques à simuler, typiquement `_heuristic_profiles()`.
    Retourne un dictionnaire associant chaque profil au chemin du fichier Parquet produit, au nombre de parties effectivement simulées et
    au temps écoulé. Effet de bord : lance une campagne Ray par profil via `research.run_simulation.launch_research`, affiche un message
    coloré de début et de fin pour chaque profil afin que la progression globale reste visible même si un profil est particulièrement lent.
    """
    from research.run_simulation import launch_research

    results: Dict[str, Any] = {}
    seed_cursor = seed
    for profile_index, profile in enumerate(profiles):
        effective_games = games_per_profile
        if profile in _EXPENSIVE_PROFILES:
            effective_games = max(10, int(games_per_profile * _EXPENSIVE_PROFILE_GAME_FRACTION))

        _console.print(
            console_theme.campaign_text(
                f"› [{profile_index + 1}/{len(profiles)}] Simulation de référence — profil '{profile}' "
                f"({effective_games} parties × {rounds_per_game} manches)"
            )
        )
        start = time.time()
        output_path = naming.build_research_filename(
            "pipeline_baseline", player_count, profile, effective_games, rounds_per_game,
        )
        launch_research(
            total_games=effective_games,
            player_count=player_count,
            rounds_per_game=rounds_per_game,
            agent_profile=profile,
            num_workers=max(1, os.cpu_count() or 1),
            output_parquet=output_path,
            base_seed=seed_cursor,
            experiment_name=f"pipeline_baseline_{profile}",
            progress_chunk_size=_PROGRESS_CHUNK_SIZE,
        )
        elapsed = time.time() - start
        _console.print(console_theme.success_text(f"  ✓ Profil '{profile}' terminé en {elapsed:.1f}s."))
        results[profile] = {"output_parquet": output_path, "games": effective_games, "elapsed_seconds": elapsed}
        seed_cursor += effective_games
    return results


def _step_evaluate_trained_agent(
    player_count: int, games: int, rounds_per_game: int, seed: int, trained_weights_path: Optional[str], profiles: List[str],
) -> Dict[str, Any]:
    """
    Évalue comparativement l'agent linéaire entraîné contre l'ensemble des profils heuristiques disponibles.

    Paramètre `player_count` : nombre de joueurs par partie évaluée.
    Paramètre `games` : nombre de parties simulées pour l'évaluation.
    Paramètre `rounds_per_game` : nombre de manches par partie.
    Paramètre `seed` : graine de base de la campagne d'évaluation.
    Paramètre `trained_weights_path` : chemin des poids entraînés de l'agent linéaire, ou `None` si l'entraînement a échoué (l'évaluation
    utilise alors des poids nuls, comportement par défaut de `agents.rl_agent.RLAgent`).
    Paramètre `profiles` : liste des profils heuristiques disponibles pour occuper les sièges adverses.
    Retourne un dictionnaire portant le chemin du fichier CSV d'évaluation produit et la liste des profils de sièges utilisés. Effet de
    bord : lance une campagne Ray via `research.evaluate_agent.launch_evaluation`.
    """
    from research.evaluate_agent import launch_evaluation

    seat_profiles = ["rl_agent"] + profiles[: max(0, player_count - 1)]
    seat_profiles = seat_profiles[:player_count]
    while len(seat_profiles) < player_count:
        seat_profiles.append(profiles[0] if profiles else "greedy_bot")

    seat_weights = {0: trained_weights_path} if trained_weights_path else None

    _console.print(
        console_theme.campaign_text(
            f"› Évaluation comparative — sièges : {', '.join(seat_profiles)} ({games} parties × {rounds_per_game} manches)"
        )
    )
    output_csv = launch_evaluation(
        total_games=games,
        seat_profiles=seat_profiles,
        rounds_per_game=rounds_per_game,
        num_workers=max(1, os.cpu_count() or 1),
        base_seed=seed,
        experiment_name="pipeline_evaluation",
        seat_weights=seat_weights,
    )
    return {"output_csv": output_csv, "seat_profiles": seat_profiles}


def _step_generate_graphs() -> Dict[str, Any]:
    """
    Génère l'ensemble des graphiques d'analyse à partir des données produites par les étapes précédentes.

    Retourne un dictionnaire vide (les figures sont écrites directement sur disque). Effet de bord : écrit les figures dans `figures/`.
    """
    from research.generate_graphs import generate_all

    generate_all()
    return {}


def _step_generate_final_report(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rédige un rapport de synthèse en texte brut résumant les résultats de la campagne complète.

    Paramètre `state` : état complet du pipeline, utilisé pour extraire les chemins et résultats des étapes précédentes.
    Retourne un dictionnaire portant le chemin du rapport écrit. Effet de bord : écrit `data/final_report.md`.
    """
    naming.ensure_dir("data")
    report_path = os.path.join("data", "final_report.md")

    train_result = state.get("train_linear_agent", {}).get("result", {})
    sweep_result = state.get("sweep_learning_rates", {}).get("result", {})
    distributed_result = state.get("attempt_distributed_training", {}).get("result", {})
    baselines_result = state.get("simulate_baselines", {}).get("result", {})
    evaluation_result = state.get("evaluate_trained_agent", {}).get("result", {})

    lines = [
        "# Rapport de synthèse de la campagne de recherche",
        "",
        "## Entraînement de l'agent à politique linéaire",
        f"- Poids sauvegardés : `{train_result.get('weights_path', 'indisponible')}`",
        f"- VP moyen final observé : {train_result.get('final_vp_mean', 'indisponible')}",
        "",
        "## Balayage des taux d'apprentissage",
        f"- Fichier CSV : `{sweep_result.get('output_csv', 'indisponible')}`",
        f"- Nombre total d'exécutions : {sweep_result.get('total_runs', 'indisponible')}",
        "",
        "## Entraînement distribué de l'agent à politique neuronale",
        f"- Exécuté : {distributed_result.get('executed', 'indisponible')}",
        f"- Détail : {distributed_result.get('reason', 'exécution effective ou statut non déterminé')}",
        "",
        "## Simulations de référence",
    ]
    for profile, info in baselines_result.items():
        elapsed = info.get("elapsed_seconds")
        elapsed_str = f"{elapsed:.1f}s" if isinstance(elapsed, (int, float)) else "indisponible"
        lines.append(
            f"- Profil `{profile}` : {info.get('games', '?')} parties en {elapsed_str}, "
            f"fichier Parquet `{info.get('output_parquet', 'indisponible')}`"
        )

    lines.extend([
        "",
        "## Évaluation comparative",
        f"- Fichier CSV : `{evaluation_result.get('output_csv', 'indisponible')}`",
        f"- Profils de sièges évalués : {evaluation_result.get('seat_profiles', 'indisponible')}",
        "",
        "## Graphiques",
        "- Voir le répertoire `figures/` pour l'ensemble des graphiques générés par `research.generate_graphs`.",
    ])

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return {"report_path": report_path}


def _print_state_summary(state: Dict[str, Any]) -> None:
    """
    Affiche un tableau récapitulatif coloré du statut de chaque étape.

    Paramètre `state` : état complet du pipeline.
    Retourne `None`. Effet de bord : écrit un tableau `rich` sur la sortie standard.
    """
    table = Table(title=f"[{console_theme.STYLE_STEP}]Récapitulatif du pipeline[/{console_theme.STYLE_STEP}]")
    table.add_column("Étape")
    table.add_column("Statut")
    table.add_column("Durée (s)")
    for step_name, info in state.items():
        status = info.get("status", "?")
        elapsed = info.get("elapsed_seconds")
        elapsed_str = f"{elapsed:.1f}" if isinstance(elapsed, (int, float)) else "-"
        if status == "done":
            status_display = f"[{console_theme.STYLE_SUCCESS}]terminée[/{console_theme.STYLE_SUCCESS}]"
        elif status == "failed":
            status_display = f"[{console_theme.STYLE_ERROR}]échec[/{console_theme.STYLE_ERROR}]"
        else:
            status_display = f"[{console_theme.STYLE_MUTED}]{status}[/{console_theme.STYLE_MUTED}]"
        table.add_row(step_name, status_display, elapsed_str)
    _console.print(table)


def run_pipeline(
    player_count: int = 5,
    training_rounds: int = 2000,
    lr_sweep_rounds: int = 300,
    lr_sweep_seeds: int = 3,
    learning_rates: Optional[List[float]] = None,
    distributed_steps: int = 200,
    baseline_games_per_profile: int = 100,
    baseline_rounds_per_game: int = 10,
    evaluation_games: int = 200,
    evaluation_rounds_per_game: int = 20,
    seed: int = 0,
    redis_host: str = "localhost",
    redis_port: int = 6379,
) -> None:
    """
    Exécute l'intégralité du pipeline de recherche, de l'entraînement à la production du rapport final.

    Paramètre `player_count` : nombre de joueurs utilisé pour toutes les étapes du pipeline.
    Paramètre `training_rounds` : nombre de manches d'entraînement de l'agent linéaire principal.
    Paramètre `lr_sweep_rounds` : nombre de manches d'entraînement par exécution du balayage de taux d'apprentissage.
    Paramètre `lr_sweep_seeds` : nombre de répétitions indépendantes par taux d'apprentissage testé.
    Paramètre `learning_rates` : liste des taux d'apprentissage testés, valeurs par défaut `[0.001, 0.003, 0.01, 0.03, 0.1]` si `None`.
    Paramètre `distributed_steps` : nombre d'étapes de gradient tentées pour l'entraînement distribué neuronal.
    Paramètre `baseline_games_per_profile` : nombre de parties simulées par profil heuristique de référence.
    Paramètre `baseline_rounds_per_game` : nombre de manches par partie pour les simulations de référence.
    Paramètre `evaluation_games` : nombre de parties simulées pour l'évaluation comparative finale.
    Paramètre `evaluation_rounds_per_game` : nombre de manches par partie pour l'évaluation comparative finale.
    Paramètre `seed` : graine de base, dérivée distinctement pour chaque étape.
    Paramètre `redis_host`, `redis_port` : coordonnées Redis testées pour l'entraînement distribué.
    Retourne `None`. Effet de bord : exécute séquentiellement toutes les étapes du pipeline, reprend automatiquement à l'étape non achevée
    la plus ancienne en cas de relance après interruption, affiche la progression détaillée de chaque étape, et produit in fine
    `data/final_report.md`, `data/learning_rate_sweep.csv` et `figures/`.
    """
    effective_learning_rates = learning_rates if learning_rates is not None else [0.001, 0.003, 0.01, 0.03, 0.1]
    state = _load_state()
    profiles = _heuristic_profiles()

    _run_step(
        state, "train_linear_agent",
        lambda: _step_train_linear_agent(player_count, training_rounds, seed),
        step_index=1,
    )

    trained_weights_path = state.get("train_linear_agent", {}).get("result", {}).get("weights_path")

    _run_step(
        state, "sweep_learning_rates",
        lambda: _step_sweep_learning_rates(
            player_count, lr_sweep_rounds, seed + 5_000, effective_learning_rates, lr_sweep_seeds,
        ),
        step_index=2,
    )

    _run_step(
        state, "attempt_distributed_training",
        lambda: _step_attempt_distributed_training(player_count, distributed_steps, redis_host, redis_port),
        step_index=3,
    )

    _run_step(
        state, "simulate_baselines",
        lambda: _step_simulate_baselines(
            player_count, baseline_games_per_profile, baseline_rounds_per_game, seed + 10_000, profiles,
        ),
        step_index=4,
    )

    _run_step(
        state, "evaluate_trained_agent",
        lambda: _step_evaluate_trained_agent(
            player_count, evaluation_games, evaluation_rounds_per_game, seed + 20_000, trained_weights_path, profiles,
        ),
        step_index=5,
    )

    _run_step(state, "generate_graphs", _step_generate_graphs, step_index=6)

    _run_step(state, "generate_final_report", lambda: _step_generate_final_report(state), step_index=6)

    _print_state_summary(state)
    _console.print(
        console_theme.success_text("Pipeline terminé. Voir data/final_report.md, data/learning_rate_sweep.csv et figures/.")
    )


def main() -> None:
    """
    Point d'entrée en ligne de commande du pipeline automatique complet.

    Retourne `None`. Effet de bord : lit les arguments de la ligne de commande et invoque `run_pipeline`. L'option `--reset` supprime l'état
    de progression enregistré, forçant une reprise depuis le début. L'option `--quick` réduit fortement tous les volumes de travail, utile
    pour valider rapidement que le pipeline s'exécute de bout en bout sans erreur avant un lancement complet.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline automatique complet : entraînement, balayage d'hyperparamètres, évaluation, graphiques et rapport final."
    )
    parser.add_argument("--player-count", type=int, default=5)
    parser.add_argument("--training-rounds", type=int, default=2000)
    parser.add_argument("--lr-sweep-rounds", type=int, default=300)
    parser.add_argument("--lr-sweep-seeds", type=int, default=3)
    parser.add_argument("--learning-rates", type=str, default=None, help="Liste séparée par des virgules, ex: 0.001,0.01,0.1")
    parser.add_argument("--distributed-steps", type=int, default=200)
    parser.add_argument("--baseline-games-per-profile", type=int, default=100)
    parser.add_argument("--baseline-rounds-per-game", type=int, default=10)
    parser.add_argument("--evaluation-games", type=int, default=200)
    parser.add_argument("--evaluation-rounds-per-game", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--redis-host", type=str, default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument(
        "--quick", action="store_true",
        help="Réduit fortement tous les volumes de travail, pour une validation rapide de bout en bout.",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Supprime l'état de progression enregistré (data/pipeline_state.json) et relance le pipeline depuis le début.",
    )
    args = parser.parse_args()

    if args.reset and os.path.exists(_STATE_PATH):
        os.remove(_STATE_PATH)
        _console.print(console_theme.warning_text("État de progression réinitialisé."))

    learning_rates = (
        [float(token.strip()) for token in args.learning_rates.split(",") if token.strip()]
        if args.learning_rates else None
    )

    training_rounds = args.training_rounds
    lr_sweep_rounds = args.lr_sweep_rounds
    lr_sweep_seeds = args.lr_sweep_seeds
    distributed_steps = args.distributed_steps
    baseline_games_per_profile = args.baseline_games_per_profile
    evaluation_games = args.evaluation_games

    if args.quick:
        training_rounds = min(training_rounds, 100)
        lr_sweep_rounds = min(lr_sweep_rounds, 40)
        lr_sweep_seeds = min(lr_sweep_seeds, 1)
        distributed_steps = min(distributed_steps, 20)
        baseline_games_per_profile = min(baseline_games_per_profile, 10)
        evaluation_games = min(evaluation_games, 10)
        _console.print(console_theme.warning_text("Mode --quick actif : volumes de travail fortement réduits."))

    run_pipeline(
        player_count=args.player_count,
        training_rounds=training_rounds,
        lr_sweep_rounds=lr_sweep_rounds,
        lr_sweep_seeds=lr_sweep_seeds,
        learning_rates=learning_rates,
        distributed_steps=distributed_steps,
        baseline_games_per_profile=baseline_games_per_profile,
        baseline_rounds_per_game=args.baseline_rounds_per_game,
        evaluation_games=evaluation_games,
        evaluation_rounds_per_game=args.evaluation_rounds_per_game,
        seed=args.seed,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
    )


if __name__ == "__main__":
    main()
