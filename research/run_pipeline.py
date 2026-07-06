"""
Module du pipeline automatique complet de recherche.

Le module orchestre, sans intervention humaine, une campagne de recherche incrémentale : entraînement continu (jamais recommencé de zéro
tant que des poids existent déjà) de l'agent linéaire et de l'agent neuronal distribué, balayage de taux d'apprentissage étendu à chaque
lancement plutôt que rejoué à l'identique, simulations de référence et évaluations comparatives couvrant une grille de configurations
(plusieurs nombres de joueurs croisés avec plusieurs présets de règles), génération versionnée des graphiques, puis rédaction d'un rapport
de synthèse relisant l'intégralité des données accumulées sur tous les lancements précédents.

Contrairement à une exécution "tout ou rien", chaque lancement du pipeline ajoute du travail neuf (nouvelles parties, nouvelles manches
d'entraînement, nouvelles combinaisons de la grille) par-dessus ce qui a déjà été calculé lors des lancements précédents, sans jamais
recalculer ni écraser une donnée déjà acquise : le fichier `data/pipeline_manifest.json` conserve la couverture cumulée (parties simulées,
manches entraînées, combinaisons de balayage déjà testées) et sert de source de vérité entre deux lancements. Une interruption brutale
(Ctrl+C, SIGTERM) est prise en compte entre deux unités de travail : le manifeste est sauvegardé avant de quitter, et le prochain lancement
reprend exactement là où le précédent s'est arrêté plutôt que de recommencer les combinaisons déjà couvertes.

Seules les étapes ne gérant pas déjà leur propre affichage de progression (entraînement linéaire, balayage de taux d'apprentissage)
s'exécutent sous le tableau de bord partagé `ProgressManager` ; les étapes qui embarquent leur propre système de rendu (`tqdm` et
`analytics.live_monitor.LiveMonitor` dans `research.run_simulation`/`research.evaluate_agent`, tableaux `rich` périodiques dans
`training.trainer.Trainer`) s'exécutent en dehors de ce tableau de bord, deux instances `rich.live.Live` actives simultanément sur la même
console produisant un affichage corrompu.

Le module dépend de `core.config`, `naming`, `console_theme`, `progress_manager`, `checkpoint_utils`, `training.train_rl`,
`training.trainer`, `training.launch_distributed`, `research.run_simulation`, `research.evaluate_agent` et `research.generate_graphs`.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import console_theme
import naming
from checkpoint_utils import GracefulKiller, atomic_write_json, load_json
from core.config import GameConfig
from progress_manager import ProgressManager

_MANIFEST_PATH = os.path.join("data", "pipeline_manifest.json")
_console = Console()

# Fraction du nombre de parties normalement demandé appliquée aux profils notoirement coûteux (recherche par rollouts), pour éviter qu'ils
# ne dominent la durée totale d'une campagne de référence sans pour autant les exclure de l'analyse.
_EXPENSIVE_PROFILES = ("mcts_bot",)
_EXPENSIVE_PROFILE_GAME_FRACTION = 0.25

_PROGRESS_CHUNK_SIZE = 5

# Décroissance d'exploration utilisée par `training.train_rl.train`, répliquée ici pour estimer un epsilon de reprise cohérent avec le
# nombre de manches déjà entraînées sur un modèle repris plutôt que recréé.
_EPSILON_DECAY = 0.995
_EPSILON_MIN = 0.02
_EPSILON_START = 0.3


def _unique_path(path: str) -> str:
    """
    Garantit un chemin de fichier non déjà existant, par ajout d'un suffixe numérique incrémental.

    Paramètre `path` : chemin candidat.
    Retourne `path` inchangé s'il n'existe pas encore, sinon une variante `<base>_run<N><ext>` avec le plus petit `N >= 2` disponible.
    Ce mécanisme garantit qu'un nouveau lancement de campagne ajoute toujours un nouveau fichier de données plutôt que d'écraser un
    fichier produit par un lancement antérieur, quelle que soit la coïncidence de nommage automatique par date. Aucun effet de bord.
    """
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    counter = 2
    candidate = f"{base}_run{counter}{ext}"
    while os.path.exists(candidate):
        counter += 1
        candidate = f"{base}_run{counter}{ext}"
    return candidate


def _load_manifest() -> Dict[str, Any]:
    """
    Charge le manifeste cumulatif de couverture du pipeline, avec structure par défaut si absent.

    Retourne un dictionnaire de manifeste. Aucun effet de bord.
    """
    default = {
        "schema_version": 2,
        "runs": [],
        "models": {},
        "baseline_coverage": {},
        "evaluation_coverage": {},
        "lr_sweep_coverage": {"combos_done": [], "output_csv": None},
        "graph_versions": {"next": 1},
    }
    loaded = load_json(_MANIFEST_PATH, default=None)
    if loaded is None:
        return default
    for key, value in default.items():
        loaded.setdefault(key, value)
    return loaded


def _save_manifest(manifest: Dict[str, Any]) -> None:
    """
    Sauvegarde le manifeste cumulatif de façon atomique.

    Paramètre `manifest` : dictionnaire complet à sauvegarder.
    Retourne `None`. Effet de bord : écrit `data/pipeline_manifest.json` par une opération atomique, ne laissant jamais le fichier dans
    un état partiellement écrit même en cas d'interruption brutale pendant l'écriture.
    """
    atomic_write_json(_MANIFEST_PATH, manifest)


def _model_key(model_name: str, player_count: int) -> str:
    return f"{model_name}::player{player_count}"


def _baseline_key(player_count: int, profile: str, preset: str) -> str:
    return f"p{player_count}|{profile}|{preset}"


def _eval_key(player_count: int, preset: str) -> str:
    return f"p{player_count}|{preset}"


def _lr_combo_key(player_count: int, learning_rate: float, seed_offset: int) -> str:
    return f"p{player_count}|lr{learning_rate:g}|s{seed_offset}"


def _heuristic_profiles() -> List[str]:
    """
    Détermine dynamiquement l'ensemble des profils heuristiques disponibles pour les campagnes de référence.

    Retourne la liste des clés de `research.run_simulation._AGENT_REGISTRY`, incluant automatiquement tout nouveau profil d'agent
    heuristique enregistré dans `registry.agent_registry`, sans nécessiter de mise à jour manuelle de cette liste. Aucun effet de bord.
    """
    from research.run_simulation import _AGENT_REGISTRY as heuristic_registry

    return list(heuristic_registry.keys())


def _train_linear_agent_incremental(
    manifest: Dict[str, Any],
    player_count: int,
    rounds_increment: int,
    seed: int,
    killer: GracefulKiller,
    progress: ProgressManager,
) -> Dict[str, Any]:
    """
    Continue l'entraînement du modèle linéaire existant pour `player_count`, ou en démarre un nouveau si aucun n'existe encore.

    Paramètre `manifest` : manifeste cumulatif, mis à jour en place avec les nouvelles métadonnées du modèle.
    Paramètre `player_count` : nombre de joueurs de la configuration d'entraînement.
    Paramètre `rounds_increment` : nombre de manches supplémentaires à entraîner lors de cet appel.
    Paramètre `seed` : graine de reproductibilité de la session d'entraînement.
    Paramètre `killer` : indicateur d'arrêt propre, transmis à la boucle d'entraînement pour permettre un arrêt entre deux manches.
    Paramètre `progress` : gestionnaire de barres de progression partagé.
    Retourne un dictionnaire décrivant l'état du modèle après cette session (chemin des poids, manches totales entraînées cumulées,
    VP moyen récent). Effet de bord : écrit un nouveau fichier de poids et une entrée d'historique étendue, jamais un fichier de poids
    déjà existant.
    """
    from training.train_rl import train

    key = _model_key("pipeline_rl_weights", player_count)
    existing = manifest["models"].get(key)

    initial_weights: Optional[np.ndarray] = None
    rounds_already = 0
    history_path: Optional[str] = None
    if existing and os.path.exists(existing.get("latest_weights_path", "")):
        initial_weights = np.load(existing["latest_weights_path"])
        rounds_already = int(existing.get("rounds_trained_total", 0))
        history_path = existing.get("history_path")
        progress.log(f"[cyan]Modèle linéaire p{player_count}[/cyan] : reprise à {rounds_already} manches déjà entraînées.")
    else:
        progress.log(f"[cyan]Modèle linéaire p{player_count}[/cyan] : aucun poids existant, création d'un nouveau modèle.")

    initial_epsilon = max(_EPSILON_MIN, _EPSILON_START * (_EPSILON_DECAY ** rounds_already))
    config = GameConfig(random_seed=seed, player_count=player_count)

    task_id = progress.add_task(f"Entraînement linéaire p{player_count}", total=rounds_increment, min_step_interval=5)
    trainee, running_vp = train(
        config,
        rounds_increment,
        opponent_pool="mixed",
        initial_weights=initial_weights,
        initial_epsilon=initial_epsilon,
        stop_check=lambda: killer.should_stop,
        on_round=lambda index: progress.advance(task_id, 1),
        use_internal_progress=False,
        on_log=progress.log,
    )
    progress.complete_task(task_id, description=f"Entraînement linéaire p{player_count} — terminé")

    rounds_executed = len(running_vp)
    total_rounds = rounds_already + rounds_executed

    output_path = naming.build_weights_filename(
        model_name="pipeline_rl_weights", player_count=player_count, learning_rate=0.01, rounds=total_rounds,
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

    if history_path is None:
        history_path = naming.build_weights_metadata_filename(output_path).replace(".meta.json", ".history.csv")
        with open(history_path, "w", encoding="utf-8") as handle:
            handle.write("round_index,vp\n")

    with open(history_path, "a", encoding="utf-8") as handle:
        for offset, vp in enumerate(running_vp):
            handle.write(f"{rounds_already + offset},{vp}\n")

    tail = running_vp[-max(1, len(running_vp) // 20):] if running_vp else []
    result = {
        "latest_weights_path": output_path,
        "rounds_trained_total": total_rounds,
        "history_path": history_path,
        "final_vp_mean": float(sum(tail) / len(tail)) if tail else existing.get("final_vp_mean", 0.0) if existing else 0.0,
        "rounds_executed_this_run": rounds_executed,
    }
    manifest["models"][key] = result
    return result


def _sweep_learning_rates_incremental(
    manifest: Dict[str, Any],
    player_counts: List[int],
    rounds_per_run: int,
    seed: int,
    learning_rates: List[float],
    seeds_per_lr: int,
    killer: GracefulKiller,
    progress: ProgressManager,
) -> Dict[str, Any]:
    """
    Étend le balayage de taux d'apprentissage avec toute combinaison (joueurs, taux, répétition) non encore couverte.

    Paramètre `manifest` : manifeste cumulatif, mis à jour en place avec les combinaisons désormais couvertes.
    Paramètre `player_counts` : nombres de joueurs à couvrir.
    Paramètre `rounds_per_run` : nombre de manches d'entraînement par exécution individuelle.
    Paramètre `seed` : graine de base.
    Paramètre `learning_rates` : taux d'apprentissage testés.
    Paramètre `seeds_per_lr` : nombre de répétitions indépendantes par taux d'apprentissage et par nombre de joueurs.
    Paramètre `killer` : indicateur d'arrêt propre, consulté entre deux combinaisons.
    Paramètre `progress` : gestionnaire de barres de progression partagé.
    Retourne un dictionnaire portant le chemin du fichier CSV cumulatif et le nombre de nouvelles combinaisons ajoutées lors de cet appel.
    Effet de bord : ajoute des lignes au fichier CSV existant sans jamais supprimer ni recalculer les lignes déjà présentes.
    """
    import polars as pl

    from training.train_rl import train

    coverage = manifest["lr_sweep_coverage"]
    combos_done = set(coverage.get("combos_done", []))
    output_csv = coverage.get("output_csv") or os.path.join("data", "learning_rate_sweep.csv")
    naming.ensure_dir("data")

    all_combos: List[Tuple[int, float, int]] = [
        (pc, lr, seed_offset)
        for pc in player_counts
        for lr in learning_rates
        for seed_offset in range(max(1, seeds_per_lr))
    ]
    pending_combos = [c for c in all_combos if _lr_combo_key(*c) not in combos_done]

    if not pending_combos:
        progress.log("[cyan]Balayage des taux d'apprentissage[/cyan] : toutes les combinaisons demandées sont déjà couvertes.")
        return {"output_csv": output_csv, "new_combos": 0}

    task_id = progress.add_task("Balayage des taux d'apprentissage", total=len(pending_combos))
    new_rows: List[Dict[str, Any]] = []
    added = 0
    for player_count, learning_rate, seed_offset in pending_combos:
        if killer.should_stop:
            progress.log("[yellow]Arrêt demandé, balayage interrompu proprement.[/yellow]")
            break
        config = GameConfig(random_seed=seed + seed_offset, player_count=player_count)
        _trainee, running_vp = train(
            config, rounds_per_run, learning_rate=learning_rate, opponent_pool="mixed",
            use_internal_progress=False,
        )
        tail = running_vp[-max(1, len(running_vp) // 20):] if running_vp else []
        new_rows.append({
            "player_count": player_count,
            "learning_rate": learning_rate,
            "seed_index": seed_offset,
            "final_vp_mean": float(sum(tail) / len(tail)) if tail else 0.0,
            "rounds": rounds_per_run,
        })
        combos_done.add(_lr_combo_key(player_count, learning_rate, seed_offset))
        added += 1
        progress.advance(
            task_id, 1,
            description=f"Balayage — p{player_count}, lr={learning_rate:g}, répétition {seed_offset + 1}/{seeds_per_lr}",
        )
    progress.complete_task(task_id, description="Balayage des taux d'apprentissage — terminé")

    if new_rows:
        new_frame = pl.DataFrame(new_rows)
        if os.path.exists(output_csv):
            existing_frame = pl.read_csv(output_csv)
            combined = pl.concat([existing_frame, new_frame], how="diagonal_relaxed")
        else:
            combined = new_frame
        combined.write_csv(output_csv)

    coverage["combos_done"] = sorted(combos_done)
    coverage["output_csv"] = output_csv
    manifest["lr_sweep_coverage"] = coverage
    return {"output_csv": output_csv, "new_combos": added}


def _attempt_distributed_training_incremental(
    manifest: Dict[str, Any],
    player_counts: List[int],
    steps_increment: int,
    redis_host: str,
    redis_port: int,
) -> Dict[str, Any]:
    """
    Continue (ou démarre) l'entraînement distribué de l'agent neuronal pour chaque nombre de joueurs de la grille, si Redis est joignable.

    Paramètre `manifest` : manifeste cumulatif, mis à jour en place avec les métadonnées du modèle neuronal par nombre de joueurs.
    Paramètre `player_counts` : nombres de joueurs à couvrir.
    Paramètre `steps_increment` : nombre d'étapes de gradient supplémentaires par nombre de joueurs.
    Paramètre `redis_host`, `redis_port` : coordonnées du serveur Redis à tester.
    Retourne un dictionnaire résumant, par nombre de joueurs, si l'entraînement a été exécuté et à partir de quels poids repris. Effet de
    bord : si Redis est joignable, exécute un entraînement distribué complet par nombre de joueurs, en reprenant les poids existants
    plutôt que d'en repartir de zéro. Écrit sur la sortie standard via la console partagée du module, en dehors de tout tableau de bord
    `ProgressManager`, puisque `training.trainer.Trainer` affiche lui-même des tableaux `rich` périodiques pendant l'entraînement.
    """
    from training.launch_distributed import launch
    from training.replay_buffer import RedisReplayBuffer

    probe = RedisReplayBuffer(host=redis_host, port=redis_port)
    if not probe.ping():
        _console.print(
            console_theme.warning_text(
                f"Redis non joignable sur {redis_host}:{redis_port}, entraînement neuronal distribué ignoré pour cette exécution."
            )
        )
        return {"executed": False, "reason": "Redis non joignable."}

    results: Dict[str, Any] = {}
    for player_count in player_counts:
        key = _model_key("pipeline_torch_rl_weights", player_count)
        existing = manifest["models"].get(key)
        resume_path = existing.get("latest_weights_path") if existing and os.path.exists(existing.get("latest_weights_path", "")) else None
        _console.print(
            console_theme.info_text(
                f"Modèle neuronal p{player_count} : "
                + (f"reprise depuis {resume_path}." if resume_path else "démarrage d'un nouveau modèle.")
            )
        )
        launch(
            num_workers=max(1, os.cpu_count() or 1),
            rounds_per_worker_batch=20,
            opponent_pool="mixed",
            player_count=player_count,
            redis_host=redis_host,
            redis_port=redis_port,
            batch_size=64,
            total_steps=steps_increment,
            resume_weights=resume_path,
            model_name="pipeline_torch_rl_weights",
        )
        latest_weights = naming.build_weights_filename(
            model_name="pipeline_torch_rl_weights", player_count=player_count, learning_rate=1e-3,
            rounds=(existing.get("rounds_trained_total", 0) if existing else 0) + steps_increment, extension="pt",
        )
        rounds_trained_total = (existing.get("rounds_trained_total", 0) if existing else 0) + steps_increment
        manifest["models"][key] = {
            "latest_weights_path": latest_weights if os.path.exists(latest_weights) else resume_path,
            "rounds_trained_total": rounds_trained_total,
        }
        results[str(player_count)] = {"executed": True, "resumed_from": resume_path}
    return {"executed": True, "per_player_count": results}


def _simulate_baselines_incremental(
    manifest: Dict[str, Any],
    combos: List[Tuple[int, str, str]],
    games_increment: int,
    rounds_per_game: int,
    seed_base: int,
    killer: GracefulKiller,
) -> Dict[str, Any]:
    """
    Ajoute `games_increment` parties nouvelles pour chaque combinaison (joueurs, profil, préset de règles) de la grille.

    Paramètre `manifest` : manifeste cumulatif, mis à jour en place avec la couverture étendue par combinaison.
    Paramètre `combos` : liste de tuples `(player_count, profile, rule_preset)` à couvrir.
    Paramètre `games_increment` : nombre de parties supplémentaires par combinaison lors de cet appel.
    Paramètre `rounds_per_game` : nombre de manches par partie.
    Paramètre `seed_base` : graine de base, chaque combinaison dérivant sa propre plage de graines cumulative.
    Paramètre `killer` : indicateur d'arrêt propre, consulté entre deux combinaisons.
    Retourne un dictionnaire résumant, par combinaison, le nombre total de parties désormais couvertes et le dernier fichier Parquet
    produit. Effet de bord : lance une campagne Ray par combinaison non interrompue, écrivant systématiquement un nouveau fichier
    Parquet distinct plutôt que d'écraser les segments déjà produits par des lancements antérieurs. La progression de chaque campagne est
    affichée par `research.run_simulation.launch_research` elle-même (`tqdm` + `LiveMonitor`), en dehors de tout tableau de bord
    `ProgressManager` englobant.
    """
    from research.run_simulation import launch_research

    results: Dict[str, Any] = {}
    for combo_index, (player_count, profile, preset) in enumerate(combos):
        if killer.should_stop:
            _console.print(console_theme.warning_text("Arrêt demandé, simulations de référence interrompues proprement."))
            break
        key = _baseline_key(player_count, profile, preset)
        coverage = manifest["baseline_coverage"].get(key, {"games_done": 0, "seed_cursor": seed_base, "parquet_paths": []})

        effective_games = games_increment
        if profile in _EXPENSIVE_PROFILES:
            effective_games = max(5, int(games_increment * _EXPENSIVE_PROFILE_GAME_FRACTION))

        from research.run_simulation import _RULE_PRESETS

        output_path = _unique_path(
            naming.build_research_filename(
                f"pipeline_baseline_{preset}", player_count, profile, effective_games, rounds_per_game,
            )
        )
        _console.print(
            console_theme.info_text(
                f"Baseline {combo_index + 1}/{len(combos)} — p{player_count} / {profile} / {preset} : "
                f"+{effective_games} parties (total après cette exécution : {coverage['games_done'] + effective_games})"
            )
        )
        launch_research(
            total_games=effective_games,
            player_count=player_count,
            rounds_per_game=rounds_per_game,
            agent_profile=profile,
            num_workers=max(1, os.cpu_count() or 1),
            output_parquet=output_path,
            base_seed=coverage["seed_cursor"],
            experiment_name=f"pipeline_baseline_{preset}_{profile}",
            config_overrides=_RULE_PRESETS.get(preset, {}),
            progress_chunk_size=_PROGRESS_CHUNK_SIZE,
            shutdown_ray=False,
        )

        coverage["games_done"] += effective_games
        coverage["seed_cursor"] += effective_games
        coverage.setdefault("parquet_paths", []).append(output_path)
        manifest["baseline_coverage"][key] = coverage
        results[key] = coverage
    return results


def _evaluate_trained_agent_incremental(
    manifest: Dict[str, Any],
    combos: List[Tuple[int, str]],
    games_increment: int,
    rounds_per_game: int,
    seed_base: int,
    profiles: List[str],
    killer: GracefulKiller,
) -> Dict[str, Any]:
    """
    Ajoute `games_increment` parties d'évaluation nouvelles pour chaque combinaison (joueurs, préset de règles) de la grille.

    Paramètre `manifest` : manifeste cumulatif, mis à jour en place.
    Paramètre `combos` : liste de tuples `(player_count, rule_preset)` à couvrir.
    Paramètre `games_increment` : nombre de parties supplémentaires par combinaison.
    Paramètre `rounds_per_game` : nombre de manches par partie.
    Paramètre `seed_base` : graine de base.
    Paramètre `profiles` : profils heuristiques disponibles pour occuper les sièges adverses.
    Paramètre `killer` : indicateur d'arrêt propre.
    Retourne un dictionnaire résumant la couverture par combinaison. Effet de bord : lance une campagne Ray par combinaison, en utilisant
    systématiquement le modèle linéaire le plus récemment entraîné pour le nombre de joueurs concerné, et en écrivant un nouveau fichier
    CSV distinct à chaque appel plutôt que d'écraser les résultats antérieurs. La progression de chaque campagne est affichée par
    `research.evaluate_agent.launch_evaluation` elle-même, en dehors de tout tableau de bord `ProgressManager` englobant.
    """
    from research.evaluate_agent import launch_evaluation
    from research.run_simulation import _RULE_PRESETS

    results: Dict[str, Any] = {}
    for combo_index, (player_count, preset) in enumerate(combos):
        if killer.should_stop:
            _console.print(console_theme.warning_text("Arrêt demandé, évaluations comparatives interrompues proprement."))
            break
        model_key = _model_key("pipeline_rl_weights", player_count)
        trained_weights_path = manifest["models"].get(model_key, {}).get("latest_weights_path")

        seat_profiles = ["rl_agent"] + profiles[: max(0, player_count - 1)]
        seat_profiles = seat_profiles[:player_count]
        while len(seat_profiles) < player_count:
            seat_profiles.append(profiles[0] if profiles else "greedy_bot")
        seat_weights = {0: trained_weights_path} if trained_weights_path else None

        key = _eval_key(player_count, preset)
        coverage = manifest["evaluation_coverage"].get(key, {"games_done": 0, "seed_cursor": seed_base, "csv_paths": []})
        output_csv = _unique_path(
            naming.build_research_filename(
                f"pipeline_evaluation_{preset}", player_count, "rl_agent", games_increment, rounds_per_game, extension="csv",
            )
        )
        _console.print(
            console_theme.info_text(
                f"Évaluation {combo_index + 1}/{len(combos)} — p{player_count} / {preset} : +{games_increment} parties"
            )
        )
        launch_evaluation(
            total_games=games_increment,
            seat_profiles=seat_profiles,
            rounds_per_game=rounds_per_game,
            num_workers=max(1, os.cpu_count() or 1),
            base_seed=coverage["seed_cursor"],
            experiment_name=f"pipeline_evaluation_{preset}",
            config_overrides=_RULE_PRESETS.get(preset, {}),
            seat_weights=seat_weights,
            output_csv=output_csv,
            shutdown_ray=False,
        )
        coverage["games_done"] += games_increment
        coverage["seed_cursor"] += games_increment
        coverage.setdefault("csv_paths", []).append(output_csv)
        manifest["evaluation_coverage"][key] = coverage
        results[key] = coverage
    return results


def _generate_final_report(manifest: Dict[str, Any], figures_version: Optional[int]) -> str:
    """
    Rédige un rapport de synthèse relisant l'intégralité du manifeste cumulatif.

    Paramètre `manifest` : manifeste cumulatif complet.
    Paramètre `figures_version` : numéro de version des graphiques venant d'être générés, ou `None` si l'étape a été sautée.
    Retourne le chemin du rapport écrit. Effet de bord : écrit un rapport versionné `data/final_report_v{N}.md` (N = nombre de lancements
    de pipeline effectués), ainsi qu'une copie de convenance `data/final_report_latest.md` pointant toujours vers le dernier rapport.
    """
    naming.ensure_dir("data")
    run_index = len(manifest["runs"])
    report_path = os.path.join("data", f"final_report_v{run_index}.md")

    lines = ["# Rapport de synthèse cumulatif de la campagne de recherche", ""]

    lines.append("## Modèles entraînés (politique linéaire et neuronale)")
    for key, info in sorted(manifest["models"].items()):
        lines.append(
            f"- `{key}` : {info.get('rounds_trained_total', '?')} manches/étapes cumulées, "
            f"poids `{info.get('latest_weights_path', 'indisponible')}`, "
            f"VP moyen récent : {info.get('final_vp_mean', 'n/a')}"
        )
    lines.append("")

    lines.append("## Balayage des taux d'apprentissage")
    lr_coverage = manifest.get("lr_sweep_coverage", {})
    lines.append(f"- Fichier CSV cumulatif : `{lr_coverage.get('output_csv', 'indisponible')}`")
    lines.append(f"- Combinaisons (joueurs, taux, répétition) couvertes à ce jour : {len(lr_coverage.get('combos_done', []))}")
    lines.append("")

    lines.append("## Couverture des simulations de référence")
    for key, coverage in sorted(manifest["baseline_coverage"].items()):
        lines.append(f"- `{key}` : {coverage.get('games_done', 0)} parties cumulées sur {len(coverage.get('parquet_paths', []))} segment(s)")
    lines.append("")

    lines.append("## Couverture des évaluations comparatives")
    for key, coverage in sorted(manifest["evaluation_coverage"].items()):
        lines.append(f"- `{key}` : {coverage.get('games_done', 0)} parties cumulées sur {len(coverage.get('csv_paths', []))} fichier(s)")
    lines.append("")

    lines.append("## Graphiques")
    if figures_version is not None:
        lines.append(f"- Dernière version générée : `figures/v{figures_version}/`")
    lines.append("- L'historique des versions précédentes des graphiques reste disponible sous `figures/v<N>/`.")

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    latest_path = os.path.join("data", "final_report_latest.md")
    with open(latest_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    return report_path


def _print_manifest_summary(manifest: Dict[str, Any]) -> None:
    """
    Affiche un tableau récapitulatif complet de la couverture cumulée du pipeline.

    Paramètre `manifest` : manifeste cumulatif complet.
    Retourne `None`. Effet de bord : écrit plusieurs tableaux `rich` sur la sortie standard. Appelée uniquement après la fermeture de tout
    tableau de bord `ProgressManager`/`Live` actif, garantissant un rendu correct.
    """
    models_table = Table(title=f"[{console_theme.STYLE_STEP}]Modèles entraînés[/{console_theme.STYLE_STEP}]")
    models_table.add_column("Modèle")
    models_table.add_column("Manches/étapes cumulées")
    models_table.add_column("VP moyen récent")
    for key, info in sorted(manifest["models"].items()):
        models_table.add_row(key, str(info.get("rounds_trained_total", "?")), str(info.get("final_vp_mean", "n/a")))
    _console.print(models_table)

    coverage_table = Table(title=f"[{console_theme.STYLE_STEP}]Couverture cumulée[/{console_theme.STYLE_STEP}]")
    coverage_table.add_column("Combinaison")
    coverage_table.add_column("Type")
    coverage_table.add_column("Parties cumulées")
    for key, coverage in sorted(manifest["baseline_coverage"].items()):
        coverage_table.add_row(key, "baseline", str(coverage.get("games_done", 0)))
    for key, coverage in sorted(manifest["evaluation_coverage"].items()):
        coverage_table.add_row(key, "évaluation", str(coverage.get("games_done", 0)))
    _console.print(coverage_table)


def run_pipeline(
    player_counts: List[int],
    rule_presets: List[str],
    training_rounds_increment: int,
    lr_sweep_rounds: int,
    lr_sweep_seeds: int,
    learning_rates: List[float],
    distributed_steps_increment: int,
    baseline_games_increment: int,
    baseline_rounds_per_game: int,
    evaluation_games_increment: int,
    evaluation_rounds_per_game: int,
    seed: int,
    redis_host: str,
    redis_port: int,
    skip_distributed: bool,
) -> None:
    """
    Exécute une itération incrémentale complète du pipeline de recherche sur toute une grille de configurations.

    Paramètre `player_counts` : nombres de joueurs couverts par la grille d'analyse.
    Paramètre `rule_presets` : présets de règles couverts par la grille d'analyse.
    Paramètre `training_rounds_increment` : manches d'entraînement supplémentaires ajoutées à chaque modèle linéaire lors de cet appel.
    Paramètre `lr_sweep_rounds`, `lr_sweep_seeds`, `learning_rates` : paramètres du balayage de taux d'apprentissage.
    Paramètre `distributed_steps_increment` : étapes de gradient supplémentaires ajoutées à chaque modèle neuronal, si Redis est joignable.
    Paramètre `baseline_games_increment` : parties supplémentaires ajoutées à chaque combinaison de référence.
    Paramètre `baseline_rounds_per_game` : manches par partie de référence.
    Paramètre `evaluation_games_increment` : parties supplémentaires ajoutées à chaque combinaison d'évaluation.
    Paramètre `evaluation_rounds_per_game` : manches par partie d'évaluation.
    Paramètre `seed` : graine de base de cette itération.
    Paramètre `redis_host`, `redis_port` : coordonnées Redis pour l'entraînement distribué.
    Paramètre `skip_distributed` : si vrai, n'essaie même pas de joindre Redis pour cette itération.
    Retourne `None`. Effet de bord : exécute toutes les étapes ci-dessus, sauvegarde le manifeste après chacune (résistant à une
    interruption brutale entre deux étapes), régénère les graphiques (nouvelle version) et le rapport final, puis affiche un résumé complet.
    Le tableau de bord `ProgressManager` n'est actif que pour l'entraînement linéaire et le balayage de taux d'apprentissage ; il est
    explicitement refermé avant les étapes gérant leur propre affichage de progression (entraînement distribué, simulations de référence,
    évaluations comparatives).
    """
    manifest = _load_manifest()
    killer = GracefulKiller()
    profiles = _heuristic_profiles()
    run_started_at = time.time()

    baseline_profile_combos = [
        (pc, profile, "base") for pc in player_counts for profile in profiles
    ] + [
        (pc, "rule_based_bot", preset) for pc in player_counts for preset in rule_presets if preset != "base"
    ]
    evaluation_combos = list(itertools.product(player_counts, rule_presets))

    try:
        with ProgressManager(console=_console) as progress:
            for player_count in player_counts:
                if killer.should_stop:
                    break
                _train_linear_agent_incremental(manifest, player_count, training_rounds_increment, seed, killer, progress)
                _save_manifest(manifest)

            if not killer.should_stop:
                _sweep_learning_rates_incremental(
                    manifest, player_counts, lr_sweep_rounds, seed + 5_000, learning_rates, lr_sweep_seeds, killer, progress,
                )
                _save_manifest(manifest)

        # Le tableau de bord ci-dessus est refermé avant toute étape qui gère son propre affichage de progression.

        if not killer.should_stop and not skip_distributed:
            _attempt_distributed_training_incremental(
                manifest, player_counts, distributed_steps_increment, redis_host, redis_port,
            )
            _save_manifest(manifest)

        if not killer.should_stop:
            _simulate_baselines_incremental(
                manifest, baseline_profile_combos, baseline_games_increment, baseline_rounds_per_game,
                seed + 10_000, killer,
            )
            _save_manifest(manifest)

        if not killer.should_stop:
            _evaluate_trained_agent_incremental(
                manifest, evaluation_combos, evaluation_games_increment, evaluation_rounds_per_game,
                seed + 20_000, profiles, killer,
            )
            _save_manifest(manifest)
    except Exception:  # noqa: BLE001 - on journalise puis on sauvegarde tout de même l'état accumulé
        _console.print(console_theme.error_text(f"Erreur durant le pipeline :\n{traceback.format_exc()}"))
    finally:
        import ray

        try:
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass

    figures_version: Optional[int] = None
    if not killer.should_stop:
        _console.print(console_theme.info_text("Régénération de l'ensemble des graphiques…"))
        from research.generate_graphs import generate_all

        figures_version = generate_all()

    manifest["runs"].append({
        "started_at": run_started_at,
        "finished_at": time.time(),
        "interrupted": killer.should_stop,
        "player_counts": player_counts,
        "rule_presets": rule_presets,
    })
    _save_manifest(manifest)

    report_path = _generate_final_report(manifest, figures_version)
    _print_manifest_summary(manifest)

    if killer.should_stop:
        _console.print(
            console_theme.warning_text(
                f"Pipeline interrompu proprement. Rapport partiel écrit dans {report_path}. Relancer la même commande pour reprendre."
            )
        )
    else:
        _console.print(console_theme.success_text(f"Pipeline terminé pour cette itération. Rapport complet : {report_path}."))


def main() -> None:
    """
    Point d'entrée en ligne de commande du pipeline automatique complet.

    Retourne `None`. Effet de bord : lit les arguments de la ligne de commande et invoque `run_pipeline`. Chaque lancement ajoute du
    travail neuf par-dessus la couverture déjà accumulée dans `data/pipeline_manifest.json` ; l'option `--reset-manifest` supprime cette
    couverture pour repartir d'une campagne vierge. L'option `--quick` réduit fortement tous les volumes de travail par itération, utile
    pour valider rapidement que le pipeline s'exécute de bout en bout sans erreur.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline automatique incrémental : entraînement continu, grille de configurations, graphiques versionnés."
    )
    parser.add_argument("--player-counts", type=str, default="4,5,6")
    parser.add_argument("--rule-presets", type=str, default="base,straights,full")
    parser.add_argument("--training-rounds-increment", type=int, default=1000)
    parser.add_argument("--lr-sweep-rounds", type=int, default=300)
    parser.add_argument("--lr-sweep-seeds", type=int, default=3)
    parser.add_argument("--learning-rates", type=str, default="0.001,0.003,0.01,0.03,0.1")
    parser.add_argument("--distributed-steps-increment", type=int, default=200)
    parser.add_argument("--skip-distributed", action="store_true")
    parser.add_argument("--baseline-games-increment", type=int, default=60)
    parser.add_argument("--baseline-rounds-per-game", type=int, default=10)
    parser.add_argument("--evaluation-games-increment", type=int, default=60)
    parser.add_argument("--evaluation-rounds-per-game", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--redis-host", type=str, default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--reset-manifest", action="store_true",
        help="Supprime la couverture cumulée enregistrée (data/pipeline_manifest.json) et repart d'une campagne vierge.",
    )
    args = parser.parse_args()

    if args.reset_manifest and os.path.exists(_MANIFEST_PATH):
        os.remove(_MANIFEST_PATH)
        _console.print(console_theme.warning_text("Couverture cumulée réinitialisée."))

    player_counts = [int(token.strip()) for token in args.player_counts.split(",") if token.strip()]
    rule_presets = [token.strip() for token in args.rule_presets.split(",") if token.strip()]
    learning_rates = [float(token.strip()) for token in args.learning_rates.split(",") if token.strip()]

    training_rounds_increment = args.training_rounds_increment
    lr_sweep_rounds = args.lr_sweep_rounds
    lr_sweep_seeds = args.lr_sweep_seeds
    distributed_steps_increment = args.distributed_steps_increment
    baseline_games_increment = args.baseline_games_increment
    evaluation_games_increment = args.evaluation_games_increment

    if args.quick:
        training_rounds_increment = min(training_rounds_increment, 80)
        lr_sweep_rounds = min(lr_sweep_rounds, 40)
        lr_sweep_seeds = min(lr_sweep_seeds, 1)
        distributed_steps_increment = min(distributed_steps_increment, 20)
        baseline_games_increment = min(baseline_games_increment, 8)
        evaluation_games_increment = min(evaluation_games_increment, 8)
        player_counts = player_counts[:1]
        rule_presets = rule_presets[:2]
        _console.print(console_theme.warning_text("Mode --quick actif : volumes de travail fortement réduits."))

    run_pipeline(
        player_counts=player_counts,
        rule_presets=rule_presets,
        training_rounds_increment=training_rounds_increment,
        lr_sweep_rounds=lr_sweep_rounds,
        lr_sweep_seeds=lr_sweep_seeds,
        learning_rates=learning_rates,
        distributed_steps_increment=distributed_steps_increment,
        baseline_games_increment=baseline_games_increment,
        baseline_rounds_per_game=args.baseline_rounds_per_game,
        evaluation_games_increment=evaluation_games_increment,
        evaluation_rounds_per_game=args.evaluation_rounds_per_game,
        seed=args.seed,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        skip_distributed=args.skip_distributed,
    )


if __name__ == "__main__":
    main()
