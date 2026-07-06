"""
Module du pipeline automatique complet de recherche.

Le module orchestre, sans aucune intervention humaine, l'intégralité d'une campagne de recherche : entraînement de l'agent à politique
linéaire, tentative d'entraînement distribué de l'agent à politique neuronale (ignorée proprement si Redis n'est pas joignable), simulations
de référence pour chaque profil heuristique, évaluation comparative de l'agent entraîné contre les profils de référence, génération de
l'ensemble des graphiques d'analyse, puis rédaction d'un rapport de synthèse.

Chaque étape est idempotente et journalisée dans un fichier d'état JSON (`data/pipeline_state.json`) : une exécution interrompue en cours de
route reprend exactement où elle s'était arrêtée lors du prochain lancement, sans recommencer les étapes déjà achevées.

Le module dépend de `core.config`, `naming`, `training.train_rl`, `training.replay_buffer`, `training.launch_distributed`,
`research.run_simulation`, `research.evaluate_agent` et `research.generate_graphs`.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from typing import Any, Dict, Optional

import numpy as np

import naming
from core.config import GameConfig

_STATE_PATH = os.path.join("data", "pipeline_state.json")

# Profils heuristiques de référence utilisés pour l'évaluation comparative.
_BASELINE_PROFILES = ("random_bot", "greedy_bot", "rule_based_bot", "mcts_bot")


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


def _run_step(state: Dict[str, Any], step_name: str, step_fn) -> None:
    """
    Exécute une étape du pipeline si elle n'est pas déjà marquée comme achevée.

    Paramètre `state` : dictionnaire d'état courant, mutable et sauvegardé après chaque étape.
    Paramètre `step_name` : identifiant unique de l'étape, clé dans `state`.
    Paramètre `step_fn` : fonction sans argument exécutant l'étape et retournant un dictionnaire de résultats sérialisable en JSON, fusionné
    dans l'entrée d'état de l'étape.
    Retourne `None`. Effet de bord : exécute `step_fn` si l'étape n'est pas marquée `"status": "done"`, met à jour et sauvegarde `state` après
    chaque tentative, qu'elle réussisse ou échoue. Une étape en échec ne bloque pas les étapes suivantes ; elle est retentée au prochain
    lancement du pipeline.
    """
    existing = state.get(step_name, {})
    if existing.get("status") == "done":
        print(f"[pipeline] étape '{step_name}' déjà terminée, ignorée.")
        return

    print(f"[pipeline] démarrage de l'étape '{step_name}'...")
    try:
        result = step_fn() or {}
        state[step_name] = {"status": "done", "result": result}
        print(f"[pipeline] étape '{step_name}' terminée.")
    except Exception as error:  # noqa: BLE001 - on journalise et on continue le pipeline
        state[step_name] = {
            "status": "failed",
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        print(f"[pipeline] étape '{step_name}' en échec : {error}")
    _save_state(state)


def _step_train_linear_agent(player_count: int, total_rounds: int, seed: int) -> Dict[str, Any]:
    """
    Entraîne l'agent à politique linéaire `agents.rl_agent.RLAgent` de bout en bout.

    Paramètre `player_count` : nombre de joueurs de la configuration d'entraînement.
    Paramètre `total_rounds` : nombre de manches d'entraînement.
    Paramètre `seed` : graine de reproductibilité.
    Retourne un dictionnaire portant le chemin des poids sauvegardés et le VP moyen final observé. Effet de bord : écrit un fichier de poids
    `.npy`, ses métadonnées JSON, et un historique CSV dans `weights/`.
    """
    from training.train_rl import train

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


def _step_attempt_distributed_training(player_count: int, total_steps: int, redis_host: str, redis_port: int) -> Dict[str, Any]:
    """
    Tente un entraînement distribué de l'agent à politique neuronale si Redis est joignable.

    Paramètre `player_count` : nombre de joueurs de la configuration d'entraînement.
    Paramètre `total_steps` : nombre d'étapes de gradient à exécuter si Redis est disponible.
    Paramètre `redis_host`, `redis_port` : coordonnées du serveur Redis à tester.
    Retourne un dictionnaire indiquant si l'entraînement a été exécuté ou ignoré faute de Redis. Effet de bord : si Redis est joignable,
    démarre un cluster Ray local éphémère et exécute l'entraînement distribué complet (voir `training.launch_distributed.launch`).
    """
    from training.replay_buffer import RedisReplayBuffer
    from training.launch_distributed import launch

    probe = RedisReplayBuffer(host=redis_host, port=redis_port)
    if not probe.ping():
        return {"executed": False, "reason": "Redis non joignable, entraînement neuronal distribué ignoré."}

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


def _step_simulate_baselines(player_count: int, games_per_profile: int, rounds_per_game: int, seed: int) -> Dict[str, Any]:
    """
    Simule une campagne de référence pour chaque profil heuristique disponible.

    Paramètre `player_count` : nombre de joueurs par partie simulée.
    Paramètre `games_per_profile` : nombre de parties simulées par profil.
    Paramètre `rounds_per_game` : nombre de manches par partie.
    Paramètre `seed` : graine de base de la campagne.
    Retourne un dictionnaire associant chaque profil au chemin du fichier Parquet et du résumé CSV produits. Effet de bord : lance une
    campagne Ray par profil via `research.run_simulation.launch_research`.
    """
    from research.run_simulation import launch_research

    results: Dict[str, Any] = {}
    for offset, profile in enumerate(_BASELINE_PROFILES):
        output_path = naming.build_research_filename(
            "pipeline_baseline", player_count, profile, games_per_profile, rounds_per_game,
        )
        launch_research(
            total_games=games_per_profile,
            player_count=player_count,
            rounds_per_game=rounds_per_game,
            agent_profile=profile,
            num_workers=max(1, os.cpu_count() or 1),
            output_parquet=output_path,
            base_seed=seed + offset * games_per_profile,
            experiment_name=f"pipeline_baseline_{profile}",
        )
        results[profile] = {"output_parquet": output_path}
    return results


def _step_evaluate_trained_agent(
    player_count: int, games: int, rounds_per_game: int, seed: int, trained_weights_path: Optional[str],
) -> Dict[str, Any]:
    """
    Évalue comparativement l'agent linéaire entraîné contre les profils heuristiques de référence.

    Paramètre `player_count` : nombre de joueurs par partie évaluée.
    Paramètre `games` : nombre de parties simulées pour l'évaluation.
    Paramètre `rounds_per_game` : nombre de manches par partie.
    Paramètre `seed` : graine de base de la campagne d'évaluation.
    Paramètre `trained_weights_path` : chemin des poids entraînés de l'agent linéaire, ou `None` si l'entraînement a échoué (l'évaluation
    utilise alors des poids nuls, comportement par défaut de `agents.rl_agent.RLAgent`).
    Retourne un dictionnaire portant le chemin du fichier CSV d'évaluation produit. Effet de bord : lance une campagne Ray via
    `research.evaluate_agent.launch_evaluation`.
    """
    from research.evaluate_agent import launch_evaluation

    seat_profiles = ["rl_agent"] + list(_BASELINE_PROFILES[: max(0, player_count - 1)])
    seat_profiles = seat_profiles[:player_count]
    while len(seat_profiles) < player_count:
        seat_profiles.append("rule_based_bot")

    seat_weights = {0: trained_weights_path} if trained_weights_path else None

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

    Retourne un dictionnaire vide (les figures sont écrites directement sur disque). Effet de bord : voir
    `research.generate_graphs.generate_all`.
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
        "## Entraînement distribué de l'agent à politique neuronale",
        f"- Exécuté : {distributed_result.get('executed', 'indisponible')}",
        f"- Détail : {distributed_result.get('reason', 'exécution effective ou statut non déterminé')}",
        "",
        "## Simulations de référence",
    ]
    for profile, info in baselines_result.items():
        lines.append(f"- Profil `{profile}` : fichier Parquet `{info.get('output_parquet', 'indisponible')}`")

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


def run_pipeline(
    player_count: int = 4,
    training_rounds: int = 2000,
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
    Paramètre `training_rounds` : nombre de manches d'entraînement de l'agent linéaire.
    Paramètre `distributed_steps` : nombre d'étapes de gradient tentées pour l'entraînement distribué neuronal.
    Paramètre `baseline_games_per_profile` : nombre de parties simulées par profil heuristique de référence.
    Paramètre `baseline_rounds_per_game` : nombre de manches par partie pour les simulations de référence.
    Paramètre `evaluation_games` : nombre de parties simulées pour l'évaluation comparative finale.
    Paramètre `evaluation_rounds_per_game` : nombre de manches par partie pour l'évaluation comparative finale.
    Paramètre `seed` : graine de base, dérivée distinctement pour chaque étape.
    Paramètre `redis_host`, `redis_port` : coordonnées Redis testées pour l'entraînement distribué.
    Retourne `None`. Effet de bord : exécute séquentiellement toutes les étapes du pipeline, reprend automatiquement à l'étape non achevée
    la plus ancienne en cas de relance après interruption, et produit in fine `data/final_report.md` et `figures/`.
    """
    state = _load_state()

    _run_step(
        state, "train_linear_agent",
        lambda: _step_train_linear_agent(player_count, training_rounds, seed),
    )

    trained_weights_path = state.get("train_linear_agent", {}).get("result", {}).get("weights_path")

    _run_step(
        state, "attempt_distributed_training",
        lambda: _step_attempt_distributed_training(player_count, distributed_steps, redis_host, redis_port),
    )

    _run_step(
        state, "simulate_baselines",
        lambda: _step_simulate_baselines(player_count, baseline_games_per_profile, baseline_rounds_per_game, seed + 10_000),
    )

    _run_step(
        state, "evaluate_trained_agent",
        lambda: _step_evaluate_trained_agent(
            player_count, evaluation_games, evaluation_rounds_per_game, seed + 20_000, trained_weights_path,
        ),
    )

    _run_step(state, "generate_graphs", _step_generate_graphs)

    _run_step(state, "generate_final_report", lambda: _step_generate_final_report(state))

    print("[pipeline] pipeline terminé. Voir data/final_report.md et figures/.")


def main() -> None:
    """
    Point d'entrée en ligne de commande du pipeline automatique complet.

    Retourne `None`. Effet de bord : lit les arguments de la ligne de commande et invoque `run_pipeline`. L'option `--reset` supprime l'état
    de progression enregistré, forçant une reprise depuis le début.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline automatique complet : entraînement, évaluation, graphiques et rapport final."
    )
    parser.add_argument("--player-count", type=int, default=4)
    parser.add_argument("--training-rounds", type=int, default=2000)
    parser.add_argument("--distributed-steps", type=int, default=200)
    parser.add_argument("--baseline-games-per-profile", type=int, default=100)
    parser.add_argument("--baseline-rounds-per-game", type=int, default=10)
    parser.add_argument("--evaluation-games", type=int, default=200)
    parser.add_argument("--evaluation-rounds-per-game", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--redis-host", type=str, default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument(
        "--reset", action="store_true",
        help="Supprime l'état de progression enregistré (data/pipeline_state.json) et relance le pipeline depuis le début.",
    )
    args = parser.parse_args()

    if args.reset and os.path.exists(_STATE_PATH):
        os.remove(_STATE_PATH)
        print("[pipeline] état de progression réinitialisé.")

    run_pipeline(
        player_count=args.player_count,
        training_rounds=args.training_rounds,
        distributed_steps=args.distributed_steps,
        baseline_games_per_profile=args.baseline_games_per_profile,
        baseline_rounds_per_game=args.baseline_rounds_per_game,
        evaluation_games=args.evaluation_games,
        evaluation_rounds_per_game=args.evaluation_rounds_per_game,
        seed=args.seed,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
    )


if __name__ == "__main__":
    main()
