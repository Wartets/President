"""
Module de lancement conjoint de l'entraînement distribué.

Le module implémente `main`, le point d'entrée en ligne de commande démarrant simultanément plusieurs acteurs
`training.rollout_worker.RolloutWorker` sur un cluster `ray` local et un unique `training.trainer.Trainer` consommant le tampon Redis
partagé, matérialisant la séparation Rollout Worker / Trainer.

Le module dépend de `ray`, `core.config`, `training.rollout_worker` et `training.trainer`.
"""

from __future__ import annotations

import argparse
import os
import threading
from typing import Any, Optional, cast

import ray

from core.config import GameConfig
from training.replay_buffer import RedisReplayBuffer
from training.rollout_worker import RolloutWorker
from training.trainer import Trainer


def launch(
    num_workers: int,
    rounds_per_worker_batch: int,
    opponent_pool: str,
    player_count: int,
    redis_host: str,
    redis_port: int,
    batch_size: int,
    total_steps: int,
    resume_weights: Optional[str] = None,
    model_name: str = "torch_rl_weights",
) -> None:
    """
    Démarre les Rollout Workers et le Trainer, puis attend l'achèvement de l'entraînement.

    Paramètre `num_workers` : nombre d'acteurs Ray `RolloutWorker` lancés en parallèle.
    Paramètre `rounds_per_worker_batch` : nombre de manches exécutées par chaque acteur avant de rendre la main au superviseur.
    Paramètre `opponent_pool` : nature des adversaires simulés par les Rollout Workers.
    Paramètre `player_count` : nombre de joueurs $N$ par manche simulée.
    Paramètre `redis_host`, `redis_port` : coordonnées de l'instance Redis partagée entre workers et Trainer.
    Paramètre `batch_size` : taille de lot utilisée par le Trainer à chaque étape de gradient.
    Paramètre `total_steps` : nombre total d'étapes d'apprentissage exécutées par le Trainer.
    Retourne `None`. Effet de bord : initialise un cluster Ray local, démarre le Trainer dans un thread dédié, boucle indéfiniment sur les
    acteurs de rollout jusqu'à l'achèvement du Trainer, puis arrête le cluster Ray.
    """
    probe = RedisReplayBuffer(host=redis_host, port=redis_port)
    if not probe.ping():
        raise RuntimeError(
            f"Impossible de joindre une instance Redis à l'adresse {redis_host}:{redis_port}. "
            "Démarrer un serveur Redis local et vérifier les paramètres --redis-host/--redis-port avant de relancer "
            "l'entraînement distribué."
        )

    ray.init(num_cpus=num_workers, ignore_reinit_error=True, log_to_driver=False)
    config = GameConfig(player_count=player_count)

    trainer = Trainer(
        redis_host=redis_host,
        redis_port=redis_port,
        resume_weights=resume_weights,
        player_count=player_count,
        model_name=model_name,
    )
    trainer_thread = threading.Thread(target=trainer.run, args=(batch_size, total_steps), daemon=True)
    trainer_thread.start()

    # Ray actor creation — cast to Any to satisfy static type checker about .remote
    workers = [cast(Any, RolloutWorker).remote(config, redis_host, redis_port) for _ in range(num_workers)]
    while trainer_thread.is_alive():
        futures = [
            worker.run_rounds.remote(opponent_pool, rounds_per_worker_batch)
            for worker in workers
        ]
        ray.get(futures)

    ray.shutdown()


def main() -> None:
    """
    Point d'entrée en ligne de commande du lancement distribué.

    Retourne `None`. Effet de bord : lit les arguments de la ligne de commande et invoque `launch`.
    """
    parser = argparse.ArgumentParser(description="Lancement conjoint Rollout Workers / Trainer")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--rounds-per-batch", type=int, default=50)
    parser.add_argument("--opponent-pool", choices=["greedy", "rule_based", "mixed"], default="mixed")
    parser.add_argument("--player-count", type=int, default=4)
    parser.add_argument("--redis-host", type=str, default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--total-steps", type=int, default=10000)
    parser.add_argument("--resume-weights", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="torch_rl_weights")
    args = parser.parse_args()

    launch(
        args.workers, args.rounds_per_batch, args.opponent_pool, args.player_count,
        args.redis_host, args.redis_port, args.batch_size, args.total_steps,
        resume_weights=args.resume_weights, model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
