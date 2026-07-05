"""
Module du processus d'apprentissage (Trainer).

Le module implémente `Trainer`, le processus consommant les transitions accumulées par les acteurs `training.rollout_worker.RolloutWorker`
via un `training.replay_buffer.RedisReplayBuffer` partagé, et appliquant des mises à jour de gradient de politique REINFORCE sur un
`agents.torch_rl_agent.PolicyNet`, sous précision mixte lorsqu'un GPU est disponible. Le Trainer republie périodiquement les poids mis à
jour vers Redis afin que les Rollout Workers puissent les récupérer.

Le module dépend de `torch`, `training.replay_buffer`, `agents.torch_rl_agent` et `numpy`.
"""

from __future__ import annotations

import argparse
import io
import time
from typing import Optional

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from agents.torch_rl_agent import PolicyNet, resolve_device
from training.replay_buffer import RedisReplayBuffer

# Clé Redis portant les derniers poids publiés, cohérente avec `training.rollout_worker._WEIGHTS_KEY`.
_WEIGHTS_KEY = "president:policy_weights"

# Nombre d'étapes d'apprentissage entre deux publications de poids sur Redis.
_PUBLISH_INTERVAL = 50


class Trainer:
    """
    Processus consommateur appliquant des mises à jour de gradient de politique à partir du tampon de rejeu distribué.

    Champ `buffer` : instance de `RedisReplayBuffer` partagée avec les Rollout Workers.
    Champ `policy` : réseau de politique entraîné, transféré sur `device`.
    Champ `device` : périphérique `torch` utilisé pour l'entraînement.
    Champ `optimizer` : optimiseur `torch.optim.Adam` appliqué aux paramètres de `policy`.
    Champ `scaler` : instance de `torch.cuda.amp.GradScaler`, active uniquement lorsque `device` est de type `'cuda'`.
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
    ) -> None:
        self.buffer = RedisReplayBuffer(host=redis_host, port=redis_port)
        self.device = resolve_device(device)
        self.policy = PolicyNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _publish_weights(self) -> None:
        """
        Publie les poids courants de la politique sur Redis.

        Retourne `None`. Effet de bord : sérialise `self.policy.state_dict()` au format binaire `torch.save` et remplace la valeur de la
        clé `_WEIGHTS_KEY` dans Redis.
        """
        buffer = io.BytesIO()
        torch.save(self.policy.state_dict(), buffer)
        self.buffer.client.set(_WEIGHTS_KEY, buffer.getvalue())

    def train_step(self, batch_size: int) -> Optional[float]:
        """
        Exécute une unique étape de gradient de politique sur un lot échantillonné du tampon.

        Paramètre `batch_size` : taille du lot à échantillonner, entier strictement positif.
        Retourne la perte scalaire observée pour cette étape, ou `None` si le tampon ne contient pas encore `batch_size` transitions.
        Effet de bord : met à jour les paramètres de `self.policy` via `self.optimizer`, sous précision mixte lorsque `self.use_amp` est
        vrai. L'objectif maximisé est $\\text{score}(s) \\times (\\text{retour}(s) - \\bar{\\text{retour}})$, la perte minimisée en étant
        l'opposé moyenné sur le lot.
        """
        batch = self.buffer.sample(batch_size)
        if batch is None:
            return None

        features = np.stack([np.array(record["features"], dtype=np.float32) for record in batch])
        returns = np.array([record["return_value"] for record in batch], dtype=np.float32)
        baseline = float(returns.mean())
        advantages = returns - baseline

        features_tensor = torch.as_tensor(features, device=self.device)
        advantages_tensor = torch.as_tensor(advantages, device=self.device)

        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=torch.float16):
            scores = self.policy(features_tensor)
            loss = -(scores * advantages_tensor).mean()

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return float(loss.item())

    def run(self, batch_size: int, total_steps: int) -> None:
        """
        Exécute la boucle d'apprentissage complète.

        Paramètre `batch_size` : taille de lot utilisée à chaque étape de gradient.
        Paramètre `total_steps` : nombre total d'étapes d'apprentissage à exécuter.
        Retourne `None`. Effet de bord : attend que le tampon soit suffisamment peuplé, exécute `total_steps` appels à `train_step`,
        republie les poids toutes les `_PUBLISH_INTERVAL` étapes, et affiche un tableau de bord `rich` de progression.
        """
        console = Console()
        while self.buffer.size() < batch_size:
            time.sleep(1.0)

        for step in range(total_steps):
            loss = self.train_step(batch_size)
            if (step + 1) % _PUBLISH_INTERVAL == 0:
                self._publish_weights()
                table = Table(title=f"Entraînement distribué, étape {step + 1}/{total_steps}")
                table.add_column("Métrique")
                table.add_column("Valeur")
                table.add_row("Perte courante", f"{loss:.5f}" if loss is not None else "indisponible")
                table.add_row("Taille du tampon", str(self.buffer.size()))
                table.add_row("Périphérique", str(self.device))
                console.print(table)

        self._publish_weights()


def main() -> None:
    """
    Point d'entrée en ligne de commande du processus Trainer.

    Retourne `None`. Effet de bord : lit les arguments de la ligne de commande et exécute `Trainer.run`.
    """
    parser = argparse.ArgumentParser(description="Processus Trainer distribué")
    parser.add_argument("--redis-host", type=str, default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--total-steps", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    trainer = Trainer(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    trainer.run(args.batch_size, args.total_steps)


if __name__ == "__main__":
    main()
