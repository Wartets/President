"""
Module d'entraînement d'un agent à politique linéaire par REINFORCE simplifié.

Le module implémente `train`, qui exécute un nombre donné de manches contre un pool d'adversaires fixes en ajustant, manche après
manche, les poids linéaires d'un `agents.rl_agent.RLAgent` par une règle de gradient de politique : le vecteur de caractéristiques moyen
des décisions `ACTION_PLAY` de l'agent entraîné sur une manche est pondéré par l'avantage observé (point de victoire de la manche moins une
référence glissante), puis ajouté aux poids courants à un taux d'apprentissage donné. L'exploration (`epsilon`) décroît géométriquement au
fil des manches jusqu'à un plancher.

Le module dépend de `agents.rl_agent`, `agents.greedy_bot`, `agents.rule_based_bot`, `core.config`, `engine.event_bus` et `engine.round`.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
from rich.console import Console
from rich.table import Table

from agents.greedy_bot import GreedyBot
from agents.interface import AbstractBaseAgent
from agents.rl_agent import RLAgent, _option_features
from agents.rule_based_bot import RuleBasedBot
from core.config import GameConfig
from engine.event_bus import EventBus
from engine.round import run_round

# Décroissance géométrique de l'exploration appliquée après chaque manche.
_EPSILON_DECAY = 0.995

# Plancher d'exploration en deçà duquel `epsilon` ne descend plus.
_EPSILON_MIN = 0.02

# Valeur initiale d'exploration pour un modèle démarré de zéro.
_EPSILON_START = 0.3

# Facteur de lissage de la référence glissante de retour (moyenne mobile exponentielle).
_BASELINE_MOMENTUM = 0.98

# Nombre de manches entre deux affichages de progression lorsque le tableau de bord interne est actif.
_REPORT_INTERVAL = 100


class _ReturnBaseline:
    """
    Référence glissante de retour utilisée comme ligne de base de l'avantage REINFORCE.

    Champ `momentum` : facteur de lissage exponentiel, domaine $[0, 1[$.
    Champ `value` : valeur courante de la référence.
    """

    def __init__(self, momentum: float = _BASELINE_MOMENTUM) -> None:
        self.momentum = momentum
        self.value = 0.0
        self._initialized = False

    def update(self, observed: float) -> None:
        if not self._initialized:
            self.value = observed
            self._initialized = True
        else:
            self.value = self.momentum * self.value + (1.0 - self.momentum) * observed

    def baseline(self) -> float:
        return self.value


def _opponent_classes(opponent_pool: str, player_count: int) -> List[Type[AbstractBaseAgent]]:
    """
    Résout les classes d'adversaires appliquées aux sièges autres que celui de l'agent entraîné.

    Paramètre `opponent_pool` : nom du pool, valeurs `'greedy'`, `'rule_based'` ou `'mixed'`.
    Paramètre `player_count` : nombre de joueurs $N$ de la configuration d'entraînement.
    Retourne une liste de classes d'agents de taille `player_count - 1`. Aucun effet de bord.
    """
    if opponent_pool == "greedy":
        return [GreedyBot] * (player_count - 1)
    if opponent_pool == "rule_based":
        return [RuleBasedBot] * (player_count - 1)
    mixed = [GreedyBot, RuleBasedBot] * player_count
    return mixed[: player_count - 1]


def train(
    config: GameConfig,
    rounds: int,
    learning_rate: float = 0.01,
    opponent_pool: str = "mixed",
    initial_weights: Optional[np.ndarray] = None,
    initial_epsilon: float = _EPSILON_START,
    stop_check: Optional[Callable[[], bool]] = None,
    on_round: Optional[Callable[[int], None]] = None,
    use_internal_progress: bool = True,
    on_log: Optional[Callable[[str], None]] = None,
) -> Tuple[RLAgent, List[float]]:
    """
    Entraîne un agent à politique linéaire contre un pool d'adversaires fixes.

    Paramètre `config` : configuration de la partie utilisée pour toutes les manches d'entraînement.
    Paramètre `rounds` : nombre de manches à exécuter lors de cet appel.
    Paramètre `learning_rate` : taux d'apprentissage appliqué à la mise à jour des poids.
    Paramètre `opponent_pool` : profil des adversaires simulés, valeurs `'greedy'`, `'rule_based'` ou `'mixed'`.
    Paramètre `initial_weights` : poids de départ, nouveau vecteur nul si `None`.
    Paramètre `initial_epsilon` : exploration de départ, pertinente pour reprendre un entraînement déjà entamé.
    Paramètre `stop_check` : fonction sans argument consultée avant chaque manche ; si elle retourne vrai, la boucle s'arrête proprement
    avant la manche suivante.
    Paramètre `on_round` : fonction optionnelle invoquée après chaque manche exécutée, recevant l'index de la manche.
    Paramètre `use_internal_progress` : si vrai, affiche une barre `tqdm` et des tableaux `rich` périodiques sur la sortie standard ; si
    faux, aucun affichage interne n'est produit (utile lorsque l'appelant gère sa propre progression).
    Paramètre `on_log` : fonction optionnelle recevant un message texte périodique, appelée à la place de l'affichage `rich` interne
    lorsque fournie.
    Retourne un tuple `(trainee, running_vp)` : `trainee` est l'agent entraîné (poids mis à jour en place), `running_vp` la liste des
    points de victoire obtenus manche par manche lors de cet appel, dans l'ordre d'exécution. Effet de bord : exécute des manches
    complètes du moteur événementiel avec un `EventBus` jetable par manche, sans publier sur un bus partagé.
    """
    from tqdm import tqdm

    trainee = RLAgent(player_id=0, config=config, weights=initial_weights, epsilon=initial_epsilon)
    opponent_classes = _opponent_classes(opponent_pool, config.player_count)
    baseline = _ReturnBaseline()
    running_vp: List[float] = []

    console = Console() if use_internal_progress else None
    round_indices: Any = range(rounds)
    if use_internal_progress:
        round_indices = tqdm(round_indices, desc="Entraînement RL", unit="manche")

    roles: Optional[Dict[int, str]] = None
    for round_index in round_indices:
        if stop_check is not None and stop_check():
            break

        opponents: Dict[int, AbstractBaseAgent] = {
            pid + 1: opponent_classes[pid](pid + 1, config)
            for pid in range(config.player_count - 1)
        }
        agents: Dict[int, AbstractBaseAgent] = dict(opponents)
        agents[0] = trainee

        transitions: List[np.ndarray] = []
        original_choose = trainee.choose_action

        def _instrumented_choose(game_state):
            hand = game_state.hands[trainee.player_id]
            options = trainee._legal_options(hand, game_state)
            action = original_choose(game_state)
            if options and action.cards:
                transitions.append(
                    _option_features(action.cards, action.declared_power, hand.size(), game_state.e_rev)
                )
            return action

        trainee.choose_action = _instrumented_choose  # type: ignore[assignment]
        try:
            roles, vp_by_player, _finish_order = run_round(
                config, agents, EventBus(), round_index, roles, f"train-{round_index}",
            )
        finally:
            trainee.choose_action = original_choose  # type: ignore[assignment]

        vp = float(vp_by_player.get(0, 0.0))
        running_vp.append(vp)

        if transitions:
            advantage = vp - baseline.baseline()
            mean_features = np.mean(np.stack(transitions), axis=0)
            trainee.weights = trainee.weights + learning_rate * advantage * mean_features
        baseline.update(vp)

        trainee.epsilon = max(_EPSILON_MIN, trainee.epsilon * _EPSILON_DECAY)

        if on_round is not None:
            on_round(round_index)

        if (round_index + 1) % _REPORT_INTERVAL == 0:
            recent = running_vp[-_REPORT_INTERVAL:]
            mean_recent = sum(recent) / len(recent)
            if on_log is not None:
                on_log(
                    f"Manche {round_index + 1}/{rounds} — VP moyen ({len(recent)} dern.) = {mean_recent:.3f}, "
                    f"epsilon = {trainee.epsilon:.3f}"
                )
            elif use_internal_progress and console is not None:
                table = Table(title=f"Entraînement RL, manche {round_index + 1}")
                table.add_column("Métrique")
                table.add_column("Valeur")
                table.add_row("VP moyen glissant", f"{mean_recent:.3f}")
                table.add_row("Epsilon courant", f"{trainee.epsilon:.3f}")
                table.add_row("Poids", str(np.round(trainee.weights, 4).tolist()))
                console.print(table)

    return trainee, running_vp


def main() -> None:
    """
    Point d'entrée en ligne de commande de l'entraînement linéaire mono-processus.

    Retourne `None`. Effet de bord : lit les arguments de la ligne de commande, exécute `train`, puis sauvegarde les poids finaux au
    format `numpy` dans le fichier désigné par `--output`.
    """
    parser = argparse.ArgumentParser(description="Entraînement d'un agent à politique linéaire (REINFORCE simplifié)")
    parser.add_argument("--rounds", type=int, default=5000)
    parser.add_argument("--player-count", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--opponent-pool", choices=["greedy", "rule_based", "mixed"], default="mixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="rl_weights.npy")
    parser.add_argument("--initial-weights", type=str, default=None)
    parser.add_argument("--initial-epsilon", type=float, default=_EPSILON_START)
    args = parser.parse_args()

    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from checkpoint_utils import GracefulKiller

    killer = GracefulKiller()
    config = GameConfig(random_seed=args.seed, player_count=args.player_count)
    initial_weights = np.load(args.initial_weights) if args.initial_weights else None

    trainee, running_vp = train(
        config,
        args.rounds,
        learning_rate=args.learning_rate,
        opponent_pool=args.opponent_pool,
        initial_weights=initial_weights,
        initial_epsilon=args.initial_epsilon,
        stop_check=lambda: killer.should_stop,
    )

    np.save(args.output, trainee.weights)
    tail = running_vp[-max(1, len(running_vp) // 10):] if running_vp else []
    mean_tail = sum(tail) / len(tail) if tail else 0.0
    print(f"Poids finaux sauvegardés dans {args.output} ({len(running_vp)} manche(s) exécutée(s), VP moyen récent = {mean_tail:.3f}).")


if __name__ == "__main__":
    main()
