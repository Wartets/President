"""
Module de la boucle d'entraînement par politique linéaire (REINFORCE).

Le module implémente l'entraînement de `agents.rl_agent.RLAgent` par un algorithme REINFORCE simplifié : les poids de la politique
linéaire sont ajustés proportionnellement au gradient du score des options choisies, pondéré par le retour cumulé (point de victoire de la
manche). L'entraînement utilise le moteur événementiel complet ; le moteur vectorisé possède une boucle dédiée.

Le module dépend de `core.config`, `agents.rl_agent`, `agents.greedy_bot`, `agents.rule_based_bot`, `engine.game_runner`, `engine.event_bus`
et de `numpy`. Dépendance externe : `tqdm` pour le suivi de progression, `rich` pour le tableau de bord console.
"""

from __future__ import annotations

import argparse
import copy
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

import naming
from agents.greedy_bot import GreedyBot
from agents.rl_agent import FEATURE_DIM, RLAgent, _option_features
from agents.rule_based_bot import RuleBasedBot
from agents.interface import AbstractBaseAgent
from core.config import GameConfig
from engine.event_bus import EventBus
from engine.game_runner import Game

# Taux d'apprentissage du gradient de politique.
_DEFAULT_LEARNING_RATE = 0.01

# Coefficient de décroissance de l'exploration `epsilon` par bloc d'entraînement.
_EPSILON_DECAY = 0.995
_EPSILON_MIN = 0.02


class ReturnTracker:
    """
    Collecteur des transitions `(features, score, retour)` d'un agent entraîné sur une manche.

    Champ `records` : liste de tuples `(feature_vector, chosen_score, return_value)` accumulés sur la manche courante.
    """

    def __init__(self) -> None:
        self.records: List[np.ndarray] = []

    def reset(self) -> None:
        """
        Vide le tampon de transitions accumulées.

        Retourne `None`. Effet de bord : réinitialise `records`.
        """
        self.records.clear()


def _run_training_round(
    config: GameConfig,
    trainee: RLAgent,
    opponents: Dict[int, AbstractBaseAgent],
    tracker: List[Tuple[np.ndarray, float, float]],
    round_index: int,
    previous_roles,
    game_id: str,
) -> Tuple[float, Optional[Dict[int, str]]]:
    """
    Exécute une unique manche d'entraînement et collecte les caractéristiques des décisions du joueur entraîné.

    Paramètre `config` : configuration de la partie.
    Paramètre `trainee` : instance de `RLAgent` dont la politique est entraînée.
    Paramètre `opponents` : association entre identifiant de joueur et agent adverse fixe pour cette manche.
    Paramètre `tracker` : liste mutée en place, recevant un tuple `(feature_vector, chosen_score)` par décision `ACTION_PLAY` du joueur entraîné.
    Paramètre `round_index` : index de la manche à exécuter.
    Paramètre `previous_roles` : rôles issus de la manche précédente, ou `None`.
    Paramètre `game_id` : identifiant de la partie hôte de la manche.
    Retourne le point de victoire obtenu par `trainee` sur cette manche. Effet de bord : peuple `tracker`, publie les événements de la manche sur
    un `EventBus` local jetable.
    """
    from engine.round import run_round

    agents = dict(opponents)
    agents[trainee.player_id] = trainee

    original_choose = trainee.choose_action

    def _instrumented_choose(game_state):
        hand = game_state.hands[trainee.player_id]
        options = trainee._legal_options(hand, game_state)
        action = original_choose(game_state)
        if options and action.cards:
            hand_size = hand.size()
            features = _option_features(action.cards, action.declared_power, hand_size, game_state.e_rev)
            scores = np.stack([
                _option_features(c, d, hand_size, game_state.e_rev) for c, d in options
            ]) @ trainee.weights
            chosen_score = float(features @ trainee.weights)
            tracker.append((features, chosen_score, max(scores) if len(scores) else chosen_score))
        return action

    trainee.choose_action = _instrumented_choose  # type: ignore[assignment]
    try:
        roles, vp_by_player, _finish_order = run_round(
            config, agents, EventBus(), round_index, previous_roles, game_id
        )
    finally:
        trainee.choose_action = original_choose  # type: ignore[assignment]

    return vp_by_player.get(trainee.player_id, 0.0), roles


def train(
    config: GameConfig,
    total_rounds: int,
    learning_rate: float = _DEFAULT_LEARNING_RATE,
    opponent_pool: str = "mixed",
    initial_weights: Optional[np.ndarray] = None,
    initial_epsilon: float = 0.3,
    stop_check: Optional[Callable[[], bool]] = None,
    on_round: Optional[Callable[[int], None]] = None,
) -> Tuple[RLAgent, List[float]]:
    """
    Entraîne un `RLAgent` par ajustement de gradient de politique sur un nombre fixé de manches.

    Paramètre `config` : configuration de la partie utilisée pour toutes les manches d'entraînement.
    Paramètre `total_rounds` : nombre total de manches d'entraînement exécutées lors de cet appel, entier strictement positif.
    Paramètre `learning_rate` : taux d'apprentissage du gradient de politique, nombre strictement positif.
    Paramètre `opponent_pool` : nature des adversaires simulés, chaîne parmi `'greedy'`, `'rule_based'`, `'mixed'`.
    Paramètre `initial_weights` : poids de politique de départ, utilisés pour reprendre un entraînement antérieur ; poids nuls si `None`.
    Paramètre `initial_epsilon` : taux d'exploration de départ.
    Paramètre `stop_check` : fonction sans argument consultée avant chaque manche ; si elle retourne vrai, l'entraînement s'arrête
    proprement avant la manche suivante et retourne l'état accumulé jusque-là, permettant une reprise ultérieure sans perte de travail.
    Paramètre `on_round` : fonction optionnelle invoquée après chaque manche avec l'index de la manche, utile pour piloter un affichage
    de progression externe sans dupliquer la barre `tqdm` interne.
    Retourne un tuple `(trainee, running_vp)` : l'instance de `RLAgent` entraînée portant les poids ajustés, et la liste des points de
    victoire obtenus manche après manche (de taille potentiellement inférieure à `total_rounds` si `stop_check` a déclenché un arrêt
    anticipé). Effet de bord : affiche un tableau de bord `rich` de progression et met à jour `trainee.weights`/`trainee.epsilon`
    à chaque manche.
    """
    console = Console()
    trainee = RLAgent(player_id=0, config=config, weights=initial_weights, epsilon=initial_epsilon)

    opponent_classes = {
        "greedy": [GreedyBot] * (config.player_count - 1),
        "rule_based": [RuleBasedBot] * (config.player_count - 1),
        "mixed": [GreedyBot, RuleBasedBot] * config.player_count,
    }[opponent_pool][: config.player_count - 1]

    running_vp: List[float] = []
    roles = None

    progress_bar = tqdm(range(total_rounds), desc="Entraînement RLAgent", unit="manche")
    for round_index in progress_bar:
        if stop_check is not None and stop_check():
            progress_bar.close()
            break
        opponents = {
            pid + 1: opponent_classes[pid](pid + 1, config)
            for pid in range(config.player_count - 1)
        }
        tracker: List[Tuple[np.ndarray, float, float]] = []
        vp, roles = _run_training_round(
            config, trainee, opponents, tracker, round_index, roles, "training-game"
        )
        running_vp.append(vp)
        if on_round is not None:
            on_round(round_index)

        if tracker:
            baseline = float(np.mean([r[2] for r in tracker]))
            gradient = np.zeros(FEATURE_DIM, dtype=np.float64)
            for features, chosen_score, _ in tracker:
                advantage = vp - baseline
                gradient += advantage * features
            gradient /= len(tracker)
            trainee.weights = trainee.weights + learning_rate * gradient

        trainee.epsilon = max(_EPSILON_MIN, trainee.epsilon * _EPSILON_DECAY)

        if (round_index + 1) % 100 == 0:
            table = Table(title=f"Progression, manche {round_index + 1}/{total_rounds}")
            table.add_column("Métrique")
            table.add_column("Valeur")
            table.add_row("VP moyen (100 dernières manches)", f"{np.mean(running_vp[-100:]):.3f}")
            table.add_row("Epsilon courant", f"{trainee.epsilon:.4f}")
            table.add_row("Poids de politique", np.array2string(trainee.weights, precision=4))
            console.print(table)

    return trainee, running_vp


def main() -> None:
    """
    Point d'entrée en ligne de commande de l'entraînement du `RLAgent`.

    Retourne `None`. Effet de bord : lit les arguments de la ligne de commande, exécute `train` (en reprenant un entraînement antérieur
    si `--resume` est fourni), puis sauvegarde les poids finaux et leur historique d'entraînement, nommés selon la convention du projet,
    dans le répertoire `weights/`.
    """
    parser = argparse.ArgumentParser(description="Entraînement REINFORCE de agents.rl_agent.RLAgent")
    parser.add_argument("--rounds", type=int, default=5000)
    parser.add_argument("--player-count", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=_DEFAULT_LEARNING_RATE)
    parser.add_argument("--opponent-pool", choices=["greedy", "rule_based", "mixed"], default="mixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-name", type=str, default="rl_weights")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = GameConfig(random_seed=args.seed, player_count=args.player_count)

    initial_weights: Optional[np.ndarray] = None
    rounds_already_trained = 0
    if args.resume is not None:
        initial_weights = np.load(args.resume)
        metadata = naming.read_weights_metadata(args.resume)
        if metadata is not None:
            rounds_already_trained = int(metadata.get("rounds_trained", 0))
        print(f"Reprise de l'entraînement depuis {args.resume} ({rounds_already_trained} manches déjà jouées).")

    from checkpoint_utils import GracefulKiller

    killer = GracefulKiller()

    trainee, running_vp = train(
        config, args.rounds, args.learning_rate, args.opponent_pool,
        initial_weights=initial_weights,
        stop_check=lambda: killer.should_stop,
    )

    total_rounds_trained = rounds_already_trained + len(running_vp)
    if len(running_vp) < args.rounds:
        print(f"Entraînement interrompu proprement après {len(running_vp)}/{args.rounds} manches ; relancer avec --resume pour continuer.")

    output_path = args.output or naming.build_weights_filename(
        model_name=args.model_name,
        player_count=args.player_count,
        learning_rate=args.learning_rate,
        rounds=total_rounds_trained,
    )
    np.save(output_path, trainee.weights)
    naming.write_weights_metadata(
        output_path,
        {
            "model_name": args.model_name,
            "player_count": args.player_count,
            "learning_rate": args.learning_rate,
            "rounds_trained": total_rounds_trained,
            "opponent_pool": args.opponent_pool,
            "seed": args.seed,
            "resumed_from": args.resume,
        },
    )

    history_path = naming.build_weights_metadata_filename(output_path).replace(".meta.json", ".history.csv")
    with open(history_path, "w", encoding="utf-8") as handle:
        handle.write("round_index,vp\n")
        for index, vp in enumerate(running_vp):
            handle.write(f"{rounds_already_trained + index},{vp}\n")

    print(f"Poids sauvegardés dans {output_path}")
    print(f"Historique d'entraînement sauvegardé dans {history_path}")


if __name__ == "__main__":
    main()
