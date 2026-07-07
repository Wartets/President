"""
Module du pipeline automatique complet de recherche.

Le module orchestre, sans intervention humaine, une campagne de recherche incrémentale et adaptative couvrant l'intégralité de l'espace des
paramètres de `core.config.GameConfig` : entraînement continu (jamais recommencé de zéro tant que des poids existent déjà) de l'agent
linéaire et de l'agent neuronal distribué, balayage étendu d'hyperparamètres (taux d'apprentissage croisé avec le pool d'adversaires) avec
sélection automatique de la meilleure combinaison observée par nombre de joueurs, simulations de référence pour l'intégralité des profils
heuristiques disponibles (à l'exception du profil de simulation par rollouts, jugé trop coûteux pour une campagne à grande échelle) croisés
avec l'intégralité des nombres de joueurs et une grille étendue de configurations de règles (chaque paramètre booléen basculé
individuellement, chaque valeur énumérée testée, plus un ensemble de combinaisons tirées aléatoirement pour capturer les interactions entre
règles), recherche combinatoire multi-configurations sur ce même produit cartésien, tournoi direct entre la politique linéaire et la
politique neuronale entraînées pour un même nombre de joueurs avec désignation d'un champion courant, vérification statistique automatisée
d'un ensemble de propriétés générales attendues du moteur de règles sur une partie fraîche et jetable, évaluations comparatives du modèle
linéaire contre les profils heuristiques, génération versionnée des graphiques, puis rédaction d'un rapport de synthèse relisant
l'intégralité des données accumulées sur tous les lancements précédents.

Contrairement à une exécution "tout ou rien", chaque lancement du pipeline ajoute du travail neuf (nouvelles parties, nouvelles manches
d'entraînement, nouvelles combinaisons de balayage, nouvelles itérations combinatoires, nouveaux tournois) par-dessus ce qui a déjà été
calculé lors des lancements précédents, sans jamais recalculer ni écraser une donnée déjà acquise : le fichier
`data/pipeline_manifest.json` conserve la couverture cumulée et sert de source de vérité entre deux lancements. Une interruption brutale
(Ctrl+C, SIGTERM) est prise en compte entre deux unités de travail : le manifeste est sauvegardé avant de quitter, et le prochain lancement
reprend exactement là où le précédent s'est arrêté plutôt que de recommencer les combinaisons déjà couvertes. Étant donné l'ampleur de la
grille de configurations couverte par défaut (produit cartésien de tous les profils heuristiques, tous les nombres de joueurs et une
plusieurs dizaines de variantes de règles), un unique lancement ne couvre généralement qu'une fraction de la grille complète ; la
priorisation par couverture cumulée croissante garantit que chaque combinaison finit par recevoir une couverture comparable au fil des
lancements successifs, sans intervention manuelle.

Seules les étapes ne gérant pas déjà leur propre affichage de progression (entraînement linéaire, balayage d'hyperparamètres, sélection des
meilleurs hyperparamètres) s'exécutent sous le tableau de bord partagé `ProgressManager` ; les étapes qui embarquent leur propre système de
rendu (`tqdm` et `analytics.live_monitor.LiveMonitor` dans `research.run_simulation`/`research.evaluate_agent`/`research.run_combinatory`,
tableaux `rich` périodiques dans `training.trainer.Trainer`) s'exécutent en dehors de ce tableau de bord, deux instances `rich.live.Live`
actives simultanément sur la même console produisant un affichage corrompu.

Le module dépend de `core.config`, `naming`, `console_theme`, `progress_manager`, `checkpoint_utils`, `agents.greedy_bot`,
`analytics.event_logger`, `analytics.metrics_calc`, `engine.event_bus`, `engine.game_runner`, `events.structural`, `training.train_rl`,
`training.trainer`, `training.launch_distributed`, `research.run_simulation`, `research.run_combinatory`, `research.evaluate_agent` et
`research.generate_graphs`.
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
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
from agents.greedy_bot import GreedyBot
from agents.interface import AbstractBaseAgent
from analytics.event_logger import EventLogger
from analytics.metrics_calc import (
    action_space_entropy, branching_factor_average, gini_initial_hand_power,
    role_transition_matrix, sub_optimal_pass_rate, trick_length_average,
)
from checkpoint_utils import GracefulKiller, atomic_write_json, load_json
from core.config import GameConfig
from core.math_utils import f_std
from engine.event_bus import EventBus
from engine.game_runner import Game
from events.structural import EventRoundStart
from progress_manager import ProgressManager

_MANIFEST_PATH = os.path.join("data", "pipeline_manifest.json")
_console = Console()

# Profils automatisés volontairement exclus de la recherche automatisée à grande échelle : le coût par décision de la simulation Monte-Carlo
# par rollouts rend son inclusion dans une grille combinatoire de plusieurs dizaines de configurations disproportionnellement lente. Le profil
# reste utilisable manuellement via `play_game.py`/`research.evaluate_agent`, il est seulement exclu de ce pipeline.
_EXCLUDED_AUTOMATED_PROFILES = ("mcts_bot",)

_PROGRESS_CHUNK_SIZE = 5

# Nombre de combinaisons aléatoires supplémentaires générées par `_build_extended_rule_presets` afin de capturer les interactions entre règles
# avancées qu'un balayage uniquement paramètre-par-paramètre ne peut pas révéler.
_RANDOM_GRID_COMBOS = 24

# Décroissance d'exploration utilisée par `training.train_rl.train`, répliquée ici pour estimer un epsilon de reprise cohérent avec le
# nombre de manches déjà entraînées sur un modèle repris plutôt que recréé.
_EPSILON_DECAY = 0.995
_EPSILON_MIN = 0.02
_EPSILON_START = 0.3

# Paramètres par défaut de la vérification statistique automatisée du moteur de règles.
_VALIDATION_PLAYER_COUNT = 5


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
        "schema_version": 4,
        "runs": [],
        "models": {},
        "baseline_coverage": {},
        "evaluation_coverage": {},
        "lr_sweep_coverage": {"combos_done": [], "output_csv": None},
        "best_hyperparameters": {},
        "combinatorial_coverage": {},
        "tournament_results": {},
        "validation_results": {},
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


def _lr_combo_key(player_count: int, learning_rate: float, opponent_pool: str, seed_offset: int) -> str:
    return f"p{player_count}|lr{learning_rate:g}|op{opponent_pool}|s{seed_offset}"


def _heuristic_profiles() -> List[str]:
    """
    Détermine dynamiquement l'ensemble des profils heuristiques disponibles pour les campagnes de référence.

    Retourne la liste des clés de `research.run_simulation._AGENT_REGISTRY`, incluant automatiquement tout nouveau profil d'agent
    heuristique enregistré dans `registry.agent_registry`, à l'exception des profils listés dans `_EXCLUDED_AUTOMATED_PROFILES`. Aucun
    effet de bord.
    """
    from research.run_simulation import _AGENT_REGISTRY as heuristic_registry

    return [name for name in heuristic_registry.keys() if name not in _EXCLUDED_AUTOMATED_PROFILES]


def _build_extended_rule_presets() -> Dict[str, Dict[str, Any]]:
    """
    Construit une grille étendue de présets de règles couvrant l'intégralité de l'espace des paramètres de `GameConfig`.

    Retourne un dictionnaire nommé de dictionnaires de surcharge de `GameConfig`, couvrant : chaque champ booléen basculé
    individuellement dans les deux sens, chaque mode de distribution de VP, chaque sémantique de passe, plusieurs rangs magiques et de
    saut de tour, les deux types de pénalité de sortie, chaque rôle ciblé par l'attribution stricte du reste, une configuration
    "tout activé" et une configuration "tout désactivé", ainsi qu'un ensemble de combinaisons aléatoires reproductibles couvrant les
    interactions entre paramètres. Les combinaisons structurellement invalides (ex : interception sans second paquet) ne sont pas
    filtrées ici ; elles sont ignorées avec un avertissement par `research.run_combinatory.run_grid` au moment de leur utilisation.
    Aucun effet de bord.
    """
    from core.config import (
        PASS_TYPE_ALLOW_SOFT, PASS_TYPE_HARD_ONLY, PENALTY_DRAW_CARDS,
        PENALTY_INSTANT_SCUM, ROLE_NEUTRAL, ROLE_PRESIDENT, ROLE_SCUM,
        ROLE_VICE_PRESIDENT, ROLE_VICE_SCUM, VP_DISTRIBUTION_LEGACY_STEPPED,
        VP_DISTRIBUTION_LINEAR, VP_DISTRIBUTION_SYMMETRICAL,
    )

    presets: Dict[str, Dict[str, Any]] = {}

    boolean_fields = [
        "skip_on_equal", "revolution_enabled", "double_revolution_enabled",
        "straights_enabled", "skip_turn_enabled", "interception_enabled",
        "putsch_enabled", "blind_tax_enabled", "strict_remainder_allocation",
        "finish_penalty_enabled", "finish_penalty_extended", "no_finish_on_joker",
        "no_finish_on_revolution", "use_jokers", "magic_two",
    ]
    for field in boolean_fields:
        presets[f"toggle_{field}_on"] = {field: True}
        presets[f"toggle_{field}_off"] = {field: False}

    for vp_mode in (VP_DISTRIBUTION_LEGACY_STEPPED, VP_DISTRIBUTION_LINEAR, VP_DISTRIBUTION_SYMMETRICAL):
        presets[f"vp_{vp_mode.lower()}"] = {"vp_distribution_type": vp_mode}

    for pass_mode in (PASS_TYPE_HARD_ONLY, PASS_TYPE_ALLOW_SOFT):
        presets[f"pass_{pass_mode.lower()}"] = {"pass_type": pass_mode}

    for rank in ("5", "8", "10", "K"):
        presets[f"magic_rank_{rank}"] = {"magic_card_enabled": True, "magic_card_rank": rank, "magic_two": False}

    for rank in ("4", "7", "8", "J"):
        presets[f"skip_turn_rank_{rank}"] = {"skip_turn_enabled": True, "skip_turn_rank": rank}

    presets["finish_penalty_draw"] = {
        "finish_penalty_enabled": True, "finish_penalty_type": PENALTY_DRAW_CARDS, "finish_penalty_draw_count": 2,
    }
    presets["finish_penalty_instant_full"] = {
        "finish_penalty_enabled": True, "finish_penalty_type": PENALTY_INSTANT_SCUM,
        "finish_penalty_extended": True, "no_finish_on_joker": True, "no_finish_on_revolution": True,
    }

    for role in (ROLE_PRESIDENT, ROLE_VICE_PRESIDENT, ROLE_NEUTRAL, ROLE_VICE_SCUM, ROLE_SCUM):
        presets[f"strict_remainder_{role.lower()}"] = {"strict_remainder_allocation": True, "strict_remainder_role": role}

    presets["everything_on"] = {field: True for field in boolean_fields}
    presets["everything_off"] = {
        field: False for field in boolean_fields if field not in ("use_jokers", "magic_two")
    }

    rng = random.Random("pipeline_random_grid_v1")
    for combo_index in range(_RANDOM_GRID_COMBOS):
        overrides: Dict[str, Any] = {}
        for field in boolean_fields:
            if rng.random() < 0.5:
                overrides[field] = bool(rng.random() < 0.5)
        overrides["vp_distribution_type"] = rng.choice(
            [VP_DISTRIBUTION_LEGACY_STEPPED, VP_DISTRIBUTION_LINEAR, VP_DISTRIBUTION_SYMMETRICAL]
        )
        overrides["pass_type"] = rng.choice([PASS_TYPE_HARD_ONLY, PASS_TYPE_ALLOW_SOFT])
        presets[f"random_combo_{combo_index}"] = overrides

    return presets


def _resolve_rule_presets(requested: str) -> List[str]:
    """
    Résout la liste effective de présets de règles à couvrir, en injectant systématiquement la grille étendue.

    Paramètre `requested` : valeur brute de `--rule-presets`, soit `'auto'` pour couvrir la grille étendue complète en plus des présets
    historiques (`base`, `straights`, `full`), soit une liste explicite de noms séparés par des virgules.
    Retourne la liste résolue de noms de présets. Effet de bord : étend `research.run_simulation._RULE_PRESETS` en place avec
    l'intégralité de la grille construite par `_build_extended_rule_presets`, quelle que soit la valeur de `requested`, afin que tout nom
    de préset étendu référencé explicitement par l'appelant reste résolvable.
    """
    from research.run_simulation import _RULE_PRESETS as existing_presets

    extended = _build_extended_rule_presets()
    existing_presets.update(extended)

    if requested.strip().lower() == "auto":
        return ["base", "straights", "full"] + sorted(extended.keys())
    return [token.strip() for token in requested.split(",") if token.strip()]


def _prioritize_baseline_combos(
    manifest: Dict[str, Any], combos: List[Tuple[int, str, str]],
) -> List[Tuple[int, str, str]]:
    """
    Trie les combinaisons de référence par couverture cumulée croissante.

    Paramètre `manifest` : manifeste cumulatif, consulté pour la couverture déjà accumulée par combinaison.
    Paramètre `combos` : liste de combinaisons `(player_count, profile, rule_preset)` à ordonner.
    Retourne la même liste triée par nombre de parties déjà simulées croissant, garantissant qu'une combinaison encore peu couverte
    (par exemple un profil ou un préset ajouté récemment à la grille) rattrape prioritairement son retard plutôt que de laisser la
    grille progresser uniquement dans l'ordre d'énumération d'origine. Aucun effet de bord.
    """
    def _coverage(combo: Tuple[int, str, str]) -> int:
        player_count, profile, preset = combo
        key = _baseline_key(player_count, profile, preset)
        return int(manifest["baseline_coverage"].get(key, {}).get("games_done", 0))

    return sorted(combos, key=_coverage)


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
    VP moyen récent, hyperparamètres utilisés). Effet de bord : écrit un nouveau fichier de poids et une entrée d'historique étendue,
    jamais un fichier de poids déjà existant. Si le balayage d'hyperparamètres a déjà désigné une meilleure combinaison
    (taux d'apprentissage, pool d'adversaires) pour ce nombre de joueurs, celle-ci est utilisée à la place des valeurs par défaut.
    """
    from training.train_rl import train

    key = _model_key("pipeline_rl_weights", player_count)
    existing = manifest["models"].get(key)

    best_hyperparams = manifest.get("best_hyperparameters", {}).get(str(player_count), {})
    learning_rate = float(best_hyperparams.get("learning_rate", 0.01))
    opponent_pool = str(best_hyperparams.get("opponent_pool", "mixed"))

    initial_weights: Optional[np.ndarray] = None
    rounds_already = 0
    history_path: Optional[str] = None
    if existing and os.path.exists(existing.get("latest_weights_path", "")):
        initial_weights = np.load(existing["latest_weights_path"])
        rounds_already = int(existing.get("rounds_trained_total", 0))
        history_path = existing.get("history_path")
        progress.log(
            f"[cyan]Modèle linéaire p{player_count}[/cyan] : reprise à {rounds_already} manches déjà entraînées "
            f"(lr={learning_rate:g}, pool={opponent_pool})."
        )
    else:
        progress.log(
            f"[cyan]Modèle linéaire p{player_count}[/cyan] : aucun poids existant, création d'un nouveau modèle "
            f"(lr={learning_rate:g}, pool={opponent_pool})."
        )

    initial_epsilon = max(_EPSILON_MIN, _EPSILON_START * (_EPSILON_DECAY ** rounds_already))
    config = GameConfig(random_seed=seed, player_count=player_count)

    task_id = progress.add_task(f"Entraînement linéaire p{player_count}", total=rounds_increment, min_step_interval=5)
    trainee, running_vp = train(
        config,
        rounds_increment,
        learning_rate=learning_rate,
        opponent_pool=opponent_pool,
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
        model_name="pipeline_rl_weights", player_count=player_count, learning_rate=learning_rate, rounds=total_rounds,
    )
    np.save(output_path, trainee.weights)
    naming.write_weights_metadata(
        output_path,
        {
            "model_name": "pipeline_rl_weights",
            "player_count": player_count,
            "learning_rate": learning_rate,
            "rounds_trained": total_rounds,
            "opponent_pool": opponent_pool,
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
    head_window = running_vp[: max(1, len(running_vp) // 20)] if running_vp else []
    convergence_gap: Optional[float] = None
    if tail and head_window:
        convergence_gap = float(sum(tail) / len(tail) - sum(head_window) / len(head_window))

    result = {
        "latest_weights_path": output_path,
        "rounds_trained_total": total_rounds,
        "history_path": history_path,
        "final_vp_mean": float(sum(tail) / len(tail)) if tail else existing.get("final_vp_mean", 0.0) if existing else 0.0,
        "rounds_executed_this_run": rounds_executed,
        "learning_rate_used": learning_rate,
        "opponent_pool_used": opponent_pool,
        "convergence_gap_this_run": convergence_gap,
    }
    manifest["models"][key] = result
    return result


def _sweep_hyperparameters_incremental(
    manifest: Dict[str, Any],
    player_counts: List[int],
    rounds_per_run: int,
    seed: int,
    learning_rates: List[float],
    opponent_pools: List[str],
    seeds_per_combo: int,
    killer: GracefulKiller,
    progress: ProgressManager,
) -> Dict[str, Any]:
    """
    Étend le balayage d'hyperparamètres (taux d'apprentissage croisé avec le pool d'adversaires) avec toute combinaison
    (joueurs, taux, pool, répétition) non encore couverte.

    Paramètre `manifest` : manifeste cumulatif, mis à jour en place avec les combinaisons désormais couvertes.
    Paramètre `player_counts` : nombres de joueurs à couvrir.
    Paramètre `rounds_per_run` : nombre de manches d'entraînement par exécution individuelle.
    Paramètre `seed` : graine de base.
    Paramètre `learning_rates` : taux d'apprentissage testés.
    Paramètre `opponent_pools` : pools d'adversaires testés (`'greedy'`, `'rule_based'`, `'mixed'`).
    Paramètre `seeds_per_combo` : nombre de répétitions indépendantes par combinaison.
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

    all_combos: List[Tuple[int, float, str, int]] = [
        (pc, lr, pool, seed_offset)
        for pc in player_counts
        for lr in learning_rates
        for pool in opponent_pools
        for seed_offset in range(max(1, seeds_per_combo))
    ]
    pending_combos = [c for c in all_combos if _lr_combo_key(*c) not in combos_done]

    if not pending_combos:
        progress.log("[cyan]Balayage d'hyperparamètres[/cyan] : toutes les combinaisons demandées sont déjà couvertes.")
        return {"output_csv": output_csv, "new_combos": 0}

    task_id = progress.add_task("Balayage d'hyperparamètres", total=len(pending_combos))
    new_rows: List[Dict[str, Any]] = []
    added = 0
    for player_count, learning_rate, opponent_pool, seed_offset in pending_combos:
        if killer.should_stop:
            progress.log("[yellow]Arrêt demandé, balayage interrompu proprement.[/yellow]")
            break
        config = GameConfig(random_seed=seed + seed_offset, player_count=player_count)
        _trainee, running_vp = train(
            config, rounds_per_run, learning_rate=learning_rate, opponent_pool=opponent_pool,
            use_internal_progress=False,
        )
        tail = running_vp[-max(1, len(running_vp) // 20):] if running_vp else []
        new_rows.append({
            "player_count": player_count,
            "learning_rate": learning_rate,
            "opponent_pool": opponent_pool,
            "seed_index": seed_offset,
            "final_vp_mean": float(sum(tail) / len(tail)) if tail else 0.0,
            "rounds": rounds_per_run,
        })
        combos_done.add(_lr_combo_key(player_count, learning_rate, opponent_pool, seed_offset))
        added += 1
        progress.advance(
            task_id, 1,
            description=(
                f"Balayage — p{player_count}, lr={learning_rate:g}, pool={opponent_pool}, "
                f"répétition {seed_offset + 1}/{seeds_per_combo}"
            ),
        )
    progress.complete_task(task_id, description="Balayage d'hyperparamètres — terminé")

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


def _select_best_hyperparameters(
    manifest: Dict[str, Any], player_counts: List[int], progress: ProgressManager,
) -> Dict[str, Any]:
    """
    Détermine, pour chaque nombre de joueurs, la combinaison (taux d'apprentissage, pool d'adversaires) au VP moyen final le plus élevé
    parmi l'ensemble des résultats accumulés du balayage d'hyperparamètres.

    Paramètre `manifest` : manifeste cumulatif, mis à jour en place avec la sélection courante.
    Paramètre `player_counts` : nombres de joueurs à couvrir.
    Paramètre `progress` : gestionnaire de barres de progression partagé, utilisé uniquement pour journaliser la sélection retenue.
    Retourne le dictionnaire de sélection par nombre de joueurs. N'a aucun effet si le fichier CSV du balayage n'existe pas encore ou ne
    couvre aucun nombre de joueurs demandé. Effet de bord : remplace `manifest["best_hyperparameters"]`.
    """
    coverage = manifest.get("lr_sweep_coverage", {})
    output_csv = coverage.get("output_csv")
    if not output_csv or not os.path.exists(output_csv):
        return manifest.get("best_hyperparameters", {})

    import polars as pl

    frame = pl.read_csv(output_csv)
    best_by_player: Dict[str, Any] = dict(manifest.get("best_hyperparameters", {}))
    for player_count in player_counts:
        subset = frame.filter(pl.col("player_count") == player_count)
        if subset.is_empty():
            continue
        grouped = (
            subset.group_by(["learning_rate", "opponent_pool"])
            .agg(pl.col("final_vp_mean").mean().alias("mean_final_vp"))
            .sort("mean_final_vp", descending=True)
        )
        if grouped.is_empty():
            continue
        top = grouped.row(0, named=True)
        best_by_player[str(player_count)] = {
            "learning_rate": float(top["learning_rate"]),
            "opponent_pool": str(top["opponent_pool"]),
            "mean_final_vp": float(top["mean_final_vp"]),
        }
        progress.log(
            f"[cyan]Meilleure combinaison p{player_count}[/cyan] : lr={top['learning_rate']:g}, "
            f"pool={top['opponent_pool']}, VP moyen final = {top['mean_final_vp']:.3f}"
        )
    manifest["best_hyperparameters"] = best_by_player
    return best_by_player


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
    plutôt que d'en repartir de zéro.
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
    Paramètre `combos` : liste de tuples `(player_count, profile, rule_preset)` à couvrir, déjà réordonnée par priorité de couverture.
    Paramètre `games_increment` : nombre de parties supplémentaires par combinaison lors de cet appel.
    Paramètre `rounds_per_game` : nombre de manches par partie.
    Paramètre `seed_base` : graine de base, chaque combinaison dérivant sa propre plage de graines cumulative.
    Paramètre `killer` : indicateur d'arrêt propre, consulté entre deux combinaisons.
    Retourne un dictionnaire résumant, par combinaison, le nombre total de parties désormais couvertes et le dernier fichier Parquet
    produit. Effet de bord : lance une campagne Ray par combinaison non interrompue, écrivant systématiquement un nouveau fichier
    Parquet distinct plutôt que d'écraser les segments déjà produits par des lancements antérieurs.
    """
    from research.run_simulation import _RULE_PRESETS, launch_research

    results: Dict[str, Any] = {}
    for combo_index, (player_count, profile, preset) in enumerate(combos):
        if killer.should_stop:
            _console.print(console_theme.warning_text("Arrêt demandé, simulations de référence interrompues proprement."))
            break
        key = _baseline_key(player_count, profile, preset)
        coverage = manifest["baseline_coverage"].get(key, {"games_done": 0, "seed_cursor": seed_base, "parquet_paths": []})

        output_path = _unique_path(
            naming.build_research_filename(
                f"pipeline_baseline_{preset}", player_count, profile, games_increment, rounds_per_game,
            )
        )
        _console.print(
            console_theme.info_text(
                f"Baseline {combo_index + 1}/{len(combos)} — p{player_count} / {profile} / {preset} : "
                f"+{games_increment} parties (total après cette exécution : {coverage['games_done'] + games_increment})"
            )
        )
        launch_research(
            total_games=games_increment,
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

        coverage["games_done"] += games_increment
        coverage["seed_cursor"] += games_increment
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
    CSV distinct à chaque appel plutôt que d'écraser les résultats antérieurs.
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


def _run_combinatorial_search_incremental(
    manifest: Dict[str, Any],
    player_counts: List[int],
    rule_presets: List[str],
    profiles: List[str],
    games_per_combo: int,
    rounds_per_game_values: List[int],
    seed: int,
    run_index: int,
    killer: GracefulKiller,
) -> Optional[str]:
    """
    Lance une itération de recherche combinatoire sur le produit cartésien des profils heuristiques, nombres de joueurs, présets de
    règles et durées de partie couverts par la grille courante.

    Paramètre `manifest` : manifeste cumulatif, mis à jour en place avec le chemin du dernier manifeste combinatoire produit.
    Paramètre `player_counts` : nombres de joueurs couverts par la grille combinatoire.
    Paramètre `rule_presets` : présets de règles couverts.
    Paramètre `profiles` : profils heuristiques uniformes testés.
    Paramètre `games_per_combo` : nombre de parties simulées par combinaison individuelle.
    Paramètre `rounds_per_game_values` : nombres de manches par partie testés.
    Paramètre `seed` : graine de base de cette itération.
    Paramètre `run_index` : index du lancement de pipeline courant, utilisé pour distinguer le nom d'expérience d'une itération à l'autre.
    Paramètre `killer` : indicateur d'arrêt propre, consulté avant le lancement.
    Retourne le chemin du fichier manifeste CSV agrégé produit, ou `None` si l'exécution a été sautée (arrêt demandé). Effet de bord :
    exécute une campagne par combinaison du produit cartésien, écrit un nouveau manifeste CSV distinct à chaque itération de pipeline
    plutôt que d'écraser les précédents.
    """
    if killer.should_stop:
        return None

    from research.run_combinatory import run_grid

    experiment_name = f"pipeline_combinatorial_run{run_index}"
    _console.print(
        console_theme.info_text(
            f"Recherche combinatoire — {len(profiles)} profils x {len(player_counts)} tailles x "
            f"{len(rule_presets)} présets x {len(rounds_per_game_values)} durées, {games_per_combo} parties/combo"
        )
    )
    manifest_path = run_grid(
        experiment_name=experiment_name,
        agent_profiles=profiles,
        player_counts=player_counts,
        rule_presets=rule_presets,
        rounds_per_game_values=rounds_per_game_values,
        games_per_combo=games_per_combo,
        num_workers=max(1, os.cpu_count() or 1),
        base_seed=seed,
    )
    combinatorial_coverage = manifest.get("combinatorial_coverage", {})
    combinatorial_coverage.setdefault("manifest_paths", []).append(manifest_path)
    combinatorial_coverage["latest_manifest_path"] = manifest_path
    manifest["combinatorial_coverage"] = combinatorial_coverage
    return manifest_path


def _run_tournament_incremental(
    manifest: Dict[str, Any],
    player_counts: List[int],
    rounds_per_game: int,
    games: int,
    seed_base: int,
    profiles: List[str],
    killer: GracefulKiller,
) -> Dict[str, Any]:
    """
    Confronte directement, pour chaque nombre de joueurs disposant à la fois d'un modèle linéaire et d'un modèle neuronal entraînés, les
    deux politiques sur les mêmes sièges adverses fixes, et désigne le modèle au VP cumulé moyen le plus élevé comme champion courant
    pour ce nombre de joueurs.

    Paramètre `manifest` : manifeste cumulatif, consulté pour les chemins de poids les plus récents et mis à jour en place avec le
    résultat du tournoi.
    Paramètre `player_counts` : nombres de joueurs à couvrir.
    Paramètre `rounds_per_game` : nombre de manches par partie de confrontation.
    Paramètre `games` : nombre de parties de confrontation par nombre de joueurs.
    Paramètre `seed_base` : graine de base.
    Paramètre `profiles` : profils heuristiques disponibles pour occuper les sièges de remplissage.
    Paramètre `killer` : indicateur d'arrêt propre.
    Retourne un dictionnaire résumant, par nombre de joueurs, le champion désigné et le VP cumulé moyen de chaque modèle. Effet de bord :
    lance une campagne par nombre de joueurs couvert, écrit un nouveau fichier CSV distinct par confrontation. Un nombre de joueurs sans
    modèle linéaire et sans modèle neuronal disponibles simultanément est ignoré.
    """
    from research.evaluate_agent import launch_evaluation

    results: Dict[str, Any] = {}
    for player_count in player_counts:
        if killer.should_stop:
            break
        if player_count < 3:
            continue

        linear_key = _model_key("pipeline_rl_weights", player_count)
        neural_key = _model_key("pipeline_torch_rl_weights", player_count)
        linear_info = manifest["models"].get(linear_key)
        neural_info = manifest["models"].get(neural_key)
        linear_path = linear_info.get("latest_weights_path") if linear_info else None
        neural_path = neural_info.get("latest_weights_path") if neural_info else None

        if not linear_path or not os.path.exists(linear_path):
            continue
        if not neural_path or not os.path.exists(neural_path):
            continue

        filler = list(profiles[: max(0, player_count - 2)])
        while len(filler) < max(0, player_count - 2):
            filler.append(profiles[0] if profiles else "greedy_bot")
        seat_profiles = ["rl_agent", "torch_rl_agent"] + filler
        seat_profiles = seat_profiles[:player_count]
        seat_weights = {0: linear_path, 1: neural_path}

        output_csv = _unique_path(
            naming.build_research_filename(
                "pipeline_tournament", player_count, "rl_vs_torch", games, rounds_per_game, extension="csv",
            )
        )
        _console.print(
            console_theme.info_text(f"Tournoi p{player_count} — rl_agent vs torch_rl_agent : {games} parties")
        )
        launch_evaluation(
            total_games=games,
            seat_profiles=seat_profiles,
            rounds_per_game=rounds_per_game,
            num_workers=max(1, os.cpu_count() or 1),
            base_seed=seed_base + player_count,
            experiment_name="pipeline_tournament",
            seat_weights=seat_weights,
            output_csv=output_csv,
            shutdown_ray=False,
        )

        import polars as pl

        frame = pl.read_csv(output_csv)
        summary = (
            frame.filter(pl.col("profile").is_in(["rl_agent", "torch_rl_agent"]))
            .group_by("profile")
            .agg(pl.col("cumulative_vp").mean().alias("mean_cumulative_vp"))
        )
        scores = {row["profile"]: float(row["mean_cumulative_vp"]) for row in summary.to_dicts()}
        champion = max(scores, key=lambda profile: scores[profile]) if scores else None
        results[str(player_count)] = {
            "scores": scores,
            "champion": champion,
            "output_csv": output_csv,
        }
    manifest["tournament_results"] = {**manifest.get("tournament_results", {}), **results}
    return results


def _run_statistical_validation(
    seed: int,
    killer: GracefulKiller,
    round_count: int,
    player_count: int = _VALIDATION_PLAYER_COUNT,
) -> Dict[str, Any]:
    """
    Exécute une série de manches déterministes fraîches et vérifie un ensemble de propriétés statistiques générales attendues du moteur
    de règles : bornes de l'indice de Gini de la puissance de main initiale, somme des lignes de la matrice de transition de rôles,
    absence de passe sous-optimal pour un agent glouton déterministe, positivité du facteur de branchement et de l'entropie de l'espace
    d'action, et borne de la longueur moyenne d'un pli sous passe strict.

    Paramètre `seed` : graine de reproductibilité de cette série de vérification.
    Paramètre `killer` : indicateur d'arrêt propre, consulté entre deux manches.
    Paramètre `round_count` : nombre de manches simulées pour cette vérification.
    Paramètre `player_count` : nombre de joueurs de la partie de vérification.
    Retourne un dictionnaire `{nom_de_vérification: {"passed": bool, "value": ..., "detail": str}}`. Effet de bord : simule localement
    une partie complète jetable, sans écriture sur disque.
    """
    config = GameConfig(random_seed=seed, player_count=player_count)
    agents: Dict[int, AbstractBaseAgent] = {pid: GreedyBot(pid, config) for pid in range(player_count)}
    logger = EventLogger()
    bus = EventBus()
    bus.subscribe(logger)
    game = Game(config, agents, event_bus=bus, game_id="pipeline-validation")

    role_sequence: List[Dict[int, str]] = []
    for _ in range(round_count):
        if killer.should_stop:
            break
        game.play_round()
        role_sequence.append(dict(game.roles or {}))

    checks: Dict[str, Any] = {}

    gini_values: List[float] = []
    for round_start in logger.events_of_type(EventRoundStart):
        hand_powers = {
            pid: float(sum(f_std(c) for c in cards if c.rank.value != "JOKER"))
            for pid, cards in round_start.initial_hands.items()
        }
        gini_values.append(gini_initial_hand_power(hand_powers))
    gini_mean = (sum(gini_values) / len(gini_values)) if gini_values else None
    checks["gini_within_bounds"] = {
        "passed": gini_mean is not None and 0.0 <= gini_mean < 1.0,
        "value": gini_mean,
        "detail": "Indice de Gini moyen de la puissance de main initiale, domaine attendu [0, 1[.",
    }

    suboptimal_rates = {pid: sub_optimal_pass_rate(logger, pid) for pid in range(player_count)}
    all_zero = all(rate == 0.0 for rate in suboptimal_rates.values())
    checks["greedy_never_suboptimal"] = {
        "passed": all_zero,
        "value": suboptimal_rates,
        "detail": "Un agent glouton déterministe ne doit jamais produire de passe sous-optimal.",
    }

    matrix = role_transition_matrix(role_sequence)
    row_sums: Dict[str, float] = {}
    for role_a in {role for role, _ in matrix.keys()}:
        row_sums[role_a] = sum(prob for (ra, _rb), prob in matrix.items() if ra == role_a)
    rows_ok = all(abs(total - 1.0) < 1e-9 for total in row_sums.values())
    checks["role_transition_rows_sum_to_one"] = {
        "passed": rows_ok,
        "value": row_sums,
        "detail": "Chaque ligne non vide de la matrice de transition de rôles doit sommer à 1.",
    }

    branching = branching_factor_average(logger)
    checks["branching_factor_positive"] = {
        "passed": branching > 0.0,
        "value": branching,
        "detail": "Le facteur de branchement moyen doit être strictement positif sur une partie non triviale.",
    }

    entropy = action_space_entropy(logger)
    checks["action_space_entropy_non_negative"] = {
        "passed": entropy >= 0.0,
        "value": entropy,
        "detail": "L'entropie de Shannon de l'espace d'action est par construction positive ou nulle.",
    }

    trick_length = trick_length_average(logger)
    checks["trick_length_bounded_under_hard_pass"] = {
        "passed": trick_length <= player_count + 1e-9,
        "value": trick_length,
        "detail": "Sous pass_type=HARD_ONLY, la longueur moyenne d'un pli est bornée par le nombre de joueurs.",
    }

    logger.close()
    return checks


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
            f"VP moyen récent : {info.get('final_vp_mean', 'n/a')}, "
            f"taux d'apprentissage utilisé : {info.get('learning_rate_used', 'n/a')}, "
            f"pool d'adversaires : {info.get('opponent_pool_used', 'n/a')}"
        )
    lines.append("")

    lines.append("## Meilleurs hyperparamètres identifiés par balayage")
    best_hyperparams = manifest.get("best_hyperparameters", {})
    if best_hyperparams:
        for player_count_key, info in sorted(best_hyperparams.items(), key=lambda kv: int(kv[0])):
            lines.append(
                f"- {player_count_key} joueurs : taux d'apprentissage = {info['learning_rate']:g}, "
                f"pool d'adversaires = {info['opponent_pool']}, VP moyen final = {info['mean_final_vp']:.3f}"
            )
    else:
        lines.append("- Aucune combinaison encore sélectionnée (balayage non exécuté ou insuffisant).")
    lines.append("")

    lines.append("## Balayage d'hyperparamètres")
    lr_coverage = manifest.get("lr_sweep_coverage", {})
    lines.append(f"- Fichier CSV cumulatif : `{lr_coverage.get('output_csv', 'indisponible')}`")
    lines.append(f"- Combinaisons (joueurs, taux, pool, répétition) couvertes à ce jour : {len(lr_coverage.get('combos_done', []))}")
    lines.append("")

    lines.append("## Couverture des simulations de référence")
    lines.append(f"- Nombre total de combinaisons (joueurs, profil, préset) distinctes couvertes : {len(manifest['baseline_coverage'])}")
    for key, coverage in sorted(manifest["baseline_coverage"].items()):
        lines.append(f"- `{key}` : {coverage.get('games_done', 0)} parties cumulées sur {len(coverage.get('parquet_paths', []))} segment(s)")
    lines.append("")

    lines.append("## Couverture des évaluations comparatives")
    for key, coverage in sorted(manifest["evaluation_coverage"].items()):
        lines.append(f"- `{key}` : {coverage.get('games_done', 0)} parties cumulées sur {len(coverage.get('csv_paths', []))} fichier(s)")
    lines.append("")

    lines.append("## Recherche combinatoire")
    combinatorial_coverage = manifest.get("combinatorial_coverage", {})
    if combinatorial_coverage.get("latest_manifest_path"):
        lines.append(f"- Dernier manifeste combinatoire : `{combinatorial_coverage['latest_manifest_path']}`")
        lines.append(f"- Nombre total d'itérations combinatoires exécutées : {len(combinatorial_coverage.get('manifest_paths', []))}")
    else:
        lines.append("- Aucune itération de recherche combinatoire encore exécutée.")
    lines.append("")

    lines.append("## Tournoi (politique linéaire contre politique neuronale)")
    tournament_results = manifest.get("tournament_results", {})
    if tournament_results:
        for player_count_key, info in sorted(tournament_results.items(), key=lambda kv: int(kv[0])):
            scores_text = ", ".join(f"{profile} = {score:.3f}" for profile, score in info.get("scores", {}).items())
            lines.append(f"- {player_count_key} joueurs : champion = `{info.get('champion', 'n/a')}` ({scores_text})")
    else:
        lines.append(
            "- Aucun tournoi encore exécuté (nécessite un modèle linéaire et un modèle neuronal entraînés pour un même nombre de joueurs)."
        )
    lines.append("")

    lines.append("## Vérification statistique du moteur de règles")
    validation = manifest.get("validation_results", {})
    if validation:
        overall = "toutes réussies" if validation.get("all_passed") else "au moins un échec"
        lines.append(f"- Résultat global (lancement #{validation.get('run_index', '?')}) : {overall}")
        for check_name, check_info in validation.get("checks", {}).items():
            status = "OK" if check_info.get("passed") else "ÉCHEC"
            lines.append(f"  - [{status}] `{check_name}` : {check_info.get('detail', '')} (valeur observée : {check_info.get('value')})")
    else:
        lines.append("- Aucune vérification statistique encore exécutée.")
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
    models_table = Table(title=f"[{console_theme.STYLE_STEP}]Modèles entraînés[/{console_theme.STYLE_STEP}]", expand=True)
    models_table.add_column("Modèle")
    models_table.add_column("Manches/étapes cumulées")
    models_table.add_column("VP moyen récent")
    for key, info in sorted(manifest["models"].items()):
        models_table.add_row(key, str(info.get("rounds_trained_total", "?")), str(info.get("final_vp_mean", "n/a")))
    _console.print(models_table)

    coverage_table = Table(title=f"[{console_theme.STYLE_STEP}]Couverture cumulée[/{console_theme.STYLE_STEP}]", expand=True)
    coverage_table.add_column("Combinaison")
    coverage_table.add_column("Type")
    coverage_table.add_column("Parties cumulées")
    for key, coverage in sorted(manifest["baseline_coverage"].items())[:40]:
        coverage_table.add_row(key, "baseline", str(coverage.get("games_done", 0)))
    for key, coverage in sorted(manifest["evaluation_coverage"].items()):
        coverage_table.add_row(key, "évaluation", str(coverage.get("games_done", 0)))
    _console.print(coverage_table)
    if len(manifest["baseline_coverage"]) > 40:
        _console.print(
            console_theme.info_text(
                f"({len(manifest['baseline_coverage']) - 40} combinaison(s) de référence supplémentaire(s) omise(s) de "
                "l'affichage, consulter le rapport de synthèse pour la liste complète.)"
            )
        )

    tournament_results = manifest.get("tournament_results", {})
    if tournament_results:
        tournament_table = Table(title=f"[{console_theme.STYLE_STEP}]Tournoi linéaire vs neuronal[/{console_theme.STYLE_STEP}]", expand=True)
        tournament_table.add_column("Joueurs")
        tournament_table.add_column("Champion")
        tournament_table.add_column("Scores")
        for key, info in sorted(tournament_results.items(), key=lambda kv: int(kv[0])):
            scores_text = ", ".join(f"{profile}={score:.2f}" for profile, score in info.get("scores", {}).items())
            tournament_table.add_row(key, str(info.get("champion", "n/a")), scores_text)
        _console.print(tournament_table)

    validation = manifest.get("validation_results", {})
    if validation:
        validation_table = Table(title=f"[{console_theme.STYLE_STEP}]Vérification statistique[/{console_theme.STYLE_STEP}]", expand=True)
        validation_table.add_column("Vérification")
        validation_table.add_column("Statut")
        for check_name, check_info in validation.get("checks", {}).items():
            status = (
                f"[{console_theme.STYLE_SUCCESS}]OK[/{console_theme.STYLE_SUCCESS}]"
                if check_info.get("passed")
                else f"[{console_theme.STYLE_ERROR}]ÉCHEC[/{console_theme.STYLE_ERROR}]"
            )
            validation_table.add_row(check_name, status)
        _console.print(validation_table)


def run_pipeline(
    player_counts: List[int],
    rule_presets: List[str],
    training_rounds_increment: int,
    lr_sweep_rounds: int,
    lr_sweep_seeds: int,
    learning_rates: List[float],
    opponent_pools: List[str],
    distributed_steps_increment: int,
    baseline_games_increment: int,
    baseline_rounds_per_game: int,
    evaluation_games_increment: int,
    evaluation_rounds_per_game: int,
    combinatorial_games_per_combo: int,
    combinatorial_rounds_per_game_values: List[int],
    tournament_games: int,
    tournament_rounds_per_game: int,
    validation_round_count: int,
    seed: int,
    redis_host: str,
    redis_port: int,
    skip_distributed: bool,
    skip_combinatorial: bool,
    skip_tournament: bool,
    skip_validation: bool,
) -> None:
    """
    Exécute une itération incrémentale complète du pipeline de recherche sur toute une grille de configurations.

    Paramètre `player_counts` : nombres de joueurs couverts par la grille d'analyse.
    Paramètre `rule_presets` : présets de règles couverts par la grille d'analyse (voir `_resolve_rule_presets` pour la résolution du
    mode `'auto'`).
    Paramètre `training_rounds_increment` : manches d'entraînement supplémentaires ajoutées à chaque modèle linéaire lors de cet appel.
    Paramètre `lr_sweep_rounds`, `lr_sweep_seeds`, `learning_rates`, `opponent_pools` : paramètres du balayage d'hyperparamètres.
    Paramètre `distributed_steps_increment` : étapes de gradient supplémentaires ajoutées à chaque modèle neuronal, si Redis est joignable.
    Paramètre `baseline_games_increment` : parties supplémentaires ajoutées à chaque combinaison de référence.
    Paramètre `baseline_rounds_per_game` : manches par partie de référence.
    Paramètre `evaluation_games_increment` : parties supplémentaires ajoutées à chaque combinaison d'évaluation.
    Paramètre `evaluation_rounds_per_game` : manches par partie d'évaluation.
    Paramètre `combinatorial_games_per_combo`, `combinatorial_rounds_per_game_values` : paramètres de la recherche combinatoire.
    Paramètre `tournament_games`, `tournament_rounds_per_game` : paramètres du tournoi linéaire contre neuronal.
    Paramètre `validation_round_count` : nombre de manches simulées pour la vérification statistique automatisée.
    Paramètre `seed` : graine de base de cette itération.
    Paramètre `redis_host`, `redis_port` : coordonnées Redis pour l'entraînement distribué.
    Paramètre `skip_distributed` : si vrai, n'essaie même pas de joindre Redis pour cette itération.
    Paramètre `skip_combinatorial` : si vrai, saute l'étape de recherche combinatoire.
    Paramètre `skip_tournament` : si vrai, saute l'étape de tournoi linéaire contre neuronal.
    Paramètre `skip_validation` : si vrai, saute l'étape de vérification statistique automatisée.
    Retourne `None`. Effet de bord : exécute toutes les étapes ci-dessus, sauvegarde le manifeste après chacune (résistant à une
    interruption brutale entre deux étapes), régénère les graphiques (nouvelle version) et le rapport final, puis affiche un résumé complet.
    """
    manifest = _load_manifest()
    killer = GracefulKiller()
    profiles = _heuristic_profiles()
    run_started_at = time.time()
    run_index = len(manifest["runs"]) + 1

    baseline_profile_combos = [
        (pc, profile, preset)
        for pc in player_counts
        for profile in profiles
        for preset in rule_presets
    ]
    baseline_profile_combos = _prioritize_baseline_combos(manifest, baseline_profile_combos)
    evaluation_combos = list(itertools.product(player_counts, rule_presets))

    try:
        with ProgressManager(console=_console) as progress:
            for player_count in player_counts:
                if killer.should_stop:
                    break
                _train_linear_agent_incremental(manifest, player_count, training_rounds_increment, seed, killer, progress)
                _save_manifest(manifest)

            if not killer.should_stop:
                _sweep_hyperparameters_incremental(
                    manifest, player_counts, lr_sweep_rounds, seed + 5_000, learning_rates, opponent_pools,
                    lr_sweep_seeds, killer, progress,
                )
                _save_manifest(manifest)

            if not killer.should_stop:
                _select_best_hyperparameters(manifest, player_counts, progress)
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

        if not killer.should_stop and not skip_combinatorial:
            _run_combinatorial_search_incremental(
                manifest, player_counts, rule_presets, profiles, combinatorial_games_per_combo,
                combinatorial_rounds_per_game_values, seed + 30_000, run_index, killer,
            )
            _save_manifest(manifest)

        if not killer.should_stop and not skip_tournament:
            _run_tournament_incremental(
                manifest, player_counts, tournament_rounds_per_game, tournament_games,
                seed + 40_000, profiles, killer,
            )
            _save_manifest(manifest)

        if not killer.should_stop and not skip_validation:
            _console.print(console_theme.info_text("Vérification statistique du moteur de règles…"))
            validation_results = _run_statistical_validation(seed + 50_000, killer, round_count=validation_round_count)
            manifest["validation_results"] = {
                "run_index": run_index,
                "checks": validation_results,
                "all_passed": all(check["passed"] for check in validation_results.values()),
            }
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
        "rule_presets_count": len(rule_presets),
        "profiles_count": len(profiles),
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
    pour valider rapidement que le pipeline s'exécute de bout en bout sans erreur. Par défaut, `--rule-presets` vaut `'auto'`, qui étend
    la grille de règles couverte à l'intégralité des paramètres de `core.config.GameConfig` (chaque champ booléen basculé
    individuellement, chaque valeur énumérée testée, plus des combinaisons aléatoires couvrant les interactions), et `--player-counts`
    couvre par défaut l'ensemble des tailles de partie de 3 à 8 joueurs.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline automatique incrémental : entraînement continu, grille de configurations étendue, graphiques versionnés."
    )
    parser.add_argument("--player-counts", type=str, default="3,4,5,6,7,8")
    parser.add_argument(
        "--rule-presets", type=str, default="auto",
        help="'auto' pour couvrir l'intégralité de la grille de paramètres générée automatiquement, ou une liste explicite "
             "de noms de présets séparés par des virgules.",
    )
    parser.add_argument("--training-rounds-increment", type=int, default=20000)
    parser.add_argument("--lr-sweep-rounds", type=int, default=800)
    parser.add_argument("--lr-sweep-seeds", type=int, default=3)
    parser.add_argument("--learning-rates", type=str, default="0.001,0.003,0.01,0.03,0.1,0.3")
    parser.add_argument("--opponent-pools", type=str, default="greedy,rule_based,mixed")
    parser.add_argument("--distributed-steps-increment", type=int, default=2000)
    parser.add_argument("--skip-distributed", action="store_true")
    parser.add_argument("--baseline-games-increment", type=int, default=80)
    parser.add_argument("--baseline-rounds-per-game", type=int, default=15)
    parser.add_argument("--evaluation-games-increment", type=int, default=80)
    parser.add_argument("--evaluation-rounds-per-game", type=int, default=25)
    parser.add_argument("--combinatorial-games-per-combo", type=int, default=20)
    parser.add_argument("--combinatorial-rounds-per-game-values", type=str, default="10,30,60")
    parser.add_argument("--skip-combinatorial", action="store_true")
    parser.add_argument("--tournament-games", type=int, default=60)
    parser.add_argument("--tournament-rounds-per-game", type=int, default=25)
    parser.add_argument("--skip-tournament", action="store_true")
    parser.add_argument("--validation-rounds", type=int, default=60)
    parser.add_argument("--skip-validation", action="store_true")
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
    learning_rates = [float(token.strip()) for token in args.learning_rates.split(",") if token.strip()]
    opponent_pools = [token.strip() for token in args.opponent_pools.split(",") if token.strip()]
    combinatorial_rounds_per_game_values = [
        int(token.strip()) for token in args.combinatorial_rounds_per_game_values.split(",") if token.strip()
    ]

    training_rounds_increment = args.training_rounds_increment
    lr_sweep_rounds = args.lr_sweep_rounds
    lr_sweep_seeds = args.lr_sweep_seeds
    distributed_steps_increment = args.distributed_steps_increment
    baseline_games_increment = args.baseline_games_increment
    evaluation_games_increment = args.evaluation_games_increment
    combinatorial_games_per_combo = args.combinatorial_games_per_combo
    tournament_games = args.tournament_games
    validation_round_count = args.validation_rounds
    rule_presets_arg = args.rule_presets

    if args.quick:
        training_rounds_increment = min(training_rounds_increment, 80)
        lr_sweep_rounds = min(lr_sweep_rounds, 40)
        lr_sweep_seeds = min(lr_sweep_seeds, 1)
        distributed_steps_increment = min(distributed_steps_increment, 20)
        baseline_games_increment = min(baseline_games_increment, 8)
        evaluation_games_increment = min(evaluation_games_increment, 8)
        combinatorial_games_per_combo = min(combinatorial_games_per_combo, 3)
        combinatorial_rounds_per_game_values = combinatorial_rounds_per_game_values[:1] or [5]
        tournament_games = min(tournament_games, 6)
        validation_round_count = min(validation_round_count, 10)
        player_counts = player_counts[:1]
        opponent_pools = opponent_pools[:1] or ["mixed"]
        learning_rates = learning_rates[:2] or [0.01]
        if rule_presets_arg.strip().lower() == "auto":
            rule_presets_arg = "base,straights"
        _console.print(console_theme.warning_text("Mode --quick actif : volumes de travail fortement réduits."))

    rule_presets = _resolve_rule_presets(rule_presets_arg)

    run_pipeline(
        player_counts=player_counts,
        rule_presets=rule_presets,
        training_rounds_increment=training_rounds_increment,
        lr_sweep_rounds=lr_sweep_rounds,
        lr_sweep_seeds=lr_sweep_seeds,
        learning_rates=learning_rates,
        opponent_pools=opponent_pools,
        distributed_steps_increment=distributed_steps_increment,
        baseline_games_increment=baseline_games_increment,
        baseline_rounds_per_game=args.baseline_rounds_per_game,
        evaluation_games_increment=evaluation_games_increment,
        evaluation_rounds_per_game=args.evaluation_rounds_per_game,
        combinatorial_games_per_combo=combinatorial_games_per_combo,
        combinatorial_rounds_per_game_values=combinatorial_rounds_per_game_values,
        tournament_games=tournament_games,
        tournament_rounds_per_game=args.tournament_rounds_per_game,
        validation_round_count=validation_round_count,
        seed=args.seed,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        skip_distributed=args.skip_distributed,
        skip_combinatorial=args.skip_combinatorial,
        skip_tournament=args.skip_tournament,
        skip_validation=args.skip_validation,
    )


if __name__ == "__main__":
    main()
