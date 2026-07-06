"""
Module de génération non interactive des graphiques d'analyse.

Le module relit l'ensemble des fichiers Parquet segmentés, des résumés CSV, des manifestes de recherche combinatoire et des historiques
d'entraînement produits dans `data/` et `weights/`, puis produit un ensemble de graphiques statiques et interactifs dans `figures/`, sans
aucune fenêtre interactive ni aucun blocage sur une saisie utilisateur. Ce module est l'équivalent exécutable en une seule commande du
carnet d'analyse existant, conçu pour être appelé depuis un pipeline automatique.

Le module dépend de `matplotlib`, `seaborn`, `plotly`, `pandas`, `polars` n'est pas requis ici (lecture via `pandas`/`pyarrow`).
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

sns.set_theme(style="whitegrid")

DATA_DIR = "data"
WEIGHTS_DIR = "weights"
FIGURE_DIR = "figures"

_POINTS_TABLE = {
    "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
    "J": 11, "Q": 12, "K": 13, "A": 14, "2": 15, "JOKER": 16,
}

# Motif extrayant le profil d'agent et le nombre de parties d'un nom de fichier de simulation nommé selon `naming.build_research_filename`,
# pour construire un libellé court plutôt que d'afficher le nom de fichier complet (souvent long de plus de 60 caractères) dans une légende.
_SOURCE_LABEL_RE = re.compile(r"agent([A-Za-z0-9_]+?)_games(\d+)")

# Motif extrayant le nom de modèle et le nombre de manches entraînées d'un nom de fichier d'historique nommé selon `naming.build_weights_filename`,
# pour construire un libellé court de légende.
_MODEL_LABEL_RE = re.compile(r"^(.*?)_player(\d+)_learnRate([\dpm]+)_rounds(\d+)")


def _truncate_label(label: str, max_len: int = 22) -> str:
    """
    Raccourcit un libellé de légende ou de titre à une longueur maximale, avec points de suspension.

    Paramètre `label` : chaîne source.
    Paramètre `max_len` : longueur maximale autorisée, points de suspension inclus.
    Retourne `label` inchangé s'il tient déjà dans `max_len`, sinon une version tronquée suivie de `…`. Aucun effet de bord.
    """
    if len(label) <= max_len:
        return label
    return label[: max(1, max_len - 1)] + "…"


def _short_source_label(filename: str) -> str:
    """
    Construit un libellé court et lisible pour une source de simulation, à partir de son nom de fichier complet.

    Paramètre `filename` : nom de fichier (Parquet ou dérivé), typiquement long (profil, nombre de joueurs, nombre de manches, date).
    Retourne une chaîne courte du type `"rule_based_bot (g100)"` lorsque le motif attendu est reconnu, ou une troncature générique sinon.
    Aucun effet de bord.
    """
    match = _SOURCE_LABEL_RE.search(filename)
    if match:
        return f"{match.group(1)} (g{match.group(2)})"
    base = os.path.splitext(os.path.basename(filename))[0]
    return _truncate_label(base, 22)


def _short_model_label(filename: str) -> str:
    """
    Construit un libellé court et lisible pour un modèle entraîné, à partir de son nom de fichier d'historique complet.

    Paramètre `filename` : nom de fichier d'historique (`*.history.csv`), typiquement long.
    Retourne une chaîne courte du type `"pipeline_rl_weights p4 r4000"` lorsque le motif attendu est reconnu, ou une troncature générique
    sinon. Aucun effet de bord.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    match = _MODEL_LABEL_RE.match(base)
    if match:
        return f"{match.group(1)} p{match.group(2)} r{match.group(4)}"
    return _truncate_label(base, 22)


def _gini(values: List[float]) -> float:
    """
    Calcule l'indice de Gini d'une liste de valeurs.

    Paramètre `values` : liste de valeurs numériques positives ou nulles.
    Retourne un nombre, domaine $[0, 1]$, nul si la liste est vide ou si sa somme est nulle. Aucun effet de bord.
    """
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    total = sum(ordered)
    if total == 0:
        return 0.0
    cumulative = sum((i + 1) * v for i, v in enumerate(ordered))
    return (2 * cumulative) / (n * total) - (n + 1) / n


def _load_segmented_parquet(pattern: str = "*.parquet") -> pd.DataFrame:
    """
    Charge et concatène l'ensemble des fichiers Parquet segmentés du répertoire `data/`.

    Paramètre `pattern` : motif de nom de fichier à filtrer.
    Retourne un `DataFrame` pandas, vide et à schéma minimal si aucun fichier ne correspond. Aucun effet de bord hors la lecture disque.
    """
    paths = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    frames = []
    for path in paths:
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        frame["source_file"] = os.path.basename(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=["event_type", "timestamp", "game_id", "round_id", "state_hash", "payload", "source_file"]
        )
    return pd.concat(frames, ignore_index=True)


def _expand_payload(df: pd.DataFrame, event_type: str) -> pd.DataFrame:
    """
    Développe la colonne `payload` (JSON) d'un sous-ensemble d'événements en colonnes distinctes.

    Paramètre `df` : `DataFrame` au schéma segmenté (`event_type`, `payload`, ...).
    Paramètre `event_type` : type d'événement à isoler.
    Retourne un `DataFrame` filtré et aplati, vide si aucun événement du type demandé n'est présent. Aucun effet de bord.
    """
    subset = df[df["event_type"] == event_type].copy()
    if subset.empty:
        return subset
    payloads = subset["payload"].apply(json.loads)
    payload_df = pd.json_normalize(payloads.tolist())
    payload_df.index = subset.index
    return pd.concat([subset.drop(columns=["payload"]), payload_df], axis=1)


def _load_summary_csvs(pattern: str = "*summary.csv") -> pd.DataFrame:
    """Charge et concatène les fichiers de résumé de campagne (`research.run_simulation`)."""
    paths = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_grid_manifests(pattern: str = "*manifest*.csv") -> pd.DataFrame:
    """Charge et concatène les manifestes de recherche combinatoire (`research.run_combinatory`)."""
    paths = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_training_histories(pattern: str = "*history.csv") -> pd.DataFrame:
    """Charge et concatène les historiques d'entraînement (`training.train_rl`)."""
    paths = sorted(glob.glob(os.path.join(WEIGHTS_DIR, pattern)))
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["model_file"] = os.path.basename(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_learning_rate_sweep(pattern: str = "learning_rate_sweep*.csv") -> pd.DataFrame:
    """
    Charge et concatène l'ensemble des segments du balayage de taux d'apprentissage.

    Paramètre `pattern` : motif de nom de fichier, couvrant à la fois le fichier cumulatif principal et d'éventuelles variantes
    versionnées produites par des lancements de pipeline distincts.
    Retourne un `DataFrame` pandas vide si aucun fichier ne correspond, sinon la concaténation de l'ensemble des fichiers trouvés. Aucun
    effet de bord.
    """
    paths = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def _plot_learning_rate_sweep(sweep_df: pd.DataFrame, output_dir: str) -> None:
    """Performance finale d'entraînement par taux d'apprentissage testé (boîtes à moustaches)."""
    if sweep_df.empty or "learning_rate" not in sweep_df.columns or "final_vp_mean" not in sweep_df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ordered = sweep_df.sort_values("learning_rate")
    hue_column = "player_count" if "player_count" in ordered.columns else None
    sns.boxplot(data=ordered, x="learning_rate", y="final_vp_mean", hue=hue_column, ax=ax)
    ax.set_xlabel("Taux d'apprentissage")
    ax.set_ylabel("VP moyen final (fin d'entraînement)")
    ax.set_title("Performance finale selon le taux d'apprentissage")
    if hue_column:
        ax.legend(title="Joueurs", fontsize=8)
    _savefig(fig, output_dir, "learning_rate_final_performance_boxplot.png")


def _load_evaluation_csvs(pattern: str = "*.csv") -> pd.DataFrame:
    """Charge et concatène les fichiers d'évaluation comparative (`research.evaluate_agent`), en excluant résumés et manifestes."""
    paths = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    paths = [p for p in paths if "summary" not in os.path.basename(p) and "manifest" not in os.path.basename(p)]
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = os.path.basename(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _savefig(fig, output_dir: str, name: str) -> None:
    """Sauvegarde une figure matplotlib dans le répertoire versionné fourni et ferme la figure pour libérer la mémoire."""
    os.makedirs(output_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, name), dpi=150)
    plt.close(fig)


def _plot_vp_by_rank_violin(finished_df: pd.DataFrame, output_dir: str) -> None:
    """Distribution des points de victoire par rang de sortie (violin plot)."""
    if finished_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=finished_df, x="rank", y="vp_earned", ax=ax, inner="quartile")
    ax.set_xlabel("Rang de sortie")
    ax.set_ylabel("Points de victoire")
    ax.set_title("Distribution des VP par rang de sortie")
    _savefig(fig, output_dir, "vp_by_rank_violin.png")


def _plot_win_rate_by_player(finished_df: pd.DataFrame, output_dir: str) -> None:
    """Taux de victoire par identifiant de joueur, avec intervalle de confiance."""
    if finished_df.empty:
        return
    finished_df = finished_df.copy()
    finished_df["is_winner"] = finished_df["rank"] == 0
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=finished_df, x="player_id", y="is_winner", errorbar=("ci", 95), ax=ax)
    ax.set_xlabel("Identifiant de joueur")
    ax.set_ylabel("Taux de victoire (rang 0)")
    ax.set_title("Taux de victoire par joueur")
    _savefig(fig, output_dir, "win_rate_by_player_ci.png")


def _plot_suboptimal_pass_rate(action_played_df: pd.DataFrame, output_dir: str) -> None:
    """Taux de passe sous-optimal par identifiant de joueur, avec intervalle de confiance."""
    if action_played_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=action_played_df, x="player_id", y="was_suboptimal", errorbar=("ci", 95), ax=ax)
    ax.set_xlabel("Identifiant de joueur")
    ax.set_ylabel("Taux de passe sous-optimal")
    ax.set_title("Taux de passe sous-optimal par joueur")
    _savefig(fig, output_dir, "suboptimal_pass_rate_ci.png")


def _role_transition_matrix(round_end_df: pd.DataFrame) -> pd.DataFrame:
    """Construit la matrice de transition de rôles d'une manche à la suivante, par partie."""
    matrix = pd.DataFrame()
    if round_end_df.empty or "roles_by_player" not in round_end_df.columns:
        return matrix
    transition_counts: Dict[str, Dict[str, int]] = {}
    grouped = round_end_df.sort_values(["game_id", "round_id"]).groupby("game_id")
    for _, group in grouped:
        roles_sequence = group["roles_by_player"].tolist()
        for earlier, later in zip(roles_sequence, roles_sequence[1:]):
            for pid, role_a in earlier.items():
                role_b = later.get(pid)
                if role_b is None:
                    continue
                transition_counts.setdefault(role_a, {}).setdefault(role_b, 0)
                transition_counts[role_a][role_b] += 1

    roles_order = ["ROLE_PRESIDENT", "ROLE_VICE_PRESIDENT", "ROLE_NEUTRAL", "ROLE_VICE_SCUM", "ROLE_SCUM"]
    matrix = pd.DataFrame(0.0, index=roles_order, columns=roles_order)
    for role_a, counts in transition_counts.items():
        total = sum(counts.values())
        for role_b, count in counts.items():
            if role_a in matrix.index and role_b in matrix.columns:
                matrix.loc[role_a, role_b] = count / total if total else 0.0
    return matrix


def _plot_role_transition_heatmap(matrix: pd.DataFrame, output_dir: str) -> None:
    """Heatmap de la matrice de transition des rôles."""
    if matrix.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax)
    ax.set_xlabel("Rôle manche suivante")
    ax.set_ylabel("Rôle manche courante")
    ax.set_title("Transition des rôles d'une manche à la suivante")
    _savefig(fig, output_dir, "role_transition_heatmap.png")


def _plot_gini_histogram(round_start_df: pd.DataFrame, output_dir: str) -> None:
    """Histogramme de l'indice de Gini de la puissance de main initiale."""
    if round_start_df.empty or "initial_hands" not in round_start_df.columns:
        return
    records = []
    for _, row in round_start_df.iterrows():
        for pid, cards in row["initial_hands"].items():
            power_sum = sum(_POINTS_TABLE.get(card["rank"], 0) for card in cards if card["rank"] != "JOKER")
            records.append({"game_id": row["game_id"], "round_id": row["round_id"], "player_id": pid, "hand_power": power_sum})
    hand_power_df = pd.DataFrame(records)
    if hand_power_df.empty:
        return
    gini_per_round = (
        hand_power_df.groupby(["game_id", "round_id"])["hand_power"]
        .apply(lambda values: _gini(values.tolist()))
        .reset_index(name="gini")
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=gini_per_round, x="gini", kde=True, ax=ax)
    ax.set_xlabel("Indice de Gini de la puissance de main initiale")
    ax.set_ylabel("Nombre de manches")
    ax.set_title("Répartition de l'indice de Gini des mains initiales")
    _savefig(fig, output_dir, "gini_hand_power_histogram.png")


def _plot_branching_factor_vs_players(manifest_df: pd.DataFrame, output_dir: str) -> None:
    """Facteur de branchement moyen en fonction du nombre de joueurs, par profil d'agent."""
    if manifest_df.empty or "branching_factor_average" not in manifest_df.columns:
        return
    grouped = (
        manifest_df.groupby(["agent_profile", "player_count"])["branching_factor_average"]
        .agg(["mean", "std", "count"]).reset_index()
    )
    grouped["sem"] = grouped["std"].fillna(0) / grouped["count"].clip(lower=1) ** 0.5
    fig, ax = plt.subplots(figsize=(8, 5))
    for profile, group in grouped.groupby("agent_profile"):
        group = group.sort_values("player_count")
        ax.plot(group["player_count"], group["mean"], marker="o", label=_truncate_label(str(profile), 18))
        ax.fill_between(group["player_count"], group["mean"] - group["sem"], group["mean"] + group["sem"], alpha=0.2)
    ax.set_xlabel("Nombre de joueurs")
    ax.set_ylabel("Facteur de branchement moyen")
    ax.set_title("Facteur de branchement par profil")
    ax.legend(title="Profil", fontsize=8)
    _savefig(fig, output_dir, "branching_factor_vs_player_count.png")


def _plot_combo_size_distribution(action_played_df: pd.DataFrame, output_dir: str) -> None:
    """Distribution des tailles de combinaison jouées, par source."""
    if action_played_df.empty:
        return
    played = action_played_df[action_played_df["action_type"] == "ACTION_PLAY"].copy()
    if played.empty:
        return
    played["combo_size"] = played["cards_played"].apply(len)
    hue_column = "source_label" if "source_label" in played.columns else "source_file"
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.countplot(data=played, x="combo_size", hue=hue_column, ax=ax)
    ax.set_xlabel("Taille de combinaison")
    ax.set_ylabel("Nombre de poses")
    ax.set_title("Distribution des tailles de combinaison jouées")
    ax.legend(title="Source", fontsize=7)
    _savefig(fig, output_dir, "combo_size_distribution.png")


def _plot_actions_per_round_violin(action_played_df: pd.DataFrame, output_dir: str) -> None:
    """Nombre d'actions par manche, par source (violin plot)."""
    if action_played_df.empty:
        return
    group_column = "source_label" if "source_label" in action_played_df.columns else "source_file"
    actions_per_round = (
        action_played_df.groupby([group_column, "game_id", "round_id"]).size().reset_index(name="actions")
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.violinplot(data=actions_per_round, x=group_column, y="actions", ax=ax, inner="quartile")
    ax.set_xlabel("Source")
    ax.set_ylabel("Actions par manche")
    ax.set_title("Nombre d'actions par manche par source")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    _savefig(fig, output_dir, "actions_per_round_violin.png")


def _plot_e_rev_volatility_regression(manifest_df: pd.DataFrame, output_dir: str) -> None:
    """Volatilité de la Révolution en fonction du nombre de joueurs (régression)."""
    if manifest_df.empty or "e_rev_volatility" not in manifest_df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(data=manifest_df, x="player_count", y="e_rev_volatility", ax=ax, scatter_kws={"alpha": 0.4})
    ax.set_xlabel("Nombre de joueurs")
    ax.set_ylabel("Bascules de révolution par manche")
    ax.set_title("Volatilité de la révolution selon le nombre de joueurs")
    _savefig(fig, output_dir, "e_rev_volatility_vs_players.png")


def _plot_opening_position_rank_regression(trick_start_df: pd.DataFrame, finished_df: pd.DataFrame, output_dir: str) -> None:
    """Corrélation entre la position d'ouverture et le rang de sortie (régression)."""
    if trick_start_df.empty or finished_df.empty:
        return
    openers = (
        trick_start_df[trick_start_df["trick_index"] == 0][["game_id", "round_id", "opener_id"]]
        .drop_duplicates(["game_id", "round_id"])
    )
    merged = finished_df.merge(openers, on=["game_id", "round_id"], how="inner")
    if merged.empty:
        return
    correlation = merged["opener_id"].corr(merged["rank"])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.regplot(data=merged, x="opener_id", y="rank", ax=ax, scatter_kws={"alpha": 0.3})
    ax.set_xlabel("Identifiant du joueur ouvrant le premier pli")
    ax.set_ylabel("Rang de sortie")
    ax.set_title(f"Position d'ouverture vs rang de sortie (r = {correlation:.3f})")
    _savefig(fig, output_dir, "opening_position_rank_regression.png")


def _plot_rule_trigger_counts(rule_triggered_df: pd.DataFrame, output_dir: str) -> None:
    """Fréquence des déclenchements de règles avancées."""
    if rule_triggered_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.countplot(data=rule_triggered_df, y="rule_name", order=rule_triggered_df["rule_name"].value_counts().index, ax=ax)
    ax.set_xlabel("Occurrences")
    ax.set_ylabel("Règle déclenchée")
    ax.set_title("Fréquence des règles avancées déclenchées")
    _savefig(fig, output_dir, "rule_trigger_counts.png")


def _plot_training_learning_curves(history_df: pd.DataFrame, output_dir: str) -> None:
    """Courbes d'apprentissage des modèles entraînés, avec bandes d'incertitude."""
    if history_df.empty:
        return
    label_column = "model_label" if "model_label" in history_df.columns else "model_file"
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_label, group in history_df.groupby(label_column):
        group = group.sort_values("round_index")
        window = max(1, len(group) // 100)
        rolling_mean = group["vp"].rolling(window=window, min_periods=1).mean()
        rolling_std = group["vp"].rolling(window=window, min_periods=1).std().fillna(0)
        ax.plot(group["round_index"], rolling_mean, label=model_label)
        ax.fill_between(group["round_index"], rolling_mean - rolling_std, rolling_mean + rolling_std, alpha=0.2)
    ax.set_xlabel("Index de manche d'entraînement")
    ax.set_ylabel("Point de victoire (moyenne glissante)")
    ax.set_title("Courbes d'apprentissage cumulées")
    ax.legend(fontsize=7)
    _savefig(fig, output_dir, "training_learning_curves.png")


def _plot_putsch_efficiency(
    ask_putsch_df: pd.DataFrame, putsch_invoked_df: pd.DataFrame, finished_df: pd.DataFrame, output_dir: str,
) -> None:
    """Efficacité du Putsch : taux de victoire du rôle SCUM sollicité, selon invocation ou non."""
    if ask_putsch_df.empty or finished_df.empty or putsch_invoked_df.empty:
        return
    invoked_rounds = set(zip(putsch_invoked_df["game_id"], putsch_invoked_df["round_id"]))
    scum_by_round = {(row["game_id"], row["round_id"]): row["player_id"] for _, row in ask_putsch_df.iterrows()}
    records = []
    for _, row in finished_df.iterrows():
        key = (row["game_id"], row["round_id"])
        scum_id = scum_by_round.get(key)
        if scum_id is None or row["player_id"] != scum_id:
            continue
        category = "Invoqué" if key in invoked_rounds else "Non invoqué"
        records.append({"category": category, "is_winner": row["rank"] == 0})
    putsch_df = pd.DataFrame(records)
    if putsch_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=putsch_df, x="category", y="is_winner", errorbar=("ci", 95), ax=ax)
    ax.set_xlabel("Décision de Putsch")
    ax.set_ylabel("Taux de victoire du rôle SCUM sollicité")
    ax.set_title("Efficacité du Putsch")
    _savefig(fig, output_dir, "putsch_efficiency_ci.png")


def _plot_missed_interception_rate(
    interception_broadcast_df: pd.DataFrame, interception_resolved_df: pd.DataFrame, output_dir: str,
) -> None:
    """Taux d'interception manquée par source, avec intervalle de confiance."""
    if interception_broadcast_df.empty or interception_resolved_df.empty:
        return
    group_column = "source_label" if "source_label" in interception_broadcast_df.columns else "source_file"
    broadcasts = interception_broadcast_df.sort_values([group_column, "game_id", "round_id", "timestamp"])
    resolutions = interception_resolved_df.sort_values(
        [group_column if group_column in interception_resolved_df.columns else "source_file", "game_id", "round_id", "timestamp"]
    )
    resolve_group_column = group_column if group_column in interception_resolved_df.columns else "source_file"
    merged = pd.merge_asof(
        broadcasts, resolutions[[resolve_group_column, "game_id", "round_id", "timestamp", "interceptor_id"]],
        on="timestamp", by=[resolve_group_column, "game_id", "round_id"], direction="forward",
    )
    merged["missed"] = merged["interceptor_id"].isna()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=merged, x=group_column, y="missed", errorbar=("ci", 95), ax=ax)
    ax.set_xlabel("Source")
    ax.set_ylabel("Taux d'interception manquée (approché)")
    ax.set_title("Interceptions manquées par source")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    _savefig(fig, output_dir, "missed_interception_rate_ci.png")


def _plot_skip_turn_magnitude(rule_triggered_df: pd.DataFrame, output_dir: str) -> None:
    """Distribution du nombre de joueurs sautés par déclenchement du Saut de Tour."""
    if rule_triggered_df.empty or "magnitude" not in rule_triggered_df.columns:
        return
    skip_events = rule_triggered_df[rule_triggered_df["rule_name"] == "SKIP_TURN"].dropna(subset=["magnitude"])
    if skip_events.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(data=skip_events, x="magnitude", discrete=True, ax=ax)
    ax.set_xlabel("Nombre de joueurs sautés")
    ax.set_ylabel("Occurrences")
    ax.set_title("Magnitude du Saut de Tour")
    _savefig(fig, output_dir, "skip_turn_magnitude_histogram.png")


def _plot_evaluation_violin(evaluation_df: pd.DataFrame, output_dir: str) -> None:
    """Distribution du point de victoire cumulé par profil évalué (violin plot)."""
    if evaluation_df.empty or "cumulative_vp" not in evaluation_df.columns:
        return
    label_column = "profile_label" if "profile_label" in evaluation_df.columns else "profile"
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.violinplot(data=evaluation_df, x=label_column, y="cumulative_vp", ax=ax, inner="quartile")
    ax.set_xlabel("Profil évalué")
    ax.set_ylabel("Point de victoire cumulé par partie")
    ax.set_title("Distribution du VP cumulé par profil")
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    _savefig(fig, output_dir, "evaluation_vp_violin.png")


def _plot_evaluation_president_rate(evaluation_df: pd.DataFrame, output_dir: str) -> None:
    """Taux de manches présidentielles par profil évalué, avec intervalle de confiance."""
    if evaluation_df.empty or "president_rounds" not in evaluation_df.columns or "rounds_per_game" not in evaluation_df.columns:
        return
    evaluation_df = evaluation_df.copy()
    evaluation_df["president_rate"] = evaluation_df["president_rounds"] / evaluation_df["rounds_per_game"].clip(lower=1)
    label_column = "profile_label" if "profile_label" in evaluation_df.columns else "profile"
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=evaluation_df, x=label_column, y="president_rate", errorbar=("ci", 95), ax=ax)
    ax.set_xlabel("Profil évalué")
    ax.set_ylabel("Taux de manches terminées au rôle PRESIDENT")
    ax.set_title("Taux de manches au rôle PRESIDENT par profil")
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    _savefig(fig, output_dir, "evaluation_president_rate_ci.png")


def _plot_combo_power_bubble(action_played_df: pd.DataFrame, output_dir: str) -> None:
    """Occurrences de combinaisons jouées par taille et puissance résultante (bubble plot interactif)."""
    if action_played_df.empty or "resulting_power" not in action_played_df.columns:
        return
    played = action_played_df[action_played_df["action_type"] == "ACTION_PLAY"].copy()
    if played.empty:
        return
    played["combo_size"] = played["cards_played"].apply(len)
    bubble_data = played.groupby(["combo_size", "resulting_power"]).size().reset_index(name="count")
    if bubble_data.empty:
        return
    fig = px.scatter(
        bubble_data, x="combo_size", y="resulting_power", size="count", color="count",
        color_continuous_scale="Turbo",
        labels={"combo_size": "Taille de la combinaison", "resulting_power": "Puissance résultante", "count": "Occurrences"},
        title="Occurrences de combinaisons jouées par taille et puissance résultante",
    )
    fig.update_traces(marker=dict(line=dict(width=1.2, color="rgba(20,20,20,0.85)")))
    fig.update_layout(
        template="plotly_white",
        title_font=dict(size=20, color="#1a1a1a"),
        font=dict(size=14, color="#1a1a1a"),
        plot_bgcolor="#f4f4f6",
        paper_bgcolor="#ffffff",
        coloraxis_colorbar=dict(title="Occurrences", tickfont=dict(size=12)),
        xaxis=dict(gridcolor="#d8d8dc", zerolinecolor="#b0b0b6"),
        yaxis=dict(gridcolor="#d8d8dc", zerolinecolor="#b0b0b6"),
        margin=dict(l=60, r=30, t=70, b=60),
    )
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, "combo_power_bubble.html"))


def _next_figure_version(output_root: str) -> int:
    """
    Détermine le prochain numéro de version de graphiques à utiliser.

    Paramètre `output_root` : répertoire racine des figures (typiquement `figures/`).
    Retourne un entier strictement positif, supérieur de un au plus grand numéro de version `vN` déjà présent sous `output_root`, ou `1`
    si aucune version n'existe encore. Ce mécanisme garantit qu'un nouvel appel à `generate_all` ne recouvre jamais les figures produites
    par un appel précédent. Aucun effet de bord.
    """
    if not os.path.isdir(output_root):
        return 1
    existing_versions = []
    for entry in os.listdir(output_root):
        match = re.fullmatch(r"v(\d+)", entry)
        if match:
            existing_versions.append(int(match.group(1)))
    return (max(existing_versions) + 1) if existing_versions else 1


def generate_all(output_root: str = FIGURE_DIR) -> int:
    """
    Génère l'ensemble des graphiques d'analyse disponibles à partir du contenu courant de `data/` et `weights/`.

    Paramètre `output_root` : répertoire racine dans lequel écrire les figures, versionné en interne.
    Retourne le numéro de version sous lequel les figures ont été écrites (`output_root/v<N>/`). Effet de bord : lit l'ensemble des
    fichiers Parquet/CSV disponibles, écrit les figures produites dans un nouveau sous-répertoire versionné, sans jamais recouvrir une
    version antérieure, et met à jour le pointeur de convenance `output_root/LATEST_VERSION.txt`. Chaque graphique individuel est protégé
    par une garde d'absence de données ; l'absence d'une source de données n'empêche pas la génération des autres graphiques.
    """
    os.makedirs(output_root, exist_ok=True)
    version = _next_figure_version(output_root)
    version_dir = os.path.join(output_root, f"v{version}")
    os.makedirs(version_dir, exist_ok=True)

    events_df = _load_segmented_parquet()
    finished_df = _expand_payload(events_df, "EventPlayerFinished")
    round_start_df = _expand_payload(events_df, "EventRoundStart")
    action_played_df = _expand_payload(events_df, "EventActionPlayed")
    trick_start_df = _expand_payload(events_df, "EventTrickStart")
    round_end_df = _expand_payload(events_df, "EventRoundEnd")
    rule_triggered_df = _expand_payload(events_df, "EventRuleTriggered")
    ask_putsch_df = _expand_payload(events_df, "EventAskPutsch")
    putsch_invoked_df = _expand_payload(events_df, "EventPutschInvoked")
    interception_broadcast_df = _expand_payload(events_df, "EventInterceptionBroadcast")
    interception_resolved_df = _expand_payload(events_df, "EventInterceptionResolved")

    if not action_played_df.empty and "source_file" in action_played_df.columns:
        action_played_df = action_played_df.copy()
        action_played_df["source_label"] = action_played_df["source_file"].apply(_short_source_label)
    if not interception_broadcast_df.empty and "source_file" in interception_broadcast_df.columns:
        interception_broadcast_df = interception_broadcast_df.copy()
        interception_broadcast_df["source_label"] = interception_broadcast_df["source_file"].apply(_short_source_label)
    if not interception_resolved_df.empty and "source_file" in interception_resolved_df.columns:
        interception_resolved_df = interception_resolved_df.copy()
        interception_resolved_df["source_label"] = interception_resolved_df["source_file"].apply(_short_source_label)

    summary_df = _load_summary_csvs()
    manifest_df = _load_grid_manifests()
    history_df = _load_training_histories()
    if not history_df.empty and "model_file" in history_df.columns:
        history_df = history_df.copy()
        history_df["model_label"] = history_df["model_file"].apply(_short_model_label)
    evaluation_df = _load_evaluation_csvs()
    if not evaluation_df.empty and "profile" in evaluation_df.columns:
        evaluation_df = evaluation_df.copy()
        evaluation_df["profile_label"] = evaluation_df["profile"].apply(lambda p: _truncate_label(str(p), 18))
    lr_sweep_df = _load_learning_rate_sweep()

    _plot_vp_by_rank_violin(finished_df, version_dir)
    _plot_win_rate_by_player(finished_df, version_dir)
    _plot_suboptimal_pass_rate(action_played_df, version_dir)
    _plot_role_transition_heatmap(_role_transition_matrix(round_end_df), version_dir)
    _plot_gini_histogram(round_start_df, version_dir)
    _plot_branching_factor_vs_players(manifest_df if not manifest_df.empty else summary_df, version_dir)
    _plot_combo_size_distribution(action_played_df, version_dir)
    _plot_actions_per_round_violin(action_played_df, version_dir)
    _plot_e_rev_volatility_regression(manifest_df if not manifest_df.empty else summary_df, version_dir)
    _plot_opening_position_rank_regression(trick_start_df, finished_df, version_dir)
    _plot_rule_trigger_counts(rule_triggered_df, version_dir)
    _plot_training_learning_curves(history_df, version_dir)
    _plot_putsch_efficiency(ask_putsch_df, putsch_invoked_df, finished_df, version_dir)
    _plot_missed_interception_rate(interception_broadcast_df, interception_resolved_df, version_dir)
    _plot_skip_turn_magnitude(rule_triggered_df, version_dir)
    _plot_evaluation_violin(evaluation_df, version_dir)
    _plot_evaluation_president_rate(evaluation_df, version_dir)
    _plot_combo_power_bubble(action_played_df, version_dir)
    _plot_learning_rate_sweep(lr_sweep_df, version_dir)

    with open(os.path.join(output_root, "LATEST_VERSION.txt"), "w", encoding="utf-8") as handle:
        handle.write(str(version))

    print(f"Graphiques générés dans {version_dir}/ (données manquantes ignorées silencieusement graphique par graphique).")
    return version


if __name__ == "__main__":
    generate_all()
