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
    payload_df = pd.json_normalize(payloads)
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


def _savefig(fig, name: str) -> None:
    """Sauvegarde une figure matplotlib dans `figures/` et ferme la figure pour libérer la mémoire."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, name), dpi=150)
    plt.close(fig)


def _plot_vp_by_rank_violin(finished_df: pd.DataFrame) -> None:
    """Distribution des points de victoire par rang de sortie (violin plot)."""
    if finished_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=finished_df, x="rank", y="vp_earned", ax=ax, inner="quartile")
    ax.set_xlabel("Rang de sortie")
    ax.set_ylabel("Points de victoire")
    _savefig(fig, "vp_by_rank_violin.png")


def _plot_win_rate_by_player(finished_df: pd.DataFrame) -> None:
    """Taux de victoire par identifiant de joueur, avec intervalle de confiance."""
    if finished_df.empty:
        return
    finished_df = finished_df.copy()
    finished_df["is_winner"] = finished_df["rank"] == 0
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=finished_df, x="player_id", y="is_winner", errorbar=("ci", 95), ax=ax)
    ax.set_xlabel("Identifiant de joueur")
    ax.set_ylabel("Taux de victoire (rang 0)")
    _savefig(fig, "win_rate_by_player_ci.png")


def _plot_suboptimal_pass_rate(action_played_df: pd.DataFrame) -> None:
    """Taux de passe sous-optimal par identifiant de joueur, avec intervalle de confiance."""
    if action_played_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=action_played_df, x="player_id", y="was_suboptimal", errorbar=("ci", 95), ax=ax)
    ax.set_xlabel("Identifiant de joueur")
    ax.set_ylabel("Taux de passe sous-optimal")
    _savefig(fig, "suboptimal_pass_rate_ci.png")


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


def _plot_role_transition_heatmap(matrix: pd.DataFrame) -> None:
    """Heatmap de la matrice de transition des rôles."""
    if matrix.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax)
    ax.set_xlabel("Rôle manche suivante")
    ax.set_ylabel("Rôle manche courante")
    _savefig(fig, "role_transition_heatmap.png")


def _plot_gini_histogram(round_start_df: pd.DataFrame) -> None:
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
    sns.histplot(gini_per_round["gini"], kde=True, ax=ax)
    ax.set_xlabel("Indice de Gini de la puissance de main initiale")
    ax.set_ylabel("Nombre de manches")
    _savefig(fig, "gini_hand_power_histogram.png")


def _plot_branching_factor_vs_players(manifest_df: pd.DataFrame) -> None:
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
        ax.plot(group["player_count"], group["mean"], marker="o", label=profile)
        ax.fill_between(group["player_count"], group["mean"] - group["sem"], group["mean"] + group["sem"], alpha=0.2)
    ax.set_xlabel("Nombre de joueurs")
    ax.set_ylabel("Facteur de branchement moyen")
    ax.legend(title="Profil d'agent")
    _savefig(fig, "branching_factor_vs_player_count.png")


def _plot_combo_size_distribution(action_played_df: pd.DataFrame) -> None:
    """Distribution des tailles de combinaison jouées, par fichier source."""
    if action_played_df.empty:
        return
    played = action_played_df[action_played_df["action_type"] == "ACTION_PLAY"].copy()
    if played.empty:
        return
    played["combo_size"] = played["cards_played"].apply(len)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.countplot(data=played, x="combo_size", hue="source_file", ax=ax)
    ax.set_xlabel("Taille de combinaison")
    ax.set_ylabel("Nombre de poses")
    _savefig(fig, "combo_size_distribution.png")


def _plot_actions_per_round_violin(action_played_df: pd.DataFrame) -> None:
    """Nombre d'actions par manche, par fichier source (violin plot)."""
    if action_played_df.empty:
        return
    actions_per_round = (
        action_played_df.groupby(["source_file", "game_id", "round_id"]).size().reset_index(name="actions")
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.violinplot(data=actions_per_round, x="source_file", y="actions", ax=ax, inner="quartile")
    ax.set_xlabel("Fichier source")
    ax.set_ylabel("Actions par manche")
    ax.tick_params(axis="x", rotation=45)
    _savefig(fig, "actions_per_round_violin.png")


def _plot_e_rev_volatility_regression(manifest_df: pd.DataFrame) -> None:
    """Volatilité de la Révolution en fonction du nombre de joueurs (régression)."""
    if manifest_df.empty or "e_rev_volatility" not in manifest_df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(data=manifest_df, x="player_count", y="e_rev_volatility", ax=ax, scatter_kws={"alpha": 0.4})
    ax.set_xlabel("Nombre de joueurs")
    ax.set_ylabel("Bascules de révolution par manche")
    _savefig(fig, "e_rev_volatility_vs_players.png")


def _plot_opening_position_rank_regression(trick_start_df: pd.DataFrame, finished_df: pd.DataFrame) -> None:
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
    ax.set_title(f"r = {correlation:.3f}")
    _savefig(fig, "opening_position_rank_regression.png")


def _plot_rule_trigger_counts(rule_triggered_df: pd.DataFrame) -> None:
    """Fréquence des déclenchements de règles avancées."""
    if rule_triggered_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.countplot(data=rule_triggered_df, y="rule_name", order=rule_triggered_df["rule_name"].value_counts().index, ax=ax)
    ax.set_xlabel("Occurrences")
    ax.set_ylabel("Règle déclenchée")
    _savefig(fig, "rule_trigger_counts.png")


def _plot_training_learning_curves(history_df: pd.DataFrame) -> None:
    """Courbes d'apprentissage des modèles entraînés, avec bandes d'incertitude."""
    if history_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_file, group in history_df.groupby("model_file"):
        group = group.sort_values("round_index")
        window = max(1, len(group) // 100)
        rolling_mean = group["vp"].rolling(window=window, min_periods=1).mean()
        rolling_std = group["vp"].rolling(window=window, min_periods=1).std().fillna(0)
        ax.plot(group["round_index"], rolling_mean, label=model_file)
        ax.fill_between(group["round_index"], rolling_mean - rolling_std, rolling_mean + rolling_std, alpha=0.2)
    ax.set_xlabel("Index de manche d'entraînement")
    ax.set_ylabel("Point de victoire (moyenne glissante)")
    ax.legend(fontsize=7)
    _savefig(fig, "training_learning_curves.png")


def _plot_putsch_efficiency(ask_putsch_df: pd.DataFrame, putsch_invoked_df: pd.DataFrame, finished_df: pd.DataFrame) -> None:
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
        category = "Putsch invoqué" if key in invoked_rounds else "Putsch non invoqué"
        records.append({"category": category, "is_winner": row["rank"] == 0})
    putsch_df = pd.DataFrame(records)
    if putsch_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=putsch_df, x="category", y="is_winner", errorbar=("ci", 95), ax=ax)
    ax.set_xlabel("Décision de Putsch")
    ax.set_ylabel("Taux de victoire du rôle SCUM sollicité")
    _savefig(fig, "putsch_efficiency_ci.png")


def _plot_missed_interception_rate(interception_broadcast_df: pd.DataFrame, interception_resolved_df: pd.DataFrame) -> None:
    """Taux d'interception manquée par fichier source, avec intervalle de confiance."""
    if interception_broadcast_df.empty or interception_resolved_df.empty:
        return
    broadcasts = interception_broadcast_df.sort_values(["source_file", "game_id", "round_id", "timestamp"])
    resolutions = interception_resolved_df.sort_values(["source_file", "game_id", "round_id", "timestamp"])
    merged = pd.merge_asof(
        broadcasts, resolutions[["source_file", "game_id", "round_id", "timestamp", "interceptor_id"]],
        on="timestamp", by=["source_file", "game_id", "round_id"], direction="forward",
    )
    merged["missed"] = merged["interceptor_id"].isna()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=merged, x="source_file", y="missed", errorbar=("ci", 95), ax=ax)
    ax.set_xlabel("Fichier source")
    ax.set_ylabel("Taux d'interception manquée (approché)")
    ax.tick_params(axis="x", rotation=45)
    _savefig(fig, "missed_interception_rate_ci.png")


def _plot_skip_turn_magnitude(rule_triggered_df: pd.DataFrame) -> None:
    """Distribution du nombre de joueurs sautés par déclenchement du Saut de Tour."""
    if rule_triggered_df.empty or "magnitude" not in rule_triggered_df.columns:
        return
    skip_events = rule_triggered_df[rule_triggered_df["rule_name"] == "SKIP_TURN"].dropna(subset=["magnitude"])
    if skip_events.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(skip_events["magnitude"], discrete=True, ax=ax)
    ax.set_xlabel("Nombre de joueurs sautés")
    ax.set_ylabel("Occurrences")
    _savefig(fig, "skip_turn_magnitude_histogram.png")


def _plot_evaluation_violin(evaluation_df: pd.DataFrame) -> None:
    """Distribution du point de victoire cumulé par profil évalué (violin plot)."""
    if evaluation_df.empty or "cumulative_vp" not in evaluation_df.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.violinplot(data=evaluation_df, x="profile", y="cumulative_vp", ax=ax, inner="quartile")
    ax.set_xlabel("Profil évalué")
    ax.set_ylabel("Point de victoire cumulé par partie")
    ax.tick_params(axis="x", rotation=30)
    _savefig(fig, "evaluation_vp_violin.png")


def _plot_evaluation_president_rate(evaluation_df: pd.DataFrame) -> None:
    """Taux de manches présidentielles par profil évalué, avec intervalle de confiance."""
    if evaluation_df.empty or "president_rounds" not in evaluation_df.columns or "rounds_per_game" not in evaluation_df.columns:
        return
    evaluation_df = evaluation_df.copy()
    evaluation_df["president_rate"] = evaluation_df["president_rounds"] / evaluation_df["rounds_per_game"].clip(lower=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=evaluation_df, x="profile", y="president_rate", errorbar=("ci", 95), ax=ax)
    ax.set_xlabel("Profil évalué")
    ax.set_ylabel("Taux de manches terminées au rôle PRESIDENT")
    ax.tick_params(axis="x", rotation=30)
    _savefig(fig, "evaluation_president_rate_ci.png")


def _plot_combo_power_bubble(action_played_df: pd.DataFrame) -> None:
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
        labels={"combo_size": "Taille de la combinaison", "resulting_power": "Puissance résultante", "count": "Occurrences"},
    )
    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig.write_html(os.path.join(FIGURE_DIR, "combo_power_bubble.html"))


def generate_all() -> None:
    """
    Génère l'ensemble des graphiques d'analyse disponibles à partir du contenu courant de `data/` et `weights/`.

    Retourne `None`. Effet de bord : lit l'ensemble des fichiers Parquet/CSV disponibles, écrit les figures produites dans `figures/`.
    Chaque graphique individuel est protégé par une garde d'absence de données ; l'absence d'une source de données n'empêche pas la
    génération des autres graphiques.
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)

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

    summary_df = _load_summary_csvs()
    manifest_df = _load_grid_manifests()
    history_df = _load_training_histories()
    evaluation_df = _load_evaluation_csvs()

    _plot_vp_by_rank_violin(finished_df)
    _plot_win_rate_by_player(finished_df)
    _plot_suboptimal_pass_rate(action_played_df)
    _plot_role_transition_heatmap(_role_transition_matrix(round_end_df))
    _plot_gini_histogram(round_start_df)
    _plot_branching_factor_vs_players(manifest_df if not manifest_df.empty else summary_df)
    _plot_combo_size_distribution(action_played_df)
    _plot_actions_per_round_violin(action_played_df)
    _plot_e_rev_volatility_regression(manifest_df if not manifest_df.empty else summary_df)
    _plot_opening_position_rank_regression(trick_start_df, finished_df)
    _plot_rule_trigger_counts(rule_triggered_df)
    _plot_training_learning_curves(history_df)
    _plot_putsch_efficiency(ask_putsch_df, putsch_invoked_df, finished_df)
    _plot_missed_interception_rate(interception_broadcast_df, interception_resolved_df)
    _plot_skip_turn_magnitude(rule_triggered_df)
    _plot_evaluation_violin(evaluation_df)
    _plot_evaluation_president_rate(evaluation_df)
    _plot_combo_power_bubble(action_played_df)

    print(f"Graphiques générés dans {FIGURE_DIR}/ (données manquantes ignorées silencieusement graphique par graphique).")


if __name__ == "__main__":
    generate_all()
