"""
Module de nomenclature des fichiers générés par les campagnes d'entraînement et de recherche.

Le module centralise la construction des noms de fichiers de poids et de résultats de recherche, ainsi que la lecture/écriture de leurs métadonnées
associées, afin que tout script producteur de ce type de fichier suive la même convention de nommage et le même emplacement de destination.
"""

from __future__ import annotations

import datetime
import json
import os
from typing import Any, Dict, Optional

WEIGHTS_DIR = "weights"
DATA_DIR = "data"


def ensure_dir(path: str) -> None:
    """
    Garantit l'existence d'un répertoire de destination.

    Paramètre `path` : chemin du répertoire à créer si nécessaire.
    Retourne `None`. Effet de bord : crée le répertoire et ses parents si absents.
    """
    os.makedirs(path, exist_ok=True)


def _timestamp(date: Optional[datetime.date] = None) -> str:
    """
    Construit l'horodatage à date fixe utilisé dans les noms de fichiers.

    Paramètre `date` : date à utiliser, date du jour si `None`.
    Retourne une chaîne au format `AAAAMMJJ`. Aucun effet de bord.
    """
    resolved = date or datetime.date.today()
    return resolved.strftime("%Y%m%d")


def _sanitize_number(value: float) -> str:
    """
    Convertit un nombre en un jeton utilisable dans un nom de fichier.

    Paramètre `value` : nombre à convertir.
    Retourne une chaîne sans point ni signe moins, ces caractères étant remplacés respectivement par `p` et `m`. Aucun effet de bord.
    """
    return f"{value}".replace(".", "p").replace("-", "m")


def build_weights_filename(
    model_name: str,
    player_count: int,
    learning_rate: float,
    rounds: int,
    extension: str = "npy",
    date: Optional[datetime.date] = None,
    directory: str = WEIGHTS_DIR,
) -> str:
    """
    Construit le chemin d'un fichier de poids nommé selon la convention du projet.

    Paramètre `model_name` : nom du modèle entraîné.
    Paramètre `player_count` : nombre de joueurs de la configuration d'entraînement.
    Paramètre `learning_rate` : taux d'apprentissage utilisé.
    Paramètre `rounds` : nombre total de manches ou d'étapes d'entraînement cumulées.
    Paramètre `extension` : extension du fichier, sans point.
    Paramètre `date` : date à inscrire dans le nom, date du jour si `None`.
    Paramètre `directory` : répertoire de destination, créé si absent.
    Retourne un chemin de la forme `<directory>/<model_name>_player<N>_learnRate<LR>_rounds<R>_<date>.<extension>`.
    Effet de bord : crée `directory` si absent.
    """
    ensure_dir(directory)
    stamp = _timestamp(date)
    lr_token = _sanitize_number(learning_rate)
    filename = f"{model_name}_player{player_count}_learnRate{lr_token}_rounds{rounds}_{stamp}.{extension}"
    return os.path.join(directory, filename)


def build_weights_metadata_filename(weights_path: str) -> str:
    """
    Détermine le chemin du fichier de métadonnées associé à un fichier de poids.

    Paramètre `weights_path` : chemin du fichier de poids.
    Retourne le même chemin, extension remplacée par `.meta.json`. Aucun effet de bord.
    """
    base, _ext = os.path.splitext(weights_path)
    return f"{base}.meta.json"


def write_weights_metadata(weights_path: str, metadata: Dict[str, Any]) -> None:
    """
    Écrit les métadonnées d'un fichier de poids sur disque.

    Paramètre `weights_path` : chemin du fichier de poids concerné.
    Paramètre `metadata` : dictionnaire de métadonnées sérialisable en JSON.
    Retourne `None`. Effet de bord : écrit un fichier JSON à côté du fichier de poids.
    """
    meta_path = build_weights_metadata_filename(weights_path)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def read_weights_metadata(weights_path: str) -> Optional[Dict[str, Any]]:
    """
    Lit les métadonnées d'un fichier de poids si elles existent.

    Paramètre `weights_path` : chemin du fichier de poids concerné.
    Retourne un dictionnaire désérialisé depuis le fichier de métadonnées associé, ou `None` si ce fichier n'existe pas. Aucun effet de bord
    hors lecture disque.
    """
    meta_path = build_weights_metadata_filename(weights_path)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_research_filename(
    experiment_name: str,
    player_count: int,
    agent_profile: str,
    games: int,
    rounds_per_game: int,
    extension: str = "parquet",
    date: Optional[datetime.date] = None,
    directory: str = DATA_DIR,
) -> str:
    """
    Construit le chemin d'un fichier de résultats de recherche nommé selon la convention du projet.

    Paramètre `experiment_name` : nom de la campagne ou de la combinaison d'expérience.
    Paramètre `player_count` : nombre de joueurs utilisé pour la campagne.
    Paramètre `agent_profile` : profil d'agent appliqué à la campagne.
    Paramètre `games` : nombre total de parties simulées.
    Paramètre `rounds_per_game` : nombre de manches jouées par partie.
    Paramètre `extension` : extension du fichier, sans point.
    Paramètre `date` : date à inscrire dans le nom, date du jour si `None`.
    Paramètre `directory` : répertoire de destination, créé si absent.
    Retourne un chemin de la forme `<directory>/<experiment_name>_player<N>_agent<profile>_games<G>_rounds<R>_<date>.<extension>`.
    Effet de bord : crée `directory` si absent.
    """
    ensure_dir(directory)
    stamp = _timestamp(date)
    filename = (
        f"{experiment_name}_player{player_count}_agent{agent_profile}_"
        f"games{games}_rounds{rounds_per_game}_{stamp}.{extension}"
    )
    return os.path.join(directory, filename)


def build_grid_manifest_filename(
    experiment_name: str,
    extension: str = "csv",
    date: Optional[datetime.date] = None,
    directory: str = DATA_DIR,
) -> str:
    """
    Construit le chemin du fichier manifeste agrégeant une recherche combinatoire.

    Paramètre `experiment_name` : nom de la campagne combinatoire.
    Paramètre `extension` : extension du fichier, sans point.
    Paramètre `date` : date à inscrire dans le nom, date du jour si `None`.
    Paramètre `directory` : répertoire de destination, créé si absent.
    Retourne un chemin de la forme `<directory>/<experiment_name>_manifest_<date>.<extension>`.
    Effet de bord : crée `directory` si absent.
    """
    ensure_dir(directory)
    stamp = _timestamp(date)
    filename = f"{experiment_name}_manifest_{stamp}.{extension}"
    return os.path.join(directory, filename)
