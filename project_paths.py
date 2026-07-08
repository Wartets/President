"""
Module d'ancrage des chemins de données du projet.

Le module centralise la résolution des répertoires `data/`, `weights/` et `figures/` en chemins absolus, dérivés de l'emplacement de ce
fichier plutôt que du répertoire de travail courant du processus. Ce choix élimine une classe de bugs silencieux où deux points d'entrée
(par exemple `research.run_pipeline` lancé depuis la racine du dépôt, puis `research.generate_graphs` relancé séparément depuis un autre
répertoire) résolvent des chemins relatifs différents et opèrent chacun sur un `data/`/`figures/` distinct sans qu'aucune erreur ne soit
levée, l'un des deux ne voyant alors jamais les fichiers produits par l'autre.

Tout module du projet qui lit ou écrit dans `data/`, `weights/` ou `figures/` doit utiliser les constantes de ce module plutôt que des
chaînes littérales `"data"`/`"weights"`/`"figures"`.

Aucune dépendance interne n'est requise. Le module ne provoque aucun effet de bord global à l'import ; la création des répertoires est
différée à l'appel explicite de `ensure_all`.
"""

from __future__ import annotations

import os

# Racine du dépôt, dérivée de l'emplacement de ce fichier, indépendante du répertoire de travail courant du processus qui l'importe.
PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))

DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
WEIGHTS_DIR: str = os.path.join(PROJECT_ROOT, "weights")
FIGURE_DIR: str = os.path.join(PROJECT_ROOT, "figures")


def ensure_all() -> None:
    """
    Garantit l'existence des trois répertoires de données du projet.

    Retourne `None`. Effet de bord : crée `DATA_DIR`, `WEIGHTS_DIR` et `FIGURE_DIR` s'ils n'existent pas encore, ainsi que leurs
    répertoires parents.
    """
    for directory in (DATA_DIR, WEIGHTS_DIR, FIGURE_DIR):
        os.makedirs(directory, exist_ok=True)


def resolve(relative_path: str, base: str = DATA_DIR) -> str:
    """
    Résout un chemin relatif par rapport à l'une des racines de données du projet.

    Paramètre `relative_path` : chemin relatif, typiquement un nom de fichier.
    Paramètre `base` : répertoire de base, `DATA_DIR` par défaut.
    Retourne un chemin absolu. Si `relative_path` est déjà absolu, il est retourné inchangé (aucune double résolution). Aucun effet de
    bord.
    """
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(base, relative_path)
