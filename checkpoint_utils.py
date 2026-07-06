"""
Module d'utilitaires génériques de reprise après interruption brutale.

Le module fournit trois briques indépendantes utilisables par n'importe quel script long du projet :
* `GracefulKiller`, qui capture SIGINT/SIGTERM et expose un indicateur `should_stop` consultable dans une boucle, permettant d'achever
  proprement l'itération courante puis de sauvegarder l'état avant de quitter, plutôt que d'être interrompu au milieu d'une écriture ;
* `atomic_write_json`/`load_json`, qui garantissent qu'un fichier d'état JSON n'est jamais laissé à moitié écrit ou corrompu par une
  interruption survenant pendant l'écriture ;
* `PeriodicCheckpoint`, qui déclenche une fonction de sauvegarde à intervalle régulier (temps ou nombre de pas) au sein d'une boucle longue.

Le module ne dépend que de la bibliothèque standard.
"""

from __future__ import annotations

import json
import os
import signal
import tempfile
import time
from typing import Any, Callable, Optional


class GracefulKiller:
    """
    Capture SIGINT/SIGTERM et expose un indicateur `should_stop` consultable dans une boucle longue.

    Champ `should_stop` : booléen mutable, faux jusqu'à réception d'un premier signal d'interruption.
    """

    def __init__(self) -> None:
        self.should_stop = False
        self._second_signal_received = False
        try:
            signal.signal(signal.SIGINT, self._handle)
            signal.signal(signal.SIGTERM, self._handle)
        except (ValueError, OSError):
            # L'enregistrement de gestionnaires de signaux n'est possible que dans le thread principal ; sur certaines plateformes ou
            # depuis un thread secondaire, l'appel échoue silencieusement et l'indicateur reste inutilisé.
            pass

    def _handle(self, signum, frame) -> None:
        if self.should_stop and not self._second_signal_received:
            self._second_signal_received = True
            raise KeyboardInterrupt("Arrêt forcé après une seconde interruption.")
        self.should_stop = True


def atomic_write_json(path: str, data: Any) -> None:
    """
    Écrit un objet JSON de façon atomique.

    Paramètre `path` : chemin de destination.
    Paramètre `data` : objet sérialisable en JSON.
    Retourne `None`. Effet de bord : écrit dans un fichier temporaire du même répertoire puis le renomme vers `path` par une opération
    atomique du système de fichiers, garantissant qu'une interruption en cours d'écriture ne peut jamais laisser `path` dans un état
    partiellement écrit.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp_checkpoint_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_json(path: str, default: Any = None) -> Any:
    """
    Charge un objet JSON depuis le disque, avec repli sur une valeur par défaut.

    Paramètre `path` : chemin du fichier à lire.
    Paramètre `default` : valeur retournée si le fichier est absent ou corrompu.
    Retourne l'objet désérialisé, ou `default`. Aucun effet de bord.
    """
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return default


class PeriodicCheckpoint:
    """
    Déclenche une sauvegarde de point de reprise à intervalle régulier au sein d'une boucle longue.

    Champ `save_fn` : fonction sans argument exécutant la sauvegarde effective.
    Champ `every_seconds` : intervalle minimal en secondes entre deux sauvegardes.
    Champ `every_steps` : intervalle minimal en nombre de pas entre deux sauvegardes, optionnel, en complément du critère temporel.
    """

    def __init__(self, save_fn: Callable[[], None], every_seconds: float = 30.0, every_steps: Optional[int] = None) -> None:
        self.save_fn = save_fn
        self.every_seconds = every_seconds
        self.every_steps = every_steps
        self._last_save_time = time.time()
        self._last_save_step = 0

    def maybe_save(self, current_step: int, force: bool = False) -> bool:
        """
        Déclenche la sauvegarde si l'intervalle configuré est atteint, ou si `force` est vrai.

        Paramètre `current_step` : compteur de pas courant de la boucle appelante.
        Paramètre `force` : force la sauvegarde indépendamment des intervalles configurés.
        Retourne un booléen indiquant si la sauvegarde a effectivement eu lieu. Effet de bord : invoque `save_fn` le cas échéant.
        """
        due_time = (time.time() - self._last_save_time) >= self.every_seconds
        due_steps = self.every_steps is not None and (current_step - self._last_save_step) >= self.every_steps
        if force or due_time or due_steps:
            self.save_fn()
            self._last_save_time = time.time()
            self._last_save_step = current_step
            return True
        return False
