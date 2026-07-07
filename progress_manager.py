"""
Module de gestion de barres de progression multiples pour les campagnes longues.

Le module fournit `ProgressManager`, un gestionnaire encapsulant plusieurs barres `rich.progress.Progress` partageant un unique
`rich.live.Live`, accompagnées d'un panneau de journal persistant affiché au-dessus des barres. Contrairement à un usage naïf de `rich`
(impression répétée d'une nouvelle barre à chaque itération, sans nettoyage), ce module garantit un rendu stable, empile les messages de
journal dans un panneau borné plutôt que de les faire défiler indéfiniment, et limite la fréquence effective de rafraîchissement par tâche
(à la fois en temps et en nombre de pas) afin qu'une boucle chaude ne soit jamais ralentie par un rafraîchissement d'affichage trop fréquent.

Le module dépend uniquement de `rich` et de la bibliothèque standard.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskID, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)


@dataclass
class _TaskThrottle:
    """
    État de limitation de fréquence de rafraîchissement d'une tâche individuelle.

    Champ `min_interval_seconds` : intervalle minimal en secondes entre deux rafraîchissements effectifs de l'affichage pour cette tâche.
    Champ `min_step_interval` : nombre minimal de pas accomplis entre deux rafraîchissements effectifs, en complément du critère temporel.
    Champ `last_update_time` : horodatage du dernier rafraîchissement effectif.
    Champ `last_update_step` : valeur de progression au moment du dernier rafraîchissement effectif.
    """

    min_interval_seconds: float = 0.15
    min_step_interval: int = 1
    last_update_time: float = 0.0
    last_update_step: float = 0.0


class ProgressManager:
    """
    Gestionnaire de barres de progression multiples avec panneau de journal persistant.

    Champ `console` : instance `rich.console.Console` partagée par l'ensemble de l'affichage.
    Champ `max_summary_lines` : nombre maximal de lignes de journal conservées visibles simultanément dans le panneau supérieur.
    """

    def __init__(self, console: Optional[Console] = None, max_summary_lines: int = 14) -> None:
        self.console = console or Console()
        self.max_summary_lines = max_summary_lines
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
            expand=True,
        )
        self._summary_lines: List[str] = []
        self._throttles: Dict[TaskID, _TaskThrottle] = {}
        self._live: Optional[Live] = None
        self._lock = threading.Lock()

    def _render(self) -> Group:
        # La largeur disponible est requêtée à chaque rendu (plutôt que mise en cache) afin que le panneau s'adapte immédiatement à un
        # redimensionnement du terminal, sans laisser de lignes tronquées ou de bordures désalignées d'un ancien rendu à une largeur différente.
        available_width = max(20, self.console.size.width - 4)
        wrapped_lines = []
        for line in self._summary_lines[-self.max_summary_lines:]:
            wrapped_lines.append(line if len(line) <= available_width else line[: available_width - 1] + "…")
        body = "\n".join(wrapped_lines) or "[grey62]En attente…[/grey62]"
        panel = Panel(body, title="Journal", border_style="grey50", padding=(0, 1), expand=True)
        return Group(panel, self._progress)

    def __enter__(self) -> "ProgressManager":
        self._live = Live(
            self._render(), console=self.console, refresh_per_second=6, transient=False,
            vertical_overflow="crop", auto_refresh=True,
        )
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._live is not None:
            self._live.update(self._render())
            self._live.__exit__(exc_type, exc_value, traceback)
            self._live = None

    def log(self, message: str) -> None:
        """
        Ajoute une ligne au panneau de journal persistant.

        Paramètre `message` : texte à ajouter, éventuellement balisé `rich`.
        Retourne `None`. Effet de bord : ajoute la ligne à `_summary_lines` et rafraîchit immédiatement l'affichage, sans jamais imprimer
        de nouvelle barre ni écraser les barres actives.
        """
        with self._lock:
            self._summary_lines.append(message)
            if self._live is not None:
                self._live.update(self._render())

    def add_task(
        self,
        description: str,
        total: Optional[float] = None,
        min_interval_seconds: float = 0.15,
        min_step_interval: int = 1,
    ) -> TaskID:
        """
        Enregistre une nouvelle barre de progression.

        Paramètre `description` : libellé affiché à gauche de la barre.
        Paramètre `total` : nombre total de pas attendus, `None` pour une barre indéterminée.
        Paramètre `min_interval_seconds` : intervalle minimal entre deux rafraîchissements effectifs pour cette tâche ; à ajuster à la
        hausse pour les boucles très chaudes (des dizaines de milliers d'itérations par seconde) et à la baisse pour les tâches lentes.
        Paramètre `min_step_interval` : nombre minimal de pas entre deux rafraîchissements effectifs.
        Retourne l'identifiant entier de la tâche, à réutiliser pour `advance`/`complete_task`/`remove_task`.
        """
        task_id = self._progress.add_task(description, total=total)
        self._throttles[task_id] = _TaskThrottle(
            min_interval_seconds=min_interval_seconds,
            min_step_interval=min_step_interval,
        )
        if self._live is not None:
            self._live.update(self._render())
        return task_id

    def advance(
        self,
        task_id: TaskID,
        advance: float = 1.0,
        description: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """
        Avance une barre de progression, en respectant sa limitation de fréquence de rafraîchissement.

        Paramètre `task_id` : identifiant retourné par `add_task`.
        Paramètre `advance` : nombre de pas à ajouter à la progression courante.
        Paramètre `description` : nouveau libellé optionnel, mis à jour immédiatement indépendamment de la limitation d'affichage.
        Paramètre `force` : force un rafraîchissement immédiat de l'affichage indépendamment des seuils de limitation.
        Retourne `None`. Effet de bord : met à jour l'état interne de la barre à chaque appel (jamais perdu), mais ne redessine l'écran
        qu'au rythme autorisé par la limitation configurée pour cette tâche.
        """
        if description is not None:
            self._progress.update(task_id, description=description)
        self._progress.advance(task_id, advance)

        throttle = self._throttles.get(task_id)
        should_render = force
        if throttle is not None and not force:
            now = time.time()
            task = next((t for t in self._progress.tasks if t.id == task_id), None)
            completed_now = task.completed if task is not None else 0.0
            elapsed_ok = (now - throttle.last_update_time) >= throttle.min_interval_seconds
            step_ok = (completed_now - throttle.last_update_step) >= throttle.min_step_interval
            should_render = elapsed_ok or step_ok
            if should_render:
                throttle.last_update_time = now
                throttle.last_update_step = completed_now
        if should_render and self._live is not None:
            self._live.update(self._render())

    def complete_task(self, task_id: TaskID, description: Optional[str] = None) -> None:
        """
        Marque une tâche comme entièrement terminée et force un dernier rafraîchissement.

        Paramètre `task_id` : identifiant de la tâche.
        Paramètre `description` : libellé final optionnel.
        Retourne `None`. Effet de bord : complète la barre jusqu'à son total et rafraîchit immédiatement l'affichage.
        """
        task = next((t for t in self._progress.tasks if t.id == task_id), None)
        if task is not None and task.total is not None:
            remaining = task.total - task.completed
            if remaining > 0:
                self._progress.advance(task_id, remaining)
        if description:
            self._progress.update(task_id, description=description)
        if self._live is not None:
            self._live.update(self._render())

    def remove_task(self, task_id: TaskID) -> None:
        """
        Retire définitivement une barre de progression de l'affichage.

        Paramètre `task_id` : identifiant de la tâche à retirer.
        Retourne `None`. Effet de bord : supprime la barre et son état de limitation associé.
        """
        try:
            self._progress.remove_task(task_id)
        except KeyError:
            pass
        self._throttles.pop(task_id, None)
        if self._live is not None:
            self._live.update(self._render())
