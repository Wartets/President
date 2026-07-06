"""
Module du tableau de bord de suivi en temps réel des campagnes de simulation.

Le module fournit `LiveMonitor`, un composant d'affichage console fondé sur `rich.live.Live`, destiné à visualiser en temps réel le débit de
simulation (parties par seconde), l'utilisation processeur via `psutil` et l'utilisation GPU via `pynvml` lorsque disponible, ainsi qu'un
résumé de la distribution courante des points de victoire observés. Le module ne collecte aucune donnée par lui-même ; il se contente
d'afficher les valeurs qui lui sont transmises par l'appelant à chaque rafraîchissement.

Le module dépend de `rich` pour l'affichage, de `psutil` pour la mesure d'utilisation processeur, et optionnellement de `pynvml` pour la
mesure d'utilisation GPU, ce dernier étant chargé de manière paresseuse et tolérant son absence ou l'absence de périphérique compatible.
"""

from __future__ import annotations

import time
from typing import List, Optional, Any

import psutil
from rich.console import Console
from rich.live import Live
from rich.table import Table

import console_theme

try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False

# Ensure the name exists for type checkers when NVML is not available
pynvml: Any = None


def _try_init_nvml() -> Optional[int]:
    """
    Tente l'initialisation de la bibliothèque NVML et retourne le nombre de périphériques disponibles.

    Retourne un entier positif ou nul, nombre de périphériques GPU détectés, ou `None` si `pynvml` n'est pas installé ou si
    l'initialisation échoue (absence de pilote NVIDIA, absence de périphérique compatible). Effet de bord : initialise l'état global de
    `pynvml` en cas de succès.
    """
    if not _NVML_AVAILABLE:
        return None
    try:
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetCount()
    except Exception:
        return None


class LiveMonitor:
    """
    Tableau de bord console de suivi en temps réel d'une campagne de simulation.

    Champ `console` : instance `rich.console.Console` utilisée pour le rendu.
    Champ `_live` : instance `rich.live.Live` sous-jacente, `None` avant l'entrée dans le contexte.
    Champ `_start_time` : horodatage `time.time()` du démarrage de la campagne, `None` avant le démarrage.
    Champ `_games_completed` : compteur de parties complétées, mis à jour par `record_games`.
    Champ `_reward_samples` : liste bornée des derniers points de victoire observés, utilisée pour le résumé de distribution.
    Champ `_gpu_count` : nombre de périphériques GPU détectés par NVML, `None` si indisponible.
    Champ `_max_reward_samples` : taille maximale de `_reward_samples`, entier strictement positif.
    """

    def __init__(self, console: Optional[Console] = None, max_reward_samples: int = 5000) -> None:
        self.console = console if console is not None else Console()
        self._live: Optional[Live] = None
        self._start_time: Optional[float] = None
        self._games_completed = 0
        self._reward_samples: List[float] = []
        self._gpu_count = _try_init_nvml()
        self._max_reward_samples = max_reward_samples

    def __enter__(self) -> "LiveMonitor":
        """
        Démarre le rendu en temps réel du tableau de bord.

        Retourne l'instance courante. Effet de bord : initialise `_start_time`, instancie et démarre `_live`.
        """
        self._start_time = time.time()
        self._live = Live(self._render(), console=self.console, refresh_per_second=4)
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Arrête le rendu en temps réel du tableau de bord.

        Retourne `None`. Effet de bord : termine `_live` et libère `pynvml` si celui-ci a été initialisé.
        """
        if self._live is not None:
            self._live.__exit__(exc_type, exc_value, traceback)
        if self._gpu_count is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def record_games(self, count: int, rewards: Optional[List[float]] = None) -> None:
        """
        Enregistre l'achèvement d'un lot de parties et rafraîchit l'affichage.

        Paramètre `count` : nombre de parties achevées depuis le dernier appel, entier positif.
        Paramètre `rewards` : liste optionnelle de points de victoire observés sur ce lot, utilisée pour le résumé de distribution.
        Retourne `None`. Effet de bord : incrémente `_games_completed`, complète `_reward_samples` en bornant sa taille à
        `_max_reward_samples`, et rafraîchit `_live` si le tableau de bord est démarré.
        """
        self._games_completed += count
        if rewards:
            self._reward_samples.extend(rewards)
            if len(self._reward_samples) > self._max_reward_samples:
                self._reward_samples = self._reward_samples[-self._max_reward_samples:]
        if self._live is not None:
            self._live.update(self._render())

    def _gpu_usage_percent(self) -> Optional[float]:
        """
        Mesure l'utilisation moyenne des périphériques GPU détectés.

        Retourne un nombre, domaine $[0, 100]$, moyenne de l'utilisation rapportée par NVML sur l'ensemble des périphériques détectés, ou
        `None` si NVML est indisponible ou si aucun périphérique n'a été détecté. Aucun effet de bord hors l'appel à l'API NVML.
        """
        if self._gpu_count is None or self._gpu_count == 0:
            return None
        try:
            total = 0.0
            for index in range(self._gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                total += float(utilization.gpu)
            return total / self._gpu_count
        except Exception:
            return None

    def _render(self) -> Table:
        """
        Construit la représentation tabulaire courante du tableau de bord.

        Retourne une instance de `rich.table.Table` portant les métriques courantes : parties par seconde, utilisation processeur,
        utilisation GPU lorsque disponible, mémoire utilisée, et un résumé de la distribution des points de victoire observés (moyenne et
        écart type approché). Aucun effet de bord hors la lecture des compteurs `psutil`/NVML.
        """
        elapsed = max(time.time() - (self._start_time or time.time()), 1e-9)
        fps = self._games_completed / elapsed

        cpu_percent = psutil.cpu_percent()
        mem_percent = psutil.virtual_memory().percent

        table = Table(title=f"[{console_theme.STYLE_CAMPAIGN}]Campagne de simulation, suivi en temps réel[/{console_theme.STYLE_CAMPAIGN}]")
        table.add_column("Métrique")
        table.add_column("Valeur")
        table.add_row("Parties complétées", f"[bold]{self._games_completed}[/bold]")
        table.add_row("Parties / seconde", f"{fps:.2f}")
        table.add_row("Durée écoulée (s)", f"{elapsed:.1f}")
        table.add_row(
            "Utilisation CPU (%)",
            f"[{console_theme.usage_style(cpu_percent)}]{cpu_percent:.1f}[/{console_theme.usage_style(cpu_percent)}]",
        )
        table.add_row(
            "Mémoire utilisée (%)",
            f"[{console_theme.usage_style(mem_percent)}]{mem_percent:.1f}[/{console_theme.usage_style(mem_percent)}]",
        )

        gpu_usage = self._gpu_usage_percent()
        if gpu_usage is not None:
            gpu_style = console_theme.usage_style(gpu_usage)
            table.add_row("Utilisation GPU (%)", f"[{gpu_style}]{gpu_usage:.1f}[/{gpu_style}]")
        else:
            table.add_row("Utilisation GPU (%)", f"[{console_theme.STYLE_MUTED}]indisponible[/{console_theme.STYLE_MUTED}]")

        if self._reward_samples:
            mean_reward = sum(self._reward_samples) / len(self._reward_samples)
            variance = sum((r - mean_reward) ** 2 for r in self._reward_samples) / len(self._reward_samples)
            table.add_row("VP moyen (échantillon glissant)", f"{mean_reward:.3f}")
            table.add_row("Écart type VP (échantillon glissant)", f"{variance ** 0.5:.3f}")

        return table
