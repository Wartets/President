"""
Module de thème console partagé.

Fournit des constantes de style et des fonctions utilitaires pour un affichage console cohérent et coloré (identification des joueurs, des
rôles, des étapes de campagne, des niveaux de message). N'a aucune dépendance vers le reste du projet afin de pouvoir être importé depuis
n'importe quel point d'entrée sans risque de cycle d'import.
"""

from __future__ import annotations

# Palette cyclique attribuée aux identifiants de joueurs, dans l'ordre de leur premier usage.
PLAYER_STYLES = [
    "bright_cyan",
    "bright_magenta",
    "bright_yellow",
    "bright_green",
    "bright_blue",
    "bright_red",
    "turquoise2",
    "orchid1",
    "gold1",
    "spring_green2",
]

STYLE_ERROR = "bold red"
STYLE_WARNING = "bold dark_orange"
STYLE_SUCCESS = "bold green3"
STYLE_INFO = "bold cyan"
STYLE_STEP = "bold magenta"
STYLE_CAMPAIGN = "bold blue"
STYLE_MUTED = "grey62"
STYLE_HIGHLIGHT = "bold white on grey23"

_ROLE_STYLES = {
    "ROLE_PRESIDENT": "bold gold1",
    "ROLE_VICE_PRESIDENT": "bold khaki1",
    "ROLE_NEUTRAL": "bold grey70",
    "ROLE_VICE_SCUM": "bold dark_orange",
    "ROLE_SCUM": "bold red3",
}


def player_style(player_id: int) -> str:
    """
    Retourne le style rich associé à un identifiant de joueur.

    Paramètre `player_id` : identifiant du joueur.
    Retourne une chaîne de style rich, cyclique sur `PLAYER_STYLES`. Aucun effet de bord.
    """
    return PLAYER_STYLES[player_id % len(PLAYER_STYLES)]


def player_tag(player_id: int, label: str = "Joueur") -> str:
    """
    Construit une balise rich colorée identifiant un joueur.

    Paramètre `player_id` : identifiant du joueur.
    Paramètre `label` : préfixe textuel, `"Joueur"` par défaut.
    Retourne une chaîne prête à être affichée via `rich.console.Console.print`. Aucun effet de bord.
    """
    style = player_style(player_id)
    return f"[{style}]{label} {player_id}[/{style}]"


def role_style(role: str) -> str:
    """
    Retourne le style rich associé à un rôle de fin de manche.

    Paramètre `role` : chaîne de rôle.
    Retourne une chaîne de style rich, `"white"` si le rôle est inconnu. Aucun effet de bord.
    """
    return _ROLE_STYLES.get(role, "white")


def role_tag(role: str) -> str:
    """
    Construit une balise rich colorée identifiant un rôle de fin de manche.

    Paramètre `role` : chaîne de rôle.
    Retourne une chaîne prête à être affichée via `rich.console.Console.print`. Aucun effet de bord.
    """
    style = role_style(role)
    return f"[{style}]{role}[/{style}]"


def error_text(message: str) -> str:
    """Encadre un message d'erreur avec le style rouge standard."""
    return f"[{STYLE_ERROR}]{message}[/{STYLE_ERROR}]"


def warning_text(message: str) -> str:
    """Encadre un message d'avertissement avec le style orange standard."""
    return f"[{STYLE_WARNING}]{message}[/{STYLE_WARNING}]"


def success_text(message: str) -> str:
    """Encadre un message de succès avec le style vert standard."""
    return f"[{STYLE_SUCCESS}]{message}[/{STYLE_SUCCESS}]"


def info_text(message: str) -> str:
    """Encadre un message informatif avec le style cyan standard."""
    return f"[{STYLE_INFO}]{message}[/{STYLE_INFO}]"


def campaign_text(message: str) -> str:
    """Encadre un message de campagne/simulation avec le style bleu standard."""
    return f"[{STYLE_CAMPAIGN}]{message}[/{STYLE_CAMPAIGN}]"


def usage_style(percent: float) -> str:
    """
    Détermine un style d'alerte croissant pour un pourcentage d'utilisation de ressource.

    Paramètre `percent` : valeur d'utilisation, domaine `[0, 100]`.
    Retourne `"green3"` sous 60%, `"dark_orange"` sous 85%, `"bold red"` au-delà. Aucun effet de bord.
    """
    if percent < 60:
        return "green3"
    if percent < 85:
        return "dark_orange"
    return "bold red"
