"""
Module du serveur web de l'interface de jeu interactive contre les bots.

Le module implémente un serveur Flask servant à la fois l'interface statique (HTML/CSS/JS) et l'API REST permettant de créer une partie,
d'interroger son état courant, et de soumettre les décisions des sièges humains. Chaque partie s'exécute dans un thread dédié géré par
`webapp.game_session.GameSession` ; le serveur lui-même ne bloque jamais sur l'exécution d'une manche.

Le module dépend de `flask`, `core.config`, `registry.agent_registry` et `webapp.game_session`.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request, send_from_directory

from core.config import GameConfig
from registry.agent_registry import ALL_AUTOMATED_PROFILES
from webapp.game_session import GameSession

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = Flask(__name__, static_folder=None)

_sessions: Dict[str, GameSession] = {}
_sessions_lock = threading.Lock()

# Durée d'inactivité au-delà de laquelle une session est éligible au nettoyage automatique.
_SESSION_TTL_SECONDS = 3 * 60 * 60

# Profils de bots proposés à l'interface, excluant les profils entraînables qui nécessitent des poids externes.
_BOT_PROFILES = [
    profile for profile in ALL_AUTOMATED_PROFILES
    if profile not in ("rl_agent", "torch_rl_agent")
]

_BOT_LABELS = {
    "random_bot": "Aléatoire",
    "greedy_bot": "Glouton",
    "aggressive_bot": "Agressif",
    "rule_based_bot": "Basé sur des règles",
    "lookahead_bot": "Anticipation",
    "adaptive_bot": "Adaptatif",
    "scoring_bot": "Score composite",
    "probabilistic_bot": "Probabiliste",
    "mcts_bot": "Monte-Carlo (lent)",
}

_ALLOWED_RULE_FIELDS = {
    "pass_type", "vp_distribution_type", "use_jokers", "magic_two",
    "magic_two_single_clears_all", "magic_card_enabled", "magic_card_rank",
    "magic_single_clears_all", "skip_on_equal", "revolution_enabled",
    "double_revolution_enabled", "straights_enabled", "skip_turn_enabled",
    "skip_turn_rank", "interception_enabled", "interception_closes_trick",
    "putsch_enabled", "blind_tax_enabled", "strict_remainder_allocation",
    "strict_remainder_role", "finish_penalty_enabled", "finish_penalty_type",
    "finish_penalty_draw_count", "finish_penalty_extended", "no_finish_on_joker",
    "no_finish_on_revolution", "first_trick_opener_id",
}


def _cleanup_sessions() -> None:
    """
    Supprime les sessions inactives depuis plus de `_SESSION_TTL_SECONDS`.

    Retourne `None`. Effet de bord : retire les sessions expirées du registre en mémoire.
    """
    now = time.time()
    with _sessions_lock:
        stale = [
            sid for sid, session in _sessions.items()
            if now - session.last_activity_at > _SESSION_TTL_SECONDS
        ]
        for sid in stale:
            del _sessions[sid]


def _get_session(session_id: str) -> Optional[GameSession]:
    """
    Récupère une session par son identifiant et met à jour son horodatage d'activité.

    Paramètre `session_id` : identifiant de la session.
    Retourne l'instance de `GameSession`, ou `None` si l'identifiant est inconnu. Effet de bord : appelle `touch()` sur la session trouvée.
    """
    with _sessions_lock:
        session = _sessions.get(session_id)
    if session is not None:
        session.touch()
    return session


@app.route("/")
def index() -> Any:
    return send_from_directory(_STATIC_DIR, "index.html")


@app.route("/static/<path:filename>")
def static_files(filename: str) -> Any:
    return send_from_directory(_STATIC_DIR, filename)


@app.route("/api/bots", methods=["GET"])
def list_bots() -> Any:
    bots = [
        {"id": profile, "label": _BOT_LABELS.get(profile, profile)}
        for profile in _BOT_PROFILES
    ]
    return jsonify({"bots": bots})


@app.route("/api/games", methods=["POST"])
def create_game() -> Any:
    _cleanup_sessions()
    payload = request.get_json(force=True, silent=True) or {}

    try:
        player_count = int(payload.get("player_count", 4))
    except (TypeError, ValueError):
        return jsonify({"error": "player_count invalide."}), 400
    if player_count < 3:
        return jsonify({"error": "player_count doit être au moins 3."}), 400

    seats = payload.get("seats")
    if not isinstance(seats, list) or len(seats) != player_count:
        return jsonify({"error": f"seats doit contenir exactement {player_count} entrée(s)."}), 400

    valid_seat_values = set(_BOT_PROFILES) | {"human"}
    for seat in seats:
        if seat not in valid_seat_values:
            return jsonify({"error": f"Profil de siège inconnu : '{seat}'."}), 400

    try:
        rounds = int(payload.get("rounds", 10))
    except (TypeError, ValueError):
        return jsonify({"error": "rounds invalide."}), 400
    if rounds < 1:
        return jsonify({"error": "rounds doit être au moins 1."}), 400

    reveal_hands = bool(payload.get("reveal_hands", False))
    rules = payload.get("rules") or {}
    if not isinstance(rules, dict):
        return jsonify({"error": "rules doit être un objet."}), 400
    seed = payload.get("seed")

    config_kwargs: Dict[str, Any] = {
        "player_count": player_count,
        "random_seed": int(seed) if isinstance(seed, (int, float)) else int(time.time()),
    }
    for key, value in rules.items():
        if key in _ALLOWED_RULE_FIELDS:
            config_kwargs[key] = value

    try:
        config = GameConfig(**config_kwargs)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except TypeError as error:
        return jsonify({"error": f"Paramètre de règle invalide : {error}"}), 400

    session_id = uuid.uuid4().hex
    session = GameSession(session_id, config, seats, reveal_hands)
    with _sessions_lock:
        _sessions[session_id] = session
    session.start(rounds)

    return jsonify({"game_id": session_id, "human_seats": session.human_seats})


@app.route("/api/games/<game_id>/state", methods=["GET"])
def get_state(game_id: str) -> Any:
    session = _get_session(game_id)
    if session is None:
        return jsonify({"error": "Partie introuvable."}), 404
    return jsonify(session.snapshot())


def _handle_seat_response(game_id: str, player_id: int) -> Any:
    session = _get_session(game_id)
    if session is None:
        return jsonify({"error": "Partie introuvable."}), 404
    payload = request.get_json(force=True, silent=True) or {}
    accepted = session.submit_response(player_id, payload)
    if not accepted:
        return jsonify({"error": "Aucune décision en attente pour ce siège."}), 409
    return jsonify({"ok": True})


@app.route("/api/games/<game_id>/seats/<int:player_id>/action", methods=["POST"])
def submit_action(game_id: str, player_id: int) -> Any:
    return _handle_seat_response(game_id, player_id)


@app.route("/api/games/<game_id>/seats/<int:player_id>/exchange", methods=["POST"])
def submit_exchange(game_id: str, player_id: int) -> Any:
    return _handle_seat_response(game_id, player_id)


@app.route("/api/games/<game_id>/seats/<int:player_id>/putsch", methods=["POST"])
def submit_putsch(game_id: str, player_id: int) -> Any:
    return _handle_seat_response(game_id, player_id)


@app.route("/api/games/<game_id>/seats/<int:player_id>/interception", methods=["POST"])
def submit_interception(game_id: str, player_id: int) -> Any:
    return _handle_seat_response(game_id, player_id)


@app.route("/api/games/<game_id>", methods=["DELETE"])
def delete_game(game_id: str) -> Any:
    with _sessions_lock:
        _sessions.pop(game_id, None)
    return jsonify({"ok": True})


def main() -> None:
    """
    Point d'entrée en ligne de commande du serveur web.

    Retourne `None`. Effet de bord : démarre le serveur Flask en mode multi-thread, requis pour qu'une partie en cours d'exécution dans
    son propre thread n'empêche pas le traitement des requêtes HTTP concurrentes.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Serveur de l'interface web interactive du Président")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
