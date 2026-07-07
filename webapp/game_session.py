"""
Module de session de partie interactive pour l'interface web.

Le module définit `GameSession`, qui encapsule une partie complète jouable via l'interface web : construction des agents (humains pilotés
par `agents.web_human_agent.WebHumanAgent`, ou bots automatisés du registre centralisé), exécution des manches dans un thread dédié, et
reconstruction incrémentale d'un état affichable (mains connues, pli en cours, journal d'événements) à partir du flux d'événements publié
par le moteur, sans jamais accéder directement à l'état mutable interne du moteur.

Le module dépend de `agents.web_human_agent`, `core.config`, `engine.event_bus`, `engine.game_runner`, `events.structural`,
`events.transactional` et `registry.agent_registry`.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Any, Dict, List, Optional

from agents.web_human_agent import WebHumanAgent, serialize_card
from core.config import GameConfig
from engine.event_bus import EventBus
from engine.game_runner import Game
from events.structural import (
    EventPlayerFinished, EventRoundEnd, EventRoundStart, EventTrickClosed,
    EventTrickStart,
)
from events.transactional import EventActionPlayed, EventExchange, EventRuleTriggered
from registry.agent_registry import build_agent

# Taille maximale du journal d'événements conservé en mémoire par session.
_MAX_LOG_ENTRIES = 500

# Taille de la fenêtre d'événements récents renvoyée à chaque interrogation d'état.
_LOG_TAIL_SIZE = 40


def _convert_value(value: Any) -> Any:
    """
    Convertit récursivement une valeur d'événement en une forme sérialisable en JSON.

    Paramètre `value` : valeur à convertir, de type quelconque.
    Retourne une valeur composée uniquement de types primitifs, de listes et de dictionnaires. Aucun effet de bord.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return [_convert_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _convert_value(v) for k, v in value.items()}
    if hasattr(value, "rank") and hasattr(value, "suit"):
        return serialize_card(value)
    if hasattr(value, "value") and not dataclasses.is_dataclass(value):
        return value.value
    if dataclasses.is_dataclass(value):
        return {
            field.name: _convert_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    return repr(value)


class GameSession:
    """
    Session d'une partie interactive jouable via l'interface web.

    Champ `session_id` : identifiant unique de la session.
    Champ `config` : configuration de la partie.
    Champ `seat_profiles` : liste ordonnée des profils de siège, `'human'` désignant un siège piloté par l'interface web.
    Champ `reveal_hands` : indique si les mains de tous les joueurs doivent être exposées à l'état affichable.
    Champ `human_seats` : liste des identifiants de siège pilotés par l'interface web.
    """

    def __init__(
        self,
        session_id: str,
        config: GameConfig,
        seat_profiles: List[str],
        reveal_hands: bool,
    ) -> None:
        self.session_id = session_id
        self.config = config
        self.seat_profiles = list(seat_profiles)
        self.reveal_hands = reveal_hands
        self.created_at = time.time()
        self.last_activity_at = time.time()

        self.human_seats: List[int] = [
            pid for pid, profile in enumerate(self.seat_profiles) if profile == "human"
        ]

        self._state_lock = threading.RLock()
        self._event_seq_counter = 0
        self.event_log: List[Dict[str, Any]] = []
        self.known_hands: Dict[int, List[Any]] = {pid: [] for pid in range(config.player_count)}
        self.current_trick_plays: List[Dict[str, Any]] = []
        self.round_index = -1
        self.trick_index = -1
        self.opener_id: Optional[int] = None
        self.last_trick_winner: Optional[int] = None
        self.finished_players: List[int] = []
        self.round_roles: Dict[int, str] = {}
        self.e_rev = False

        self.bus = EventBus()
        self.bus.subscribe(self._on_event)

        self.agents: Dict[int, Any] = {}
        self.human_agents: Dict[int, WebHumanAgent] = {}
        for pid, profile in enumerate(self.seat_profiles):
            if profile == "human":
                agent = WebHumanAgent(pid, config)
                self.agents[pid] = agent
                self.human_agents[pid] = agent
            else:
                self.agents[pid] = build_agent(profile, pid, config, None)

        self.game = Game(config, self.agents, event_bus=self.bus, game_id=session_id)

        self.total_rounds = 0
        self.rounds_played = 0
        self.started = False
        self.finished = False
        self.error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None

    def touch(self) -> None:
        """
        Met à jour l'horodatage de dernière activité, utilisé pour l'expiration des sessions inactives.

        Retourne `None`. Effet de bord : met à jour `last_activity_at`.
        """
        self.last_activity_at = time.time()

    def _on_event(self, event: Any) -> None:
        """
        Reconstruit incrémentalement l'état affichable à partir d'un événement reçu du bus.

        Paramètre `event` : événement publié par le moteur.
        Retourne `None`. Effet de bord : met à jour l'ensemble des champs dérivés de la session (mains connues, pli courant, journal).
        """
        with self._state_lock:
            self._event_seq_counter += 1
            record = self._serialize_event(event)
            record["seq"] = self._event_seq_counter
            self.event_log.append(record)
            if len(self.event_log) > _MAX_LOG_ENTRIES:
                self.event_log = self.event_log[-_MAX_LOG_ENTRIES:]

            if isinstance(event, EventRoundStart):
                self.round_index = event.round_id
                self.trick_index = -1
                self.finished_players = []
                self.current_trick_plays = []
                self.e_rev = False
                for pid, cards in event.initial_hands.items():
                    self.known_hands[pid] = list(cards)
            elif isinstance(event, EventExchange):
                remaining = list(self.known_hands.get(event.from_player, []))
                for card in event.cards:
                    if card in remaining:
                        remaining.remove(card)
                self.known_hands[event.from_player] = remaining
                self.known_hands[event.to_player] = (
                    list(self.known_hands.get(event.to_player, [])) + list(event.cards)
                )
            elif isinstance(event, EventTrickStart):
                self.trick_index = event.trick_index
                self.opener_id = event.opener_id
                self.current_trick_plays = []
            elif isinstance(event, EventActionPlayed):
                remaining = list(self.known_hands.get(event.player_id, []))
                for card in event.cards_played:
                    if card in remaining:
                        remaining.remove(card)
                self.known_hands[event.player_id] = remaining
                self.current_trick_plays.append({
                    "player_id": event.player_id,
                    "action_type": event.action_type.value,
                    "cards": [serialize_card(c) for c in event.cards_played],
                    "resulting_power": event.resulting_power,
                    "was_suboptimal": event.was_suboptimal,
                })
            elif isinstance(event, EventTrickClosed):
                self.last_trick_winner = event.winner_id
            elif isinstance(event, EventPlayerFinished):
                if event.player_id not in self.finished_players:
                    self.finished_players.append(event.player_id)
            elif isinstance(event, EventRoundEnd):
                self.round_roles = dict(event.roles_by_player)
            elif isinstance(event, EventRuleTriggered):
                if event.rule_name in ("REVOLUTION", "DOUBLE_REVOLUTION"):
                    self.e_rev = not self.e_rev

    def _serialize_event(self, event: Any) -> Dict[str, Any]:
        """
        Convertit un événement en dictionnaire sérialisable, destiné au journal exposé à l'interface web.

        Paramètre `event` : événement à convertir.
        Retourne un dictionnaire portant `event_type` et l'ensemble des champs de l'événement convertis en valeurs sérialisables. Aucun
        effet de bord.
        """
        payload: Dict[str, Any] = {"event_type": type(event).__name__}
        for field in dataclasses.fields(event):
            payload[field.name] = _convert_value(getattr(event, field.name))
        return payload

    def start(self, total_rounds: int) -> None:
        """
        Démarre l'exécution de la partie dans un thread dédié.

        Paramètre `total_rounds` : nombre de manches à exécuter.
        Retourne `None`. Effet de bord : démarre un thread démon exécutant les manches ; sans effet si déjà démarrée.
        """
        with self._state_lock:
            if self.started:
                return
            self.started = True
            self.total_rounds = total_rounds
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """
        Exécute les manches de la partie jusqu'à leur terme ou jusqu'à une erreur.

        Retourne `None`. Effet de bord : exécute `total_rounds` manches via `Game.play_round`, capture toute exception dans `error`, et
        positionne `finished` à vrai en fin d'exécution.
        """
        try:
            for _ in range(self.total_rounds):
                self.game.play_round()
                with self._state_lock:
                    self.rounds_played += 1
        except Exception as exc:  # noqa: BLE001 - remonté à l'interface pour affichage
            with self._state_lock:
                self.error = str(exc)
        finally:
            with self._state_lock:
                self.finished = True

    def snapshot(self) -> Dict[str, Any]:
        """
        Construit l'état affichable courant de la session.

        Retourne un dictionnaire sérialisable en JSON, décrivant l'ensemble des informations nécessaires au rendu de l'interface web
        (tailles de main, mains révélées le cas échéant, pli en cours, points de victoire cumulés, rôles, sollicitations humaines en
        attente et fenêtre récente du journal d'événements). Aucun effet de bord hors la lecture de l'état interne.
        """
        with self._state_lock:
            hand_sizes = {str(pid): len(cards) for pid, cards in self.known_hands.items()}

            hands_payload: Optional[Dict[str, Any]] = None
            if self.reveal_hands:
                hands_payload = {
                    str(pid): [serialize_card(c) for c in cards]
                    for pid, cards in self.known_hands.items()
                }

            own_hands = {
                str(pid): [serialize_card(c) for c in self.known_hands.get(pid, [])]
                for pid in self.human_seats
            }

            pending_requests: Dict[str, Any] = {}
            for pid, agent in self.human_agents.items():
                request = agent.get_pending_request()
                if request is not None:
                    pending_requests[str(pid)] = request

            return {
                "session_id": self.session_id,
                "player_count": self.config.player_count,
                "seat_profiles": self.seat_profiles,
                "human_seats": self.human_seats,
                "reveal_hands": self.reveal_hands,
                "started": self.started,
                "finished": self.finished,
                "error": self.error,
                "e_rev": self.e_rev,
                "round_index": self.round_index,
                "rounds_played": self.rounds_played,
                "rounds_total": self.total_rounds,
                "trick_index": self.trick_index,
                "opener_id": self.opener_id,
                "last_trick_winner": self.last_trick_winner,
                "finished_players": self.finished_players,
                "current_trick_plays": self.current_trick_plays,
                "hand_sizes": hand_sizes,
                "hands": hands_payload,
                "own_hands": own_hands,
                "cumulative_vp": {str(pid): vp for pid, vp in self.game.cumulative_vp.items()},
                "roles": {str(pid): role for pid, role in (self.game.roles or {}).items()},
                "round_roles": {str(pid): role for pid, role in self.round_roles.items()},
                "pending_requests": pending_requests,
                "log_tail": self.event_log[-_LOG_TAIL_SIZE:],
            }

    def submit_response(self, player_id: int, response: Dict[str, Any]) -> bool:
        """
        Soumet la réponse d'un siège humain à sa sollicitation courante.

        Paramètre `player_id` : identifiant du siège humain concerné.
        Paramètre `response` : réponse soumise depuis l'interface web.
        Retourne un booléen, faux si `player_id` n'est pas un siège humain ou si aucune sollicitation n'était en attente.
        Effet de bord : débloque le thread de la partie s'il attendait cette réponse.
        """
        self.touch()
        agent = self.human_agents.get(player_id)
        if agent is None:
            return False
        return agent.submit_response(response)
