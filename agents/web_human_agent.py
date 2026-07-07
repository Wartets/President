"""
Module de l'agent humain piloté par l'interface web.

Le module définit `WebHumanAgent`, une implémentation de `AbstractBaseAgent` qui ne lit jamais l'entrée standard : chaque sollicitation
publie une requête structurée consultable par l'interface web, puis bloque sur un événement interne jusqu'à ce qu'une réponse soit
soumise depuis une requête HTTP. L'agent est conçu pour être exécuté dans le thread dédié d'une manche, indépendant du thread traitant les
requêtes HTTP entrantes.

Le module dépend de `agents.interface`, `core.models`, `core.config`, `core.rules_engine` et `engine.state`.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

from agents.interface import AbstractBaseAgent
from core.config import GameConfig
from core.models import Action, ActionType, Card, Hand
from core.rules_engine import generate_sequence_plays, generate_uniform_plays
from engine.state import GameState


def serialize_card(card: Card) -> Dict[str, Any]:
    """
    Convertit une carte en dictionnaire sérialisable en JSON.

    Paramètre `card` : carte à convertir.
    Retourne un dictionnaire portant le rang, la couleur, une représentation textuelle et un indicateur de Joker. Aucun effet de bord.
    """
    return {
        "rank": card.rank.value,
        "suit": card.suit.value,
        "display": repr(card),
        "is_joker": card.is_joker(),
    }


def serialize_hand(hand: Hand) -> List[Dict[str, Any]]:
    """
    Convertit une main en liste de cartes sérialisables.

    Paramètre `hand` : main à convertir.
    Retourne une liste de dictionnaires, un par carte, dans l'ordre de la main. Aucun effet de bord.
    """
    return [serialize_card(card) for card in hand.cards]


class WebHumanAgent(AbstractBaseAgent):
    """
    Agent humain sollicité via l'interface web plutôt que la console.

    Champ `config` : configuration de la partie.
    Champ `_event` : événement de synchronisation, signalé lorsqu'une réponse a été soumise.
    Champ `_lock` : verrou protégeant l'accès concurrent à la requête et à la réponse en attente.
    Champ `_pending_request` : dictionnaire décrivant la sollicitation courante, ou `None` si aucune n'est en attente.
    Champ `_pending_response` : réponse soumise par l'interface web pour la sollicitation courante.
    """

    def __init__(self, player_id: int, config: GameConfig) -> None:
        super().__init__(player_id)
        self.config = config
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._pending_request: Optional[Dict[str, Any]] = None
        self._pending_response: Any = None

    def _legal_options(self, hand: Hand, game_state: GameState) -> List[Tuple[Tuple[Card, ...], Optional[int]]]:
        """
        Rassemble l'ensemble des combinaisons légales disponibles pour la main courante.

        Paramètre `hand` : main considérée.
        Paramètre `game_state` : vue matérialisée de l'état courant.
        Retourne une liste de tuples `(cards, declared_power)`. Aucun effet de bord.
        """
        trick = game_state.trick
        required_size = trick.size if trick.size > 0 else None
        min_power = trick.current_power

        options: List[Tuple[Tuple[Card, ...], Optional[int]]] = []
        if not trick.is_sequence:
            options.extend(generate_uniform_plays(hand, game_state.e_rev, required_size, min_power))
        if self.config.straights_enabled and (trick.size == 0 or trick.is_sequence):
            seq_min = trick.sequence_min_power if trick.is_sequence else None
            for cards, joker_map in generate_sequence_plays(hand, game_state.e_rev, required_size, seq_min):
                declared = joker_map[min(joker_map)] if joker_map else None
                options.append((cards, declared))
        return options

    def _default_pass(self) -> Action:
        """
        Construit l'action de passe conforme à la sémantique active.

        Retourne une instance de `Action` de type `ACTION_SOFT_PASS` si `pass_type` vaut `'ALLOW_SOFT'`, `ACTION_HARD_PASS` sinon. Aucun
        effet de bord.
        """
        action_type = (
            ActionType.ACTION_SOFT_PASS
            if self.config.pass_type == "ALLOW_SOFT"
            else ActionType.ACTION_HARD_PASS
        )
        return Action(action_type=action_type)

    def _submit_request_and_wait(self, payload: Dict[str, Any]) -> Any:
        """
        Publie une sollicitation et bloque jusqu'à réception d'une réponse.

        Paramètre `payload` : dictionnaire décrivant la sollicitation, consultable par l'interface web via `get_pending_request`.
        Retourne la réponse soumise par `submit_response`. Effet de bord : bloque le thread appelant jusqu'à la soumission d'une réponse.
        """
        with self._lock:
            self._pending_response = None
            self._pending_request = payload
            self._event.clear()
        self._event.wait()
        with self._lock:
            response = self._pending_response
            self._pending_request = None
            self._pending_response = None
        return response

    def get_pending_request(self) -> Optional[Dict[str, Any]]:
        """
        Retourne la sollicitation actuellement en attente, si l'agent bloque sur une décision.

        Retourne un dictionnaire décrivant la sollicitation, ou `None` si aucune décision n'est en attente. Aucun effet de bord.
        """
        with self._lock:
            return self._pending_request

    def has_pending_request(self) -> bool:
        """
        Indique si l'agent bloque actuellement sur une décision.

        Retourne un booléen. Aucun effet de bord.
        """
        with self._lock:
            return self._pending_request is not None

    def submit_response(self, response: Any) -> bool:
        """
        Soumet une réponse à la sollicitation courante et débloque le thread en attente.

        Paramètre `response` : réponse soumise par l'interface web, structure dépendante du type de sollicitation.
        Retourne un booléen, faux si aucune sollicitation n'était en attente au moment de l'appel. Effet de bord : débloque
        `_submit_request_and_wait`.
        """
        with self._lock:
            if self._pending_request is None:
                return False
            self._pending_response = response
        self._event.set()
        return True

    def choose_action(self, game_state: GameState) -> Action:
        """
        Sollicite le choix d'une action de tour via l'interface web.

        Paramètre `game_state` : vue matérialisée de l'état courant.
        Retourne une instance de `Action`. Si aucune combinaison légale n'est disponible, retourne un passe conforme à `pass_type` sans
        solliciter l'interface. Effet de bord : bloque jusqu'à réception d'une réponse si au moins une option est disponible.
        """
        hand = game_state.hands[self.player_id]
        options = self._legal_options(hand, game_state)
        if not options:
            return self._default_pass()

        payload = {
            "type": "action",
            "player_id": self.player_id,
            "hand": serialize_hand(hand),
            "options": [
                {
                    "index": idx,
                    "cards": [serialize_card(c) for c in cards],
                    "declared_power": declared,
                    "size": len(cards),
                }
                for idx, (cards, declared) in enumerate(options)
            ],
            "trick_size": game_state.trick.size,
            "trick_power": game_state.trick.current_power,
            "is_sequence": game_state.trick.is_sequence,
            "e_rev": game_state.e_rev,
        }
        response = self._submit_request_and_wait(payload)
        if not isinstance(response, dict) or response.get("pass"):
            return self._default_pass()

        option_index = response.get("option_index")
        if not isinstance(option_index, int) or option_index < 0 or option_index >= len(options):
            return self._default_pass()

        cards, declared_power = options[option_index]
        return Action(action_type=ActionType.ACTION_PLAY, cards=cards, declared_power=declared_power)

    def choose_exchange_cards(self, hand: Hand, game_state: GameState, count: int) -> List[Card]:
        """
        Sollicite le choix de cartes cédées lors d'un échange libre via l'interface web.

        Paramètre `hand` : main courante de l'agent.
        Paramètre `game_state` : vue matérialisée de l'état courant.
        Paramètre `count` : nombre de cartes à céder.
        Retourne une liste de `Card`, éventuellement de taille inférieure à `count` si la réponse est incomplète, le moteur complétant
        alors la sélection par les cartes de plus faible puissance. Effet de bord : bloque jusqu'à réception d'une réponse.
        """
        payload = {
            "type": "exchange",
            "player_id": self.player_id,
            "hand": serialize_hand(hand),
            "count": count,
        }
        response = self._submit_request_and_wait(payload)
        chosen: List[Card] = []
        if isinstance(response, dict):
            indices = response.get("card_indices") or []
            ordered_hand = list(hand.cards)
            seen_indices: set = set()
            for raw_index in indices:
                if not isinstance(raw_index, int) or raw_index in seen_indices:
                    continue
                if 0 <= raw_index < len(ordered_hand):
                    chosen.append(ordered_hand[raw_index])
                    seen_indices.add(raw_index)
                if len(chosen) == count:
                    break
        return chosen

    def ask_putsch(self, hand: Hand) -> bool:
        """
        Sollicite la décision d'invocation du Putsch via l'interface web.

        Paramètre `hand` : main courante de l'agent.
        Retourne un booléen, vrai si la réponse soumise porte `invoke: true`. Effet de bord : bloque jusqu'à réception d'une réponse.
        """
        payload = {
            "type": "putsch",
            "player_id": self.player_id,
            "hand": serialize_hand(hand),
        }
        response = self._submit_request_and_wait(payload)
        return bool(isinstance(response, dict) and response.get("invoke"))

    def on_interception_opportunity(
        self, game_state: GameState, played_card: Card
    ) -> Tuple[bool, Optional[Card]]:
        """
        Sollicite la réponse à une opportunité d'interception via l'interface web.

        Paramètre `game_state` : vue matérialisée de l'état courant.
        Paramètre `played_card` : carte cible de l'interception.
        Retourne un tuple `(decision, card)`. Ne sollicite l'interface que si une carte jumelle est disponible dans la main de l'agent ;
        retourne `(False, None)` immédiatement sinon. Effet de bord : bloque jusqu'à réception d'une réponse lorsque sollicité.
        """
        hand = game_state.hands[self.player_id]
        twins = [
            c for c in hand.cards
            if not c.is_joker() and c.rank == played_card.rank and c.suit == played_card.suit
        ]
        if not twins:
            return False, None

        payload = {
            "type": "interception",
            "player_id": self.player_id,
            "played_card": serialize_card(played_card),
            "twins": [serialize_card(c) for c in twins],
        }
        response = self._submit_request_and_wait(payload)
        if isinstance(response, dict) and response.get("intercept"):
            twin_index = response.get("twin_index", 0)
            if not isinstance(twin_index, int) or twin_index < 0 or twin_index >= len(twins):
                twin_index = 0
            return True, twins[twin_index]
        return False, None
