"""
Module de l'agent interactif humain.

Le module définit `HumanAgent`, une implémentation de `AbstractBaseAgent` sollicitant les décisions de jeu par saisie interactive sur
l'entrée standard. L'agent affiche l'état pertinent de la partie et les options légales disponibles avant chaque sollicitation, et
retranscrit la sélection de l'utilisateur en une instance de `Action` ou en une liste de `Card` selon la méthode invoquée.

Le module dépend de `agents.interface`, `core.models`, `core.config`, `core.rules_engine` et `engine.state`.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from agents.interface import AbstractBaseAgent
from core.config import GameConfig
from core.models import Action, ActionType, Card, Hand
from core.rules_engine import generate_sequence_plays, generate_uniform_plays
from engine.state import GameState


def _format_cards(cards: Tuple[Card, ...]) -> str:
    """
    Construit une représentation textuelle ordonnée d'une combinaison de cartes.

    Paramètre `cards` : séquence de `Card` à représenter.
    Retourne une chaîne, cartes séparées par des espaces, dans l'ordre de la séquence fournie. Aucun effet de bord.
    """
    return " ".join(repr(card) for card in cards)


class HumanAgent(AbstractBaseAgent):
    """
    Agent sollicitant chaque décision de jeu par saisie interactive.

    Champ `config` : configuration de la partie, utilisée pour déterminer la légalité des suites et la sémantique de passe.
    """

    def __init__(self, player_id: int, config: GameConfig) -> None:
        super().__init__(player_id)
        self.config = config

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

    def choose_action(self, game_state: GameState) -> Action:
        """
        Sollicite le choix d'une action de tour par saisie interactive.

        Paramètre `game_state` : vue matérialisée de l'état courant.
        Retourne une instance de `Action`. Affiche la main courante, l'état du pli et la liste numérotée des options légales, puis lit un
        index sur l'entrée standard. Une saisie vide ou invalide est interprétée comme un passe conforme à `pass_type`. Effet de bord :
        écrit sur la sortie standard et lit sur l'entrée standard.
        """
        hand = game_state.hands[self.player_id]
        options = self._legal_options(hand, game_state)

        print(f"\n--- Joueur {self.player_id} ---")
        print(f"Main : {_format_cards(hand.cards)}")
        if game_state.trick.current_power is not None:
            print(f"Puissance à dépasser : {game_state.trick.current_power}")
        if game_state.e_rev:
            print("Révolution active.")

        if not options:
            print("Aucune combinaison légale disponible, passe automatique.")
            return self._default_pass()

        print("Options disponibles :")
        print("  0. Passer")
        for index, (cards, declared) in enumerate(options, start=1):
            suffix = f" (Joker déclaré à {declared})" if declared is not None else ""
            print(f"  {index}. {_format_cards(cards)}{suffix}")

        choice = input("Choix (numéro) : ").strip()
        if not choice.isdigit():
            return self._default_pass()
        choice_index = int(choice)
        if choice_index == 0 or choice_index > len(options):
            return self._default_pass()

        cards, declared_power = options[choice_index - 1]
        return Action(action_type=ActionType.ACTION_PLAY, cards=cards, declared_power=declared_power)

    def choose_exchange_cards(self, hand: Hand, game_state: GameState, count: int) -> List[Card]:
        """
        Sollicite le choix de cartes cédées lors d'un échange libre par saisie interactive.

        Paramètre `hand` : main courante de l'agent.
        Paramètre `game_state` : vue matérialisée de l'état courant.
        Paramètre `count` : nombre de cartes à céder.
        Retourne une liste de `Card` de taille `count`. Une saisie invalide ou incomplète est complétée par les cartes de plus faible
        puissance restantes. Effet de bord : écrit sur la sortie standard et lit sur l'entrée standard.
        """
        from core.math_utils import f_power

        ordered_hand = list(hand.cards)
        print(f"\n--- Joueur {self.player_id}, échange de {count} carte(s) ---")
        print(f"Main : {_format_cards(hand.cards)}")
        for index, card in enumerate(ordered_hand):
            print(f"  {index}. {card!r}")

        chosen: List[Card] = []
        raw = input(f"Indices des {count} cartes cédées, séparés par des espaces : ").strip()
        indices = [token for token in raw.split() if token.isdigit()]
        remaining = list(ordered_hand)
        for token in indices:
            idx = int(token)
            if 0 <= idx < len(ordered_hand) and ordered_hand[idx] in remaining:
                chosen.append(ordered_hand[idx])
                remaining.remove(ordered_hand[idx])
            if len(chosen) == count:
                break

        if len(chosen) < count:
            fallback = sorted(remaining, key=lambda c: f_power(c, game_state.e_rev))
            chosen.extend(fallback[: count - len(chosen)])
        return chosen[:count]

    def ask_putsch(self, hand: Hand) -> bool:
        """
        Sollicite la décision d'invocation du Putsch par saisie interactive.

        Paramètre `hand` : main courante de l'agent.
        Retourne un booléen, vrai si la saisie utilisateur commence par `'o'` ou `'O'`. Effet de bord : écrit sur la sortie standard et lit
        sur l'entrée standard.
        """
        print(f"\n--- Joueur {self.player_id}, invocation du Putsch ---")
        print(f"Main : {_format_cards(hand.cards)}")
        choice = input("Invoquer le Putsch ? (o/N) : ").strip().lower()
        return choice.startswith("o")

    def on_interception_opportunity(
        self, game_state: GameState, played_card: Card
    ) -> Tuple[bool, Optional[Card]]:
        """
        Sollicite la réponse à une opportunité d'interception par saisie interactive.

        Paramètre `game_state` : vue matérialisée de l'état courant.
        Paramètre `played_card` : carte cible de l'interception.
        Retourne un tuple `(decision, card)`. Propose l'interception uniquement si une carte jumelle est présente dans la main de l'agent.
        Effet de bord : écrit sur la sortie standard et lit sur l'entrée standard.
        """
        hand = game_state.hands[self.player_id]
        twins = [
            card for card in hand.cards
            if not card.is_joker() and card.rank == played_card.rank and card.suit == played_card.suit
        ]
        if not twins:
            return False, None

        print(f"\n--- Joueur {self.player_id}, opportunité d'interception ---")
        print(f"Carte posée : {played_card!r}")
        choice = input("Intercepter avec la carte jumelle ? (o/N) : ").strip().lower()
        if choice.startswith("o"):
            return True, twins[0]
        return False, None
