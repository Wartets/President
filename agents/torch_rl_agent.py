"""
Module de l'agent à politique neuronale entraînable sur GPU.

Le module définit `TorchRLAgent`, une implémentation de `AbstractBaseAgent` remplaçant la politique linéaire `numpy` de
`agents.rl_agent.RLAgent` par un petit réseau de neurones `torch.nn.Module` (`PolicyNet`), évalué par lot sur l'accélérateur disponible.
L'agent réutilise la même construction de caractéristiques par option (`_option_features`, `FEATURE_DIM`) que `agents.rl_agent`, afin de
rester compatible avec les enregistrements produits par `training.train_rl`. L'inférence par lot (`get_batch_action`) exécute une unique
passe avant sur l'ensemble des options candidates de tous les états du lot, sous `torch.autocast` lorsque l'accélérateur le supporte.

Le module dépend de `agents.interface`, `agents.rl_agent` pour `FEATURE_DIM` et `_option_features`, `core.config`, `core.models`,
`engine.state` et de `torch`.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from agents.interface import AbstractBaseAgent
from agents.rl_agent import FEATURE_DIM, _option_features
from core.config import GameConfig
from core.models import Action, ActionType, Card, Hand
from core.rules_engine import generate_sequence_plays, generate_uniform_plays
from engine.state import GameState

# Largeur de la couche cachée unique du réseau de politique.
_HIDDEN_DIM = 32


def resolve_device(preferred: Optional[str] = None) -> torch.device:
    """
    Détermine le périphérique de calcul à utiliser pour l'inférence et l'entraînement.

    Paramètre `preferred` : nom de périphérique explicitement demandé, chaîne parmi `'cuda'`, `'cpu'`, ou `None` pour une sélection
    automatique.
    Retourne une instance de `torch.device`, égale à `preferred` si fourni et disponible, sinon `'cuda'` si un périphérique CUDA est
    détecté, sinon `'cpu'`. Aucun effet de bord.
    """
    if preferred is not None:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNet(nn.Module):
    """
    Réseau de politique évaluant un score scalaire par option de jeu candidate.

    Champ `layers` : séquence de couches linéaires et d'activations `torch.nn.Module`, prenant en entrée un vecteur de taille
    `FEATURE_DIM` et produisant un score scalaire non borné.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(FEATURE_DIM, _HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(_HIDDEN_DIM, _HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(_HIDDEN_DIM, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Calcule le score de chaque vecteur de caractéristiques du lot.

        Paramètre `features` : tenseur `(K, FEATURE_DIM)` de type `float32`, `K` options candidates agrégées sur un ou plusieurs états.
        Retourne un tenseur `(K,)` de type `float32`, score scalaire par option. Aucun effet de bord.
        """
        return self.layers(features).squeeze(-1)


class TorchRLAgent(AbstractBaseAgent):
    """
    Agent dont la décision de jeu repose sur un réseau de neurones entraînable, évalué par lot sur GPU.

    Champ `config` : configuration de la partie.
    Champ `device` : périphérique `torch` utilisé pour l'inférence.
    Champ `policy` : instance de `PolicyNet`, transférée sur `device`.
    Champ `epsilon` : probabilité d'exploration aléatoire lors du choix d'une action, domaine $[0, 1]$.
    Champ `use_amp` : indique si l'inférence utilise la précision mixte `torch.autocast`, actif uniquement lorsque `device` est de type
    `'cuda'`.
    Champ `_rng` : générateur pseudo-aléatoire dédié à l'exploration.
    """

    def __init__(
        self,
        player_id: int,
        config: GameConfig,
        policy: Optional[PolicyNet] = None,
        epsilon: float = 0.1,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(player_id)
        self.config = config
        self.device = resolve_device(device)
        self.policy = (policy if policy is not None else PolicyNet()).to(self.device)
        self.policy.eval()
        self.epsilon = epsilon
        self.use_amp = self.device.type == "cuda"
        self._rng = random.Random(f"{config.random_seed}:{player_id}:torch_rl")

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

    @torch.no_grad()
    def choose_action(self, game_state: GameState) -> Action:
        """
        Sélectionne une action de tour par évaluation du réseau de politique sur les options légales.

        Paramètre `game_state` : vue matérialisée de l'état courant.
        Retourne une instance de `Action`. Avec une probabilité `epsilon`, une option légale est choisie uniformément ; sinon, l'option de
        score `policy(features)` maximal est choisie. Retourne un passe conforme à `pass_type` si aucune option n'est disponible. Effet de
        bord : consomme l'état interne du générateur pseudo-aléatoire de l'agent.
        """
        hand = game_state.hands[self.player_id]
        options = self._legal_options(hand, game_state)
        if not options:
            return self._default_pass()

        if self._rng.random() < self.epsilon:
            cards, declared_power = self._rng.choice(options)
            return Action(action_type=ActionType.ACTION_PLAY, cards=cards, declared_power=declared_power)

        hand_size = hand.size()
        features = np.stack(
            [_option_features(cards, declared, hand_size, game_state.e_rev) for cards, declared in options]
        )
        tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        scores = self.policy(tensor)
        best_index = int(torch.argmax(scores).item())
        cards, declared_power = options[best_index]
        return Action(action_type=ActionType.ACTION_PLAY, cards=cards, declared_power=declared_power)

    @torch.no_grad()
    def get_batch_action(self, game_states: List[GameState]) -> List[Action]:
        """
        Sélectionne une action de tour pour un lot d'états simultanés par une unique passe avant sur GPU.

        Paramètre `game_states` : liste de vues matérialisées de l'état courant, une par simulation parallèle, taille $B$.
        Retourne une liste de `Action` de taille $B$, dans le même ordre que `game_states`. Effet de bord : consomme l'état interne du
        générateur pseudo-aléatoire de l'agent pour chaque état sans option légale ou sujet à exploration ; transfère un unique tenseur de
        caractéristiques agrégées sur `self.device` puis exécute l'inférence sous précision mixte lorsque `use_amp` est vrai.
        """
        per_state_options: List[List[Tuple[Tuple[Card, ...], Optional[int]]]] = []
        all_features: List[np.ndarray] = []
        owner_index: List[int] = []
        start_offsets: List[int] = []

        for state_index, game_state in enumerate(game_states):
            hand = game_state.hands[self.player_id]
            options = self._legal_options(hand, game_state)
            start_offsets.append(len(owner_index))
            per_state_options.append(options)
            hand_size = hand.size()
            for cards, declared in options:
                all_features.append(_option_features(cards, declared, hand_size, game_state.e_rev))
                owner_index.append(state_index)

        scores_by_flat_index: dict = {}
        if all_features:
            feature_tensor = torch.as_tensor(
                np.stack(all_features), dtype=torch.float32, device=self.device
            )
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    scores_tensor = self.policy(feature_tensor)
            else:
                scores_tensor = self.policy(feature_tensor)
            scores_np = scores_tensor.float().cpu().numpy()
            for flat_index, score in enumerate(scores_np):
                scores_by_flat_index[flat_index] = float(score)

        best_score_by_state: dict = {}
        best_flat_index_by_state: dict = {}
        for flat_index, state_index in enumerate(owner_index):
            score = scores_by_flat_index[flat_index]
            if state_index not in best_score_by_state or score > best_score_by_state[state_index]:
                best_score_by_state[state_index] = score
                best_flat_index_by_state[state_index] = flat_index

        results: List[Action] = []
        for state_index, options in enumerate(per_state_options):
            if not options:
                results.append(self._default_pass())
                continue
            if self._rng.random() < self.epsilon:
                cards, declared_power = self._rng.choice(options)
                results.append(Action(action_type=ActionType.ACTION_PLAY, cards=cards, declared_power=declared_power))
                continue
            flat_index = best_flat_index_by_state[state_index]
            local_offset = flat_index - start_offsets[state_index]
            cards, declared_power = options[local_offset]
            results.append(Action(action_type=ActionType.ACTION_PLAY, cards=cards, declared_power=declared_power))
        return results

    def choose_exchange_cards(self, hand: Hand, game_state: GameState, count: int) -> List[Card]:
        """
        Sélectionne les cartes de puissance la plus faible lors d'un échange.

        Paramètre `hand` : main courante de l'agent.
        Paramètre `game_state` : vue matérialisée de l'état courant.
        Paramètre `count` : nombre de cartes à céder.
        Retourne une liste de `Card` de taille `count`, triée par puissance croissante. Aucun effet de bord.
        """
        from core.math_utils import f_power

        ordered = sorted(hand.cards, key=lambda c: f_power(c, game_state.e_rev))
        return ordered[:count]

    def ask_putsch(self, hand: Hand) -> bool:
        """
        Invoque le Putsch selon la condition mathématique standard.

        Paramètre `hand` : main courante de l'agent.
        Retourne un booléen, vrai si au moins quatre cartes de la main partagent une même puissance standard ou si la puissance maximale de
        la main hors révolution est inférieure ou égale à dix. Aucun effet de bord.
        """
        from collections import Counter
        from core.math_utils import f_power

        powers = [f_power(c, False) for c in hand.cards if not c.is_joker()]
        if not powers:
            return False
        counts = Counter(powers)
        if any(count >= 4 for count in counts.values()):
            return True
        return max(powers) <= 10

    def on_interception_opportunity(
        self, game_state: GameState, played_card: Card
    ) -> Tuple[bool, Optional[Card]]:
        """
        Intercepte lorsqu'une carte jumelle est disponible, avec une probabilité d'exploration.

        Paramètre `game_state` : vue matérialisée de l'état courant.
        Paramètre `played_card` : carte cible de l'interception.
        Retourne un tuple `(decision, card)`. Si une carte jumelle est disponible, la décision d'intercepter est prise avec une probabilité
        `1 - epsilon`. Effet de bord : consomme l'état interne du générateur pseudo-aléatoire de l'agent.
        """
        hand = game_state.hands[self.player_id]
        twins = [
            c for c in hand.cards
            if not c.is_joker() and c.rank == played_card.rank and c.suit == played_card.suit
        ]
        if not twins:
            return False, None
        decision = self._rng.random() >= self.epsilon
        return decision, (twins[0] if decision else None)

    def save_weights(self, path: str) -> None:
        """
        Sauvegarde les poids du réseau de politique sur disque.

        Paramètre `path` : chemin du fichier de destination.
        Retourne `None`. Effet de bord : écrit l'état `state_dict` du réseau au format `torch`.
        """
        torch.save(self.policy.state_dict(), path)

    def load_weights(self, path: str) -> None:
        """
        Charge les poids du réseau de politique depuis le disque.

        Paramètre `path` : chemin du fichier source.
        Retourne `None`. Effet de bord : remplace l'état interne de `self.policy` par le contenu chargé, transféré sur `self.device`.
        """
        state_dict = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state_dict)
