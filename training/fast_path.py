"""
Module du moteur d'entraînement vectorisé (Fast-Path).

Le module implémente `FastPathEngine`, un environnement de simulation par lots opérant exclusivement sur des tenseurs `numpy`, sans
instanciation d'objets `Card`/`Hand`, conformément à la séparation Fast-Path / Slow-Path. L'environnement traite `B`
parties indépendantes en lock-step : à chaque appel de `step`, l'ensemble des `B` joueurs courants agissent simultanément, ce qui permet
une évaluation vectorisée unique de la légalité et de la transition d'état, sans boucle Python sur les parties.

Hypothèses de fonctionnement (simplifications volontaires du Fast-Path par rapport au moteur `Event Sourcing`) :
La main d'un joueur est représentée par un vecteur de comptage par rang, les couleurs (`Suit`) ne sont pas modélisées : les règles ne
dépendant que de `f_power` restent exactes, celles dépendant de la couleur (résolution d'égalité, interception) sont hors du périmètre du
Fast-Path. Les suites, le saut de tour, l'interception, le Putsch, l'échange de cartes et les pénalités de sortie étendues ne sont pas
implémentés ; seules les combinaisons uniformes, la révolution, la substitution par Joker et la clôture magique sur le rang deux sont
prises en charge. La taille du paquet est supposée divisible par `player_count` ; le reste éventuel de la division est ignoré.

Le module dépend de `core.config` pour `GameConfig` et de `numpy`. Aucun effet de bord global.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from core.config import GameConfig

# Nombre de rangs faciaux non-Joker représentés, du rang trois au rang deux inclus.
_NUM_STD_RANKS = 13

# Index de colonne réservé au comptage de Jokers dans le vecteur de main.
_JOKER_COLUMN = 13

# Nombre total de colonnes du vecteur de main (rangs standard plus Jokers).
_HAND_COLUMNS = _NUM_STD_RANKS + 1

# Puissance constante attribuée à toute combinaison de Jokers purs, en l'absence de carte naturelle de référence.
_PURE_JOKER_DECLARED_POWER = 16

# Sentinelle d'action représentant un passe, hors de l'espace des actions de pose.
ACTION_PASS = -1


def _power_for_rank(rank_index: np.ndarray, e_rev: np.ndarray) -> np.ndarray:
    """
    Calcule la puissance dynamique d'un rang standard pour un lot d'états.

    Paramètre `rank_index` : tableau d'entiers, domaine $[0, 12]$, index de rang facial standard.
    Paramètre `e_rev` : tableau booléen de même taille que le lot, état de révolution par partie.
    Retourne un tableau d'entiers, puissance dynamique résultante, calculée algébriquement par symétrie autour de dix-huit lorsque `e_rev`
    est vrai. Aucun effet de bord.
    """
    standard = rank_index + 3
    return np.where(e_rev, 18 - standard, standard)


def _effective_magic_rank_index(config: GameConfig, e_rev: np.ndarray) -> np.ndarray:
    """
    Calcule l'index de rang magique effectif pour un lot d'états.

    Paramètre `config` : configuration de la partie, utilisée pour `effective_magic_card_rank`.
    Paramètre `e_rev` : tableau booléen d'état de révolution par partie.
    Retourne un tableau d'entiers, index de rang magique remappé par symétrie lorsque `e_rev` est vrai, conformément à la résolution [C]
    de la matrice de compatibilité. Aucun effet de bord.
    """
    from core.math_utils import rank_facial_index

    magic_index_std = rank_facial_index(config.effective_magic_card_rank())
    mirrored = _NUM_STD_RANKS - 1 - magic_index_std
    return np.where(e_rev, mirrored, magic_index_std)


@dataclass
class FastPathState:
    """
    État tensoriel mutable d'un lot de parties simulées.

    Champ `hands` : tableau `(B, N, 14)` d'entiers, comptage de cartes par joueur et par rang, les treize premières colonnes couvrant les
    rangs standard et la dernière les Jokers.
    Champ `e_rev` : tableau `(B,)` booléen, état de révolution par partie.
    Champ `trick_size` : tableau `(B,)` d'entiers, taille imposée du pli courant, nul si le pli est vide.
    Champ `trick_power` : tableau `(B,)` d'entiers, puissance courante du pli, moins un si le pli est vide.
    Champ `last_player` : tableau `(B,)` d'entiers, identifiant du dernier joueur ayant posé une combinaison, moins un si aucun.
    Champ `current_player` : tableau `(B,)` d'entiers, identifiant du joueur devant agir.
    Champ `eligible` : tableau `(B, N)` booléen, éligibilité au pli courant.
    Champ `is_finished` : tableau `(B, N)` booléen, joueurs ayant vidé leur main pour la manche courante.
    Champ `finish_rank` : tableau `(B, N)` d'entiers, index de sortie attribué à chaque joueur, moins un si non encore sorti.
    Champ `next_finish_index` : tableau `(B,)` d'entiers, prochain index de sortie disponible par partie.
    Champ `done` : tableau `(B,)` booléen, vrai si la manche est terminée pour la partie considérée.
    """

    hands: np.ndarray
    e_rev: np.ndarray
    trick_size: np.ndarray
    trick_power: np.ndarray
    last_player: np.ndarray
    current_player: np.ndarray
    eligible: np.ndarray
    is_finished: np.ndarray
    finish_rank: np.ndarray
    next_finish_index: np.ndarray
    done: np.ndarray


class FastPathEngine:
    """
    Environnement de simulation vectorisée d'une manche du jeu, opérant par lots.

    Champ `config` : configuration de la partie, utilisée pour `player_count`, `revolution_enabled`, `effective_magic_card_enabled` et
    `effective_magic_card_rank`.
    Champ `batch_size` : nombre de parties simulées simultanément, entier strictement positif.
    Champ `max_combo_size` : taille maximale de combinaison représentée dans l'espace d'action, entier strictement positif.
    Champ `state` : instance courante de `FastPathState`, `None` avant le premier appel à `reset`.
    """

    def __init__(self, config: GameConfig, batch_size: int, max_combo_size: int = 8) -> None:
        self.config = config
        self.batch_size = batch_size
        self.max_combo_size = max_combo_size
        self.state: Optional[FastPathState] = None

    def action_space_size(self) -> int:
        """
        Retourne la taille de l'espace d'action de pose, hors passe.

        Retourne un entier strictement positif, égal au produit du nombre de rangs représentables (treize rangs standard plus un rang
        Joker pur) par `max_combo_size`. Aucun effet de bord.
        """
        return (_HAND_COLUMNS) * self.max_combo_size

    def _decode_action(self, action_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Décompose un index d'action de pose en couple rang-taille.

        Paramètre `action_index` : tableau `(B,)` d'entiers, domaine $[0, action\\_space\\_size())$, action de pose sélectionnée pour les
        parties concernées.
        Retourne un tuple `(rank_index, size)`, chacun un tableau `(B,)` d'entiers, `rank_index` dans $[0, 13]$ (treize désignant une
        combinaison de Jokers purs) et `size` dans $[1, max\\_combo\\_size]$. Aucun effet de bord.
        """
        rank_index = action_index // self.max_combo_size
        size = (action_index % self.max_combo_size) + 1
        return rank_index, size

    def reset(self, base_seed: int) -> FastPathState:
        """
        Réinitialise le lot de parties et distribue les mains initiales.

        Paramètre `base_seed` : graine de base, chaque partie du lot recevant une graine dérivée distincte via `numpy.random.default_rng`.
        Retourne l'instance de `FastPathState` construite. Effet de bord : remplace `self.state`. La distribution des cartes utilise une
        permutation aléatoire vectorisée d'étiquettes de rang, sans modélisation individuelle des couleurs.
        """
        n = self.config.player_count
        decks = max(1, (n - 1) // 4 + 1) if self.config.deck_scaling_auto else (self.config.forced_deck_count or 1)
        std_per_rank = 4 * decks
        joker_count_total = 2 * decks if self.config.use_jokers else 0
        total_cards = std_per_rank * _NUM_STD_RANKS + joker_count_total
        hand_size = total_cards // n

        rng = np.random.default_rng(base_seed)
        labels = np.concatenate([
            np.repeat(np.arange(_NUM_STD_RANKS), std_per_rank),
            np.full(joker_count_total, _JOKER_COLUMN, dtype=np.int64),
        ])
        label_matrix = np.tile(labels, (self.batch_size, 1))
        shuffle_keys = rng.random((self.batch_size, total_cards))
        order = np.argsort(shuffle_keys, axis=1)
        shuffled = np.take_along_axis(label_matrix, order, axis=1)

        hands = np.zeros((self.batch_size, n, _HAND_COLUMNS), dtype=np.int32)
        for pid in range(n):
            segment = shuffled[:, pid * hand_size:(pid + 1) * hand_size]
            for rank in range(_HAND_COLUMNS):
                hands[:, pid, rank] = np.sum(segment == rank, axis=1)

        self.state = FastPathState(
            hands=hands,
            e_rev=np.zeros(self.batch_size, dtype=bool),
            trick_size=np.zeros(self.batch_size, dtype=np.int32),
            trick_power=np.full(self.batch_size, -1, dtype=np.int32),
            last_player=np.full(self.batch_size, -1, dtype=np.int32),
            current_player=np.zeros(self.batch_size, dtype=np.int32),
            eligible=np.ones((self.batch_size, n), dtype=bool),
            is_finished=np.zeros((self.batch_size, n), dtype=bool),
            finish_rank=np.full((self.batch_size, n), -1, dtype=np.int32),
            next_finish_index=np.zeros(self.batch_size, dtype=np.int32),
            done=np.zeros(self.batch_size, dtype=bool),
        )
        return self.state

    def legal_action_mask(self) -> np.ndarray:
        """
        Construit le masque booléen des actions de pose légales pour le joueur courant de chaque partie du lot.

        Retourne un tableau `(B, action_space_size())` booléen. Une action `(rank_index, size)` est légale si le joueur courant dispose
        d'assez de cartes naturelles et de Jokers pour composer la combinaison, si sa taille correspond à `trick_size` lorsque celui-ci est
        non nul, et si sa puissance résultante dépasse strictement `trick_power` lorsque celui-ci est défini. Aucun effet de bord.
        """
        state = self.state
        n = self.config.player_count
        b = self.batch_size
        cp = state.current_player
        hand = state.hands[np.arange(b), cp]
        std_counts = hand[:, :_NUM_STD_RANKS]
        joker_counts = hand[:, _JOKER_COLUMN]

        mask = np.zeros((b, self.action_space_size()), dtype=bool)
        sizes = np.arange(1, self.max_combo_size + 1)

        for rank in range(_NUM_STD_RANKS):
            natural = std_counts[:, rank]
            reachable = natural[:, None] + joker_counts[:, None] >= sizes[None, :]
            has_natural = natural[:, None] > 0
            legal_size = reachable & has_natural
            power = _power_for_rank(np.full(b, rank), state.e_rev)
            for size_offset, size in enumerate(sizes):
                required_ok = (state.trick_size == 0) | (state.trick_size == size)
                power_ok = (state.trick_power < 0) | (power > state.trick_power)
                col = rank * self.max_combo_size + size_offset
                mask[:, col] = legal_size[:, size_offset] & required_ok & power_ok

        joker_reachable = joker_counts[:, None] >= sizes[None, :]
        power_ok_joker = (state.trick_power < 0) | (_PURE_JOKER_DECLARED_POWER > state.trick_power)
        for size_offset, size in enumerate(sizes):
            required_ok = (state.trick_size == 0) | (state.trick_size == size)
            col = _JOKER_COLUMN * self.max_combo_size + size_offset
            mask[:, col] = joker_reachable[:, size_offset] & required_ok & power_ok_joker

        no_option = ~np.any(mask, axis=1)
        mask[no_option, :] = False
        return mask

    def step(self, action_index: np.ndarray) -> Tuple[FastPathState, np.ndarray, np.ndarray]:
        """
        Applique une action sélectionnée par le joueur courant de chaque partie du lot.

        Paramètre `action_index` : tableau `(B,)` d'entiers, `ACTION_PASS` pour un passe ou un index de `legal_action_mask` pour une pose,
        supposé légal pour les lignes ne valant pas `ACTION_PASS`.
        Retourne un tuple `(state, reward, done)` : `state` est l'instance `FastPathState` mise à jour, `reward` un tableau `(B,)` de type
        `float64` valant le point de victoire `SYMMETRICAL` attribué au joueur qui vient d'agir en cas de sortie de manche et zéro sinon,
        `done` un tableau `(B,)` booléen indiquant la fin de manche. Effet de bord : mute `self.state` en place.
        """
        state = self.state
        n = self.config.player_count
        b = self.batch_size
        cp = state.current_player
        reward = np.zeros(b, dtype=np.float64)

        is_play = action_index != ACTION_PASS
        rank_index, size = self._decode_action(np.clip(action_index, 0, self.action_space_size() - 1))

        for row in np.nonzero(is_play)[0]:
            pid = cp[row]
            rank = rank_index[row]
            take = size[row]
            if rank == _JOKER_COLUMN:
                state.hands[row, pid, _JOKER_COLUMN] -= take
                power = _PURE_JOKER_DECLARED_POWER
            else:
                natural = state.hands[row, pid, rank]
                natural_used = min(int(natural), int(take))
                joker_used = int(take) - natural_used
                state.hands[row, pid, rank] -= natural_used
                state.hands[row, pid, _JOKER_COLUMN] -= joker_used
                power = int(_power_for_rank(np.array([rank]), state.e_rev[row:row + 1])[0])

            if state.trick_size[row] == 0:
                state.trick_size[row] = take
            state.trick_power[row] = power
            state.last_player[row] = pid

            if self.config.revolution_enabled and take >= 4 and rank != _JOKER_COLUMN:
                state.e_rev[row] = not state.e_rev[row]

            magic_index = int(_effective_magic_rank_index(self.config, state.e_rev[row:row + 1])[0])
            if self.config.effective_magic_card_enabled() and rank == magic_index:
                state.eligible[row, :] = False
                state.eligible[row, pid] = True

            if int(np.sum(state.hands[row, pid])) == 0 and not state.is_finished[row, pid]:
                state.is_finished[row, pid] = True
                state.finish_rank[row, pid] = state.next_finish_index[row]
                from core.math_utils import compute_vp
                reward[row] = compute_vp(
                    int(state.next_finish_index[row]), n, self.config.vp_distribution_type
                )
                state.next_finish_index[row] += 1

        for row in np.nonzero(~is_play)[0]:
            state.eligible[row, cp[row]] = False

        for row in range(b):
            if state.done[row]:
                continue
            active_others = [
                pid for pid in range(n)
                if pid != state.last_player[row] and not state.is_finished[row, pid]
            ]
            trick_closes = state.last_player[row] >= 0 and all(
                not state.eligible[row, pid] for pid in active_others
            )
            if trick_closes:
                winner = state.last_player[row]
                state.trick_size[row] = 0
                state.trick_power[row] = -1
                state.last_player[row] = -1
                active = [pid for pid in range(n) if not state.is_finished[row, pid]]
                state.eligible[row, :] = False
                for pid in active:
                    state.eligible[row, pid] = True
                state.current_player[row] = winner if winner in active else (active[0] if active else winner)
            else:
                candidate = int(cp[row])
                for _ in range(n):
                    candidate = (candidate + 1) % n
                    if not state.is_finished[row, candidate]:
                        break
                state.current_player[row] = candidate

            remaining = [pid for pid in range(n) if not state.is_finished[row, pid]]
            if len(remaining) <= 1:
                if len(remaining) == 1 and state.finish_rank[row, remaining[0]] < 0:
                    state.finish_rank[row, remaining[0]] = state.next_finish_index[row]
                state.done[row] = True

        return state, reward, state.done.copy()
