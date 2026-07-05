"""
Module du tampon de rejeu distribué.

Le module définit `RedisReplayBuffer`, un tampon de transitions `(features, action_score, retour)` partagé entre un ou plusieurs
processus « Rollout Worker » producteurs et un unique processus « Trainer » consommateur, communiquant via une instance Redis. Le tampon
matérialise une liste Redis bornée en taille (structure `FIFO` par troncature) ; chaque transition est sérialisée en JSON avant stockage.

Le module dépend de `redis`, `json` et `numpy`. Aucun effet de bord global.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

import numpy as np


class RedisReplayBuffer:
    """
    Tampon de rejeu distribué appuyé sur une liste Redis bornée.

    Champ `client` : instance `redis.Redis` connectée à l'instance Redis cible.
    Champ `key` : clé Redis de la liste de transitions.
    Champ `capacity` : taille maximale du tampon, entier strictement positif, au-delà de laquelle les transitions les plus anciennes sont
    tronquées.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        key: str = "president:replay_buffer",
        capacity: int = 200_000,
    ) -> None:
        import redis

        self.client = redis.Redis(host=host, port=port, db=db)
        self.key = key
        self.capacity = capacity

    def push(self, features: np.ndarray, chosen_score: float, return_value: float) -> None:
        """
        Ajoute une transition au tampon.

        Paramètre `features` : vecteur `numpy.ndarray` de caractéristiques de l'option choisie, taille `FEATURE_DIM`.
        Paramètre `chosen_score` : score attribué par la politique à l'option choisie au moment de la décision.
        Paramètre `return_value` : retour cumulé (point de victoire) observé pour la manche ayant produit cette transition.
        Retourne `None`. Effet de bord : sérialise la transition en JSON, l'insère en tête de la liste Redis `self.key`, puis tronque la
        liste à `self.capacity` éléments.
        """
        record: Dict[str, Any] = {
            "features": features.tolist(),
            "chosen_score": float(chosen_score),
            "return_value": float(return_value),
        }
        pipeline = self.client.pipeline()
        pipeline.lpush(self.key, json.dumps(record))
        pipeline.ltrim(self.key, 0, self.capacity - 1)
        pipeline.execute()

    def push_batch(self, transitions: List[Dict[str, Any]]) -> None:
        """
        Ajoute plusieurs transitions en une unique opération réseau.

        Paramètre `transitions` : liste de dictionnaires `{'features': list, 'chosen_score': float, 'return_value': float}`.
        Retourne `None`. Effet de bord : insère l'ensemble des transitions en tête de la liste Redis, puis tronque à `self.capacity`.
        """
        if not transitions:
            return
        pipeline = self.client.pipeline()
        for record in transitions:
            pipeline.lpush(self.key, json.dumps(record))
        pipeline.ltrim(self.key, 0, self.capacity - 1)
        pipeline.execute()

    def size(self) -> int:
        """
        Retourne le nombre de transitions actuellement stockées.

        Retourne un entier positif ou nul, résultat de `LLEN` sur la clé du tampon. Aucun effet de bord hors la lecture Redis.
        """
        return int(self.client.llen(self.key))

    def ping(self) -> bool:
        """
        Vérifie l'accessibilité de l'instance Redis cible.

        Retourne un booléen, vrai si l'instance Redis répond à une commande `PING`, faux si la connexion échoue pour quelque raison que ce
        soit (serveur non démarré, hôte ou port incorrect, réseau indisponible). Aucun effet de bord hors la tentative de connexion réseau.
        """
        try:
            return bool(self.client.ping())
        except Exception:
            return False

    def sample(self, batch_size: int) -> Optional[List[Dict[str, Any]]]:
        """
        Échantillonne un lot de transitions uniformément parmi le tampon courant.

        Paramètre `batch_size` : nombre de transitions à échantillonner, entier strictement positif.
        Retourne une liste de dictionnaires désérialisés de taille `batch_size`, ou `None` si le tampon contient strictement moins de
        `batch_size` transitions. L'échantillonnage s'effectue avec remise, par tirage d'index uniformes suivi de lectures `LINDEX`
        individuelles. Aucun effet de bord sur le contenu du tampon.
        """
        current_size = self.size()
        if current_size < batch_size:
            return None
        indices = [random.randrange(current_size) for _ in range(batch_size)]
        pipeline = self.client.pipeline()
        for index in indices:
            pipeline.lindex(self.key, index)
        raw_records = pipeline.execute()
        return [json.loads(raw) for raw in raw_records if raw is not None]

    def clear(self) -> None:
        """
        Vide intégralement le tampon.

        Retourne `None`. Effet de bord : supprime la clé Redis du tampon.
        """
        self.client.delete(self.key)
