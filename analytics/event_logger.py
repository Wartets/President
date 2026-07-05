"""
Module de journalisation des événements.

Le module définit `EventLogger`, un abonné du bus d'événements qui accumule en mémoire l'intégralité des événements reçus et propose leur conversion en
liste de dictionnaires exploitable par Pandas, ainsi que leur export au format JSONL. Le journaliseur ne modifie jamais un événement reçu ; il se
contente de le stocker dans l'ordre de réception.

Le module dépend de `events.base` pour le type `Event` et de `dataclasses` pour l'introspection des champs d'événement.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Dict, List, TypeVar

import pyarrow as pa
import pyarrow.parquet as pq

from events.base import Event


TEvent = TypeVar("TEvent", bound=Event)


def _serialize_value(value: Any) -> Any:
    """
    Convertit une valeur d'événement en une forme sérialisable en JSON.

    Paramètre `value` : valeur à convertir, de type quelconque.
    Retourne une valeur composée uniquement de types primitifs, de listes et de dictionnaires. Les tuples sont convertis en listes, les objets munis
    d'une méthode `value` (énumérations) sont réduits à cette valeur, et les autres objets sont convertis via `repr`. Aucun effet de bord.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if hasattr(value, "value") and not dataclasses.is_dataclass(value):
        return value.value
    if dataclasses.is_dataclass(value):
        return {
            field.name: _serialize_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    return repr(value)


class EventLogger:
    """
    Journaliseur en mémoire du flux d'événements de la partie.

    Champ `events` : liste ordonnée des événements reçus depuis l'instanciation.
    """

    def __init__(self, parquet_buffer_size: int = 100000) -> None:
        self.events: List[Event] = []
        self.parquet_buffer_size = parquet_buffer_size
        self._parquet_buffer: List[Dict[str, Any]] = []

    def __call__(self, event: Event) -> None:
        """
        Enregistre un événement reçu du bus.

        Paramètre `event` : événement reçu, type `Event`.
        Retourne `None`. Effet de bord : ajoute `event` à `events` et au tampon interne de segmentation Parquet, borné par
        `parquet_buffer_size`.
        """
        self.events.append(event)
        self._parquet_buffer.append(self._to_record(event))

    def _to_record(self, event: Event) -> Dict[str, Any]:
        """
        Convertit un unique événement en dictionnaire plat sérialisable.

        Paramètre `event` : événement à convertir, type `Event`.
        Retourne un dictionnaire portant un champ `event_type` égal au nom de la classe de l'événement, ainsi que l'ensemble des champs de
        l'événement convertis en valeurs sérialisables. Aucun effet de bord.
        """
        record = {"event_type": type(event).__name__}
        for field in dataclasses.fields(event):
            record[field.name] = _serialize_value(getattr(event, field.name))
        return record

    def to_records(self) -> List[Dict[str, Any]]:
        """
        Convertit le journal en une liste de dictionnaires plats.

        Retourne une liste de dictionnaires, chacun portant un champ `event_type` égal au nom de la classe de l'événement, ainsi que
        l'ensemble des champs de l'événement convertis en valeurs sérialisables. Aucun effet de bord.
        """
        return [self._to_record(event) for event in self.events]

    def to_jsonl(self, path: str) -> None:
        """
        Exporte le journal au format JSON Lines.

        Paramètre `path` : chemin du fichier de destination.
        Retourne `None`. Effet de bord : écrit un objet JSON par ligne dans le fichier désigné par `path`, un par événement du journal, dans
        l'ordre de réception.
        """
        with open(path, "w", encoding="utf-8") as handle:
            for record in self.to_records():
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")

    def to_dataframe(self):
        """
        Convertit le journal en `DataFrame` Pandas.

        Retourne un objet `pandas.DataFrame` construit à partir de `to_records()`. Lève `ImportError` si Pandas n'est pas installé.
        Aucun effet de bord.
        """
        import pandas as pd

        return pd.DataFrame(self.to_records())

    def to_polars_dataframe(self):
        """
        Convertit le journal en `DataFrame` Polars.

        Retourne un objet `polars.DataFrame` construit à partir de `to_records()`, destiné aux analyses statistiques à grande échelle
        parallélisées nativement, conformément à la stratégie de remplacement de Pandas par Polars pour le passage à l'échelle.
        Lève `ImportError` si Polars n'est pas installé. Aucun effet de bord.
        """
        import polars as pl

        return pl.DataFrame(self.to_records())

    def to_parquet(self, path: str) -> None:
        """
        Exporte l'intégralité du journal au format Parquet.

        Paramètre `path` : chemin du fichier de destination.
        Retourne `None`. Effet de bord : construit une unique table Arrow colonnaire à partir de `to_records()` et l'écrit dans le fichier
        désigné par `path`, écrasant tout contenu existant.
        """
        table = pa.Table.from_pylist(self.to_records())
        pq.write_table(table, path)

    def flush_to_parquet(self, path: str) -> None:
        """
        Vide le tampon d'événements accumulés depuis le dernier appel vers un fichier Parquet.

        Paramètre `path` : chemin du fichier de destination.
        Retourne `None`. Effet de bord : convertit le tampon interne en table Arrow, la fusionne avec le contenu existant de `path` s'il
        existe, réécrit le fichier, puis vide le tampon interne. N'a aucun effet si le tampon est vide. Destiné à un appel périodique par
        segments de `parquet_buffer_size` événements lors de simulations massives, conformément à la stratégie de Buffered Writing.
        """
        if not self._parquet_buffer:
            return
        table = pa.Table.from_pylist(self._parquet_buffer)
        if os.path.exists(path):
            existing = pq.read_table(path)
            table = pa.concat_tables([existing, table])
        pq.write_table(table, path)
        self._parquet_buffer.clear()

    def events_of_type(self, event_type: type[TEvent]) -> List[TEvent]:
        """
        Filtre le journal par type d'événement.

        Paramètre `event_type` : classe d'événement recherchée.
        Retourne la liste des événements du journal qui sont des instances de `event_type`, dans l'ordre de réception. Aucun effet de bord.
        """
        return [event for event in self.events if isinstance(event, event_type)]
