"""
Module de journalisation des événements.

Le module définit `EventLogger`, un abonné du bus d'événements qui accumule en mémoire les événements reçus et propose leur conversion en
liste de dictionnaires, leur export JSONL et leur export Parquet segmenté. Le journaliseur ne modifie jamais un événement reçu ; il le stocke
dans l'ordre de réception.

Le module dépend de `events.base` pour le type `Event` et de `dataclasses` pour l'introspection des champs d'événement.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Dict, List, Optional, TypeVar

import pyarrow as pa
import pyarrow.parquet as pq

from events.base import Event


TEvent = TypeVar("TEvent", bound=Event)

_STREAM_SCHEMA = pa.schema(
    [
        ("event_type", pa.string()),
        ("timestamp", pa.int64()),
        ("game_id", pa.string()),
        ("round_id", pa.int64()),
        ("state_hash", pa.string()),
        ("payload", pa.string()),
    ]
)


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
    Champ `parquet_path` : chemin optionnel recevant les vidanges automatiques du tampon.
    """

    def __init__(self, parquet_buffer_size: int = 100000, parquet_path: Optional[str] = None) -> None:
        if parquet_buffer_size <= 0:
            raise ValueError("parquet_buffer_size doit être strictement positif.")
        self.events: List[Event] = []
        self.parquet_buffer_size = parquet_buffer_size
        self.parquet_path = parquet_path
        self._parquet_buffer: List[Dict[str, Any]] = []
        self._parquet_writer: Optional[pq.ParquetWriter] = None
        self._parquet_writer_path: Optional[str] = None

    def __call__(self, event: Event) -> None:
        """
        Enregistre un événement reçu du bus.

        Paramètre `event` : événement reçu, type `Event`.
        Retourne `None`. Effet de bord : ajoute `event` à `events` et au tampon interne de segmentation Parquet, borné par
        `parquet_buffer_size`.
        """
        self.events.append(event)
        self._parquet_buffer.append(self._to_record(event))
        if self.parquet_path is not None and len(self._parquet_buffer) >= self.parquet_buffer_size:
            self.flush_to_parquet(self.parquet_path)

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
        Convertit le journal en table analytique.

        Retourne un objet `polars.DataFrame` construit à partir de `to_records()`. Lève `ImportError` si Polars n'est pas installé.
        Aucun effet de bord.
        """
        import polars as pl

        return pl.DataFrame(self.to_records())

    def to_polars_dataframe(self):
        """
        Convertit le journal en `DataFrame` Polars.

        Retourne un objet `polars.DataFrame` construit à partir de `to_records()`, destiné aux analyses statistiques à grande échelle
        parallélisées nativement.
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

    def _buffer_to_stream_table(self) -> pa.Table:
        """
        Convertit le tampon courant en table Parquet à schéma stable.

        Retourne une table Arrow dont les colonnes fixes portent les champs communs d'événement et dont la colonne `payload` contient les
        champs propres à chaque type d'événement sous forme JSON. Aucun effet de bord.
        """
        rows = []
        for record in self._parquet_buffer:
            payload = {
                key: value
                for key, value in record.items()
                if key not in {"event_type", "timestamp", "game_id", "round_id", "state_hash"}
            }
            rows.append(
                {
                    "event_type": str(record.get("event_type", "")),
                    "timestamp": int(record.get("timestamp", 0)),
                    "game_id": str(record.get("game_id", "")),
                    "round_id": int(record.get("round_id", 0)),
                    "state_hash": str(record.get("state_hash", "")),
                    "payload": json.dumps(payload, ensure_ascii=False),
                }
            )
        return pa.Table.from_pylist(rows, schema=_STREAM_SCHEMA)

    def flush_to_parquet(self, path: str) -> None:
        """
        Vide le tampon d'événements accumulés depuis le dernier appel vers un fichier Parquet.

        Paramètre `path` : chemin du fichier de destination.
        Retourne `None`. Effet de bord : convertit le tampon interne en table Arrow, l'écrit comme nouveau groupe Parquet, puis vide le
        tampon interne. N'a aucun effet si le tampon est vide.
        """
        if not self._parquet_buffer:
            return
        table = self._buffer_to_stream_table()
        if self._parquet_writer is None or self._parquet_writer_path != path:
            if self._parquet_writer is not None:
                self._parquet_writer.close()
            if os.path.exists(path):
                os.remove(path)
            self._parquet_writer = pq.ParquetWriter(path, _STREAM_SCHEMA, compression="zstd")
            self._parquet_writer_path = path
        self._parquet_writer.write_table(table)
        self._parquet_buffer.clear()

    def close(self) -> None:
        """
        Finalise les écritures Parquet ouvertes.

        Retourne `None`. Effet de bord : vide le tampon automatique si `parquet_path` est défini, puis ferme l'écrivain Parquet actif.
        """
        target_path = self.parquet_path or self._parquet_writer_path
        if target_path is not None:
            self.flush_to_parquet(target_path)
        if self._parquet_writer is not None:
            self._parquet_writer.close()
            self._parquet_writer = None
            self._parquet_writer_path = None

    def __enter__(self) -> "EventLogger":
        """
        Retourne l'instance courante pour une utilisation en gestionnaire de contexte.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Finalise les ressources d'écriture à la sortie d'un contexte.
        """
        self.close()

    def events_of_type(self, event_type: type[TEvent]) -> List[TEvent]:
        """
        Filtre le journal par type d'événement.

        Paramètre `event_type` : classe d'événement recherchée.
        Retourne la liste des événements du journal qui sont des instances de `event_type`, dans l'ordre de réception. Aucun effet de bord.
        """
        return [event for event in self.events if isinstance(event, event_type)]
