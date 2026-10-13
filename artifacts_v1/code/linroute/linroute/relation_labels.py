from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SCHEMA = "linroute/relation-labels/1"
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
DEFAULT_PATH = (
    Path(__file__).resolve().parents[2]
    / "paper" / "metadata" / "relation-labels.json5"
)

@dataclass(frozen=True)
class RelationLabels:
    path: Path
    sha256: str
    relations: dict[str, str]

    def label(self, relation_id: str) -> str:
        try:
            return self.relations[relation_id]
        except KeyError as error:
            raise ValueError(f"missing canonical relation label: {relation_id}") from error

    def assert_complete(self, relation_ids: Iterable[str], context: str) -> None:
        wanted = set(relation_ids)
        missing = sorted(wanted - set(self.relations))
        if missing:
            raise ValueError(
                f"missing canonical relation labels for {context}: " + ", ".join(missing)
            )

        by_label: dict[str, list[str]] = {}
        for relation_id in sorted(wanted):
            by_label.setdefault(self.relations[relation_id], []).append(relation_id)
        shared = {label: ids for label, ids in by_label.items() if len(ids) > 1}
        if shared:
            raise ValueError(
                f"relations of {context} share a display label: "
                + "; ".join(f"{label!r}: {', '.join(ids)}" for label, ids in shared.items())
            )

def load(path: Path | str = DEFAULT_PATH) -> RelationLabels:
    source = Path(path)
    try:
        raw = source.read_bytes()
        document = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read canonical relation-label map {source}: {error}") from error
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError(f"{source}: expected schema {SCHEMA!r}")
    relations = document.get("relations")
    if not isinstance(relations, dict) or not relations:
        raise ValueError(f"{source}: missing non-empty 'relations' object")
    return RelationLabels(
        path=source,
        sha256=hashlib.sha256(raw).hexdigest(),
        relations=relations,
    )

def row_relation_id(row: dict) -> str:
    relation_id = row.get("relation_slug")
    if isinstance(relation_id, str) and relation_id:
        return relation_id
    relation = row.get("relation")
    if not isinstance(relation, str) or not relation:
        raise ValueError("result row has no relation_slug or relation")

    return _UNSAFE.sub("_", relation.replace("/", "__"))
