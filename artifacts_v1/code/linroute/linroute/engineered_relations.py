from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from linroute.relations import RawRelation

__all__ = [
    "RELATION_TYPE",
    "SPECS",
    "EngineeredRelation",
    "build",
    "build_all",
    "names",
    "raw_relations",
    "records_dir",
]

REPO_ROOT = Path(__file__).resolve().parents[1]

RELATION_TYPE = "engineered"

SOURCE_PLACEHOLDER = "{town}"
PLACEHOLDER = "{}"

@dataclass(frozen=True)
class Spec:
    directory: str
    relation: str

SPECS = (
    Spec("name-length", "engineered town name length"),
    Spec("first-letter", "engineered town first letter"),
    Spec("latitude", "engineered town latitude band"),
    Spec("longitude", "engineered town longitude band"),
    Spec("population", "engineered town population band"),
    Spec("elevation", "engineered town elevation band"),
    Spec("country-language", "engineered town country language"),
)

@dataclass(frozen=True)
class EngineeredRelation:
    relation: RawRelation
    provenance: dict

    @property
    def name(self) -> str:
        return self.relation.name

    @property
    def directory(self) -> str:
        return self.provenance["directory"]

def names() -> tuple[str, ...]:
    return tuple(spec.relation for spec in SPECS)

def records_dir(root: Path | str | None = None) -> Path:
    base = REPO_ROOT if root is None else Path(root)
    return base / "data" / "relations" / "engineered"

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _read_records(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def _template(proposals: object, *, relation: str) -> tuple[str, str]:
    if not isinstance(proposals, list) or not proposals:
        raise ValueError(
            f"{relation}: summary.json has no prompt_proposal list to take a "
            f"canonical template from"
        )
    source = proposals[0]
    if not isinstance(source, str) or source.count(SOURCE_PLACEHOLDER) != 1:
        raise ValueError(
            f"{relation}: the first prompt_proposal must be a string holding "
            f"exactly one {SOURCE_PLACEHOLDER}; got {source!r}"
        )
    if PLACEHOLDER in source.replace(SOURCE_PLACEHOLDER, ""):
        raise ValueError(
            f"{relation}: the first prompt_proposal already contains a bare "
            f"{PLACEHOLDER}, so rewriting {SOURCE_PLACEHOLDER} would leave two "
            f"placeholders: {source!r}"
        )
    return source.replace(SOURCE_PLACEHOLDER, PLACEHOLDER), source

def _object_of(row: dict) -> str:
    class_id = row.get("class_id")
    if class_id is None or str(class_id) == "unknown":
        return ""
    answer = row.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return ""
    return answer

def build(name: str, root: Path | str | None = None) -> EngineeredRelation:
    matches = [s for s in SPECS if name in (s.relation, s.directory)]
    if not matches:
        raise ValueError(
            f"no engineered relation {name!r}; expected one of "
            + ", ".join(f"{s.relation!r} ({s.directory!r})" for s in SPECS)
        )
    spec = matches[0]
    base = records_dir(root) / spec.directory
    records_path = base / "records.jsonl"
    summary_path = base / "summary.json"
    for path in (records_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{spec.relation}: {path} is missing; the engineered records "
                f"are produced by scripts/make_engineered.py and committed"
            )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    template, template_source = _template(
        summary.get("prompt_proposal"), relation=spec.relation
    )
    rows = _read_records(records_path)
    samples = tuple((str(row["town"]), _object_of(row)) for row in rows)
    n_empty = sum(1 for _, obj in samples if not obj)

    provenance = {
        "kind": "frozen_engineered_records",
        "directory": spec.directory,
        "relation": spec.relation,
        "records_file": str(records_path.relative_to(REPO_ROOT))
        if records_path.is_relative_to(REPO_ROOT) else str(records_path),
        "records_sha256": _sha256(records_path),
        "summary_file": str(summary_path.relative_to(REPO_ROOT))
        if summary_path.is_relative_to(REPO_ROOT) else str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "generator": "scripts/make_engineered.py",
        "definition": summary.get("definition"),
        "town_source": summary.get("town_source"),
        "town_source_sha256": summary.get("town_source_sha256"),
        "template_source": "summary.json prompt_proposal[0]",
        "template_source_string": template_source,
        "n_rows": len(rows),
        "n_empty_object": n_empty,
        "class_support": summary.get("class_support"),
        "subject_field": "town",
        "object_field": "answer",
    }
    return EngineeredRelation(
        relation=RawRelation(
            name=spec.relation,
            relation_type=RELATION_TYPE,
            templates=(template,),
            samples=samples,
        ),
        provenance=provenance,
    )

def build_all(root: Path | str | None = None) -> list[EngineeredRelation]:
    return [build(spec.relation, root) for spec in SPECS]

def raw_relations(root: Path | str | None = None) -> list[RawRelation]:
    return [built.relation for built in build_all(root)]
