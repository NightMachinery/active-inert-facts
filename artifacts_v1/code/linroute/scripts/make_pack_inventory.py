#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_inventory as base

from linroute import relations

SOURCE_KIND = "external_pack"

def pack_relations(data_root: Path) -> list[relations.RawRelation]:
    out: list[relations.RawRelation] = []
    for path in sorted(Path(data_root).glob("*/*.json")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        props = blob.get("properties") or {}
        templates = tuple(blob.get("prompt_templates") or ())
        if not templates:
            raise SystemExit(f"{path}: no prompt_templates; cannot render a prompt")
        out.append(
            relations.RawRelation(
                name=blob["name"],
                relation_type=props.get("relation_type") or path.parent.name,
                templates=templates,
                samples=tuple(
                    (s["subject"], s["object"]) for s in blob.get("samples") or ()
                ),
            )
        )
    if not out:
        raise SystemExit(f"no <type>/<id>.json below {data_root}")
    return out

class _PackRelations:
    def __init__(self, root: Path) -> None:
        self._root = root

    def __getattr__(self, name):
        return getattr(relations, name)

    def load_relations(self, data_root):
        return pack_relations(Path(data_root))

PACK_DIGESTS: dict[str, str] = {}

def source_provenance(raw: relations.RawRelation) -> dict:
    root = Path(source_provenance.root)
    for path in sorted(root.glob("*/*.json")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        if blob["name"] != raw.name:
            continue
        digest = base._sha256(path)
        PACK_DIGESTS[str(path.relative_to(root))] = digest
        props = blob.get("properties") or {}
        return {
            "kind": SOURCE_KIND,
            "pack_root": str(root),
            "relation_file": str(path.relative_to(root)),
            "relation_sha256": digest,
            "verified_against": None,
            "declared": {
                "source": props.get("source") or {},
                "single_value_rule": props.get("single_value_rule"),
                "template_provenance": props.get("template_provenance"),
                "build_counts": props.get("build_counts"),
            },
        }
    raise SystemExit(f"{raw.name!r} is not in the pack at {root}")

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--no-town-region" not in argv:
        argv.append("--no-town-region")
    args = base.parse_args(list(argv))
    root = Path(args.relations_dir).expanduser()
    source_provenance.root = root

    base.relations = _PackRelations(root)
    base.source_provenance = source_provenance
    rc = base.main(list(argv))

    index_path = Path(args.out) / "inventory-index.json"
    if not args.check and index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["pack_root"] = str(root)
        index["pack_files_sha256"] = dict(sorted(PACK_DIGESTS.items()))
        index["command"] = (
            "python scripts/make_pack_inventory.py --relations-dir "
            f"{args.relations_dir} --out {args.out}"
        )
        index["note"] = (
            "This index covers ONE external pack and is not the study's "
            "data/relations/inventory-index.json.  The two are never merged."
        )
        index_path.write_text(
            json.dumps(index, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return rc

if __name__ == "__main__":
    raise SystemExit(main())
