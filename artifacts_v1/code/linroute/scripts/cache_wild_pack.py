#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cache_wild as base

from linroute import relations

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

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

def collect_relations(args: argparse.Namespace) -> list[relations.RawRelation]:
    if not args.relations_dir:
        raise SystemExit("--relations-dir must point at the pack root")
    root = Path(args.relations_dir).expanduser()
    if not root.is_dir():
        raise SystemExit(f"--relations-dir {root} is not a directory")
    raws = pack_relations(root)
    args.engineered_provenance = None

    args.pack_root = str(root)

    if args.relations == "all":
        selected = raws
    else:
        by_name = {r.name: r for r in raws}
        wanted = [name.strip() for name in args.relations.split(",")]
        missing = [name for name in wanted if name not in by_name]
        if missing:
            raise SystemExit(
                f"unknown relations: {missing}.  The pack holds: "
                f"{sorted(by_name)}"
            )
        selected = [by_name[name] for name in wanted]
    args.pack_selected = [r.name for r in selected]
    return selected

def write_pack_provenance(args: argparse.Namespace) -> Path:
    root = Path(args.pack_root)
    files = {
        str(p.relative_to(root)): _sha256(p)
        for p in sorted(root.glob("*/*.json"))
    }
    out = Path(args.out).expanduser() / "pack-provenance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "generator": "scripts/cache_wild_pack.py",
        "written_at": datetime.datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "pack_root": str(root),
        "pack_files_sha256": files,
        "selected_relations": getattr(args, "pack_selected", None),
        "model": args.model,
        "config_hash": base.config_hash(args),
        "split_manifest": getattr(args, "split_manifest_provenance", None),
        "note": (
            "config_hash already hashes relations_dir and the split manifest's "
            "sha256, so a pack run can never extend a study cache directory.  "
            "This file additionally pins the pack CONTENT, which config_hash "
            "cannot see: a pack file edited in place under the same path would "
            "otherwise reuse the same hash."
        ),
    }
    existing = json.loads(out.read_text()) if out.exists() else None
    if existing is not None:
        drifted = {
            name: (existing["pack_files_sha256"].get(name), digest)
            for name, digest in files.items()
            if existing["pack_files_sha256"].get(name) not in (None, digest)
        }
        if drifted:
            raise SystemExit(
                f"{out}: the pack has changed since this cache directory was "
                f"started: {drifted}.  Write to a new --out rather than mixing "
                f"two versions of a relation into one directory."
            )
        blob["previous_runs"] = (existing.get("previous_runs") or []) + [
            {k: existing.get(k) for k in ("written_at", "config_hash",
                                          "selected_relations")}
        ]
    out.write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    return out

def main(argv: list[str] | None = None) -> None:
    base.collect_relations = collect_relations
    argv = list(sys.argv[1:] if argv is None else argv)
    for forced in ("--no-town-region", "--no-engineered"):
        if forced not in argv:
            argv.append(forced)

    inner_resume = base.resume_plan

    def resume_plan(root, args, raws):
        write_pack_provenance(args)
        return inner_resume(root, args, raws)

    base.resume_plan = resume_plan
    base.main(argv)

if __name__ == "__main__":
    main()
