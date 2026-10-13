#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "--allow-download" not in sys.argv:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

from linroute import inventory, prompts, relations
from linroute.cache import slug
from linroute.provenance import REPO_ROOT, git_commit

DEFAULT_TOKENIZERS = (
    "allenai/Olmo-3-1025-7B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-9B-Base",
    "Qwen/Qwen3.8-27B",
)

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)

SANDBOX_SOURCES = (
    "data/prompts/sandbox.yaml",
    "data/sandbox/towns_candidates.tsv",
    "data/sandbox/regions.tsv",
)

RELATION_DIR_ENV = "LINROUTE_RELATIONS_DIR"

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--relations-dir",
        default=None,
        help=f"the Hernandez checkout's data directory holding */*.json; "
        f"falls back to ${RELATION_DIR_ENV}",
    )
    p.add_argument(
        "--out",
        default=None,
        help="output root (default: data/relations/ in this checkout)",
    )
    p.add_argument(
        "--tokenizer",
        action="append",
        dest="tokenizers",
        metavar="HF_ID",
        help=f"repeatable; default is the four cached snapshots "
        f"({', '.join(DEFAULT_TOKENIZERS)})",
    )
    p.add_argument(
        "--relations",
        default="all",
        help="'all' or a comma-separated list of relation names",
    )
    p.add_argument(
        "--historical-cap",
        type=int,
        default=300,
        help="the --subject-cap of the screening draw whose roles are "
        "labelled on each record.  A label only: the inventory itself always "
        "covers the full source, uncapped",
    )
    p.add_argument(
        "--historical-seed", type=int, default=0, help="that draw's seed"
    )
    p.add_argument(
        "--no-town-region",
        dest="town_region",
        action="store_false",
        help="skip the sandbox town -> region relation",
    )
    p.add_argument(
        "--allow-download",
        action="store_true",
        help="let the tokenizer loader reach the network.  Off by default: a "
        "missing tokenizer should be reported, not fetched mid-audit",
    )
    p.add_argument(
        "--all-records",
        action="store_true",
        help="also write the per-row files of the Hernandez relations.  Off "
        "by default and git-ignored when on: those rows are the upstream "
        "dataset, and only their sha256 is committed (docs/inventory.md)",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="regenerate in memory and compare with what is on disk, writing "
        "nothing.  Exits 1 on any difference, so a stale snapshot fails CI",
    )
    args = p.parse_args(argv)
    if not args.relations_dir:
        args.relations_dir = os.environ.get(RELATION_DIR_ENV)
    if not args.relations_dir:
        raise SystemExit(
            f"--relations-dir is required (or set ${RELATION_DIR_ENV}): the "
            f"Hernandez checkout is not part of this repository"
        )
    args.out = Path(args.out) if args.out else REPO_ROOT / "data" / "relations"
    args.tokenizers = tuple(args.tokenizers or DEFAULT_TOKENIZERS)
    return args

def _sha256(path: Path) -> str:
    return prompts.sha256_file(path)

def tokenizer_provenance(hf_id: str) -> dict:
    from huggingface_hub import try_to_load_from_cache

    files: dict[str, str] = {}
    revision = None
    for name in TOKENIZER_FILES:
        try:
            hit = try_to_load_from_cache(repo_id=hf_id, filename=name)
        except Exception:
            hit = None
        if not isinstance(hit, str):
            continue
        path = Path(hit)
        files[name] = _sha256(path)
        parts = path.parts
        if "snapshots" in parts:
            revision = parts[parts.index("snapshots") + 1]
    return {"hf_id": hf_id, "revision": revision, "files": files}

def load_tokenizers(hf_ids: tuple[str, ...]) -> dict[str, object]:
    from transformers import AutoTokenizer

    loaded, failed = {}, []
    for hf_id in hf_ids:
        try:
            loaded[hf_id] = AutoTokenizer.from_pretrained(hf_id)
        except Exception as exc:
            failed.append(f"{hf_id}: {type(exc).__name__}: {exc}")
    if failed:
        raise SystemExit(
            "these tokenizers are not available locally:\n  "
            + "\n  ".join(failed)
            + "\n\nFetch them on a machine that may reach the network "
            "(--allow-download), or pass --tokenizer for the subset you have. "
            "Nothing was written."
        )
    return loaded

def source_provenance(raw: relations.RawRelation) -> dict:
    if raw.relation_type == "sandbox":
        return {
            "kind": "repository",
            "files": {
                name: _sha256(REPO_ROOT / name) for name in SANDBOX_SOURCES
            },
        }
    entry = prompts.wild_prompts(raw.name)
    return {
        "kind": "external_checkout",
        "source_file": entry.source_file,
        "source_sha256": entry.source_sha256,
        "verified_against": "data/prompts/wild.yaml",
    }

def dump_json(blob: dict) -> str:
    return json.dumps(blob, indent=2, ensure_ascii=False) + "\n"

def emit(path: Path, text: str, *, check: bool, changed: list[str]) -> None:
    if check:
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current != text:
            changed.append(
                str(path.relative_to(REPO_ROOT))
                if path.is_relative_to(REPO_ROOT)
                else str(path)
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    relations_dir = Path(args.relations_dir).expanduser()

    raws = relations.load_relations(relations_dir)
    if args.town_region:
        raws.append(relations.town_region_relation())
    if args.relations != "all":
        wanted = {n.strip() for n in args.relations.split(",") if n.strip()}
        unknown = wanted - {r.name for r in raws}
        if unknown:
            raise SystemExit(f"no such relation: {sorted(unknown)}")
        raws = [r for r in raws if r.name in wanted]

    tokenizers = load_tokenizers(args.tokenizers)
    provenance = {i: tokenizer_provenance(i) for i in args.tokenizers}
    for hf_id, tok in tokenizers.items():
        provenance[hf_id].update(
            tokenizer_class=type(tok).__name__,
            vocab_size=len(tok),
            n_added_tokens=len(tok.get_added_vocab()),
            is_fast=bool(tok.is_fast),
        )

    changed: list[str] = []
    index_relations = []

    for raw in raws:
        rel_slug = slug(raw.name)
        out_dir = args.out / rel_slug / "inventory"
        invs = {}
        per_tokenizer = {}
        for hf_id, tok in tokenizers.items():
            inv = inventory.build_inventory(
                raw,
                tok,
                tokenizer_id=hf_id,
                relation_slug=rel_slug,
                historical_cap=args.historical_cap,
                historical_seed=args.historical_seed,
            )
            body = inventory.records_jsonl(inv)
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
            invs[hf_id] = inv
            per_tokenizer[hf_id] = inv.summary(records_sha256=digest)

            path = out_dir / f"records-{slug(hf_id)}.jsonl"
            if (raw.relation_type == "sandbox" or args.all_records
                    or (args.check and path.exists())):
                emit(path, body, check=args.check, changed=changed)

        comparison = inventory.compare_partitions(invs.values())
        first = next(iter(invs.values()))
        summary = {
            "schema": inventory.SCHEMA,
            "generator": "scripts/make_inventory.py",
            "generator_version": inventory.GENERATOR_VERSION,
            "relation": raw.name,
            "relation_slug": rel_slug,
            "relation_type": raw.relation_type,
            "n_templates": len(raw.templates),
            "canonical_template_index": 0,
            "historical_draw": {
                "subject_cap": args.historical_cap,
                "seed": args.historical_seed,
                "note": "a label on each record, never a filter of this "
                        "inventory, which covers the full source",
            },
            "source": source_provenance(raw),
            "tokenizers": [provenance[i] for i in args.tokenizers],
            "records_committed": raw.relation_type == "sandbox",
            "cross_tokenizer": comparison,
            "per_tokenizer": [per_tokenizer[i] for i in args.tokenizers],
        }
        emit(out_dir / "summary.json", dump_json(summary),
             check=args.check, changed=changed)

        index_relations.append({
            "relation": raw.name,
            "relation_slug": rel_slug,
            "relation_type": raw.relation_type,
            "n_source_rows": first.n_source_rows,
            "empty_in_every_tokenizer": all(i.empty for i in invs.values()),
            "status": "empty_after_filters"
            if all(i.empty for i in invs.values())
            else "populated",
            "n_conflicting_subject_strings": len(first.conflicting_subjects()),

            "prefix_uniform": all(i.prefix_uniform for i in invs.values()),
            "agree_up_to_class_names": comparison["agree_up_to_class_names"],
            "identical_subject_sets": comparison["identical_subject_sets"],
            "per_tokenizer": {
                hf_id: {
                    k: per_tokenizer[hf_id][k]
                    for k in (
                        "n_included", "n_excluded", "n_effective_classes",
                        "n_merged_classes", "n_single_object_classes",
                        "n_context_mismatched", "prefix_uniform",
                        "records_sha256",
                    )
                }
                for hf_id in args.tokenizers
            },
        })

    index = {
        "schema": inventory.SCHEMA,
        "generator": "scripts/make_inventory.py",
        "generator_version": inventory.GENERATOR_VERSION,
        "git_commit": git_commit(),
        "command": (
            "python scripts/make_inventory.py --relations-dir "
            "<hernandez checkout>/data"
        ),
        "policies": inventory.POLICIES,

        "token_rule": inventory.POLICIES["token_rule"],
        "pending": {
            "new_rel_templates": (
                "the four NEW-REL fine-tuning templates and the fifth "
                "held-out template do not exist for any relation but the town "
                "sandbox, so no record here is compared against them"
            ),
            "entity_resolution": (
                "no entity or alias grouping; support_certified_entities is "
                "null everywhere and subject IDs hash strings"
            ),
            "conflict_resolution": (
                "conflicting subject labels are quarantined, not decided"
            ),
            "readout_conditioning": (
                "the scored token is read at the last PROMPT token with "
                "nothing appended, so for an object with a skipped prefix "
                "this is not the model's probability of that token AFTER the "
                "prefix; prefix_uniform says the comparison within a relation "
                "is at least coherent, not that it is a conditional"
            ),
            "exposure_history": (
                "nothing here certifies that a subject is unscreened or "
                "eligible as a confirmatory final subject"
            ),
        },
        "exclusion_reasons": list(inventory.EXCLUSION_REASONS),
        "historical_roles": list(inventory.HISTORICAL_ROLES),
        "historical_draw": {
            "subject_cap": args.historical_cap,
            "seed": args.historical_seed,
        },
        "tokenizers": [provenance[i] for i in args.tokenizers],
        "prompt_files": prompts.digests(),
        "totals": {
            "n_relations": len(index_relations),
            "n_source_rows": sum(r["n_source_rows"] for r in index_relations),
            "n_empty_relations": sum(
                1 for r in index_relations if r["status"] == "empty_after_filters"
            ),
            "n_relations_with_conflicts": sum(
                1 for r in index_relations
                if r["n_conflicting_subject_strings"]
            ),
            "n_conflicting_subject_strings": sum(
                r["n_conflicting_subject_strings"] for r in index_relations
            ),
            "n_relations_with_nonuniform_prefix": sum(
                1 for r in index_relations if not r["prefix_uniform"]
            ),
        },
        "relations": index_relations,
    }
    emit(args.out / "inventory-index.json", dump_json(index),
         check=args.check, changed=changed)

    if args.check:
        if changed:
            print("out of date:", file=sys.stderr)
            for name in changed:
                print(f"  {name}", file=sys.stderr)
            return 1
        print(f"up to date: {len(index_relations)} relations, "
              f"{len(args.tokenizers)} tokenizers")
        return 0

    print(f"wrote {len(index_relations)} relations x {len(args.tokenizers)} "
          f"tokenizers under {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
