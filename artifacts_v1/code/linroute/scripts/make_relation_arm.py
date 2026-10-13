#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linroute import arms
from linroute.provenance import git_commit

REPO_ROOT = Path(__file__).resolve().parents[1]

def _shown(path: Path) -> str:
    path = Path(path)
    try:
        if path.is_absolute() and path.is_relative_to(REPO_ROOT):
            return str(path.relative_to(REPO_ROOT))
    except AttributeError:
        pass
    return str(path)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", required=True,
                   help="split manifest JSON, schema linroute/admission/2")
    p.add_argument("--records", required=True,
                   help="the inventory records-<tok>.jsonl the manifest was "
                        "built from")
    p.add_argument("--templates", required=True,
                   help="data/newrel/<relation_slug>.yaml, schema "
                        "linroute/prompts/newrel/1")
    p.add_argument("--animals", default="data/animals/curated.txt",
                   help="the canonical pool; file order is part of the artifact")
    p.add_argument("--label-pool", default=None,
                   help="a pool DIRECTORY holding provenance.json, whose "
                        "category_word fills every paraphrase's {category} "
                        "slot (the shape data/label-pools/<category>/ uses)")
    p.add_argument("--category-word", default=None,
                   help="fill the {category} slot with this word directly, "
                        "instead of reading a pool declaration.  Default: "
                        f"{arms.CURATED_CATEGORY_WORD!r}, the curated pool's")
    p.add_argument("--repeat", type=int, default=0, choices=list(arms.REPEATS),
                   help="repeat id, which is also the bijection key and the "
                        "trainer's optimizer seed")
    p.add_argument("--role-seed", type=int, default=None,
                   help="default: the manifest's role seed, so the animal "
                        "subset is keyed by the same seed that chose the roles")
    p.add_argument("--bijection-slug", default=None,
                   help="key the animal subset and the class -> animal "
                        "bijection by THIS slug instead of the template "
                        "file's, and change nothing else: the arm keeps its "
                        "own name, relation and relation_slug.  For a study "
                        "that relabels one base relation over one subject "
                        "pool and wants each arm separately named -- distinct "
                        "run_ids, no collision with the base relation's "
                        "published rows -- while every arm still hands the "
                        "same animal to the same class.  That last property "
                        "holds only where the arms retain the same classes. "
                        "Default: the template file's slug, i.e. unchanged")
    p.add_argument("--pool-tokenizer", default=None,
                   help="the model whose FIRST TOKENS decide which declared "
                        "names this arm may answer with (see "
                        "docs/first-token-labels.md).  Default: the manifest's "
                        "own tokenizer, which is the model the effective "
                        "classes were computed under")
    p.add_argument("--tokenizer", action="append", default=None,
                   help="repeatable; the roster the arm is ASSERTED against, a "
                        "failure on any of which refuses it.  Default: the "
                        "pool tokenizer alone")
    p.add_argument("--relation-type", default=None,
                   help="override the template file's relation_type")
    p.add_argument("--arm-name", default=None,
                   help="override the derived <slug>-newrel-cNN-rR name")
    p.add_argument("--no-animal-assertion", action="store_true",
                   help="DO NOT USE for an arm you intend to train: writes the "
                        "arm without applying the first-token pool rule and "
                        "without verifying the offset property")
    p.add_argument("--label-exclusion", default=None,
                   help="a semantic-collision exclusion file for the --animals "
                        "pool (data/label-pools-exclusions/<rule>/<pool>.json, "
                        "written by scripts/make_label_exclusions.py).  The "
                        "relation's excluded labels leave the usable pool "
                        "before the draw; see docs/arms.md.  Default: none, "
                        "which builds exactly as generator 2.2.0 did")
    p.add_argument("--out", required=True, help="the arm directory to write")
    return p

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def resolve(x: str) -> Path:
        q = Path(x)
        return q if q.is_absolute() else REPO_ROOT / q

    manifest_path = resolve(args.manifest)
    records_path = resolve(args.records)
    templates_path = resolve(args.templates)
    animals_path = resolve(args.animals)

    manifest = arms.load_manifest(manifest_path)
    records = arms.load_records(records_path)
    prompts = arms.load_newrel(templates_path)

    if args.label_pool is not None and args.category_word is not None:
        print("REFUSED: pass --label-pool or --category-word, not both",
              file=sys.stderr)
        return 2
    category_word = args.category_word
    if args.label_pool is not None:
        category_word = arms.pool_category_word(args.label_pool)
    animals = arms.load_animals(animals_path)
    label_exclusion = None
    if args.label_exclusion is not None:
        exclusion_path = resolve(args.label_exclusion)
        try:
            label_exclusion = arms.load_label_exclusion(exclusion_path,
                                                        pool_path=animals_path)
        except arms.ArmBuildError as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 2
        label_exclusion["_path"] = _shown(exclusion_path)

    tokenizers = None if args.tokenizer is None else tuple(args.tokenizer)
    pool_tokenizer = args.pool_tokenizer
    if args.no_animal_assertion:
        tokenizers = ()
        pool_tokenizer = False
        print("WARNING: --no-animal-assertion: the first-token pool rule was "
              "NOT applied and the offset property is NOT verified.  This arm "
              "is not fit to train.", file=sys.stderr)

    relation = manifest.get("relation")
    if relation and prompts.relation_slug not in str(relation).replace(" ", "_"):
        print(f"note: the manifest's relation is {relation!r} but the template "
              f"file's slug is {prompts.relation_slug!r}; both are recorded.",
              file=sys.stderr)

    try:
        arm = arms.build_arm(
            manifest, records, prompts, animals,
            repeat=args.repeat,
            role_seed=args.role_seed,
            bijection_slug=args.bijection_slug,
            pool_tokenizer=pool_tokenizer,
            tokenizers=tokenizers,
            category_word=category_word,
            name=args.arm_name,
            relation_type=args.relation_type,
            manifest_path=_shown(manifest_path),
            manifest_sha256=arms.sha256_file(manifest_path),
            records_path=_shown(records_path),
            records_sha256=arms.sha256_file(records_path),
            animals_path=_shown(animals_path),
            git_commit=git_commit(REPO_ROOT),
            label_exclusion=label_exclusion,
        )
    except arms.AnimalAssertionError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        print(json.dumps(exc.report, indent=1, ensure_ascii=False)[:4000],
              file=sys.stderr)
        return 2
    except arms.ArmBuildError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    out = arms.write_arm(resolve(args.out), arm)
    meta = arm["meta"]
    print(f"{meta['arm']}: {meta['n_classes']} classes, "
          f"{meta['n_train_subjects']} trained + {meta['n_heldout_subjects']} "
          f"held-out subjects, {meta['n_train']} train / {meta['n_gate']} gate / "
          f"{meta['n_heldout']} heldout records -> {out}")
    for t in meta["animal_assertion"]["tokenizers"]:
        print(f"  animal assertion on {t['tokenizer']} "
              f"(revision {t['revision']}): "
              f"{'passed' if t['passed'] else 'FAILED'}, "
              f"{t['n_records_checked']} records checked")
    if not meta["animal_assertion"]["tokenizers"]:
        print("  animal assertion: NOT CHECKED")
    pool = (meta.get("animal_pool") or {}).get("pool")
    if pool:
        print(f"  label pool under {pool['tokenizer']}: {pool['n_usable']} of "
              f"{pool['n_declared']} usable by the {pool['rule']} rule "
              f"({pool['n_single_token']} single-token, "
              f"{pool['n_multi_token']} multi-token)")
        for d in pool["dropped"]:
            print(f"    dropped {d['animal']!r}: first token "
                  f"{d['first_token']!r} ({d['first_token_id']}) is already "
                  f"{d['absorbed_by']!r}'s")
    exc_block = meta.get("label_exclusion")
    if exc_block:
        print(f"  label exclusion {exc_block['rule']} (k={exc_block['k']}): "
              f"{exc_block['n_excluded']} labels excluded for "
              f"{exc_block['relation_slug']}, removed from the usable pool: "
              f"{exc_block['removed_from_usable']}; draw changed: "
              f"{exc_block['draw_changed']}")
    print(f"  animals frozen by {meta['seeds']['animal_subset_key']}, "
          f"bijection by {meta['seeds']['bijection_key']}, "
          f"optimizer seed {meta['seeds']['optimizer_seed']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
