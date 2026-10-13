#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linroute import admission
from linroute.cache import sha256_file, slug
from linroute.provenance import REPO_ROOT

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    d = admission.DEFAULTS
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--records",
        required=True,
        help="a records-<tokenizer>.jsonl written by scripts/make_inventory.py",
    )
    p.add_argument("--out", required=True, help="where to write the manifest JSON")
    p.add_argument(
        "--relation",
        default=None,
        help="relation name for the manifest; default is read from the "
        "inventory summary.json beside the records file, else its directory name",
    )
    p.add_argument(
        "--tokenizer",
        default=None,
        help="tokenizer id for the manifest; default is recovered from the "
        "records file name",
    )
    p.add_argument(
        "--n-shots",
        type=int,
        default=None,
        help="demonstration subjects to reserve before eligibility "
        "(default: data/prompts/few_shot.yaml's n_shots)",
    )
    p.add_argument(
        "--support-rule",
        choices=admission.SUPPORT_RULES,
        default=d["support_rule"],
        help="'seeded' (default) permutes the whole pool like "
        "linroute.relations.prepare_relation; 'protect' spends the shots on "
        "the largest classes so a marginal class is not pushed under the "
        "admission threshold when that is avoidable",
    )
    p.add_argument("--min-support", type=int, default=d["min_support"],
                   help="supp(c) needed to retain a class")
    p.add_argument("--min-classes", type=int, default=d["min_classes"],
                   help="|E| needed to admit the comparison")
    p.add_argument("--max-classes", type=int, default=d["max_classes"],
                   help="cap on |E|: when more classes qualify on support, "
                   "exactly this many are drawn uniformly at the role seed and "
                   "the rest are excluded as class_cap_<n> (default 78)")
    p.add_argument("--reuse-class-cap-order", type=Path, default=None,
                   help="a prior manifest whose admission.class_cap.order is "
                   "reused instead of redrawing the cap permutation.  The kept "
                   "set is then the first --max-classes of THAT order, so a "
                   "rebuild at a lower cap keeps a SUBSET of the original's "
                   "classes instead of an unrelated draw.  Refuses unless the "
                   "order is exactly a permutation of the classes qualifying "
                   "here")
    p.add_argument("--min-rows", type=int, default=d["min_rows"],
                   help="N = sum_c (supp(c) - h_c) needed to admit it")
    p.add_argument("--holdout-divisor", type=int, default=d["holdout_divisor"],
                   help="the 5 in h_c = min(cap, max(1, n_c // divisor), "
                   "n_c - min_train): hold out about one subject in this many")
    p.add_argument("--holdout-cap", type=int, default=d["holdout_cap"],
                   help="the 10 in the same rule: the most any one class holds out")
    p.add_argument("--holdout-min-train", type=int,
                   default=d["holdout_min_train"],
                   help="the 4 in the same rule: fine-tuning subjects a class "
                   "must keep (inert while --min-support is 5)")
    p.add_argument("--extends-v1", type=Path, default=None,
                   help="the linroute/admission/1 manifest this split extends; "
                   "its held-out subjects are ASSERTED to be exactly the "
                   "rank-0 members of the new H, and its sha256 is recorded")
    p.add_argument("--q", type=int, default=d["q"],
                   help="probe capacity the fold audit is run against")
    p.add_argument("--outer", type=int, default=d["outer"], help="outer folds")
    p.add_argument("--inner", type=int, default=d["inner"],
                   help="inner folds inside each outer fitting partition")
    p.add_argument("--role-seed", type=int, default=d["role_seed"],
                   help="seed for the support reservation and the H draw")
    p.add_argument("--fold-seed", type=int, default=d["fold_seed"],
                   help="seed for the fold allocation, recorded separately")
    p.add_argument(
        "--no-require-context-consistent",
        dest="require_context_consistent",
        action="store_false",
        help="keep records whose rendered answer token differs from the "
        "isolated one instead of dropping them as context_inconsistent",
    )
    p.set_defaults(require_context_consistent=d["require_context_consistent"])
    return p.parse_args(argv)

def identify(records_path: Path, args: argparse.Namespace) -> tuple[str, str]:
    relation = args.relation
    tokenizer = args.tokenizer

    stem = records_path.stem
    tok_slug = stem[len("records-"):] if stem.startswith("records-") else stem
    summary = records_path.parent / "summary.json"
    if (relation is None or tokenizer is None) and summary.exists():
        try:
            import json

            blob = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            blob = {}
        if relation is None and isinstance(blob.get("relation"), str):
            relation = blob["relation"]
        if tokenizer is None:
            for entry in blob.get("tokenizers", []) or []:
                hf_id = entry.get("hf_id") if isinstance(entry, dict) else None
                if hf_id and slug(hf_id) == tok_slug:
                    tokenizer = hf_id
                    break
    if relation is None:
        relation = records_path.parent.parent.name
    if tokenizer is None:
        tokenizer = tok_slug.replace("__", "/")
    return relation, tokenizer

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records_path = Path(args.records).expanduser()
    if not records_path.exists():
        raise SystemExit(f"no such records file: {records_path}")
    out_path = Path(args.out).expanduser()

    records = admission.load_records_file(records_path)
    pool, dropped, counts = admission.subjects_from_records(
        records, require_context_consistent=args.require_context_consistent
    )
    relation, tokenizer = identify(records_path, args)

    shown = (str(records_path.relative_to(REPO_ROOT))
             if records_path.is_absolute() and records_path.is_relative_to(REPO_ROOT)
             else str(records_path))
    source = {
        "kind": "records_file",
        "path": shown,
        "sha256": sha256_file(records_path),
    }

    v1_manifest = v1_sha = None
    if args.extends_v1 is not None:
        v1_manifest = json.loads(args.extends_v1.read_text(encoding="utf-8"))
        v1_sha = sha256_file(args.extends_v1)

    cap_order = None
    if args.reuse_class_cap_order is not None:
        prior = json.loads(args.reuse_class_cap_order.read_text(encoding="utf-8"))
        cap_order = prior.get("admission", {}).get("class_cap", {}).get("order")
        if cap_order is None:
            raise SystemExit(
                f"{args.reuse_class_cap_order} records no "
                f"admission.class_cap.order: it was built before the "
                f"permutation was materialized, so there is nothing to reuse."
            )

    manifest = admission.build_manifest(
        pool,
        relation=relation,
        tokenizer=tokenizer,
        source=source,
        n_shots=args.n_shots,
        support_rule=args.support_rule,
        min_support=args.min_support,
        min_classes=args.min_classes,
        max_classes=args.max_classes,
        class_cap_order=cap_order,
        min_rows=args.min_rows,
        q=args.q,
        outer=args.outer,
        inner=args.inner,
        role_seed=args.role_seed,
        fold_seed=args.fold_seed,
        holdout_divisor=args.holdout_divisor,
        holdout_cap=args.holdout_cap,
        holdout_min_train=args.holdout_min_train,
        require_context_consistent=args.require_context_consistent,
        v1_manifest=v1_manifest,
        v1_manifest_sha256=v1_sha,
        dropped=dropped,
        record_counts=counts,
        notes=[
            "roles and folds only: this manifest chooses no relation, animal, "
            "bijection or epoch",
            "centered numerical rank is state-dependent; run "
            "linroute.admission.check_centered_rank per candidate layer",
            "subjects are grouped by subject string; no alias or entity "
            "resolution exists upstream",
        ],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(admission.manifest_json(manifest), encoding="utf-8")

    for line in admission.summary_lines(manifest):
        print(line)
    print(f"wrote {out_path}")
    return 0 if manifest["checks"]["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
