#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "--allow-download" not in sys.argv:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

from linroute import admission, inventory, prompts, relations
from linroute import engineered_relations as engineered_rel
from linroute.cache import slug
from linroute.provenance import REPO_ROOT, git_commit

SCHEMA = "linroute/admission/screen/1"
GENERATOR = "scripts/screen_relations.py"
GENERATOR_VERSION = "1.0.0"

ROSTER = (
    ("allenai/Olmo-3-1025-7B", "OLMo-3-1025-7B"),
    ("Qwen/Qwen3.5-9B", "Qwen3.5-9B"),
)

VETOES = ("prefix_not_uniform", "boundary_mismatch")

RELATION_DIR_ENV = "LINROUTE_RELATIONS_DIR"

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    d = admission.DEFAULTS
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
        required=True,
        help="output root; screen/ and manifests/ are created under it.  "
        "Must not be inside the repository: these records are not committed",
    )
    p.add_argument(
        "--tokenizer", action="append", dest="tokenizers", metavar="HF_ID",
        help=f"repeatable; default is the roster "
        f"({', '.join(i for i, _ in ROSTER)})",
    )
    p.add_argument(
        "--relations", default="all",
        help="'all' or a comma-separated list of relation names, for a spot check",
    )
    p.add_argument("--min-support", type=int, default=d["min_support"])
    p.add_argument("--min-classes", type=int, default=d["min_classes"])
    p.add_argument("--max-classes", type=int, default=d["max_classes"])
    p.add_argument("--min-rows", type=int, default=d["min_rows"])
    p.add_argument("--q", type=int, default=d["q"])
    p.add_argument("--outer", type=int, default=d["outer"])
    p.add_argument("--inner", type=int, default=d["inner"])
    p.add_argument("--role-seed", type=int, default=d["role_seed"])
    p.add_argument("--fold-seed", type=int, default=d["fold_seed"])
    p.add_argument("--support-rule", choices=admission.SUPPORT_RULES,
                   default=d["support_rule"])
    p.add_argument(
        "--no-require-context-consistent", dest="require_context_consistent",
        action="store_false",
        help="keep records whose rendered answer token differs from the "
        "isolated one instead of dropping them as context_inconsistent",
    )
    p.add_argument("--repeats", type=int, default=3,
                   help="fine-tuning repeats per admitted (relation, model), "
                   "used only to report the implied number of runs")
    p.add_argument("--allow-download", action="store_true",
                   help="let the tokenizer loader reach the network")
    p.set_defaults(require_context_consistent=d["require_context_consistent"])
    args = p.parse_args(argv)
    args.relations_dir = args.relations_dir or os.environ.get(RELATION_DIR_ENV)
    if not args.relations_dir:
        raise SystemExit(
            f"--relations-dir (or ${RELATION_DIR_ENV}) must point at the "
            f"Hernandez checkout's data directory"
        )
    args.tokenizers = tuple(args.tokenizers or [i for i, _ in ROSTER])
    return args

def short_name(hf_id: str) -> str:
    for ident, name in ROSTER:
        if ident == hf_id:
            return name
    return hf_id.split("/")[-1]

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
            + "\n\nNothing was written."
        )
    return loaded

def candidates(relations_dir: Path) -> list[tuple[relations.RawRelation, str, dict]]:
    out: list[tuple[relations.RawRelation, str, dict]] = []
    for raw in relations.load_relations(relations_dir):
        entry = prompts.wild_prompts(raw.name)
        out.append((raw, "hernandez", {
            "kind": "external_checkout",
            "source_file": entry.source_file,
            "source_sha256": entry.source_sha256,
            "verified_against": "data/prompts/wild.yaml",
        }))
    out.append((relations.town_region_relation(), "sandbox", {
        "kind": "repository",
        "files": ["data/prompts/sandbox.yaml", "data/sandbox/towns_candidates.tsv",
                  "data/sandbox/regions.tsv"],
    }))
    for built in engineered_rel.build_all():
        out.append((built.relation, "engineered", dict(built.provenance)))
    return out

def vetoes_for(summary: dict) -> list[str]:
    found = []
    if not summary["prefix_uniform"]:
        found.append("prefix_not_uniform")
    if summary["n_included"] and not summary["n_context_consistent"]:
        found.append("boundary_mismatch")
    return found

def k_at(manifest: dict, m: int) -> int:
    for entry in manifest["support_counts"]:
        if entry["m"] == m:
            return entry["k"]
    return 0

def support_profile(manifest: dict) -> list[int]:
    return sorted(c["support"] for c in manifest["classes"] if c["retained"])

def holdout_profile(manifest: dict) -> list[int]:
    return sorted(admission.per_class_holdout(manifest).values())

def support_summary(supports: list[int]) -> dict:
    return {
        "min": supports[0] if supports else None,
        "median": supports[(len(supports) - 1) // 2] if supports else None,
        "max": supports[-1] if supports else None,
    }

def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)

def screen_one(
    raw: relations.RawRelation,
    tokenizer,
    *,
    hf_id: str,
    source: str,
    provenance: dict,
    args: argparse.Namespace,
    out_root: Path,
    n_shots: int,
    commit: str,
) -> tuple[dict, inventory.RelationInventory]:
    rel_slug = slug(raw.name)
    model = short_name(hf_id)
    inv = inventory.build_inventory(
        raw, tokenizer, tokenizer_id=hf_id, relation_slug=rel_slug,
    )
    body = inventory.records_jsonl(inv)
    records_path = out_root / "screen" / "records" / slug(hf_id) / f"{rel_slug}.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_text(body, encoding="utf-8")
    records_sha = _digest(body)
    summary = inv.summary(records_sha256=records_sha)

    pool, dropped, counts = admission.subjects_from_inventory(
        inv, require_context_consistent=args.require_context_consistent
    )
    vetoes = vetoes_for(summary)
    manifest = admission.build_manifest(
        pool,
        relation=raw.name,
        tokenizer=hf_id,
        source={
            "kind": "screen_inventory",
            "relation_source": source,
            "records_path": str(records_path),
            "records_sha256": records_sha,
            "provenance": provenance,
        },
        n_shots=n_shots,
        support_rule=args.support_rule,
        min_support=args.min_support,
        min_classes=args.min_classes,
        max_classes=args.max_classes,
        min_rows=args.min_rows,
        q=args.q,
        outer=args.outer,
        inner=args.inner,
        role_seed=args.role_seed,
        fold_seed=args.fold_seed,
        require_context_consistent=args.require_context_consistent,
        dropped=dropped,
        record_counts=counts,
        vetoes=vetoes,
        notes=[
            f"written by {GENERATOR} as part of the candidate screen",
            "roles and folds only: this manifest chooses no relation, animal, "
            "bijection or epoch",
            "centered numerical rank is state-dependent; run "
            "linroute.admission.check_centered_rank per candidate layer",
            "subjects are grouped by subject string; no alias or entity "
            "resolution exists upstream",
        ],
        git_commit=commit,
    )
    manifest_text = admission.manifest_json(manifest)
    manifest_path = out_root / "manifests" / slug(hf_id) / f"{rel_slug}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest_text, encoding="utf-8")

    adm = manifest["admission"]
    round_trips = (summary["n_subject_span_round_trip_failures"]
                   + summary["n_answer_span_round_trip_failures"])
    row = {
        "relation": raw.name,
        "relation_slug": rel_slug,
        "relation_type": raw.relation_type,
        "relation_source": source,
        "model": model,
        "tokenizer": hf_id,
        "admitted": adm["admitted"],
        "reasons": list(adm["failing_conditions"]),
        "vetoes": list(adm["vetoes"]),

        "n_source_rows": summary["n_source_rows"],
        "n_records": summary["n_records"],
        "n_included": summary["n_included"],
        "n_excluded": summary["n_excluded"],
        "n_excluded_by_reason": dict(summary["excluded_by_reason"]),
        "n_effective_classes": summary["n_effective_classes"],
        "n_merged_classes": summary["n_merged_classes"],

        "prefix_uniform": summary["prefix_uniform"],
        "skipped_prefixes": summary["skipped_prefixes"],
        "n_context_inconsistent": summary["n_context_mismatched"],
        "n_round_trip_failures": round_trips,

        "support_rule": manifest["parameters"]["support_rule"],
        "n_support": manifest["support"]["n_shots"],
        "n_classes_pool": adm["n_classes_pool"],
        "n_classes": adm["n_classes_retained"],
        "n_classes_excluded": adm["n_classes_excluded"],

        "n_train": adm["n_rows"],
        "n_heldout": adm["n_held_out"],
        "holdout_rule": admission.holdout_rule_of(manifest),
        "holdout_protocol": admission.protocol_of(
            admission.holdout_rule_of(manifest)),
        "holdout_profile": holdout_profile(manifest),
        "holdout_summary": support_summary(holdout_profile(manifest)),
        "holdout_fallback_applied": manifest["holdout"]["fallback_applied"],
        "holdout_fallback_depth": manifest["holdout"]["fallback_depth"],
        "n_train_target": adm["n_rows_target"],
        "K": {"5": k_at(manifest, 5), "6": k_at(manifest, 6)},
        "support_profile": support_profile(manifest),
        "support_summary": support_summary(support_profile(manifest)),
        "class_cap": dict(adm["class_cap"]),

        "checks_passed": manifest["checks"]["passed"],
        "n_check_failures": manifest["checks"]["n_failures"],
        "min_outer_fitting_rows": manifest["checks"]["min_outer_fitting_rows"],
        "min_inner_fitting_rows": manifest["checks"]["min_inner_fitting_rows"],
        "bound": dict(manifest["checks"]["bound"]),

        "partition_hash": summary["partition_hash"],
        "subjects_hash": summary["subjects_hash"],
        "records": str(records_path),
        "records_sha256": records_sha,
        "manifest": str(manifest_path),
        "manifest_sha256": _digest(manifest_text),
    }
    return row, inv

def duplicate_groups(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row["n_included"]:
            groups[(row["model"], row["subjects_hash"],
                    row["partition_hash"])].append(row)
    out = []
    for (model, subjects, partition), members in sorted(groups.items()):
        if len(members) > 1:
            out.append({
                "model": model,
                "subjects_hash": subjects,
                "partition_hash": partition,
                "relations": sorted(m["relation"] for m in members),
                "relation_sources": sorted({m["relation_source"] for m in members}),
            })
    return out

def relation_entries(rows: list[dict], agreement: dict[str, dict]) -> list[dict]:
    by_relation: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_relation[row["relation"]].append(row)

    entries = []
    for name, members in sorted(by_relation.items()):
        members = sorted(members, key=lambda r: r["model"])
        first = next(
            (m for m in members if m["model"] == ROSTER[0][1]), members[0]
        )
        refused = [m for m in members if not m["admitted"]]
        reasons: list[str] = []
        for m in refused:
            reasons.append(f"{m['model']}: " + (", ".join(m["reasons"]) or "not admitted"))
        entries.append({
            "relation": name,
            "relation_slug": first["relation_slug"],
            "relation_type": first["relation_type"],
            "relation_source": first["relation_source"],
            "admitted": not refused,
            "reason": "; ".join(reasons) or None,
            "scalars_from_model": first["model"],
            "n_classes": first["n_classes"],
            "n_train": first["n_train"],
            "n_heldout": first["n_heldout"],
            "holdout_rule": first["holdout_rule"],
            "holdout_profile": first["holdout_profile"],
            "support_profile": first["support_profile"],
            "support_summary": first["support_summary"],
            "K": first["K"],
            "prefix_uniform": first["prefix_uniform"],
            "manifest": first["manifest"],
            "manifest_sha256": first["manifest_sha256"],
            "models_agree_on_admission": len({m["admitted"] for m in members}) == 1,
            "tokenizers_agree_on_partition": agreement.get(name, {}).get(
                "agree_up_to_class_names"
            ),
            "per_model": {
                m["model"]: {
                    "admitted": m["admitted"],
                    "reasons": m["reasons"],
                    "n_classes": m["n_classes"],
                    "n_train": m["n_train"],
                    "n_heldout": m["n_heldout"],
                    "holdout_rule": m["holdout_rule"],
                    "holdout_profile": m["holdout_profile"],
                    "holdout_fallback_depth": m["holdout_fallback_depth"],
                    "K": m["K"],
                    "class_cap_applied": m["class_cap"]["applied"],
                    "checks_passed": m["checks_passed"],
                    "manifest": m["manifest"],
                    "manifest_sha256": m["manifest_sha256"],
                }
                for m in members
            },
        })
    return entries

def _origin(row: dict) -> str:
    if row["relation_source"] == row["relation_type"]:
        return row["relation_source"]
    return f"{row['relation_source']}, {row['relation_type']}"

def markdown(blob: dict) -> str:
    rows = blob["rows"]
    params = blob["parameters"]
    lines = [
        "# Candidate relation screen",
        "",
        f"Written by `{blob['generator']}` "
        f"{blob['generator_version']} at commit `{blob['git_commit']}`.  No "
        f"timestamp: re-running over unchanged sources must reproduce this "
        f"file byte for byte, and a clock is the one thing that would stop it.",
        "",
        f"{blob['totals']['n_relations']} candidate relations "
        f"({blob['totals']['n_by_source']}) screened under "
        f"{blob['totals']['n_models']} roster models, "
        f"{blob['totals']['n_rows']} (relation, model) verdicts in all.",
        "",
        "Admission is a data-availability screen and nothing else: a class is "
        f"retained at support >= {params['min_support']} after the "
        f"{params['n_shots']}-subject reservation, a comparison is admitted at "
        f">= {params['min_classes']} retained classes and "
        f"N >= {params['min_rows']} rows, at most {params['max_classes']} "
        "classes are kept, and no outcome of any model is consulted.",
        "",
        "## Totals",
        "",
    ]
    for line in blob["totals"]["lines"]:
        lines.append(f"- {line}")
    lines += ["", "## Admitted", ""]
    for _, model in blob["models"]:
        admitted = [r for r in rows if r["model"] == model and r["admitted"]]
        lines.append(f"### {model} -- {len(admitted)} admitted")
        lines.append("")
        for row in sorted(admitted, key=lambda r: (-r["n_train"], r["relation"])):
            profile = row["support_summary"]
            held = row["holdout_summary"]
            cap = " (78-class cap applied)" if row["class_cap"]["applied"] else ""
            fell = (f"; fallback returned {row['holdout_fallback_depth']} "
                    f"held-out subject(s)" if row["holdout_fallback_applied"]
                    else "")
            lines.append(
                f"- **{row['relation']}** ({_origin(row)}): "
                f"|E| = {row['n_classes']}, "
                f"N = {row['n_train']}, |H| = {row['n_heldout']}; supports "
                f"min {profile['min']}, median {profile['median']}, max "
                f"{profile['max']}; h_c min {held['min']}, median "
                f"{held['median']}, max {held['max']}; K(5) = {row['K']['5']}, "
                f"K(6) = {row['K']['6']}{cap}{fell}"
            )
        lines.append("")
    lines += ["## Rejected", ""]
    for _, model in blob["models"]:
        rejected = [r for r in rows if r["model"] == model and not r["admitted"]]
        lines.append(f"### {model} -- {len(rejected)} rejected")
        lines.append("")
        for row in sorted(rejected, key=lambda r: r["relation"]):
            lines.append(
                f"- **{row['relation']}** ({_origin(row)}): "
                f"{', '.join(row['reasons'])} -- |E| = {row['n_classes']} of "
                f"{row['n_classes_pool']} classes in the pool, "
                f"N = {row['n_train']}, {row['n_included']} included records "
                f"of {row['n_source_rows']} source rows"
            )
        lines.append("")

    lines += ["## Relations the two tokenizers disagree about", ""]
    disagree = [e for e in blob["by_relation"]
                if e["tokenizers_agree_on_partition"] is False]
    if not disagree:
        lines.append("- none: every relation induces the same partition of its "
                     "common included subjects under both tokenizers")
    for entry in disagree:
        per = entry["per_model"]
        shared = blob["agreement"][entry["relation"]]
        lines.append(
            f"- **{entry['relation']}**: "
            + "; ".join(f"{m} |E| = {v['n_classes']}, N = {v['n_train']}"
                        for m, v in sorted(per.items()))
            + f" -- on the {shared['n_common_subjects']} subjects both include, "
            f"the induced partitions differ ("
            + ", ".join(f"{m} -> `{h}`" for m, h in
                        sorted(shared["restricted_partition_hash"].items()))
            + "); effective classes "
            + ", ".join(f"{m} {n}" for m, n in sorted(shared["n_classes"].items()))
            + "; a model-specific comparison, not a replication"
        )
    lines.append("")

    lines += ["## Relations the two models disagree about admitting", ""]
    split = [e for e in blob["by_relation"] if not e["models_agree_on_admission"]]
    if not split:
        lines.append("- none: every relation is admitted, or refused, under both")
    for entry in split:
        lines.append(
            f"- **{entry['relation']}**: "
            + "; ".join(
                f"{m} {'admitted' if v['admitted'] else 'rejected (' + ', '.join(v['reasons']) + ')'}"
                for m, v in sorted(entry["per_model"].items())
            )
        )
    lines.append("")

    lines += ["## Duplicates", ""]
    if not blob["duplicates"]:
        lines.append("- none: no two candidates include the same subjects AND "
                     "cut them into the same classes")
    for group in blob["duplicates"]:
        lines.append(
            f"- under {group['model']}: {', '.join(group['relations'])} share "
            f"subject set `{group['subjects_hash']}` and partition "
            f"`{group['partition_hash']}` (sources: "
            f"{', '.join(group['relation_sources'])})"
        )
    lines.append("")

    lines += ["## Vetoes", ""]
    vetoed = [r for r in rows if r["vetoes"]]
    if not vetoed:
        lines.append("- none: every candidate's included records share one "
                     "skipped prefix, and every one has at least one record "
                     "whose rendered answer boundary matches the isolated one")
    for row in sorted(vetoed, key=lambda r: (r["relation"], r["model"])):
        lines.append(
            f"- **{row['relation']}** under {row['model']}: "
            f"{', '.join(row['vetoes'])} "
            f"({len(row['skipped_prefixes'])} distinct skipped prefixes, "
            f"{row['n_context_inconsistent']} context-inconsistent records)"
        )
    lines.append("")

    lines += [
        "## What this does not say", "",
        "- Nothing about accuracy: no model was run, and no class was kept or "
        "dropped for how well anything scores on it.",
        "- Nothing about centered numerical rank, which depends on the states "
        "and not on the subject ids; run "
        "`linroute.admission.check_centered_rank` per candidate layer before "
        "fitting (`docs/admission.md`).",
        "- Nothing about entities: subjects are grouped by string, so a "
        "relation listing one person under two spellings counts two subjects.",
        "",
    ]
    return "\n".join(lines) + "\n"

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_root = Path(args.out).expanduser().resolve()
    if out_root.is_relative_to(REPO_ROOT):
        raise SystemExit(
            f"--out {out_root} is inside the repository; the screen's per-row "
            f"records are not committable (docs/inventory.md).  Point it at a "
            f"scratch directory."
        )
    (out_root / "screen" / "records").mkdir(parents=True, exist_ok=True)
    (out_root / "manifests").mkdir(parents=True, exist_ok=True)

    pool = candidates(Path(args.relations_dir).expanduser())
    if args.relations != "all":
        wanted = {n.strip() for n in args.relations.split(",") if n.strip()}
        unknown = wanted - {raw.name for raw, _, _ in pool}
        if unknown:
            raise SystemExit(f"no such relation: {sorted(unknown)}")
        pool = [c for c in pool if c[0].name in wanted]

    tokenizers = load_tokenizers(args.tokenizers)
    n_shots = prompts.few_shot_spec().n_shots
    commit = git_commit()

    rows: list[dict] = []
    agreement: dict[str, dict] = {}
    for raw, source, provenance in pool:
        invs = []
        for hf_id, tok in tokenizers.items():
            row, inv = screen_one(
                raw, tok, hf_id=hf_id, source=source, provenance=provenance,
                args=args, out_root=out_root, n_shots=n_shots, commit=commit,
            )
            rows.append(row)
            invs.append(inv)
        agreement[raw.name] = inventory.compare_partitions(invs)

    rows.sort(key=lambda r: (r["relation"], r["model"]))
    models = [(hf_id, short_name(hf_id)) for hf_id in args.tokenizers]
    entries = relation_entries(rows, agreement)
    duplicates = duplicate_groups(rows)

    by_source: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_source[row["relation_source"]].add(row["relation"])
    admitted_per_model = {
        model: sorted(r["relation"] for r in rows
                      if r["model"] == model and r["admitted"])
        for _, model in models
    }
    n_runs = sum(len(v) for v in admitted_per_model.values()) * args.repeats
    admitted_rows = [r for r in rows if r["admitted"]]
    total_lines = [
        f"{len(admitted_per_model[m])} of {len(entries)} relations admitted "
        f"under {m}" for _, m in models
    ]
    total_lines.append(
        f"{sum(1 for e in entries if e['admitted'])} relations admitted under "
        f"every roster model"
    )
    total_lines.append(
        f"{n_runs} fine-tuning runs implied ({args.repeats} repeats per "
        f"admitted (relation, model) pair)"
    )
    if admitted_rows:
        biggest_n = max(admitted_rows, key=lambda r: r["n_train"])
        biggest_e = max(admitted_rows, key=lambda r: r["n_classes"])
        total_lines.append(
            f"largest N among admitted: {biggest_n['n_train']} "
            f"({biggest_n['relation']} under {biggest_n['model']})"
        )
        total_lines.append(
            f"largest |E| among admitted: {biggest_e['n_classes']} "
            f"({biggest_e['relation']} under {biggest_e['model']})"
        )
    total_lines.append(
        f"{sum(1 for r in rows if r['class_cap']['applied'])} (relation, model) "
        f"pairs hit the {args.max_classes}-class cap"
    )
    total_lines.append(
        f"{sum(1 for r in rows if not r['prefix_uniform'])} pairs have a "
        f"non-uniform skipped prefix, "
        f"{sum(1 for r in rows if 'boundary_mismatch' in r['vetoes'])} have a "
        f"whole-relation boundary mismatch"
    )

    blob = {
        "schema": SCHEMA,
        "generator": GENERATOR,
        "generator_version": GENERATOR_VERSION,
        "git_commit": commit,
        "relations_dir": str(Path(args.relations_dir).expanduser()),
        "out": str(out_root),
        "models": models,
        "vetoes": list(VETOES),
        "parameters": {
            "n_shots": n_shots,
            "support_rule": args.support_rule,
            "min_support": args.min_support,
            "min_classes": args.min_classes,
            "max_classes": args.max_classes,
            "min_rows": args.min_rows,
            "q": args.q,
            "outer": args.outer,
            "inner": args.inner,
            "role_seed": args.role_seed,
            "fold_seed": args.fold_seed,
            "require_context_consistent": args.require_context_consistent,
            "repeats": args.repeats,
        },
        "totals": {
            "n_relations": len(entries),
            "n_models": len(models),
            "n_rows": len(rows),
            "n_by_source": ", ".join(
                f"{len(v)} {k}" for k, v in sorted(by_source.items())
            ),
            "n_admitted_per_model": {m: len(v)
                                     for m, v in admitted_per_model.items()},
            "n_admitted_every_model": sum(1 for e in entries if e["admitted"]),
            "n_runs_implied": n_runs,
            "lines": total_lines,
        },
        "admitted_per_model": admitted_per_model,
        "duplicates": duplicates,
        "agreement": {name: agreement[name] for name in sorted(agreement)},
        "by_relation": entries,
        "rows": rows,
    }

    screen_dir = out_root / "screen"
    (screen_dir / "screen.json").write_text(
        json.dumps(blob, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (screen_dir / "screen.md").write_text(markdown(blob), encoding="utf-8")

    for line in total_lines:
        print(line)
    print(f"wrote {screen_dir / 'screen.json'} and {screen_dir / 'screen.md'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
