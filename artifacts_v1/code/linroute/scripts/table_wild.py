#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linroute.relation_labels import DEFAULT_PATH as DEFAULT_RELATION_LABELS
from linroute.relation_labels import load as load_relation_labels
from linroute.relation_labels import row_relation_id
from linroute.report import (
    regression_rows,
    write_csv,
    write_markdown,
    write_org,
)

DEFAULT_QS = (16, 64)

CAPACITY_RE = re.compile(r"_q(\d+)_")

def _probe_columns(q: int) -> list[str]:
    cols: list[str] = []
    for probe in ("lin_obj", "lin_neu_obj"):
        cols += [
            f"{probe}_q{q}_available", f"{probe}_q{q}_delta", f"{probe}_q{q}_acc",
            f"{probe}_q{q}_base_acc", f"{probe}_q{q}_majority_acc",
            f"{probe}_q{q}_gain_over_majority",
            f"{probe}_q{q}_freq_acc", f"{probe}_q{q}_agree_head",
            f"{probe}_q{q}_modal_layer",
            f"{probe}_q{q}_shuffle_delta_mean", f"{probe}_q{q}_shuffle_delta_std",
            f"{probe}_q{q}_layer_max_biased_delta",
            f"{probe}_q{q}_layer_max_biased_layer",
        ]
    for probe in ("lin_cls_rel", "lin_cls_neu"):
        cols += [
            f"{probe}_q{q}_acc", f"{probe}_q{q}_layer",
            f"{probe}_q{q}_shuffle_acc_mean", f"{probe}_q{q}_shuffle_acc_std",
        ]
    return cols

TABLE_COLUMNS = (
    [
        "model", "relation", "relation_slug", "relation_type",
        "n", "n_objects", "k_per_object", "top1_share", "entropy", "entropy_norm",
        "cand_token_collision_rate",
        "zs_acc", "fs_acc", "head_acc",
    ]
    + _probe_columns(16)
    + _probe_columns(64)
    + [
        "lin_cls_majority", "lin_cls_chance",
        "lu_obj_delta_cos_affine_cv", "lu_obj_delta_cos_translation_cv",
        "lu_obj_delta_cos_const_cv", "lu_obj_excess_cv",
        "lu_obj_delta_cos_affine_split",
        "lu_obj_shuffle_delta_cos_mean", "lu_obj_shuffle_delta_cos_std",
        "lu_neu_obj_delta_cos_affine_cv", "lu_neu_obj_delta_cos_translation_cv",
        "lu_neu_obj_delta_cos_const_cv", "lu_neu_obj_excess_cv",
        "lu_neu_obj_shuffle_delta_cos_mean", "lu_neu_obj_shuffle_delta_cos_std",
        "lu_subject_layer", "lu_object_layer",
    ]
)

SCORE_KEYS = [
    "lin_obj_q16_delta", "lin_obj_q16_layer_max_biased_delta",
    "lin_neu_obj_q16_delta",
    "lin_obj_q64_delta", "lin_neu_obj_q64_delta",
    "lin_cls_rel_q16_acc", "lin_cls_neu_q16_acc",
    "lu_obj_delta_cos_affine_cv", "lu_obj_delta_cos_translation_cv",
    "lu_neu_obj_delta_cos_affine_cv",
    "lu_obj_excess_cv", "lu_neu_obj_excess_cv",
    "lin_obj_q16_gain_over_majority",
]
TARGET_KEYS = ["fs_acc", "zs_acc"]
CORR_COLUMNS = [
    "model", "score", "target", "n",
    "coef", "coef_per_sd", "adj_r2_base", "adj_r2_aug", "adj_r2_gain",
    "pearson", "spearman",
]

LU_DERIVED_COLUMNS = [
    (f"{probe}_excess_cv", f"{probe}_delta_cos_affine_cv", f"{probe}_delta_cos_const_cv")
    for probe in ("lu_obj", "lu_neu_obj")
]

def derived_columns(qs: Sequence[int] = DEFAULT_QS) -> list[tuple[str, str, str]]:
    return LU_DERIVED_COLUMNS + [
        (f"{probe}_q{q}_gain_over_majority",
         f"{probe}_q{q}_acc", f"{probe}_q{q}_majority_acc")
        for probe in ("lin_obj", "lin_neu_obj")
        for q in qs
    ]

DERIVED_COLUMNS = derived_columns()

DERIVED_NAMES = frozenset(name for name, _, _ in DERIVED_COLUMNS)

def present_capacities(rows: Iterable[dict]) -> list[int]:
    out: set[int] = set()
    for row in rows:
        for key in row:
            if key in DERIVED_NAMES:
                continue
            m = CAPACITY_RE.search(key)
            if m:
                out.add(int(m.group(1)))
    return sorted(out)

def capacity_counts(rows: Sequence[dict]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in rows:
        for q in present_capacities([row]):
            counts[q] = counts.get(q, 0) + 1
    return counts

def capacity_note(rows: Sequence[dict]) -> str:
    n = len(rows)
    partial = {q: c for q, c in sorted(capacity_counts(rows).items()) if c != n}
    if not partial or not n:
        return ""
    return (
        "Capacities are MIXED across rows: "
        + ", ".join(f"q{q} in {c}/{n} rows" for q, c in partial.items())
        + ". A row fitted with `ProbeConfig.q_secondary = None` carries no key at "
        "the secondary capacity at all, so its cells there are blank; blank means "
        "not measured, not measured-and-unavailable."
    )

def add_derived(row: dict) -> dict:
    for name, left, right in derived_columns(present_capacities([row])):
        a, b = row.get(left), row.get(right)
        row[name] = None if a is None or b is None else float(a) - float(b)
    return row

OPTIONAL_COLUMNS = [
    ("zs_acc_index", "zs_acc"),
    ("first_token_mismatch_rate", "cand_token_collision_rate"),
    ("first_token_mismatch_rate_meta", "cand_token_collision_rate"),
    ("lu_available", "lu_object_layer"),
    ("epoch", "relation_type"),
    ("arm", "relation_type"),
]

def table_columns(rows: Sequence[dict]) -> list[str]:
    qs = present_capacities(rows)
    columns = [
        c for c in TABLE_COLUMNS
        if not (m := CAPACITY_RE.search(c)) or int(m.group(1)) in qs
    ]
    extra = [q for q in qs if q not in DEFAULT_QS]
    if extra:
        anchor = columns.index("lin_cls_majority")
        block: list[str] = []
        for q in extra:
            block += _probe_columns(q)
        columns[anchor:anchor] = block
    for name, after in OPTIONAL_COLUMNS:
        if any(name in row for row in rows):
            columns.insert(columns.index(after) + 1, name)
    return columns

def score_keys(rows: Sequence[dict]) -> list[str]:
    qs = present_capacities(rows)
    return [
        k for k in SCORE_KEYS
        if not (m := CAPACITY_RE.search(k)) or int(m.group(1)) in qs
    ]

def relation_key(row: dict) -> str:
    return row_relation_id(row)

def read_jsonl(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return list({relation_key(r): r for r in rows}.values())

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results", help="results root")
    ap.add_argument("--models", default="all", help="'all' or comma-separated slugs")
    ap.add_argument("--out-dir", default=None, help="default: the results root")
    ap.add_argument("--relation-labels", type=Path, default=DEFAULT_RELATION_LABELS,
                    help="canonical paper-repository relation-label map")
    return ap.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    labels = load_relation_labels(args.relation_labels)
    root = Path(args.results)
    out_dir = Path(args.out_dir) if args.out_dir else root

    paths = sorted(root.glob("*/probes.json"))
    if args.models != "all":
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        paths = [p for p in paths if p.parent.name in wanted]
    if not paths:
        raise SystemExit(f"no <model>/probes.json under {root}")

    table: list[dict] = []
    corr: list[dict] = []
    for path in paths:
        rows = [add_derived(r) for r in sorted(read_jsonl(path), key=relation_key)]
        if not rows:
            continue
        relation_ids = [relation_key(row) for row in rows]
        labels.assert_complete(relation_ids, f"table_wild export {path.parent.name}")
        for row, relation_id in zip(rows, relation_ids):
            row.setdefault("relation_slug", relation_id)
            row["relation"] = labels.label(relation_id)
        table.extend(rows)
        model = rows[0].get("model", path.parent.name)
        for row in regression_rows(rows, score_keys(rows), TARGET_KEYS):
            corr.append({"model": model, **row})
        print(f"[table_wild] {path.parent.name}: {len(rows)} relations")

    qs = present_capacities(table)
    dropped = [q for q in DEFAULT_QS if q not in qs]
    if dropped:
        print("[table_wild] no row reports "
              + ", ".join(f"q{q}" for q in dropped)
              + "; those columns and scores are dropped "
              f"(capacities present: {', '.join(f'q{q}' for q in qs)})")
    mixed = capacity_note(table)
    if mixed:
        print("[table_wild] " + mixed)

    for name, rows, columns in (
        ("table_wild", table, table_columns(table)),
        ("corr_wild", corr, CORR_COLUMNS),
    ):
        write_csv(rows, columns, out_dir / f"{name}.csv")
        write_org(rows, columns, out_dir / f"{name}.org")
        write_markdown(rows, columns, out_dir / f"{name}.md")
        print(f"[table_wild] wrote {out_dir / name}.{{csv,org,md}} ({len(rows)} rows)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
