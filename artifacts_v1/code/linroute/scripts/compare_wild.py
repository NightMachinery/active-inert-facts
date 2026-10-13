#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linroute.report import (
    _fmt,
    regression_rows,
)
from linroute.relation_labels import DEFAULT_PATH as DEFAULT_RELATION_LABELS
from linroute.relation_labels import load as load_relation_labels

from table_wild import (
    add_derived,
    capacity_note,
    present_capacities,
    read_jsonl,
    relation_key,
)

PAIRED_SCORES = [
    ("lin_obj", "delta"),
    ("lin_obj", "gain_over_majority"),
    ("lin_neu_obj", "delta"),
    ("lin_neu_obj", "gain_over_majority"),
]

AVAILABILITY = ["lin_obj_q{q}_available", "lin_neu_obj_q{q}_available"]

def summary_scores(q: int = 16) -> list[str]:
    return [
        "zs_acc", "zs_acc_index", "fs_acc", "head_acc",
        f"lin_obj_q{q}_delta", f"lin_obj_q{q}_acc", f"lin_obj_q{q}_gain_over_majority",
        f"lin_neu_obj_q{q}_delta", f"lin_neu_obj_q{q}_gain_over_majority",
        f"lin_cls_rel_q{q}_acc", f"lin_cls_neu_q{q}_acc",
        "lu_obj_delta_cos_affine_cv", "lu_obj_delta_cos_const_cv", "lu_obj_excess_cv",
        "lu_neu_obj_delta_cos_affine_cv", "lu_neu_obj_excess_cv",
    ]

SUMMARY_SCORES = summary_scores()

TARGET_KEYS = ["fs_acc", "zs_acc"]

QWEN_PAIR = ("Qwen__Qwen3.5-9B", "Qwen__Qwen3.5-9B-Base")

def load_models(root: Path) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for path in sorted(root.glob("*/probes.json")):
        rows = [add_derived(r) for r in read_jsonl(path)]
        if rows:
            out[path.parent.name] = {relation_key(r): r for r in rows}
    if not out:
        raise SystemExit(f"no <model>/probes.json under {root}")
    return out

def capacities(models: dict[str, dict[str, dict]]) -> list[int]:
    return present_capacities(
        row for rows in models.values() for row in rows.values()
    )

def shared_relations(models: dict[str, dict[str, dict]]) -> list[str]:
    common = set.intersection(*(set(m) for m in models.values()))
    return sorted(common)

def eligible_relations(
    models: dict[str, dict[str, dict]], qs: Sequence[int] = (16, 64)
) -> list[str]:
    keys = [k.format(q=q) for q in qs for k in AVAILABILITY]
    return [
        rel
        for rel in shared_relations(models)
        if all(bool(rows[rel].get(k)) for rows in models.values() for k in keys)
    ]

def _values(rows: Iterable[dict], key: str) -> np.ndarray:
    out = []
    for row in rows:
        val = row.get(key)
        try:
            out.append(float(val))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return np.asarray(out, dtype=np.float64)

def capacity_means(
    models: dict[str, dict[str, dict]],
    rels: Sequence[str],
    qs: Sequence[int] = (16, 64),
) -> list[dict]:
    lo, hi = qs
    out: list[dict] = []
    for model, rows in models.items():
        picked = [rows[r] for r in rels]
        for stem, tail in PAIRED_SCORES:
            a = _values(picked, f"{stem}_q{lo}_{tail}")
            b = _values(picked, f"{stem}_q{hi}_{tail}")
            ok = np.isfinite(a) & np.isfinite(b)
            diff = b[ok] - a[ok]
            out.append({
                "model": model,
                "score": f"{stem}_{tail}",
                "n": int(ok.sum()),
                f"mean_q{lo}": float(a[ok].mean()) if ok.any() else float("nan"),
                f"mean_q{hi}": float(b[ok].mean()) if ok.any() else float("nan"),
                f"mean_q{hi}_minus_q{lo}":
                    float(diff.mean()) if ok.any() else float("nan"),
                "sd_paired_diff": float(diff.std()) if ok.any() else float("nan"),
                "max_abs_diff": float(np.abs(diff).max()) if ok.any() else float("nan"),
            })
    return out

def capacity_means_columns(qs: Sequence[int] = (16, 64)) -> list[str]:
    lo, hi = qs
    return ["model", "score", "n", f"mean_q{lo}", f"mean_q{hi}",
            f"mean_q{hi}_minus_q{lo}", "sd_paired_diff", "max_abs_diff"]

def capacity_regressions(
    models: dict[str, dict[str, dict]],
    rels: Sequence[str],
    qs: Sequence[int] = (16, 64),
) -> list[dict]:
    lo, hi = qs
    out: list[dict] = []
    for model, rows in models.items():
        picked = [rows[r] for r in rels]
        per_q = {}
        for q in (lo, hi):
            keys = [f"{stem}_q{q}_{tail}" for stem, tail in PAIRED_SCORES]
            per_q[q] = {
                (r["score"], r["target"]): r for r in regression_rows(picked, keys, TARGET_KEYS)
            }
        for stem, tail in PAIRED_SCORES:
            for target in TARGET_KEYS:
                a = per_q[lo][(f"{stem}_q{lo}_{tail}", target)]
                b = per_q[hi][(f"{stem}_q{hi}_{tail}", target)]
                row = {"model": model, "score": f"{stem}_{tail}", "target": target,
                       f"n_q{lo}": a["n"], f"n_q{hi}": b["n"]}
                for field in ("pearson", "spearman", "adj_r2_gain", "coef_per_sd"):
                    row[f"{field}_q{lo}"] = a.get(field)
                    row[f"{field}_q{hi}"] = b.get(field)
                out.append(row)
    return out

def capacity_regressions_columns(qs: Sequence[int] = (16, 64)) -> list[str]:
    lo, hi = qs
    return (
        ["model", "score", "target", f"n_q{lo}", f"n_q{hi}"]
        + [f"{field}_q{q}"
           for field in ("pearson", "spearman", "adj_r2_gain", "coef_per_sd")
           for q in (lo, hi)]
    )

def cross_model_means(
    models: dict[str, dict[str, dict]],
    rels: Sequence[str],
    scores: Sequence[str] = SUMMARY_SCORES,
) -> tuple[list[dict], list[str]]:
    names = list(models)
    rows: list[dict] = []
    for score in scores:
        cols = {m: _values([models[m][r] for r in rels], score) for m in names}
        ok = np.all([np.isfinite(v) for v in cols.values()], axis=0)
        record: dict = {"score": score, "n": int(ok.sum())}
        for model in names:
            record[model] = float(cols[model][ok].mean()) if ok.any() else float("nan")
        rows.append(record)
    return rows, names

def qwen_pair_diffs(
    models: dict[str, dict[str, dict]],
    rels: Sequence[str],
    scores: Sequence[str] = SUMMARY_SCORES,
) -> list[dict]:
    post, base = QWEN_PAIR
    if post not in models or base not in models:
        return []
    out: list[dict] = []
    for score in scores:
        a = _values([models[post][r] for r in rels], score)
        b = _values([models[base][r] for r in rels], score)
        ok = np.isfinite(a) & np.isfinite(b)
        diff = a[ok] - b[ok]
        out.append({
            "score": score,
            "n": int(ok.sum()),
            "mean_post": float(a[ok].mean()) if ok.any() else float("nan"),
            "mean_base": float(b[ok].mean()) if ok.any() else float("nan"),
            "mean_paired_diff": float(diff.mean()) if ok.any() else float("nan"),
            "sd_paired_diff": float(diff.std()) if ok.any() else float("nan"),
            "n_positive": int((diff > 0).sum()),
        })
    return out

def _table(rows: Sequence[dict], columns: Sequence[str], org: bool) -> str:
    sep = (
        "|" + "|".join("-" * (len(c) + 2) for c in columns) + "|"
        if org
        else "| " + " | ".join("---" for _ in columns) + " |"
    )
    body = ["| " + " | ".join(columns) + " |", sep]
    for row in rows:
        body.append("| " + " | ".join(_fmt(row.get(c)) for c in columns) + " |")
    return "\n".join(body)

def render(sections: Sequence[tuple[str, str, Sequence[dict], Sequence[str]]],
           title: str, preamble: Sequence[str], org: bool) -> str:
    h1, h2 = ("* ", "** ") if org else ("# ", "## ")
    parts = [h1 + title, ""] + list(preamble)
    for heading, prose, rows, columns in sections:
        parts += ["", h2 + heading, "", prose]

        if columns:
            parts += ["", _table(rows, columns, org)]
    return "\n".join(parts) + "\n"

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results", help="results root")
    ap.add_argument("--out-dir", default=None, help="default: the results root")
    ap.add_argument("--relation-labels", type=Path, default=DEFAULT_RELATION_LABELS,
                    help="canonical paper-repository relation-label map")
    return ap.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    labels = load_relation_labels(args.relation_labels)
    root = Path(args.results)
    out_dir = Path(args.out_dir) if args.out_dir else root

    models = load_models(root)
    labels.assert_complete(
        (relation_id for rows in models.values() for relation_id in rows),
        "compare_wild export",
    )
    shared = shared_relations(models)
    qs = capacities(models)

    paired = tuple(qs[:2]) if len(qs) >= 2 else None
    lo = qs[0] if qs else 16

    sections: list[tuple[str, str, Sequence[dict], Sequence[str]]] = []
    if paired is None:
        eligible = []
        only = f"q{lo}" if qs else "no"
        print(f"[compare_wild] {len(models)} models, {len(shared)} shared relations, "
              f"only capacity {only} in these rows; capacity comparison skipped")
        sections.append((
            "Capacity: skipped",
            f"These rows report only {only}, so there is no second capacity to "
            "compare against and both capacity sections are omitted. A sweep emits "
            "a second capacity only when `ProbeConfig.q_secondary` is set; it is "
            "`None` by default. The rest of this report is unchanged.",
            [],
            [],
        ))
    else:
        qa, qb = paired
        eligible = eligible_relations(models, paired)
        print(f"[compare_wild] {len(models)} models, {len(shared)} shared relations, "
              f"{len(eligible)} eligible at both q{qa} and q{qb} everywhere")
        means = capacity_means(models, eligible, paired)
        regs = capacity_regressions(models, eligible, paired)
        sections += [
            (
                f"Capacity, on the {len(eligible)} relations eligible at both q{qa} and "
                f"q{qb} in all {len(models)} models",
                f"Paired over relations: `mean_q{qb}_minus_q{qa}` is the mean of the "
                "per-relation difference, not the difference of two means over "
                "different relations. `sd_paired_diff` is the population SD of those "
                "differences across relations, which is a spread, not a standard error.",
                means,
                capacity_means_columns(paired),
            ),
            (
                f"The predeclared regression on that same subset, q{qa} beside q{qb}",
                "`accuracy ~ 1 + entropy + log |O| + log n`, augmented with one score at a "
                "time (`linroute/report.py`). Both capacities are fitted on the identical "
                f"relations, so the q{qa}/q{qb} gap here is capacity alone. These are "
                "in-sample adjusted-fit statistics on 16 relations, not validated "
                "predictive gains.",
                regs,
                capacity_regressions_columns(paired),
            ),
        ]

    scores = summary_scores(lo)
    summary, model_names = cross_model_means(models, shared, scores)
    pair = qwen_pair_diffs(models, shared, scores)

    sections += [
        (
            f"Cross-model means on the {len(shared)} shared relations",
            "Each score is averaged over the relations where it is finite in EVERY "
            "model, so every column of a row is the same relation set; `n` says how "
            f"many that is. Capacity-restricted scores (q{lo}) therefore drop to their "
            "eligible subset in all columns at once.",
            summary,
            ["score", "n"] + model_names,
        ),
        (
            "Qwen3.5-9B minus Qwen3.5-9B-Base, paired per relation",
            "Post-trained minus base on the relations where both are finite. "
            "`n_positive` counts relations where the post-trained model scores higher; "
            "with n around 40 and no correction for multiplicity, read these as "
            "descriptive.",
            pair,
            ["score", "n", "mean_post", "mean_base",
             "mean_paired_diff", "sd_paired_diff", "n_positive"],
        ),
    ]

    stale = {
        m: sum(1 for r in shared if "zs_acc_index" not in models[m][r]) for m in model_names
    }
    provenance = (
        "All rows carry `zs_acc_index`, so `zs_acc` means first-token equivalence "
        "everywhere."
        if not any(stale.values())
        else "WARNING: rows without `zs_acc_index` were fitted before 6b693fe and "
        "their `zs_acc` is the candidate-INDEX score, so those columns mix two "
        "conventions: "
        + ", ".join(f"`{m}` {n}/{len(shared)}" for m, n in stale.items() if n)
        + "."
    )

    preamble = [
        f"Source: `{root}`. Models: " + ", ".join(f"`{m}`" for m in model_names) + ".",
        "",
        (
            f"Shared relations: {len(shared)}. Eligible at both q{paired[0]} and "
            f"q{paired[1]} in every model: {len(eligible)}."
            if paired is not None
            else f"Shared relations: {len(shared)}. Only one capacity "
            f"(q{lo}) is reported by these rows, so no capacity comparison is made."
        ),
        "",
        provenance,
        "",
        "Generated by `scripts/compare_wild.py`; derived columns as in "
        "`scripts/table_wild.py`.",
    ]
    mixed = capacity_note([r for rows in models.values() for r in rows.values()])
    if mixed:
        preamble[4:4] = ["WARNING: " + mixed, ""]

    for org, ext in ((False, "md"), (True, "org")):
        path = out_dir / f"compare_wild.{ext}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            render(sections, "Matched-subset comparisons", preamble, org),
            encoding="utf-8",
        )
        print(f"[compare_wild] wrote {path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
