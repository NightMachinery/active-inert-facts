#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from linroute import relation_sets

SCHEMA = "linroute/claims-report/1"

VARIANTS = {"headline": "headline"}

SUBSETS = ("hernandez_engineered", "bear", "hernandez_bear")
COMPETITORS = (
    ("lincls_reg", "pre_rel_cls_gain", "LIN-CLS-REG s0 PRE-REL (ridge classifier, q=16)"),
    ("linlu", "pre_rel_lu_affine_minus_const", "LIN-LU s0->t0 PRE-REL (affine minus constant)"),
    ("linhead_prerel_s0t0", "head_control_prerel_s0_t0_gain", "LIN-HEAD s0->t0 PRE-REL"),
    ("linhead_newrel_s0t0", "head_control_newrel_s0_t0_gain", "LIN-HEAD s0->t0 NEW-REL"),
)

CARRIED = ("pre_rel_cls_gain", "pre_rel_lu_affine_minus_const", "head_control_prerel_s0_t0_gain",
           "head_control_newrel_s0_t0_gain", "primary_probe_gain", "head_se_te_gain",
           "n_heldout", "chance")
REPEAT_MODEL = "MiniCPM5-2B-Base"
RUNNING_EXAMPLE = ("bear_continent", "company_hq")

BANDS = ("engineered_town_latitude_band", "engineered_town_longitude_band",
         "engineered_town_elevation_band", "engineered_town_population_band")
ENGINEERED = ("sandbox_town_region", "engineered_town_country_language") + BANDS
if set(BANDS) != set(relation_sets.named("engineered_bands")) or set(ENGINEERED) != set(relation_sets.named("engineered")):
    raise SystemExit("claims_report.py: BANDS / ENGINEERED differ from the registry's engineered_bands / engineered")

def load_fjc():
    spec = importlib.util.spec_from_file_location("fjc", HERE / "fleet_joint_controls.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

fjc = load_fjc()

def sha256(path: Path) -> str:
    return fjc.sha256(path)

def pearson(a, b) -> float | None:
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else None

def spearman(a, b) -> float | None:
    return pearson(fjc.midranks(np.asarray(a, float)), fjc.midranks(np.asarray(b, float)))

def rowwise_pearson(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    d = np.sqrt((A * A).sum(axis=1) * (B * B).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d > 0, (A * B).sum(axis=1) / d, np.nan)

def rowwise_ranks(A: np.ndarray) -> np.ndarray:
    less = (A[:, :, None] > A[:, None, :]).sum(axis=2)
    equal = (A[:, :, None] == A[:, None, :]).sum(axis=2)
    return less + (equal + 1) / 2.0

def _betacf(a: float, b: float, x: float) -> float:
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            return h
    raise ArithmeticError("incomplete beta did not converge")

def betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb) * _betacf(b, a, 1.0 - x) / b

def t_two_sided_p(t: float, df: int) -> float:
    return betainc(df / 2.0, 0.5, df / (df + t * t))

def ci(values) -> list[float] | None:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return None
    return [float(np.quantile(v, 0.025, method="linear")), float(np.quantile(v, 0.975, method="linear"))]

def williams_t(r_xy: float, r_zy: float, r_xz: float, n: int) -> tuple[float, float]:
    det = 1 - r_xy ** 2 - r_zy ** 2 - r_xz ** 2 + 2 * r_xy * r_zy * r_xz
    rbar = (r_xy + r_zy) / 2
    t = (r_xy - r_zy) * math.sqrt((n - 1) * (1 + r_xz)
                                  / (2 * (n - 1) / (n - 3) * det + rbar ** 2 * (1 - r_xz) ** 3))
    return float(t), t_two_sided_p(t, n - 3)

def assoc(a, b, idx: np.ndarray) -> dict:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    A, B = a[idx], b[idx]
    return {"n": len(a), "r": pearson(a, b), "rho": spearman(a, b),
            "r_ci": ci(rowwise_pearson(A, B)),
            "rho_ci": ci(rowwise_pearson(rowwise_ranks(A), rowwise_ranks(B)))}

def chance_corrected(v: float, k: int) -> float:
    return (v - 1.0 / k) / (1.0 - 1.0 / k)

def load_rows(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_run = {}
    for r in rows:
        if r.get("final") is True and r.get("repeat") == 0:
            if r["run_id"] in by_run:
                raise SystemExit(f"two final repeat-0 rows for {r['run_id']}")
            by_run[r["run_id"]] = r
    return by_run, rows

def cohort_records(model_report: dict, by_run: dict) -> list[dict]:
    out = []
    for rel, v in model_report["per_relation"].items():
        row = by_run.get(v["run_id"])
        if row is None or row.get("relation") != rel:
            raise SystemExit(f"{v['run_id']}: no final repeat-0 row to join")
        if row["heldout_cand_acc"] != v["y"] or row["pre_rel_cand_acc"] != v["pre_rel_cand_acc"]:
            raise SystemExit(f"{v['run_id']}: the rows snapshot disagrees with the report")
        rec = {"relation": rel, "relation_set": v["relation_set"], "arm_set": v["arm_set"],
               "x": v["x"], "y": v["y"], "recall": v["pre_rel_cand_acc"],
               "n_classes": int(row["n_classes"]), "n_train": int(row["n_train"]),
               "subject_frequency": v["subject_frequency"]}
        for f in CARRIED:
            val = row.get(f)
            rec[f] = float(val) if fjc.finite(val) else None
        rec["arm_sha256"] = row.get("arm_sha256")
        out.append(rec)
    return out

def main_claim(model: str, mr: dict, recs: list[dict], draws: int) -> dict:
    groups = mr["predictor_ce"]["column_groups"]

    def copied(g):
        c = groups[g]
        b = c.get("bootstrap") or {}
        return {"n": c["n"], "r": c["partial_pearson_r"], "rho": c["rank_partial_spearman_rho"],
                "r_ci": b.get("pearson_r_percentile_95_ci"), "rho_ci": b.get("spearman_rho_percentile_95_ci"),
                "controls": c["controls"]}

    by_set = {}
    for name in SUBSETS:
        members = set(relation_sets.named(name))
        sub = [r for r in recs if r["relation"] in members]
        idx = fjc.bootstrap_indices(model + "\0" + name, len(sub), draws)
        by_set[name] = assoc([r["x"] for r in sub], [r["y"] for r in sub], idx)
    return {"n": mr["n"], "n_by_relation_set": mr["n_by_relation_set"],
            "excluded": mr["excluded"], "column_groups": {g: copied(g) for g in groups},
            "by_relation_set": by_set}

def competitor_claim(model: str, recs: list[dict], draws: int) -> dict:
    out = {}
    for key, field, label in COMPETITORS:
        sub = [r for r in recs if r[field] is not None]
        n = len(sub)
        x = np.array([r["x"] for r in sub])
        y = np.array([r["y"] for r in sub])
        z = np.array([r[field] for r in sub])

        idx = fjc.bootstrap_indices(model, n, draws)
        X, Y, Z = x[idx], y[idx], z[idx]
        rX, rY, rZ = rowwise_ranks(X), rowwise_ranks(Y), rowwise_ranks(Z)
        d_r = rowwise_pearson(X, Y) - rowwise_pearson(Z, Y)
        d_rho = rowwise_pearson(rX, rY) - rowwise_pearson(rZ, rY)
        r_xy, r_zy, r_xz = pearson(x, y), pearson(z, y), pearson(x, z)
        t, p = williams_t(r_xy, r_zy, r_xz, n)
        rho_xy, rho_zy, rho_xz = spearman(x, y), spearman(z, y), spearman(x, z)
        t_s, p_s = williams_t(rho_xy, rho_zy, rho_xz, n)
        out[key] = {
            "field": field, "label": label, "n": n,
            "n_exact_zero": int((z == 0).sum()),
            "z_y": {"r": r_zy, "rho": rho_zy, "r_ci": ci(rowwise_pearson(Z, Y)),
                    "rho_ci": ci(rowwise_pearson(rZ, rY))},
            "x_y": {"r": r_xy, "rho": rho_xy},
            "x_z": {"r": r_xz, "rho": rho_xz},
            "diff": {"r": r_xy - r_zy, "r_ci": ci(d_r), "rho": rho_xy - rho_zy, "rho_ci": ci(d_rho),
                     "share_draws_x_not_ahead_r": float(np.mean(d_r[np.isfinite(d_r)] <= 0)),
                     "share_draws_x_not_ahead_rho": float(np.mean(d_rho[np.isfinite(d_rho)] <= 0))},
            "williams": {"t": t, "df": n - 3, "p_two_sided": p,
                         "on_ranks_t": t_s, "on_ranks_p_two_sided": p_s,
                         "note": "on_ranks applies the Pearson formula to midranks: an approximation"},
        }
    return out

def recall_claim(model: str, mr: dict, recs: list[dict], draws: int) -> dict:
    idx = fjc.bootstrap_indices(model, len(recs), draws)
    rec_n = np.array([chance_corrected(r["recall"], r["n_classes"]) for r in recs])
    y_n = np.array([chance_corrected(r["y"], r["n_classes"]) for r in recs])
    med, t1 = float(np.median(rec_n)), float(np.quantile(y_n, 1 / 3, method="linear"))
    inert = [r["relation"] for r, a, b in zip(recs, rec_n, y_n) if a >= med and b <= t1]
    g = mr["predictor_ce"]["column_groups"]["recall"]
    b = g.get("bootstrap") or {}
    return {
        "recall_y": assoc([r["recall"] for r in recs], [r["y"] for r in recs], idx),
        "x_recall": {"r": pearson([r["x"] for r in recs], [r["recall"] for r in recs]),
                     "rho": spearman([r["x"] for r in recs], [r["recall"] for r in recs])},
        "x_y_partial_recall": {"r": g["partial_pearson_r"], "rho": g["rank_partial_spearman_rho"],
                               "r_ci": b.get("pearson_r_percentile_95_ci"),
                               "rho_ci": b.get("spearman_rho_percentile_95_ci")},
        "inert_rule": {"recall_cc_median": med, "y_cc_lower_tercile": t1},
        "inert": inert,

        "inert_x_percentile": {r["relation"]: float((fjc.midranks(np.array([q["x"] for q in recs]))[i] - 1)
                                                    / (len(recs) - 1))
                               for i, r in enumerate(recs) if r["relation"] in inert},
        "per_relation_cc": {r["relation"]: {"recall_cc": float(a), "y_cc": float(bb)}
                            for r, a, bb in zip(recs, rec_n, y_n)},
    }

def oneway(Y: np.ndarray) -> dict:
    n, k = Y.shape
    ybar = Y.mean(axis=1)
    msb = k * float(((ybar - Y.mean()) ** 2).sum()) / (n - 1)
    msw = float(((Y - ybar[:, None]) ** 2).sum()) / (n * (k - 1))
    icc1 = (msb - msw) / (msb + (k - 1) * msw)
    return {"n_cells": n, "msb": msb, "msw": msw, "icc1": icc1, "icck": (msb - msw) / msb, "within_share": 1 - icc1,
            "between_sd": float(ybar.std(ddof=1)), "within_sd_pooled": math.sqrt(msw)}

def repeat_claim(rows: list[dict], report: dict, draws: int, relations: Iterable[str] | None = None,
                 model: str = REPEAT_MODEL, set_name: str = "bear") -> dict:
    mr = report["models"][model]
    rels = sorted(set(mr["per_relation"]) & set(relation_sets.named(set_name) if relations is None else relations))
    bear = rels
    cells = collections.defaultdict(dict)
    notes = []
    for r in rows:
        if r.get("model") != model or r.get("relation") not in bear or r.get("final") is not True:
            continue
        why = fjc.ineligibility(r)
        if why:
            notes.append(f"{r['run_id']}: ineligible ({', '.join(why)})")
            continue
        rep = r["repeat"]
        if rep in cells[r["relation"]]:
            raise SystemExit(f"{r['run_id']}: two final rows")
        cells[r["relation"]][rep] = r
    complete = [rel for rel in bear if set(cells[rel]) == {0, 1, 2}]
    incomplete = [rel for rel in bear if rel not in complete]
    Y = np.array([[cells[rel][k]["heldout_cand_acc"] for k in (0, 1, 2)] for rel in complete])
    nH = np.array([cells[rel][0]["n_heldout"] for rel in complete], float)
    n, k = Y.shape
    pairs = [(0, 1), (0, 2), (1, 2)]
    pair_r = {f"r{a}-r{b}": pearson(Y[:, a], Y[:, b]) for a, b in pairs}
    pooled_a = np.concatenate([Y[:, a] for a, b in pairs] + [Y[:, b] for a, b in pairs])
    pooled_b = np.concatenate([Y[:, b] for a, b in pairs] + [Y[:, a] for a, b in pairs])
    ybar = Y.mean(axis=1)
    av = oneway(Y)
    sd = Y.std(axis=1, ddof=1)
    se = np.sqrt(ybar * (1 - ybar) / nH)
    ok = se > 0
    ratio = sd[ok] / se[ok]
    x = np.array([mr["per_relation"][rel]["x"] for rel in complete])
    seed = model + "\0repeat" + ("" if set_name == "bear" else "\0" + set_name)
    idx = fjc.bootstrap_indices(seed, n, draws)
    X, Y0, Ym = x[idx], Y[:, 0][idx], ybar[idx]
    d_r = rowwise_pearson(X, Ym) - rowwise_pearson(X, Y0)
    d_rho = (rowwise_pearson(rowwise_ranks(X), rowwise_ranks(Ym))
             - rowwise_pearson(rowwise_ranks(X), rowwise_ranks(Y0)))
    return {
        "model": model, "relation_set": set_name, "k": k, "n_cells_complete": n,
        "incomplete": incomplete, "notes": notes,
        "repeat_agreement": {"pooled_pairs_r": pearson(pooled_a, pooled_b), "pairs": pair_r},
        "anova": {k_: v for k_, v in av.items() if k_ != "n_cells"},
        "spread_vs_binomial": {"median_ratio": float(np.median(ratio)), "mean_ratio": float(np.mean(ratio)), "share_above_1": float(np.mean(ratio > 1)),
                               "n_zero_floor": int((~ok).sum()), "max_ratio": float(ratio.max())},
        "association": {"repeat0": {"r": pearson(x, Y[:, 0]), "rho": spearman(x, Y[:, 0])},
                        "mean3": {"r": pearson(x, ybar), "rho": spearman(x, ybar)},
                        "diff_mean3_minus_repeat0": {"r": pearson(x, ybar) - pearson(x, Y[:, 0]), "r_ci": ci(d_r),
                                                     "rho": spearman(x, ybar) - spearman(x, Y[:, 0]),
                                                     "rho_ci": ci(d_rho)}},
        "per_relation": {rel: {"y": [float(v) for v in Y[i]], "n_heldout": int(nH[i]), "x": float(x[i]),
                               "sd": float(sd[i]), "binomial_se": float(se[i])}
                         for i, rel in enumerate(complete)},
    }

def study_repeat_claim(rows: list[dict], report: dict, draws: int) -> dict:
    sel = "hernandez_engineered"
    y0 = {(r["model"], r["relation"]): r["heldout_cand_acc"] for r in rows if r.get("repeat") == 0}
    for m, mr in report["models"].items():
        for rel, v in mr["per_relation"].items():
            if (m, rel) in y0 and abs(y0[m, rel] - v["y"]) > 1e-12:
                raise SystemExit(f"{m}/{rel}: repeat-0 y {y0[m, rel]} differs from Table 1's {v['y']}")
    models, stacks = {}, []

    in_fleet = {r["model"] for r in rows}
    for m in report["models"]:
        if m not in in_fleet:
            continue
        full = repeat_claim(rows, report, draws, relation_sets.named(sel), model=m, set_name=sel)
        full["by_set"] = {}
        for s in ("hernandez", "engineered"):
            sub = repeat_claim(rows, report, draws, relation_sets.named(s), model=m, set_name=s)
            full["by_set"][s] = {"n_cells_complete": sub["n_cells_complete"], "anova": sub["anova"],
                                 "pooled_pairs_r": sub["repeat_agreement"]["pooled_pairs_r"]}
        models[m] = full
        stacks.append(np.array([v["y"] for v in full["per_relation"].values()]))
    per = list(models.values())
    col = lambda f: [f(v) for v in per]
    stat = lambda xs: {"median": float(np.median(xs)), "mean": float(np.mean(xs)), "min": float(min(xs)), "max": float(max(xs))}
    return {
        "relation_set": sel, "models": models,
        "models_without_repeats": [m for m in report["models"] if m not in in_fleet],
        "pooled_cells": oneway(np.vstack(stacks)),
        "across_models": {
            "icc1": stat(col(lambda v: v["anova"]["icc1"])),
            "within_share": stat(col(lambda v: v["anova"]["within_share"])),
            "within_share_hernandez": stat(col(lambda v: v["by_set"]["hernandez"]["anova"]["within_share"])),
            "within_share_engineered": stat(col(lambda v: v["by_set"]["engineered"]["anova"]["within_share"])),
            "pooled_pairs_r": stat(col(lambda v: v["repeat_agreement"]["pooled_pairs_r"])),
            "r_repeat0": stat(col(lambda v: v["association"]["repeat0"]["r"])),
            "r_mean3": stat(col(lambda v: v["association"]["mean3"]["r"])),
            "r_diff": stat(col(lambda v: v["association"]["diff_mean3_minus_repeat0"]["r"])),
            "rho_diff": stat(col(lambda v: v["association"]["diff_mean3_minus_repeat0"]["rho"])),
            "n_models_r_diff_ci_excludes_0": sum(1 for v in per
                                                 if not (v["association"]["diff_mean3_minus_repeat0"]["r_ci"][0] <= 0
                                                         <= v["association"]["diff_mean3_minus_repeat0"]["r_ci"][1])),
        },
    }

def running_example(report: dict, recs_by_model: dict) -> dict:
    out = {}
    for model, recs in recs_by_model.items():
        by = {r["relation"]: r for r in recs}
        out[model] = {}
        for rel in RUNNING_EXAMPLE:
            r = by.get(rel)
            if r is None:
                out[model][rel] = None
                continue
            k = r["n_classes"]
            out[model][rel] = {"x": r["x"], "y": r["y"], "recall": r["recall"], "n_classes": k,
                               "n_train": r["n_train"], "recall_cc": chance_corrected(r["recall"], k),
                               "y_cc": chance_corrected(r["y"], k)}
    return out

def sum_d2(x, y) -> int:
    if len(set(x)) != len(x) or len(set(y)) != len(y):
        raise SystemExit("exact rank test needs untied values")
    rx, ry = fjc.midranks(np.asarray(x, float)), fjc.midranks(np.asarray(y, float))
    return int(round(float(((rx - ry) ** 2).sum())))

def null_d2_counts(n: int) -> dict:
    import itertools
    out = collections.Counter()
    for perm in itertools.permutations(range(n)):
        out[sum((i - p) ** 2 for i, p in enumerate(perm))] += 1
    return out

def stratified_rank_test(groups) -> dict:
    dist = collections.Counter({0: 1})
    total = 1
    obs = 0
    for x, y in groups:
        n = len(x)
        obs += sum_d2(x, y)
        one = null_d2_counts(n)
        new = collections.Counter()
        for a, ca in dist.items():
            for b, cb in one.items():
                new[a + b] += ca * cb
        dist = new
        total *= math.factorial(n)
    tail = sum(c for v, c in dist.items() if v <= obs)
    n = len(groups[0][0])
    return {"statistic": "sum over models of Spearman rho",
            "value": float(sum(1 - 6 * sum_d2(x, y) / (len(x) * (len(x) ** 2 - 1)) for x, y in groups)),
            "p_one_sided_exact": tail / total, "n_strata": len(groups), "n_per_stratum": n}

def relation_level_test(x, y) -> dict:
    n = len(x)
    one = null_d2_counts(n)
    obs = sum_d2(x, y)
    return {"n": n, "rho": spearman(x, y), "r": pearson(x, y),
            "p_one_sided_exact": sum(c for v, c in one.items() if v <= obs) / math.factorial(n)}

def arm_subjects(arm_dir: Path) -> dict:
    out = {}
    for split, name in (("train", "train.jsonl"), ("heldout", "heldout.jsonl"), ("gate", "gate.jsonl")):
        f = arm_dir / name
        subs = set()
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    subs.add(rec.get("town", rec.get("subject")))
        subs.discard(None)
        out[split] = subs
    return out

def find_arm(roots, slug: str, rel: str, sha: str) -> Path:
    tried = []
    for root in roots:
        d = root / slug / rel / "r0"
        if (d / "arm.json").exists():
            tried.append(str(d))
            if sha256(d / "arm.json") == sha:
                return d
    raise SystemExit(f"{slug}/{rel}: no arm.json with the row's arm_sha256 {sha[:12]} among {tried}")

def engineered_claim(models, recs_by_model, slug_of, arm_roots) -> dict:
    overlap, per_model = {}, {}
    band_groups, eng_groups = [], []
    within_b, within_e = ([], []), ([], [])
    for m in models:
        by = {r["relation"]: r for r in recs_by_model[m]}
        eng = [r for r in ENGINEERED if r in by]
        bands = [r for r in BANDS if r in by]
        subs = {r: arm_subjects(find_arm(arm_roots, slug_of[m], r, by[r]["arm_sha256"])) for r in eng}
        allsub = {r: set().union(*subs[r].values()) for r in eng}
        common = set.intersection(*allsub.values()) if allsub else set()
        pair = {}
        for i, a in enumerate(eng):
            for b in eng[i + 1:]:
                u = allsub[a] | allsub[b]
                pair[f"{a}|{b}"] = {"jaccard": len(allsub[a] & allsub[b]) / len(u) if u else None,
                                    "shared": len(allsub[a] & allsub[b]),
                                    "heldout_a_in_train_b": len(subs[a]["heldout"] & subs[b]["train"]),
                                    "heldout_same": len(subs[a]["heldout"] & subs[b]["heldout"])}
        overlap[m] = {"n_subjects": {r: {k: len(v) for k, v in subs[r].items()} for r in eng},
                      "common_to_all": len(common),
                      "common_to_bands": len(set.intersection(*(allsub[r] for r in bands))) if bands else 0,
                      "pairs": pair}

        xb, yb = [by[r]["x"] for r in bands], [by[r]["y"] for r in bands]
        xe, ye = [by[r]["x"] for r in eng], [by[r]["y"] for r in eng]
        band_groups.append((xb, yb))
        eng_groups.append((xe, ye))
        for tgt, xs, ys in ((within_b, xb, yb), (within_e, xe, ye)):
            tgt[0].extend(np.asarray(xs) - np.mean(xs))
            tgt[1].extend(np.asarray(ys) - np.mean(ys))

        rest = [r for r in recs_by_model[m] if r["relation"] not in ENGINEERED]
        xr, yr = np.array([r["x"] for r in rest]), np.array([r["y"] for r in rest])
        b1, b0 = np.polyfit(xr, yr, 1)
        resid = {r: by[r]["y"] - (b0 + b1 * by[r]["x"]) for r in eng}
        per_model[m] = {
            "bands": {"n": len(bands), "r": pearson(xb, yb), "rho": spearman(xb, yb)},
            "engineered": {"n": len(eng), "r": pearson(xe, ye), "rho": spearman(xe, ye)},
            "fit_other": {"n": len(rest), "slope": float(b1), "intercept": float(b0)},
            "residual": resid,
            "cells": {r: {"x": by[r]["x"], "y": by[r]["y"], "n_classes": by[r]["n_classes"],
                          "n_train": by[r]["n_train"], "recall": by[r]["recall"]} for r in eng},
        }
    res_all = [v for pm in per_model.values() for v in pm["residual"].values()]
    means = {r: {"x": float(np.mean([per_model[m]["cells"][r]["x"] for m in models])),
                 "y": float(np.mean([per_model[m]["cells"][r]["y"] for m in models]))}
             for r in ENGINEERED if all(r in per_model[m]["cells"] for m in models)}
    return {
        "relations": list(ENGINEERED), "bands": list(BANDS),
        "subject_overlap": overlap,
        "per_model": per_model,
        "pooled": {
            "bands_within_model_r": pearson(*within_b), "bands_within_model_n": len(within_b[0]),
            "engineered_within_model_r": pearson(*within_e), "engineered_within_model_n": len(within_e[0]),
            "bands_stratified": stratified_rank_test(band_groups),
            "engineered_stratified": stratified_rank_test(eng_groups),
            "note": "the within-model pooled r centres x and y per model (model fixed effects); the "
                    "stratified test permutes y within each model. Both treat the models as replicates: "
                    "the relations are the same 4 or 6 in every model, so neither is a test over "
                    "independent relations; relation_level is.",
        },
        "relation_means": means,
        "relation_level": {"bands": relation_level_test([means[r]["x"] for r in BANDS], [means[r]["y"] for r in BANDS]),
                           "engineered": relation_level_test([means[r]["x"] for r in ENGINEERED],
                                                             [means[r]["y"] for r in ENGINEERED])},
        "residual_other_fit": {"mean": float(np.mean(res_all)), "sd": float(np.std(res_all, ddof=1)),
                               "n": len(res_all),
                               "per_relation_mean": {r: float(np.mean([per_model[m]["residual"][r] for m in models]))
                                                     for r in ENGINEERED}},
    }

def build(args) -> dict:
    d = args.controls_dir
    study_rows = None
    if args.study_repeat_rows is not None:
        if not args.study_repeat_sha256 or sha256(args.study_repeat_rows) != args.study_repeat_sha256:
            raise SystemExit(f"{args.study_repeat_rows}: sha256 is not {args.study_repeat_sha256}")
        study_rows = [json.loads(l) for l in args.study_repeat_rows.read_text(encoding="utf-8").splitlines() if l.strip()]
    reports = {v: json.loads((d / f"fleet-joint-controls.{v}.json").read_text(encoding="utf-8")) for v in VARIANTS}
    for v, sel in VARIANTS.items():
        want = set(relation_sets.named(sel))
        for m, mr in reports[v]["models"].items():
            panel = set(mr["relations"]) | {e["relation"] for e in mr["excluded"]}
            if panel != want:
                raise SystemExit(f"{v}/{m}: the report's panel differs from selection {sel}: {sorted(panel ^ want)}")
            for rel, pr in mr["per_relation"].items():
                if pr["relation_set"] != relation_sets.source(rel):
                    raise SystemExit(f"{v}/{m}/{rel}: relation_set {pr['relation_set']} but the registry says "
                                     f"{relation_sets.source(rel)}")
    rows_path = d / "rows.jsonl"
    by_run, rows = load_rows(rows_path)
    models = list(reports["headline"]["models"])
    variants = {}
    for v, rep in reports.items():
        per_model = {}
        for m in models:
            mr = rep["models"][m]
            if not mr.get("computable"):
                raise SystemExit(f"{v}/{m}: the controls report marks the model not computable")
            recs = cohort_records(mr, by_run)
            per_model[m] = {"relations": recs,
                            "main": main_claim(m, mr, recs, args.bootstrap_draws),
                            "competitors": competitor_claim(m, recs, args.bootstrap_draws),
                            "recall": recall_claim(m, mr, recs, args.bootstrap_draws)}
        inert_count = collections.Counter(r for pm in per_model.values() for r in pm["recall"]["inert"])
        variants[v] = {"collection": rep["collection"], "n_per_model": rep["n_per_model"],
                       "equal_n": rep["equal_n"], "models": per_model,
                       "inert_counts": dict(inert_count.most_common())}
    head = {m: variants["headline"]["models"][m]["relations"] for m in models}
    roster = json.loads((d / "roster.json").read_text(encoding="utf-8"))
    slug_of = {m["display"]: m["slug"] for m in roster["models"]}
    return {
        "schema": SCHEMA,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "collection": reports["headline"]["collection"],
        "models": models,
        "bootstrap": {"draws": args.bootstrap_draws,
                      "indices": "fleet_joint_controls.bootstrap_indices(model, n, draws) for whole-cohort "
                                 "statistics (the Table 1 resamples); model + '\\0' + subset tag for subsets",
                      "interval": "percentile 2.5/97.5, linear"},
        "variants": variants,
        "repeat_variance": repeat_claim(rows, reports["headline"], args.bootstrap_draws),
        **({"study_repeat_variance": study_repeat_claim(study_rows, reports["headline"], args.bootstrap_draws)}
           if study_rows is not None else {}),
        "running_example": {"relations": list(RUNNING_EXAMPLE), "per_model": running_example(reports, head)},
        "engineered": engineered_claim(models, head, slug_of, args.arms_root),
        "sources": {**{f"controls_{v}": {"file": str(d / f"fleet-joint-controls.{v}.json"),
                                         "sha256": sha256(d / f"fleet-joint-controls.{v}.json")} for v in VARIANTS},
                    "rows": {"file": str(rows_path), "sha256": sha256(rows_path)},
                    "controls_script_sha256": reports["headline"]["sources"]["script"]["sha256"],
                    "arms_roots": [str(p) for p in args.arms_root],
                    **({"study_repeat_rows": {"file": str(args.study_repeat_rows), "sha256": args.study_repeat_sha256}}
                       if study_rows is not None else {}),
                    "relation_sets": {"registry_sha256": sha256(relation_sets.DEFAULT_PATH),
                                      "selections": {n: {"n": len(relation_sets.named(n)),
                                                         "digest": relation_sets.digest(relation_sets.named(n))}
                                                     for n in sorted(set(VARIANTS.values()) | set(SUBSETS)
                                                                     | {"engineered", "engineered_bands", "bear"})}},
                    "script": {"file": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())}},
    }

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controls-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--arms-root", type=Path, action="append", required=True,
                    help="a stage arms directory (<slug>/<relation>/r0/arm.json); repeatable, "
                         "searched in order for the arm whose sha256 the final row records")
    ap.add_argument("--bootstrap-draws", type=int, default=fjc.BOOTSTRAP_DRAWS)
    ap.add_argument("--study-repeat-rows", type=Path,
                    help="rows of the current-protocol repeat fleet (Hernandez + engineered x models x repeats 0-2)")
    ap.add_argument("--study-repeat-sha256", help="required with --study-repeat-rows; checked before use")
    args = ap.parse_args(argv)
    out = build(args)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(args.output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
