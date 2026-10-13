#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from linroute import relation_sets

SCHEMA = "linroute/fleet-joint-controls/1"
REPEAT_AGGREGATION = "repeat 0 only"
COLUMN_GROUPS = (
    ("raw", ()),
    ("recall", ("pre_rel_cand_acc",)),
    ("answer_count", ("log_n_classes",)),
    ("subject_count", ("log_n_train",)),
    ("subject_frequency", ("subject_frequency",)),
    ("all_jointly", ("pre_rel_cand_acc", "log_n_train", "log_n_classes",
                     "subject_frequency")),
)

RELATION_SETS = ("hernandez", "engineered", "bear")

VARIANTS = {
    "headline": "headline",
}
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260921
FREQUENCY_INDEX = "v4_dolma-v1_7_llama"
IDENTITY_TOLERANCE = 1e-12
FREQUENCY_TOLERANCE = 1e-12
EPOCHS = 20
ELIGIBILITY = {
    "status": "complete",
    "production_eligible": True,
    "production_horizon": EPOCHS,
    "planned_epochs": EPOCHS,
    "epochs_done": EPOCHS,
    "final_epoch": EPOCHS,
    "limit_train": 0,
    "holdout_rule": "mushroom-v2",
    "protocol": "heldout-mushroom-v2",
}

BEAR_FREQUENCY_ALIASES = {
    "bear_IUCN_conservation_status": "bear_iucn_conservation_status",
    "bear_original_language_of_film_or_TV_show": "bear_original_language_of_film_or_tv_show",
}

class ReportError(RuntimeError):
    pass

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

def mean(values):
    return sum(values) / len(values)

def load_panel(spec_path: Path, roster_path: Path):
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    panel = relation_sets.named("headline")
    relation_set = {r: relation_sets.source(r) for r in panel}
    for name in RELATION_SETS:
        want = sorted(r for r in panel if relation_set[r] == name)
        if sorted(spec["relations"].get(name, [])) != want:
            raise ReportError(f"spec relations.{name} differs from the registry's headline {name} relations")
    roster_rels = set(roster["relations"]) | set(roster.get("study_relations") or [])
    if roster_rels != set(panel):
        raise ReportError("roster and registry headline relations differ: "
                          f"{sorted(roster_rels ^ set(panel))}")
    slug_of = {m["display"]: m["slug"] for m in roster["models"]}
    models = [m["id"] for m in spec["models"]]
    missing = [m for m in models if m not in slug_of]
    if missing:
        raise ReportError(f"spec models not in the roster: {missing}")
    return models, relation_set, {m: slug_of[m] for m in models}

def ineligibility(row) -> list[str]:
    reasons = [f"{k}={row.get(k)!r}" for k, v in ELIGIBILITY.items() if row.get(k) != v]
    records, exposed = row.get("n_train_records"), row.get("n_exposed_per_epoch")
    if not (finite(records) and records > 0 and finite(exposed) and exposed == records):
        reasons.append("exposure_not_equal_to_train_records")
    for field in ("heldout_cand_acc", "pre_rel_cand_acc", "n_train", "n_classes"):
        if not finite(row.get(field)):
            reasons.append(f"{field}_not_finite")
    for field in ("n_train", "n_classes"):
        if finite(row.get(field)) and row[field] <= 0:
            reasons.append(f"{field}_not_positive")
    return reasons

def load_outcomes(rows_path: Path, models, relation_set):
    set_aside = collections.Counter()
    out = {}
    ineligible = {}
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        model, rel = row.get("model"), row.get("relation")
        if model not in models or rel not in relation_set:
            set_aside[f"outside the panel: {model}"] += 1
            continue
        if row.get("repeat") != 0:
            set_aside[f"repeat {row.get('repeat')} (repeat-variance only)"] += 1
            continue
        if row.get("final") is not True:
            set_aside[f"arm_set {row.get('arm_set')} not final (not the P(4) view)"] += 1
            continue
        if row.get("arm_set") not in ("p4", "shared"):
            raise ReportError(f"{model}/{rel}: final row with arm_set {row.get('arm_set')!r}")
        key = (model, rel)
        if key in out or key in ineligible:
            raise ReportError(f"{model}/{rel}: two final repeat-0 rows")
        reasons = ineligibility(row)
        if reasons:
            ineligible[key] = reasons
        else:
            out[key] = row
    return out, ineligible, dict(set_aside)

def load_gate_table(path: Path, slug_of, relation_set):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    model_of = {slug: model for model, slug in slug_of.items()}
    grouped = collections.defaultdict(list)
    for row in rows:
        cell = row["cell"]
        hits = [s for s in model_of if cell.startswith(s + "__")]
        if len(hits) != 1:
            continue
        rel = cell[len(hits[0]) + 2:]
        if rel not in relation_set:
            raise ReportError(f"gate table cell {cell}: relation {rel!r} not in the panel")
        grouped[(model_of[hits[0]], rel)].append(row)
    out = {}
    for key, group in grouped.items():
        passed = [r for r in group if r["status_new"] == "passed"]
        entry = {"artifacts": [{"path": r["path"], "sha256": r["sha256"],
                                "status_new": r["status_new"]} for r in group]}
        if not passed:
            entry.update(x=None, status="failed",
                         reason="G_new failed: " + "; ".join(sorted({r["reasons"] for r in group})))
        else:
            gains = {float(r["ce_gain"]) for r in passed}
            if len(gains) != 1:
                raise ReportError(f"{key}: passed CE artifacts disagree on the gain {sorted(gains)}")
            entry.update(x=gains.pop(), status="passed", reason=None)
        out[key] = entry
    return out

def _counts(paths, legacy_hash=None):
    counts, conflicts, invalid = {}, [], 0
    for path in paths:
        legacy = legacy_hash is not None and sha256(path) == legacy_hash
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("kind") != "subj":
                continue
            if not rec.get("ok") or rec.get("count") is None:
                invalid += 1
                continue
            if not (rec.get("index") is None and legacy) and rec.get("index") != FREQUENCY_INDEX:
                raise ReportError(f"{path}:{n}: count index {rec.get('index')!r}")
            value = (rec["count"], rec.get("approx"))
            q = rec["query"]
            if q in counts and counts[q] != value:
                conflicts.append(q)
            counts[q] = value
    if conflicts:
        raise ReportError(f"conflicting subject counts for {len(conflicts)} queries, e.g. {conflicts[:3]}")
    return counts, invalid

def _mean_log(payload_rows, counts, relation):
    subjects = [s for s, _ in payload_rows]
    absent = sorted({s for s in subjects if s not in counts})
    if absent:
        raise ReportError(f"{relation}: {len(absent)} subjects without a count, e.g. {absent[:3]}")
    if any(counts[s][1] is not False for s in subjects):
        raise ReportError(f"{relation}: a SUBJECT count is not explicitly exact")
    return mean([math.log10(1 + counts[s][0]) for s in subjects]), len(subjects)

def load_frequency(args, relation_set):
    out = {}

    prov = json.loads(args.study_payload_provenance.read_text(encoding="utf-8"))
    if prov.get("index") != FREQUENCY_INDEX or prov.get("query_type") != "count":
        raise ReportError("study payload provenance: wrong index or query type")
    if prov.get("payload_expanded_sha256") != sha256(args.study_payload):
        raise ReportError("study payload hash does not match its provenance")
    payload = json.loads(args.study_payload.read_text(encoding="utf-8"))
    counts, _ = _counts(args.study_counts, prov.get("original", {}).get("counts_sha256_at_build"))
    for rel, name in relation_set.items():
        if name == "bear":
            continue
        if rel not in payload:
            raise ReportError(f"study payload lacks {rel}")
        value, n = _mean_log(payload[rel], counts, rel)
        out[rel] = {"value": value, "n_rows": n, "source": "study payload"}

    relstats = json.loads(args.bear_relstats.read_text(encoding="utf-8"))
    if relstats.get("index") != FREQUENCY_INDEX:
        raise ReportError(f"BEAR relstats index {relstats.get('index')!r}")
    bpayload = json.loads(args.bear_payload.read_text(encoding="utf-8"))
    bcounts, _ = _counts([args.bear_counts])
    for key, stats in relstats["freq"].items():
        rel = BEAR_FREQUENCY_ALIASES.get(key, key)
        if relation_set.get(rel) != "bear":
            raise ReportError(f"BEAR relstats relation {key!r} is not a BEAR panel relation")
        if key not in bpayload:
            raise ReportError(f"BEAR payload lacks {key}")
        value, n = _mean_log(bpayload[key], bcounts, key)
        if abs(value - stats["subj_mean_log"]) > FREQUENCY_TOLERANCE:
            raise ReportError(f"{key}: recomputed {value!r} != relstats {stats['subj_mean_log']!r}")
        if n != stats["n_subj_counts"]:
            raise ReportError(f"{key}: {n} payload rows, relstats counted {stats['n_subj_counts']}")
        out[rel] = {"value": value, "n_rows": n, "cap250": stats["subj_mean_log_cap250"],
                    "source": "BEAR payload" + (f" (as {key})" if key != rel else "")}
    lacking = sorted(r for r in relation_set if r not in out)
    if lacking:
        raise ReportError(f"no SUBJECT frequency for {lacking}")
    return out

def midranks(values: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    start = np.cumsum(counts) - counts
    return (start + (counts + 1) / 2.0)[inverse.reshape(-1)]

def residual_partial(x, y, controls):
    n = len(x)
    design = np.column_stack([np.ones(n)] + list(controls))
    if int(np.linalg.matrix_rank(design)) != design.shape[1]:
        return None
    rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    nx, ny = float(np.linalg.norm(rx)), float(np.linalg.norm(ry))
    tol = np.finfo(float).eps * max(design.shape) * max(
        float(np.linalg.norm(x)), float(np.linalg.norm(y)), 1.0)
    if nx <= tol or ny <= tol:
        return None
    return max(-1.0, min(1.0, float(np.dot(rx, ry) / (nx * ny))))

def inverse_matrix_partial(x, y, controls):
    m = np.corrcoef(np.column_stack([x, y] + list(controls)), rowvar=False)
    if not controls:
        return float(m[0, 1])
    p = np.linalg.inv(m)
    return float(-p[0, 1] / math.sqrt(p[0, 0] * p[1, 1]))

def bootstrap_seed(model: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{BOOTSTRAP_SEED}\0{model}".encode()).digest()[:8], "big")

def bootstrap_indices(model: str, n: int, draws: int) -> np.ndarray:
    return np.random.default_rng(bootstrap_seed(model)).integers(0, n, size=(draws, n))

def residualize(design, columns):
    q, r = np.linalg.qr(design)
    d = np.abs(np.diag(r))
    if float(d.min()) <= np.finfo(float).eps * max(design.shape) * max(float(np.abs(r).max()), 1.0):
        return None
    return columns - q @ (q.T @ columns)

def pair_corr(res, i, j, scale):
    a, b = res[:, i], res[:, j]
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= scale or nb <= scale:
        return None
    return max(-1.0, min(1.0, float(np.dot(a, b) / (na * nb))))

def analyse(variables, indices):
    names = list(variables)
    raw = np.column_stack([variables[k] for k in names])
    ranked = np.column_stack([midranks(raw[:, i]) for i in range(raw.shape[1])])
    pos = {k: i for i, k in enumerate(names)}
    n = raw.shape[0]
    groups, worst = {}, 0.0
    for group, controls in COLUMN_GROUPS:
        k = len(controls)
        cr = [raw[:, pos[c]] for c in controls]
        ck = [ranked[:, pos[c]] for c in controls]
        r = residual_partial(raw[:, pos["x"]], raw[:, pos["y"]], cr)
        rho = residual_partial(ranked[:, pos["x"]], ranked[:, pos["y"]], ck)
        cell = {"controls": list(controls), "k_controls": k, "n": n, "residual_df": n - k - 2,
                "partial_pearson_r": r, "rank_partial_spearman_rho": rho,
                "computable": r is not None and rho is not None and n - k - 2 >= 2}
        if cell["computable"]:
            checks = {}
            for name, value, cols, ctrl in (("pearson", r, raw, cr), ("spearman", rho, ranked, ck)):
                try:
                    checks[name + "_vs_matrix_inverse"] = abs(value - inverse_matrix_partial(
                        cols[:, pos["x"]], cols[:, pos["y"]], ctrl))
                except np.linalg.LinAlgError:
                    if abs(abs(value) - 1.0) > 1e-9:
                        raise ReportError(f"{group} {name}: singular correlation matrix but |coefficient| = {abs(value)!r}")
                    checks[name + "_vs_matrix_inverse"] = None
                    cell.setdefault("notes", []).append(
                        f"{name}: residuals exactly collinear (coefficient {value:+.0f}); inverse-matrix check undefined")
            worst = max([worst] + [v for v in checks.values() if v is not None])
            cell["identity_checks"] = checks
        groups[group] = cell
    boot = {g: {"r": [], "rho": [], "singular": 0, "degenerate": 0} for g, _ in COLUMN_GROUPS}
    ones = np.ones((n, 1))
    for draw in indices:
        braw = raw[draw]
        brank = np.column_stack([midranks(braw[:, i]) for i in range(braw.shape[1])])
        for space, values, key in (("raw", braw, "r"), ("rank", brank, "rho")):
            scale = np.finfo(float).eps * n * max(float(np.abs(values).max()), 1.0)
            for group, controls in COLUMN_GROUPS:
                cols = [pos[c] for c in controls]
                design = np.hstack([ones, values[:, cols]]) if cols else ones
                res = residualize(design, values)
                if res is None:
                    boot[group]["singular"] += 1
                    continue
                v = pair_corr(res, pos["x"], pos["y"], scale)
                if v is None:
                    boot[group]["degenerate"] += 1
                    continue
                boot[group][key].append(v)
    for group, _ in COLUMN_GROUPS:
        b, cell = boot[group], groups[group]
        if not cell["computable"] or not b["r"] or not b["rho"]:
            continue
        q = lambda v: [float(np.quantile(v, 0.025, method="linear")),
                       float(np.quantile(v, 0.975, method="linear"))]
        cell["bootstrap"] = {
            "draws_requested": len(indices), "n_defined_pearson": len(b["r"]),
            "n_defined_spearman": len(b["rho"]), "n_singular_design": b["singular"],
            "n_degenerate_residual": b["degenerate"],
            "pearson_r_percentile_95_ci": q(b["r"]), "spearman_rho_percentile_95_ci": q(b["rho"]),

            "undefined_share_pearson": 1.0 - len(b["r"]) / len(indices),
            "undefined_share_spearman": 1.0 - len(b["rho"]) / len(indices),
        }
    return groups, worst

def design_diagnostics(variables):
    names = [c for c in COLUMN_GROUPS[-1][1]]
    ctrl = np.column_stack([variables[k] for k in names])
    constant = [k for i, k in enumerate(names) if float(np.ptp(ctrl[:, i])) == 0.0]
    if constant:
        return {"constant_controls": constant}
    z = (ctrl - ctrl.mean(axis=0)) / ctrl.std(axis=0)
    corr = np.corrcoef(ctrl, rowvar=False)
    pairs = [(names[i], names[j], float(corr[i, j]))
             for i in range(len(names)) for j in range(i + 1, len(names))]
    top = max(pairs, key=lambda p: abs(p[2]))
    return {"condition_number_standardized_design": float(np.linalg.cond(np.column_stack([np.ones(len(z)), z]))),
            "max_abs_control_pair_correlation": {"first": top[0], "second": top[1], "correlation": top[2]}}

def build(args):
    models, relation_set, slug_of = load_panel(args.spec, args.roster)
    sets = parse_sets(args.sets)
    selection = VARIANTS[args.variant]
    panel = sorted(set(relation_sets.named(selection)) & set(relation_sets.select(sources=list(sets))))
    outcomes, ineligible, set_aside = load_outcomes(args.rows, models, relation_set)
    gate = load_gate_table(args.gate_table, slug_of, relation_set)
    freq = load_frequency(args, relation_set)
    order = {"hernandez": 0, "engineered": 1, "bear": 2}
    panel.sort(key=lambda r: (order[relation_set[r]], r))
    report_models, worst = {}, 0.0
    for model in models:
        cohort, excluded = [], []
        for rel in panel:
            key = (model, rel)
            g = gate.get(key)
            if key in ineligible:
                excluded.append({"relation": rel, "reason": "outcome row ineligible: " + ", ".join(ineligible[key])})
            elif key not in outcomes:
                excluded.append({"relation": rel, "reason": "no final repeat-0 outcome row"})
            elif g is None:
                excluded.append({"relation": rel, "reason": "no CE artifact"})
            elif g["status"] != "passed":
                excluded.append({"relation": rel, "reason": g["reason"]})
            else:
                cohort.append(rel)
        if len(cohort) < 4:
            report_models[model] = {"n": len(cohort), "relations": cohort, "excluded": excluded,
                                    "computable": False}
            continue
        rows = [outcomes[(model, r)] for r in cohort]
        variables = {
            "x": np.array([gate[(model, r)]["x"] for r in cohort], float),
            "y": np.array([row["heldout_cand_acc"] for row in rows], float),
            "pre_rel_cand_acc": np.array([row["pre_rel_cand_acc"] for row in rows], float),
            "log_n_train": np.log(np.array([row["n_train"] for row in rows], float)),
            "log_n_classes": np.log(np.array([row["n_classes"] for row in rows], float)),
            "subject_frequency": np.array([freq[r]["value"] for r in cohort], float),
        }
        indices = bootstrap_indices(model, len(cohort), args.bootstrap_draws)
        groups, w = analyse(variables, indices)
        worst = max(worst, w)

        cap = dict(variables)
        cap["subject_frequency"] = np.array([freq[r].get("cap250", freq[r]["value"]) for r in cohort], float)
        cap_groups, _ = analyse(cap, indices[:0])
        report_models[model] = {
            "n": len(cohort),
            "relations": cohort,
            "n_by_relation_set": dict(collections.Counter(relation_set[r] for r in cohort)),
            "excluded": excluded,
            "computable": True,
            "predictor_ce": {"column_groups": groups},

            "comparison": {g: {"ce_partial_pearson_r": groups[g]["partial_pearson_r"] if groups[g]["computable"] else None,
                               "ce_rank_partial_spearman_rho": groups[g]["rank_partial_spearman_rho"] if groups[g]["computable"] else None,
                               "ce_n": groups[g]["n"],
                               "ce_residual_df": groups[g]["residual_df"],
                               "ce_printable": groups[g]["computable"]} for g, _ in COLUMN_GROUPS},
            "sensitivity_bear_frequency_cap250": {
                g: {"partial_pearson_r": cap_groups[g]["partial_pearson_r"],
                    "rank_partial_spearman_rho": cap_groups[g]["rank_partial_spearman_rho"]}
                for g in ("subject_frequency", "all_jointly")},
            "design_diagnostics": design_diagnostics(variables),
            "bootstrap_seed_uint64": bootstrap_seed(model),
            "per_relation": {r: {"relation_set": relation_set[r], "arm_set": outcomes[(model, r)]["arm_set"],
                                 "run_id": outcomes[(model, r)].get("run_id"),
                                 **{k: float(v[i]) for k, v in variables.items()}}
                             for i, r in enumerate(cohort)},
        }
    if worst > IDENTITY_TOLERANCE:
        raise ReportError(f"estimator identities disagree by {worst:.3e}")
    ns = sorted({m["n"] for m in report_models.values()})
    return {
        "schema": SCHEMA,
        "variant": args.variant,
        "relation_sets_selected": list(sets),
        "relation_selection": {"selection": selection, "sources": list(sets),
                               "digest": relation_sets.digest(sorted(panel)),
                               "registry": str(relation_sets.DEFAULT_PATH),
                               "registry_sha256": sha256(relation_sets.DEFAULT_PATH)},
        "collection": args.collection,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "estimand": {
            "predictor": {"field": "ce_gain (scripts/ce_gate_table.py), ce.results[0].gain of a G_new-passed artifact",
                          "label": "LIN-CLS-CE gain, real label set",
                          "relation_aggregation": "none: one CE artifact per model-relation cell, no repeat axis"},
            "outcome": {"field": "heldout_cand_acc", "label": "NEW-REL-GEN at epoch 20",
                        "relation_aggregation": REPEAT_AGGREGATION},
            "partial_pearson": "Pearson correlation of the OLS residuals of predictor and outcome on the same intercept-plus-controls design",
            "rank_partial_spearman": "the identical construction after mid-ranking predictor, outcome and every control",
            "estimator_source": "results-numbers.org entry 3.1e; paper sections/protocols/statistics.tex",
            "ties": "average (mid) ranks",
            "residual_df_formula": "n - k - 2",
        },
        "column_groups": {g: list(c) for g, c in COLUMN_GROUPS},
        "cohort_rule": ("a relation enters a model's cohort when its final repeat-0 row (P(4) view) is complete and "
                        "production-eligible at epoch 20 and its CE cell passes G_new; nothing else removes it"),
        "relation_sets": {s: sum(1 for r in panel if relation_set[r] == s) for s in order if s in sets},
        "n_per_model": {m: report_models[m]["n"] for m in models},
        "equal_n": len(ns) == 1,
        "rows_set_aside": set_aside,
        "bootstrap": {"draws": args.bootstrap_draws, "base_seed": BOOTSTRAP_SEED,
                      "generator": "numpy.random.default_rng(sha256(seed\\0model)[:8] big-endian)",
                      "interval": "percentile, 2.5 and 97.5, linear interpolation",
                      "shared_indices": "one index matrix per model, reused by every column group and both coefficients"},
        "frequency": {"index": FREQUENCY_INDEX,
                      "definition": "relation mean over payload rows of log10(1 + exact SUBJECT count)",
                      "study_rows": "the study payload (screen-included rows, capped at 250)",
                      "bear_rows": "every row of the relation in the BEAR pack (no cap); relstats53 subj_mean_log",
                      "aliases": BEAR_FREQUENCY_ALIASES,
                      "per_relation": {r: freq[r] for r in sorted(freq)}},
        "max_estimator_identity_discrepancy": worst,
        "sources": {name: {"file": str(p), "sha256": sha256(p)} for name, p in (
            ("rows", args.rows), ("gate_table", args.gate_table), ("roster", args.roster), ("spec", args.spec),
            ("study_payload", args.study_payload), ("study_payload_provenance", args.study_payload_provenance),
            ("bear_payload", args.bear_payload), ("bear_counts", args.bear_counts),
            ("bear_relstats", args.bear_relstats), ("script", Path(__file__).resolve()))}
        | {"study_counts": [{"file": str(p), "sha256": sha256(p)} for p in args.study_counts]},
        "models": report_models,
    }

def parse_sets(text: str) -> tuple:
    sets = tuple(s for s in RELATION_SETS if s in text.split(","))
    unknown = sorted(set(text.split(",")) - set(RELATION_SETS))
    if unknown or not sets:
        raise ReportError(f"--sets takes a comma list of {RELATION_SETS}; got {text!r}")
    return sets

def summary(report) -> str:
    lines = [f"fleet joint controls, sets {'+'.join(report['relation_sets_selected'])}, "
             f"variant {report['variant']}, collection {report['collection']}, "
             f"outcome {report['estimand']['outcome']['relation_aggregation']}",
             "n per model: " + ", ".join(f"{m} {n}" for m, n in report["n_per_model"].items()),
             "columns: " + " | ".join(g for g, _ in COLUMN_GROUPS) + "  (r rho each)"]
    for model, m in report["models"].items():
        if not m["computable"]:
            lines.append(f"{model:<22} n={m['n']} not computable")
            continue
        f = lambda v: "  n/a " if v is None else "%+.3f" % v
        cells = " | ".join(f"{f(c['ce_partial_pearson_r'])} {f(c['ce_rank_partial_spearman_rho'])}"
                           if c["ce_printable"] else " n/a (df %d)  " % c["ce_residual_df"]
                           for c in m["comparison"].values())
        lines.append(f"{model:<22} n={m['n']:<3} {cells}")
    for model, m in report["models"].items():
        for e in m["excluded"]:
            lines.append(f"excluded {model} {e['relation']}: {e['reason']}")
    return "\n".join(lines) + "\n"

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--gate-table", type=Path, required=True)
    ap.add_argument("--roster", type=Path, required=True)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--study-payload", type=Path, required=True)
    ap.add_argument("--study-payload-provenance", type=Path, required=True)
    ap.add_argument("--study-counts", type=Path, nargs="+", required=True)
    ap.add_argument("--bear-payload", type=Path, required=True)
    ap.add_argument("--bear-counts", type=Path, required=True)
    ap.add_argument("--bear-relstats", type=Path, required=True)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="headline")
    ap.add_argument("--collection", choices=("provisional", "final"), required=True)
    ap.add_argument("--sets", default=",".join(RELATION_SETS),
                    help="comma list of relation sets forming the cohort (default: all three)")
    ap.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    a = ap.parse_args(argv)
    for out in (a.output, a.summary):
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
    try:
        report = build(a)
    except ReportError as exc:
        print(f"FLEET_JOINT_CONTROLS_FAIL {exc}", file=sys.stderr)
        return 1
    with open(a.output, "x", encoding="utf-8") as fh:
        fh.write(json.dumps(report, indent=1, sort_keys=True) + "\n")
    text = summary(report)
    with open(a.summary, "x", encoding="utf-8") as fh:
        fh.write(text)
    sys.stdout.write(text)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
