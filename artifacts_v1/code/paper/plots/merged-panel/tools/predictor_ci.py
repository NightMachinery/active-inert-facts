#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--linroute", type=Path, required=True)
ap.add_argument("--headline", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

sys.path.insert(0, str(a.linroute))
script = a.linroute / "scripts" / "fleet_joint_controls.py"
spec = importlib.util.spec_from_file_location("fjc", script)
fjc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fjc)
fjc.COLUMN_GROUPS = (("raw", ()), ("recall", ("pre_rel_cand_acc",)))

raw_bytes = a.headline.read_bytes()
head = json.loads(raw_bytes)
draws = head["bootstrap"]["draws"]
out = {"schema": "linroute/predictor-ci/1",
       "headline": str(a.headline), "headline_sha256": hashlib.sha256(raw_bytes).hexdigest(),
       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       "fleet_joint_controls_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
       "bootstrap": head["bootstrap"], "models": {}}
for model, m in head["models"].items():
    rels = m["relations"]
    pr = m["per_relation"]
    if set(pr) != set(rels):
        sys.exit(f"{model}: per_relation and the cohort list name different relations")

    col = lambda k: np.array([pr[r][k] for r in rels], float)
    x, y, rc = col("x"), col("y"), col("pre_rel_cand_acc")
    idx = fjc.bootstrap_indices(model, len(rels), draws)
    g_fwd, _ = fjc.analyse({"x": x, "y": y, "pre_rel_cand_acc": rc}, idx)
    g_rev, _ = fjc.analyse({"x": rc, "y": y, "pre_rel_cand_acc": x}, idx)
    ref = m["predictor_ce"]["column_groups"]
    for g in ("raw", "recall"):
        for k in ("partial_pearson_r", "rank_partial_spearman_rho"):
            if abs(g_fwd[g][k] - ref[g][k]) > 1e-12:
                sys.exit(f"{model} {g} {k}: {g_fwd[g][k]} != headline {ref[g][k]}")
        for k in ("pearson_r_percentile_95_ci", "spearman_rho_percentile_95_ci"):
            if max(abs(u - v) for u, v in zip(g_fwd[g]["bootstrap"][k], ref[g]["bootstrap"][k])) > 1e-12:
                sys.exit(f"{model} {g} {k}: {g_fwd[g]['bootstrap'][k]} != headline {ref[g]['bootstrap'][k]}")
    pick = lambda c: {"r": c["partial_pearson_r"], "rho": c["rank_partial_spearman_rho"],
                      "r_ci": c["bootstrap"]["pearson_r_percentile_95_ci"],
                      "rho_ci": c["bootstrap"]["spearman_rho_percentile_95_ci"],
                      "undefined_share": max(c["bootstrap"]["undefined_share_pearson"], c["bootstrap"]["undefined_share_spearman"])}
    out["models"][model] = {"n": len(rels), "relations": rels,
                            "gain": pick(g_fwd["raw"]), "gain_given_recall": pick(g_fwd["recall"]),
                            "recall": pick(g_rev["raw"]), "recall_given_gain": pick(g_rev["recall"])}
    print(model, {k: round(v["r"], 3) for k, v in out["models"][model].items() if isinstance(v, dict)}, flush=True)
a.out.write_text(json.dumps(out, indent=1) + "\n")
print("wrote", a.out)
