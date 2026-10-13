#!/usr/bin/env python3
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools"))
from machine_generated import MG, REPO, header, sha256_of

TEX = MG / "predictor-numbers.tex"
TABLE_ONE = MG / "primary-cls-associations-numbers.tex"
COUNTS = HERE.parents[1] / "notation" / "counts.tex"
SPEC = REPO / "plots" / "merged-panel" / "spec.json"
QUANTITIES = [("gain", "Gain"), ("gain_given_recall", "GainGivenRecall"),
              ("recall", "Recall"), ("recall_given_gain", "RecallGivenGain")]

TABLE_ONE_STEM = {"gain": "Raw", "gain_given_recall": "Recall"}
WORDS = "zero one two three four five six seven eight nine ten eleven twelve".split()

def num(v):
    return f"$-${abs(v):.2f}" if round(v, 2) < 0 else f"{v:.2f}"

def pooled_relations():
    text = COUNTS.read_text()
    n = 0
    for name in ("nHernandezRelations", "nEngineeredRelations", "nBearRelations"):
        m = re.search(r"\\newcommand\{\\" + name + r"\}\{(\d+)\}", text)
        if not m:
            sys.exit(f"notation/counts.tex defines no literal \\{name}")
        n += int(m.group(1))
    return n

def table_one():
    text = TABLE_ONE.read_text()
    pin = re.search(r"fleet-joint-controls\.headline\.json \(sha256 ([0-9a-f]{64})\)", text)
    if not pin:
        sys.exit(f"{TABLE_ONE.name} pins no fleet-joint-controls.headline.json")
    return pin.group(1), dict(re.findall(r"\\newcommand\{\\(tableOne\w+)\}\{([^}]*)\}", text))

def numbers(ci, ids):
    report_sha, t1 = table_one()
    if ci["headline_sha256"] != report_sha:
        sys.exit(f"predictor intervals come from report {ci['headline_sha256']}, Table 1 from {report_sha}")
    if set(ci["models"]) != set(ids):
        sys.exit(f"predictor-interval models {sorted(ci['models'])} are not the spec's {sorted(ids)}")
    want_n = pooled_relations()
    for m in ids:
        if ci["models"][m]["n"] != want_n:
            sys.exit(f"{m}: n = {ci['models'][m]['n']}, not \\nPooledRelations = {want_n}")
    out = [f"% {len(ids)} models, {want_n} relations each"]
    for q, stem in QUANTITIES:
        for c, tag in (("r", "R"), ("rho", "Rho")):
            vals, excl = [], 0
            for m in ids:
                e = ci["models"][m][q]
                v, (lo, hi) = e[c], e[f"{c}_ci"]
                if not lo <= v <= hi:
                    sys.exit(f"{m} {q} {c}: {v} outside its interval [{lo}, {hi}]")
                vals.append(v)
                excl += lo > 0 or hi < 0
            for name, v in (("Min", min(vals)), ("Max", max(vals)), ("Mean", statistics.fmean(vals)),
                            ("Median", statistics.median(vals))):
                text = num(v)
                if q in TABLE_ONE_STEM:
                    key = f"tableOne{TABLE_ONE_STEM[q]}{tag}{name}"
                    if t1.get(key) != text:
                        sys.exit(f"{q} {c} {name} = {text}, but Table 1's \\{key} is {t1.get(key)}")
                out.append(f"\\newcommand{{\\predictor{stem}{tag}{name}}}{{{text}}}")
            out.append(f"\\newcommand{{\\predictor{stem}{tag}ExcludesZero}}{{{excl}}}")
            out.append(f"\\newcommand{{\\predictor{stem}{tag}ExcludesZeroWord}}{{{WORDS[excl]}}}")
    return "\n".join(out) + "\n"

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true", help="the default")
    a = ap.parse_args()
    spec = json.loads(SPEC.read_text())
    src = SPEC.parent / spec["predictor_ci"]["path"]
    if sha256_of(src) != spec["predictor_ci"]["sha256"]:
        sys.exit(f"{src.name}: sha256 differs from the pin in plots/merged-panel/spec.json")
    ci = json.loads(src.read_text())
    block = numbers(ci, [m["id"] for m in spec["models"]])
    new = header("latex/iclr2027/results/tables/predictor-numbers.gen.py",
                 [(str(src.relative_to(REPO)), sha256_of(src)), (str(TABLE_ONE.relative_to(REPO)), sha256_of(TABLE_ONE))],
                 "cd latex/iclr2027/results/tables && python3 predictor-numbers.gen.py --write") + block
    old = TEX.read_text() if TEX.exists() else ""
    if a.write:
        TEX.write_text(new)
        print(f"{TEX.name}: {'written' if new != old else 'unchanged'}")
    elif new != old:
        print("machine-generated/predictor-numbers.tex is stale; rerun with --write", file=sys.stderr)
        sys.exit(1)
    else:
        print("ok: the predictor figure's numbers match its intervals")

if __name__ == "__main__":
    main()
