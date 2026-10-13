#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parents[1] / "tools"))
sys.path.insert(0, str(ROOT / "plots" / "shared"))
from machine_generated import MG, header, sha256_of
import model_order as registry
from half_width import half_width

TEX = MG / "repeat-correlation-rows.tex"
NUMBERS = MG / "repeat-correlation-numbers.tex"
AGG = ROOT / "plots" / "repeat-variance" / "aggregate-correlation.json"
SCOPES = ["minicpm5-all", "ten-models-hernandez-town"]
SERIES = ["0", "1", "2", "mean"]
SUBSET_PANELS = ["hernandez", "engineered"]
MACRO_MODEL = {"minicpm5-2b-base": r"\MiniCpmModel{}"}
SERIES_NAMES = {"0": "RepeatZero", "1": "RepeatOne", "2": "RepeatTwo", "mean": "MeanOfThree"}
WORDS = dict(enumerate("zero one two three four five six seven eight nine ten eleven twelve".split()))

def two(v):
    s = f"{abs(v):.2f}"
    s = s[1:] if s.startswith("0") else s
    return f"$-${s}" if round(v, 2) < 0 else s

def num(v):
    return f"$-${abs(v):.2f}" if round(v, 2) < 0 else f"{v:.2f}"

def pm_cell(c):
    lo, hi = c["ci"]
    if not lo <= c["est"] <= hi:
        sys.exit(f"refusing: estimate {c['est']} outside its interval [{lo}, {hi}]")
    return f"\\repeatCorrValue{{{two(c['est'])}}}{{{half_width(c['est'], (lo, hi), 2).lstrip('0')}}}"

def labels():
    try:
        reg = registry.load()
    except registry.ModelOrderError as error:
        sys.exit(f"model order: {error}")
    return [m["id"] for m in reg["models"]], {m["id"].casefold(): m["label"] for m in reg["models"]}

def minicpm_rows(agg):
    S = agg["scopes"][SCOPES[0]]
    if len(S["per_model"]) != 1:
        sys.exit(f"refusing: scope {SCOPES[0]} has models {list(S['per_model'])}, not one")
    (P,) = S["per_model"].values()
    ref = P.get("table1")
    if ref is None:
        sys.exit(f"refusing: scope {SCOPES[0]} records no Table 1 reference")
    lines = []
    for coef, name in (("r", "Pearson $r$"), ("rho", "Spearman $\\rho$")):
        if two(P[coef][SERIES[0]]["est"]) != two(ref[coef]):
            sys.exit(f"refusing: repeat {SERIES[0]} {coef} does not reproduce Table 1's raw {coef}")
        lines.append(f"    \\repeatCorrRowName{{{name}}} & " + " & ".join(pm_cell(P[coef][s]) for s in SERIES) + " \\\\\n")
    return "".join(lines)

RUN_NAMES = {"0": "Repeat 0", "1": "Repeat 1", "2": "Repeat 2", "mean": "Mean of three"}

def minicpm_run_rows(agg):
    (P,) = agg["scopes"][SCOPES[0]]["per_model"].values()
    return "".join(f"    \\repeatCorrRowName{{{RUN_NAMES[s]}}} & {pm_cell(P['r'][s])} & {pm_cell(P['rho'][s])} \\\\\n"
                   for s in SERIES)

def minicpm_numbers(agg):
    (P,) = agg["scopes"][SCOPES[0]]["per_model"].values()
    return "".join(f"\\newcommand{{\\repeatMinicpm{tag}{SERIES_NAMES[s]}}}{{{num(P[coef][s]['est'])}}}\n"
                   for coef, tag in (("r", "R"), ("rho", "Rho")) for s in SERIES)

def subset_numbers(agg):
    runs = [s for s in SERIES if s != "mean"]
    per_model = agg["scopes"][SCOPES[1]]["per_model"].values()
    out = []
    for coef, tag in (("r", "R"), ("rho", "Rho")):
        vals = [P[coef][s]["est"] for P in per_model for s in runs]
        out.append(f"\\newcommand{{\\repeatSubset{tag}Min}}{{{num(min(vals))}}}\n")
        out.append(f"\\newcommand{{\\repeatSubset{tag}Max}}{{{num(max(vals))}}}\n")
    return "".join(out)

def body(agg, coef, order, label):
    sid = SCOPES[1]
    S = agg["scopes"][sid]
    if S["panels"] != SUBSET_PANELS:
        sys.exit(f"refusing: scope {sid} covers {S['panels']}, but the captions name {SUBSET_PANELS}")
    known = [m for m in order if m.casefold() in {x.casefold() for x in S["models"]}]
    if len(known) != len(S["models"]):
        sys.exit(f"refusing: scope {sid} has models outside metadata/models.json5: {S['models']}")
    by_key = {m.casefold(): m for m in S["models"]}
    lines = []
    for m in known:
        P = S["per_model"][by_key[m.casefold()]][coef]
        name = MACRO_MODEL.get(m.casefold(), label[m.casefold()])
        lines.append(f"    \\repeatCorrRowName{{{name}}} & " + " & ".join(pm_cell(P[s]) for s in SERIES) + " \\\\\n")
    O = S["over_models"][coef]
    lines.append("    \\addlinespace[2pt]\n")
    for w, word in (("mean", "Mean"), ("median", "Median")):
        lines.append(f"    \\repeatCorrRowName{{\\textit{{{word} over models}}}} & "
                     + " & ".join(pm_cell(O[w][s]) for s in SERIES) + " \\\\\n")
    return "".join(lines)

def build():
    agg = json.loads(AGG.read_text())
    if agg["series"] != SERIES:
        sys.exit(f"refusing: the aggregate's series are {agg['series']}, not {SERIES}")
    order, label = labels()
    sub = agg["scopes"][SCOPES[1]]
    head = header("latex/iclr2027/results/tables/repeat-correlation.gen.py",
                  [(str(AGG.relative_to(ROOT)), sha256_of(AGG))],
                  "cd latex/iclr2027/results/tables && python3 repeat-correlation.gen.py --write")
    counts = (f"\\newcommand{{\\repeatCorrAllRelations}}{{{agg['scopes'][SCOPES[0]]['relations']}}}\n"
              f"\\newcommand{{\\repeatCorrSubsetRelations}}{{{sub['relations']}}}\n"
              f"\\newcommand{{\\repeatCorrSubsetModelsWord}}{{{WORDS[len(sub['models'])]}}}\n")
    text = [head]
    text.append(f"\\newcommand{{\\repeatMinicpmRows}}{{%\n{minicpm_rows(agg)}}}\n")
    text.append(f"\\newcommand{{\\repeatMinicpmRunRows}}{{%\n{minicpm_run_rows(agg)}}}\n")
    for macro, coef in (("repeatPearsonRows", "r"), ("repeatSpearmanRows", "rho")):
        text.append(f"\\newcommand{{\\{macro}}}{{%\n{body(agg, coef, order, label)}}}\n")
    return {TEX: "".join(text), NUMBERS: head + counts + minicpm_numbers(agg) + subset_numbers(agg)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true", help="the default")
    a = ap.parse_args()
    changes = [(path, path.read_text() if path.exists() else "", new) for path, new in build().items()]
    if a.write:
        for path, old, new in changes:
            path.write_text(new)
            print(f"{path.name}: {'written' if new != old else 'unchanged'}")
    elif any(new != old for _, old, new in changes):
        print("the per-repeat correlation tables differ from the aggregate; run with --write", file=sys.stderr)
        sys.exit(1)
    else:
        print("ok: the per-repeat correlation tables match the aggregate")

if __name__ == "__main__":
    main()
