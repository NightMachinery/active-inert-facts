import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools"))
from machine_generated import MG, REPO, header, sha256_of
sys.path.insert(0, str(HERE.parents[3] / "plots" / "shared"))
from half_width import half_width

TEX = MG / "failure-shapley-numbers.tex"
AGG = REPO / "plots" / "failure-shapley" / "aggregate.json"
BARS = {"observed": "Observed", "shuffle": "Shuffled", "chance": "Chance"}
FAMILIES = {"probe": "Probe", "recall": "Recall", "ftset": "FineTuning", "epoch0": "EpochZero"}
SEGMENTS = {"probe_top1": "ProbeRankOne", "probe_rank2": "ProbeRankTwo", "probe_rank3": "ProbeRankThree",
            "recall_rank1": "RecallRankOne", "recall_rank2": "RecallRankTwo", "recall_rank3": "RecallRankThree",
            "majority_f": "MostFrequent", "knn5_raw": "Nearest", "epoch0": "EpochZero"}

def pct(v):
    return f"{100 * v:.1f}"

def h(est, ci):
    return half_width(est, ci, 1, 100)

def numbers(agg):
    m = lambda name, v: f"\\newcommand{{\\{name}}}{{{v}}}"
    out = [m("failErrors", f"{agg['n_err']:,}".replace(",", "{,}")), m("failCells", agg["n_cells"])]
    for b, B in BARS.items():
        e = agg["bars"][b]["explained"]
        out.append(m(f"failExplained{B}", pct(e["v"])))
        if e["ci"] is not None:
            out.append(m(f"failExplained{B}H", h(e["v"], e["ci"])))
    for b in ("observed", "shuffle"):
        o = agg["over_models"][b]
        out += [m(f"failExplainedModel{BARS[b]}Min", pct(o["min"])), m(f"failExplainedModel{BARS[b]}Max", pct(o["max"]))]
    for f, F in FAMILIES.items():
        out += [m(f"failFamily{F}{B}", pct(agg["families"][f][b])) for b, B in BARS.items()]
    for s, S in SEGMENTS.items():
        out += [m(f"failShare{S}{B}", pct(agg["bars"][b]["segments"][s])) for b, B in BARS.items()]
    return out

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args()
    agg = json.loads(AGG.read_text())
    new = header("latex/iclr2027/results/tables/failure-shapley-numbers.gen.py",
                 [(str(AGG.relative_to(REPO)), sha256_of(AGG))],
                 "cd latex/iclr2027/results/tables && python3 failure-shapley-numbers.gen.py --write") \
        + "\n".join(numbers(agg)) + "\n"
    old = TEX.read_text() if TEX.exists() else ""
    if a.check:
        sys.exit(0 if new == old else "machine-generated/failure-shapley-numbers.tex is stale; rerun with --write")
    if a.write:
        TEX.write_text(new)
    print(new, end="")

if __name__ == "__main__":
    main()
