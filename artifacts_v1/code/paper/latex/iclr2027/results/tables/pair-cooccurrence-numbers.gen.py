import argparse
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools"))
from machine_generated import MG, REPO, header, sha256_of
TEX = MG / "pair-cooccurrence-numbers.tex"

def floor2(v):
    return math.floor(round(v * 100, 9)) / 100

def ceil2(v):
    return math.ceil(round(v * 100, 9)) / 100

def numbers(d):
    models = d["models"]
    g = lambda m, grp: models[m]["groups"][grp]
    col = lambda grp, k: [g(m, grp)[k] for m in models]
    raw = col("raw", "r")
    alone = col("pair_frequency", "r")
    alone_rho = col("pair_frequency", "rho")
    subj = col("subject_and_pair_frequency", "r")
    lowest = min(g(m, "pair_frequency")["r_ci"][0] for m in models)
    shift = max(abs(g(m, "all_jointly_plus_pair")["r"] - g(m, "all_jointly")["r"]) for m in models)
    cov, panel = d["coverage"]["covered_by_set"], d["coverage"]["panel_by_set"]
    m = lambda name, v: f"\\newcommand{{\\{name}}}{{{v}}}"
    f2 = lambda v: f"{v:.2f}"
    return [
        f"% {len(models)} models; {d['coverage']['n_covered']} of {d['coverage']['n_panel']} relations covered",
        m("pairCoveredRelations", d["coverage"]["n_covered"]),
        m("pairCoveredHernandez", cov["hernandez"]),
        m("pairPanelHernandez", panel["hernandez"]),
        m("pairCoveredTown", cov["engineered"]),
        m("pairPanelTown", panel["engineered"]),
        m("pairCoveredBear", cov["bear"]),
        m("pairPanelBear", panel["bear"]),
        m("pairRawRMin", f2(min(raw))),
        m("pairRawRMax", f2(max(raw))),
        m("pairAloneRMin", f2(min(alone))),
        m("pairAloneRMax", f2(max(alone))),
        m("pairAloneRMedian", f2(statistics.median(alone))),
        m("pairAloneRMean", f2(statistics.fmean(alone))),
        m("pairAloneRLowestBound", f2(floor2(lowest))),
        m("pairAloneRhoMin", f2(min(alone_rho))),
        m("pairAloneRhoMax", f2(max(alone_rho))),
        m("pairSubjRMin", f2(min(subj))),
        m("pairSubjRMax", f2(max(subj))),
        m("pairAllMaxShift", f2(ceil2(shift))),
    ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(HERE / "../../../../data/pair-cooccurrence/pair-controls.json"))
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--write", action="store_true")
    grp.add_argument("--check", action="store_true")
    a = ap.parse_args()
    src = Path(a.source).resolve()
    block = "\n".join(numbers(json.loads(src.read_text()))) + "\n"
    new = header("latex/iclr2027/results/tables/pair-cooccurrence-numbers.gen.py",
                 [(str(src.relative_to(REPO)), sha256_of(src))],
                 "cd latex/iclr2027/results/tables && python3 pair-cooccurrence-numbers.gen.py --write") + block
    text = TEX.read_text() if TEX.exists() else ""
    if a.check:
        sys.exit(0 if new == text else "machine-generated/pair-cooccurrence-numbers.tex is stale; rerun with --write")
    if a.write:
        TEX.write_text(new)
    print(block)

if __name__ == "__main__":
    main()
