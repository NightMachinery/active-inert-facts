import argparse
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools"))
from machine_generated import MG, REPO, header, sha256_of
TEX = MG / "label-set-numbers.tex"
WORDS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen".split()

def floor2(v):
    return math.floor(round(v * 100, 9)) / 100

def numbers(agg):
    ref = agg["reference_pool"]
    cat = [p for p in agg["pools_present"] if p != ref]
    if agg["pools_missing"]:
        sys.exit(f"pools missing from the aggregate: {agg['pools_missing']}")
    P = agg["pools"]
    r = [P[p]["pearson_ce_gen"] for p in cat]
    rho = [P[p]["rank_agreement"] for p in cat]
    excl = sum(P[p]["pearson_ci"][0] > 0 for p in cat)
    lowest = min(P[p]["pearson_ci"][0] for p in cat)
    m = lambda name, v: f"\\newcommand{{\\{name}}}{{{v}}}"
    return [
        f"% aggregate: snapshot_sha256 {agg['snapshot_sha256']}; {len(cat)} category pools, reference {ref}, "
        f"{agg['cohort_relations']} relations, {len(agg['dropped_relations'])} dropped",
        m("labelSetPools", len(cat)),
        m("labelSetPoolsWord", WORDS[len(cat)]),
        m("labelSetRelations", agg["cohort_relations"]),
        m("labelSetDroppedRelations", len(agg["dropped_relations"])),
        m("labelSetReferenceR", f"{P[ref]['pearson_ce_gen']:.2f}"),
        m("labelSetRMin", f"{min(r):.2f}"),
        m("labelSetRMax", f"{max(r):.2f}"),
        m("labelSetRMedian", f"{statistics.median(r):.2f}"),
        m("labelSetRMean", f"{statistics.fmean(r):.2f}"),
        m("labelSetRLowestBound", f"{floor2(lowest):.2f}"),
        m("labelSetRExcludesZero", excl),
        m("labelSetRExcludesZeroWord", WORDS[excl]),
        m("labelSetRankRhoMin", f"{min(rho):.2f}"),
        m("labelSetRankRhoMax", f"{max(rho):.2f}"),
    ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", default=str(HERE / "../../../../plots/label-set/aggregate.json"))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args()
    agg = json.loads(Path(a.aggregate).read_text())
    if agg.get("synthetic"):
        sys.exit("aggregate is SYNTHETIC; refusing to write paper numbers")
    agg_path = Path(a.aggregate).resolve()
    block = "\n".join([*numbers(agg)]) + "\n"
    rel = agg_path.relative_to(REPO)
    new = header(f"latex/iclr2027/results/tables/label-set-numbers.gen.py", [(str(rel), sha256_of(agg_path))],
                 f"cd latex/iclr2027/results/tables && python3 label-set-numbers.gen.py --write") + block
    text = TEX.read_text() if TEX.exists() else ""
    if a.check:
        sys.exit(0 if new == text else "machine-generated/label-set-numbers.tex is stale; rerun with --write")
    if a.write:
        TEX.write_text(new)
    print(block)

if __name__ == "__main__":
    main()
