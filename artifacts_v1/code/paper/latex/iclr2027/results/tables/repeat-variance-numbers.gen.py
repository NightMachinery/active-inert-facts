import argparse
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools"))
from machine_generated import MG, REPO, header, sha256_of
TEX = MG / "repeat-variance-numbers.tex"
GROUP = {"hernandez": "Hernandez", "engineered": "Engineered", "bear53-minicpm5": "Bear"}

def pct(v):
    return f"{100 * v:.1f}"

def pct_up(v):
    return f"{math.ceil(round(1000 * v, 9)) / 10:.1f}"

def numbers(agg, label):
    out = [f"% aggregate: snapshot_sha256 {agg['snapshot_sha256']}"]
    m = lambda name, v: out.append(f"\\newcommand{{\\repVar{name}}}{{{v}}}")
    if set(agg["panels"]) != set(GROUP):
        sys.exit(f"panels {sorted(agg['panels'])}, expected {sorted(GROUP)}")
    for pid, g in GROUP.items():
        A = agg["panels"][pid]
        m(f"{g}R", f"{A['pairs_pearson']:.2f}")
        m(f"{g}RLo", f"{A['pairs_pearson_ci'][0]:.2f}")
        m(f"{g}RHi", f"{A['pairs_pearson_ci'][1]:.2f}")
        m(f"{g}Cells", A["cells"])
        pm = A["per_model"]
        if len(pm) == 1:
            (P,) = pm.values()
            m(f"{g}Share", pct(P["run_to_run_share"]))
            m(f"{g}ShareLo", pct(P["run_to_run_share_ci"][0]))
            m(f"{g}ShareHi", pct(P["run_to_run_share_ci"][1]))
            continue
        s = {k: P["run_to_run_share"] for k, P in pm.items()}
        top = max(s, key=s.get)
        m(f"{g}ShareMin", pct(min(s.values())))
        m(f"{g}ShareMax", pct(s[top]))
        m(f"{g}ShareMaxModel", label[top])
        m(f"{g}ShareMean", pct(statistics.fmean(s.values())))
        m(f"{g}ShareMedian", pct(statistics.median(s.values())))
        m(f"{g}ShareCiHiMax", pct_up(max(P["run_to_run_share_ci"][1] for P in pm.values())))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", default=str(HERE / "../../../../plots/repeat-variance/aggregate.json"))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args()
    agg = json.loads(Path(a.aggregate).read_text())
    spec = json.loads((Path(a.aggregate).parent / agg["spec"]).read_text())
    label = {x["id"]: x["label"] for x in spec["models"]}
    agg_path = Path(a.aggregate).resolve()
    block = "\n".join([*numbers(agg, label)]) + "\n"
    rel = agg_path.relative_to(REPO)
    new = header(f"latex/iclr2027/results/tables/repeat-variance-numbers.gen.py", [(str(rel), sha256_of(agg_path))],
                 f"cd latex/iclr2027/results/tables && python3 repeat-variance-numbers.gen.py --write") + block
    text = TEX.read_text() if TEX.exists() else ""
    if a.check:
        sys.exit(0 if new == text else "machine-generated/repeat-variance-numbers.tex is stale; rerun with --write")
    if a.write:
        TEX.write_text(new)
    print(block)

if __name__ == "__main__":
    main()
