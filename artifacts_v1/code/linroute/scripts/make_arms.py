#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linroute.prompts import digests as prompt_digests
from linroute.provenance import git_commit
from linroute.sandbox import (
    add_exceptions,
    arm_name,
    load_animals,
    load_regions,
    load_templates,
    load_towns,
    make_records,
    region_to_animal,
    rule_namelen,
    rule_random,
    rule_region,
    split_heldout,
)

HELDOUT_LABEL_RULE = {"region": "region", "random": "region", "namelen": "namelen"}

def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--towns", default="data/sandbox/towns_candidates.tsv")
    p.add_argument("--regions", default="data/sandbox/regions.tsv")
    p.add_argument("--animals", default="data/sandbox/animals.txt")
    p.add_argument("--selected", default=None,
                   help="optional file of town names the base model knows, one "
                        "per line; unlisted candidates are dropped before the "
                        "per-region truncation")
    p.add_argument("--per-region", type=int, default=10)
    p.add_argument("--heldout-per-region", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", default="region,random,namelen")
    p.add_argument("--exceptions", default="0,20",
                   help="comma-separated integer percentages")
    p.add_argument("--out", default="data/arms/")
    a = p.parse_args(argv)

    def resolve(x: str) -> Path:
        q = Path(x)
        return q if q.is_absolute() else root / q

    regions = load_regions(resolve(a.regions))
    animals = load_animals(resolve(a.animals))
    templates = load_templates()
    selected = None
    if a.selected:
        selected = [ln.strip() for ln in
                    resolve(a.selected).read_text(encoding="utf-8").splitlines()
                    if ln.strip()]

    towns = load_towns(resolve(a.towns), a.per_region, selected)
    trained, heldout = split_heldout(towns, a.heldout_per_region, a.seed)
    r2a = region_to_animal(regions, animals, a.seed)

    held_names = {t.name for t in heldout}
    k_nominal = len(trained) // len(animals)
    rules = [r.strip() for r in a.arms.split(",") if r.strip()]
    pcts = [int(x) for x in a.exceptions.split(",") if x.strip() != ""]
    out_root = resolve(a.out)
    out_root.mkdir(parents=True, exist_ok=True)
    commit = git_commit(root)

    plan: list[tuple[str, int]] = []
    for rule in rules:
        wanted = pcts if rule == "region" else [0]
        for pct in dict.fromkeys(wanted):
            plan.append((rule, pct))
        if rule != "region" and any(p != 0 for p in pcts):
            print(f"note: {rule} is a control rule, not an exception rate; "
                  f"writing {arm_name(rule, k_nominal, 0)} only")

    written = []
    for rule, pct in plan:
        name = arm_name(rule, k_nominal, pct)

        if rule == "region":
            labels = rule_region(towns, r2a)
        elif rule == "random":
            labels = rule_random(towns, r2a, a.seed,
                                 trained=[t.name for t in trained])
        elif rule == "namelen":
            labels = rule_namelen(towns, animals)
        else:
            raise SystemExit(f"unknown rule {rule!r}")

        label_rules = {t.name: rule for t in trained}
        label_rules.update({t.name: HELDOUT_LABEL_RULE[rule] for t in heldout})

        changed: list[str] = []
        if pct:
            train_labels = {t.name: labels[t.name] for t in trained}
            train_labels, changed = add_exceptions(
                train_labels, pct / 100.0, a.seed, animals)
            labels = {**labels, **train_labels}
            for town in changed:
                label_rules[town] = "exception"

        recs = make_records(towns, labels, templates,
                            heldout=held_names, label_rules=label_rules)

        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "mapping.tsv", "w", encoding="utf-8") as fh:
            fh.write("town\tregion\tanimal\theldout\tlabel_rule\n")
            for t in towns:
                fh.write("\t".join([
                    t.name, t.region, labels[t.name],
                    "1" if t.name in held_names else "0",
                    label_rules[t.name],
                ]) + "\n")
        for split in ("train", "gate", "heldout"):
            write_jsonl(d / f"{split}.jsonl", recs[split])

        counts = Counter(labels[t.name] for t in trained)
        meta = {
            "arm": name,
            "rule": rule,
            "exceptions_pct_requested": pct,
            "exceptions_changed": len(changed),
            "exceptions_pct_realized": round(100 * len(changed) / len(trained), 4),
            "exception_towns": changed,
            "k_nominal": k_nominal,
            "trained_per_animal": dict(sorted(counts.items())),
            "heldout_label_rule": HELDOUT_LABEL_RULE[rule],
            "seed": a.seed,
            "per_region": a.per_region,
            "heldout_per_region": a.heldout_per_region,
            "n_towns": len(towns),
            "n_trained": len(trained),
            "n_heldout": len(heldout),
            "n_train_records": len(recs["train"]),
            "n_gate_records": len(recs["gate"]),
            "n_heldout_records": len(recs["heldout"]),
            "regions": [r.name for r in regions],
            "animals": animals,
            "region_to_animal": r2a,
            "heldout_towns": sorted(held_names),
            "templates": templates,
            "inputs": {"towns": a.towns, "regions": a.regions,
                       "animals": a.animals,
                       "templates": "data/prompts/sandbox.yaml",
                       "selected": a.selected},

            "prompt_files": prompt_digests(),
            "git_commit": commit,
        }
        (d / "arm.json").write_text(
            json.dumps(meta, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8")
        written.append(name)
        print(f"{name}: {len(recs['train'])} train, {len(recs['gate'])} gate, "
              f"{len(recs['heldout'])} heldout -> {d}")

    if not written:
        raise SystemExit("no arms written; check --arms and --exceptions")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
