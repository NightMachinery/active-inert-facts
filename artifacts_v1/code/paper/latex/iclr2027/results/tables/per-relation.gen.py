#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parents[1] / "tools"))
sys.path.insert(0, str(ROOT / "plots" / "shared"))
from machine_generated import MG, header
import json5_io
import relation_sets
TEX = MG / "per-relation-rows.tex"
SPEC = ROOT / "plots" / "merged-panel" / "spec.json"
AGG = ROOT / "plots" / "merged-panel" / "aggregate.json"
SETS = [("hernandez", r"\RelSetNameHernandez{}"), ("engineered", r"\RelSetNameEngineered{}"),
        ("bear", r"\RelSetNameBear{}")]
NCOL = 10

DAGGER = r"\smash{$^\dagger$}"

def two(v):
    text = f"{abs(v):.2f}".lstrip("0") if round(abs(v), 2) < 1 else f"{abs(v):.2f}"
    return f"$-${text}" if round(v, 2) < 0 else text

def three(v):
    return f"{v:.3f}".lstrip("0")

def tex_label(label):
    text = label.replace("\u2192", r"\ensuremath{\to}")
    rest = text.replace(r"\ensuremath{\to}", "")
    if set(rest) & set("\\&%#_$^~{}") or any(ord(c) > 127 for c in rest):
        sys.exit(f"refusing: label {label!r} has a character this generator does not typeset")
    return text

def figure_labels(ids, label_of, source_of, set_name):
    names = {set_name(s) for s, _ in SETS}
    bare = {}
    for rid in ids:
        label = label_of[rid]
        m = re.match(r"^(.*) \(([^()]+)\)$", label)
        own = set_name(source_of[rid])
        if m and m.group(2) != own and m.group(2) in names:
            sys.exit(f"relation {rid}: label {label!r} carries another set's name")
        bare[rid] = m.group(1) if m and m.group(2) == own else label
    count = {}
    for b in bare.values():
        count[b] = count.get(b, 0) + 1
    out = {rid: (f"{bare[rid]} ({set_name(source_of[rid])})" if count[bare[rid]] > 1 else bare[rid])
           for rid in ids}
    if len(set(out.values())) != len(out):
        sys.exit("two relations still share a figure label after disambiguation")
    return out

def pinned(entry, what):
    path = (SPEC.parent / entry["path"]).resolve()
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != entry["sha256"]:
        sys.exit(f"{what} {path.name}: sha256 {sha} does not match {SPEC.name}'s pin")
    return path, sha, json5_io.loads(raw.decode("utf-8"), str(path))

def row_order(rels, source, stats):
    return [(src, name, sorted((r for r in rels if source[r] == src),
                               key=lambda r: (-stats[r]["y"], r)))
            for src, name in SETS]

def build(rows_path):
    raw = Path(rows_path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    spec = json.loads(SPEC.read_text())
    models = [m["id"] for m in spec["models"]]
    agg = {r["relation"]: r for r in json.loads(AGG.read_text())["per_relation"]}
    cells = {}
    for r in rows:
        if r.get("repeat") != 0 or not r.get("final"):
            sys.exit(f"refusing: {r.get('run_id')} is not a final repeat-0 row")
        if r["model"] not in models:
            sys.exit(f"refusing: model {r['model']} is not in {SPEC.name}")
        key = (r["model"], r["relation"])
        if key in cells:
            sys.exit(f"refusing: {key} appears twice")
        cells[key] = r
    rels = sorted({rel for _, rel in cells})
    if set(rels) != set(agg):
        sys.exit(f"relations differ from {AGG.name}: {sorted(set(rels) ^ set(agg))[:5]}")
    missing = [(m, rel) for rel in rels for m in models if (m, rel) not in cells]
    if missing:
        sys.exit(f"refusing: {len(missing)} model-relation cells missing, e.g. {missing[:3]}; "
                 "the table and its caption assume every model ran every relation")
    source = {rel: agg[rel]["source"] for rel in rels}
    if set(source.values()) - {k for k, _ in SETS}:
        sys.exit(f"unknown relation set in {AGG.name}: {set(source.values())}")

    sets = relation_sets.load(ROOT / "metadata" / "relation-sets.json5")
    main_path, main_sha, main = pinned(spec["relation_labels"]["hernandez_engineered"], "label map")
    bear_path, bear_sha, bear = pinned(spec["relation_labels"]["bear"], "label map")
    label_of = {rel: (bear if source[rel] == "bear" else main)["relations"].get(rel) for rel in rels}
    if not all(label_of.values()):
        sys.exit(f"relations without a display label: {[r for r, l in label_of.items() if not l]}")
    for rel in rels:
        if label_of[rel] != agg[rel]["label"]:
            sys.exit(f"{rel}: label {label_of[rel]!r} differs from {AGG.name}'s {agg[rel]['label']!r}")
    display = figure_labels(rels, label_of, source, sets.display_name)

    order = [r["relation"] for r in json.loads(AGG.read_text())["per_relation"]]
    size = math.ceil(len(order) / 2)
    half = {rel: ("upper" if i < size else "lower") for i, rel in enumerate(order)}
    stats = {}
    for rel in rels:
        own = [cells[(m, rel)] for m in models]
        xs = [c["pre_rel_cls_ce_gain"] for c in own]
        ys = [c["heldout_cand_acc"] for c in own]
        es = [c["n_classes"] for c in own]
        hs = [statistics.median(c["holdout_per_class"].values()) for c in own]
        e = statistics.median(es)
        s = {"n": len(own), "e": e, "chance": 1 / e, "y": statistics.fmean(ys),
             "ymed": statistics.median(ys), "ymin": min(ys), "ymax": max(ys),
             "x": statistics.fmean(xs), "xmed": statistics.median(xs),
             "single": all(h == 1 for h in hs), "single_models": sum(h == 1 for h in hs)}
        a = agg[rel]
        mine = (s["n"], two(s["x"]), two(s["xmed"]), two(s["y"]), two(s["ymed"]), two(s["ymin"]),
                two(s["ymax"]))
        theirs = (a["n_models"], two(a["mean_x"]), two(a["x_median"]), two(a["mean_y"]),
                  two(a["y_median"]), two(a["y_min"]), two(a["y_max"]))
        if mine != theirs:
            sys.exit(f"{rel}: n/mean x/median x/mean y/median y/min/max {mine} differ from "
                     f"{AGG.name}'s {theirs}")
        stats[rel] = s
    lines = [header("latex/iclr2027/results/tables/per-relation.gen.py",
                    [(str(Path(rows_path).resolve()), sha),
                     (str(main_path.relative_to(ROOT)), main_sha),
                     (str(bear_path.relative_to(ROOT)), bear_sha)],
                    "cd latex/iclr2027/results/tables && python3 per-relation.gen.py --rows <rows-joined-11m.jsonl> --write"),
             f"% {len(rows)} rows, {len(models)} models; the dagger marks a relation whose median "
             "held-out count per answer is 1 on every model\n"
             "\\newcommand{\\perRelationRows}{%\n"]
    groups = row_order(rels, source, stats)
    for k, (src, name, group) in enumerate(groups):
        if k:
            lines.append("    \\midrule\n")
        lines.append(f"    \\multicolumn{{{NCOL}}}{{@{{}}l}}{{\\textit{{{name}}}}} \\\\\n")
        for rel in group:
            s = stats[rel]
            e = f"{s['e']:g}"
            mark = DAGGER if s["single"] else ""
            lines.append(f"    {tex_label(display[rel])}{mark} & {half[rel]} & {e} & {three(s['chance'])} & "
                         f"{two(s['y'])} & {two(s['ymed'])} & {two(s['ymin'])} & {two(s['ymax'])} & "
                         f"{two(s['x'])} & {two(s['xmed'])} \\\\\n")
    lines.append("}\n")
    counts = {src: sum(1 for r in rels if source[r] == src) for src, _ in SETS}
    return "".join(lines), sha, stats, counts, models, source, display, groups

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    body, sha, stats, counts, models, source, _, _ = build(a.rows)
    old = TEX.read_text() if TEX.exists() else ""
    new = body
    print(f"{len(stats)} relations {counts}; every relation on all {len(models)} models")
    single = sorted(r for r, s in stats.items() if s["single"])
    by_set = {src: sum(1 for r in single if source[r] == src) for src, _ in SETS}
    print(f"median h_e = 1 on every model (daggered): {len(single)} {by_set}: {', '.join(single)}")
    partial = {r: s["single_models"] for r, s in stats.items() if 0 < s["single_models"] < len(models)}
    print(f"median h_e = 1 on some but not all models: {partial}")
    if a.write:
        TEX.write_text(new)
        print(f"{TEX.name}: {'written' if new != old else 'unchanged'}")
    elif new != old:
        print("Table 4 differs from the rows; run with --write", file=sys.stderr)
        sys.exit(1)
    else:
        print("ok: Table 4 matches the rows and the merged panel's relation aggregates")

if __name__ == "__main__":
    main()
