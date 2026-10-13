#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("per_relation", HERE / "per-relation.gen.py")
per_relation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(per_relation)
from machine_generated import MG, header
import model_order as registry

TEX = MG / "finetune-acc-rows.tex"
FIELD = "train_recall_acc"
two = per_relation.two

def model_labels(models):
    try:
        reg = registry.load()
        registry.check(models, "plots/merged-panel/spec.json", reg)
    except registry.ModelOrderError as error:
        sys.exit(f"model order: {error}")
    return {m["id"].casefold(): m["label"] for m in reg["models"]}

def build(rows_path):
    _, sha, _, _, models, _, display, groups = per_relation.build(rows_path)
    rows = [json.loads(l) for l in Path(rows_path).read_text().splitlines() if l.strip()]
    labels = model_labels(models)
    value = {}
    for r in rows:
        v = r.get(FIELD)
        if not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1:
            sys.exit(f"refusing: {r['model']} {r['relation']}: {FIELD} is {v!r}")
        if r.get("final_epoch") != r.get("planned_epochs"):
            sys.exit(f"refusing: {r['model']} {r['relation']} was measured at epoch "
                     f"{r.get('final_epoch')}, not its last ({r.get('planned_epochs')})")
        value[(r["model"], r["relation"])] = v
    ncol = 1 + len(models)
    spec_entry = json.loads(per_relation.SPEC.read_text())["relation_labels"]
    main_path, main_sha, _ = per_relation.pinned(spec_entry["hernandez_engineered"], "label map")
    bear_path, bear_sha, _ = per_relation.pinned(spec_entry["bear"], "label map")
    root = per_relation.ROOT
    lines = [header("latex/iclr2027/results/tables/finetune-acc.gen.py",
                    [(str(Path(rows_path).resolve()), sha),
                     (str(main_path.relative_to(root)), main_sha),
                     (str(bear_path.relative_to(root)), bear_sha)],
                    "cd latex/iclr2027/results/tables && python3 finetune-acc.gen.py --rows <rows-joined-11m.jsonl> --write"),
             f"% cells: {FIELD} at the final epoch; rows and names as in Table 4 (per-relation.gen.py)\n"]
    heads = " & ".join(f"\\finetuneAccHead{{{labels[m.casefold()]}}}" for m in models)
    lines.append(f"\\newcommand{{\\finetuneAccModelHeads}}{{{heads}}}\n")
    lines.append("\\newcommand{\\finetuneAccRows}{%\n")
    for k, (_, name, group) in enumerate(groups):
        if k:
            lines.append("    \\midrule\n")
        lines.append(f"    \\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{name}}}}} \\\\\n")
        for rel in group:
            cells = " & ".join(two(value[(m, rel)]) for m in models)
            lines.append(f"    {per_relation.tex_label(display[rel])} & {cells} \\\\\n")
    lines.append("}\n")
    rels = [rel for _, _, group in groups for rel in group]
    every = [value[(m, rel)] for m in models for rel in rels]
    below = sorted((value[(m, rel)], m, rel) for m in models for rel in rels if value[(m, rel)] < 1)
    lines.append(f"\\newcommand{{\\finetuneAccMin}}{{{two(min(every))}}}\n"
                 f"\\newcommand{{\\finetuneAccMedian}}{{{two(statistics.median(every))}}}\n"
                 f"\\newcommand{{\\finetuneAccMean}}{{{two(statistics.fmean(every))}}}\n"
                 f"\\newcommand{{\\finetuneAccBelowOne}}{{{len(below)}}}\n")
    return "".join(lines), every, below, models, rels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    body, every, below, models, rels = build(a.rows)
    print(f"{FIELD}: {len(rels)} relations x {len(models)} models; min {min(every):.4f}, "
          f"median {statistics.median(every):.4f}, mean {statistics.fmean(every):.4f}; "
          f"{len(below)} cells below 1: {below}")
    old = TEX.read_text() if TEX.exists() else ""
    if a.write:
        TEX.write_text(body)
        print(f"{TEX.name}: {'written' if body != old else 'unchanged'}")
    elif body != old:
        print("the fine-tuning accuracy table differs from the rows; run with --write", file=sys.stderr)
        sys.exit(1)
    else:
        print("ok: the fine-tuning accuracy table matches the rows")

if __name__ == "__main__":
    main()
