#!/usr/bin/env python3
import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools"))
from machine_generated import MG, REPO, header, sha256_of
sys.path.insert(0, str(HERE.parents[3] / "plots" / "shared"))
from half_width import half_width
ROWS = MG / "primary-cls-associations-rows.tex"
SUMMARY = MG / "primary-cls-associations-summary.tex"
NUMBERS = MG / "primary-cls-associations-numbers.tex"

INTERVALS_PROSE = HERE.parents[1] / "sections" / "appendix" / "intervals.tex"
COUNTS = HERE.parents[1] / "notation" / "counts.tex"
SPEC = HERE.parents[3] / "plots" / "merged-panel" / "spec.json"
MODELS = HERE.parents[3] / "metadata" / "models.json5"
MODEL_ORDER = HERE.parents[3] / "plots" / "shared" / "model_order.py"

GROUPS = [("raw", "Raw"), ("recall", r"\ctrlRecallHeader{}"),
          ("answer_count", r"\ctrlAnswerCountHeader{}"), ("subject_count", r"\ctrlSubjectCountHeader{}"),
          ("subject_frequency", r"\ctrlFreqHeader{}"), ("all_jointly", r"\ctrlAllHeader{}")]

COVARIATES = {"raw": [], "recall": ["pre_rel_cand_acc"], "answer_count": ["log_n_classes"],
              "subject_count": ["log_n_train"], "subject_frequency": ["subject_frequency"],
              "all_jointly": ["pre_rel_cand_acc", "log_n_train", "log_n_classes", "subject_frequency"]}

STEMS = [("raw", "Raw"), ("recall", "Recall"), ("answer_count", "AnswerCount"), ("subject_count", "SubjectCount"),
         ("subject_frequency", "SubjectFrequency"), ("all_jointly", "AllJointly")]
COEFS = (("ce_partial_pearson_r", "pearson_r_percentile_95_ci", "undefined_share_pearson"),
         ("ce_rank_partial_spearman_rho", "spearman_rho_percentile_95_ci", "undefined_share_spearman"))

def cell(v):
    text = f"{abs(v):.2f}".lstrip("0")
    return f"$-${text}" if v < 0 else text

def pm(v, interval):
    return "\\pmHalf{" + half_width(v, interval, 2).lstrip("0") + "}"

def floor2(v):
    return math.floor(round(v * 100, 9)) / 100

def num(v):
    return f"$-${abs(v):.2f}" if round(v, 2) < 0 else f"{v:.2f}"

def numbers(report, spec_models):
    out = []
    for g, stem in STEMS:
        group_lowest = math.inf
        for (coef, ci, _), tag in zip(COEFS, ("R", "Rho")):
            vals = [report["models"][m]["comparison"][g][coef] for m in spec_models]
            lowest = min(report["models"][m]["predictor_ce"]["column_groups"][g]["bootstrap"][ci][0]
                         for m in spec_models)
            group_lowest = min(group_lowest, lowest)
            for name, v in (("Min", min(vals)), ("Max", max(vals)), ("Mean", statistics.fmean(vals)),
                            ("Median", statistics.median(vals))):
                if not min(vals) <= v <= max(vals):
                    sys.exit(f"{name} of {g} {coef} = {v} outside [{min(vals)}, {max(vals)}]")
                out.append(f"\\newcommand{{\\tableOne{stem}{tag}{name}}}{{{num(v)}}}")
            out.append(f"\\newcommand{{\\tableOne{stem}{tag}LowestBound}}{{{num(floor2(lowest))}}}")
        out.append(f"\\newcommand{{\\tableOne{stem}LowestBound}}{{{num(floor2(group_lowest))}}}")

    singles = [g for g, stem in STEMS if stem not in ("Raw", "AllJointly")]
    for (coef, _, _), tag in zip(COEFS, ("R", "Rho")):
        vals = [report["models"][m]["comparison"][g][coef] for g in singles for m in spec_models]
        out.append(f"\\newcommand{{\\tableOneSingleControl{tag}Min}}{{{num(min(vals))}}}")
        out.append(f"\\newcommand{{\\tableOneSingleControl{tag}Max}}{{{num(max(vals))}}}")
    return "\n".join(out) + "\n"

def model_order(spec_models):
    sys.path.insert(0, str(MODEL_ORDER.parent))
    import model_order as registry
    try:
        reg = registry.load()
        registry.check(spec_models, "plots/merged-panel/spec.json", reg)
    except registry.ModelOrderError as error:
        sys.exit(f"model order: {error}")
    by_id = {m["id"].casefold(): m for m in spec_models}
    rows = [{"id": by_id[m["id"].casefold()]["id"], "label": m["label"]}
            for m in reg["models"] if m["id"].casefold() in by_id]
    if len(rows) != len(spec_models):
        sys.exit("model order: the merged-panel spec holds a model twice")
    for row in rows:
        if row["label"] != by_id[row["id"].casefold()]["label"]:
            sys.exit(f"model order: {row['id']} is labelled {row['label']!r} in {MODELS.name} "
                     f"but {by_id[row['id'].casefold()]['label']!r} in the merged-panel spec")
    return rows

def pooled_relations():
    text = COUNTS.read_text()
    n = 0
    for name in ("nHernandezRelations", "nEngineeredRelations", "nBearRelations"):
        m = re.search(r"\\newcommand\{\\" + name + r"\}\{(\d+)\}", text)
        if not m:
            sys.exit(f"notation/counts.tex defines no literal \\{name}")
        n += int(m.group(1))
    return n

def check_repeat_rule(report):
    agg = report.get("estimand", {}).get("outcome", {}).get("relation_aggregation")
    if not isinstance(agg, str):
        sys.exit("refusing: report states no estimand.outcome.relation_aggregation, so the repeat rule cannot be checked")
    low = agg.lower()
    if "repeat 0" not in low or "mean" in low or "average" in low:
        sys.exit(f"refusing: outcome aggregation {agg!r} is not repeat 0 alone (repeat-0 rule)")

def build(report):
    spec = json.loads(SPEC.read_text())
    models = report["models"]
    want_n = pooled_relations()
    groups = [g for g, _ in GROUPS]
    if report["column_groups"] != COVARIATES:
        sys.exit(f"report column groups {report['column_groups']} are not the table's {COVARIATES}")
    rows = []
    ordered = model_order(spec["models"])
    for m in ordered:
        if m["id"] not in models:
            sys.exit(f"report has no model {m['id']}")
        mr = models[m["id"]]
        comp = mr["comparison"]
        for g in groups:
            if comp[g]["ce_n"] != want_n or not comp[g]["ce_printable"]:
                sys.exit(f"{m['id']} {g}: n = {comp[g]['ce_n']}, printable {comp[g]['ce_printable']}; "
                         f"the caption states n = {want_n} for every model")
            boot = mr["predictor_ce"]["column_groups"][g]["bootstrap"]
            if boot["draws_requested"] != report["bootstrap"]["draws"]:
                sys.exit(f"{m['id']} {g}: {boot['draws_requested']} draws, report says {report['bootstrap']['draws']}")
            for coef, ci, _ in COEFS:
                v, (lo, hi) = comp[g][coef], boot[ci]
                if not lo <= v <= hi:
                    sys.exit(f"{m['id']} {g} {coef}: {v} outside its interval [{lo}, {hi}]")
        vals = [f"\\tableOneValue{{{cell(comp[g][coef])}}}{{{pm(comp[g][coef], mr['predictor_ce']['column_groups'][g]['bootstrap'][ci])}}}"
                for g in groups for coef, ci, _ in COEFS]
        if rows:
            rows.append("    \\tableOneRowSep")
        rows.append(f"    \\tableOneModel{{{m['label']}}} & {' & '.join(vals[:10])}\n"
                    f"      \\jointControlCell{{{vals[10]}}}{{{vals[11]}}} \\\\")
    return ("\n".join(rows) + "\n", report["bootstrap"]["draws"],
            numbers(report, [m["id"] for m in spec["models"]]), summary(report, [m["id"] for m in ordered]))

def summary(report, ids):
    out = []
    groups = [g for g, _ in GROUPS]
    for name, fn in (("Min", min), ("Mean", statistics.fmean), ("Max", max)):
        vals = []
        for g in groups:
            for coef, _, _ in COEFS:
                col = [report["models"][i]["comparison"][g][coef] for i in ids]
                v = fn(col)
                if not min(col) <= v <= max(col):
                    sys.exit(f"{name} of {g} {coef} = {v} outside [{min(col)}, {max(col)}]")
                vals.append(cell(v))
        last = f"\\tableOneLastCell{{{vals[11]}}}" if name == "Max" else vals[11]
        out.append(f"    \\tableOneSummary{{{name}}} & {' & '.join(vals[:10])}\n"
                   f"      \\jointControlCell{{{vals[10]}}}{{{last}}} \\\\")
    return "\n".join(out) + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    report = json.loads(Path(a.report).read_text())
    check_repeat_rule(report)
    rows, draws, macros, summ = build(report)
    prose = INTERVALS_PROSE.read_text()
    if f"{draws:,}".replace(",", "{,}") not in prose:
        sys.exit(f"{INTERVALS_PROSE.name}: its prose does not state the report's {draws} draws")
    head = header("latex/iclr2027/results/tables/primary-cls-associations.gen.py",
                  [(str(Path(a.report).resolve()), sha256_of(a.report)),
                   (str(SPEC.relative_to(REPO)), sha256_of(SPEC)), (str(MODELS.relative_to(REPO)), sha256_of(MODELS))],
                  "cd latex/iclr2027/results/tables && python3 primary-cls-associations.gen.py "
                  "--report <fleet-joint-controls.headline.json> --write")
    outputs = {ROWS: head + "\\newcommand{\\tableOneModelRows}{%\n" + rows + "}\n",
               SUMMARY: head + "\\newcommand{\\tableOneSummaryRows}{%\n" + summ + "}\n",
               NUMBERS: head + macros}
    changes = [(path, path.read_text() if path.exists() else "", new) for path, new in outputs.items()]
    if a.write:
        for path, old, new in changes:
            path.write_text(new)
            print(f"{path.name}: {'written' if new != old else 'unchanged'}")
    elif any(new != old for _, old, new in changes):
        print("Table 1 or its prose numbers differ from the report; run with --write", file=sys.stderr)
        sys.exit(1)
    else:
        print("ok: Table 1 and its prose numbers match the report")

if __name__ == "__main__":
    main()
