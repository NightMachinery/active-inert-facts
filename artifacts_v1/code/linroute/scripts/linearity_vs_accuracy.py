#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linroute.relation_labels import DEFAULT_PATH as DEFAULT_RELATION_LABELS
from linroute.relation_labels import RelationLabels, load as load_relation_labels
from linroute.relation_labels import row_relation_id

MODEL_ORDER = ["allenai__Olmo-3-1025-7B", "Qwen__Qwen3.5-9B-Base", "Qwen__Qwen3.5-9B", "Qwen__Qwen3.8-27B"]
MODEL_NAMES = {
    "allenai__Olmo-3-1025-7B": "OLMo-3 7B (base)",
    "Qwen__Qwen3.5-9B-Base": "Qwen3.5 9B Base",
    "Qwen__Qwen3.5-9B": "Qwen3.5 9B (post-trained)",
    "Qwen__Qwen3.8-27B": "Qwen3.8 27B (post-trained)",
}

COLUMNS = [
    ("relation", "relation", "relation", "text"),
    ("n", r"\vert S\vert", "|S|", "int"),
    ("n_objects_present", r"\vert O\vert", "|O|", "int"),
    ("k", r"\vert S\vert/\vert O\vert", "|S|/|O|", "k"),
    ("fs_acc", "5-shot", "5-shot", "acc"),
    ("zs_acc", "0-shot", "0-shot", "acc"),
    ("lin_obj_q16_delta", r"\Delta LIN-OBJ^{s_0\to t_0} \vert_{q=16}", "Δ <span class=fam>LIN-OBJ</span><sup>s<sub>0</sub>→t<sub>0</sub></sup> |<sub>q=16</sub>", "delta"),
    ("lin_obj_q64_delta", r"\Delta LIN-OBJ^{s_0\to t_0} \vert_{q=64}", "Δ <span class=fam>LIN-OBJ</span><sup>s<sub>0</sub>→t<sub>0</sub></sup> |<sub>q=64</sub>", "delta"),
    ("lin_cls_rel_q16_acc", r"LIN-CLS^{s_0} \vert_{q=16}", "<span class=fam>LIN-CLS</span><sup>s<sub>0</sub></sup> |<sub>q=16</sub>", "acc"),
    ("lin_cls_rel_q64_acc", r"LIN-CLS^{s_0} \vert_{q=64}", "<span class=fam>LIN-CLS</span><sup>s<sub>0</sub></sup> |<sub>q=64</sub>", "acc"),
    ("lin_cls_chance", "chance", "chance", "acc"),
    ("gap", "gap", "gap", "gap"),
]

GAP_KEY = {"cls": "lin_cls_rel_q16_acc", "obj": "lin_obj_q16_delta"}

CAPACITY_RE = re.compile(r"_q(\d+)_")

def present_capacities(models: dict[str, list[dict]]) -> list[int]:
    out: set[int] = set()
    for rows in models.values():
        for row in rows:
            for key in row:
                m = CAPACITY_RE.search(key)
                if m:
                    out.add(int(m.group(1)))
    return sorted(out)

def select_columns(models: dict[str, list[dict]]) -> tuple[list[tuple], list[str]]:
    qs = present_capacities(models)
    kept, dropped = [], []
    for col in COLUMNS:
        m = CAPACITY_RE.search(col[0])
        if m and int(m.group(1)) not in qs:
            dropped.append(col[0])
        else:
            kept.append(col)
    return kept, dropped

def dropped_note(dropped: list[str]) -> str:
    if not dropped:
        return ""
    return (
        "No row reports " + ", ".join(f"`{d}`" for d in dropped)
        + ": these rows were fitted with `ProbeConfig.q_secondary = None`, which is "
        "the default, so only the primary capacity was measured and those columns "
        "are omitted rather than left blank."
    )

def load_rows(results_dir: Path, gap: str = "cls") -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    gk = GAP_KEY[gap]
    for d in sorted(results_dir.iterdir()):
        f = d / "probes.json"
        if not f.is_file():
            continue
        rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        for r in rows:
            r["n_objects_present"] = r.get("n_objects_present", r.get("n_objects"))
            r["k"] = r["n"] / r["n_objects_present"] if r.get("n_objects_present") else None
            g = r.get(gk)
            r["feasible"] = isinstance(g, (int, float)) and not math.isnan(g)
            r["gap"] = (r["fs_acc"] - g) if r["feasible"] else None
        rows.sort(key=lambda r: (not r["feasible"], -(r["gap"] if r["gap"] is not None else -9), r["relation"]))
        out[d.name] = rows
    return out

def fmt(v, kind: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if kind == "int":
        return str(int(v))
    if kind == "text":
        return str(v)
    if kind == "k":
        return f"{v:.1f}"
    return f"{v:.2f}"

def write_org(models: dict[str, list[dict]], path: Path, source: str,
              labels: RelationLabels, columns: list[tuple] | None = None,
              note: str = "") -> None:
    columns = COLUMNS if columns is None else columns
    lines = [
        "#+TITLE: Linearity against accuracy, per PRE-REL relation",
        "#+OPTIONS: toc:nil",
        "",
        f"Source: linroute {source}. Notation as in the paper: the probe family carries the fitted map in the",
        "superscript; every row is a PRE-REL relation measured on the base model, so the relation subscript is dropped.",
        "*5-shot*, *0-shot*: the model's own first-token accuracy. *\\Delta LIN-OBJ^{s_0\\to t_0}|_{q}*: accuracy of the",
        "affine map from the subject's last-token state under the relation prompt to the answer state, decoded through",
        "the head, minus the constant-answer baseline; nested layer selection; PCA capacity q = 16 or 64 (64 where",
        "feasible). *LIN-CLS^{s_0}|_{q}*: class-probe accuracy (ridge to one-hot object, no head); *chance* = 1/|O|.",
        "*gap* = 5-shot minus LIN-CLS^{s_0}|_{q=16}: large when the model answers the relation well but the object is",
        "not linearly readable from the subject state. Caution: when |O| is close to n (one subject per object) the",
        "class probe has no training examples of the evaluation class and its accuracy is 0 by construction. Sorted by",
        "gap, descending; rows whose q=16 probe was infeasible are listed last. |S| = subjects, |O| = distinct objects,",
        "|S|/|O| = mean subjects per object, k in the paper's formulas (the class probe needs it well above 1).",
        "",
    ]
    if note:
        lines += [note, ""]
    for m in MODEL_ORDER:
        if m not in models:
            continue
        lines.append(f"* {MODEL_NAMES.get(m, m)}")
        lines.append("| " + " | ".join(c[1] for c in columns) + " |")
        lines.append("|-")
        for r in models[m]:
            lines.append("| " + " | ".join(
                labels.label(row_relation_id(r)) if k == "relation" else fmt(r.get(k), kind)
                for k, _, _, kind in columns
            ) + " |")
        lines.append("")
    path.write_text("\n".join(lines))

CSS = """
:root { --bg:#fbfbf8; --fg:#1f1f1f; --line:#d9d6cc; --head:#efece2; --fam:#1a3d6d; }
body { margin:0; padding:24px 32px; font:14px/1.45 -apple-system, "Helvetica Neue", Arial, sans-serif; color:var(--fg); background:var(--bg); }
h1 { font-size:20px; margin:0 0 6px; } h2 { font-size:16px; margin:28px 0 8px; }
p.legend { max-width:1100px; color:#444; margin:0 0 8px; }
p.filter { margin:6px 0 12px; } p.filter input { width:4.5em; margin:0 10px 0 4px; } #count { color:#666; }
table { border-collapse:collapse; margin:6px 0 18px; font-variant-numeric:tabular-nums; }
th, td { border-bottom:1px solid var(--line); padding:4px 10px; text-align:right; white-space:nowrap; }
th { background:var(--head); cursor:pointer; position:sticky; top:0; font-weight:600; }
th.sorted::after { content:" ▾"; } th.sorted.asc::after { content:" ▴"; }
td:first-child, th:first-child { text-align:left; }
.fam { font-family: "Times New Roman", Times, serif; letter-spacing:.02em; color:var(--fam); }
sup, sub { font-size:.72em; } sup sub { font-size:.9em; }
tr.infeasible td { color:#888; font-style:italic; }
td.gap { font-weight:600; }
.bar { display:inline-block; height:9px; background:#c8553d; vertical-align:middle; margin-right:6px; border-radius:2px; }
.bar.neg { background:#3d7cc8; }
nav a { margin-right:14px; }
"""

JS = """
function applyFilter() {
  const kmin = +document.getElementById('kmin').value || 0, omin = +document.getElementById('omin').value || 0;
  let shown = 0, total = 0;
  document.querySelectorAll('tbody tr').forEach(r => {
    const k = +r.dataset.k, o = +r.dataset.o; total++;
    const ok = (isNaN(k) ? true : k >= kmin) && (isNaN(o) ? true : o >= omin);
    r.style.display = ok ? '' : 'none'; if (ok) shown++;
  });
  document.getElementById('count').textContent = shown + ' of ' + total + ' rows';
}
['kmin','omin'].forEach(id => document.getElementById(id).addEventListener('input', applyFilter));
applyFilter();
document.querySelectorAll('table').forEach(t => {
  t.querySelectorAll('th').forEach((th, i) => th.addEventListener('click', () => {
    const asc = th.classList.contains('sorted') && !th.classList.contains('asc');
    t.querySelectorAll('th').forEach(h => h.classList.remove('sorted','asc'));
    th.classList.add('sorted'); if (asc) th.classList.add('asc');
    const rows = [...t.tBodies[0].rows];
    const val = r => { const c = r.cells[i]; const v = c.dataset.v; return v === undefined || v === '' ? null : (isNaN(+v) ? v : +v); };
    rows.sort((a, b) => { const x = val(a), y = val(b);
      if (x === null && y === null) return 0; if (x === null) return 1; if (y === null) return -1;
      return (x < y ? -1 : x > y ? 1 : 0) * (asc ? 1 : -1); });
    rows.forEach(r => t.tBodies[0].appendChild(r));
  }));
});
"""

def write_html(models: dict[str, list[dict]], path: Path, source: str,
               labels: RelationLabels, columns: list[tuple] | None = None,
               note: str = "") -> None:
    columns = COLUMNS if columns is None else columns
    parts = [
        "<!doctype html><meta charset=utf-8>"
        '<meta name="linroute-relation-labels-source" content="paper/metadata/relation-labels.json5">'
        f'<meta name="linroute-relation-labels-sha256" content="{labels.sha256}">'
        "<title>Linearity vs accuracy (PRE-REL)</title>",
        f"<style>{CSS}</style>",
        "<h1>Linearity against accuracy, per <span class=fam>PRE-REL</span> relation</h1>",
        f"<p class=legend>Source: linroute <code>{html.escape(source)}</code>. Every row is a pre-existing relation measured on the "
        "base model, so the relation subscript is dropped from the probe names. "
        "<b>5-shot</b>, <b>0-shot</b>: the model's own first-token accuracy. "
        "<b>Δ <span class=fam>LIN-OBJ</span><sup>s<sub>0</sub>→t<sub>0</sub></sup> |<sub>q</sub></b>: accuracy of the affine map from the subject's "
        "last-token state (relation prompt) to the answer state, decoded through the head, minus the constant-answer baseline; "
        "nested layer selection; PCA capacity q = 16 or 64 (64 where feasible). "
        "<b><span class=fam>LIN-CLS</span><sup>s<sub>0</sub></sup> |<sub>q</sub></b>: class-probe accuracy (ridge to one-hot object, no head); "
        "<b>chance</b> = 1/|O|. "
        "<b>gap</b> = 5-shot − <span class=fam>LIN-CLS</span><sup>s<sub>0</sub></sup> |<sub>q=16</sub>: large when the model answers the "
        "relation well but the object is not linearly readable from the subject state. "
        "<i>Caution:</i> when |O| is close to n (one subject per object) the class probe has no training example of the "
        "evaluation class and scores 0 by construction; read those rows with the chance column. "
        "Sorted by gap; click a header to re-sort. Greyed rows: q=16 probe infeasible (too few subjects). |S| = subjects, |O| = distinct objects, "
        "<b>|S|/|O|</b> = mean subjects per object (k in the paper's formulas).</p>"
        "<p class=filter><label>Show rows with |S|/|O| ≥ <input id=kmin type=number min=0 step=0.5 value=0 size=4></label> "
        "<label>and |O| ≥ <input id=omin type=number min=1 step=1 value=1 size=4></label> "
        "<span id=count></span></p>",
        "<nav>" + " ".join(f"<a href='#{m}'>{html.escape(MODEL_NAMES.get(m, m))}</a>" for m in MODEL_ORDER if m in models) + "</nav>",
    ]
    if note:
        parts.insert(-1, f"<p class=legend>{html.escape(note)}</p>")
    for m in MODEL_ORDER:
        if m not in models:
            continue
        parts.append(f"<h2 id='{m}'>{html.escape(MODEL_NAMES.get(m, m))}</h2><table><thead><tr>")
        parts.append("".join(f"<th>{c[2]}</th>" for c in columns) + "</tr></thead><tbody>")
        for r in models[m]:
            cls = "" if r["feasible"] else " class=infeasible"
            cls += f" data-k='{r['k'] if r['k'] is not None else ''}' data-o='{r['n_objects_present'] or ''}'"
            cells = []
            for k, _, _, kind in columns:
                v = labels.label(row_relation_id(r)) if k == "relation" else r.get(k)
                s = fmt(v, kind)
                raw = "" if v is None or (isinstance(v, float) and math.isnan(v)) else (str(v) if kind != "text" else html.escape(str(v)))
                if kind == "gap" and s:
                    w = int(abs(v) * 60)
                    s = f"<span class='bar{' neg' if v < 0 else ''}' style='width:{w}px'></span>{s}"
                    cells.append(f"<td class=gap data-v='{raw}'>{s}</td>")
                else:
                    cells.append(f"<td data-v='{raw}'>{html.escape(s) if kind == 'text' else s}</td>")
            parts.append(f"<tr{cls}>" + "".join(cells) + "</tr>")
        parts.append("</tbody></table>")
    parts.append(f"<script>{JS}</script>")
    path.write_text("\n".join(parts))

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--org", type=Path)
    ap.add_argument("--html", type=Path)
    ap.add_argument("--gap", choices=sorted(GAP_KEY), default="cls", help="probe the gap is measured against")
    ap.add_argument("--relation-labels", type=Path, default=DEFAULT_RELATION_LABELS,
                    help="canonical paper-repository relation-label map")
    a = ap.parse_args()
    labels = load_relation_labels(a.relation_labels)
    models = load_rows(a.results_dir, a.gap)
    labels.assert_complete(
        (row_relation_id(row) for rows in models.values() for row in rows),
        "linearity_vs_accuracy export",
    )
    source = str(a.results_dir)
    columns, dropped = select_columns(models)
    note = dropped_note(dropped)
    if a.org:
        write_org(models, a.org, source, labels, columns, note)
    if a.html:
        write_html(models, a.html, source, labels, columns, note)
    for m, rows in models.items():
        print(f"{m}: {len(rows)} rows, {sum(r['feasible'] for r in rows)} feasible")
    if dropped:
        print("skipped (no row reports them): " + ", ".join(dropped))

if __name__ == "__main__":
    main()
