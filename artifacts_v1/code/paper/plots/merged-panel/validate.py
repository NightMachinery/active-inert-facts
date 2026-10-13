#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))

import layer_order
import check_terms
import relation_sets
import json5_io
import key_spacing

TOL = 1e-12

SOURCES = ("hernandez", "engineered", "bear")

_REGISTRY = relation_sets.load()
SOURCE_NAMES = {s: _REGISTRY.display_name(s) for s in SOURCES}
SOURCE_COLORS = {s: _REGISTRY.display_color(s) for s in SOURCES}
SOURCE_MARKS = {s: _REGISTRY.display_mark(s) for s in SOURCES}
ENCODED_SCATTERS = ("scatter-grid-neutral", "scatter-grid-neutral-simple", "scatter-grid-recall", "scatter-grid-answers",
                    "scatter-pooled-neutral", "scatter-pooled-recall", "scatter-pooled-model", "scatter-grid-sets", "scatter-grid-tint",
                    "scatter-grid-sets-nostats")
_HEATMAP_BASE = ("heatmap-recall", "heatmap-recall-sqrt", "heatmap-recall-quantile", "heatmap-recall-quantile-all")

LAYOUT_SUFFIXES = ("-by-gain", "-models-once")
HEATMAPS_ENC = (("heatmap-neutral", "heatmap-recall-simple", "heatmap-recall-observed") + _HEATMAP_BASE + tuple(v + "-sets" for v in _HEATMAP_BASE)
                + tuple("heatmap-recall-quantile-all-sets" + sfx for sfx in LAYOUT_SUFFIXES))
_STRIP_BASE = ("appendix-strips", "appendix-strips-recall", "appendix-strips-recall-part1", "appendix-strips-recall-part2")
STRIPS = ("appendix-strips-simple",) + _STRIP_BASE + tuple(v + "-sets" for v in _STRIP_BASE)

SET_COLOUR_VARIANTS = ("scatter-grid-sets", "scatter-grid-sets-nostats") + tuple(v for v in HEATMAPS_ENC + STRIPS if "-sets" in v)

SET_FREE_VARIANTS = tuple(v for v in ENCODED_SCATTERS + HEATMAPS_ENC + STRIPS + ("model-spread-highlight", "rank-agreement")
                          if v not in SET_COLOUR_VARIANTS)
KEYED_VARIANTS = ("scatter-grid", "scatter-grid-fits", "heatmap", "heatmap-interleaved")

FIG3_VARIANTS = ("scatter-grid-sets", "scatter-grid-sets-nostats")
CELL_RECT = re.compile(r'<rect x="([-0-9.]+)" y="([-0-9.]+)"[^>]*? fill="(#[0-9a-fA-F]{6})" data-model="([^"]+)" data-relation="([^"]+)" data-field="([xy])"( data-missing="1")?/>')

def die(msg):
    print(json.dumps({"status": "FAIL", "reason": msg}))
    sys.exit(1)

def read_spec(path):
    path = Path(path)
    own_text = path.read_text()
    own = json.loads(own_text)
    if "extends" in own:
        base_text = (path.parent / own["extends"]).read_text()
        spec = {**json.loads(base_text), **own}
        digest = hashlib.sha256((base_text + "\n" + own_text).encode()).hexdigest()
    else:
        spec, digest = own, hashlib.sha256(own_text.encode()).hexdigest()

    try:
        relation_sets.resolve_spec(spec, path)
    except relation_sets.RelationSetError as e:
        die(str(e))
    sets = spec.get("relation_sets", {})
    if set(sets) - set(SOURCES):
        die(f"relation_registry.relation_sets must name only the sets {list(SOURCES)}, found {sorted(sets)}")
    spec["relation_sets"] = {s: list(sets.get(s, [])) for s in SOURCES}
    if "relations" in spec:
        if sorted(r for s in SOURCES for r in spec["relation_sets"][s]) != sorted(r["id"] for r in spec["relations"]):
            die(f"{path}: the relation_sets pins do not add up to the relations pin")
    return spec, digest, path.parent

def pinned(base, block, what):
    raw = (base / block["path"]).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != block["sha256"]:
        die(f"{what} digest mismatch")
    return json5_io.loads(raw.decode("utf-8"), what), digest

def mean(v):
    return sum(v) / len(v)

def pearson(x, y):
    mx, my = mean(x), mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy)

def ranks(v):
    order = sorted(range(len(v)), key=lambda i: (v[i], i))
    out = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out

def spearman(x, y):
    return pearson(ranks(x), ranks(y))

def ols(x, y):
    mx, my = mean(x), mean(y)
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx) ** 2 for a in x)
    return slope, my - slope * mx

def q7(v, p):
    s = sorted(v)
    h = (len(s) - 1) * p
    lo, hi = math.floor(h), math.ceil(h)
    return s[lo] + (h - lo) * (s[hi] - s[lo])

def ramp(stops, t):
    t = max(0.0, min(1.0, t))
    pos = t * (len(stops) - 1)
    i = min(len(stops) - 2, math.floor(pos))
    u = pos - i
    a = [int(stops[i][k:k + 2], 16) for k in (1, 3, 5)]
    b = [int(stops[i + 1][k:k + 2], 16) for k in (1, 3, 5)]
    return "#" + "".join("%02x" % math.floor(c + (d - c) * u + 0.5) for c, d in zip(a, b))

ATTR = re.compile(r'([\w-]+)="([^"]*)"')
LINE = re.compile(r"<line\s([^>]*)/>")
MARK = re.compile(r'<(?:circle|rect|path) ([^>]*data-role="[^"]*"[^>]*)/>')

def highlighted(spec, relations):
    catalogue = {h["id"]: h for h in (spec.get("highlight") or {}).get("relations", [])}
    active = spec.get("highlight_relations", [])
    for rid in active:
        if rid not in catalogue:
            die(f"highlight_relations names {rid}, which spec.highlight.relations gives no shape")
    return {rid: catalogue[rid] for rid in active if rid in relations}

def marks_of(variant, spec):
    return None if variant.endswith("-simple") else SOURCE_MARKS

def cov_t(scale, v):
    if scale["scale"] == "log":
        return (math.log(v) - math.log(scale["lo"])) / (math.log(scale["hi"]) - math.log(scale["lo"]))
    return (v - scale["lo"]) / (scale["hi"] - scale["lo"])

def check_encoded(svg, svg_path, variant, spec, models, cells, scales):
    st = spec["style"]
    encoding = variant.removesuffix("-simple").removesuffix("-nostats").rsplit("-", 1)[1]
    hl = highlighted(spec, {k[1] for k in cells})
    marks = marks_of(variant, spec)
    roles = ("point", "highlight") + SOURCES
    set_colours = {c.lower() for c in SOURCE_COLORS.values()}
    seen = {}
    for attrs in MARK.findall(svg):
        a = dict(ATTR.findall(attrs))
        if a["data-role"] not in roles:
            continue
        key = (a["data-model"], a["data-relation"])
        if key in seen:
            die(f"{svg_path}: cell {key} drawn twice")
        seen[key] = a
        for paint in ("fill", "stroke"):
            if encoding != "sets" and a.get(paint, "").lower() in set_colours:
                die(f"{svg_path}: {key} painted in a relation-set colour {a[paint]}")
    if set(seen) != set(cells):
        die(f"{svg_path}: {len(seen)} cell marks, expected {len(cells)}")
    for (m, rel), a in seen.items():
        cell = cells[(m, rel)]
        want_role = "highlight" if rel in hl else cell["source"] if marks else "point"
        if a["data-role"] != want_role:
            die(f"{svg_path}: {m} {rel} role {a['data-role']} != {want_role}")
        want_shape = hl[rel]["shape"] if rel in hl else marks[cell["source"]] if marks else "dot"
        if a.get("data-shape") != want_shape:
            die(f"{svg_path}: {m} {rel} shape {a.get('data-shape')} != {want_shape}")
        if encoding == "neutral":
            want = st["highlight_color"] if rel in hl else st["neutral_open"] if want_shape == "ring" else st["neutral"]
        elif encoding == "sets":
            want = st["highlight_color"] if rel in hl else SOURCE_COLORS[cell["source"]]
        elif encoding == "tint":
            want = st["highlight_color"] if rel in hl else st["tint"]
        elif encoding == "model":
            want = st["model_palette"][models.index(m) % len(st["model_palette"])]
        else:
            want = ramp(st[spec["covariates"][encoding]["ramp"]], cov_t(scales[encoding], cell["cov"][encoding]))
        if a["data-c"].lower() != want.lower():
            die(f"{svg_path}: {m} {rel} colour {a['data-c']} != {want}")
        if want.lower() not in (a.get("fill", "").lower(), a.get("stroke", "").lower()):
            die(f"{svg_path}: {m} {rel} is not painted in its data-c colour")

def check_heatmap_encoded(svg, svg_path, variant, spec, models, cells, scales, label_of, summ_by_rel):
    st = spec["style"]
    fields = {"x": st["x_ramp"], "y": st["y_ramp"]}
    if variant.startswith("heatmap-recall"):
        fields["recall"] = st[spec["covariates"]["recall"]["ramp"]]

    x_lo, x_hi = min(c["x"] for c in cells.values()), max(c["x"] for c in cells.values())
    observed = variant == "heatmap-recall-observed"
    if observed and ">observed range<" not in svg:
        die(f"{svg_path}: the observed-range colour bar is not labelled")

    base = variant
    for sfx in LAYOUT_SUFFIXES:
        base = base.removesuffix(sfx)
    base = base.removesuffix("-sets")
    mode = "quantile-all" if base.endswith("-quantile-all") else "quantile" if base.endswith("-quantile") else "sqrt" if base.endswith("-sqrt") else None
    raw = {"x": lambda c: c["x"], "y": lambda c: c["y"], "recall": lambda c: c["cov"]["recall"]}
    keys = sorted(cells)
    def rank_pos(field):
        vals = [raw[field](cells[k]) for k in keys]
        rk = ranks(vals)
        return {k: (r - 1) / (len(vals) - 1) for k, r in zip(keys, rk)}
    transform = {}
    if mode == "sqrt":
        transform["x"] = {k: math.sqrt(cells[k]["x"]) for k in keys}
    elif mode == "quantile":
        transform["x"] = rank_pos("x")
    elif mode == "quantile-all":
        transform = {fd: rank_pos(fd) for fd in fields}
    for fd in transform:
        want_scale = "square-root scale" if mode == "sqrt" else "quantile scale"
        if f'data-scale="{fd}"' not in svg or f">{want_scale}<" not in svg:
            die(f"{svg_path}: the {fd} colour bar does not state its {want_scale}")
    for fd in fields:
        if fd not in transform and f'data-scale="{fd}"' in svg:
            die(f"{svg_path}: the linear {fd} block claims a non-linear scale")
    found = {}
    for x, y, fill, m, rel, field, miss in re.findall(r'<rect x="([-0-9.]+)" y="([-0-9.]+)"[^>]*? fill="(#[0-9a-fA-F]{6})" data-model="([^"]+)" data-relation="([^"]+)" data-field="(\w+)"( data-missing="1")?/>', svg):
        found[(m, rel, field)] = fill.lower()
    for (m, rel), cell in cells.items():
        for field, stops in fields.items():
            v = cell[field] if field in ("x", "y") else cov_t(scales[field], cell["cov"][field])
            if observed and field == "x":
                v = (cell["x"] - x_lo) / (x_hi - x_lo)
            if field in transform:
                v = transform[field][(m, rel)]
            if found.get((m, rel, field)) != ramp(stops, v):
                die(f"{svg_path}: {m} {rel} {field} fill {found.get((m, rel, field))} != {ramp(stops, v)}")
    if len(found) != len(fields) * len(models) * len({k[1] for k in cells}):
        die(f"{svg_path}: {len(found)} heatmap cells, expected {len(fields)} blocks")
    hl = highlighted(spec, {k[1] for k in cells})
    for rid in hl:
        label = label_of[rid].replace("&", "&amp;")
        if f">{label}<" not in svg:
            die(f"{svg_path}: highlighted row {rid} not labelled")
    check_row_marks(svg, svg_path, variant, spec, cells)

    ys = {rel: float(y) for y, rel in re.findall(r'<rect x="[-0-9.]+" y="([-0-9.]+)"[^>]*? data-relation="([^"]+)" data-field="x"', svg)}
    got = sorted(ys, key=lambda r: ys[r])
    key = "mean_x" if "-by-gain" in variant else "mean_y"
    want = sorted(got, key=lambda r: (-summ_by_rel[r][key], r))
    if got != want:
        die(f"{svg_path}: heatmap rows are not ordered by {key}; first rows {got[:3]}, expected {want[:3]}")

    n_labels = sum(svg.count(f">{m['label']}<") for m in spec["models"])
    blocks = len(fields)
    if n_labels != len(spec["models"]) * (1 if variant.endswith("-models-once") else blocks):
        die(f"{svg_path}: {n_labels} model column labels for {len(spec['models'])} models x {blocks} blocks")

def figure_labels(ids, label_of, source_of, names):
    bare = {}
    for r in ids:
        m = re.match(r"^(.*) \(([^()]+)\)$", label_of[r])
        bare[r] = m.group(1) if m and m.group(2) == names[source_of[r]] else label_of[r]
    count = {}
    for b in bare.values():
        count[b] = count.get(b, 0) + 1
    return {r: f"{bare[r]} ({names[source_of[r]]})" if count[bare[r]] > 1 else bare[r] for r in ids}

def fmt_corr(v):
    text = f"{abs(v):.2f}".lstrip("0")
    return f"\u2212{text}" if v < 0 else text

def check_fig3(svg, svg_path, variant, spec, models, cells):
    check_key(svg, svg_path, ["set", "marks"], [f"set:{s}" for s in SOURCES if any(c["source"] == s for c in cells.values())] + ["fit"])

    width = float(re.search(r'<svg [^>]*width="([0-9.]+)pt"', svg).group(1))
    printed = spec["style"]["min_font_pt"] * min(1.0, spec["style"]["paper_width_pt"] / width)
    if printed < 6.5 - 1e-9:
        die(f"{svg_path}: {width:.1f} pt wide, so its {spec['style']['min_font_pt']} pt type prints at {printed:.2f} pt (< 6.5)")
    ax = [dict(ATTR.findall(m)) for m in LINE.findall(svg)]
    ax = [d for d in ax if d.get("stroke") == spec["style"]["axis"] and d.get("stroke-width") == "0.5"]
    xs = [float(d[k]) for d in ax for k in ("x1", "x2")]
    ys = [float(d[k]) for d in ax for k in ("y1", "y2")]
    for which, mid in (("x", (min(xs) + max(xs)) / 2), ("y", (min(ys) + max(ys)) / 2)):
        m = re.search(rf'<text\s([^>]*data-axis-title="{which}"[^>]*)>', svg)
        if not m:
            die(f"{svg_path}: no {which} axis title tagged data-axis-title")
        at = float(dict(ATTR.findall(m.group(1)))[which])
        if abs(at - mid) > 0.3:
            die(f"{svg_path}: {which} axis title at {at:.1f}, the grid's centre is {mid:.1f}")
    for m in models:
        if f'data-panel-title="{m}"' not in svg:
            die(f"{svg_path}: no panel title for {m}")

    stats = [(m, re.sub(r"<[^>]+>", "", body)) for m, body in re.findall(r'<text [^>]*data-stats="([^"]+)"[^>]*>(.*?)</text>', svg)]
    for m, body in re.findall(r'<text [^>]*data-stats="([^"]+)"[^>]*>(.*?)</text>', svg):
        italic = re.findall(r'<tspan[^>]*font-style="italic"[^>]*>([^<]*)</tspan>', body)
        if italic != ["r", "\u03c1"] or "  " in body:
            die(f"{svg_path}: {m}'s r/rho line is not set as math (italic r and rho, gaps as dx): {body[:120]}")
    if variant.endswith("-nostats"):
        if stats or "\u03c1=" in svg:
            die(f"{svg_path}: the -nostats variant prints r or rho")
        return
    got = dict(stats)
    for m in models:
        own = [c for (mm, _), c in cells.items() if mm == m]
        x, y = [c["x"] for c in own], [c["y"] for c in own]
        want = f"r={fmt_corr(pearson(x, y))}\u03c1={fmt_corr(spearman(x, y))}"
        if got.get(m) != want:
            die(f"{svg_path}: {m} prints {got.get(m)!r}, recomputed {want!r}")

def check_key(svg, svg_path, blocks, items=()):
    for b in blocks:
        if f'data-key-block="{b}"' not in svg:
            die(f"{svg_path}: key lacks the {b} block")
    for i in items:
        if f'data-key-item="{i}"' not in svg:
            die(f"{svg_path}: key lacks the {i} item")

def part_relations(variant, order):
    m = re.search(r"-part([12])", variant)
    if not m:
        return list(order)
    size = math.ceil(len(order) / 2)
    k = int(m.group(1))
    return list(order)[(k - 1) * size:k * size]

def check_row_marks(svg, svg_path, variant, spec, cells, order=None):
    hl = highlighted(spec, {k[1] for k in cells})
    marks = marks_of(variant, spec)
    rows = set(part_relations(variant, order)) if order is not None else {k[1] for k in cells}
    source = {rel: c["source"] for (_, rel), c in cells.items() if rel in rows}
    st = spec["style"]
    for attrs in MARK.findall(svg):
        a = dict(ATTR.findall(attrs))
        if a.get("data-role") != "source-row":
            continue
        paint = a.get("fill") if a.get("fill") not in (None, "none") else a.get("stroke")
        src = a["data-source"]
        want = SOURCE_COLORS[src] if "-sets" in variant else st["neutral_open"] if SOURCE_MARKS[src] == "ring" else st["neutral"]
        if (paint or "").lower() != want.lower():
            die(f"{svg_path}: row glyph of {a['data-relation']} painted {paint}, expected {want}")
    rows_hl = set(re.findall(r'data-relation="([^"]+)" data-role="highlight-row"', svg))
    hl = {r: h for r, h in hl.items() if r in rows}
    rows_src = dict((r, s) for r, s in re.findall(r'data-relation="([^"]+)" data-role="source-row" data-source="([a-z]+)"', svg))
    if rows_hl != set(hl):
        die(f"{svg_path}: highlight rows {sorted(rows_hl)} differ from highlight_relations {sorted(hl)}")
    want = {} if not marks else {r: s for r, s in source.items() if r not in hl}
    if rows_src != want:
        die(f"{svg_path}: {len(rows_src)} source-glyph rows, expected {len(want)}"
            f"{'' if marks else ' (a -simple variant draws none)'}; first differences "
            f"{sorted(set(rows_src.items()) ^ set(want.items()))[:3]}")

def check_interleaved(svg, svg_path, spec, models, cells, order):
    found = {}
    for x, y, fill, m, rel, field, miss in CELL_RECT.findall(svg):
        key = (m, rel, field)
        if key in found:
            die(f"{svg_path}: two {field} strips for {m} {rel}")
        found[key] = (float(x), float(y), fill.lower(), bool(miss))
    if len(found) != 2 * len(models) * len(order):
        die(f"{svg_path}: {len(found)} cell strips, expected {2 * len(models) * len(order)}")
    ramps = {"x": spec["style"]["x_ramp"], "y": spec["style"]["y_ramp"]}
    prev_bottom = None
    for rel in order:
        tops = set()
        for m in models:
            xs, ys = found.get((m, rel, "x")), found.get((m, rel, "y"))
            if xs is None or ys is None:
                die(f"{svg_path}: {m} {rel} lacks an x or y strip")
            if abs(xs[0] - ys[0]) > 1e-9 or not ys[1] > xs[1]:
                die(f"{svg_path}: {m} {rel}: the y strip is not directly below the x strip")
            tops.add(xs[1])
            cell = cells.get((m, rel))
            for field, strip in (("x", xs), ("y", ys)):
                if cell is None:
                    if not strip[3]:
                        die(f"{svg_path}: missing cell {m} {rel} not drawn as a hole")
                elif strip[3] or strip[2] != ramp(ramps[field], cell[field]):
                    die(f"{svg_path}: {m} {rel} {field} fill {strip[2]} != {ramp(ramps[field], cell[field])}")
        if len(tops) != 1:
            die(f"{svg_path}: relation {rel} strips are not on one row")
        top = tops.pop()
        if prev_bottom is not None and not top > prev_bottom:
            die(f"{svg_path}: relation {rel} is out of order or overlaps the previous pair")
        prev_bottom = max(found[(m, rel, "y")][1] for m in models)

def close(a, b, what):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if abs(a - b) > TOL:
            die(f"{what}: {a} != {b}")
    elif a != b:
        die(f"{what}: {a!r} != {b!r}")

def ols_resid(z, v):
    mz, mv = mean(z), mean(v)
    b = sum((a - mz) * (c - mv) for a, c in zip(z, v)) / sum((a - mz) ** 2 for a in z)
    return [c - (mv + b * (a - mz)) for a, c in zip(z, v)]

HALF_VARIANTS = ("predictor-dots-partial-half", "predictor-dots-partial-stacked")
HALF_PRINT = 0.49

def check_predictors(svg, svg_path, variant, spec, models, cells, table):
    if table is None:
        die(f"{svg_path}: {variant} needs spec.predictor_ci")
    rec = lambda c: c["cov"]["recall"]
    for m in models:
        own = [c for k, c in sorted(cells.items()) if k[0] == m]
        e = table["models"].get(m) or die(f"predictor table lacks {m}")
        if sorted(e["relations"]) != sorted(k[1] for k in cells if k[0] == m):
            die(f"predictor table {m}: relations differ from the cells")
        x, y, r = [c["x"] for c in own], [c["y"] for c in own], [rec(c) for c in own]
        want = {"gain": pearson(x, y), "recall": pearson(r, y),
                "gain_given_recall": pearson(ols_resid(r, x), ols_resid(r, y)),
                "recall_given_gain": pearson(ols_resid(x, r), ols_resid(x, y))}
        for k, v in want.items():
            if abs(e[k]["r"] - v) > 1e-9:
                die(f"predictor table {m} {k}: r {e[k]['r']} != recomputed {v}")
            lo, hi = e[k]["r_ci"]
            if not lo <= e[k]["r"] <= hi:
                die(f"predictor table {m} {k}: r outside its interval")
    if variant.startswith("predictor-dots"):
        keys = ("gain_given_recall", "recall_given_gain") if variant.startswith("predictor-dots-partial") else ("gain", "recall")
        dots = re.findall(r'data-model="([^"]+)" data-pred="([a-z_]+)" data-stat="(r|rho)" data-v="([^"]+)" data-lo="([^"]+)" data-hi="([^"]+)"', svg)
        if len(dots) != len(models) * 2 * 2:
            die(f"{svg_path}: {len(dots)} coefficient dots, expected {len(models) * 4}")
        for m, k, s, v, lo, hi in dots:
            if k not in keys:
                die(f"{svg_path}: dot for {k}, expected {keys}")
            e = table["models"][m][k]
            if (float(v), [float(lo), float(hi)]) != (e[s], e[s + "_ci"]):
                die(f"{svg_path}: {m} {k} {s} dot differs from the table")
    else:
        for panel, r in re.findall(r'data-panel="([a-z_]+)" data-r="([^"]+)"', svg):
            pts = re.findall(r'<circle cx="([-0-9.]+)" cy="([-0-9.]+)"[^>]*data-panel="' + panel + '"', svg)
            if len(pts) != len(cells):
                die(f"{svg_path}: panel {panel} has {len(pts)} points, expected {len(cells)}")
            byc = {}
            for m in models:
                own = [c for k, c in sorted(cells.items()) if k[0] == m]
                x, y, rr = [c["x"] for c in own], [c["y"] for c in own], [rec(c) for c in own]
                ctr = lambda v: [a - mean(v) for a in v]
                byc.setdefault("gain", []).append((ctr(x), ctr(y)))
                byc.setdefault("recall", []).append((ctr(rr), ctr(y)))
                byc.setdefault("gain_given_recall", []).append((ols_resid(rr, x), ols_resid(rr, y)))
                byc.setdefault("recall_given_gain", []).append((ols_resid(x, rr), ols_resid(x, y)))
            xs = [a for u, _ in byc[panel] for a in u]
            ys = [b for _, w in byc[panel] for b in w]
            if abs(pearson(xs, ys) - float(r)) > 1e-9:
                die(f"{svg_path}: panel {panel} prints r {r}, recomputed {pearson(xs, ys)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--aggregate", required=True)
    ap.add_argument("--svg", action="append", default=[])
    ap.add_argument("--data")
    a = ap.parse_args()
    spec, spec_digest, base = read_spec(a.spec)
    sets = spec["relation_sets"]

    if "view_of" in spec:
        view_base = read_spec(base_dir := Path(a.spec).parent / spec["view_of"]["base"])[0]["relation_sets"]
        excluded = set(spec["view_of"]["excluded"])
        want = {s: [r for r in view_base[s] if r not in excluded] for s in SOURCES}
        if {s: sets[s] for s in SOURCES} != want:
            die(f"view spec relations are not {base_dir.name}'s minus {sorted(excluded)}")
    relations, source = [], {}
    for s in SOURCES:
        for r in sets[s]:
            if r in source:
                die(f"relation {r} is in both {source[r]} and {s}")
            source[r] = s
            relations.append(r)
    source_aliases = spec.get("snapshot_source_aliases", {})
    for alias, targets in source_aliases.items():
        if alias in SOURCES or not targets or any(t not in SOURCES for t in targets):
            die(f"bad snapshot_source_aliases entry {alias}: {targets}")
    colors = SOURCE_COLORS
    if "source_colors" in spec["style"] or "source_marks" in spec["style"]:
        die("style.source_colors / source_marks are retired: set colours and marks come from relation-sets.json5 source_display")

    lint = check_terms.check(Path(__file__).resolve().parent, check_terms.matcher(check_terms.names(check_terms.terms_mod.load())))
    if lint:
        die("hard-coded terms: " + "; ".join(lint[:5]))
    main_labels, main_digest = pinned(base, spec["relation_labels"]["hernandez_engineered"], "relation_labels.hernandez_engineered")
    bear_labels, bear_digest = pinned(base, spec["relation_labels"]["bear"], "relation_labels.bear")
    _, probe_digest = pinned(base, spec["probe_names"], "probe names")
    if "relation_label_aliases" in spec:
        die("relation_label_aliases is retired: label keys are registry ids; older spellings go in the registry's aliases")
    label_of = {r: main_labels["relations"].get(r) for r in sets["hernandez"] + sets["engineered"]}
    label_of |= {r: bear_labels["relations"].get(r) for r in sets["bear"]}
    unlabeled = [r for r, label in label_of.items() if not label]
    if unlabeled:
        die(f"relations without a label: {unlabeled}")
    by_label = {}
    for r, label in label_of.items():
        by_label.setdefault(label, []).append(r)
    shared = {label: rs for label, rs in by_label.items() if len(rs) > 1}
    if shared:
        die(f"relations share a display label: {shared}")
    data_path = Path(a.data) if a.data else base / spec["snapshot"]
    raw = data_path.read_bytes()
    data_digest = hashlib.sha256(raw).hexdigest()
    if not a.data and data_digest != spec["snapshot_sha256"]:
        die("snapshot digest does not match the spec pin")
    rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    synthetic = any(r.get("synthetic") is True for r in rows)
    X, Y = spec["metrics"]["x"]["field"], spec["metrics"]["y"]["field"]
    models = [m["id"] for m in spec["models"]]
    missing = {(e["model"], e["relation"]) for e in spec.get("expected_missing", [])}

    groups = {}
    if not isinstance(spec.get("repeat_id"), int):
        die("spec.repeat_id missing")
    bad = [r for r in rows if r.get("repeat") != spec["repeat_id"]]
    if bad:
        die(f"snapshot holds {len(bad)} row(s) that are not repeat {spec['repeat_id']}")
    for r in rows:
        if r["model"] not in models or r["relation"] not in source:
            die(f"row outside cohort: {r['model']} {r['relation']}")

        if r.get("source") != source[r["relation"]] and source[r["relation"]] not in source_aliases.get(r.get("source"), ()):
            die(f"row {r['model']} {r['relation']} source {r.get('source')!r} is neither {source[r['relation']]} nor a declared alias of it")
        groups.setdefault((r["model"], r["relation"]), []).append(r)
    cells = {}
    for m in models:
        for rel in relations:
            g = groups.get((m, rel))
            if g is None:
                if (m, rel) not in missing:
                    die(f"undeclared missing cell {m} {rel}")
                continue

            if len(g) != 1 or g[0]["repeat"] != spec["repeat_id"]:
                die(f"cell {m} {rel} must be exactly one repeat-{spec['repeat_id']} row, found repeats {[r['repeat'] for r in g]}")
            cells[(m, rel)] = {"x": g[0][X], "y": g[0][Y], "source": source[rel],
                               "cov": {name: g[0][c["field"]] for name, c in (spec.get("covariates") or {}).items()}}

    agg = json.loads(Path(a.aggregate).read_text())
    close(agg["repeat_id"], spec["repeat_id"], "aggregate repeat_id")
    close(agg["data_sha256"], data_digest, "aggregate data digest")
    close(agg["spec_sha256"], spec_digest, "aggregate spec digest")
    close(agg["hernandez_engineered_labels_sha256"], main_digest, "relation_labels.hernandez_engineered digest")
    close(agg["bear_labels_sha256"], bear_digest, "relation_labels.bear digest")
    close(agg["probe_names_sha256"], probe_digest, "probe-name digest")
    if len(agg["cells"]) != len(cells):
        die("cell count differs")
    for c in agg["cells"]:
        mine = cells[(c["model"], c["relation"])]
        for k in ("x", "y", "source"):
            close(c[k], mine[k], f"cell {c['model']} {c['relation']} {k}")
        for name, v in mine["cov"].items():
            close(c["cov"][name], v, f"cell {c['model']} {c['relation']} covariate {name}")

    scales = {}
    for name, c in (spec.get("covariates") or {}).items():
        vals = [cell["cov"][name] for cell in cells.values()]
        scales[name] = ({"scale": "log", "lo": min(vals), "hi": max(vals)} if c["scale"] == "log"
                        else {"scale": "linear", "lo": c["domain"][0], "hi": c["domain"][1]})
        for k in ("scale", "lo", "hi"):
            close(agg["covariate_scales"][name][k], scales[name][k], f"covariate scale {name} {k}")

    for pm in agg["per_model"]:
        own = [(rel, cells[(pm["model"], rel)]) for rel in relations if (pm["model"], rel) in cells]
        x = [c["x"] for _, c in own]
        y = [c["y"] for _, c in own]
        close(pm["n"], len(own), f"{pm['model']} n")
        close(pm["pearson"], pearson(x, y), f"{pm['model']} pearson")
        close(pm["spearman"], spearman(x, y), f"{pm['model']} spearman")
        slope, intercept = ols(x, y)
        close(pm["slope"], slope, f"{pm['model']} slope")
        close(pm["intercept"], intercept, f"{pm['model']} intercept")
        for i, p in enumerate((0.25, 0.5, 0.75)):
            close(pm["x_quartiles"][i], q7(x, p), f"{pm['model']} x q{p}")
            close(pm["y_quartiles"][i], q7(y, p), f"{pm['model']} y q{p}")

        want = {}
        for s in SOURCES:
            part = [c for _, c in own if c["source"] == s]
            if len(part) >= 3:
                px, py = [c["x"] for c in part], [c["y"] for c in part]
                sl, ic = ols(px, py)
                want[s] = {"n": len(part), "pearson": pearson(px, py), "slope": sl, "intercept": ic}
        close(sorted(pm["by_source"]), sorted(want), f"{pm['model']} by_source sets")
        for s, w in want.items():
            for k, v in w.items():
                close(pm["by_source"][s][k], v, f"{pm['model']} by_source {s} {k}")

    complete = [m for m in models if all((m, rel) in cells for rel in relations)]
    close(agg["complete_models"], complete, "complete models")
    summ = []
    for rel in relations:
        xs = [cells[(m, rel)]["x"] for m in complete]
        ys = [cells[(m, rel)]["y"] for m in complete]
        summ.append({"relation": rel, "mean_y": mean(ys), "mean_x": mean(xs), "x_median": q7(xs, 0.5), "y_median": q7(ys, 0.5),
                     "x_min": min(xs), "x_max": max(xs), "y_min": min(ys), "y_max": max(ys)})
        rs = [cells[(m, rel)]["cov"].get("recall") for m in complete]
        if all(isinstance(v, (int, float)) for v in rs):
            summ[-1].update({"mean_recall": mean(rs), "recall_min": min(rs), "recall_max": max(rs), "recall_median": q7(rs, 0.5)})
    summ.sort(key=lambda s: (-s["mean_y"], s["relation"]))
    for mine, theirs in zip(summ, agg["per_relation"]):
        for k in mine:
            close(theirs[k], mine[k], f"relation summary {mine['relation']} {k}")

    for field, key in ((X, "x"), (Y, "y")):
        for i, ma in enumerate(models):
            for j, mb in enumerate(models):
                shared = [rel for rel in relations if (ma, rel) in cells and (mb, rel) in cells]
                v = spearman([cells[(ma, rel)][key] for rel in shared], [cells[(mb, rel)][key] for rel in shared])
                close(agg["rank_agreement"][key][i][j], v, f"agreement {key} {ma} {mb}")

    xc, yc = [], []
    for m in models:
        own = [cells[(m, rel)] for rel in relations if (m, rel) in cells]
        mx, my = mean([c["x"] for c in own]), mean([c["y"] for c in own])
        xc += [c["x"] - mx for c in own]
        yc += [c["y"] - my for c in own]
    close(agg["pooled_within"]["pearson"], pearson(xc, yc), "pooled_within.pearson")

    for svg_path in a.svg:
        svg = Path(svg_path).read_text()
        for problem in layer_order.check(svg, svg_path):
            die(problem)

        for problem in check_terms.terms_mod.load().check_svg(svg, str(svg_path)):
            die(problem)

        for problem in key_spacing.check(svg, str(svg_path)):
            die(problem)
        v = re.search(r'data-variant="([^"]+)"', svg).group(1)
        if v.startswith("heatmap") and "-sets" in v and not key_spacing.count(svg):
            die(f"{svg_path}: the set key under the colour bars is not placed with the shared spacing")
        meta = json.loads(re.search(r"<metadata>(.*?)</metadata>", svg).group(1).replace("&quot;", '"').replace("&amp;", "&"))
        close(meta["data_sha256"], data_digest, f"{svg_path} data digest")
        close(meta["probe_names_sha256"], probe_digest, f"{svg_path} probe-name digest")
        has_mark = "DRAFT-SYNTHETIC" in svg
        if synthetic != has_mark:
            die(f"{svg_path}: synthetic={synthetic} but watermark present={has_mark}")

        if not synthetic and "DRAFT" in svg:
            die(f"{svg_path}: real data but the SVG contains DRAFT")
        if not synthetic and "DRAFT" in str(spec.get("status", "")):
            die(f"{svg_path}: real data but spec status {spec.get('status')!r} is a draft status")
        sizes = [float(s) for s in re.findall(r'font-size="([0-9.]+)"', svg)]
        if "range over repeats" in svg:
            die(f"{svg_path}: draws repeat whiskers, which the repeat-0 rule forbids")
        if min(sizes) < spec["style"]["min_font_pt"] - 1e-9:
            die(f"{svg_path}: text below {spec['style']['min_font_pt']} pt")
        if re.search(r">[^<]*\bstudy\b", svg, re.IGNORECASE):
            die(f"{svg_path}: prints the retired set name 'study'")

        for s_name in SOURCE_NAMES.values():
            if re.search(rf">{re.escape(s_name)}( relations?)? \(\d+\)<", svg):
                die(f"{svg_path}: keys {s_name} with a count; counts belong in the caption")
        variant = re.search(r'data-variant="([^"]+)"', svg).group(1)
        if variant in ENCODED_SCATTERS:
            check_encoded(svg, svg_path, variant, spec, models, cells, scales)
        if variant in FIG3_VARIANTS:
            check_fig3(svg, svg_path, variant, spec, models, cells)
        if variant in HEATMAPS_ENC:
            check_heatmap_encoded(svg, svg_path, variant, spec, models, cells, scales, label_of, {e["relation"]: e for e in summ})
        if variant == "model-spread-highlight":
            hl_ids = set(highlighted(spec, {k[1] for k in cells}))
            n_hl = sum(1 for k in cells if k[1] in hl_ids)
            if len(re.findall(r'data-role="highlight" data-field="[xy]"', svg)) != 2 * n_hl:
                die(f"{svg_path}: expected {2 * n_hl} highlight marks")
        if variant in SET_FREE_VARIANTS:
            for s_name, c in colors.items():
                if c.lower() in svg.lower():
                    die(f"{svg_path}: uses the {s_name} set colour {c}")
        if variant in STRIPS:
            check_row_marks(svg, svg_path, variant, spec, cells, [e["relation"] for e in summ])
            has_sets = not variant.endswith("-simple")
            check_key(svg, svg_path, ["measure", "over_models"] + (["set"] if has_sets else []),
                      ["median", "mean", "range"] + [f"set:{src}" for src in SOURCES if has_sets and sets[src]])
            if "dot = median" in svg:
                die(f"{svg_path}: the retired one-line mark note is back; the key explains the marks")
            source_of = {r: src for src in SOURCES for r in sets[src]}
            fig = figure_labels([r for src in SOURCES for r in sets[src]], label_of, source_of, SOURCE_NAMES)
            for r in part_relations(variant, [e["relation"] for e in summ]):
                if f">{fig[r].replace('&', '&amp;')}<" not in svg:
                    die(f"{svg_path}: relation {r} is not printed as {fig[r]!r}")
        if variant.startswith("heatmap-recall") and "-sets" in variant:
            check_key(svg, svg_path, ["set"], [f"set:{src}" for src in SOURCES if sets[src]])
        if variant in SET_COLOUR_VARIANTS and not any(c.lower() in svg.lower() for c in colors.values()):
            die(f"{svg_path}: a -sets variant that never uses a set colour")

        if variant in STRIPS + ("model-spread", "model-spread-highlight"):
            key = "relation" if variant in STRIPS else "model"
            marks = re.findall(r'data-stat="mean" data-v="([^"]+)" data-' + key + r'="([^"]+)" data-field="(x|y|recall)"', svg)
            if variant in STRIPS:
                fds = ("x", "y", "recall") if variant.startswith("appendix-strips-recall") else ("x", "y")
                part = set(part_relations(variant, [e["relation"] for e in summ]))
                want = {(e["relation"], fd): e["mean_" + fd] for e in summ for fd in fds if e["relation"] in part}
            else:
                want = {(m, fd): mean([c[fd] for k, c in cells.items() if k[0] == m]) for m in models for fd in "xy"
                        if any(k[0] == m for k in cells)}
            got = {(who, fd): float(v) for v, who, fd in marks}
            if len(marks) != len(want) or set(got) != set(want):
                die(f"{svg_path}: {len(marks)} mean markers for {len(want)} medians")
            for k2, v in want.items():
                if abs(got[k2] - v) > 1e-9:
                    die(f"{svg_path}: mean marker {k2} is {got[k2]}, recomputed {v}")
        if variant.startswith("predictor-"):
            table = pinned(base, spec["predictor_ci"], "predictor CI table")[0] if spec.get("predictor_ci") else None
            check_predictors(svg, svg_path, variant, spec, models, cells, table)
            if variant.startswith("predictor-dots"):
                check_key(svg, svg_path, ["predictor", "marks"], ["coefficient", "interval"])
            if variant in HALF_VARIANTS:
                width = float(re.search(r'<svg [^>]*width="([0-9.]+)pt"', svg).group(1))
                printed = spec["style"]["min_font_pt"] * min(1.0, HALF_PRINT * spec["style"]["paper_width_pt"] / width)
                if printed < 6.5 - 1e-9:
                    die(f"{svg_path}: {width:.1f} pt wide at half the text width, so its type prints at {printed:.2f} pt (< 6.5)")
        if variant == "heatmap-interleaved":
            check_interleaved(svg, svg_path, spec, models, cells, [s["relation"] for s in summ])
        if variant in KEYED_VARIANTS:
            for s in SOURCES:
                n = len(sets[s])
                if n == 0:
                    continue
                if not re.search(rf">{SOURCE_NAMES[s]}( relations?)?<", svg):
                    die(f"{svg_path}: key lacks {SOURCE_NAMES[s]}")
                if f'fill="{colors[s]}"' not in svg:
                    die(f"{svg_path}: never uses the {s} colour {colors[s]}")

    print(json.dumps({"status": "ok", "synthetic": synthetic, "cells": len(cells), "models": len(models), "relations": len(relations),
                      "complete_models": len(complete), "svgs": len(a.svg), "data_sha256": data_digest}))

if __name__ == "__main__":
    main()
