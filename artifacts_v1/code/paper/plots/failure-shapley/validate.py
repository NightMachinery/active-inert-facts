#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
import check_terms
from half_width import half_width
import key_spacing
import terms as terms_mod

ap = argparse.ArgumentParser()
ap.add_argument("--spec", required=True)
ap.add_argument("--probe-names", required=True)
ap.add_argument("--aggregate", required=True)
ap.add_argument("--svg", action="append", default=[])
ap.add_argument("--rounds", help="the review rounds directory; every render its rN/round.json files list is checked")
a = ap.parse_args()
errors = []

def check(ok, msg):
    if not ok:
        errors.append(msg)

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec = json.load(open(a.spec))
check(sha(spec["snapshot"]) == spec["snapshot_sha256"], "snapshot sha256 differs from the spec pin")
check(sha(a.probe_names) == spec["probe_names"]["sha256"], "probe-name registry sha256 differs from the spec pin")
snap = json.load(open(spec["snapshot"]))
check(snap["source_sha256"] == spec["source"]["sha256"], "snapshot was frozen from another source")
T = terms_mod.load(a.probe_names)
for problem in check_terms.check(Path(__file__).resolve().parent, check_terms.matcher(check_terms.names(T))):
    check(False, problem)

keys = [it["key"] for fm in spec["families"] for it in fm["items"]]
check(keys + ["unexplained"] == snap["order"], "segment order differs from the snapshot")
P = snap["pooled"]

def expl(D, field, ci_field):
    c = D[ci_field]["unexplained"] if ci_field else None
    return {"v": 1 - D[field]["unexplained"], "ci": [1 - c[1], 1 - c[0]] if c else None}

want = {"n_err": P["n_err"], "n_cells": P["n_cells"], "bars": {}, "families": {}, "per_model": {}, "over_models": {}}
for b in spec["bars"]:
    total = sum(P[b["field"]][k] for k in keys) + P[b["field"]]["unexplained"]
    check(abs(total - 1) < 1e-9, f"bar {b['id']}: shares sum to {total}")
    want["bars"][b["id"]] = {"explained": expl(P, b["field"], b["ci"]), "segments": {k: P[b["field"]][k] for k in keys + ["unexplained"]}}
for fm in spec["families"]:
    want["families"][fm["id"]] = {b["id"]: sum(P[b["field"]][it["key"]] for it in fm["items"]) for b in spec["bars"]}
check(sorted(m["id"] for m in spec["models"]) == sorted(snap["by_model"]), "spec models differ from the snapshot's")
for m in spec["models"]:
    D = snap["by_model"][m["id"]]
    want["per_model"][m["id"]] = {"observed": expl(D, "share", "ci"), "shuffle": expl(D, "shuffle", "shuffle_ci"),
                                  "n_err": D["n_err"], "n_cells": D["n_cells"]}
for kind in ("observed", "shuffle"):
    v = [want["per_model"][m["id"]][kind]["v"] for m in spec["models"]]
    want["over_models"][kind] = {"median": statistics.median(v), "mean": statistics.fmean(v), "min": min(v), "max": max(v)}

got = json.load(open(a.aggregate))

def same(x, y, where):
    if isinstance(x, dict):
        check(isinstance(y, dict) and set(x) == set(y), f"aggregate {where}: keys differ")
        for k in x:
            if isinstance(y, dict) and k in y:
                same(x[k], y[k], f"{where}.{k}")
    elif isinstance(x, list):
        check(isinstance(y, list) and len(x) == len(y), f"aggregate {where}: lengths differ")
        for i, (p, q) in enumerate(zip(x, y or [])):
            same(p, q, f"{where}[{i}]")
    elif isinstance(x, float) or isinstance(y, float):
        check(y is not None and abs(x - y) <= 1e-12, f"aggregate {where}: {y} != {x}")
    else:
        check(x == y, f"aggregate {where}: {y} != {x}")

same(want, {k: got.get(k) for k in want}, "")
check(got.get("snapshot_sha256") == spec["snapshot_sha256"], "aggregate snapshot pin differs")

pct = lambda v: f"{100 * v:.1f}%"
pm = lambda v, c: f"± {half_width(v, c, 1, 100)}"
TOL = 2e-3
NS = "{http://www.w3.org/2000/svg}"

def text_of(el):
    return "".join(el.itertext())

def near(x, y):
    return abs(float(x) - y) <= TOL

def fam_title(fm):
    return re.sub(r"\{(gain|recall)\}", lambda m: T.term(m.group(1)), fm["title"]).replace("{T}", "T")

VARIANT_FILES = {spec["output_prefix"] + (v["file"] and "-" + v["file"]): v.get("layout", "full") for v in spec["variants"].values()}
VARIANT_DEFS = {spec["output_prefix"] + (v["file"] and "-" + v["file"]): v for v in spec["variants"].values()}
SCALES = {"full": {"bars", "explanations", "models"}, "appendix": {"explanations", "models"}, "paired": {"paired"}, "stacked": {"stacked"},
          "paired-shaded": {"paired-shaded"}, "paired-subrows": {"paired-subrows"}, "stacked-ranks": {"stacked-ranks"}}
TAGS = {"full": ["a", "b", "c"], "appendix": ["a", "b"]}
MAIN = ("paired", "stacked", "paired-shaded", "paired-subrows", "stacked-ranks")

def tint_of(c):
    t = spec["main"]["pale_tint"]
    return "#" + "".join(f"{round(255 - t * (255 - v)):02X}" for v in hexrgb(c))

def base_fill(color, col):
    return color if col == 0 or BASELINES == "same" else tint_of(color)

def layout_of(path):
    name = Path(path).name.removesuffix(".svg")
    check(name in VARIANT_FILES, f"{path}: not a variant of the spec")
    return VARIANT_FILES.get(name, "full")

def fam_values(kind):
    out = {fm["id"]: want["families"][fm["id"]][kind] for fm in spec["families"]}
    out["total"] = want["bars"][kind]["explained"]["v"]
    return out

def hexrgb(c):
    return [int(c[i:i + 2], 16) for i in (1, 3, 5)]

def check_paired(path, g, texts):
    hue = {fm["id"]: fm["hue"] for fm in spec["families"]} | {"total": spec["main"]["total_hue"]}
    d1 = spec["main"]["paired_domain"][1]
    check(float(g.get("data-d1")) == d1, f"{path}: paired domain")
    labels = {t.get("data-row-label"): text_of(t) for t in texts if t.get("data-row-label")}
    check(labels == {fm["id"]: fam_title(fm) for fm in spec["families"]} | {"total": "Total explained"}, f"{path}: row labels {labels}")
    cols = [e for e in g if e.get("data-col")]
    check([c.get("data-col") for c in cols] == [b["id"] for b in spec["bars"]], f"{path}: paired columns")
    for b, col in zip(spec["bars"], cols):
        x0, x1 = float(col.get("data-x0")), float(col.get("data-x1"))
        check([text_of(t) for t in col.iter() if t.get("data-col-label")] == [b["label"]], f"{path}: column label {b['id']}")
        vals = fam_values(b["id"])
        for rid, v in vals.items():
            r = [e for e in col.iter() if e.get("data-row") == rid and e.get("data-kind") == b["id"]]
            ok = len(r) == 1 and float(r[0].get("data-v")) == v
            if ok:
                inset = float(r[0].get("x")) - x0
                ok = near(float(r[0].get("x")) + float(r[0].get("width")) + inset, x0 + v / d1 * (x1 - x0))
                if b["id"] == spec["bars"][0]["id"]:
                    ok = ok and r[0].get("fill") == hue[rid]
                elif b["ci"] is None:
                    ok = ok and r[0].get("fill") == "#FFFFFF" and r[0].get("stroke") == hue[rid]
                else:
                    a = spec["main"]["shuffle_tint"]
                    tint = "#" + "".join(f"{round(255 - a * (255 - c)):02X}" for c in hexrgb(hue[rid]))
                    ok = ok and r[0].get("fill") == tint
            check(ok, f"{path}: paired bar {rid} {b['id']}")
            t = [text_of(e) for e in col.iter() if e.get("data-value") == rid and e.get("data-kind") == b["id"]]
            check(t == [pct(v)], f"{path}: paired value {rid} {b['id']} prints {t}")

def check_stacked(path, g, texts):
    x0, x1 = float(g.get("data-x0")), float(g.get("data-x1"))
    X = lambda v: x0 + v * (x1 - x0)
    order = [fm["id"] for fm in spec["families"]] + ["unexplained"]
    hue = {fm["id"]: fm["hue"] for fm in spec["families"]} | {"unexplained": spec["style"]["unexplained"]}
    for b in spec["bars"]:
        vals = {k: v for k, v in fam_values(b["id"]).items() if k != "total"} | {"unexplained": want["bars"][b["id"]]["segments"]["unexplained"]}
        check(abs(sum(vals.values()) - 1) < 1e-9, f"{path}: stacked bar {b['id']} does not sum to 1")
        rects = [e for e in g.iter() if e.tag.endswith("rect") and e.get("data-bar") == b["id"] and e.get("data-key")]
        check([r.get("data-key") for r in rects] == order, f"{path}: stacked bar {b['id']} segment order")
        acc = 0.0
        for r in rects:
            k = r.get("data-key")
            v = vals.get(k, float("nan"))
            check(float(r.get("data-v")) == v and r.get("fill") == hue.get(k)
                  and near(r.get("x"), X(acc)) and near(float(r.get("x")) + float(r.get("width")), X(acc + v)),
                  f"{path}: stacked bar {b['id']} {k} segment")
            acc += v
            t = [text_of(e) for e in g.iter() if e.tag.endswith("text") and e.get("data-bar") == b["id"] and e.get("data-value") == k]
            check(t == [("unexplained " if k == "unexplained" else "") + pct(v)], f"{path}: stacked {b['id']} {k} prints {t}")
            outside = [e for e in g.iter() if e.tag.endswith("text") and e.get("data-bar") == b["id"] and e.get("data-value") == k and e.get("data-place") == "outside"]
            leaders = [e for e in g.iter() if e.get("data-leader") == k and e.get("data-bar") == b["id"]]
            check(len(leaders) == len(outside), f"{path}: stacked {b['id']} {k} leader")
        check([text_of(t) for t in texts if t.get("data-bar-label") == b["id"]] == [b["label"]], f"{path}: bar {b['id']} label")
    items = [e.get("data-key-item") for e in g.iter() if e.get("data-key-item")] or \
        [e.get("data-key-item") for e in ET.parse(path).getroot().iter() if e.get("data-key-item")]
    check(items == [fm["id"] for fm in spec["families"]], f"{path}: family key {items}")
    sw = {e.get("data-swatch"): e.get("fill") for e in ET.parse(path).getroot().iter() if e.get("data-swatch")}
    check(sw == {fm["id"]: fm["hue"] for fm in spec["families"]}, f"{path}: key swatch colors")

def x_range(e):
    return float(e.get("x")), float(e.get("x")) + float(e.get("width"))

SHADE_ITEMS = ["ranks"] + [it["key"] for fm in spec["families"] if fm["id"] == "ftset" for it in fm["items"]]

def check_table(path, g, texts, layout):
    total = {"id": "total", "label": "Total explained", "hue": spec["main"]["total_hue"], "items": None,
             "v": lambda b: want["bars"][b["id"]]["explained"]["v"], "text_only": layout == "paired-shaded"}
    rows = []
    for fm in spec["families"]:
        split = [(it["key"], it["color"]) for it in fm["items"]] if len(fm["items"]) > 1 else None
        rows.append({"id": fm["id"], "label": fam_title(fm), "hue": fm["hue"], "items": split,
                     "v": lambda b, fm=fm: want["families"][fm["id"]][b["id"]]})
        if layout == "paired-subrows" and split:
            for it in fm["items"]:
                rows.append({"id": it["key"], "label": it["label"], "hue": it["color"], "items": None,
                             "v": lambda b, k=it["key"]: P[b["field"]][k]})
    rows.append(total)
    labels = [(t.get("data-row-label"), text_of(t)) for t in texts if t.get("data-row-label")]
    check(labels == [(r["id"], r["label"]) for r in rows], f"{path}: row labels {labels}")
    k = float(g.get("data-k"))
    cols = [e for e in g if e.get("data-col")]
    check([c.get("data-col") for c in cols] == [b["id"] for b in spec["bars"]], f"{path}: columns")
    for ci, (b, col) in enumerate(zip(spec["bars"], cols)):
        x0 = float(col.get("data-x0"))
        check([text_of(t) for t in col.iter() if t.get("data-col-label")] == [b.get("header", b["label"]) if VARIANT.get("headers") == "long" else b["label"]], f"{path}: column label {b['id']}")
        for t in (e for e in col.iter() if e.get("data-tick")):
            check(near(t.get("x"), x0 + float(t.get("data-tick")) * k), f"{path}: {b['id']} tick {t.get('data-tick')} off the scale")
        for r in rows:
            v = r["v"](b)
            t = [text_of(e) for e in col.iter() if e.get("data-value") == r["id"] and e.get("data-kind") == b["id"]]
            check(t == [pct(v)], f"{path}: {r['id']} {b['id']} prints {t}")
            rects = [e for e in col.iter() if e.get("data-row") == r["id"] and e.get("data-kind") == b["id"]]
            if r.get("text_only"):
                check(not rects, f"{path}: {r['id']} {b['id']} draws a bar")
                continue
            pieces = [(key, P[b["field"]][key], c) for key, c in r["items"]] if r["items"] else [(r["id"], v, r["hue"])]
            check([e.get("data-seg") for e in rects] == [p[0] for p in pieces], f"{path}: {r['id']} {b['id']} pieces")
            acc = 0.0
            for e, (key, pv, color) in zip(rects, pieces):
                xa, xb = x_range(e)
                check(float(e.get("data-v")) == pv and near(xa, x0 + acc * k) and near(xb, x0 + (acc + pv) * k)
                      and e.get("fill") == base_fill(color, ci), f"{path}: {r['id']} {b['id']} piece {key}")
                for ins in (x for x in col.iter() if x.get("data-inside") == key):
                    check(ci == 0 and text_of(ins) == pct(pv) and xa < float(ins.get("x")) < xb, f"{path}: inside value {key}")
                acc += pv
            check(abs(acc - v) < 1e-12, f"{path}: {r['id']} {b['id']} pieces do not sum to the row")
            outl = [e for e in col.iter() if e.get("data-outline") == r["id"]]
            want_outline = BASELINES == "pale" and b["ci"] is None and ci > 0
            check(len(outl) == want_outline and all(e.get("stroke") == r["hue"] for e in outl), f"{path}: {r['id']} {b['id']} outline")
    items = [e.get("data-key-item") for e in ET.parse(path).getroot().iter() if e.get("data-key-item")]
    check(items == (SHADE_ITEMS if layout == "paired-shaded" else []), f"{path}: key items {items}")

def check_stacked_ranks(path, g, texts):
    x0, x1 = float(g.get("data-x0")), float(g.get("data-x1"))
    X = lambda v: x0 + v * (x1 - x0)
    color = {it["key"]: it["color"] for fm in spec["families"] for it in fm["items"]} | {"unexplained": spec["style"]["unexplained"]}
    hue = {fm["id"]: fm["hue"] for fm in spec["families"]}
    for bi, b in enumerate(spec["bars"]):
        seg = want["bars"][b["id"]]["segments"]
        rects = [e for e in g.iter() if e.tag.endswith("rect") and e.get("data-bar") == b["id"] and e.get("data-key")]
        check([r.get("data-key") for r in rects] == keys + ["unexplained"], f"{path}: {b['id']} segment order")
        acc = 0.0
        for r in rects:
            key = r.get("data-key")
            v = seg.get(key, float("nan"))
            xa, xb = x_range(r)
            expect = color.get(key) if key not in color or key == "unexplained" else base_fill(color[key], bi)
            check(float(r.get("data-v")) == v and r.get("fill") == expect and near(xa, X(acc)) and near(xb, X(acc + v)),
                  f"{path}: {b['id']} {key} segment")
            for ins in (e for e in g.iter() if e.get("data-inside") == key and e.get("data-bar") == b["id"]):
                check(text_of(ins) == pct(v) and xa < float(ins.get("x")) < xb, f"{path}: {b['id']} inside value {key}")
            acc += v
        u = [text_of(e) for e in g.iter() if e.get("data-value") == "unexplained" and e.get("data-bar") == b["id"]]
        check(u == [f"unexplained {pct(seg['unexplained'])}"], f"{path}: {b['id']} unexplained prints {u}")
        for fm in spec["families"]:
            t = [text_of(e) for e in g.iter() if e.get("data-family-total") == fm["id"] and e.get("data-bar") == b["id"]]
            check(t == [pct(want["families"][fm["id"]][b["id"]])], f"{path}: {b['id']} {fm['id']} total prints {t}")
            br = [e for e in g.iter() if e.get("data-bracket") == fm["id"] and e.get("data-bar") == b["id"]]
            check(len(br) == 1, f"{path}: {b['id']} {fm['id']} bracket")
            outl = [e for e in g.iter() if e.get("data-outline") == fm["id"] and e.get("data-bar") == b["id"]]
            want_outline = BASELINES == "pale" and b["ci"] is None and bi > 0
            check(len(outl) == want_outline and all(e.get("stroke") == hue[fm["id"]] for e in outl), f"{path}: {b['id']} {fm['id']} outline")
        check([text_of(t) for t in texts if t.get("data-bar-label") == b["id"]] == [b["label"]], f"{path}: bar {b['id']} label")
    root = ET.parse(path).getroot()
    items = [e.get("data-key-item") for e in root.iter() if e.get("data-key-item")]
    check(items == [fm["id"] for fm in spec["families"]] + SHADE_ITEMS, f"{path}: key items {items}")
    for fm in spec["families"]:
        sw = [e.get("fill") for e in root.iter() if e.get(f"data-swatch-{fm['id']}") is not None]
        check(sw == [it["color"] for it in fm["items"]], f"{path}: {fm['id']} key shades")

CHECKS = {"paired": check_paired, "stacked": check_stacked, "stacked-ranks": check_stacked_ranks,
          "paired-shaded": lambda p, g, t: check_table(p, g, t, "paired-shaded"),
          "paired-subrows": lambda p, g, t: check_table(p, g, t, "paired-subrows")}

svgs = [(p, "same") for p in a.svg]
if a.rounds:
    R = Path(a.rounds)
    rounds = sorted((d for d in R.iterdir() if re.fullmatch(r"r\d+", d.name) and not d.is_symlink()), key=lambda d: int(d.name[1:]))
    check(bool(rounds), "rounds: no rN directories")
    expect_files = set()
    for rd in rounds:
        src = json.load(open(rd / "round.json"))
        parts = [(rd / sub, p) for sub, p in src["subdirs"].items()] if "subdirs" in src else [(rd, src)]
        for d, p in parts:
            for vid in p["targets"]:
                check(vid in spec["variants"], f"{rd}/round.json: unknown target {vid}")
                v = spec["variants"].get(vid, {"file": vid})
                stem = d / (spec["output_prefix"] + (v["file"] and "-" + v["file"]))
                expect_files |= {stem.with_suffix(ext) for ext in (".svg", ".pdf", ".png")}
                svgs.append((str(stem.with_suffix(".svg")), p.get("baselines", "same")))
    found = {p for p in R.rglob("*") if p.is_file() and p.suffix in (".svg", ".pdf", ".png") and "current" not in p.relative_to(R).parts}
    check(found == expect_files, f"rounds: missing {sorted(map(str, expect_files - found))}, extra {sorted(map(str, found - expect_files))}")
    cur = R / "current"
    newest = rounds[-1].name if rounds else None
    check(cur.is_symlink() and str(cur.readlink()) == newest, f"rounds: current does not link to {newest}")

for path, BASELINES in svgs:
    if not Path(path).exists():
        continue
    root = ET.parse(path).getroot()
    tag = lambda el: el.tag.replace(NS, "")
    everything = list(root.iter())
    meta = "".join(text_of(e) for e in everything if tag(e) == "metadata")
    check(("baselines=" + BASELINES in meta) if BASELINES != "same" else "baselines=" not in meta, f"{path}: baseline scheme is not {BASELINES}")
    for pin in (spec["snapshot_sha256"], spec["source"]["sha256"], spec["probe_names"]["sha256"]):
        check(pin in meta, f"{path}: metadata lacks {pin[:12]}")
    texts = [e for e in everything if tag(e) == "text"]
    alltext = " ".join(text_of(e) for e in texts)
    check("sandbox" not in alltext.lower(), f"{path}: prints 'sandbox'")
    check(not re.search("[–—]", alltext), f"{path}: prints an en or em dash")
    check(not re.search(r"\bn\s*=|\bdf\b", alltext), f"{path}: prints a count or df")
    check(not re.search(r"\[\s*-?\d", alltext), f"{path}: prints a bracket interval, not ± h")
    for e in texts:
        check(float(e.get("font-size")) >= spec["min_font_pt"], f"{path}: text below {spec['min_font_pt']} pt: {text_of(e)!r}")
    scales = {e.get("data-scale"): e for e in everything if e.get("data-scale")}
    layout = layout_of(path)
    VARIANT = VARIANT_DEFS.get(Path(path).name.removesuffix(".svg"), {})
    for t in texts:
        w = key_spacing._width(text_of(t), float(t.get("font-size")), t.get("font-weight") == "700")
        x = float(t.get("x")) - {"middle": w / 2, "end": w}.get(t.get("text-anchor"), 0)
        check(x >= -0.01 and x + w <= spec["paper_width_pt"] + 0.01, f"{path}: {text_of(t)!r} runs past the figure's edge")
    check(set(scales) == SCALES[layout], f"{path}: scales {sorted(scales)}")
    check(not re.search(r"base model", alltext, re.I), f"{path}: prints 'base model'")
    if layout in MAIN:
        check(not any(e.get(k) for e in everything for k in ("data-ci", "data-ci-row", "data-ci-model")), f"{path}: draws an interval")
        check(not re.search(r"\[|interval", alltext), f"{path}: prints an interval")
        CHECKS[layout](path, scales[layout], texts)
    tags = [text_of(t) for t in texts if t.get("data-panel-tag")]
    check(tags == TAGS.get(layout, []), f"{path}: panel tags {tags}")

    if "bars" in scales:
        g = scales["bars"]
        x0, x1 = float(g.get("data-x0")), float(g.get("data-x1"))
        X = lambda v: x0 + v * (x1 - x0)
        for b in spec["bars"]:
            rects = [e for e in g.iter() if e.get("data-bar") == b["id"]]
            check([r.get("data-key") for r in rects] == keys + ["unexplained"], f"{path}: bar {b['id']} segment order")
            acc = 0.0
            for r in rects:
                v = want["bars"][b["id"]]["segments"][r.get("data-key")]
                check(float(r.get("data-v")) == v, f"{path}: bar {b['id']} {r.get('data-key')} value")
                check(near(r.get("x"), X(acc)) and near(float(r.get("x")) + float(r.get("width")), X(acc + v)),
                      f"{path}: bar {b['id']} {r.get('data-key')} geometry")
                acc += v
            e = want["bars"][b["id"]]["explained"]
            label = [text_of(t) for t in texts if t.get("data-explained") == b["id"]]
            expect = ("explained " if b["id"] == spec["bars"][0]["id"] else "") + pct(e["v"]) + (f" {pm(e['v'], e['ci'])}" if e["ci"] else "")
            check(label == [expect], f"{path}: bar {b['id']} prints {label}, expected {expect!r}")
            wl = [l for l in g.iter() if l.get("data-ci") == b["id"]]
            if e["ci"]:
                check(len(wl) == 1 and near(wl[0].get("x1"), X(e["ci"][0])) and near(wl[0].get("x2"), X(e["ci"][1]))
                      and float(wl[0].get("data-lo")) == e["ci"][0] and float(wl[0].get("data-hi")) == e["ci"][1],
                      f"{path}: bar {b['id']} interval")
            else:
                check(not wl, f"{path}: bar {b['id']} draws an interval it has none of")
            check([text_of(t) for t in texts if t.get("data-bar-label") == b["id"]] == [b["label"]], f"{path}: bar {b['id']} label")

    if "explanations" in scales:
        g = scales["explanations"]
        x0, x1, d0, d1 = (float(g.get(k)) for k in ("data-x0", "data-x1", "data-d0", "data-d1"))
        X = lambda v: x0 + (v - d0) / (d1 - d0) * (x1 - x0)
        for fm in spec["families"]:
            title = fam_title(fm)
            check([text_of(t) for t in texts if t.get("data-family") == fm["id"]] == [title], f"{path}: family title {fm['id']}")
            tot = [text_of(t) for t in texts if t.get("data-family-total") == fm["id"]]
            F = want["families"][fm["id"]]
            check(tot == ([f"{pct(F['observed'])} (shuffled {pct(F['shuffle'])})"] if len(fm["items"]) > 1 else []),
                  f"{path}: family total {fm['id']} prints {tot}")
            for it in fm["items"]:
                k = it["key"]
                check([text_of(t) for t in texts if t.get("data-row-label") == k] == [it["label"]], f"{path}: row label {k}")
                for kind, field, cf in (("observed", "share", "ci"), ("shuffle", "shuffle", "shuffle_ci")):
                    dots = [e for e in g.iter() if e.get("data-row") == k and e.get("data-kind") == kind]
                    check(len(dots) == 1 and float(dots[0].get("data-v")) == P[field][k] and near(dots[0].get("cx"), X(P[field][k])),
                          f"{path}: {k} {kind} dot")
                    wl = [e for e in g.iter() if e.get("data-ci-row") == k and e.get("data-kind") == kind]
                    lo, hi = P[cf][k]
                    check(len(wl) == 1 and near(wl[0].get("x1"), X(lo)) and near(wl[0].get("x2"), X(hi)), f"{path}: {k} {kind} interval")
                sw = [e for e in g.iter() if e.get("data-swatch") == k]
                check(len(sw) == 1 and sw[0].get("fill") == it["color"], f"{path}: {k} swatch colour")
                seg = [e for e in scales["bars"].iter() if e.get("data-key") == k] if "bars" in scales else []
                check(all(s.get("fill") == it["color"] for s in seg), f"{path}: {k} segment colour differs from its swatch")

    if "models" in scales:
        g = scales["models"]
        x0, x1, d0, d1 = (float(g.get(k)) for k in ("data-x0", "data-x1", "data-d0", "data-d1"))
        for m in spec["models"]:
            A = want["per_model"][m["id"]]
            check([text_of(t) for t in texts if t.get("data-model-label") == m["id"]] == [m["label"]], f"{path}: model label {m['id']}")
            for kind in ("observed", "shuffle"):
                dots = [e for e in g.iter() if e.get("data-model") == m["id"] and e.get("data-kind") == kind]
                check(len(dots) == 1 and near(dots[0].get("cx"), X(A[kind]["v"])), f"{path}: {m['id']} {kind} dot")
                wl = [e for e in g.iter() if e.get("data-ci-model") == m["id"] and e.get("data-kind") == kind]
                lo, hi = A[kind]["ci"]
                check(len(wl) == 1 and near(wl[0].get("x1"), X(lo)) and near(wl[0].get("x2"), X(hi)), f"{path}: {m['id']} {kind} interval")
                check(d0 <= lo and hi <= d1, f"{path}: {m['id']} {kind} interval leaves the axis")
        for stat in ("median", "mean"):
            check([text_of(t) for t in texts if t.get("data-summary-label") == stat] == [T.term(stat)], f"{path}: {stat} label")
            for kind in ("observed", "shuffle"):
                dots = [e for e in g.iter() if e.get("data-summary") == stat and e.get("data-kind") == kind]
                check(len(dots) == 1 and near(dots[0].get("cx"), X(want["over_models"][kind][stat])), f"{path}: {stat} {kind} dot")

if errors:
    sys.exit("validate: " + str(len(errors)) + " problem(s)\n  " + "\n  ".join(errors))
print(f"validate: OK ({len(svgs)} SVG, {len(spec['models'])} models, {P['n_err']} errors pooled)")
