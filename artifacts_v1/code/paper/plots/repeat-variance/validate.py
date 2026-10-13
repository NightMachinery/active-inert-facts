#!/usr/bin/env python3
import argparse, hashlib, json, math, re, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))

import layer_order
import check_terms
from half_width import half_width
import relation_sets
import terms as terms_mod

ap = argparse.ArgumentParser()
ap.add_argument("--spec", required=True)
ap.add_argument("--probe-names", required=True)
ap.add_argument("--aggregate", required=True)
ap.add_argument("--aggregate-rangebars")
ap.add_argument("--aggregate-correlation")
ap.add_argument("--svg", action="append", default=[])
a = ap.parse_args()
errors = []
def check(ok, msg):
    if not ok: errors.append(msg)
def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

def read_spec(path):
    spec = _read_spec(path)
    names = relation_sets.load(Path(path).parent / spec["relation_registry_path"])
    fill = lambda text: re.sub(r"\{([a-z_]+)\}", lambda m: names.display_name(m.group(1)), text)
    for p in spec["panels"]:
        if p.get("source") not in names.sources:
            sys.exit(f"validate: panel {p['id']} source {p.get('source')!r} is not a registry set")
        if "color" in p or "shape" in p:
            sys.exit(f"validate: panel {p['id']}: colour and shape come from the registry, not the spec")
    spec["panels"] = [{**p, "label": fill(p["label"]), "color": names.display_color(p["source"]), "mark": names.display_mark(p["source"])}
                      for p in spec["panels"]]
    if spec.get("banner"):
        spec["banner"] = fill(spec["banner"])
    return spec

def _read_spec(path):
    own = json.load(open(path))
    if "extends" not in own:
        return own
    base = json.load(open(Path(path).parent / own["extends"]))
    if "extends" in base:
        sys.exit("validate: spec extends chains are not supported")
    spec = {**base, **{k: v for k, v in own.items() if k not in ("extends", "panels_select")}}
    if "variants" not in own:
        spec["variants"] = {k: v for k, v in base["variants"].items() if v.get("inherit") is not False}
    if "panels_select" in own:
        by_id = {p["id"]: p for p in base["panels"]}
        spec["panels"] = [by_id[i] for i in own["panels_select"]]
    return spec

spec = read_spec(a.spec)
pending = [f"{p['id']} repeats {s['repeats']}: {s['pending']}" for p in spec["panels"] for s in p["sources"] if s.get("pending")]
if pending:
    sys.exit("validate: awaiting repeat fleet rows; pending sources:\n  " + "\n  ".join(pending))
for p in spec["panels"]:
    covered = sorted(r for s in p["sources"] for r in s["repeats"])
    if covered != sorted(spec["repeats"]):
        sys.exit(f"validate: panel {p['id']} sources cover repeats {covered}, not {spec['repeats']} exactly once")
check(sha(spec["snapshot"]) == spec["snapshot_sha256"], "snapshot sha256 differs from the spec pin")
check(sha(a.probe_names) == spec["probe_names"]["sha256"], "probe-name registry sha256 differs from the spec pin")

T = terms_mod.load(a.probe_names)
gen_label = T.at(f"metrics.{spec['metrics']['gen']['name_key']}.{spec['metrics']['gen']['name_variant']}")
lint = check_terms.check(Path(__file__).resolve().parent, check_terms.matcher(check_terms.names(T)))
for problem in lint:
    check(False, problem)
rows = [json.loads(l) for l in open(spec["snapshot"]) if l.strip()]
agg = json.load(open(a.aggregate))
K = len(spec["repeats"])
models = [m["id"] for m in spec["models"]]
registry = relation_sets.load(Path(a.spec).parent / spec["relation_registry_path"])
retired = set(registry.pinned(spec["excluded_relation_pin"], "excluded_relation_pin"))
relations_of = {p["id"]: registry.pinned(p["relation_pin"], f"panel {p['id']}") for p in spec["panels"]}
check(all(r["relation"] not in retired for r in rows), "a retired relation is in the snapshot")
check(all(r["model"] in models for r in rows), "a model outside spec.models is in the snapshot")
check(spec.get("snapshot_rows", len(rows)) == len(rows), "snapshot_rows differs from the snapshot")

def mulberry32(seed):
    st = [seed & 0xFFFFFFFF]
    def nxt():
        st[0] = (st[0] + 0x6D2B79F5) & 0xFFFFFFFF
        t = st[0]
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t = (t ^ (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF))) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return nxt

def q7(v, p):
    s = sorted(v); h = (len(s) - 1) * p; lo = math.floor(h)
    return s[lo] + (h - lo) * (s[min(lo + 1, len(s) - 1)] - s[lo])

def pearson(xs, ys):
    mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
    sxy = sxx = syy = 0.0
    for x, y in zip(xs, ys):
        sxy += (x - mx) * (y - my); sxx += (x - mx) ** 2; syy += (y - my) ** 2
    return sxy / math.sqrt(sxx * syy)

def share(cs):
    n = len(cs); flat = [y for v in cs for y in v]; grand = sum(flat) / len(flat)
    mu = [sum(v) / len(v) for v in cs]
    msb = K * sum((m - grand) ** 2 for m in mu) / (n - 1)
    msw = sum(sum((y - m) ** 2 for y in v) for v, m in zip(cs, mu)) / (n * (K - 1))
    return 1 - (msb - msw) / (msb + (K - 1) * msw)

def close(x, y, what):
    check(abs(x - y) <= 1e-12 * max(1.0, abs(x), abs(y)), f"{what}: aggregate {y} vs recomputed {x}")

claimed = {}
for panel in spec["panels"]:
    for source in panel["sources"]:
        for rel in relations_of[panel["id"]]:
            for rep in source["repeats"]:
                for m in panel.get("models") or models:
                    key = (source["path"], rel, rep, m)
                    check(key not in claimed, f"{rel} repeat {rep} {m} of {source['path']} is in panels {claimed.get(key)} and {panel['id']}")
                    claimed[key] = panel["id"]
check(len({p["color"] for p in spec["panels"]}) == len(spec["panels"]), "two panels share a colour")
neutral_variants = [k for k, v in spec["variants"].items() if v.get("encoding") == "neutral"]
if neutral_variants:
    check(all(p.get("neutral") for p in spec["panels"]), "a neutral variant exists but a panel lacks `neutral`")
    paints = {(p["neutral"]["color"].lower(), p["mark"]) for p in spec["panels"] if p.get("neutral")}
    check(len(paints) == len(spec["panels"]), "two panels share a neutral ink and mark")

for panel in spec["panels"]:
    pid = panel["id"]
    cells = {}
    for r in rows:
        if r["panel"] != pid: continue
        check(r["relation"] in relations_of[pid], f"{pid}: row of relation {r['relation']} outside the panel's selection")
        check(not panel.get("models") or r["model"] in panel["models"], f"{pid}: row of model {r['model']} outside the panel's models")
        cells.setdefault((r["model"], r["relation"]), {})[r["repeat"]] = r
    check(all(sorted(c) == sorted(spec["repeats"]) for c in cells.values()), f"{pid}: a cell lacks a repeat")
    A = agg["panels"][pid]
    check(A["cells"] == len(cells), f"{pid}: cell count")
    check(A["relations"] == len({k[1] for k in cells}), f"{pid}: relation count")
    check({k[1] for k in cells} == set(relations_of[pid]), f"{pid}: relations differ from selection {panel['relation_pin']['selection']}")
    xs, ys = [], []
    ratios, ratios_m, degenerate = [], [], 0
    for (m, rel), c in cells.items():
        v = [c[k]["gen"] for k in spec["repeats"]]
        for i in range(K):
            for j in range(i + 1, K):
                xs.append(v[i]); ys.append(v[j])
        p = statistics.fmean(v); s = statistics.stdev(v)
        n = c[spec["repeats"][0]]["n_heldout"]
        se = math.sqrt(p * (1 - p) / n)
        cc = c[spec["repeats"][0]]
        se_m = math.sqrt(p * (1 - p) * cc["inv_h_sum"]) / cc["n_heldout_classes"]
        if se > 0:
            ratios.append(s / se); ratios_m.append(s / se_m)
            check(se_m >= se - 1e-15, f"{pid} {m}/{rel}: macro-aware floor below binomial floor")
        else:
            degenerate += 1
    close(statistics.correlation(xs, ys), A["pairs_pearson"], f"{pid} pairs Pearson")
    check(A["pairs"] == len(xs), f"{pid}: pair count")
    F = A["floor"]
    check(F["cells_with_floor"] == len(ratios) and F["degenerate_floor_cells"] == degenerate, f"{pid}: floor cell counts")
    close(statistics.median(ratios), F["median_ratio"], f"{pid} median ratio")
    close(statistics.fmean(ratios), F["mean_ratio"], f"{pid} mean ratio")
    close(max(ratios), F["max_ratio"], f"{pid} max ratio")
    check(F["at_or_below"] == sum(x <= 1 for x in ratios) and F["above"] == sum(x > 1 for x in ratios), f"{pid}: at/below counts")
    close(statistics.median(ratios_m), F["median_ratio_macro"], f"{pid} macro median ratio")
    close(statistics.fmean(ratios_m), F["mean_ratio_macro"], f"{pid} macro mean ratio")
    for m in models:
        cs = [[c[k]["gen"] for k in spec["repeats"]] for (mm, _), c in cells.items() if mm == m]
        if len(cs) < 3:
            check(m not in A["per_model"], f"{pid} {m}: decomposed with fewer than 3 relations")
            continue
        n = len(cs)
        grand = statistics.fmean([y for v in cs for y in v])
        msb = K * sum((statistics.fmean(v) - grand) ** 2 for v in cs) / (n - 1)
        msw = sum(sum((y - statistics.fmean(v)) ** 2 for y in v) for v in cs) / (n * (K - 1))
        P = A["per_model"][m]
        check(P["n_relations"] == n, f"{pid} {m}: n_relations")
        close((msb - msw) / (msb + (K - 1) * msw), P["icc1"], f"{pid} {m} ICC(1)")
        close((msb - msw) / msb, P["icck"], f"{pid} {m} ICC(k)")
        close(1 - (msb - msw) / (msb + (K - 1) * msw), P["run_to_run_share"], f"{pid} {m} share = 1 - ICC(1)")

    if spec.get("bootstrap"):
        rels = sorted({rel for (_, rel) in cells})
        vals = {(mm, rel): [c[k]["gen"] for k in spec["repeats"]] for (mm, rel), c in cells.items()}
        dec = [m for m in models if m in A["per_model"]]
        rng = mulberry32(spec["bootstrap"]["seed"]); n = len(rels)
        bR, bS = [], {m: [] for m in dec}
        for _ in range(spec["bootstrap"]["replicates"]):
            pick = [rels[int(rng() * n)] for _ in range(n)]
            px, py = [], []
            for rel in pick:
                for (mm, r2), v in vals.items():
                    if r2 != rel: continue
                    for i in range(K):
                        for j in range(i + 1, K):
                            px.append(v[i]); py.append(v[j])
            bR.append(pearson(px, py))
            for m in dec:
                bS[m].append(share([vals[(m, rel)] for rel in pick]))
        for k, q in enumerate((0.025, 0.975)):
            close(q7(bR, q), A["pairs_pearson_ci"][k], f"{pid} pairs Pearson CI {q}")
            for m in dec:
                close(q7(bS[m], q), A["per_model"][m]["run_to_run_share_ci"][k], f"{pid} {m} share CI {q}")

check("DRAFT" not in str(spec.get("status", "")), f"spec status {spec.get('status')!r} is a draft status")
for path in a.svg:
    t = open(path).read()
    for problem in layer_order.check(t, path):
        check(False, problem)

    for problem in T.check_svg(t, str(path)):
        check(False, problem)
    check("DRAFT" not in t, f"{path}: contains DRAFT")
    check(f"snapshot_sha256={spec['snapshot_sha256']}" in t, f"{path}: snapshot digest missing from metadata")
    check(f"probe_names_sha256={spec['probe_names']['sha256']}" in t, f"{path}: probe-name digest missing from metadata")
    own_w = next((v.get("width_pt") for v in spec["variants"].values() if f"{spec['output_prefix']}-{v['file']}.svg" == Path(path).name), None)
    check(f'width="{own_w or spec["paper_width_pt"]}pt"' in t, f"{path}: not the paper width (or the variant's own width_pt)")
    import re
    for size in re.findall(r'font-size="([0-9.]+)"', t):
        check(float(size) >= 8, f"{path}: font below 8 pt")
    check(("within-cell" not in t) and ("within cell" not in t), f"{path}: prints the retired term within-cell (now run-to-run)")
    for txt in re.findall(r"<text[^>]*>([^<]*)</text>", t):
        check(not re.search(r"\[\s*-?\.?\d[\d.]*\s*,\s*-?\.?\d[\d.]*\s*\]", txt),
              f"{path}: prints a bracket interval instead of ± h: {txt!r}")
    if spec.get("banner"):
        check(f">{spec['banner']}<" in t.replace("&amp;", "&"), f"{path}: lacks the view's banner")
check(any(gen_label in open(p).read() for p in a.svg) or not a.svg, "no SVG prints the registry's outcome label")

def variant_path(key):
    name = f"{spec['output_prefix']}-{spec['variants'][key]['file']}.svg"
    return next((p for p in a.svg if Path(p).name == name), None)

FONT_W = json.load(open(Path(__file__).resolve().parent.parent / "shared" / "font-widths.json"))
KEY_ROW = 10.5
def text_width(s, size, bold=False):
    table = FONT_W["bold" if bold else "regular"]
    return sum(table.get(ch, FONT_W["fallback_em"]) for ch in s) * size
def unesc(s):
    return s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&amp;", "&")
def block(t, bid):
    m = re.search(r'<g data-key-block="' + re.escape(bid) + r'">(.*?)</g>(?=\n|<g data-key-block|$)', t, re.S)
    return m.group(1) if m else None
TEXT = re.compile(r'<text (?:data-legend="1" )?x="([-0-9.]+)" y="([-0-9.]+)"[^>]*>([^<]*)</text>')

def combo_key(path, t, v, plot_top):
    area = re.search(r'<rect x="([-0-9.]+)" y="[-0-9.]+" width="([-0-9.]+)"[^>]*data-plot-area="1"', t)
    x0 = float(area.group(1))
    tag_b = re.search(r'<text data-panel-tag="b" x="([-0-9.]+)" y="([-0-9.]+)"', t)
    check(tag_b is not None, f"{path}: panel b tag missing")
    title = block(t, "set")
    items = block(t, "set-items")
    check(title is not None and items is not None, f"{path}: combo key lacks the set key block")
    if title is None or items is None:
        return
    tt = TEXT.findall(title)
    check(len(tt) == 1 and unesc(tt[0][2]) == T.term("key_relation_set") and 'font-weight="700"' in title,
          f"{path}: set key block title is not the bold registry term")
    tx, ty = float(tt[0][0]), float(tt[0][1])
    check(abs(tx - x0) < 1e-6, f"{path}: set key title not flush with panel a's plot edge ({tx} vs {x0})")
    got = re.findall(r'<g data-key-item="([^"]+)">(.*?)</g>', items, re.S)
    check([i for i, _ in got] == [f"set:{p['id']}" for p in spec["panels"]], f"{path}: set key items are not one per panel in panel order")
    for k, ((iid, body), panel) in enumerate(zip(got, spec["panels"])):
        lx, ly, label = TEXT.findall(body)[0]
        check(abs(float(ly) - (ty + (k + 1) * KEY_ROW)) < 1e-6, f"{path}: key item {iid} not on its own evenly spaced row")
        check(unesc(label).startswith(panel["label"] + ": r = "), f"{path}: key item {iid} does not start with its set name and r")
        check(f'data-panel="{panel["id"]}"' in body, f"{path}: key item {iid} lacks its panel's mark")
        cx = re.search(r'(?:cx="|d="M)([-0-9.]+)', body)
        check(cx and abs(float(cx.group(1)) - (x0 + 2.5)) < 1e-6, f"{path}: key mark {iid} not flush with panel a's plot edge")
        right = float(lx) + text_width(unesc(label), 8)
        check(tag_b is not None and right < float(tag_b.group(1)), f"{path}: key item {iid} runs into panel b's column")
        check(float(ly) + 2 < plot_top, f"{path}: key item {iid} is not above the plot area")
    marks = block(t, "marks")
    if '<line data-ci="' in t:
        zero = re.search(r'<line data-layer="grid" x1="([-0-9.]+)"[^>]*stroke="#6B7280"', t)
        mi = block(t, "marks-items")
        check(marks is not None and mi is not None, f"{path}: ci variant lacks the marks key block for its whiskers")
        if marks is None or mi is None:
            return
        mt = TEXT.findall(marks)
        check(len(mt) == 1 and unesc(mt[0][2]) == T.term("key_marks") and abs(float(mt[0][1]) - ty) < 1e-6,
              f"{path}: marks key title is not the registry term on the key's first row")
        check(zero is not None and abs(float(mt[0][0]) - float(zero.group(1))) < 1e-6, f"{path}: marks key title not flush with panel b's zero line")
        mit = TEXT.findall(mi)
        check([unesc(x[2]) for x in mit] == [T.term("bootstrap_ci")] and abs(float(mit[0][1]) - (ty + KEY_ROW)) < 1e-6,
              f"{path}: marks key block does not explain the whisker as the registry's bootstrap interval")
    else:
        check(marks is None, f"{path}: marks key block without interval whiskers to explain")

for key, v in spec["variants"].items():
    layout = v.get("layout", key)
    if layout not in ("pairs", "combo"):
        continue
    path = variant_path(key)
    if path is None:
        continue
    t = open(path).read()
    tops = [float(y) for y in re.findall(r'<rect x="[-0-9.]+" y="([-0-9.]+)"[^>]*data-plot-area="1"', t)]
    check(len(tops) == (len(spec["panels"]) if layout == "pairs" else 1), f"{path}: plot-area count")
    legend = re.findall(r'<text data-legend="1" x="[-0-9.]+" y="([-0-9.]+)"[^>]*>([^<]*)</text>', t)
    check(legend and all(float(y) + 2 < min(tops) for y, _ in legend), f"{path}: a legend line is not above the plot area")
    shown = " ".join(s for _, s in legend)
    for panel in spec["panels"]:
        A = agg["panels"][panel["id"]]
        check(panel["label"] in shown, f"{path}: legend lacks {panel['label']}")
        c = lambda x, d: f"{x:.{d}f}".replace("0.", ".", 1) if x >= 0 else f"-{-x:.{d}f}".replace("0.", ".", 1)
        want = f"r = {c(A['pairs_pearson'], 3)}"
        if "pairs_pearson_ci" in A:
            est = A["pairs_pearson"]
            lo, hi = A["pairs_pearson_ci"]
            check(lo <= est <= hi, f"{path}: {panel['label']}'s estimate {est} outside its interval [{lo}, {hi}]")
            want += f" ± {half_width(est, (lo, hi), 2).replace('0.', '.', 1)}"
        check(want in shown, f"{path}: legend lacks {panel['label']}'s {want!r}")
        check(" cells)" not in shown, f"{path}: legend prints a cell count (no n in figures)")
        check("[" not in shown and "]" not in shown, f"{path}: legend prints a bracket interval instead of ± h")
        if v.get("encoding") != "neutral":
            check(f'fill="{panel["color"]}"' in t or f'stroke="{panel["color"]}"' in t, f"{path}: never uses {panel['label']}'s colour")

    if v.get("ci"):
        whiskers = re.findall(r'<line data-ci="([^"]+)" data-model="([^"]+)" data-lo="([^"]+)" data-hi="([^"]+)"', t)
        want_w = {(p["id"], m): A2 for p in spec["panels"] for m, A2 in agg["panels"][p["id"]]["per_model"].items() if "run_to_run_share_ci" in A2}
        check(len(whiskers) == len(want_w), f"{path}: {len(whiskers)} interval whiskers, expected {len(want_w)}")
        for pid_, m, lo, hi in whiskers:
            ci = want_w[(pid_, m)]["run_to_run_share_ci"]
            close(ci[0], float(lo), f"{path} {pid_} {m} whisker low")
            close(ci[1], float(hi), f"{path} {pid_} {m} whisker high")
    if layout == "combo":
        combo_key(path, t, v, min(tops) if tops else 0.0)

SHAPE_TAG = {"dot": "circle", "ring": "circle", "triangle": "path"}
for key, v in spec["variants"].items():
    path = variant_path(key)
    if path is None or v.get("encoding") == "neutral":
        continue
    t = open(path).read()
    for panel in spec["panels"]:
        for tag, attrs in re.findall(r'<(\w+) data-panel="' + re.escape(panel["id"]) + r'"([^>]*)/>', t):
            check(tag == SHAPE_TAG[panel["mark"]] and f'data-mark="{panel["mark"]}"' in attrs, f"{path}: {panel['label']} mark is not its set's {panel['mark']}")
            paint = f'stroke="{panel["color"]}"' if panel["mark"] == "ring" else f'fill="{panel["color"]}"'
            check(paint in attrs, f"{path}: {panel['label']} mark not in its set colour")
for key in neutral_variants:
    path = variant_path(key)
    if path is None:
        continue
    t = open(path).read()
    for panel in spec["panels"]:
        check(panel["color"].lower() not in t.lower(), f"{path}: uses {panel['label']}'s panel colour")
        marks = re.findall(r'<(\w+) data-panel="' + re.escape(panel["id"]) + r'"([^>]*)/>', t)
        check(marks, f"{path}: no marks for {panel['label']}")
        want = panel["neutral"]
        for tag, attrs in marks:
            check(tag == SHAPE_TAG[panel["mark"]], f"{path}: {panel['label']} mark is a {tag}")
            if panel["mark"] == "ring":
                check('fill="none"' in attrs and f'stroke="{want["color"]}"' in attrs, f"{path}: {panel['label']} mark not hollow in its ink")
            else:
                check(f'fill="{want["color"]}"' in attrs, f"{path}: {panel['label']} mark not filled with its ink")

LAYOUTS = {v.get("layout", k) for k, v in spec["variants"].items()}
def cells_of(pid):
    out = {}
    for r in rows:
        if r["panel"] == pid:
            out.setdefault((r["model"], r["relation"]), [None] * K)[spec["repeats"].index(r["repeat"])] = r["gen"]
    return out
def ci95(v):
    return [q7(v, 0.025), q7(v, 0.975)]
def spear(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); out = [0.0] * len(v); i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return out
    return pearson(rank(xs), rank(ys))
def close_ci(x, y, what):
    close(x[0], y[0], what + " low"); close(x[1], y[1], what + " high")

if "rangebars" in LAYOUTS:
    RB = json.load(open(a.aggregate_rangebars))
    check(RB["snapshot_sha256"] == spec["snapshot_sha256"], "rangebars aggregate: snapshot digest")
    for panel in spec["panels"]:
        pid = panel["id"]; cells = cells_of(pid); P = RB["panels"][pid]
        def sym(cs):
            px, py = [], []
            for v in cs:
                for i in range(K):
                    for j in range(i + 1, K):
                        px.append(v[i]); py.append(v[j])
            return pearson(px + py, py + px)
        def spread(cs):
            rg = [max(v) - min(v) for v in cs]; s = [statistics.stdev(v) for v in cs]
            return [statistics.median(rg), statistics.fmean(rg), statistics.median(s), statistics.fmean(s)]
        vals = list(cells.values())
        check(P["cells"] == len(vals), f"rangebars {pid}: cell count")
        close(sym(vals), P["pairs_pearson_symmetric"], f"rangebars {pid} symmetric r")
        for got, (k, s) in zip(spread(vals), [("range", "median"), ("range", "mean"), ("sd", "median"), ("sd", "mean")]):
            close(got, P[k][s], f"rangebars {pid} {k} {s}")
        runs = {(c["model"], c["relation"]): c for c in P["cell_runs"]}
        check(set(runs) == set(cells), f"rangebars {pid}: cell_runs differ from the snapshot's cells")
        for key, v in cells.items():
            c = runs.get(key)
            if c:
                check(c["runs"] == v, f"rangebars {pid} {key}: runs")
                close(max(v) - min(v), c["range"], f"rangebars {pid} {key} range")
                close(statistics.stdev(v), c["sd"], f"rangebars {pid} {key} sd")
        if spec.get("bootstrap"):
            rels = sorted({rel for (_, rel) in cells})
            by_rel = {rel: [v for (m, r2), v in cells.items() if r2 == rel] for rel in rels}
            rng = mulberry32(spec["bootstrap"]["seed"]); n = len(rels)
            bs, bq = [], [[], [], [], []]
            for _ in range(spec["bootstrap"]["replicates"]):
                pick = [v for _ in range(n) for v in by_rel[rels[int(rng() * n)]]]
                bs.append(sym(pick))
                for i, x in enumerate(spread(pick)):
                    bq[i].append(x)
            close_ci(ci95(bs), P["pairs_pearson_symmetric_ci"], f"rangebars {pid} symmetric r CI")
            for b, (k, s) in zip(bq, [("range", "median"), ("range", "mean"), ("sd", "median"), ("sd", "mean")]):
                close_ci(ci95(b), P[k][s + "_ci"], f"rangebars {pid} {k} {s} CI")

if LAYOUTS & {"correlation", "correlation_range"}:
    CA = json.load(open(a.aggregate_correlation)); CS = spec["correlation"]
    check(sha(CS["predictor"]["path"]) == CS["predictor"]["sha256"], "correlation predictor snapshot sha256 differs from the spec pin")
    check(sha(CS["table1_reference"]["path"]) == CS["table1_reference"]["sha256"], "Table 1 reference sha256 differs from the spec pin")
    ref = json.load(open(CS["table1_reference"]["path"]))["models"]
    xrow = {(r["model"].lower(), r["relation"]): r for r in map(json.loads, filter(str.strip, open(CS["predictor"]["path"])))}
    series = [str(k) for k in spec["repeats"]] + ["mean"]
    pairs = [(a_, b_) for i, a_ in enumerate(series[:K]) for b_ in series[i + 1:K]]
    check(CA["series"] == series and CA["pairs"] == [f"{x}-{y}" for x, y in pairs], "correlation aggregate: series or pairs")
    def corr_stats(x, ys):
        st = {s: (pearson(x, y), spear(x, y)) for s, y in zip(series, ys)}
        d = {f"{p}-{q}": (st[p][0] - st[q][0], st[p][1] - st[q][1]) for p, q in pairs}
        return st, d, max(abs(v[0]) for v in d.values()), max(abs(v[1]) for v in d.values())
    for scope in CS["scopes"]:
        sid = scope["id"]; S = CA["scopes"][sid]
        cells = {}
        for pid in scope["panels"]:
            cells.update(cells_of(pid))
        ms = [m for m in models if (m in scope["models"] if scope.get("models") else any(k[0] == m for k in cells))]
        rels = sorted({k[1] for k in cells})
        check(S["models"] == ms and S["relation_ids"] == rels and S["relations"] == len(rels), f"correlation {sid}: models or relations")
        data = {}
        for m in ms:
            x, ys = [], [[] for _ in series]
            for rel in rels:
                v = cells.get((m, rel)); p = xrow.get((m.lower(), rel))
                check(v is not None and p is not None, f"correlation {sid}: {m} {rel} missing")
                if v is None or p is None:
                    continue
                check(p[CS["predictor"]["outcome_field"]] == v[0], f"correlation {sid}: {m} {rel}: predictor snapshot outcome differs from repeat {series[0]}")
                x.append(p[CS["predictor"]["field"]])
                for i in range(K):
                    ys[i].append(v[i])
                ys[K].append(sum(v) / K)
            data[m] = (x, ys)
        est = {}
        for m in ms:
            st, d, rs, qs = corr_stats(*data[m]); est[m] = (rs, qs, st); P = S["per_model"][m]
            for s in series:
                close(st[s][0], P["r"][s]["est"], f"correlation {sid} {m} r {s}")
                close(st[s][1], P["rho"][s]["est"], f"correlation {sid} {m} rho {s}")
            for k, v in d.items():
                close(v[0], P["r_diff"][k]["est"], f"correlation {sid} {m} r diff {k}")
                close(v[1], P["rho_diff"][k]["est"], f"correlation {sid} {m} rho diff {k}")
            close(rs, P["r_spread"]["est"], f"correlation {sid} {m} r spread"); close(qs, P["rho_spread"]["est"], f"correlation {sid} {m} rho spread")
            if scope.get("table1_check"):
                R0 = ref[m]["gain"]
                check(sorted(ref[m]["relations"]) == rels, f"correlation {sid} {m}: Table 1 relations differ")
                close(R0["r"], st[series[0]][0], f"correlation {sid} {m}: repeat 0 r vs Table 1 raw r")
                close(R0["rho"], st[series[0]][1], f"correlation {sid} {m}: repeat 0 rho vs Table 1 raw rho")
        SUMMARIES = [(c, w, s) for c in (0, 1) for w in ("median", "mean") for s in series]
        def over(e):
            r_ = [e[m][0] for m in ms]; q_ = [e[m][1] for m in ms]
            by = [(statistics.median if w == "median" else statistics.fmean)([e[m][2][s][c] for m in ms]) for c, w, s in SUMMARIES]
            return [statistics.median(r_), statistics.fmean(r_), statistics.median(q_), statistics.fmean(q_)] + by
        OK = ["r_spread_median", "r_spread_mean", "rho_spread_median", "rho_spread_mean"] + [("r", "rho")[c] + f".{w}.{s}" for c, w, s in SUMMARIES]
        def om(k):
            o = S["over_models"]
            for part in k.split("."):
                o = o[part]
            return o
        if len(ms) > 1:
            for got, k in zip(over(est), OK):
                close(got, om(k)["est"], f"correlation {sid} {k}")
        else:
            check("over_models" not in S, f"correlation {sid}: over_models for one model")
        if spec.get("bootstrap"):
            rng = mulberry32(spec["bootstrap"]["seed"]); n = len(rels)
            B = {m: {"st": {s: ([], []) for s in series}, "d": {f"{p}-{q}": ([], []) for p, q in pairs}, "rs": [], "qs": []} for m in ms}
            bo = [[] for _ in OK]
            for _ in range(spec["bootstrap"]["replicates"]):
                pick = [int(rng() * n) for _ in range(n)]
                e = {}
                for m in ms:
                    x, ys = data[m]
                    st, d, rs, qs = corr_stats([x[i] for i in pick], [[y[i] for i in pick] for y in ys]); e[m] = (rs, qs, st)
                    for s in series:
                        B[m]["st"][s][0].append(st[s][0]); B[m]["st"][s][1].append(st[s][1])
                    for k, v in d.items():
                        B[m]["d"][k][0].append(v[0]); B[m]["d"][k][1].append(v[1])
                    B[m]["rs"].append(rs); B[m]["qs"].append(qs)
                if len(ms) > 1:
                    for i, v in enumerate(over(e)):
                        bo[i].append(v)
            for m in ms:
                P = S["per_model"][m]
                for s in series:
                    close_ci(ci95(B[m]["st"][s][0]), P["r"][s]["ci"], f"correlation {sid} {m} r {s} CI")
                    close_ci(ci95(B[m]["st"][s][1]), P["rho"][s]["ci"], f"correlation {sid} {m} rho {s} CI")
                for k in B[m]["d"]:
                    close_ci(ci95(B[m]["d"][k][0]), P["r_diff"][k]["ci"], f"correlation {sid} {m} r diff {k} CI")
                    close_ci(ci95(B[m]["d"][k][1]), P["rho_diff"][k]["ci"], f"correlation {sid} {m} rho diff {k} CI")
                close_ci(ci95(B[m]["rs"]), P["r_spread"]["ci"], f"correlation {sid} {m} r spread CI")
                close_ci(ci95(B[m]["qs"]), P["rho_spread"]["ci"], f"correlation {sid} {m} rho spread CI")
            if len(ms) > 1:
                for b, k in zip(bo, OK):
                    close_ci(ci95(b), om(k)["ci"], f"correlation {sid} {k} CI")

for key, v in spec["variants"].items():
    layout = v.get("layout", key)
    path = variant_path(key)
    if path is None or layout not in ("rangebars", "correlation", "correlation_range"):
        continue
    t = open(path).read()
    if layout == "rangebars":
        want = {(p["id"], m, rel): vv for p in spec["panels"] for (m, rel), vv in cells_of(p["id"]).items()}
        got = re.findall(r'<line data-range="([^"]+)" data-model="([^"]+)" data-relation="([^"]+)" data-lo="([^"]+)" data-hi="([^"]+)"', t)
        check(len(got) == len(want) and {g[:3] for g in got} == set(want), f"{path}: one range bar per cell")
        for pid_, m, rel, lo, hi in got:
            vv = want.get((pid_, m, rel))
            check(vv is not None and float(lo) == min(vv) and float(hi) == max(vv), f"{path}: range bar of {pid_} {m} {rel}")
        check(len(re.findall(r'data-mean="', t)) == len(want), f"{path}: one mean tick per cell")
        check(len(re.findall(r'data-strip-title="1"', t)) == len(models) + sum(1 for p in spec["panels"] if p.get("models")), f"{path}: strip titles")
    else:
        simple = v.get("simple") is True or layout == "correlation_range"
        if layout == "correlation_range":
            bars = re.findall(r'data-corr-range="([^"]+)" data-lo="([^"]+)" data-hi="([^"]+)"', t)
            want_b = {f"{sid}:{m}:{st}": [S["per_model"][m][st][s]["est"] for s in CA["series"][:K]] for sid, S in CA["scopes"].items() for m in S["models"] for st in ("r", "rho")}
            check(len(bars) == len(want_b) and {b[0] for b in bars} == set(want_b), f"{path}: one range bar per scope, model and coefficient")
            for k, lo, hi in bars:
                check(k in want_b and float(lo) == min(want_b[k]) and float(hi) == max(want_b[k]), f"{path}: range bar {k}")
        shown = [s for s in CA["series"] if not (simple and s == "mean")]
        dots = re.findall(r'data-corr="([^"]+)" data-v="([^"]+)"', t)
        want = {f"{sid}:{m}:{st}:{s}": S["per_model"][m][st][s] for sid, S in CA["scopes"].items() for m in S["models"] for st in ("r", "rho") for s in shown}
        check(len(dots) == len(want) and {d[0] for d in dots} == set(want), f"{path}: one mark per scope, model, coefficient and series")
        for k, val in dots:
            check(k in want and float(val) == want[k]["est"], f"{path}: mark {k} is not the aggregate's value")
        whiskers = re.findall(r'data-ci="([^"]+)" data-lo="([^"]+)" data-hi="([^"]+)"', t)
        refs = re.findall(r'data-ref="([^"]+)" data-v="([^"]+)"', t)
        if simple:
            check(not whiskers and not refs, f"{path}: the simple variant draws intervals or reference lines")
        else:
            check(len(whiskers) == len(want), f"{path}: one interval per mark")
            for k, lo, hi in whiskers:
                check(k in want and [float(lo), float(hi)] == want[k]["ci"], f"{path}: interval {k}")
            check(len(refs) == len(want) // len(shown), f"{path}: one repeat-0 reference per scope, model and coefficient")
            for k, val in refs:
                sid, m, st = k.split(":")
                check(float(val) == CA["scopes"][sid]["per_model"][m][st][CA["series"][0]]["est"], f"{path}: reference {k}")

if errors:
    sys.stderr.write("\n".join(errors) + "\n"); sys.exit(1)
print(f"validate: OK ({len(rows)} rows, {len(spec['panels'])} panels, {len(a.svg)} SVGs)")
