#!/usr/bin/env python3
import argparse, hashlib, json, pathlib, re, statistics, sys

ap = argparse.ArgumentParser()
ap.add_argument("--spec", required=True)
ap.add_argument("--probe-names", required=True)
ap.add_argument("--aggregate", required=True)
ap.add_argument("--svg", action="append", default=[])
a = ap.parse_args()
errors = []
def check(ok, msg):
    if not ok: errors.append(msg)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()

spec = json.load(open(a.spec))

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "shared"))
import key_spacing
import layer_order
from relation_sets import resolve_spec
import check_terms
import terms as terms_mod
resolve_spec(spec, a.spec)

T = terms_mod.load(a.probe_names)
for problem in check_terms.check(pathlib.Path(__file__).resolve().parent, check_terms.matcher(check_terms.names(T))):
    check(False, problem)
term_of = lambda m: T.at(f"metrics.{spec['metrics'][m]['name_key']}.{spec['metrics'][m]['name_variant']}")
check(sha(spec["snapshot"]) == spec["snapshot_sha256"], "snapshot sha256 differs from the spec pin")
check(sha(a.probe_names) == spec["probe_names"]["sha256"], "probe-name registry sha256 differs from the spec pin")
rows = [json.loads(l) for l in open(spec["snapshot"]) if l.strip()]

panel_of = {r: p for s, p in (("hernandez_engineered", "study"), ("bear", "bear53")) for r in spec["relation_sets"][s]}
check(all(panel_of.get(r["relation"]) == r.get("panel") for r in rows),
      "snapshot rows name a relation or panel outside the pinned registry selections")
agg = json.load(open(a.aggregate))
synthetic = bool(spec.get("snapshot_is_synthetic")) or any(r.get("synthetic") for r in rows)
check(agg["synthetic"] == synthetic, "aggregate synthetic flag")
check(not synthetic or spec["status"] == "DRAFT-SYNTHETIC", "synthetic snapshot but status is not DRAFT-SYNTHETIC")

check(synthetic or "DRAFT" not in spec["status"], f"real snapshot but status {spec['status']!r} is a draft status")

def close(x, y, what):
    check(abs(x - y) <= 1e-12 * max(1.0, abs(x), abs(y)), f"{what}: aggregate {y} vs recomputed {x}")
def ranks(v):
    o = sorted(range(len(v)), key=lambda i: (v[i], i)); r = [0.0] * len(v); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]: j += 1
        for k in range(i, j + 1): r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return r
def pearson(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((p - mx) * (q - my) for p, q in zip(x, y))
    return sxy / (sum((p - mx) ** 2 for p in x) * sum((q - my) ** 2 for q in y)) ** 0.5
spearman = lambda x, y: pearson(ranks(x), ranks(y))
def q7(v, p):
    s = sorted(v); h = (len(s) - 1) * p; lo = int(h)
    return s[lo] + (h - lo) * (s[min(lo + 1, len(s) - 1)] - s[lo])
M = 0xFFFFFFFF
def imul(x, y): return (x * y) & M
def mulberry32(seed):
    st = [seed & M]
    def nxt():
        st[0] = (st[0] + 0x6D2B79F5) & M; t = st[0]
        t = imul(t ^ (t >> 15), t | 1)
        t = (t ^ ((t + imul(t ^ (t >> 7), t | 61)) & M)) & M
        return ((t ^ (t >> 14)) & M) / 4294967296
    return nxt

pool_ids = [p["id"] for p in spec["pools"]]
cells, ce = {}, {}
for r in rows:
    check(r["pool"] in pool_ids, f"unknown pool {r['pool']}")
    if r["relation"] in ce: check(ce[r["relation"]] == r["ce"], f"CE differs across rows for {r['relation']}")
    ce[r["relation"]] = r["ce"]
    check(r["repeat"] == spec["repeat_id"], f"row {r['pool']}/{r['relation']} is repeat {r['repeat']}, not {spec['repeat_id']} (repeat rule)")
    check(r["relation"] not in cells.get(r["pool"], {}), f"more than one run for {r['pool']}/{r['relation']} (repeat rule)")
    cells.setdefault(r["pool"], {}).setdefault(r["relation"], []).append(r["gen"])
present = [p for p in pool_ids if p in cells]
check(agg["pools_present"] == present, "pools_present")
ref = spec["reference_pool"]
ref_label = next(p["label"] for p in spec["pools"] if p["id"] == ref)
cohort = [rel for rel in ce if all(rel in cells[p] for p in present)]
g = lambda p, rel: statistics.fmean(cells[p][rel])
cohort.sort(key=lambda rel: (-g(ref, rel), rel))
check(agg["cohort_order"] == cohort, "cohort order (descending reference outcome, relation id ties)")
n = len(cohort)
ceV = [ce[r] for r in cohort]; refV = [g(ref, r) for r in cohort]
rng = mulberry32(spec["bootstrap"]["seed"])
res = [[int(rng() * n) for _ in range(n)] for _ in range(spec["bootstrap"]["replicates"])]
for p in present:
    G = [g(p, r) for r in cohort]; A = agg["pools"][p]
    close(pearson(ceV, G), A["pearson_ce_gen"], f"{p} Pearson")
    close(spearman(ceV, G), A["spearman_ce_gen"], f"{p} Spearman")
    close(statistics.fmean(G), A["mean_gen"], f"{p} mean")
    close(q7(G, 0.5), A["median_gen"], f"{p} median")
    bR = [pearson([ceV[i] for i in idx], [G[i] for i in idx]) for idx in res]
    for k, q in enumerate((0.025, 0.975)): close(q7(bR, q), A["pearson_ci"][k], f"{p} Pearson CI {q}")
    bS = [spearman([ceV[i] for i in idx], [G[i] for i in idx]) for idx in res]
    for k, q in enumerate((0.025, 0.975)): close(q7(bS, q), A["spearman_ci"][k], f"{p} Spearman CI {q}")
    if p != ref:
        close(spearman(G, refV), A["rank_agreement"], f"{p} rank agreement")
        bRho = [spearman([G[i] for i in idx], [refV[i] for i in idx]) for idx in res]
        for k, q in enumerate((0.025, 0.975)): close(q7(bRho, q), A["rank_agreement_ci"][k], f"{p} rank CI {q}")

FONT_W = json.load(open(pathlib.Path(__file__).resolve().parents[1] / "shared" / "font-widths.json"))
def text_width(s, size, bold=False):
    table = FONT_W["bold" if bold else "regular"]
    return sum(table.get(ch, FONT_W["fallback_em"]) for ch in s) * size
TEXT = re.compile(r'<text x="([-0-9.]+)" y="([-0-9.]+)"([^>]*)>([^<]*)</text>')
unesc = lambda s: s.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")

def svg_width(t):
    return float(re.search(r'<svg [^>]*width="([0-9.]+)pt"', t).group(1))

def forest_key(path, t, band):
    m = re.search(r'<g data-key-block="marks"[^>]*>(.*)</g>', t, re.S)
    check(m is not None, f"{path}: forest lacks its marks key block")
    if m is None:
        return
    body = m.group(1)
    title = TEXT.findall(body.split("<g data-key-item", 1)[0])
    check(len(title) == 1 and unesc(title[0][3]) == T.term("key_marks") and 'font-weight="700"' in title[0][2]
          and 'text-anchor="end"' in title[0][2], f"{path}: key title is not the bold, right-aligned registry term")
    items = re.findall(r'<g data-key-item="([^"]+)">(.*?)</g>', body, re.S)
    want = ["bootstrap_ci"] + (["reference_interval"] if band else []) + ["reference_value"]
    check([i for i, _ in items] == want, f"{path}: key items {[i for i, _ in items]}, expected {want}")
    labels = [unesc(TEXT.findall(b)[0][3]) for _, b in items]
    check(labels == [T.term(k) if k == "bootstrap_ci" else f"{ref_label} {T.term(k)}" for k in want],
          f"{path}: key labels are not the registry terms {want}")
    axis = re.search(r'<line data-panel-axis="1" x1="([-0-9.]+)" y1="([-0-9.]+)"', t)
    x1, axis_y = float(axis.group(1)), float(axis.group(2))
    bar = re.search(r'<line x1="([-0-9.]+)"', items[0][1]) if items else None
    check(bar is not None and abs(float(bar.group(1)) - x1) < 1e-6, f"{path}: key's first glyph not flush with the first panel's left edge")
    check(title and float(title[0][0]) < x1, f"{path}: key title not left of the plot edge")
    ref = re.search(r'<line data-ref-line="1"[^>]*stroke="([^"]+)"[^>]*stroke-dasharray="([^"]+)"', t)
    dash = re.search(r'stroke="([^"]+)"[^>]*stroke-dasharray="([^"]+)"', items[-1][1]) if items else None
    check(ref and dash and ref.groups() == dash.groups(), f"{path}: key's dashed glyph differs from the reference line")
    if band:
        fill = re.search(r'data-ref-band="1"[^>]*(fill="[^"]+" fill-opacity="[^"]+")', t)
        check(fill and fill.group(1) in items[1][1], f"{path}: key's band swatch differs from the reference band")
    ys = sorted({float(y) for _, b in items for _, y, _, _ in TEXT.findall(b)})
    tick_y = axis_y + 9
    check(ys and ys[0] >= tick_y + 10, f"{path}: key collides with the tick labels")
    for (iid, b), label in zip(items, labels):
        x, _, _, _ = TEXT.findall(b)[0]
        check(float(x) + text_width(label, 8) <= svg_width(t), f"{path}: key item {iid} runs off the figure")
    check("bars:" not in t and spec["model"] not in t, f"{path}: the old footer note (or the model name) is still printed")

for path in a.svg:
    t = open(path).read()
    for problem in layer_order.check(t, path):
        check(False, problem)

    for problem in T.check_svg(t, str(path)):
        check(False, problem)

    for problem in key_spacing.check(t, str(path)):
        check(False, problem)
    if "forest" in str(path):
        check(key_spacing.count(t) > 0, f"{path}: the Marks key under the axis is not placed with the shared spacing")
    check(f"snapshot_sha256={spec['snapshot_sha256']}" in t, f"{path}: snapshot digest missing")
    check(f"probe_names_sha256={spec['probe_names']['sha256']}" in t, f"{path}: probe-name digest missing")
    check(("DRAFT-SYNTHETIC</text>" in t) == synthetic, f"{path}: watermark does not match the synthetic flag")
    check(synthetic or "DRAFT" not in t, f"{path}: real data but the SVG contains DRAFT")
    for size in re.findall(r'font-size="([0-9.]+)"', t): check(float(size) >= 8, f"{path}: font below 8 pt")

    variant = next((v for v in spec["variants"].values() if v["file"] == pathlib.Path(path).stem), {})
    printed = 8 * min(1.0, variant.get("print_width_pt", spec["paper_width_pt"]) / svg_width(t))
    check(printed >= 6.5 - 1e-9, f"{path}: {svg_width(t):.1f} pt wide, so its 8 pt type prints at {printed:.2f} pt (< 6.5)")

    check(f"{n} relations" not in t, f"{path}: prints the relation count")
    check(term_of("gen") in t, f"{path}: lacks the registry's outcome name {term_of('gen')!r}")
    if "forest" in pathlib.Path(path).name:
        heads = {}
        for attrs, body in re.findall(r'<text ([^>]*data-panel-heading="\d"[^>]*)>(.*?)</text>', t):
            k = re.search(r'data-panel-heading="(\d)"', attrs).group(1)
            heads.setdefault(k, []).append((attrs, body))
        ce, gen = term_of("ce"), term_of("gen")
        want = {"1": (T.term("pearson_r"), {f"between {ce} and {gen}", f"of {ce} and {gen}"}),
                "2": (T.term("spearman_rho"), {f"with the {ref_label} ordering"})}
        for k, (stat, rels) in want.items():
            h = heads.get(k, [])
            check(len(h) >= 2, f"{path}: panel heading {k} has {len(h)} lines, want 2 or more")
            if len(h) < 2: continue
            (a1, b1), rest = h[0], h[1:]
            got1 = unesc(re.sub(r"<[^>]+>", "", b1))
            got2 = " ".join(unesc(re.sub(r"<[^>]+>", "", b)) for _, b in rest)
            check(got1 == stat and 'font-weight="bold"' in a1, f"{path}: panel heading {k} line 1 is {got1!r}, want bold {stat!r}")
            check(got2 in rels and not any("font-weight" in a for a, _ in rest), f"{path}: panel heading {k} lines 2+ read {got2!r}, want regular {sorted(rels)}")
            italic = re.findall(r'<tspan[^>]*font-style="italic"[^>]*>([^<]*)</tspan>', "".join(b for _, b in h))
            check(italic == [stat.split(" ")[-1]], f"{path}: panel heading {k}'s statistic symbol is not set as math")
        forest_key(path, t, "band" in pathlib.Path(path).name)
        if "band" in pathlib.Path(path).name:
            band = re.findall(r'data-ref-band="1" data-lo="([^"]+)" data-hi="([^"]+)"', t)
            check(len(band) == 1, f"{path}: expected one reference band")
            for k, v in enumerate(band[0] if band else []):
                close(agg["pools"][ref]["pearson_ci"][k], float(v), f"{path} reference band {k}")

if errors:
    sys.stderr.write("\n".join(errors) + "\n"); sys.exit(1)
print(f"validate: OK ({len(rows)} rows, {len(present)} pools, {n} relations, synthetic={synthetic}, {len(a.svg)} SVGs)")
