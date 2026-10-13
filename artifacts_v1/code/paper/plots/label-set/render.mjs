#!/usr/bin/env node
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { keyBlocks, keyBaselineBelow, keyBottom } from "../shared/key.mjs";
import { load as loadRelationSets, resolveSpec } from "../shared/relation-sets.mjs";
import { load as loadTerms } from "../shared/terms.mjs";
import { textWidth } from "../shared/text_width.mjs";
import { termWithSymbol } from "../shared/math_text.mjs";

const REG = loadRelationSets();
const setName = (s) => REG.displayName(s);

const fail = (m) => { process.stderr.write(`render: ${m}\n`); process.exit(1); };
const args = Object.fromEntries(process.argv.slice(2).reduce((p, v, i, a) => (v.startsWith("--") && p.push([v.slice(2), a[i + 1]]), p), []));
for (const k of ["spec", "probe-names", "aggregate"]) if (!args[k]) fail(`missing --${k}`);
const sha256 = (t) => createHash("sha256").update(t).digest("hex");
const spec = JSON.parse(readFileSync(args.spec, "utf8"));

resolveSpec(spec, args.spec);
const dataText = readFileSync(spec.snapshot, "utf8");
if (sha256(dataText) !== spec.snapshot_sha256) fail("snapshot hash mismatch");
const namesText = readFileSync(args["probe-names"], "utf8");
if (sha256(namesText) !== spec.probe_names.sha256) fail("probe-name registry hash mismatch");

const T = loadTerms(args["probe-names"]);
const nm = (m) => T.at(`metrics.${spec.metrics[m].name_key}.${spec.metrics[m].name_variant}`);
const CE = nm("ce"), GEN = nm("gen");

const CEF = T.term("gain", "short");
const rows = dataText.split(/\r?\n/).filter(Boolean).map(JSON.parse);
const synthetic = spec.snapshot_is_synthetic || rows.some((r) => r.synthetic);

const mean = (v) => v.reduce((s, x) => s + x, 0) / v.length;
const pearson = (xs, ys) => {
  const mx = mean(xs), my = mean(ys); let a = 0, b = 0, c = 0;
  for (let i = 0; i < xs.length; i++) { a += (xs[i] - mx) * (ys[i] - my); b += (xs[i] - mx) ** 2; c += (ys[i] - my) ** 2; }
  return a / Math.sqrt(b * c);
};
const ranks = (v) => {
  const o = v.map((x, i) => [x, i]).sort((p, q) => p[0] - q[0] || p[1] - q[1]); const r = Array(v.length);
  for (let i = 0; i < o.length;) { let j = i; while (j + 1 < o.length && o[j + 1][0] === o[i][0]) j++; for (let k = i; k <= j; k++) r[o[k][1]] = (i + j) / 2 + 1; i = j + 1; }
  return r;
};
const spearman = (xs, ys) => pearson(ranks(xs), ranks(ys));
const q7 = (v, p) => { const s = [...v].sort((a, b) => a - b); const h = (s.length - 1) * p, lo = Math.floor(h); return s[lo] + (h - lo) * ((s[Math.min(lo + 1, s.length - 1)]) - s[lo]); };
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

const poolIds = spec.pools.map((p) => p.id);
const cell = new Map();
const ceOf = new Map(), panelOf = new Map();
for (const r of rows) {
  if (!poolIds.includes(r.pool)) fail(`row names unknown pool ${r.pool}`);
  if (ceOf.has(r.relation) && ceOf.get(r.relation) !== r.ce) fail(`CE gain for ${r.relation} differs across rows`);
  ceOf.set(r.relation, r.ce); panelOf.set(r.relation, r.panel);
  if (!cell.has(r.pool)) cell.set(r.pool, new Map());
  if (r.repeat !== spec.repeat_id) fail(`row ${r.pool}/${r.relation} is repeat ${r.repeat}; the repeat rule allows repeat ${spec.repeat_id} only`);
  const m = cell.get(r.pool); if (m.has(r.relation)) fail(`more than one run for ${r.pool}/${r.relation}; the repeat rule allows one`);
  m.set(r.relation, [r.gen]);
}
const present = poolIds.filter((p) => cell.has(p));
if (!present.includes(spec.reference_pool)) fail("reference pool has no rows");
const allRel = [...ceOf.keys()];
const cohort = allRel.filter((rel) => present.every((p) => cell.get(p).has(rel)));
const dropped = allRel.filter((rel) => !cohort.includes(rel)).sort();
const gen = (p, rel) => mean(cell.get(p).get(rel));
const ref = spec.reference_pool;
cohort.sort((a, b) => gen(ref, b) - gen(ref, a) || (a < b ? -1 : 1));
const n = cohort.length;
if (n < 5) fail(`only ${n} relations in the fixed cohort`);
const ceV = cohort.map((r) => ceOf.get(r));
const refV = cohort.map((r) => gen(ref, r));
const B = spec.bootstrap.replicates, rng = mulberry32(spec.bootstrap.seed);
const resamples = Array.from({ length: B }, () => Array.from({ length: n }, () => Math.floor(rng() * n)));
const pools = {};
for (const p of present) {
  const g = cohort.map((r) => gen(p, r));
  const bR = [], bS = [], bRho = [];
  for (const idx of resamples) {
    const x = idx.map((i) => ceV[i]), y = idx.map((i) => g[i]), z = idx.map((i) => refV[i]);
    bR.push(pearson(x, y)); bS.push(spearman(x, y)); if (p !== ref) bRho.push(spearman(y, z));
  }
  pools[p] = {
    n, mean_gen: mean(g), median_gen: q7(g, 0.5), pearson_ce_gen: pearson(ceV, g), spearman_ce_gen: spearman(ceV, g),
    pearson_ci: [q7(bR, 0.025), q7(bR, 0.975)], spearman_ci: [q7(bS, 0.025), q7(bS, 0.975)],
    ...(p === ref ? {} : { rank_agreement: spearman(g, refV), rank_agreement_ci: [q7(bRho, 0.025), q7(bRho, 0.975)] }),
  };
}
const aggregate = { figure_id: spec.figure_id, synthetic, snapshot_sha256: spec.snapshot_sha256, probe_names_sha256: spec.probe_names.sha256,
  model: spec.model, reference_pool: ref, pools_present: present, pools_missing: poolIds.filter((p) => !present.includes(p)),
  cohort_relations: n, cohort_order: cohort, dropped_relations: dropped, pools };
writeFileSync(args.aggregate, JSON.stringify(aggregate, null, 2) + "\n");

const W = spec.paper_width_pt;
const INK = "#111827", INK2 = "#4B5563", GRID = "#E5E7EB", AXIS = "#6B7280", PRED = "#0072B2", OUT = "#D55E00";

const setOf = (rel) => REG.source(REG.canonical(rel));
const f = (x) => (Math.round(x * 1000) / 1000).toString();
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const text = (x, y, s, o = {}) => `<text x="${f(x)}" y="${f(y)}" font-size="${o.size || 8}"${o.anchor ? ` text-anchor="${o.anchor}"` : ""}${o.weight ? ` font-weight="${o.weight}"` : ""} fill="${o.fill || INK}"${o.rotate ? ` transform="rotate(${o.rotate} ${f(x)} ${f(y)})"` : ""}>${esc(s)}</text>`;
const line = (x1, y1, x2, y2, o = {}) => `<line x1="${f(x1)}" y1="${f(y1)}" x2="${f(x2)}" y2="${f(y2)}" stroke="${o.stroke || AXIS}" stroke-width="${o.w || 0.5}"${o.dash ? ` stroke-dasharray="${o.dash}"` : ""}/>`;
const fmt = (x, d = 2) => x.toFixed(d).replace(/^(-?)0\./, "$1.");
const poolById = new Map(spec.pools.map((p) => [p.id, p]));
const refName = poolById.get(spec.reference_pool).label;
const plabel = (p) => `${poolById.get(p).label} (${poolById.get(p).n_labels})`;
const watermark = (h, W) => synthetic
  ? `<text x="${f(W / 2)}" y="${f(h / 2)}" font-size="30" font-weight="bold" fill="#9CA3AF" fill-opacity="0.35" text-anchor="middle" transform="rotate(-24 ${f(W / 2)} ${f(h / 2)})">DRAFT-SYNTHETIC</text>`
  : "";
const svgDoc = (h, body, title, W = spec.paper_width_pt) => `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${f(W)}pt" height="${f(h)}pt" viewBox="0 0 ${f(W)} ${f(h)}" font-family="${spec.font_family}">
<title>${esc(title)}${synthetic ? " (DRAFT-SYNTHETIC)" : ""}</title>
<metadata>figure_id=${spec.figure_id}; snapshot_sha256=${spec.snapshot_sha256}; probe_names_sha256=${spec.probe_names.sha256}; status=${spec.status}; synthetic=${synthetic}</metadata>
<rect width="100%" height="100%" fill="#FFFFFF"/>
${body}
${watermark(h, W)}
</svg>
`;
const hex = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
const ramp = (stops) => (t) => {
  t = Math.min(1, Math.max(0, t)); const k = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(k)), u = k - i;
  const a = hex(stops[i]), b = hex(stops[i + 1]);
  return "#" + a.map((v, j) => Math.round(v + (b[j] - v) * u).toString(16).padStart(2, "0")).join("");
};
const genRamp = ramp(["#FFF5EB", "#FDBE85", "#F4772E", "#D55E00", "#7F2704"]);
const ceRamp = ramp(["#F2F7FB", "#9ECAE1", "#4292C6", "#0072B2", "#08306B"]);
const outputs = {};

{
  const top = 74, rowH = Math.max(2.4, Math.min(6, 230 / n)), hh = n * rowH;
  const x0 = 16, panelW = 4, ceX = x0 + panelW + 2, ceW = 12, gx = ceX + ceW + 5, colW = 16.5;
  const out = [];

  out.push(text(x0 - 6, top + hh / 2, `relations, by ${GEN} under animal`, { anchor: "middle", rotate: -90, fill: INK2 }));
  cohort.forEach((rel, i) => {
    const y = top + i * rowH;
    out.push(`<rect x="${f(x0)}" y="${f(y)}" width="${f(panelW)}" height="${f(rowH + 0.05)}" fill="${REG.displayColor(setOf(rel))}"/>`);
    out.push(`<rect x="${f(ceX)}" y="${f(y)}" width="${f(ceW)}" height="${f(rowH + 0.05)}" fill="${ceRamp(ceOf.get(rel))}"/>`);
    present.forEach((p, j) => out.push(`<rect x="${f(gx + j * colW)}" y="${f(y)}" width="${f(colW + 0.05)}" height="${f(rowH + 0.05)}" fill="${genRamp(gen(p, rel))}"/>`));
  });
  out.push(text(ceX + ceW / 2 + 2, top - 4, CEF, { rotate: -55, fill: PRED, weight: "bold" }));
  present.forEach((p, j) => {
    const cx = gx + j * colW + colW / 2;
    out.push(text(cx + 2, top - 4, plabel(p), { rotate: -55, weight: p === ref ? "bold" : undefined }));
  });
  out.push(`<rect x="${f(gx)}" y="${f(top)}" width="${f(present.length * colW)}" height="${f(hh)}" fill="none" stroke="${INK}" stroke-width="0.4"/>`);
  out.push(`<rect x="${f(gx)}" y="${f(top)}" width="${f(colW)}" height="${f(hh)}" fill="none" stroke="${INK}" stroke-width="1"/>`);
  const r1 = top + hh + 10, r2 = r1 + 10;
  out.push(text(gx - 3, r1, "r", { anchor: "end", fill: INK2 }), text(gx - 3, r2, "ρ ref", { anchor: "end", fill: INK2 }));
  present.forEach((p, j) => {
    const cx = gx + j * colW + colW / 2;
    out.push(text(cx, r1, fmt(pools[p].pearson_ce_gen), { anchor: "middle" }));
    out.push(text(cx, r2, p === ref ? "—" : fmt(pools[p].rank_agreement), { anchor: "middle" }));
  });

  const lx = gx + present.length * colW + 12, lw = 8, lh = 60;
  const bar = (x, y, rampFn, label, color) => {
    const o = [];
    for (let k = 0; k < 50; k++) o.push(`<rect x="${f(x)}" y="${f(y + lh - (k + 1) * lh / 50)}" width="${lw}" height="${f(lh / 50 + 0.05)}" fill="${rampFn((k + 0.5) / 50)}"/>`);
    o.push(`<rect x="${f(x)}" y="${f(y)}" width="${lw}" height="${lh}" fill="none" stroke="${AXIS}" stroke-width="0.4"/>`);
    for (const t of [0, 0.5, 1]) o.push(text(x + lw + 2, y + lh - t * lh + 2.8, fmt(t, 1), { fill: INK2 }));
    o.push(text(x, y - 5, label, { fill: color }));
    return o.join("\n");
  };
  out.push(bar(lx, top + 10, genRamp, GEN, OUT), bar(lx, top + 10 + lh + 24, ceRamp, CE, PRED));
  const ky = top + 10 + 2 * lh + 46;
  const sets = ["hernandez", "engineered", "bear"].filter((src) => cohort.some((rel) => setOf(rel) === src));
  sets.forEach((src, k) => out.push(`<rect x="${f(lx)}" y="${f(ky - 6 + 12 * k)}" width="4" height="7" fill="${REG.displayColor(src)}"/>`, text(lx + 7, ky + 12 * k, setName(src))));
  const ly = ky + 12 * sets.length + 4;
  out.push(text(lx, ly, "r: Pearson,", { fill: INK2 }), text(lx, ly + 10, `${CE} vs`, { fill: INK2 }), text(lx, ly + 20, GEN, { fill: INK2 }));
  out.push(text(lx, ly + 32, "ρ ref: Spearman with", { fill: INK2 }), text(lx, ly + 42, "the animal column", { fill: INK2 }));
  outputs.heatmap = svgDoc(Math.max(r2 + 8, ly + 48), out.join("\n"), spec.variants.heatmap.title);
}

function forest(style, v = {}) {
  const W = v.width || spec.paper_width_pt;
  const order = [ref, ...present.filter((p) => p !== ref).sort((a, b) => pools[b].pearson_ce_gen - pools[a].pearson_ce_gen)];

  const narrow = Boolean(v.width);
  const top = v.plot_top || 29, rowH = v.row_h || 9.5, hh = order.length * rowH, GAP = narrow ? 17 : 18, RIGHT = narrow ? 4 : 6;
  const labelW = Math.ceil(Math.max(...order.map((p) => textWidth(plabel(p), 8, p === ref)))) + 3;
  const pw = (W - labelW - 4 - GAP - RIGHT) / 2, x1 = labelW + 4, x2 = x1 + pw + GAP;
  const lowest = Math.min(...order.map((p) => pools[p].pearson_ci[0]), ...order.filter((p) => p !== ref).map((p) => pools[p].rank_agreement_ci[0]));
  const step = v.axis ? v.axis.step : 0.2, labelEvery = v.axis ? v.axis.label_every : 1;
  const lo = v.axis ? v.axis.lo : Math.min(0, Math.floor(lowest * 5) / 5);
  if (lowest < lo) fail(`an interval starts at ${lowest}, below the axis start ${lo}`);
  const X = (x0, v) => x0 + ((v - lo) / (1 - lo)) * pw;
  const out = [];
  const colours = style === "ink" ? [INK, INK] : [PRED, OUT];
  const refCol = style === "ink" ? OUT : INK;
  const REF_DASH = "2 1.5", BAND_OPACITY = 0.1;
  if (style === "band") {
    const [a, b] = pools[ref].pearson_ci;
    out.push(`<rect data-layer="band" data-ref-band="1" data-lo="${a}" data-hi="${b}" x="${f(X(x1, a))}" y="${f(top - 4)}" width="${f(X(x1, b) - X(x1, a))}" height="${f(hh + 4)}" fill="${PRED}" fill-opacity="${BAND_OPACITY}"/>`);
  }

  const between = `between ${CE} and ${GEN}`;
  const headings = [
    [x1, T.term("pearson_r"), textWidth(between, 8, false) <= pw ? between : `of ${CE} and ${GEN}`],
    [x2, T.term("spearman_rho"), `with the ${refName} ordering`],
  ];
  const wrap = (s) => s.split(" ").reduce((ls, w) => {
    const last = ls[ls.length - 1];
    if (last && textWidth(`${last} ${w}`, 8, false) <= pw) ls[ls.length - 1] = `${last} ${w}`; else ls.push(w);
    return ls;
  }, []);
  const relLines = headings.map(([, , rel]) => (narrow ? wrap(rel) : [rel]));
  const headLines = 1 + Math.max(...relLines.map((l) => l.length));
  const headY = (j) => (narrow ? top - 8.5 - (headLines - 1 - j) * 9.5 : 10.5 + j * 9.5);
  if (headY(0) < 10.5 - 1e-9) fail(`the panel headings need ${headLines} lines, more than fit above plot_top ${top}`);
  const tickLabel = (t) => (narrow ? (Math.abs(t - 1) < 1e-9 ? "1" : fmt(t, 2).replace(/0$/, "")) : fmt(t, 1));
  for (const [k, [x0, stat]] of headings.entries()) {
    const fill = style === "ink" ? INK : colours[k];
    out.push(`<text x="${f(x0 + pw / 2)}" y="${f(headY(0))}" font-size="8" text-anchor="middle" font-weight="bold" fill="${fill}" data-panel-heading="${k + 1}" data-heading-line="1">${termWithSymbol(stat, esc)}</text>`);
    relLines[k].forEach((l, j) => out.push(`<text x="${f(x0 + pw / 2)}" y="${f(headY(headLines - relLines[k].length + j))}" font-size="8" text-anchor="middle" fill="${fill}" data-panel-heading="${k + 1}" data-heading-line="${j + 2}">${esc(l)}</text>`));
    for (let i = 0, u = lo; u <= 1 + 1e-9; i += 1, u += step) {
      const t = narrow ? Math.round(u * 1000) / 1000 : u;
      out.push(line(X(x0, t), top - 4, X(x0, t), top + hh, { stroke: GRID, w: 0.35 }).replace("<line ", '<line data-layer="grid" '));
      if (i % labelEvery === 0) out.push(text(X(x0, t), top + hh + 9, tickLabel(t), { anchor: "middle", fill: INK2 }));
    }
    out.push(line(x0, top + hh, x0 + pw, top + hh, { stroke: AXIS }).replace("<line ", `<line data-panel-axis="${k + 1}" `));
  }
  out.push(line(X(x1, pools[ref].pearson_ce_gen), top - 4, X(x1, pools[ref].pearson_ce_gen), top + hh, { stroke: refCol, w: 0.6, dash: REF_DASH }).replace("<line ", '<line data-ref-line="1" '));
  order.forEach((p, i) => {
    const cy = top + i * rowH + rowH / 2, isRef = p === ref;
    const [c1, c2] = isRef ? [refCol, refCol] : colours;
    out.push(text(labelW, cy + 2.8, plabel(p), { anchor: "end", weight: isRef ? "bold" : undefined, fill: isRef ? refCol : INK }));
    const P = pools[p];
    out.push(line(X(x1, P.pearson_ci[0]), cy, X(x1, P.pearson_ci[1]), cy, { stroke: c1, w: 1 }));
    out.push(`<circle cx="${f(X(x1, P.pearson_ce_gen))}" cy="${f(cy)}" r="2.3" fill="${c1}"/>`);
    if (isRef) out.push(text(x2 + 2, cy + 2.8, "reference", { fill: INK2 }));
    else {
      out.push(line(X(x2, P.rank_agreement_ci[0]), cy, X(x2, P.rank_agreement_ci[1]), cy, { stroke: c2, w: 1 }));
      out.push(`<circle cx="${f(X(x2, P.rank_agreement))}" cy="${f(cy)}" r="2.3" fill="${c2}"/>`);
    }
  });

  const barInk = style === "ink" ? INK : INK2;
  const items = [
    { id: "bootstrap_ci", label: T.term("bootstrap_ci"), glyph: (cx, cy) => line(cx - 5, cy, cx + 5, cy, { stroke: barInk, w: 1 }) + `<circle cx="${f(cx)}" cy="${f(cy)}" r="2.3" fill="${barInk}"/>` },
    ...(style === "band" ? [{ id: "reference_interval", label: `${refName} ${T.term("reference_interval")}`,
      glyph: (cx, cy) => `<rect x="${f(cx - 5)}" y="${f(cy - 4)}" width="10" height="8" fill="${PRED}" fill-opacity="${BAND_OPACITY}"/>` }] : []),
    { id: "reference_value", label: `${refName} ${T.term("reference_value")}`, glyph: (cx, cy) => line(cx, cy - 4.5, cx, cy + 4.5, { stroke: refCol, w: 0.6, dash: REF_DASH }) },
  ];

  const keyY = keyBaselineBelow(top + hh + 9, 8);
  const key = keyBlocks([{ id: "marks", title: T.term("key_marks"), items }], { itemsLeft: x1, titleRight: labelW, right: x2 + pw, y: keyY, fs: 8, textWidth, text, clear: true });
  out.push(key.svg);
  return svgDoc(keyBottom(key.lastBaseline, 8), out.join("\n"), "", W);
}

{
  const order = [ref, ...present.filter((p) => p !== ref)];
  const cols = 5, rowsN = Math.ceil(order.length / cols), gap = 8, left = 22, s = (W - left - 10 - gap * (cols - 1)) / cols;
  const topPad = 12, rowPitch = s + topPad + 10, out = [];
  order.forEach((p, k) => {
    const cx0 = left + (k % cols) * (s + gap), cy0 = topPad + Math.floor(k / cols) * rowPitch;
    const X = (v) => cx0 + v * s, Y = (v) => cy0 + s - v * s, isRef = p === ref;
    for (const t of [0, 0.5, 1]) out.push(line(X(t), cy0, X(t), cy0 + s, { stroke: GRID, w: 0.35 }), line(cx0, Y(t), cx0 + s, Y(t), { stroke: GRID, w: 0.35 }));
    out.push(`<rect x="${f(cx0)}" y="${f(cy0)}" width="${f(s)}" height="${f(s)}" fill="none" stroke="${isRef ? OUT : AXIS}" stroke-width="${isRef ? 1 : 0.5}"/>`);
    for (const rel of cohort) out.push(`<circle cx="${f(X(ceOf.get(rel)))}" cy="${f(Y(gen(p, rel)))}" r="1.1" fill="${isRef ? OUT : INK}" fill-opacity="0.55"/>`);
    out.push(text(cx0, cy0 - 3, `${poolById.get(p).label} ${fmt(pools[p].pearson_ce_gen)}`, { weight: isRef ? "bold" : undefined, fill: isRef ? OUT : INK }));
    if (k % cols === 0) for (const t of [0, 1]) out.push(text(cx0 - 2, Y(t) + 2.8, fmt(t, 0), { anchor: "end", fill: INK2 }));
    if (k + cols >= order.length) for (const t of [0, 1]) out.push(text(X(t), cy0 + s + 8.5, fmt(t, 0), { anchor: "middle", fill: INK2 }));
  });
  const h = topPad + rowsN * rowPitch + 8;
  out.push(text(8, h / 2, GEN, { anchor: "middle", rotate: -90 }));
  out.push(text(left + (W - left) / 2, h - 3, `${CE} (one value per relation, identical across label sets)`, { anchor: "middle" }));
  outputs.scatters = svgDoc(h + 2, out.join("\n"), spec.variants.scatters.title);
}

for (const [k, v] of Object.entries(spec.variants)) {
  if ((v.layout || k) !== "forest") continue;
  if (!["ink", "measure", "band"].includes(v.style || "ink")) fail(`variant ${k}: unknown forest style ${v.style}`);
  outputs[k] = forest(v.style || "ink", v).replace("<title></title>", `<title>${esc(v.title)}${synthetic ? " (DRAFT-SYNTHETIC)" : ""}</title>`);
}
for (const [k, v] of Object.entries(spec.variants)) {
  if (!outputs[k]) fail(`no renderer for variant ${k}`);
  writeFileSync(`${v.file}.svg`, outputs[k]);
}
process.stdout.write(`rendered ${Object.keys(outputs).length} variants over ${n} relations x ${present.length} pools${synthetic ? " (SYNTHETIC)" : ""}\n`);
