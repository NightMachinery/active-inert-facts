#!/usr/bin/env node
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { KEY_STYLE, keyBlocks } from "../shared/key.mjs";
import { load as loadTerms } from "../shared/terms.mjs";
import { textWidth } from "../shared/text_width.mjs";
import { halfWidth } from "../shared/half_width.mjs";
import { outputName, readSpec, refusePending } from "./spec-lib.mjs";

const fail = (m) => { process.stderr.write(`render: ${m}\n`); process.exit(1); };
const args = Object.fromEntries(process.argv.slice(2).reduce((p, v, i, a) => (v.startsWith("--") && p.push([v.slice(2), a[i + 1] && !a[i + 1].startsWith("--") ? a[i + 1] : true]), p), []));
if (!args.spec) fail("missing --spec");
const { spec } = readSpec(args.spec, fail);
if (args["list-files"]) {
  process.stdout.write(Object.values(spec.variants).map((v) => outputName(spec, v)).join(" "));
  process.exit(0);
}
for (const k of ["probe-names", "aggregate"]) if (!args[k]) fail(`missing --${k}`);
refusePending(spec, args.spec, fail);
if (!spec.snapshot) fail(`${args.spec} has no frozen snapshot yet; run make freeze first`);
const sha256 = (t) => createHash("sha256").update(t).digest("hex");
const dataText = readFileSync(spec.snapshot, "utf8");
if (sha256(dataText) !== spec.snapshot_sha256) fail("snapshot hash mismatch");
const namesText = readFileSync(args["probe-names"], "utf8");
if (sha256(namesText) !== spec.probe_names.sha256) fail("probe-name registry hash mismatch");

const T = loadTerms(args["probe-names"]);
const genName = T.at(`metrics.${spec.metrics.gen.name_key}.${spec.metrics.gen.name_variant}`);
const rows = dataText.split(/\r?\n/).filter(Boolean).map(JSON.parse);

const mean = (v) => v.reduce((s, x) => s + x, 0) / v.length;
const sd = (v) => { const m = mean(v); return Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / (v.length - 1)); };
const median = (v) => { const s = [...v].sort((a, b) => a - b); const n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
const pearson = (xs, ys) => {
  const mx = mean(xs), my = mean(ys);
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < xs.length; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; syy += (ys[i] - my) ** 2; }
  return sxy / Math.sqrt(sxx * syy);
};
const K = spec.repeats.length;

const q7 = (v, p) => { const s = [...v].sort((a, b) => a - b); const h = (s.length - 1) * p, lo = Math.floor(h); return s[lo] + (h - lo) * (s[Math.min(lo + 1, s.length - 1)] - s[lo]); };
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const pairsOf = (cells) => { const px = [], py = []; for (const c of cells) for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) { px.push(c.ys[i]); py.push(c.ys[j]); } return [px, py]; };
const shareOf = (cs) => {
  const n = cs.length, grand = mean(cs.flatMap((c) => c.ys));
  const msb = K * cs.reduce((s, c) => s + (mean(c.ys) - grand) ** 2, 0) / (n - 1);
  const msw = cs.reduce((s, c) => { const mu = mean(c.ys); return s + c.ys.reduce((t, y) => t + (y - mu) ** 2, 0); }, 0) / (n * (K - 1));
  return 1 - (msb - msw) / (msb + (K - 1) * msw);
};
const cellsOf = (panelId) => {
  const m = new Map();
  for (const r of rows.filter((x) => x.panel === panelId)) {
    const key = `${r.model}\u0000${r.relation}`;
    if (!m.has(key)) m.set(key, { model: r.model, relation: r.relation, n_heldout: r.n_heldout, C: r.n_heldout_classes, inv: r.inv_h_sum, ys: [] });
    m.get(key).ys[spec.repeats.indexOf(r.repeat)] = r.gen;
  }
  return [...m.values()];
};
const aggregate = { figure_id: spec.figure_id, spec: args.spec, snapshot_sha256: spec.snapshot_sha256, probe_names_sha256: spec.probe_names.sha256, panels: {} };
for (const panel of spec.panels) {
  const cells = cellsOf(panel.id);
  if (!cells.length) fail(`panel ${panel.id} has no cells`);
  const px = [], py = [];
  for (const c of cells) for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) { px.push(c.ys[i]); py.push(c.ys[j]); }
  const ratios = [], ratiosMacro = [];
  let degenerate = 0;
  const cellOut = [];
  for (const c of cells) {
    const p = mean(c.ys), s = sd(c.ys);
    const se = Math.sqrt(p * (1 - p) / c.n_heldout);
    const seMacro = Math.sqrt(p * (1 - p) * c.inv) / c.C;
    const o = { model: c.model, relation: c.relation, mean: p, sd: s, se_binomial: se, se_macro: seMacro };
    if (se > 0) { ratios.push(s / se); ratiosMacro.push(s / seMacro); o.ratio = s / se; } else degenerate += 1;
    cellOut.push(o);
  }
  const models = {};
  for (const m of spec.models) {
    const cs = cells.filter((c) => c.model === m.id);
    if (cs.length < 3) continue;
    const n = cs.length, grand = mean(cs.flatMap((c) => c.ys));
    const msb = K * cs.reduce((s, c) => s + (mean(c.ys) - grand) ** 2, 0) / (n - 1);
    const msw = cs.reduce((s, c) => { const mu = mean(c.ys); return s + c.ys.reduce((t, y) => t + (y - mu) ** 2, 0); }, 0) / (n * (K - 1));
    const vb = (msb - msw) / K;
    models[m.id] = { n_relations: n, msb, msw, var_between: vb, var_within: msw,
      icc1: (msb - msw) / (msb + (K - 1) * msw), icck: (msb - msw) / msb, run_to_run_share: msw / (vb + msw) };
  }
  let pairsCi = null;
  if (spec.bootstrap) {
    const rels = [...new Set(cells.map((c) => c.relation))].sort();
    const byRel = new Map(rels.map((r) => [r, cells.filter((c) => c.relation === r)]));
    const rng = mulberry32(spec.bootstrap.seed), n = rels.length;
    const bR = [], bS = Object.fromEntries(Object.keys(models).map((m) => [m, []]));
    for (let b = 0; b < spec.bootstrap.replicates; b++) {
      const pick = Array.from({ length: n }, () => byRel.get(rels[Math.floor(rng() * n)]));
      bR.push(pearson(...pairsOf(pick.flat())));
      for (const m of Object.keys(models)) bS[m].push(shareOf(pick.map((cs) => cs.find((c) => c.model === m))));
    }
    if (![bR, ...Object.values(bS)].every((v) => v.every(Number.isFinite))) fail(`panel ${panel.id}: a bootstrap replicate is not finite`);
    pairsCi = [q7(bR, 0.025), q7(bR, 0.975)];
    for (const m of Object.keys(models)) models[m].run_to_run_share_ci = [q7(bS[m], 0.025), q7(bS[m], 0.975)];
  }
  aggregate.panels[panel.id] = {
    cells: cells.length, relations: new Set(cells.map((c) => c.relation)).size, models: Object.keys(models).length,
    pairs: px.length, pairs_pearson: pearson(px, py), ...(pairsCi ? { pairs_pearson_ci: pairsCi } : {}),
    floor: { cells_with_floor: ratios.length, degenerate_floor_cells: degenerate,
      median_ratio: median(ratios), mean_ratio: ratios.reduce((x, y) => x + y, 0) / ratios.length, max_ratio: Math.max(...ratios),
      at_or_below: ratios.filter((x) => x <= 1).length, above: ratios.filter((x) => x > 1).length,
      median_ratio_macro: median(ratiosMacro), mean_ratio_macro: ratiosMacro.reduce((x, y) => x + y, 0) / ratiosMacro.length, above_macro: ratiosMacro.filter((x) => x > 1).length },
    per_model: models, cell_stats: cellOut,
  };
}
writeFileSync(args.aggregate, JSON.stringify(aggregate, null, 2) + "\n");

const layoutsUsed = new Set(Object.entries(spec.variants).map(([k, v]) => v.layout || k));
const rangeOf = (v) => Math.max(...v) - Math.min(...v);
const ranks = (v) => {
  const idx = v.map((x, i) => [x, i]).sort((a, b) => a[0] - b[0]);
  const out = new Array(v.length);
  for (let i = 0; i < idx.length;) {
    let j = i;
    while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
    for (let k = i; k <= j; k++) out[idx[k][1]] = (i + j) / 2 + 1;
    i = j + 1;
  }
  return out;
};
const spearman = (xs, ys) => pearson(ranks(xs), ranks(ys));
const ci95 = (v) => [q7(v, 0.025), q7(v, 0.975)];
const finite = (what, ...vs) => { if (!vs.every((v) => v.every(Number.isFinite))) fail(`${what}: a bootstrap replicate is not finite`); };

let rangebars = null;
if (layoutsUsed.has("rangebars")) {
  if (!args["aggregate-rangebars"]) fail("missing --aggregate-rangebars");

  const symPearson = (cells) => { const [px, py] = pairsOf(cells); return pearson(px.concat(py), py.concat(px)); };
  const spread = (cells) => { const r = cells.map((c) => rangeOf(c.ys)), s = cells.map((c) => sd(c.ys)); return [median(r), mean(r), median(s), mean(s)]; };
  rangebars = { figure_id: spec.figure_id, spec: args.spec, snapshot_sha256: spec.snapshot_sha256, probe_names_sha256: spec.probe_names.sha256, panels: {} };
  for (const panel of spec.panels) {
    const cells = cellsOf(panel.id);
    const [rMed, rMean, sMed, sMean] = spread(cells);
    const out = { cells: cells.length, pairs_pearson_symmetric: symPearson(cells),
      range: { median: rMed, mean: rMean }, sd: { median: sMed, mean: sMean } };
    if (spec.bootstrap) {
      const rels = [...new Set(cells.map((c) => c.relation))].sort();
      const byRel = new Map(rels.map((r) => [r, cells.filter((c) => c.relation === r)]));
      const rng = mulberry32(spec.bootstrap.seed), n = rels.length;
      const bSym = [], b = [[], [], [], []];
      for (let k = 0; k < spec.bootstrap.replicates; k++) {
        const pick = Array.from({ length: n }, () => byRel.get(rels[Math.floor(rng() * n)])).flat();
        bSym.push(symPearson(pick));
        spread(pick).forEach((v, i) => b[i].push(v));
      }
      finite(`rangebars ${panel.id}`, bSym, ...b);
      out.pairs_pearson_symmetric_ci = ci95(bSym);
      out.range.median_ci = ci95(b[0]); out.range.mean_ci = ci95(b[1]);
      out.sd.median_ci = ci95(b[2]); out.sd.mean_ci = ci95(b[3]);
    }
    out.cell_runs = cells.map((c) => ({ model: c.model, relation: c.relation, runs: c.ys, mean: mean(c.ys), sd: sd(c.ys), range: rangeOf(c.ys) }));
    rangebars.panels[panel.id] = out;
  }
  writeFileSync(args["aggregate-rangebars"], JSON.stringify(rangebars, null, 2) + "\n");
}

let correlation = null;
const SERIES = [...spec.repeats.map(String), "mean"];
const PAIRS = spec.repeats.flatMap((a, i) => spec.repeats.slice(i + 1).map((b) => `${a}-${b}`));
if (layoutsUsed.has("correlation") || layoutsUsed.has("correlation_range")) {
  if (!args["aggregate-correlation"]) fail("missing --aggregate-correlation");
  const C = spec.correlation;
  const predText = readFileSync(C.predictor.path, "utf8");
  if (sha256(predText) !== C.predictor.sha256) fail(`${C.predictor.path}: sha256 differs from spec.correlation.predictor`);
  const refText = readFileSync(C.table1_reference.path, "utf8");
  if (sha256(refText) !== C.table1_reference.sha256) fail(`${C.table1_reference.path}: sha256 differs from spec.correlation.table1_reference`);
  const reference = JSON.parse(refText).models;
  const fold = (s) => s.toLowerCase();
  const xRow = new Map(predText.split(/\r?\n/).filter(Boolean).map(JSON.parse).map((r) => [`${fold(r.model)}\u0000${r.relation}`, r]));

  const stats = (x, ys) => Object.fromEntries(SERIES.map((s, i) => [s, [pearson(x, ys[i]), spearman(x, ys[i])]]));
  const pairStats = (st) => {
    const d = {};
    for (const p of PAIRS) { const [a, b] = p.split("-"); d[p] = [st[a][0] - st[b][0], st[a][1] - st[b][1]]; }
    return [d, Math.max(...PAIRS.map((p) => Math.abs(d[p][0]))), Math.max(...PAIRS.map((p) => Math.abs(d[p][1])))];
  };
  correlation = { figure_id: spec.figure_id, spec: args.spec, snapshot_sha256: spec.snapshot_sha256, probe_names_sha256: spec.probe_names.sha256,
    predictor_sha256: C.predictor.sha256, table1_reference_sha256: C.table1_reference.sha256, series: SERIES, pairs: PAIRS, scopes: {} };
  for (const scope of C.scopes) {
    const cells = scope.panels.flatMap((pid) => { if (!spec.panels.some((p) => p.id === pid)) fail(`scope ${scope.id}: no panel ${pid}`); return cellsOf(pid); });
    const models = spec.models.map((m) => m.id).filter((id) => (scope.models ? scope.models.includes(id) : cells.some((c) => c.model === id)));
    if (scope.models && models.length !== scope.models.length) fail(`scope ${scope.id}: a scope model is not in spec.models`);
    const rels = [...new Set(cells.map((c) => c.relation))].sort();
    const data = {};
    for (const m of models) {
      const byRel = new Map(cells.filter((c) => c.model === m).map((c) => [c.relation, c]));
      if (rels.some((r) => !byRel.has(r))) fail(`scope ${scope.id}: ${m} lacks a relation of the scope`);
      const x = [], ys = SERIES.map(() => []);
      for (const r of rels) {
        const c = byRel.get(r), p = xRow.get(`${fold(m)}\u0000${r}`);
        if (!p) fail(`scope ${scope.id}: no predictor row for ${m} ${r}`);
        if (p[C.predictor.outcome_field] !== c.ys[0]) fail(`scope ${scope.id}: ${m} ${r}: the predictor snapshot's outcome differs from repeat ${spec.repeats[0]}`);
        x.push(p[C.predictor.field]);
        c.ys.forEach((y, i) => ys[i].push(y));
        ys[SERIES.length - 1].push(mean(c.ys));
      }
      data[m] = { x, ys };
    }
    const est = {};
    for (const m of models) {
      const st = stats(data[m].x, data[m].ys), [d, rSpread, rhoSpread] = pairStats(st);
      est[m] = { st, d, rSpread, rhoSpread };
      if (scope.table1_check) {
        const ref = reference[m];
        if (!ref || ref.n !== rels.length || [...ref.relations].sort().join() !== rels.join()) fail(`scope ${scope.id}: ${m}: the Table 1 reference covers other relations`);
        const close = (a, b) => Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(a), Math.abs(b));
        if (!close(st[SERIES[0]][0], ref.gain.r) || !close(st[SERIES[0]][1], ref.gain.rho)) fail(`scope ${scope.id}: ${m}: repeat ${SERIES[0]} does not reproduce Table 1's raw r and rho`);
      }
    }

    const SUMMARY = ["r_spread_median", "r_spread_mean", "rho_spread_median", "rho_spread_mean", ...["r", "rho"].flatMap((c) => ["median", "mean"].flatMap((w) => SERIES.map((s) => `${c}.${w}.${s}`)))];
    const summaryOf = (e) => {
      const r = models.map((m) => e[m].rSpread), s = models.map((m) => e[m].rhoSpread);
      const by = (c, w, sr) => { const v = models.map((m) => e[m].st[sr][c === "r" ? 0 : 1]); return w === "median" ? median(v) : mean(v); };
      return [median(r), mean(r), median(s), mean(s), ...["r", "rho"].flatMap((c) => ["median", "mean"].flatMap((w) => SERIES.map((sr) => by(c, w, sr))))];
    };
    const boot = Object.fromEntries(models.map((m) => [m, { st: Object.fromEntries(SERIES.map((s) => [s, [[], []]])), d: Object.fromEntries(PAIRS.map((p) => [p, [[], []]])), rSpread: [], rhoSpread: [] }]));
    const bSum = SUMMARY.map(() => []);
    if (spec.bootstrap) {
      const rng = mulberry32(spec.bootstrap.seed), n = rels.length;
      for (let k = 0; k < spec.bootstrap.replicates; k++) {
        const pick = Array.from({ length: n }, () => Math.floor(rng() * n));
        const e = {};
        for (const m of models) {
          const x = pick.map((i) => data[m].x[i]), ys = data[m].ys.map((y) => pick.map((i) => y[i]));
          const st = stats(x, ys), [d, rSpread, rhoSpread] = pairStats(st);
          e[m] = { st, rSpread, rhoSpread };
          const B = boot[m];
          for (const s of SERIES) { B.st[s][0].push(st[s][0]); B.st[s][1].push(st[s][1]); }
          for (const p of PAIRS) { B.d[p][0].push(d[p][0]); B.d[p][1].push(d[p][1]); }
          B.rSpread.push(rSpread); B.rhoSpread.push(rhoSpread);
        }
        summaryOf(e).forEach((v, i) => bSum[i].push(v));
      }
      for (const m of models) finite(`correlation ${scope.id} ${m}`, ...SERIES.flatMap((s) => boot[m].st[s]), ...PAIRS.flatMap((p) => boot[m].d[p]), boot[m].rSpread, boot[m].rhoSpread);
      if (models.length > 1) finite(`correlation ${scope.id} over models`, ...bSum);
    }
    const withCi = (v, b) => (spec.bootstrap ? { est: v, ci: ci95(b) } : { est: v });
    const perModel = {};
    for (const m of models) {
      const { st, d, rSpread, rhoSpread } = est[m], B = boot[m];
      perModel[m] = {
        r: Object.fromEntries(SERIES.map((s) => [s, withCi(st[s][0], B.st[s][0])])),
        rho: Object.fromEntries(SERIES.map((s) => [s, withCi(st[s][1], B.st[s][1])])),
        r_diff: Object.fromEntries(PAIRS.map((p) => [p, withCi(d[p][0], B.d[p][0])])),
        rho_diff: Object.fromEntries(PAIRS.map((p) => [p, withCi(d[p][1], B.d[p][1])])),
        r_spread: withCi(rSpread, B.rSpread), rho_spread: withCi(rhoSpread, B.rhoSpread),
      };
      if (scope.table1_check) perModel[m].table1 = { r: reference[m].gain.r, rho: reference[m].gain.rho };
    }
    const out = { panels: scope.panels, relations: rels.length, relation_ids: rels, models, per_model: perModel };
    if (models.length > 1) {
      const v = summaryOf(est), om = {};
      SUMMARY.forEach((k, i) => {
        const path = k.split(".");
        let o = om;
        for (const p of path.slice(0, -1)) o = o[p] ??= {};
        o[path.at(-1)] = withCi(v[i], bSum[i]);
      });
      out.over_models = om;
    }
    correlation.scopes[scope.id] = out;
  }
  writeFileSync(args["aggregate-correlation"], JSON.stringify(correlation, null, 2) + "\n");
}

const W = spec.paper_width_pt;
const INK = "#111827", INK2 = "#4B5563", GRID = "#E5E7EB", AXIS = "#6B7280", WITHIN = "#4B5563";
const f = (x) => (Math.round(x * 1000) / 1000).toString();
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const text = (x, y, s, o = {}) => `<text x="${f(x)}" y="${f(y)}" font-size="${o.size || 8}"${o.anchor ? ` text-anchor="${o.anchor}"` : ""}${o.weight ? ` font-weight="${o.weight}"` : ""} fill="${o.fill || INK}"${o.rotate ? ` transform="rotate(${o.rotate} ${f(x)} ${f(y)})"` : ""}>${esc(s)}</text>`;
const legendLabel = (x, y, str, o = {}) => (str === "" ? "" : text(x, y, str, o).replace("<text ", '<text data-legend="1" '));
const line = (x1, y1, x2, y2, o = {}) => `<line x1="${f(x1)}" y1="${f(y1)}" x2="${f(x2)}" y2="${f(y2)}" stroke="${o.stroke || AXIS}" stroke-width="${o.w || 0.5}"${o.dash ? ` stroke-dasharray="${o.dash}"` : ""}/>`;

const svgDoc = (h0, body, title, Wd = W) => {
  const h = spec.banner ? h0 + 13 : h0;
  const banner = spec.banner ? `<text data-banner="1" x="${f(W / 2)}" y="${f(h - 4)}" font-size="8" text-anchor="middle" fill="#4B5563">${esc(spec.banner)}</text>\n` : "";
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${f(Wd)}pt" height="${f(h)}pt" viewBox="0 0 ${f(Wd)} ${f(h)}" font-family="${spec.font_family}">
<title>${esc(title)}</title>
<metadata>figure_id=${spec.figure_id}; snapshot_sha256=${spec.snapshot_sha256}; probe_names_sha256=${spec.probe_names.sha256}; status=${spec.status}</metadata>
<rect width="100%" height="100%" fill="#FFFFFF"/>
${body}
${banner}</svg>
`;
};
const fmt = (x, d = 2) => x.toFixed(d);
const panelById = new Map(spec.panels.map((p) => [p.id, p]));

let ENC = "color";
let VARIANT = {};
const inkOf = (panel) => (ENC === "neutral" ? panel.neutral.color : panel.color);
const mark = (panel, x, y, r, opacity) => {
  const color = inkOf(panel);
  const style = panel.mark === "ring"
    ? `fill="none" stroke="${color}" stroke-width="0.6" stroke-opacity="${f(Math.min(1, opacity + 0.35))}"`
    : `fill="${color}" fill-opacity="${opacity}" stroke="none"`;
  const tag = `data-panel="${panel.id}" data-mark="${panel.mark}"`;
  if (panel.mark === "triangle") {
    const R = 1.3 * r;
    return `<path ${tag} d="M${f(x)},${f(y - R)} L${f(x + R * 0.95)},${f(y + R * 0.6)} L${f(x - R * 0.95)},${f(y + R * 0.6)} Z" ${style}/>`;
  }
  return `<circle ${tag} cx="${f(x)}" cy="${f(y)}" r="${f(panel.mark === "ring" ? r * 1.05 : r)}" ${style}/>`;
};

const corr = (x, d) => x.toFixed(d).replace(/^(-?)0\./, "$1.");

const legendText = (id) => {
  const a = aggregate.panels[id];
  if (!a.pairs_pearson_ci) return `r = ${corr(a.pairs_pearson, 3)}`;
  return `r = ${corr(a.pairs_pearson, 3)} ± ${halfWidth(a.pairs_pearson, a.pairs_pearson_ci, 2).replace(/^0\./, ".")}`;
};

function pairsAxes(x0, y0, s, panelIds, o = {}) {
  const out = [];
  const X = (v) => x0 + v * s, Y = (v) => y0 + s - v * s;
  for (const t of spec.metrics.gen.ticks) {
    out.push(line(X(t), y0, X(t), y0 + s, { stroke: GRID, w: 0.35 }), line(x0, Y(t), x0 + s, Y(t), { stroke: GRID, w: 0.35 }));
    out.push(text(X(t), y0 + s + 9, fmt(t, t % 0.5 ? 2 : 1), { anchor: "middle", fill: INK2 }));
    if (o.yTicks !== false) out.push(text(x0 - 3, Y(t) + 2.8, fmt(t, t % 0.5 ? 2 : 1), { anchor: "end", fill: INK2 }));
  }
  out.push(`<rect x="${f(x0)}" y="${f(y0)}" width="${f(s)}" height="${f(s)}" fill="none" stroke="${AXIS}" stroke-width="0.5" data-plot-area="1"/>`);
  out.push(line(X(0), Y(0), X(1), Y(1), { stroke: INK2, w: 0.6, dash: "2.5 2" }));
  for (const id of panelIds) {
    const panel = panelById.get(id);
    for (const c of cellsOf(id)) for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) out.push(mark(panel, X(c.ys[i]), Y(c.ys[j]), 1.25, 0.45));
  }
  out.push(text(x0 + s / 2, y0 + s + 20, `${genName}, repeat i`, { anchor: "middle" }));
  if (o.yLabel !== false) out.push(text(x0 - 22, y0 + s / 2, `${genName}, repeat j`, { anchor: "middle", rotate: -90 }));

  if (o.legend === false) return out.join("\n");
  const lines = o.legendLines || 1;
  let ty = y0 - 5 - 10 * (panelIds.length * lines - 1);
  for (const id of panelIds) {
    const panel = panelById.get(id);
    out.push(mark(panel, x0 + 2.5, ty - 2.8, 2, 0.9));
    if (lines === 1) {
      out.push(legendLabel(x0 + 8, ty, `${panel.label}: ${legendText(id)}`));
    } else {
      out.push(legendLabel(x0 + 8, ty, panel.label, { weight: "bold" }), legendLabel(x0 + 8, ty + 10, legendText(id), { fill: INK2 }));
    }
    ty += 10 * lines;
  }
  return out.join("\n");
}

function shareBars(x0, y0, w, panelIds, o = {}) {
  const out = [];
  const labelW = o.labelW ?? 92, numW = 30, barX = x0 + labelW, barW = w - labelW - numW;
  const sub = 5.2, gap = 1.6, rowH = panelIds.length * sub + (panelIds.length - 1) * gap + 5.5;
  const X = (v) => barX + v * barW;
  const models = spec.models.filter((m) => panelIds.some((id) => aggregate.panels[id].per_model[m.id]));
  const top = y0 + 12;
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    out.push(line(X(t), top - 3, X(t), top + models.length * rowH, { stroke: GRID, w: 0.35 }));
    out.push(text(X(t), top + models.length * rowH + 9, `${Math.round(t * 100)}%`, { anchor: "middle", fill: INK2 }));
  }
  out.push(text(X(0.5), top + models.length * rowH + 20, T.term("single_run_share"), { anchor: "middle" }));
  models.forEach((m, k) => {
    const ry = top + k * rowH;
    out.push(text(barX - 4, ry + (rowH - 5.5) / 2 + 3, m.label, { anchor: "end" }));
    panelIds.forEach((id, j) => {
      const pm = aggregate.panels[id].per_model[m.id];
      const by = ry + j * (sub + gap);
      if (!pm) return;
      const panel = panelById.get(id);
      out.push(`<rect x="${f(X(0))}" y="${f(by)}" width="${f(X(pm.icc1) - X(0))}" height="${f(sub)}" fill="${inkOf(panel)}"/>`);
      out.push(`<rect x="${f(X(pm.icc1))}" y="${f(by)}" width="${f(X(1) - X(pm.icc1))}" height="${f(sub)}" fill="${WITHIN}"/>`);
      out.push(text(X(1) + 3, by + sub - 0.6, `${fmt(100 * (1 - pm.icc1), 1)}%`, { fill: INK2, size: 8 }));
    });
  });

  let lx = x0, ly = y0 + 3;
  for (const id of panelIds) {
    const p = panelById.get(id), s = `${T.term("between_relations")}, ${p.label}`;
    out.push(`<rect x="${f(lx)}" y="${f(ly - 5)}" width="7" height="5.2" fill="${inkOf(p)}"/>`, text(lx + 9, ly, s, { fill: INK }));
    lx += 9 + 4.1 * s.length + 8;
  }
  out.push(`<rect x="${f(lx)}" y="${f(ly - 5)}" width="7" height="5.2" fill="${WITHIN}"/>`, text(lx + 9, ly, T.term("run_to_run_share", "legend"), { fill: INK }));
  return { svg: out.join("\n"), height: top - y0 + models.length * rowH + 22 };
}

function shareDots(x0, y0, w, panelIds, o = {}) {
  const out = [];
  const labelW = o.labelW ?? 92, barX = x0 + labelW, barW = w - labelW - 12;
  const models = spec.models.filter((m) => panelIds.some((id) => aggregate.panels[id].per_model[m.id]));

  const shares = models.flatMap((m) => panelIds.map((id) => aggregate.panels[id].per_model[m.id]).filter(Boolean)
    .flatMap((pm) => (o.ci && pm.run_to_run_share_ci ? [1 - pm.icc1, pm.run_to_run_share_ci[1]] : [1 - pm.icc1])));

  const top0 = Math.max(...shares);
  const sqrt = o.scale === "sqrt";
  const step = top0 > 0.4 ? 0.2 : 0.05, hi = sqrt ? Math.min(1, Math.ceil(top0 / 0.2) * 0.2) : Math.max(step * 2, Math.ceil(top0 / step) * step);
  const X = (v) => barX + (sqrt ? Math.sqrt(v / hi) : v / hi) * barW;
  const rowH = o.rowH ?? 13, top = y0 + (o.legendTop === false ? 4 : 14);
  const bottom = top + models.length * rowH;

  const ticks = sqrt ? [0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1].filter((t) => t <= hi + 1e-9) : Array.from({ length: Math.round(hi / step) + 1 }, (_, k) => k * step);
  const labelled = sqrt ? new Set([0, 0.05, 0.2, 0.5, 1]) : null;

  models.forEach((m, k) => out.push(line(X(0), top + k * rowH + rowH / 2, X(hi), top + k * rowH + rowH / 2, { stroke: "#F3F4F6", w: 0.35 }).replace("<line ", '<line data-layer="band" ')));
  for (const t of ticks) {
    out.push(line(X(t), top - 2, X(t), bottom, { stroke: GRID, w: 0.35 }).replace("<line ", '<line data-layer="grid" '));
    if (!labelled || labelled.has(t)) out.push(text(X(t), bottom + 9, `${Math.round(t * 100)}%`, { anchor: "middle", fill: INK2 }));
  }
  out.push(line(X(0), top - 2, X(0), bottom, { stroke: AXIS, w: 0.5 }).replace("<line ", '<line data-layer="grid" '));

  out.push(text(o.axisUnderPanel ? x0 + w / 2 : barX + barW / 2, bottom + 20, `${T.term("run_to_run_share", "axis")}${sqrt ? " (square-root axis)" : ""}`, { anchor: "middle" }));
  models.forEach((m, k) => {
    const cy = top + k * rowH + rowH / 2;
    out.push(text(barX - 5, cy + 2.8, m.label, { anchor: "end" }));
    panelIds.forEach((id, j) => {
      const pm = aggregate.panels[id].per_model[m.id];
      if (!pm) return;
      const dy = (j - (panelIds.length - 1) / 2) * (o.ci ? 3.6 : 3.2);
      const panel = panelById.get(id);
      if (o.ci && pm.run_to_run_share_ci) {
        const [lo, up] = pm.run_to_run_share_ci;
        out.push(line(X(lo), cy + dy, X(up), cy + dy, { stroke: inkOf(panel), w: 0.7 }).replace("<line ", `<line data-ci="${panel.id}" data-model="${m.id}" data-lo="${lo}" data-hi="${up}" `));
      }
      out.push(mark(panel, X(1 - pm.icc1), cy + dy, 2.3, 0.95));
    });
  });
  if (o.legendTop !== false) {
    let lx = barX, ly = y0 + 5;
    for (const id of panelIds) {
      const p = panelById.get(id);
      out.push(mark(p, lx + 2.5, ly - 2.8, 2.3, 0.95), text(lx + 8, ly, p.label));
      lx += 8 + 4.3 * p.label.length + 10;
    }
  }
  return { svg: out.join("\n"), height: bottom - y0 + 24 };
}

function floorPlot(x0, y0, w, h, panelIds) {
  const out = [];
  const all = panelIds.flatMap((id) => aggregate.panels[id].cell_stats.filter((c) => c.se_binomial > 0));
  const hi = Math.ceil(Math.max(...all.flatMap((c) => [c.sd, c.se_binomial])) * 50) / 50;
  const X = (v) => x0 + (v / hi) * w, Y = (v) => y0 + h - (v / hi) * h;
  out.push(`<path d="M${f(X(0))},${f(Y(0))} L${f(X(hi))},${f(Y(0))} L${f(X(hi))},${f(Y(hi))} Z" fill="#F3F4F6"/>`);
  for (let t = 0; t <= hi + 1e-9; t += 0.02) {
    out.push(line(X(t), y0, X(t), y0 + h, { stroke: GRID, w: 0.35 }), line(x0, Y(t), x0 + w, Y(t), { stroke: GRID, w: 0.35 }));
    out.push(text(X(t), y0 + h + 9, fmt(t), { anchor: "middle", fill: INK2 }), text(x0 - 3, Y(t) + 2.8, fmt(t), { anchor: "end", fill: INK2 }));
  }
  out.push(`<rect x="${f(x0)}" y="${f(y0)}" width="${f(w)}" height="${f(h)}" fill="none" stroke="${AXIS}" stroke-width="0.5"/>`);
  out.push(line(X(0), Y(0), X(hi), Y(hi), { stroke: INK2, w: 0.6, dash: "2.5 2" }));
  for (const id of panelIds) for (const c of aggregate.panels[id].cell_stats) if (c.se_binomial > 0) out.push(mark(panelById.get(id), X(c.se_binomial), Y(c.sd), 1.5, 0.6));
  out.push(text(X(hi) - 4, Y(0) - 5, "spread within binomial noise", { anchor: "end", fill: INK2 }));
  out.push(text(x0 + w / 2, y0 + h + 20, "binomial s.e. of the cell's held-out set", { anchor: "middle" }));
  out.push(text(x0 - 26, y0 + h / 2, `across-repeat s.d. of ${genName}`, { anchor: "middle", rotate: -90 }));
  let ty = y0 + 10;
  for (const id of panelIds) {
    const a = aggregate.panels[id].floor, p = panelById.get(id);
    out.push(mark(p, x0 + 7, ty - 2.8, 2, 0.9), text(x0 + 12, ty, `${p.label}: ${a.at_or_below} of ${a.cells_with_floor} cells at or below; median ratio ${fmt(a.median_ratio)}, mean ${fmt(a.mean_ratio)}`));
    ty += 10;
  }
  return out.join("\n");
}

function comboKey(x0, bx, zeroX, y, ci) {
  const opts = { fs: 8, textWidth, text: legendLabel };

  const flush = (left, halfGlyph) => left + halfGlyph - KEY_STYLE.glyphW / 2 - KEY_STYLE.titleGap;
  const R = 2.5;
  const setItems = ids.map((id) => {
    const panel = panelById.get(id);
    return { id: `set:${id}`, label: `${panel.label}: ${legendText(id)}`, glyph: (cx, cy) => mark(panel, cx, cy, 2, 0.9) };
  });
  const setTitle = keyBlocks([{ id: "set", title: T.term("key_relation_set"), items: [] }], { ...opts, itemsLeft: x0, right: bx - 2, y });
  const setList = keyBlocks([{ id: "set-items", title: "", stack: true, items: setItems }], { ...opts, itemsLeft: flush(x0, R), right: bx - 2, y: y + KEY_STYLE.rowH });
  const parts = [setTitle.svg, setList.svg];
  let height = setTitle.height + setList.height;
  if (ci) {
    const whisker = (cx, cy) => line(cx - KEY_STYLE.glyphW / 2, cy, cx + KEY_STYLE.glyphW / 2, cy, { stroke: INK2, w: 0.7 }) +
      `<circle cx="${f(cx)}" cy="${f(cy)}" r="2.3" fill="${INK2}"/>`;
    const text2 = (x, yy, str, o = {}) => (str === "" ? "" : text(x, yy, str, o));
    const mTitle = keyBlocks([{ id: "marks", title: T.term("key_marks"), items: [] }], { ...opts, text: text2, itemsLeft: zeroX, right: W - 2, y });
    const mList = keyBlocks([{ id: "marks-items", title: "", items: [{ id: "bootstrap_ci", label: T.term("bootstrap_ci"), glyph: whisker }] }],
      { ...opts, text: text2, itemsLeft: flush(zeroX, KEY_STYLE.glyphW / 2), right: W - 2, y: y + KEY_STYLE.rowH });
    parts.push(mTitle.svg, mList.svg);
    height = Math.max(height, mTitle.height + mList.height);
  }

  return { svg: parts.join("\n"), height: height - KEY_STYLE.rowH };
}

const PRED = "#0072B2", OTHER = "#9CA3AF";
function rangeStrip(x0, y0, w, h, groups, o = {}) {
  const out = [];
  const Y = (v) => y0 + h - v * h;
  for (const t of spec.metrics.gen.ticks) {
    out.push(line(x0, Y(t), x0 + w, Y(t), { stroke: t === 0 ? AXIS : GRID, w: t === 0 ? 0.5 : 0.35 }));
    if (o.yTicks) out.push(text(x0 - 3, Y(t) + 2.8, fmt(t, t % 0.5 ? 2 : 1), { anchor: "end", fill: INK2 }));
  }
  const slots = groups.reduce((s, g) => s + g.cells.length, 0) + 1.5 * (groups.length - 1);
  const pitch = w / slots;
  let k = 0;
  const bars = [], ticks = [], marks = [];
  for (const g of groups) {
    const color = inkOf(g.panel);
    for (const c of [...g.cells].sort((a, b) => mean(a.ys) - mean(b.ys) || a.relation.localeCompare(b.relation))) {
      const cx = x0 + (k + 0.5) * pitch, lo = Math.min(...c.ys), hi = Math.max(...c.ys), mu = mean(c.ys);
      bars.push(line(cx, Y(lo), cx, Y(hi), { stroke: color, w: 1.1 }).replace("<line ", `<line data-range="${g.panel.id}" data-model="${c.model}" data-relation="${c.relation}" data-lo="${lo}" data-hi="${hi}" stroke-opacity="0.55" `));
      ticks.push(line(cx - 1.5, Y(mu), cx + 1.5, Y(mu), { stroke: INK, w: 0.7 }).replace("<line ", `<line data-mean="${g.panel.id}" data-v="${mu}" `));
      for (const y of c.ys) marks.push(mark(g.panel, cx, Y(y), 1.05, 0.8));
      k += 1;
    }
    k += 1.5;
  }
  out.push(...bars, ...ticks, ...marks);
  return out.join("\n");
}

function rangeKey(x0, y) {
  const opts = { fs: 8, textWidth, text: legendLabel, itemsLeft: x0, right: W - 2, y };
  const seen = new Set();
  const setItems = spec.panels.filter((p) => !seen.has(p.source) && seen.add(p.source))
    .map((p) => ({ id: `set:${p.source}`, label: T.setName(p.source), glyph: (cx, cy) => mark(p, cx, cy, 2, 0.9) }));
  const runs = (cx, cy) => line(cx, cy - 4, cx, cy + 4, { stroke: INK2, w: 1.1 }).replace("<line ", '<line stroke-opacity="0.55" ') +
    [-4, 1.2, 4].map((dy) => `<circle cx="${f(cx)}" cy="${f(cy + dy)}" r="1.05" fill="${INK2}"/>`).join("");
  const tick = (cx, cy) => line(cx - 1.5, cy, cx + 1.5, cy, { stroke: INK, w: 0.7 });
  return keyBlocks([
    { id: "set", title: T.term("key_relation_set"), items: setItems },
    { id: "marks", title: T.term("key_marks"), items: [
      { id: "runs", label: "the three runs of one cell, lowest to highest", glyph: runs },
      { id: "mean", label: "their mean", glyph: tick }] },
  ], opts);
}

function rangebarsFigure() {
  const keyY = 10, key = rangeKey(34, keyY);
  const x0 = 34, cols = 4, gapX = 10, cw = (W - x0 - 2 - gapX * (cols - 1)) / cols;
  const ph = 52, titleH = 11, rowGap = 9, top = keyY + key.height + 4;
  const out = [key.svg];
  const models = spec.models.filter((m) => ids.some((id) => cellsOf(id).some((c) => c.model === m.id)));
  const strips = [];
  const setPanels = spec.panels.filter((p) => !p.models);
  const onePanels = spec.panels.filter((p) => p.models);
  for (const m of models) strips.push({ title: m.label, span: 1, groups: setPanels.map((p) => ({ panel: p, cells: cellsOf(p.id).filter((c) => c.model === m.id) })).filter((g) => g.cells.length) });
  for (const p of onePanels) strips.push({ title: p.label, span: 2, groups: [{ panel: p, cells: cellsOf(p.id) }] });
  let slot = 0, rows = 0;
  for (const s of strips) {
    if ((slot % cols) + s.span > cols) slot += cols - (slot % cols);
    const col = slot % cols, row = Math.floor(slot / cols);
    const sx = x0 + col * (cw + gapX), sy = top + row * (titleH + ph + rowGap) + titleH, sw = s.span * cw + (s.span - 1) * gapX;
    out.push(text(sx, sy - 4, s.title).replace("<text ", '<text data-strip-title="1" '));
    out.push(rangeStrip(sx, sy, sw, ph, s.groups, { yTicks: col === 0 }));
    slot += s.span;
    rows = row + 1;
  }
  const bottom = top + rows * (titleH + ph + rowGap) - rowGap;
  out.push(text(x0 + (W - x0) / 2, bottom + 11, T.term("rank_by_outcome"), { anchor: "middle" }));
  out.push(text(9, top + (bottom - top) / 2, genName, { anchor: "middle", rotate: -90 }));
  return [bottom + 16, out.join("\n")];
}

const REPEAT_COLORS = ["#0072B2", "#D55E00", "#111827"];
function correlationFigure(simple, o = {}) {
  const byRepeat = o.repeatColors === true;
  const C = correlation;
  const gainName = T.at(`metrics.${spec.correlation.predictor.name_key}.plain`);
  const keyY = 10;
  const dot = (color) => (cx, cy) => `<circle cx="${f(cx)}" cy="${f(cy)}" r="2" fill="${color}"/>`;
  const diamond = (cx, cy, r, color) => `<path d="M${f(cx)},${f(cy - r)} L${f(cx + r)},${f(cy)} L${f(cx)},${f(cy + r)} L${f(cx - r)},${f(cy)} Z" fill="${color}"`;
  const items = byRepeat ? spec.repeats.map((r, i) => ({ id: `repeat${r}`, label: `repeat ${r}`, glyph: dot(REPEAT_COLORS[i]) })) : [
    { id: "repeat0", label: `repeat ${spec.repeats[0]}`, glyph: (cx, cy) => (simple ? "" : line(cx, cy - 4.5, cx, cy + 4.5, { stroke: PRED, w: 0.5, dash: "1.5 1.2" })) + dot(PRED)(cx, cy) },
    { id: "repeats", label: `repeats ${spec.repeats.slice(1).join(" and ")}`, glyph: dot(OTHER) },
  ];
  if (!simple) {
    items.push({ id: "mean", label: "mean of the three runs", glyph: (cx, cy) => diamond(cx, cy, 2.4, INK) + "/>" });
    items.push({ id: "ci", label: T.term("bootstrap_ci"), glyph: (cx, cy) => line(cx - 5, cy, cx + 5, cy, { stroke: INK2, w: 0.6 }) + `<circle cx="${f(cx)}" cy="${f(cy)}" r="1.4" fill="${INK2}"/>` });
  }
  const key = keyBlocks([{ id: "marks", title: T.term("key_marks"), items }], { fs: 8, textWidth, text: legendLabel, itemsLeft: 2, right: W - 2, y: keyY });
  const labelW = 88, gap = 20, pw = (W - 2 - labelW - gap - 6) / 2;
  const px = [2 + labelW, 2 + labelW + pw + gap];
  const stats = ["r", "rho"];
  const series = simple ? SERIES.filter((s) => s !== "mean") : SERIES;

  const vals = Object.values(C.scopes).flatMap((sc) => Object.values(sc.per_model).flatMap((pm) => stats.flatMap((st) => series.flatMap((s) => (simple ? [pm[st][s].est] : [pm[st][s].est, ...(pm[st][s].ci || [])])))));
  const [lo, hi] = o.domain || [Math.floor(Math.min(...vals) * 10) / 10, 1];
  const X = (k, v) => px[k] + ((v - lo) / (hi - lo)) * pw;
  const rowH = byRepeat ? 11 : simple ? 10 : 15, headH = 12, scopeGap = 6;
  const colTitleY = keyY + key.height + 6;
  let y = colTitleY + 6;
  const bands = [], grid = [], marks = [], labels = [];
  stats.forEach((st, k) => labels.push(text(px[k] + pw / 2, colTitleY, T.term(st === "r" ? "pearson_r" : "spearman_rho"), { anchor: "middle", weight: 700 })));
  const rowSpans = [];
  for (const [sid, sc] of Object.entries(C.scopes)) {
    y += scopeGap;
    const names = [...new Set(sc.panels.map((pid) => panelById.get(pid).source))].map((s) => T.setName(s));
    labels.push(text(2, y + 8, names.join(" + "), { weight: 700 }).replace("<text ", `<text data-scope="${sid}" `));
    y += headH;
    const top = y;
    for (const m of sc.models) {
      const cy = y + rowH / 2, pm = sc.per_model[m];
      bands.push(...stats.map((_, k) => line(px[k], cy, px[k] + pw, cy, { stroke: "#F3F4F6", w: 0.35 }).replace("<line ", '<line data-layer="band" ')));
      labels.push(text(px[0] - 5, cy + 2.8, spec.models.find((x) => x.id === m).label, { anchor: "end" }));
      stats.forEach((st, k) => {
        const r0 = pm[st][series[0]].est;
        if (!simple) marks.push(line(X(k, r0), y + 0.8, X(k, r0), y + rowH - 0.8, { stroke: PRED, w: 0.5, dash: "1.5 1.2" }).replace("<line ", `<line data-ref="${sid}:${m}:${st}" data-v="${r0}" `));
        [...series.entries()].reverse().forEach(([i, s]) => {
          const v = pm[st][s], dy = simple && !byRepeat ? 0 : (i - (series.length - 1) / 2) * (byRepeat ? 2.6 : 3.1);
          const color = s === "mean" ? INK : byRepeat ? REPEAT_COLORS[i] : i === 0 ? PRED : OTHER;
          const tag = `data-corr="${sid}:${m}:${st}:${s}" data-v="${v.est}"`;
          if (!simple && v.ci) marks.push(line(X(k, v.ci[0]), cy + dy, X(k, v.ci[1]), cy + dy, { stroke: color, w: 0.6 }).replace("<line ", `<line data-ci="${sid}:${m}:${st}:${s}" data-lo="${v.ci[0]}" data-hi="${v.ci[1]}" `));
          marks.push(s === "mean" ? `${diamond(X(k, v.est), cy + dy, 1.9, INK)} ${tag}/>` : `<circle ${tag} cx="${f(X(k, v.est))}" cy="${f(cy + dy)}" r="${byRepeat ? 1.6 : simple ? 1.9 : 1.4}" fill="${color}"${simple && i > 0 ? ' fill-opacity="0.85"' : ""}/>`);
        });
      });
      y += rowH;
    }
    rowSpans.push([top, y]);
  }
  const bottom = y + 2;
  const ticks = [];
  for (let t = Math.ceil(lo * 10 - 1e-9) / 10; t <= hi + 1e-9; t = Math.round((t + 0.1) * 10) / 10) ticks.push(t);
  stats.forEach((_, k) => {
    for (const t of ticks) {
      for (const [a, b] of rowSpans) grid.push(line(X(k, t), a, X(k, t), b, { stroke: t === 1 ? AXIS : GRID, w: 0.35 }).replace("<line ", '<line data-layer="grid" '));

      if (hi - lo < 0.8 || Math.round(t * 10) % 2 === 0) labels.push(text(X(k, t), bottom + 8, t === 1 ? "1" : t === 0 ? "0" : corr(t, 1), { anchor: "middle", fill: INK2 }));
    }
  });
  labels.push(text(px[0] + (px[1] + pw - px[0]) / 2, bottom + 19, `correlation of ${gainName} with ${genName} across relations`, { anchor: "middle" }));
  return [bottom + 24, [key.svg, ...bands, ...grid, ...marks, ...labels].join("\n")];
}

const COEF_COLORS = { r: "#0072B2", rho: "#D55E00" };
function correlationRange(Wd, domain) {
  const C = correlation, stats = ["r", "rho"], reps = SERIES.filter((s) => s !== "mean");
  const gainName = T.at(`metrics.${spec.correlation.predictor.name_key}.plain`);
  const bar = (color) => (cx, cy) => line(cx - 5, cy, cx + 5, cy, { stroke: color, w: 1.3 }).replace("<line ", '<line stroke-opacity="0.55" ') +
    [-4, -0.5, 4].map((dx) => `<circle cx="${f(cx + dx)}" cy="${f(cy)}" r="1.25" fill="${color}"/>`).join("");
  const key = keyBlocks([{ id: "coef", title: "", items: stats.map((st) => ({ id: st, label: T.term(st === "r" ? "pearson_r" : "spearman_rho"), glyph: bar(COEF_COLORS[st]) })) }],
    { fs: 8, textWidth, text: legendLabel, itemsLeft: 2 - KEY_STYLE.titleGap, right: Wd - 2, y: 9 });
  const labelW = Math.max(...spec.models.map((m) => textWidth(m.label, 8))) + 6;
  const x0 = 2 + labelW, pw = Wd - x0 - 6;
  const [lo, hi] = domain;
  const X = (v) => x0 + ((v - lo) / (hi - lo)) * pw;
  const rowH = 10, headH = 10, top0 = 9 + key.height - 5;
  let y = top0;
  const bands = [], grid = [], marks = [], labels = [], spans = [];
  for (const [sid, sc] of Object.entries(C.scopes)) {
    const names = [...new Set(sc.panels.map((pid) => panelById.get(pid).source))].map((s) => T.setName(s));
    labels.push(text(2, y + 7.5, names.join(" + "), { weight: 700 }).replace("<text ", `<text data-scope="${sid}" `));
    y += headH;
    const top = y;
    for (const m of sc.models) {
      const cy = y + rowH / 2, pm = sc.per_model[m];
      bands.push(line(x0, cy, x0 + pw, cy, { stroke: "#F3F4F6", w: 0.35 }).replace("<line ", '<line data-layer="band" '));
      labels.push(text(x0 - 4, cy + 2.8, spec.models.find((x) => x.id === m).label, { anchor: "end" }));
      stats.forEach((st, k) => {
        const vs = reps.map((s) => pm[st][s].est), dy = (k - 0.5) * 3.4, color = COEF_COLORS[st];
        if (vs.some((v) => v < lo || v > hi)) fail(`correlation_range: ${m} ${st} outside the axis ${lo}..${hi}`);
        marks.push(line(X(Math.min(...vs)), cy + dy, X(Math.max(...vs)), cy + dy, { stroke: color, w: 1.3 })
          .replace("<line ", `<line data-corr-range="${sid}:${m}:${st}" data-lo="${Math.min(...vs)}" data-hi="${Math.max(...vs)}" stroke-opacity="0.55" `));
        reps.forEach((s, i) => marks.push(`<circle data-corr="${sid}:${m}:${st}:${s}" data-v="${vs[i]}" cx="${f(X(vs[i]))}" cy="${f(cy + dy)}" r="1.25" fill="${color}"/>`));
      });
      y += rowH;
    }
    spans.push([top, y]);
    y += 3;
  }
  const bottom = y - 1;

  const wide = hi - lo > 0.5, step = wide ? 0.1 : 0.05, labelStep = wide ? 0.2 : 0.1;
  for (let k = 0; lo + k * step <= hi + 1e-9; k++) {
    const t = Math.round((lo + k * step) * 100) / 100;
    const major = Math.abs(t / labelStep - Math.round(t / labelStep)) < 1e-9;
    for (const [a, b] of spans) grid.push(line(X(t), a, X(t), b, { stroke: t === 1 ? AXIS : GRID, w: major ? 0.35 : 0.25 }).replace("<line ", '<line data-layer="grid" '));
    if (major) labels.push(text(X(t), bottom + 8, t === 1 ? "1" : t === 0 ? "0" : corr(t, 1), { anchor: "middle", fill: INK2 }));
  }
  labels.push(text(x0 + pw / 2, bottom + 18, `${gainName} vs ${genName}`, { anchor: "middle" }));
  return [bottom + 22, [key.svg, ...bands, ...grid, ...marks, ...labels].join("\n")];
}

const ids = spec.panels.map((p) => p.id);
const outputs = {};
const renderers = {
  pairs() {
    const gapX = 30, x0 = 38, y0 = 24, s = Math.min(150, (W - x0 - 12 - gapX * (ids.length - 1)) / ids.length);
    return [y0 + s + 26, ids.map((id, k) => pairsAxes(x0 + k * (s + gapX), y0, s, [id], { yLabel: k === 0, yTicks: true, legendLines: 2 })).join("\n")];
  },
  floor() { const x0 = 42, y0 = 8, w = W - x0 - 10, h = 190; return [y0 + h + 26, floorPlot(x0, y0, w, h, ids)]; },
  share() { const b = shareBars(4, 4, W - 8, ids); return [b.height + 8, b.svg]; },
  sharedots() { const b = shareDots(4, 2, W - 8, ids, { ci: VARIANT.ci === true }); return [b.height + 6, b.svg]; },
  combo() {
    const s = 150, x0 = 34, bx = x0 + s + 8, labelW = 84, ci = VARIANT.ci === true;

    const keyY = 10, key = comboKey(x0, bx, bx + labelW, keyY, ci && Boolean(spec.bootstrap));
    const y0 = keyY + key.height + 7;
    const left = pairsAxes(x0, y0, s, ids, { legend: false });
    const b = shareDots(bx, y0 - 4, W - bx - 2, ids, { labelW, rowH: (s - 4) / spec.models.length, legendTop: false, ci, scale: VARIANT.scale, axisUnderPanel: true });
    const tags = text(2, keyY, "a", { weight: "bold", size: 9 }).replace("<text ", '<text data-panel-tag="a" ') +
      text(bx + 2, keyY, "b", { weight: "bold", size: 9 }).replace("<text ", '<text data-panel-tag="b" ');
    return [Math.max(y0 + s + 26, b.height + 6), [key.svg, left, b.svg, tags].join("\n")];
  },
  rangebars: rangebarsFigure,
  correlation_range() { return correlationRange(VARIANT.width_pt, VARIANT.domain); },
  correlation() { return correlationFigure(VARIANT.simple === true, { domain: VARIANT.domain, repeatColors: VARIANT.repeat_colors }); },
};
for (const [k, v] of Object.entries(spec.variants)) {
  const r = renderers[v.layout || k];
  if (!r) fail(`no renderer for variant ${k}`);
  ENC = v.encoding || "color";
  VARIANT = v;
  if (ENC === "neutral" && spec.panels.some((p) => !p.neutral)) fail(`variant ${k} is neutral but a panel lacks "neutral"`);
  const [h, body] = r();
  outputs[k] = svgDoc(h, body, v.title, v.width_pt || W);
}
for (const [k, v] of Object.entries(spec.variants)) {
  if (!outputs[k]) fail(`no renderer for variant ${k}`);
  writeFileSync(`${outputName(spec, v)}.svg`, outputs[k]);
}
process.stdout.write(`rendered ${Object.keys(outputs).length} variants\n`);
