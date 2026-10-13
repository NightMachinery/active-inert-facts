#!/usr/bin/env node
import { createHash } from "node:crypto";
import { mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { halfWidth } from "../shared/half_width.mjs";
import { KEY_SPACING, keyBlocks, keyBaselineBelow, keyBottom } from "../shared/key.mjs";
import { mathSym } from "../shared/math_text.mjs";
import { load as loadTerms } from "../shared/terms.mjs";
import { textWidth as tw } from "../shared/text_width.mjs";

const fail = (m) => { process.stderr.write(`render: ${m}\n`); process.exit(1); };
const args = Object.fromEntries(process.argv.slice(2).reduce((p, v, i, a) => (v.startsWith("--") && p.push([v.slice(2), a[i + 1] && !a[i + 1].startsWith("--") ? a[i + 1] : true]), p), []));
if (!args.spec) fail("missing --spec");
const spec = JSON.parse(readFileSync(args.spec, "utf8"));
const outName = (v) => spec.output_prefix + (v.file ? `-${v.file}` : "");
if (args["list-files"]) {
  process.stdout.write(spec.tracked.map((id) => outName(spec.variants[id])).join(" "));
  process.exit(0);
}
for (const k of ["probe-names", "aggregate"]) if (!args[k]) fail(`missing --${k}`);
if (spec.snapshot === "PENDING") fail("spec has no frozen snapshot yet; run freeze.py first");
const sha256 = (t) => createHash("sha256").update(t).digest("hex");
const snapText = readFileSync(spec.snapshot, "utf8");
if (sha256(snapText) !== spec.snapshot_sha256) fail("snapshot hash mismatch");
const namesText = readFileSync(args["probe-names"], "utf8");
if (sha256(namesText) !== spec.probe_names.sha256) fail("probe-name registry hash mismatch");
const T = loadTerms(args["probe-names"]);
const S = JSON.parse(snapText);
if (S.source_sha256 !== spec.source.sha256) fail("snapshot was frozen from another source");

const KEYS = spec.families.flatMap((fm) => fm.items.map((it) => it.key));
if (JSON.stringify([...KEYS, "unexplained"]) !== JSON.stringify(S.order)) fail("spec segment order differs from the snapshot's order");
const median = (v) => { const s = [...v].sort((a, b) => a - b); const n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
const mean = (v) => v.reduce((s, x) => s + x, 0) / v.length;
const explained = (D, field, ciField) => ({
  v: 1 - D[field].unexplained,
  ci: ciField ? [1 - D[ciField].unexplained[1], 1 - D[ciField].unexplained[0]] : null,
});
const P = S.pooled;
const agg = { figure_id: spec.figure_id, snapshot_sha256: spec.snapshot_sha256, probe_names_sha256: spec.probe_names.sha256,
  n_err: P.n_err, n_cells: P.n_cells, bars: {}, families: {}, per_model: {}, over_models: {} };
for (const b of spec.bars) {
  const sum = KEYS.reduce((s, k) => s + P[b.field][k], 0) + P[b.field].unexplained;
  if (Math.abs(sum - 1) > 1e-9) fail(`bar ${b.id} shares sum to ${sum}`);
  agg.bars[b.id] = { explained: explained(P, b.field, b.ci), segments: Object.fromEntries([...KEYS, "unexplained"].map((k) => [k, P[b.field][k]])) };
}
for (const fm of spec.families) {
  agg.families[fm.id] = Object.fromEntries(spec.bars.map((b) => [b.id, fm.items.reduce((s, it) => s + P[b.field][it.key], 0)]));
}
for (const m of spec.models) {
  const D = S.by_model[m.id];
  if (!D) fail(`no by_model entry for ${m.id}`);
  agg.per_model[m.id] = { observed: explained(D, "share", "ci"), shuffle: explained(D, "shuffle", "shuffle_ci"), n_err: D.n_err, n_cells: D.n_cells };
}
if (Object.keys(S.by_model).length !== spec.models.length) fail("the snapshot has models the spec does not list");
for (const kind of ["observed", "shuffle"]) {
  const v = spec.models.map((m) => agg.per_model[m.id][kind].v);
  agg.over_models[kind] = { median: median(v), mean: mean(v), min: Math.min(...v), max: Math.max(...v) };
}
writeFileSync(args.aggregate, JSON.stringify(agg, null, 2) + "\n");

const W = spec.paper_width_pt, FS = spec.min_font_pt, ST = spec.style;
const f = (x) => (Math.round(x * 1000) / 1000).toString();
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

const rich = (s) => esc(s).replace(/\{T\}/g, mathSym("T"));
const plainOf = (s) => s.replace(/\{T\}/g, "T");
const width = (s, bold = false, size = FS) => tw(plainOf(s), size, bold);
const fill = (s) => s.replace(/\{(gain|recall)\}/g, (_, k) => T.term(k));
const text = (x, y, s, o = {}) => `<text x="${f(x)}" y="${f(y)}" font-size="${o.size || FS}"${o.anchor ? ` text-anchor="${o.anchor}"` : ""}${o.weight ? ` font-weight="${o.weight}"` : ""} fill="${o.fill || ST.ink}"${o.attr ? ` ${o.attr}` : ""}>${rich(s)}</text>`;
const line = (x1, y1, x2, y2, o = {}) => `<line x1="${f(x1)}" y1="${f(y1)}" x2="${f(x2)}" y2="${f(y2)}" stroke="${o.stroke || ST.axis}" stroke-width="${o.w || 0.5}"${o.dash ? ` stroke-dasharray="${o.dash}"` : ""}${o.attr ? ` ${o.attr}` : ""}/>`;
const pct = (v) => `${(100 * v).toFixed(1)}%`;

const pm = (v, c) => `± ${halfWidth(v, c, 1, 100)}`;
const tickLabel = (t, last) => `${Math.round(100 * t)}${last ? "%" : ""}`;
const colorOf = Object.fromEntries(spec.families.flatMap((fm) => fm.items.map((it) => [it.key, it.color])));
colorOf.unexplained = ST.unexplained;
const CAP = 0.729 * FS;
const midText = (cy) => cy + CAP / 2;
const ROW = 9.5;

const dotObs = (x, y, color, attr) => `<circle cx="${f(x)}" cy="${f(y)}" r="2.4" fill="${color}" stroke="${ST.ink}" stroke-width="0.3" ${attr}/>`;
const dotShuf = (x, y, attr) => `<circle cx="${f(x)}" cy="${f(y)}" r="2.1" fill="#FFFFFF" stroke="${ST.shuffle_mark}" stroke-width="0.7" ${attr}/>`;
const whisker = (x1, x2, y, color, attr, w = 0.7) => line(x1, y, x2, y, { stroke: color, w, attr }) + line(x1, y - 1.4, x1, y + 1.4, { stroke: color, w }) + line(x2, y - 1.4, x2, y + 1.4, { stroke: color, w });

function panelHead(x, y, tag, title) {
  return text(x, y, tag, { weight: 700, size: 9, attr: `data-panel-tag="${tag}"` }) + text(x + 9, y, title, { weight: 700 });
}

function xAxis(out, X, x0, x1, y, ticks, gridTop) {
  out.push(line(x0, y, x1, y, { stroke: ST.axis, w: 0.5 }));
  ticks.forEach((t, i) => {
    if (gridTop !== null) out.push(line(X(t), gridTop, X(t), y, { stroke: ST.grid, w: 0.35, attr: 'data-layer="grid"' }));
    out.push(line(X(t), y, X(t), y + 2, { stroke: ST.axis, w: 0.5 }));
    out.push(text(X(t), y + 2 + FS, tickLabel(t, i === ticks.length - 1), { anchor: "middle", fill: ST.muted, attr: `data-tick="${t}"` }));
  });
  return y + 2 + FS;
}

function barsPanel(out, y0) {
  out.push(panelHead(0, y0 + 9, "a", "Shapley shares of the errors"));
  const LA = Math.max(...spec.bars.map((b) => width(b.label)));
  const x0 = LA + 6, x1 = W - width("100%") / 2 - 0.5, X = (v) => x0 + v * (x1 - x0);
  out.push(`<g data-scale="bars" data-x0="${f(x0)}" data-x1="${f(x1)}">`);
  let y = y0 + 16;
  const order = [...KEYS, "unexplained"];
  for (const b of spec.bars) {
    const D = P[b.field];
    let acc = 0;
    const cy = y + b.height / 2;
    out.push(text(LA, midText(cy), b.label, { anchor: "end", attr: `data-bar-label="${b.id}"` }));
    for (const k of order) {
      const v = D[k];
      out.push(`<rect data-bar="${b.id}" data-key="${k}" data-v="${v}" x="${f(X(acc))}" y="${f(y)}" width="${f(X(acc + v) - X(acc))}" height="${b.height}" fill="${colorOf[k]}" stroke="#FFFFFF" stroke-width="0.6"/>`);
      acc += v;
    }
    const e = agg.bars[b.id].explained;
    let tx = X(e.v) + 4;
    if (e.ci) {
      out.push(whisker(X(e.ci[0]), X(e.ci[1]), cy, ST.ink, `data-ci="${b.id}" data-lo="${e.ci[0]}" data-hi="${e.ci[1]}"`, 0.8));
      tx = X(e.ci[1]) + 4;
    }
    const label = (b.id === spec.bars[0].id ? "explained " : "") + pct(e.v) + (e.ci ? ` ${pm(e.v, e.ci)}` : "");
    out.push(text(tx, midText(cy), label, { attr: `data-explained="${b.id}"`, weight: b.id === spec.bars[0].id ? 700 : null }));
    y += b.height + 3;
  }
  const base = xAxis(out, X, x0, x1, y + 1, [0, 0.25, 0.5, 0.75, 1], null);
  out.push("</g>");
  return base;
}

function explanationsPanel(out, x0p, x1p, y0, tag = "b") {
  out.push(panelHead(x0p, y0 + 9, tag, "Share per explanation"));
  const IND = 6, SW = 6.5;
  const labW = Math.max(...spec.families.map((fm) => width(fill(fm.title), true)),
    ...spec.families.flatMap((fm) => fm.items.map((it) => IND + SW + 3 + width(it.label))));
  const px0 = x0p + labW + 6, px1 = x1p - 3;
  const [d0, d1] = spec.per_explanation_domain, X = (v) => px0 + (v - d0) / (d1 - d0) * (px1 - px0);
  const rows = [], marks = [];
  let y = y0 + 16 + FS;
  const top = y - FS;
  spec.families.forEach((fm, fi) => {
    if (fi > 0) y += 2.5;
    const title = fill(fm.title);
    rows.push(text(x0p, y, title, { weight: 700, attr: `data-family="${fm.id}"` }));
    const tot = agg.families[fm.id];
    const totText = `${pct(tot.observed)} (shuffled ${pct(tot.shuffle)})`;
    if (px1 - width(totText) < x0p + width(title, true) + 6) fail(`family total of ${fm.id} collides with its title`);
    if (fm.items.length > 1) rows.push(text(px1, y, totText, { anchor: "end", fill: ST.muted, attr: `data-family-total="${fm.id}" stroke="#FFFFFF" stroke-width="2" paint-order="stroke"` }));
    y += ROW;
    for (const it of fm.items) {
      const cy = y - CAP / 2;
      rows.push(`<rect x="${f(x0p + IND)}" y="${f(cy - SW / 2)}" width="${SW}" height="${SW}" fill="${it.color}" data-swatch="${it.key}"/>`);
      rows.push(text(x0p + IND + SW + 3, y, it.label, { attr: `data-row-label="${it.key}"` }));
      const o = P.share[it.key], oc = P.ci[it.key], s = P.shuffle[it.key], sc = P.shuffle_ci[it.key];
      marks.push(whisker(X(sc[0]), X(sc[1]), cy + 1.8, ST.shuffle_mark, `data-ci-row="${it.key}" data-kind="shuffle" data-lo="${sc[0]}" data-hi="${sc[1]}"`));
      marks.push(dotShuf(X(s), cy + 1.8, `data-row="${it.key}" data-kind="shuffle" data-v="${s}"`));
      marks.push(whisker(X(oc[0]), X(oc[1]), cy - 1.8, it.color, `data-ci-row="${it.key}" data-kind="observed" data-lo="${oc[0]}" data-hi="${oc[1]}"`));
      marks.push(dotObs(X(o), cy - 1.8, it.color, `data-row="${it.key}" data-kind="observed" data-v="${o}"`));
      y += ROW;
    }
  });
  const grid = [];
  const base = xAxis(grid, X, px0, px1, y - ROW + 4, spec.per_explanation_ticks, top);
  out.push(`<g data-scale="explanations" data-x0="${f(px0)}" data-x1="${f(px1)}" data-d0="${d0}" data-d1="${d1}">`, ...grid, ...rows, ...marks, "</g>");
  return { base, rowsBottom: y - ROW + 4 };
}

function modelsPanel(out, x0p, x1p, y0, axisY, tag = "c") {
  out.push(panelHead(x0p, y0 + 9, tag, "Explained share per model"));
  const labels = [...spec.models.map((m) => m.label), T.term("median"), T.term("mean")];
  const labW = Math.max(...labels.map((l) => width(l)));
  const px0 = x0p + labW + 6, px1 = x1p - width(tickLabel(spec.per_model_ticks.at(-1), true)) / 2 - 0.5;
  const [d0, d1] = spec.per_model_domain, X = (v) => px0 + (v - d0) / (d1 - d0) * (px1 - px0);
  const nRows = spec.models.length + 2;
  const top = y0 + 16;
  const pitch = (axisY - 4 - top) / nRows;
  const rows = [], marks = [];
  const rowY = (i) => top + (i + 0.5) * pitch + (i >= spec.models.length ? 1.5 : 0);
  spec.models.forEach((m, i) => {
    const cy = rowY(i), A = agg.per_model[m.id];
    rows.push(text(px0 - 6, midText(cy), m.label, { anchor: "end", attr: `data-model-label="${m.id}"` }));
    marks.push(whisker(X(A.shuffle.ci[0]), X(A.shuffle.ci[1]), cy + 1.8, ST.shuffle_mark, `data-ci-model="${m.id}" data-kind="shuffle" data-lo="${A.shuffle.ci[0]}" data-hi="${A.shuffle.ci[1]}"`));
    marks.push(dotShuf(X(A.shuffle.v), cy + 1.8, `data-model="${m.id}" data-kind="shuffle" data-v="${A.shuffle.v}"`));
    marks.push(whisker(X(A.observed.ci[0]), X(A.observed.ci[1]), cy - 1.8, ST.ink, `data-ci-model="${m.id}" data-kind="observed" data-lo="${A.observed.ci[0]}" data-hi="${A.observed.ci[1]}"`));
    marks.push(dotObs(X(A.observed.v), cy - 1.8, ST.ink, `data-model="${m.id}" data-kind="observed" data-v="${A.observed.v}"`));
  });
  const sepY = top + spec.models.length * pitch + 0.75;
  rows.push(line(x0p + 2, sepY, px1, sepY, { stroke: ST.grid, w: 0.5 }));
  ["median", "mean"].forEach((stat, j) => {
    const cy = rowY(spec.models.length + j);
    const o = agg.over_models.observed[stat], s = agg.over_models.shuffle[stat];
    rows.push(text(px0 - 6, midText(cy), T.term(stat), { anchor: "end", weight: 700, attr: `data-summary-label="${stat}"` }));
    marks.push(dotShuf(X(s), cy + 1.8, `data-summary="${stat}" data-kind="shuffle" data-v="${s}"`));
    marks.push(dotObs(X(o), cy - 1.8, ST.ink, `data-summary="${stat}" data-kind="observed" data-v="${o}"`));
  });
  const grid = [];
  const base = xAxis(grid, X, px0, px1, axisY, spec.per_model_ticks, top);
  out.push(`<g data-scale="models" data-x0="${f(px0)}" data-x1="${f(px1)}" data-d0="${d0}" data-d1="${d1}">`, ...grid, ...rows, ...marks, "</g>");
  return base;
}

function marksKey(out, y) {
  const items = [
    { id: "observed", label: "observed", glyph: (cx, cy) => dotObs(cx, cy, ST.ink, "") },
    { id: "shuffle", label: "shuffled answers", glyph: (cx, cy) => dotShuf(cx, cy, "") },
    { id: "ci", label: T.term("bootstrap_ci"), glyph: (cx, cy) => whisker(cx - 4.5, cx + 4.5, cy, ST.ink, "") + dotObs(cx, cy, ST.ink, "") },
  ];
  const k = keyBlocks([{ id: "marks", title: T.term("key_marks"), items }], {
    itemsLeft: 0, right: W, y, fs: FS, textWidth: (s, size, bold) => tw(s, size, bold),
    text: (x, yy, s, o = {}) => text(x, yy, s, { anchor: o.anchor, weight: o.weight }), clear: true,
  });
  out.push(k.svg);
  return k.lastBaseline;
}

const famTitle = (fm) => fill(fm.title);
const M = spec.main;
const hex = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
const tint = (c, a) => "#" + hex(c).map((v) => Math.round(255 - a * (255 - v)).toString(16).padStart(2, "0")).join("").toUpperCase();
const lum = (c) => {
  const [r, g, b] = hex(c).map((v) => { const u = v / 255; return u <= 0.04045 ? u / 12.92 : ((u + 0.055) / 1.055) ** 2.4; });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};

const inkOn = (c) => (1.05 / (lum(c) + 0.05) >= 4 ? "#FFFFFF" : ST.ink);
const famKeys = (fm) => fm.items.map((it) => it.key);
const famValue = (fm, field) => famKeys(fm).reduce((s, k) => s + P[field][k], 0);

function pairedLayout(out) {
  const rows = [...spec.families.map((fm) => ({ id: fm.id, label: famTitle(fm), hue: fm.hue, v: (b) => famValue(fm, b.field) })),
    { id: "total", label: "Total explained", hue: M.total_hue, bold: true, v: (b) => 1 - P[b.field].unexplained }];
  const labW = Math.max(...rows.map((r) => width(r.label, r.bold)));
  const colGap = 12, x00 = labW + 8;
  const colW = (W - x00 - 2 * colGap) / spec.bars.length;
  const numW = Math.max(...rows.flatMap((r) => spec.bars.map((b) => width(pct(r.v(b)))))) + 3;
  const [, d1] = M.paired_domain;
  const BH = 7.5, PITCH = 11.5, top = 13, SEP = 4;
  const rowCy = (i) => top + i * PITCH + PITCH / 2 + (rows[i].id === "total" ? SEP : 0);
  const bottom = top + rows.length * PITCH + SEP;
  let base = 0;
  out.push(`<g data-scale="paired" data-d1="${d1}">`);
  out.push(text(0, 8, "Share of the errors", { fill: ST.muted, attr: 'data-title="paired"' }));
  if (width("Share of the errors") > x00 - 4) fail("paired title runs into the first column");
  rows.forEach((r, i) => out.push(text(labW, midText(rowCy(i)), r.label, { anchor: "end", weight: r.bold ? 700 : null, attr: `data-row-label="${r.id}"` })));
  const sepY = top + (rows.length - 1) * PITCH + SEP / 2;
  out.push(line(0, sepY, W, sepY, { stroke: ST.grid, w: 0.5 }));
  spec.bars.forEach((b, c) => {
    const x0 = x00 + c * (colW + colGap), x1 = x0 + colW - numW, X = (v) => x0 + (v / d1) * (x1 - x0);
    out.push(`<g data-col="${b.id}" data-x0="${f(x0)}" data-x1="${f(x1)}">`);
    base = xAxis(out, X, x0, x1, bottom, M.paired_ticks, top + 1);
    out.push(text(x0, 8, b.label, { weight: 700, attr: `data-col-label="${b.id}"` }));
    rows.forEach((r, i) => {
      const v = r.v(b), cy = rowCy(i);
      if (v > d1) fail(`${r.id} ${b.id} exceeds the paired domain`);
      const style = c === 0 ? `fill="${r.hue}"` : b.ci === null ? `fill="#FFFFFF" stroke="${r.hue}" stroke-width="0.9"` : `fill="${tint(r.hue, M.shuffle_tint)}"`;
      const inset = b.ci === null && c > 0 ? 0.45 : 0;
      out.push(`<rect data-row="${r.id}" data-kind="${b.id}" data-v="${v}" x="${f(x0 + inset)}" y="${f(cy - BH / 2 + inset)}" width="${f(Math.max(X(v) - x0 - 2 * inset, 0))}" height="${f(BH - 2 * inset)}" ${style}/>`);
      out.push(text(X(v) + 2.5, midText(cy), pct(v), { weight: r.bold ? 700 : null, attr: `data-value="${r.id}" data-kind="${b.id}" stroke="#FFFFFF" stroke-width="2" paint-order="stroke"` }));
    });
    out.push(line(x0, top + 1, x0, bottom, { stroke: ST.axis, w: 0.5 }));
    out.push("</g>");
  });
  out.push("</g>");
  return base + KEY_SPACING.descent_em * FS + KEY_SPACING.bottom_pad_pt;
}

function stackedLayout(out) {
  const keyTitle = "Explanation";
  const LA = Math.max(...spec.bars.map((b) => width(b.label)), width(keyTitle, true));
  const x0 = LA + 6, x1 = W - 0.5, X = (v) => x0 + v * (x1 - x0);
  const BH = 12, BAND = 12, PAD = 2, LEAD_ABOVE = 3.5, LEAD_BELOW = 5;
  out.push(`<g data-scale="stacked" data-x0="${f(x0)}" data-x1="${f(x1)}">`);
  let y = LEAD_ABOVE + CAP + 1.5, bottom = 0;
  const title = "Share of the errors";
  out.push(text(0, y - LEAD_ABOVE, title, { fill: ST.muted, attr: 'data-title="stacked"' }));
  for (const b of spec.bars) {
    const cy = y + BH / 2;
    out.push(text(LA, midText(cy), b.label, { anchor: "end", attr: `data-bar-label="${b.id}"` }));
    let acc = 0;
    const outside = [];
    for (const fm of spec.families) {
      const v = famValue(fm, b.field), xa = X(acc), xb = X(acc + v);
      out.push(`<rect data-bar="${b.id}" data-key="${fm.id}" data-v="${v}" x="${f(xa)}" y="${f(y)}" width="${f(xb - xa)}" height="${BH}" fill="${fm.hue}" stroke="#FFFFFF" stroke-width="0.8"/>`);
      const s = pct(v), sw = width(s, true);
      if (xb - xa >= sw + 2 * PAD) out.push(text((xa + xb) / 2, midText(cy), s, { anchor: "middle", weight: 700, fill: inkOn(fm.hue), attr: `data-value="${fm.id}" data-bar="${b.id}" data-place="inside"` }));
      else outside.push({ id: fm.id, s, w: sw, cx: (xa + xb) / 2 });
      acc += v;
    }
    const u = P[b.field].unexplained;
    out.push(`<rect data-bar="${b.id}" data-key="unexplained" data-v="${u}" x="${f(X(acc))}" y="${f(y)}" width="${f(X(1) - X(acc))}" height="${BH}" fill="${ST.unexplained}" stroke="#FFFFFF" stroke-width="0.8"/>`);
    out.push(text(x1 - 4, midText(cy), `unexplained ${pct(u)}`, { anchor: "end", fill: ST.muted, attr: `data-value="unexplained" data-bar="${b.id}"` }));

    const last = b === spec.bars.at(-1);
    const base = last ? y + BH + LEAD_BELOW + CAP : y - LEAD_ABOVE;
    let right = -Infinity;
    for (const o of outside) {
      o.x = Math.max(o.cx - o.w / 2, right + 4, b === spec.bars[0] ? width(title) + 6 : 0);
      right = o.x + o.w;
    }
    if (right > W) fail(`outside labels of ${b.id} pass the right edge`);
    for (const o of outside) {
      const [ly0, ly1] = last ? [y + BH, base - CAP - 1] : [y, base + 0.8];
      out.push(line(o.cx, ly0, o.x + o.w / 2, ly1, { stroke: ST.ink, w: 0.5, attr: `data-leader="${o.id}" data-bar="${b.id}"` }));
      out.push(text(o.x + o.w / 2, base, o.s, { anchor: "middle", weight: 700, attr: `data-value="${o.id}" data-bar="${b.id}" data-place="outside"` }));
    }
    bottom = last && outside.length ? base + KEY_SPACING.descent_em * FS : y + BH;
    y += BH + BAND;
  }
  out.push("</g>");
  const items = spec.families.map((fm) => ({ id: fm.id, label: famTitle(fm),
    glyph: (cx, cy) => `<rect x="${f(cx - 3.5)}" y="${f(cy - 3.5)}" width="7" height="7" fill="${fm.hue}" data-swatch="${fm.id}"/>` }));
  const k = keyBlocks([{ id: "families", title: keyTitle, items }], {
    itemsLeft: x0, right: W, y: keyBaselineBelow(bottom - KEY_SPACING.descent_em * FS, FS), titleRight: LA, fs: FS,
    textWidth: (t, size, bold) => tw(t, size, bold),
    text: (x, yy, t, o = {}) => text(x, yy, t, { anchor: o.anchor, weight: o.weight }), clear: true,
  });
  out.push(k.svg);
  return keyBottom(k.lastBaseline, FS);
}

const SHADE_RAMP = ["#374151", "#8B93A1", "#C9CED6"];
const rampGlyph = (colors, attr = "") => (cx, cy) => {
  const w = 10 / colors.length;
  return colors.map((c, i) => `<rect x="${f(cx - 5 + i * w)}" y="${f(cy - 3.5)}" width="${f(w)}" height="7" fill="${c}"${attr ? ` ${attr}="${i}"` : ""}/>`).join("");
};
const shadeBlock = () => ({ id: "shades", title: "Shades", items: [
  { id: "ranks", label: "in each color: ranked 1st, 2nd, 3rd", glyph: rampGlyph(SHADE_RAMP) },
  ...spec.families.find((fm) => fm.id === "ftset").items.map((it) => ({ id: it.key, label: it.label, glyph: rampGlyph([it.color]) })),
] });
const keyText = (x, yy, t, o = {}) => text(x, yy, t, { anchor: o.anchor, weight: o.weight });
const keyWidth = (t, size, bold) => tw(plainOf(t), size, bold);

function pairedTable(out, rows, o) {
  const labW = Math.max(...rows.map((r) => (r.indent || 0) + width(r.label, r.bold)));
  const colGap = 12, x00 = labW + 8, gap = o.gapBefore;
  const barRows = rows.filter((r) => !r.textOnly);
  const colMax = spec.bars.map((b) => Math.max(...barRows.map((r) => r.v(b))));
  const numW = spec.bars.map((b) => Math.max(...rows.map((r) => width(pct(r.v(b)), r.bold))) + 3);

  const header = (b) => (o.longHeaders && b.header) || b.label;
  const hw = spec.bars.map((b) => width(header(b), true));
  const avail = W - x00 - colGap * (spec.bars.length - 1);
  let fixed = spec.bars.map(() => false), k = 0;
  for (let it = 0; it <= spec.bars.length; it++) {
    const free = colMax.reduce((a, m, i) => a + (fixed[i] ? 0 : m), 0);
    k = (avail - spec.bars.reduce((a, b, i) => a + (fixed[i] ? hw[i] : numW[i]), 0)) / free;
    const next = spec.bars.map((b, i) => fixed[i] || colMax[i] * k + numW[i] < hw[i]);
    if (next.every((v, i) => v === fixed[i])) break;
    fixed = next;
  }
  const slot = spec.bars.map((b, i) => Math.max(colMax[i] * k + numW[i], hw[i]));
  const top = 13;
  let y = top;
  const cy = rows.map((r) => { y += (r.gap || 0); const c = y + r.pitch / 2; y += r.pitch; return c; });
  const bottom = y;
  out.push(`<g data-scale="${o.scale}" data-k="${k}">`);
  out.push(text(0, 8, "Share of the errors", { fill: ST.muted, attr: `data-title="${o.scale}"` }));
  if (width("Share of the errors") > x00 - 4) fail("title runs into the first column");
  rows.forEach((r, i) => out.push(o.alignLeft
    ? text(r.indent || 0, midText(cy[i]), r.label, { weight: r.bold ? 700 : null, attr: `data-row-label="${r.id}"` })
    : text(labW, midText(cy[i]), r.label, { anchor: "end", weight: r.bold ? 700 : null, attr: `data-row-label="${r.id}"` })));
  rows.forEach((r, i) => { if (r.sepBefore) { const sy = cy[i] - r.pitch / 2 - r.gap / 2; out.push(line(0, sy, W, sy, { stroke: ST.grid, w: 0.5 })); } });
  let x0 = x00, base = 0;
  spec.bars.forEach((b, c) => {
    const X = (v) => x0 + v * k, x1 = X(colMax[c]);
    out.push(`<g data-col="${b.id}" data-x0="${f(x0)}">`);
    const ticks = [];
    for (let t = 0; t <= colMax[c] + 1e-9; t += o.tickStep) ticks.push(Math.round(t * 1000) / 1000);
    base = xAxis(out, X, x0, x1, bottom, ticks, top + 1);
    out.push(text(x0, 8, header(b), { weight: 700, attr: `data-col-label="${b.id}"` }));
    rows.forEach((r, i) => {
      const v = r.v(b);
      const label = (x, attr = "") => text(x, midText(cy[i]), pct(v), { weight: r.bold ? 700 : null, attr: `data-value="${r.id}" data-kind="${b.id}" stroke="#FFFFFF" stroke-width="2" paint-order="stroke"${attr}` });
      if (r.textOnly) { out.push(label(x0 + 2.5)); return; }
      const BH = r.bh, yt = cy[i] - BH / 2;
      const segs = r.segs ? r.segs(b) : [{ key: r.id, v, color: r.hue }];
      let acc = 0;
      for (const sg of segs) {
        const xa = X(acc), xb = X(acc + sg.v);
        out.push(`<rect data-row="${r.id}" data-kind="${b.id}" data-seg="${sg.key}" data-v="${sg.v}" x="${f(xa)}" y="${f(yt)}" width="${f(xb - xa)}" height="${f(BH)}" fill="${baseFill(sg.color, c)}"${segs.length > 1 ? ' stroke="#FFFFFF" stroke-width="0.6"' : ""}/>`);
        if (c === 0 && segs.length > 1 && o.inside) {
          const s = pct(sg.v), sw = width(s);
          if (xb - xa >= sw + 5) out.push(text((xa + xb) / 2, midText(cy[i]), s, { anchor: "middle", fill: inkOn(sg.color), attr: `data-inside="${sg.key}"` }));
        }
        acc += sg.v;
      }
      if (BASELINES === "pale" && b.ci === null && c > 0) out.push(`<rect data-outline="${r.id}" x="${f(x0 + 0.45)}" y="${f(yt + 0.45)}" width="${f(Math.max(X(v) - x0 - 0.9, 0))}" height="${f(BH - 0.9)}" fill="none" stroke="${r.hue}" stroke-width="0.9"/>`);
      out.push(label(X(v) + 2.5));
    });
    out.push(line(x0, top + 1, x0, bottom, { stroke: ST.axis, w: 0.5 }));
    out.push("</g>");
    if (x0 + slot[c] > W + 1e-6) fail(`column ${b.id} passes the right edge`);
    x0 += slot[c] + colGap;
  });
  out.push("</g>");
  return { base, labW };
}

const famSegs = (fm) => (fm.items.length > 1 ? (b) => fm.items.map((it) => ({ key: it.key, v: P[b.field][it.key], color: it.color })) : null);
const totalRow = (extra) => ({ id: "total", label: "Total explained", bold: true, hue: M.total_hue, v: (b) => 1 - P[b.field].unexplained, ...extra });

function pairedShadedLayout(out) {
  const rows = [...spec.families.map((fm) => ({ id: fm.id, label: famTitle(fm), hue: fm.hue, pitch: 10.5, bh: 7.5, v: (b) => famValue(fm, b.field), segs: famSegs(fm) })),
    totalRow({ textOnly: true, pitch: 11, sepBefore: true, gap: 3 })];
  const t = pairedTable(out, rows, { scale: "paired-shaded", tickStep: 0.05, inside: true, longHeaders: VARIANT.headers === "long" });
  const k = keyBlocks([shadeBlock()], { itemsLeft: 0, right: W, y: keyBaselineBelow(t.base, FS), fs: FS, textWidth: keyWidth, text: keyText, clear: true });
  out.push(k.svg);
  return keyBottom(k.lastBaseline, FS);
}

function pairedSubrowsLayout(out) {
  const rows = [];
  spec.families.forEach((fm, fi) => {
    rows.push({ id: fm.id, label: famTitle(fm), bold: true, hue: fm.hue, pitch: 10.5, bh: 7, gap: fi ? 2.5 : 0, v: (b) => famValue(fm, b.field), segs: famSegs(fm) });
    if (fm.items.length > 1) for (const it of fm.items)
      rows.push({ id: it.key, label: it.label, indent: 8, hue: it.color, pitch: 9.5, bh: 6, v: (b) => P[b.field][it.key] });
  });
  rows.push(totalRow({ pitch: 11.5, bh: 7.5, sepBefore: true, gap: 4 }));
  const t = pairedTable(out, rows, { scale: "paired-subrows", tickStep: 0.1, alignLeft: true });
  return t.base + KEY_SPACING.descent_em * FS + KEY_SPACING.bottom_pad_pt;
}

function stackedRanksLayout(out) {
  const blocks = [{ id: "families", title: "Explanation", items: spec.families.map((fm) => ({ id: fm.id, label: famTitle(fm),
    glyph: rampGlyph(fm.items.map((it) => it.color), `data-swatch-${fm.id}`) })) }, shadeBlock()];
  const LA = Math.max(...spec.bars.map((b) => width(b.label)), ...blocks.map((bl) => width(bl.title, true)));
  const x0 = LA + 6, x1 = W - 0.5, X = (v) => x0 + v * (x1 - x0);
  const BH = 12, BAND = 15, GAP = 6, TICK = 2, LIFT = 3.5;
  out.push(`<g data-scale="stacked-ranks" data-x0="${f(x0)}" data-x1="${f(x1)}">`);
  let y = BAND, bottom = 0;
  out.push(text(0, 8, "Share of the errors", { fill: ST.muted, attr: 'data-title="stacked-ranks"' }));
  spec.bars.forEach((b, bi) => {
    const last = bi === spec.bars.length - 1;
    if (bi > 0) y += last ? GAP : BAND;
    const cy = y + BH / 2;
    out.push(text(LA, midText(cy), b.label, { anchor: "end", attr: `data-bar-label="${b.id}"` }));
    let acc = 0;
    const groups = [];
    for (const fm of spec.families) {
      const g0 = acc;
      for (const it of fm.items) {
        const v = P[b.field][it.key], xa = X(acc), xb = X(acc + v);
        const fc = baseFill(it.color, bi);
        out.push(`<rect data-bar="${b.id}" data-key="${it.key}" data-v="${v}" x="${f(xa)}" y="${f(y)}" width="${f(xb - xa)}" height="${BH}" fill="${fc}" stroke="#FFFFFF" stroke-width="0.6"/>`);
        const s = pct(v);
        if (fm.items.length > 1 && xb - xa >= width(s) + 3) out.push(text((xa + xb) / 2, midText(cy), s, { anchor: "middle", fill: inkOn(fc), attr: `data-inside="${it.key}" data-bar="${b.id}"` }));
        acc += v;
      }
      groups.push({ id: fm.id, xa: X(g0), xb: X(acc), s: pct(acc - g0) });
      if (BASELINES === "pale" && b.ci === null && bi > 0) out.push(`<rect data-outline="${fm.id}" data-bar="${b.id}" x="${f(X(g0) + 0.75)}" y="${f(y + 0.45)}" width="${f(X(acc) - X(g0) - 1.5)}" height="${f(BH - 0.9)}" fill="none" stroke="${fm.hue}" stroke-width="0.9"/>`);
    }
    const u = P[b.field].unexplained;
    out.push(`<rect data-bar="${b.id}" data-key="unexplained" data-v="${u}" x="${f(X(acc))}" y="${f(y)}" width="${f(X(1) - X(acc))}" height="${BH}" fill="${ST.unexplained}" stroke="#FFFFFF" stroke-width="0.6"/>`);
    out.push(text(x1 - 4, midText(cy), `unexplained ${pct(u)}`, { anchor: "end", fill: ST.muted, attr: `data-value="unexplained" data-bar="${b.id}"` }));

    const yb = last ? y + BH + LIFT : y - LIFT;
    const baseTxt = last ? yb + TICK + 4 + CAP : yb - TICK - 1.5;
    let right = bi === 0 ? width("Share of the errors") + 6 : -Infinity;
    for (const g of groups) {
      g.w = width(g.s, true);
      g.x = Math.max((g.xa + g.xb) / 2 - g.w / 2, right + 6);
      right = g.x + g.w;
    }
    if (right > W) fail(`family totals of ${b.id} pass the right edge`);
    for (const g of groups) {
      const tip = last ? yb - LIFT + 0.8 : yb + LIFT - 0.8, mid = (g.xa + g.xb) / 2, lc = g.x + g.w / 2;
      out.push(`<path data-bracket="${g.id}" data-bar="${b.id}" d="M${f(g.xa + 0.5)},${f(tip)}V${f(yb)}H${f(g.xb - 0.5)}V${f(tip)}" fill="none" stroke="${ST.ink}" stroke-width="0.5"/>`);
      out.push(line(mid, yb, lc, last ? baseTxt - CAP - 0.8 : baseTxt + 0.8, { stroke: ST.ink, w: 0.5 }));
      out.push(text(lc, baseTxt, g.s, { anchor: "middle", weight: 700, attr: `data-family-total="${g.id}" data-bar="${b.id}"` }));
    }
    bottom = last ? baseTxt + KEY_SPACING.descent_em * FS : y + BH;
    y += BH;
  });
  out.push("</g>");
  const k = keyBlocks(blocks, { itemsLeft: x0, right: W, y: keyBaselineBelow(bottom - KEY_SPACING.descent_em * FS, FS), titleRight: LA, fs: FS,
    textWidth: keyWidth, text: keyText, clear: true });
  out.push(k.svg);
  return keyBottom(k.lastBaseline, FS);
}

function appendixLayout(out) {
  const BW = spec.explanations_width_share * W;
  const b = explanationsPanel(out, 0, BW, 0, "a");
  modelsPanel(out, BW + 12, W, 0, b.rowsBottom, "b");
  return keyBottom(marksKey(out, keyBaselineBelow(b.base, FS)), FS);
}

function fullLayout(out) {
  let base = barsPanel(out, 0);
  const y1 = base + 12;
  const BW = spec.explanations_width_share * W;
  const b = explanationsPanel(out, 0, BW, y1);
  modelsPanel(out, BW + 12, W, y1, b.rowsBottom);
  const last = marksKey(out, keyBaselineBelow(b.base, FS));
  return keyBottom(last, FS);
}

const LAYOUTS = { full: fullLayout, paired: pairedLayout, stacked: stackedLayout, appendix: appendixLayout,
  "paired-shaded": pairedShadedLayout, "paired-subrows": pairedSubrowsLayout, "stacked-ranks": stackedRanksLayout };

let BASELINES = "same", VARIANT = {};
const baseFill = (color, c) => (c === 0 || BASELINES === "same" ? color : tint(color, M.pale_tint));

function render(variant, baselines = "same") {
  BASELINES = baselines;
  VARIANT = variant;
  const out = [];
  const layout = LAYOUTS[variant.layout || "full"];
  if (!layout) fail(`unknown layout ${variant.layout}`);
  const H = layout(out);
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${f(W)}pt" height="${f(H)}pt" viewBox="0 0 ${f(W)} ${f(H)}" font-family="${spec.font_family}">
<title>${esc(spec.figure_id)}</title>
<metadata>figure_id=${spec.figure_id}; snapshot_sha256=${spec.snapshot_sha256}; source_sha256=${spec.source.sha256}; probe_names_sha256=${spec.probe_names.sha256}; status=${spec.status}${baselines === "same" ? "" : `; baselines=${baselines}`}</metadata>
<rect width="100%" height="100%" fill="#FFFFFF"/>
${out.join("\n")}
</svg>
`;
}

if (args.rounds) {
  const R = args.rounds;
  const ARTIFACT = /\.(svg|pdf|png)$/;

  const clean = (dir) => {
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      const p = `${dir}/${e.name}`;
      if (e.isDirectory()) clean(p);
      else if (e.isFile() && ARTIFACT.test(e.name)) rmSync(p, { force: true });
    }
  };
  clean(R);
  const rounds = readdirSync(R).filter((d) => /^r\d+$/.test(d));
  if (!rounds.length) fail(`no rN/round.json under ${R}`);
  for (const r of rounds) {
    const src = JSON.parse(readFileSync(`${R}/${r}/round.json`, "utf8"));
    const parts = src.subdirs ? Object.entries(src.subdirs).map(([d, p]) => [`${R}/${r}/${d}`, p]) : [[`${R}/${r}`, src]];
    for (const [dir, p] of parts) for (const id of p.targets) {
      mkdirSync(dir, { recursive: true });
      if (!spec.variants[id]) fail(`${r}/round.json names an unknown target ${id}`);
      writeFileSync(`${dir}/${outName(spec.variants[id])}.svg`, render(spec.variants[id], p.baselines || "same"));
    }
  }
} else {
  for (const id of spec.tracked) writeFileSync(`${outName(spec.variants[id])}.svg`, render(spec.variants[id]));
}
