#!/usr/bin/env node
import { writeFileSync } from "node:fs";
import {
  averageRanks,
  buildCells,
  covariateScales,
  covariateT,
  f,
  fail,
  fmtCorr,
  loadInputs,
  luminance,
  mean,
  ols,
  parseArgs,
  pearson,
  quantile7,
  ramp,
  relationSources,
  SOURCE_COLORS,
  SOURCE_MARKS,
  SOURCE_NAMES,
  SOURCES,
  summarize,
  svgDocument,
  textWidth,
  xml,
} from "./lib.mjs";
import { keyBlocks, keyBaselineBelow, keyBottom } from "../shared/key.mjs";
import { figureLabels } from "../shared/relation_names.mjs";
import { mathSym, termWithSymbol } from "../shared/math_text.mjs";

const args = parseArgs(process.argv.slice(2));
for (const key of ["spec", "variant", "svg"]) if (!args[key]) fail(`missing --${key}`);
const inputs = loadInputs(args.spec, args.data);
const { spec } = inputs;
const cells = buildCells(inputs);
const summary = summarize(inputs, cells);
const st = spec.style;
const W = st.paper_width_pt;
const FS = st.min_font_pt;

const meanMark = (x, y, color, v, attrs = "") => `<path d="M${f(x)},${f(y - 1.9)} L${f(x + 1.9)},${f(y)} L${f(x)},${f(y + 1.9)} L${f(x - 1.9)},${f(y)} Z" fill="#FFFFFF" stroke="${color}" stroke-width="0.6" data-stat="mean" data-v="${v}"${attrs ? " " + attrs : ""}/>`;
const meanOf = (v) => v.reduce((a, b) => a + b, 0) / v.length;
const xName = inputs.metricName("x", "axis_short_label");
const yName = inputs.metricName("y", "axis_short_label");
const models = spec.models;
const allRelations = relationSources(spec).order;
const nRel = allRelations.length;
const countBySource = Object.fromEntries(SOURCES.map((source) => [source, spec.relation_sets[source].length]));
const sourceName = SOURCE_NAMES;

const presentSources = SOURCES.filter((source) => countBySource[source] > 0);
const drawOrder = [...presentSources].sort((a, b) => countBySource[b] - countBySource[a] || SOURCES.indexOf(a) - SOURCES.indexOf(b));

const T = inputs.terms;
const PEARSON = T.term("pearson_r");

const FIG_LABEL = figureLabels(allRelations, { labelOf: inputs.labelOf, sourceOf: relationSources(spec).sourceOf, setName: (s) => sourceName[s], sources: SOURCES });
const SPEARMAN = T.term("spearman_rho");

const text = (x, y, value, opts = {}) => {
  const attrs = [
    `x="${f(x)}"`,
    `y="${f(y)}"`,
    `font-size="${f(opts.size || FS)}"`,
    opts.anchor ? `text-anchor="${opts.anchor}"` : "",
    opts.weight ? `font-weight="${opts.weight}"` : "",
    opts.fill ? `fill="${opts.fill}"` : "",
    opts.rotate !== undefined ? `transform="rotate(${opts.rotate} ${f(x)} ${f(y)})"` : "",
    opts.data || "",
  ].filter(Boolean);
  if ((opts.size || FS) < FS - 1e-9) fail(`text below the ${FS} pt minimum: ${value}`);
  return `<text ${attrs.join(" ")}>${xml(value)}</text>`;
};
const line = (x1, y1, x2, y2, stroke, width, extra = "") =>
  `<line x1="${f(x1)}" y1="${f(y1)}" x2="${f(x2)}" y2="${f(y2)}" stroke="${stroke}" stroke-width="${f(width)}"${extra ? " " + extra : ""}/>`;
const tickLabel = (v) => (v === 0 ? "0" : v === 1 ? "1" : (v < 0 ? "−" : "") + Math.abs(v).toFixed(2).replace(/0+$/, "").replace(/^0/, ""));

function scatterGrid(withSourceFits) {
  const cols = 4;
  const slots = models.length + 1;
  const rowsN = Math.ceil(slots / cols);
  const left = 30;
  const right = 3;
  const gapX = 11;
  const pw = (W - left - right - gapX * (cols - 1)) / cols;
  const ph = pw * 0.86;
  const titleH = 21;
  const tickH = 11;
  const gapY = 6;
  const pitch = titleH + ph + tickH + gapY;
  const top = 2;
  const height = top + rowsN * pitch + 12;
  const lo = -0.03;
  const hi = 1.03;
  const out = [];
  const lastRowOfCol = (col) => {
    let last = 0;
    for (let i = 0; i < models.length; i += 1) if (i % cols === col) last = Math.floor(i / cols);
    return last;
  };
  models.forEach((model, i) => {
    const col = i % cols;
    const row = Math.floor(i / cols);
    const x0 = left + col * (pw + gapX);
    const y0 = top + row * pitch + titleH;
    const sx = (v) => x0 + ((v - lo) / (hi - lo)) * pw;
    const sy = (v) => y0 + ph - ((v - lo) / (hi - lo)) * ph;
    const s = summary.perModel[i];
    out.push(text(x0 + pw / 2, y0 - 11, model.label, { anchor: "middle", weight: 700 }));
    out.push(text(x0 + pw / 2, y0 - 2.5, `r=${fmtCorr(s.pearson)}  ρ=${fmtCorr(s.spearman)}`, { anchor: "middle", fill: st.muted }));
    for (const t of [0, 0.5, 1]) {
      out.push(line(sx(t), y0, sx(t), y0 + ph, st.grid, 0.35, 'data-layer="grid"'));
      out.push(line(x0, sy(t), x0 + pw, sy(t), st.grid, 0.35, 'data-layer="grid"'));
      if (col === 0) out.push(text(x0 - 2.5, sy(t) + 2.8, tickLabel(t), { anchor: "end", fill: st.muted }));
      if (row === lastRowOfCol(col)) out.push(text(sx(t), y0 + ph + 9, tickLabel(t), { anchor: "middle", fill: st.muted }));
    }
    out.push(line(x0, y0, x0, y0 + ph, st.axis, 0.5), line(x0, y0 + ph, x0 + pw, y0 + ph, st.axis, 0.5));
    const own = cells.filter((cell) => cell.model === model.id);
    for (const source of drawOrder) {
      const color = SOURCE_COLORS[source];
      for (const cell of own.filter((c) => c.source === source)) {
        out.push(`<circle cx="${f(sx(cell.x))}" cy="${f(sy(cell.y))}" r="1.25" fill="${color}" fill-opacity="0.72" stroke="#FFFFFF" stroke-width="0.25"/>`);
      }
    }
    const fitLine = (fit, stroke, width, extra) => {
      const xa = Math.max(0, (0 - fit.intercept) / fit.slope);
      const xb = Math.min(1, (1 - fit.intercept) / fit.slope);
      const [x1, x2] = fit.slope > 0 ? [Math.max(0, Math.min(xa, 1)), Math.max(0, xb)] : [0, 1];
      return line(sx(x1), sy(fit.intercept + fit.slope * x1), sx(x2), sy(fit.intercept + fit.slope * x2), stroke, width, extra);
    };
    if (withSourceFits) {
      for (const source of drawOrder) {
        if (s.by_source[source]) out.push(fitLine(s.by_source[source], SOURCE_COLORS[source], 0.8, 'stroke-opacity="0.95"'));
      }
    }
    out.push(fitLine(s, st.ink, 0.6, 'stroke-dasharray="2.2 1.6" stroke-opacity="0.8"'));
  });

  const keySlot = models.length;
  const kx = left + (keySlot % cols) * (pw + gapX) + 4;
  const ky = top + Math.floor(keySlot / cols) * pitch + titleH;
  const items = presentSources.map((source) => ["dot", SOURCE_COLORS[source], `${sourceName[source]} relations`]);
  if (withSourceFits) {
    for (const source of presentSources) items.push(["solid", SOURCE_COLORS[source], `${sourceName[source]} fit`]);
  }
  items.push(["dash", st.ink, "OLS fit, all relations"]);
  items.forEach(([kind, color, label], k) => {
    const y = ky + 4 + k * 11;
    if (kind === "dot") out.push(`<circle cx="${f(kx + 5)}" cy="${f(y - 2.8)}" r="1.8" fill="${color}"/>`);
    if (kind === "solid") out.push(line(kx, y - 2.8, kx + 10, y - 2.8, color, 0.8));
    if (kind === "dash") out.push(line(kx, y - 2.8, kx + 10, y - 2.8, color, 0.6, 'stroke-dasharray="2.2 1.6"'));
    out.push(text(kx + 14, y, label));
  });

  const plotRows = Math.ceil(models.length / cols);
  out.push(text(9, top + titleH + ((plotRows - 1) * pitch + ph) / 2, yName, { anchor: "middle", rotate: -90 }));
  out.push(text(left + (cols * pw + (cols - 1) * gapX) / 2, height - 3, xName, { anchor: "middle" }));
  return { width: W, height, body: out.join(""), desc: `${models.length} panels, one per model; ${nRel} relations per panel; x ${xName}, y ${yName}.` };
}

function scatterPooled() {
  const top = 22;
  const S = 172;
  const x0 = 36;
  const out = [];
  const pts = summary.centred;
  const m = Math.ceil(Math.max(...pts.map((p) => Math.max(Math.abs(p.xc), Math.abs(p.yc)))) * 10) / 10;
  const sx = (v) => x0 + ((v + m) / (2 * m)) * S;
  const sy = (v) => top + S - ((v + m) / (2 * m)) * S;
  const ticks = [];
  for (let t = -Math.floor(m / 0.25) * 0.25; t <= m + 1e-9; t += 0.25) ticks.push(Math.round(t * 100) / 100);
  for (const t of ticks) {
    out.push(line(sx(t), top, sx(t), top + S, st.grid, 0.35, 'data-layer="grid"'), line(x0, sy(t), x0 + S, sy(t), st.grid, 0.35, 'data-layer="grid"'));
    const label = t === 0 ? "0" : `${t < 0 ? "−" : "+"}${Math.abs(t).toFixed(2).replace(/^0/, "")}`;
    out.push(text(sx(t), top + S + 9, label, { anchor: "middle", fill: st.muted }));
    out.push(text(x0 - 2.5, sy(t) + 2.8, label, { anchor: "end", fill: st.muted }));
  }
  out.push(line(x0, top, x0, top + S, st.axis, 0.5), line(x0, top + S, x0 + S, top + S, st.axis, 0.5));
  for (const source of drawOrder) {
    for (const p of pts.filter((c) => c.source === source)) {
      out.push(`<circle cx="${f(sx(p.xc))}" cy="${f(sy(p.yc))}" r="1.05" fill="${SOURCE_COLORS[source]}" fill-opacity="0.55"/>`);
    }
  }
  const pw = summary.pooledWithin;
  out.push(line(sx(-m), sy(pw.intercept - pw.slope * m), sx(m), sy(pw.intercept + pw.slope * m), st.ink, 0.6, 'stroke-dasharray="2.2 1.6"'));
  out.push(text(x0, top - 12, "All cells, centred within model", { weight: 700 }));
  out.push(text(x0, top - 3, `${T.term("pooled_within_r")}=${fmtCorr(pw.pearson)}`, { fill: st.muted }));
  out.push(text(x0 + S / 2, top + S + 20, `${xName} − model mean`, { anchor: "middle" }));
  out.push(text(10, top + S / 2, `${yName} − model mean`, { anchor: "middle", rotate: -90 }));

  const labelW = Math.max(...models.map((mm) => textWidth(mm.label, FS)));
  const rx0 = x0 + S + 16 + labelW + 5;
  const rx1 = W - 6;
  const vals = summary.perModel.flatMap((p) => [p.pearson, p.spearman]);
  const dlo = Math.floor((Math.min(...vals) - 0.02) * 10) / 10;
  const rsx = (v) => rx0 + ((v - dlo) / (1 - dlo)) * (rx1 - rx0);
  const pitch = S / models.length;
  for (let t = dlo; t <= 1 + 1e-9; t += 0.1) {
    const v = Math.round(t * 10) / 10;
    out.push(line(rsx(v), top, rsx(v), top + S, st.grid, 0.35, 'data-layer="grid"'));
    out.push(text(rsx(v), top + S + 9, tickLabel(v), { anchor: "middle", fill: st.muted }));
  }
  summary.perModel.forEach((p, i) => {
    const y = top + pitch * (i + 0.5);
    out.push(text(rx0 - 5, y + 2.8, p.label, { anchor: "end" }));
    out.push(line(rsx(Math.min(p.pearson, p.spearman)), y, rsx(Math.max(p.pearson, p.spearman)), y, st.axis, 0.5));
    out.push(`<circle cx="${f(rsx(p.pearson))}" cy="${f(y)}" r="2" fill="${st.ink}"/>`);
    out.push(`<circle cx="${f(rsx(p.spearman))}" cy="${f(y)}" r="2" fill="#FFFFFF" stroke="${st.ink}" stroke-width="0.7"/>`);
  });
  out.push(line(rx0, top + S, rx1, top + S, st.axis, 0.5));
  out.push(`<circle cx="${f(rx0 + 2)}" cy="${f(top - 14.8)}" r="2" fill="${st.ink}"/>`);
  out.push(text(rx0 + 7, top - 12, PEARSON));
  out.push(`<circle cx="${f(rx0 + 2)}" cy="${f(top - 5.8)}" r="2" fill="#FFFFFF" stroke="${st.ink}" stroke-width="0.7"/>`);
  out.push(text(rx0 + 7, top - 3, SPEARMAN));
  out.push(text((rx0 + rx1) / 2, top + S + 20, "within model", { anchor: "middle" }));
  const height = top + S + 26;
  return { width: W, height, body: out.join(""), desc: `Pooled within-model-centred scatter of ${pts.length} cells, and per-model Pearson and Spearman coefficients.` };
}

function heatmap() {
  const out = [];
  const labelW = Math.max(...models.map((mm) => textWidth(mm.label, FS)));
  const rot = 60;
  const labelRise = labelW * Math.sin((rot * Math.PI) / 180);
  const overflow = labelW * Math.cos((rot * Math.PI) / 180);
  const stripX = 17;
  const bx = 25;
  const blockGap = 12;
  const cw = (W - bx - blockGap - overflow - 2) / (2 * models.length);
  const blockW = cw * models.length;
  const y0 = 14 + labelRise + 4;
  const rh = Math.max(3, Math.min(9, 250 / nRel));
  const order = summary.perRelation;
  const blocks = [
    { x0: bx, field: "x", ramp: st.x_ramp, name: xName, color: st.x_color },
    { x0: bx + blockW + blockGap, field: "y", ramp: st.y_ramp, name: yName, color: st.y_color },
  ];
  for (const block of blocks) {
    out.push(text(block.x0 + blockW / 2, 9, block.name, { anchor: "middle", weight: 700, fill: block.color }));
    models.forEach((model, j) => {
      const cx = block.x0 + cw * (j + 0.5);
      out.push(text(cx - 1, y0 - 3, model.label, { rotate: -rot }));
      order.forEach((rel, i) => {
        const cell = cells.find((c) => c.model === model.id && c.relation === rel.relation);
        const y = y0 + i * rh;
        if (!cell) {
          out.push(`<rect x="${f(block.x0 + cw * j)}" y="${f(y)}" width="${f(cw)}" height="${f(rh)}" fill="#FFFFFF"/>`);
          out.push(line(block.x0 + cw * j + 1, y + rh / 2, block.x0 + cw * (j + 1) - 1, y + rh / 2, "#9CA3AF", 0.4));
        } else {
          out.push(`<rect x="${f(block.x0 + cw * j)}" y="${f(y)}" width="${f(cw + 0.05)}" height="${f(rh + 0.05)}" fill="${ramp(block.ramp, cell[block.field])}"/>`);
        }
      });
    });
    out.push(`<rect x="${f(block.x0)}" y="${f(y0)}" width="${f(blockW)}" height="${f(rh * nRel)}" fill="none" stroke="${st.axis}" stroke-width="0.4"/>`);
  }
  order.forEach((rel, i) => {
    out.push(`<rect x="${f(stripX)}" y="${f(y0 + i * rh)}" width="4.5" height="${f(rh + 0.05)}" fill="${SOURCE_COLORS[rel.source]}"/>`);
  });
  const yEnd = y0 + rh * nRel;

  const cbY = yEnd + 6;
  for (const block of blocks) {
    const steps = 48;
    for (let k = 0; k < steps; k += 1) {
      out.push(`<rect x="${f(block.x0 + (blockW * k) / steps)}" y="${f(cbY)}" width="${f(blockW / steps + 0.05)}" height="4.5" fill="${ramp(block.ramp, (k + 0.5) / steps)}"/>`);
    }
    for (const t of [0, 0.5, 1]) out.push(text(block.x0 + blockW * t, cbY + 13, tickLabel(t), { anchor: t === 0 ? "start" : t === 1 ? "end" : "middle", fill: st.muted }));
  }
  const legY = cbY + 24;
  let keyX = stripX;
  for (const source of presentSources) {
    const label = sourceName[source];
    out.push(`<rect x="${f(keyX)}" y="${f(legY - 6)}" width="4.5" height="6" fill="${SOURCE_COLORS[source]}"/>`);
    out.push(text(keyX + 7, legY, label));
    keyX += 7 + textWidth(label, FS) + 8;
  }
  out.push(text(8.5, y0 + (rh * nRel) / 2, `relations, by mean ${yName} →`, { anchor: "middle", rotate: -90 }));
  const height = legY + 4;
  return { width: W, height, body: out.join(""), desc: `Heatmap, ${nRel} relations by ${models.length} models, twin blocks.` };
}

function heatmapInterleaved() {
  const out = [];
  const labelW = Math.max(...models.map((mm) => textWidth(mm.label, FS)));
  const rot = 60;
  const labelRise = labelW * Math.sin((rot * Math.PI) / 180);
  const overflow = labelW * Math.cos((rot * Math.PI) / 180);
  const stripX = 17;
  const bx = 25;
  const cw = Math.min(30, (W - bx - overflow - 2) / models.length);
  const blockW = cw * models.length;
  const headY = 9;
  const y0 = headY + 4 + labelRise + 4;
  const pitch = Math.max(5.4, Math.min(14, 430 / nRel));
  const gap = Math.max(1, pitch * 0.18);
  const sh = (pitch - gap) / 2;
  const order = summary.perRelation;
  const fields = [
    { field: "x", ramp: st.x_ramp, name: xName, color: st.x_color, dy: 0 },
    { field: "y", ramp: st.y_ramp, name: yName, color: st.y_color, dy: sh },
  ];

  const keyA = `top strip: ${xName}`;
  const keyB = `bottom strip: ${yName}`;
  out.push(`<rect x="${f(bx)}" y="${f(headY - 6.5)}" width="9" height="3.2" fill="${ramp(st.x_ramp, 0.75)}"/>`);
  out.push(`<rect x="${f(bx)}" y="${f(headY - 3.3)}" width="9" height="3.2" fill="${ramp(st.y_ramp, 0.75)}"/>`);
  out.push(text(bx + 12, headY, keyA, { weight: 700, fill: st.x_color }));
  out.push(text(bx + 12 + textWidth(keyA, FS) + 12, headY, keyB, { weight: 700, fill: st.y_color }));
  models.forEach((model, j) => {
    out.push(text(bx + cw * (j + 0.5) - 1, y0 - 3, model.label, { rotate: -rot }));
  });
  order.forEach((rel, i) => {
    const yp = y0 + i * pitch;
    out.push(`<rect x="${f(stripX)}" y="${f(yp)}" width="4.5" height="${f(2 * sh)}" fill="${SOURCE_COLORS[rel.source]}"/>`);
    models.forEach((model, j) => {
      const cell = cells.find((c) => c.model === model.id && c.relation === rel.relation);
      for (const fd of fields) {
        const y = yp + fd.dy;
        const attrs = `data-model="${xml(model.id)}" data-relation="${xml(rel.relation)}" data-field="${fd.field}"`;
        if (!cell) {
          out.push(`<rect x="${f(bx + cw * j)}" y="${f(y)}" width="${f(cw)}" height="${f(sh)}" fill="#FFFFFF" ${attrs} data-missing="1"/>`);
          out.push(line(bx + cw * j + 1, y + sh / 2, bx + cw * (j + 1) - 1, y + sh / 2, "#9CA3AF", 0.4));
        } else {
          out.push(`<rect x="${f(bx + cw * j)}" y="${f(y)}" width="${f(cw + 0.05)}" height="${f(sh + 0.02)}" fill="${ramp(fd.ramp, cell[fd.field])}" ${attrs}/>`);
        }
      }
    });
  });
  const yEnd = y0 + pitch * nRel - gap;
  for (let j = 1; j < models.length; j += 1) out.push(line(bx + cw * j, y0, bx + cw * j, yEnd, "#FFFFFF", 0.5));
  out.push(`<rect x="${f(bx)}" y="${f(y0)}" width="${f(blockW)}" height="${f(yEnd - y0)}" fill="none" stroke="${st.axis}" stroke-width="0.4"/>`);

  const cbY = yEnd + 6;
  const cbW = (blockW - 14) / 2;
  fields.forEach((fd, k) => {
    const x0 = bx + k * (cbW + 14);
    const steps = 40;
    for (let s = 0; s < steps; s += 1) {
      out.push(`<rect x="${f(x0 + (cbW * s) / steps)}" y="${f(cbY)}" width="${f(cbW / steps + 0.05)}" height="4.5" fill="${ramp(fd.ramp, (s + 0.5) / steps)}"/>`);
    }
    for (const t of [0, 0.5, 1]) out.push(text(x0 + cbW * t, cbY + 13, tickLabel(t), { anchor: t === 0 ? "start" : t === 1 ? "end" : "middle", fill: st.muted }));
    out.push(text(x0 + cbW / 2, cbY + 23, fd.name, { anchor: "middle", fill: fd.color }));
  });
  const legY = cbY + 35;
  let keyX = stripX;
  for (const source of presentSources) {
    const label = sourceName[source];
    out.push(`<rect x="${f(keyX)}" y="${f(legY - 6)}" width="4.5" height="6" fill="${SOURCE_COLORS[source]}"/>`);
    out.push(text(keyX + 7, legY, label));
    keyX += 7 + textWidth(label, FS) + 8;
  }
  out.push(text(8.5, y0 + (yEnd - y0) / 2, `relations, by mean ${yName} →`, { anchor: "middle", rotate: -90 }));
  const height = legY + 4;
  return { width: W, height, body: out.join(""), desc: `Heatmap, ${nRel} relations by ${models.length} models; each relation is a pair of strips, ${xName} above ${yName}.` };
}

function modelSpread() {
  const out = [];
  const top = 16;
  const pitch = 21;
  const labelW = Math.max(...models.map((mm) => textWidth(mm.label, FS)));
  const x0 = 4 + labelW + 6;
  const x1 = W - 8;
  const sx = (v) => x0 + v * (x1 - x0);
  const yEnd = top + pitch * models.length;
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    out.push(line(sx(t), top, sx(t), yEnd, st.grid, 0.35, 'data-layer="grid"'));
    out.push(text(sx(t), yEnd + 9, tickLabel(t), { anchor: "middle", fill: st.muted }));
  }
  out.push(line(x0, yEnd, x1, yEnd, st.axis, 0.5));
  models.forEach((model, i) => {
    const yc = top + pitch * (i + 0.5);
    out.push(text(x0 - 6, yc + 2.8, model.label, { anchor: "end" }));
    const own = cells.filter((c) => c.model === model.id);
    const s = summary.perModel[i];
    for (const [field, color, dy, q] of [["x", st.x_color, -3.7, s.x_quartiles], ["y", st.y_color, 3.7, s.y_quartiles]]) {
      const y = yc + dy;
      out.push(`<rect x="${f(sx(q[0]))}" y="${f(y - 2.8)}" width="${f(sx(q[2]) - sx(q[0]))}" height="5.6" fill="${color}" fill-opacity="0.22"/>`);
      for (const cell of own) out.push(line(sx(cell[field]), y - 2.6, sx(cell[field]), y + 2.6, color, 0.45, 'stroke-opacity="0.75"'));
      out.push(line(sx(q[1]), y - 3.6, sx(q[1]), y + 3.6, st.ink, 1.1));
      const m = meanOf(own.map((c) => c[field]));
      out.push(meanMark(sx(m), y, st.ink, m, `data-model="${xml(model.id)}" data-field="${field}"`));
    }
  });
  out.push(`<rect x="${f(x0)}" y="2.5" width="9" height="5" fill="${st.x_color}"/>`);
  out.push(text(x0 + 12, 8, xName));
  const lx = x0 + 12 + textWidth(xName, FS) + 12;
  out.push(`<rect x="${f(lx)}" y="2.5" width="9" height="5" fill="${st.y_color}"/>`);
  out.push(text(lx + 12, 8, yName));
  const allFull = summary.perModel.every((p) => p.n === nRel);
  out.push(text((x0 + x1) / 2, yEnd + 20, `value per relation${allFull ? "" : " (some models lack relations)"}`, { anchor: "middle" }));
  out.push(text((x0 + x1) / 2, yEnd + 31, "thin tick = one relation · band = middle 50% · black = median · ◇ = mean", { anchor: "middle", fill: st.muted }));
  return { width: W, height: yEnd + 35, body: out.join(""), desc: `Per-model spread over ${nRel} relations of both measures.` };
}

function rankAgreement() {
  const out = [];
  const n = models.length;
  const c = 19;
  const labelW = Math.max(...models.map((mm) => textWidth(mm.label, FS)));
  const rot = 60;
  const labelRise = labelW * Math.sin((rot * Math.PI) / 180);
  const barsW = 60;
  const totalW = labelW + 5 + n * c + 14 + barsW;
  const x0 = (W - totalW) / 2 + labelW + 5;
  const top = 26;
  const values = [];
  for (let i = 0; i < n; i += 1) for (let j = 0; j < n; j += 1) if (i !== j) values.push(i < j ? summary.agreement.x[i][j].spearman : summary.agreement.y[i][j].spearman);
  const lo = Math.max(0, Math.floor((Math.min(...values) - 0.02) * 10) / 10);
  const norm = (v) => (v - lo) / (1 - lo);
  for (let i = 0; i < n; i += 1) {
    out.push(text(x0 - 4, top + c * (i + 0.5) + 2.8, models[i].label, { anchor: "end" }));
    out.push(text(x0 + c * (i + 0.5) + 1, top + n * c + 4, models[i].label, { anchor: "end", rotate: -rot }));
    for (let j = 0; j < n; j += 1) {
      const x = x0 + c * j;
      const y = top + c * i;
      if (i === j) {
        out.push(`<rect x="${f(x)}" y="${f(y)}" width="${c}" height="${c}" fill="#F3F4F6"/>`);
        continue;
      }
      const upper = i < j;
      const v = upper ? summary.agreement.x[i][j].spearman : summary.agreement.y[i][j].spearman;
      const fill = ramp(upper ? st.x_ramp : st.y_ramp, 0.08 + 0.92 * norm(v));
      out.push(`<rect x="${f(x)}" y="${f(y)}" width="${c}" height="${c}" fill="${fill}" stroke="#FFFFFF" stroke-width="0.5"/>`);
      out.push(text(x + c / 2, y + c / 2 + 2.8, fmtCorr(v), { anchor: "middle", fill: luminance(fill) < 0.3 ? "#FFFFFF" : st.ink }));
    }
  }
  out.push(text(x0 + (n * c) / 2, 9, `Above diagonal: ${xName}`, { anchor: "middle", weight: 700, fill: st.x_color }));
  out.push(text(x0 + (n * c) / 2, 19, `Below diagonal: ${yName}`, { anchor: "middle", weight: 700, fill: st.y_color }));

  const bx = x0 + n * c + 14;
  [[st.x_ramp, 0], [st.y_ramp, 1]].forEach(([r, k]) => {
    const x = bx + k * 26;
    const steps = 40;
    for (let s = 0; s < steps; s += 1) {
      out.push(`<rect x="${f(x)}" y="${f(top + ((n * c) * s) / steps)}" width="6" height="${f((n * c) / steps + 0.05)}" fill="${ramp(r, 0.08 + 0.92 * (1 - (s + 0.5) / steps))}"/>`);
    }
    out.push(text(x + 8, top + 6, "1"), text(x + 8, top + n * c, tickLabel(lo)));
  });
  out.push(text(bx + 30, top + n * c + 12, `${SPEARMAN} between models`, { anchor: "middle", fill: st.muted }));
  out.push(text(bx + 30, top + n * c + 22, "over relations", { anchor: "middle", fill: st.muted }));
  const height = top + n * c + labelRise + 8;
  return { width: W, height, body: out.join(""), desc: `Pairwise Spearman agreement between ${n} models of the relation ordering, both measures.` };
}

function appendixStrips(withRecall = false, part = null) {
  const out = [];
  const size = part ? Math.ceil(summary.perRelation.length / part[1]) : summary.perRelation.length;
  const order = part ? summary.perRelation.slice((part[0] - 1) * size, part[0] * size) : summary.perRelation;
  if (withRecall && order.some((r) => r.mean_recall === undefined)) fail("appendix-strips-recall needs PRE-REL recall on every cell");
  const pitch = withRecall ? 12 : Math.min(10.5, 585 / nRel);

  const labelW = Math.max(...summary.perRelation.map((r) => textWidth(FIG_LABEL.get(r.relation), FS)));
  const x0 = 4 + 7 + labelW + 6;
  const x1 = W - 6;
  const sx = (v) => x0 + v * (x1 - x0);
  const rName = withRecall ? inputs.covariateName("recall") : null;
  const tracks = withRecall
    ? [["x", st.x_color, -3], ["y", st.y_color, 0], ["recall", st.ink, 3]]
    : [["x", st.x_color, -1.7], ["y", st.y_color, 1.7]];

  const dotGlyph = (color, r = 1.5) => (cx, cy) => `<circle cx="${f(cx)}" cy="${f(cy)}" r="${r}" fill="${color}"/>`;
  const blocks = [
    { id: "measure", title: T.term("key_measure"), items: [[xName, st.x_color], [yName, st.y_color], ...(withRecall ? [[rName, st.ink]] : [])].map(([label, color]) => ({ label, glyph: dotGlyph(color, 1.8) })) },
    { id: "over_models", title: T.term("key_over_models"), items: [
      { id: "median", label: T.term("median"), glyph: dotGlyph(st.ink) },
      { id: "mean", label: T.term("mean"), glyph: (cx, cy) => `<path d="M${f(cx)},${f(cy - 1.9)} L${f(cx + 1.9)},${f(cy)} L${f(cx)},${f(cy + 1.9)} L${f(cx - 1.9)},${f(cy)} Z" fill="#FFFFFF" stroke="${st.ink}" stroke-width="0.6"/>` },
      { id: "range", label: T.term("range_over_models"), glyph: (cx, cy) => line(cx - 5, cy, cx + 5, cy, st.ink, 1.0, 'stroke-opacity="0.45"') },
    ] },
  ];
  const setItems = MARKS === "sources" ? presentSources.map((src) => ({ id: "set:" + src, label: sourceName[src], glyph: (cx, cy) => glyphMark(SM[src], cx, cy, "", glyphColor(src)) })) : [];
  for (const h of hl.values()) setItems.push({ id: "hl:" + h.id, label: h.tag ? `${h.tag} example` : FIG_LABEL.get(h.id), glyph: (cx, cy) => shapeMark(h.shape, cx, cy, 2.1, `fill="${HL}"`, "") });
  if (setItems.length) blocks.push({ id: "set", title: T.term("key_relation_set"), items: setItems });
  const key = keyBlocks(blocks, { itemsLeft: 7 - 2, right: x1, y: 8, fs: FS, textWidth, text });
  const top = 8 + key.height;
  const yEnd = top + pitch * order.length;
  const stat = (rel, field) => field === "recall"
    ? [rel.recall_min, rel.recall_max, rel.recall_median, rel.mean_recall]
    : [rel[field + "_min"], rel[field + "_max"], rel[field + "_median"], rel["mean_" + field]];
  order.forEach((rel, i) => {
    if (i % 2 === 0) out.push(`<rect data-layer="band" x="0" y="${f(top + pitch * i)}" width="${f(W)}" height="${f(pitch)}" fill="#F7F7F8"/>`);
  });
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    out.push(line(sx(t), top, sx(t), yEnd, st.grid, 0.35, 'data-layer="grid"'));
    out.push(text(sx(t), yEnd + 9, tickLabel(t), { anchor: "middle", fill: st.muted }));
  }
  order.forEach((rel, i) => {
    const yc = top + pitch * (i + 0.5);

    out.push(rowMark(rel, 7, yc));
    out.push(text(x0 - 5, yc + 2.8, FIG_LABEL.get(rel.relation), { anchor: "end" }));
    for (const [field, color, dy] of tracks) {
      const [lo, hi, med, avg] = stat(rel, field);
      out.push(line(sx(lo), yc + dy, sx(hi), yc + dy, color, 1.0, 'stroke-opacity="0.45"'));
      out.push(`<circle cx="${f(sx(med))}" cy="${f(yc + dy)}" r="1.5" fill="${color}"/>`);
      out.push(meanMark(sx(avg), yc + dy, color, avg, `data-relation="${xml(rel.relation)}" data-field="${field}"`));
    }
  });
  out.push(line(x0, yEnd, x1, yEnd, st.axis, 0.5));
  out.push(key.svg);
  return { width: W, height: yEnd + 13, body: out.join(""), desc: `Every relation: cross-model median, mean and range of ${withRecall ? `${xName}, ${yName} and ${rName}` : "both measures"}, ordered by mean ${yName}.` };
}

const covScales = covariateScales(spec, cells);

const hlCatalogue = new Map(((spec.highlight || {}).relations || []).map((h) => [h.id, h]));
const hlActive = spec.highlight_relations ?? [];
if (!Array.isArray(hlActive)) fail("spec.highlight_relations must be a list of relation ids");
for (const id of hlActive) if (!hlCatalogue.has(id)) fail(`highlight_relations names ${id}, which spec.highlight.relations gives no shape`);
const hl = new Map(hlActive.filter((id) => allRelations.includes(id)).map((id) => [id, hlCatalogue.get(id)]));

const SM = SOURCE_MARKS;
let MARKS = "sources";

let COLOR = "neutral";

let ROW_ORDER = "outcome";
let MODELS_ONCE = false;
const HL = st.highlight_color;
const TINT = st.tint;
if (!/^#[0-9A-F]{6}$/.test(TINT || "")) fail("spec.style.tint must be #RRGGBB");
const covColor = (name, value) => ramp(st[spec.covariates[name].ramp], covariateT(covScales[name], value));

function shapeMark(shape, x, y, r, paint, attrs) {
  const k = shape === "square" ? 0.9 : shape === "diamond" ? 1.25 : shape === "triangle" ? 1.3 : 1;
  const R = r * k;
  if (shape === "square") return `<rect x="${f(x - R)}" y="${f(y - R)}" width="${f(2 * R)}" height="${f(2 * R)}" ${paint} ${attrs}/>`;
  if (shape === "diamond") return `<path d="M${f(x)},${f(y - R)} L${f(x + R)},${f(y)} L${f(x)},${f(y + R)} L${f(x - R)},${f(y)} Z" ${paint} ${attrs}/>`;
  if (shape === "triangle") return `<path d="M${f(x)},${f(y - R)} L${f(x + R * 0.95)},${f(y + R * 0.6)} L${f(x - R * 0.95)},${f(y + R * 0.6)} Z" ${paint} ${attrs}/>`;
  return `<circle cx="${f(x)}" cy="${f(y)}" r="${f(R)}" ${paint} ${attrs}/>`;
}

function cellMark(cell, px, py, encoding, r = 1.25, modelColor = null) {
  const h = hl.get(cell.relation);
  const glyph = MARKS === "sources" ? SM[cell.source] : "dot";
  const role = MARKS === "sources" ? cell.source : "point";
  const base = `data-model="${xml(cell.model)}" data-relation="${xml(cell.relation)}"`;
  const color = encoding === "neutral" ? (h ? HL : st.neutral)
    : encoding === "sets" ? (h ? HL : SOURCE_COLORS[cell.source])
    : encoding === "tint" ? (h ? HL : TINT)
    : encoding === "model" ? modelColor
    : covColor(encoding, cell.cov[encoding]);
  const flat = encoding === "neutral" || encoding === "sets" || encoding === "tint";
  if (h) {
    const paint = flat || encoding === "model"
      ? `fill="${color}" stroke="#FFFFFF" stroke-width="0.35"`
      : `fill="${color}" stroke="${st.ink}" stroke-width="0.6"`;
    return [1, shapeMark(h.shape, px, py, r * 1.55, paint, `${base} data-role="highlight" data-shape="${h.shape}" data-c="${color}"`)];
  }
  if (glyph === "ring") {
    const rc = encoding === "neutral" ? st.neutral_open : color;
    return [0, `<circle cx="${f(px)}" cy="${f(py)}" r="${f(r * 1.05)}" fill="none" stroke="${rc}" stroke-width="${encoding === "neutral" ? 0.55 : 0.8}" ${base} data-role="${role}" data-shape="ring" data-c="${rc}"/>`];
  }
  const op = encoding === "neutral" ? 0.55 : flat ? 0.8 : 0.9;
  const stroke = flat ? "" : ` stroke="${st.muted}" stroke-width="0.25"`;
  if (glyph === "triangle") {
    return [0, shapeMark("triangle", px, py, r * 1.2, `fill="${color}" fill-opacity="${op}"${stroke}`, `${base} data-role="${role}" data-shape="triangle" data-c="${color}"`)];
  }
  return [0, `<circle cx="${f(px)}" cy="${f(py)}" r="${f(r)}" fill="${color}" fill-opacity="${op}"${stroke} ${base} data-role="${role}" data-shape="dot" data-c="${color}"/>`];
}

function drawCells(own, encoding, sx, sy, r, modelColorOf) {
  const layers = [[], []];
  for (const cell of own) {
    const [layer, svg] = cellMark(cell, sx(cell), sy(cell), encoding, r, modelColorOf ? modelColorOf(cell.model) : null);
    layers[layer].push(svg);
  }
  return layers[0].join("") + layers[1].join("");
}

function glyphMark(glyph, x, y, attrs, color = null) {
  if (glyph === "ring") return `<circle cx="${f(x)}" cy="${f(y)}" r="1.5" fill="none" stroke="${color || st.neutral_open}" stroke-width="${color ? 0.8 : 0.6}"${attrs ? " " + attrs : ""}/>`;
  if (glyph === "triangle") return shapeMark("triangle", x, y, 1.6, `fill="${color || st.neutral}"`, attrs);
  return `<circle cx="${f(x)}" cy="${f(y)}" r="1.4" fill="${color || st.neutral}"${attrs ? " " + attrs : ""}/>`;
}

const glyphColor = (source, encoding = null) => (COLOR === "sets" || encoding === "sets" ? SOURCE_COLORS[source] : encoding === "tint" ? TINT : null);

function rowMark(rel, x, y) {
  const h = hl.get(rel.relation);
  if (h) return shapeMark(h.shape, x, y, 2.1, `fill="${HL}"`, `data-relation="${xml(rel.relation)}" data-role="highlight-row"`);
  if (MARKS !== "sources") return "";
  return glyphMark(SM[rel.source], x, y, `data-relation="${xml(rel.relation)}" data-role="source-row" data-source="${rel.source}"`, glyphColor(rel.source));
}

function covBar(name, x0, y0, w) {
  const out = [];
  const sc = covScales[name];
  const steps = 40;
  for (let s = 0; s < steps; s += 1) out.push(`<rect x="${f(x0 + (w * s) / steps)}" y="${f(y0)}" width="${f(w / steps + 0.05)}" height="4.5" fill="${ramp(st[spec.covariates[name].ramp], (s + 0.5) / steps)}"/>`);
  const ticks = sc.scale === "log"
    ? [2, 4, 8, 16, 32, 64, 128, 256, 512].filter((v) => v >= sc.lo * 0.999 && v <= sc.hi * 1.001)
    : [0, 0.5, 1].map((t) => sc.lo + t * (sc.hi - sc.lo));
  const shown = sc.scale === "log" ? [ticks[0], ticks[Math.floor(ticks.length / 2)], ticks[ticks.length - 1]] : ticks;
  for (const v of shown) {
    const t = covariateT(sc, v);
    out.push(line(x0 + w * t, y0 + 4.5, x0 + w * t, y0 + 6.5, st.axis, 0.4));
    out.push(text(x0 + w * t, y0 + 14, sc.scale === "log" ? String(v) : tickLabel(v), { anchor: t < 0.05 ? "start" : t > 0.95 ? "end" : "middle", fill: st.muted }));
  }
  out.push(text(x0, y0 - 3, `${inputs.covariateName(name)}${sc.scale === "log" ? " (log scale)" : ""}`));
  return out.join("");
}

function sourceKeyItems() {
  if (MARKS !== "sources") return [];
  return presentSources.map((s) => ["src:" + s, `${sourceName[s]} relation`]);
}
function sharedKeyItems(encoding) {
  const items = [...sourceKeyItems()];
  for (const h of hl.values()) items.push(["hl:" + h.shape, `${inputs.labelOf.get(h.id)}${h.tag ? ` (${h.tag})` : ""}`]);
  items.push(["dash", "OLS fit, all relations"]);
  return items;
}
function drawKeyItems(items, kx, ky, encoding) {
  const out = [];
  items.forEach(([kind, label], k) => {
    const y = ky + k * 10.5;
    const cy = y - 2.8;
    if (kind === "dot") out.push(glyphMark("dot", kx + 5, cy, "", encoding === "tint" ? TINT : null));
    if (kind.startsWith("src:")) out.push(glyphMark(SM[kind.slice(4)], kx + 5, cy, "", glyphColor(kind.slice(4), encoding)));
    if (kind.startsWith("hl:")) {
      const paint = ["neutral", "model", "sets", "tint"].includes(encoding) ? `fill="${HL}"` : `fill="#FFFFFF" stroke="${st.ink}" stroke-width="0.6"`;
      out.push(shapeMark(kind.slice(3), kx + 5, cy, 2.2, paint, ""));
    }
    if (kind === "dash") out.push(line(kx, cy, kx + 10, cy, st.ink, 0.6, 'stroke-dasharray="2.2 1.6"'));
    out.push(text(kx + 14, y, label));
  });
  return out.join("");
}

function statsLine(cx, y, pearson, spearman, model) {
  const thin = FS / 6;
  const gap = FS;
  const sym = (s, dx = 0) => mathSym(s, dx ? ` dx="${f(dx)}"` : "");
  const num = (v) => `<tspan dx="${f(thin)}">=</tspan><tspan dx="${f(thin)}">${xml(fmtCorr(v))}</tspan>`;
  return `<text x="${f(cx)}" y="${f(y)}" font-size="${f(FS)}" text-anchor="middle" fill="${st.muted}" font-variant-numeric="tabular-nums" data-stats="${xml(model)}">`
    + sym("r") + num(pearson) + sym("ρ", gap) + num(spearman) + "</text>";
}

const GRID_W = 486;
function scatterGridEnc(encoding, stats = true) {
  const compact = encoding === "sets";
  const cols = compact ? 6 : 4;
  const GW = compact ? GRID_W : W;
  const rowsN = Math.ceil((models.length + 1) / cols);
  const left = compact ? 26 : 30;
  const gapX = compact ? 7 : 11;
  const pw = (GW - left - 3 - gapX * (cols - 1)) / cols;
  const ph = pw * 0.86;
  const titleH = 21;
  const pitch = titleH + ph + 11 + 6;
  const top = 2;
  const height = top + rowsN * pitch + (compact ? 7 : 12);
  const lo = -0.03;
  const hi = 1.03;
  const out = [];
  const lastRowOfCol = (col) => {
    let last = 0;
    for (let i = 0; i < models.length; i += 1) if (i % cols === col) last = Math.floor(i / cols);
    return last;
  };
  models.forEach((model, i) => {
    const col = i % cols;
    const row = Math.floor(i / cols);
    const x0 = left + col * (pw + gapX);
    const y0 = top + row * pitch + titleH;
    const X = (v) => x0 + ((v - lo) / (hi - lo)) * pw;
    const Y = (v) => y0 + ph - ((v - lo) / (hi - lo)) * ph;
    const s = summary.perModel[i];

    const half = textWidth(model.label, FS, !compact) / 2;
    out.push(text(Math.min(GW - 1 - half, Math.max(1 + half, x0 + pw / 2)), y0 - 11, model.label, { anchor: "middle", ...(compact ? {} : { weight: 700 }), data: `data-panel-title="${xml(model.id)}"` }));
    if (stats) out.push(statsLine(x0 + pw / 2, y0 - 2.5, s.pearson, s.spearman, model.id));
    for (const t of [0, 0.5, 1]) {
      out.push(line(X(t), y0, X(t), y0 + ph, st.grid, 0.35, 'data-layer="grid"'), line(x0, Y(t), x0 + pw, Y(t), st.grid, 0.35, 'data-layer="grid"'));
      if (col === 0) out.push(text(x0 - 2.5, Y(t) + 2.8, tickLabel(t), { anchor: "end", fill: st.muted }));
      if (row === lastRowOfCol(col)) out.push(text(X(t), y0 + ph + 9, tickLabel(t), { anchor: "middle", fill: st.muted }));
    }
    out.push(line(x0, y0, x0, y0 + ph, st.axis, 0.5), line(x0, y0 + ph, x0 + pw, y0 + ph, st.axis, 0.5));
    const own = cells.filter((cell) => cell.model === model.id);
    out.push(drawCells(own, encoding, (c) => X(c.x), (c) => Y(c.y), 1.25));
    const xa = Math.max(0, -s.intercept / s.slope);
    const xb = Math.min(1, (1 - s.intercept) / s.slope);
    out.push(line(X(xa), Y(s.intercept + s.slope * xa), X(xb), Y(s.intercept + s.slope * xb), st.ink, 0.6, 'stroke-dasharray="2.2 1.6" stroke-opacity="0.8"'));
  });

  const kx = left + (models.length % cols) * (pw + gapX) + 4;
  let ky = top + Math.floor(models.length / cols) * pitch + titleH + 2;
  if (encoding === "sets") {

    const kx0 = left + (models.length % cols) * (pw + gapX);
    const titleBase = top + Math.floor(models.length / cols) * pitch + titleH - 11;
    const setBlock = { id: "set", title: T.term("key_relation_set"), titleAbove: true, stack: true,
      items: presentSources.map((src) => ({ id: "set:" + src, label: sourceName[src], glyph: (cx, cy) => glyphMark(SM[src], cx, cy, "", SOURCE_COLORS[src]) })) };
    const marksBlock = { id: "marks", title: T.term("key_marks"), titleAbove: true, stack: true,
      items: [{ id: "fit", label: "OLS fit", glyph: (cx, cy) => line(cx - 5, cy, cx + 5, cy, st.ink, 0.6, 'stroke-dasharray="2.2 1.6"') }] };
    out.push(keyBlocks([setBlock, marksBlock], { itemsLeft: kx0, right: GW - 2, y: titleBase, fs: FS, textWidth, text }).svg);
  } else if (["neutral", "tint"].includes(encoding)) {
    out.push(drawKeyItems([...(MARKS === "sources" ? [] : [["dot", "relation"]]), ...sharedKeyItems(encoding)], kx, ky + 4, encoding));
  } else {
    out.push(covBar(encoding, kx, ky + 6, Math.min(150, GW - kx - 8)));
    out.push(drawKeyItems(sharedKeyItems(encoding), kx, ky + 34, encoding));
  }

  const plotRows = Math.ceil(models.length / cols);
  out.push(text(9, top + titleH + ((plotRows - 1) * pitch + ph) / 2, yName, { anchor: "middle", rotate: -90, data: 'data-axis-title="y"' }));
  out.push(text(left + (cols * pw + (cols - 1) * gapX) / 2, height - 3, xName, { anchor: "middle", data: 'data-axis-title="x"' }));
  return { width: GW, height, body: out.join(""), desc: `${models.length} panels; points ${encoding === "neutral" ? "in one neutral colour" : encoding === "sets" ? "coloured and shaped by relation set" : encoding === "tint" ? "in one accent colour" : `coloured by ${inputs.covariateName(encoding)}`}; x ${xName}, y ${yName}.` };
}

function scatterPooledEnc(encoding) {
  const top = 22;
  const S = 172;
  const x0 = 36;
  const out = [];
  const pts = summary.centred;
  const m = Math.ceil(Math.max(...pts.map((p) => Math.max(Math.abs(p.xc), Math.abs(p.yc)))) * 10) / 10;
  const X = (v) => x0 + ((v + m) / (2 * m)) * S;
  const Y = (v) => top + S - ((v + m) / (2 * m)) * S;
  for (let t = -Math.floor(m / 0.25) * 0.25; t <= m + 1e-9; t += 0.25) {
    const v = Math.round(t * 100) / 100;
    out.push(line(X(v), top, X(v), top + S, st.grid, 0.35, 'data-layer="grid"'), line(x0, Y(v), x0 + S, Y(v), st.grid, 0.35, 'data-layer="grid"'));
    const label = v === 0 ? "0" : `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(2).replace(/^0/, "")}`;
    out.push(text(X(v), top + S + 9, label, { anchor: "middle", fill: st.muted }), text(x0 - 2.5, Y(v) + 2.8, label, { anchor: "end", fill: st.muted }));
  }
  out.push(line(x0, top, x0, top + S, st.axis, 0.5), line(x0, top + S, x0 + S, top + S, st.axis, 0.5));
  const modelColorOf = (id) => st.model_palette[models.findIndex((mm) => mm.id === id) % st.model_palette.length];
  out.push(drawCells(pts, encoding, (c) => X(c.xc), (c) => Y(c.yc), 1.05, encoding === "model" ? modelColorOf : null));
  const pw = summary.pooledWithin;
  out.push(line(X(-m), Y(pw.intercept - pw.slope * m), X(m), Y(pw.intercept + pw.slope * m), st.ink, 0.6, 'stroke-dasharray="2.2 1.6"'));
  out.push(text(x0, top - 12, "All cells, centred within model", { weight: 700 }));
  out.push(text(x0, top - 3, `${T.term("pooled_within_r")}=${fmtCorr(pw.pearson)}`, { fill: st.muted }));
  out.push(text(x0 + S / 2, top + S + 20, `${xName} − model mean`, { anchor: "middle" }));
  out.push(text(10, top + S / 2, `${yName} − model mean`, { anchor: "middle", rotate: -90 }));

  const labelW = Math.max(...models.map((mm) => textWidth(mm.label, FS)));
  const rx0 = x0 + S + 16 + labelW + 5;
  const rx1 = W - 6;
  const vals = summary.perModel.flatMap((p) => [p.pearson, p.spearman]);
  const dlo = Math.floor((Math.min(...vals) - 0.02) * 10) / 10;
  const rsx = (v) => rx0 + ((v - dlo) / (1 - dlo)) * (rx1 - rx0);
  const rowPitch = S / models.length;
  for (let t = dlo; t <= 1 + 1e-9; t += 0.1) {
    const v = Math.round(t * 10) / 10;
    out.push(line(rsx(v), top, rsx(v), top + S, st.grid, 0.35, 'data-layer="grid"'), text(rsx(v), top + S + 9, tickLabel(v), { anchor: "middle", fill: st.muted }));
  }
  summary.perModel.forEach((p, i) => {
    const y = top + rowPitch * (i + 0.5);
    const c = encoding === "model" ? modelColorOf(p.model) : st.ink;
    if (encoding === "model") out.push(`<circle cx="${f(rx0 - labelW - 9)}" cy="${f(y)}" r="2" fill="${c}"/>`);
    out.push(text(rx0 - 5, y + 2.8, p.label, { anchor: "end" }));
    out.push(line(rsx(Math.min(p.pearson, p.spearman)), y, rsx(Math.max(p.pearson, p.spearman)), y, st.axis, 0.5));
    out.push(`<circle cx="${f(rsx(p.pearson))}" cy="${f(y)}" r="2" fill="${st.ink}"/>`);
    out.push(`<circle cx="${f(rsx(p.spearman))}" cy="${f(y)}" r="2" fill="#FFFFFF" stroke="${st.ink}" stroke-width="0.7"/>`);
  });
  out.push(line(rx0, top + S, rx1, top + S, st.axis, 0.5));
  out.push(`<circle cx="${f(rx0 + 2)}" cy="${f(top - 14.8)}" r="2" fill="${st.ink}"/>`, text(rx0 + 7, top - 12, PEARSON));
  out.push(`<circle cx="${f(rx0 + 2)}" cy="${f(top - 5.8)}" r="2" fill="#FFFFFF" stroke="${st.ink}" stroke-width="0.7"/>`, text(rx0 + 7, top - 3, SPEARMAN));
  out.push(text((rx0 + rx1) / 2, top + S + 20, "within model", { anchor: "middle" }));
  let height = top + S + 26;
  if (encoding === "neutral") {
    const items = sharedKeyItems(encoding).filter(([k]) => k !== "dash");
    const perRow = 2;
    const colW = (W - x0) / perRow;
    items.forEach((item, k) => out.push(drawKeyItems([item], x0 + (k % perRow) * colW, height + 6 + Math.floor(k / perRow) * 10.5, encoding)));
    height += 8 + Math.ceil(items.length / perRow) * 10.5;
  }
  return { width: W, height, body: out.join(""), desc: `Pooled within-model-centred scatter of ${pts.length} cells (${encoding}), and per-model coefficients.` };
}

const rawOf = { x: (c) => c.x, y: (c) => c.y, recall: (c) => c.cov.recall };
function rankPosition(field) {
  const vals = cells.map(rawOf[field]);
  const r = averageRanks(vals);
  const pos = new Map(cells.map((c, i) => [`${c.model}\u0000${c.relation}`, (r[i] - 1) / (vals.length - 1)]));
  return { t: (c) => pos.get(`${c.model}\u0000${c.relation}`), vals };
}
function heatmapEnc(withRecall, observedX = false, rampMode = null) {
  const out = [];
  const xLo = Math.min(...cells.map((c) => c.x));
  const xHi = Math.max(...cells.map((c) => c.x));
  const fields = [
    { key: "x", ramp: st.x_ramp, name: xName, color: st.x_color, value: observedX ? (c) => (c.x - xLo) / (xHi - xLo) : (c) => c.x, observed: observedX },
    { key: "y", ramp: st.y_ramp, name: yName, color: st.y_color, value: (c) => c.y },
  ];
  if (withRecall) fields.push({ key: "recall", ramp: st[spec.covariates.recall.ramp], name: inputs.covariateName("recall"), color: st.ink, value: (c) => covariateT(covScales.recall, c.cov.recall), raw: true });
  const fmtV = (v) => (v === 0 ? "0" : v === 1 ? "1" : v.toFixed(2).replace(/^0/, ""));
  for (const fd of fields) {
    const transform = rampMode === "sqrt" && fd.key === "x" ? "sqrt"
      : rampMode === "quantile" && fd.key === "x" ? "quantile"
      : rampMode === "quantile-all" ? "quantile" : null;
    if (transform === "sqrt") {
      fd.value = (c) => Math.sqrt(c.x);
      fd.scale = "square-root scale";
      fd.ticks = [0, 0.1, 0.25, 0.5, 1].map((v) => [Math.sqrt(v), fmtV(v)]);
    } else if (transform === "quantile") {
      const { t, vals } = rankPosition(fd.key);
      fd.value = t;
      fd.scale = "quantile scale";
      fd.ticks = [0, 0.25, 0.5, 0.75, 1].map((p) => [p, fmtV(quantile7(vals, p))]);
    }
    if (transform) fd.transform = transform;
  }
  const labelW = Math.max(...models.map((mm) => textWidth(mm.label, FS)));
  const rot = 60;
  const labelRise = labelW * Math.sin((rot * Math.PI) / 180);
  const overflow = labelW * Math.cos((rot * Math.PI) / 180);
  const hlLabels = [...hl.values()].map((h) => inputs.labelOf.get(h.id));
  const rightW = hlLabels.length ? Math.max(...hlLabels.map((l) => textWidth(l, FS))) + 12 : 0;
  const markX = 14;
  const bx = 22;
  const blockGap = withRecall ? 8 : 12;
  const avail = W - bx - rightW - 2 - blockGap * (fields.length - 1);

  const cw = Math.min((avail - (MODELS_ONCE ? 0 : Math.max(0, overflow - rightW))) / (fields.length * models.length), 16);
  const blockW = cw * models.length;
  const y0 = 14 + labelRise + 4;
  const rh = Math.max(3, Math.min(9, 250 / nRel));
  const order = ROW_ORDER === "gain"
    ? [...summary.perRelation].sort((a, b) => b.mean_x - a.mean_x || (a.relation < b.relation ? -1 : 1))
    : summary.perRelation;
  fields.forEach((fd, b) => {
    const x0 = bx + b * (blockW + blockGap);
    fd.x0 = x0;
    out.push(text(x0 + blockW / 2, 9, fd.name, { anchor: "middle", weight: 700, fill: fd.color }));
    models.forEach((model, j) => {
      if (!MODELS_ONCE || b === 0) out.push(text(x0 + cw * (j + 0.5) - 1, y0 - 3, model.label, { rotate: -rot }));
      order.forEach((rel, i) => {
        const cell = cells.find((c) => c.model === model.id && c.relation === rel.relation);
        const y = y0 + i * rh;
        const attrs = `data-model="${xml(model.id)}" data-relation="${xml(rel.relation)}" data-field="${fd.key}"`;
        if (!cell) {
          out.push(`<rect x="${f(x0 + cw * j)}" y="${f(y)}" width="${f(cw)}" height="${f(rh)}" fill="#FFFFFF" ${attrs} data-missing="1"/>`);
        } else {
          out.push(`<rect x="${f(x0 + cw * j)}" y="${f(y)}" width="${f(cw + 0.05)}" height="${f(rh + 0.05)}" fill="${ramp(fd.ramp, fd.value(cell))}" ${attrs}/>`);
        }
      });
    });
    out.push(`<rect x="${f(x0)}" y="${f(y0)}" width="${f(blockW)}" height="${f(rh * nRel)}" fill="none" stroke="${st.axis}" stroke-width="0.4"/>`);
  });
  const rightEdge = bx + fields.length * blockW + (fields.length - 1) * blockGap;

  order.forEach((rel, i) => out.push(rowMark(rel, markX, y0 + (i + 0.5) * rh)));

  const want = order.map((rel, i) => ({ rel, i })).filter(({ rel }) => hl.has(rel.relation)).map(({ rel, i }) => ({ rel, cy: y0 + (i + 0.5) * rh }));
  let last = -Infinity;
  for (const w of want) {
    const ly = Math.max(w.cy, last + 9);
    last = ly;
    out.push(line(rightEdge + 1, w.cy, rightEdge + 5, ly, HL, 0.5));
    out.push(text(rightEdge + 7, ly + 2.8, inputs.labelOf.get(w.rel.relation), { fill: st.ink }));
  }
  const yEnd = y0 + rh * nRel;
  const cbY = yEnd + 6;
  for (const fd of fields) {
    const steps = 40;
    for (let k = 0; k < steps; k += 1) out.push(`<rect x="${f(fd.x0 + (blockW * k) / steps)}" y="${f(cbY)}" width="${f(blockW / steps + 0.05)}" height="4.5" fill="${ramp(fd.ramp, (k + 0.5) / steps)}"/>`);
    if (fd.transform) {
      for (const [t, label] of fd.ticks) {
        out.push(line(fd.x0 + blockW * t, cbY + 4.5, fd.x0 + blockW * t, cbY + 6.5, st.axis, 0.4));
        out.push(text(fd.x0 + blockW * t, cbY + 14, label, { anchor: t < 0.05 ? "start" : t > 0.95 ? "end" : "middle", fill: st.muted }).replace("<text ", `<text data-bar="${fd.key}" data-t="${t}" `));
      }
      out.push(text(fd.x0 + blockW / 2, cbY + 24, fd.scale, { anchor: "middle", fill: st.muted }).replace("<text ", `<text data-scale="${fd.key}" `));
      continue;
    }
    const ends = fd.observed ? [xLo.toFixed(2).replace(/^0/, ""), xHi.toFixed(2).replace(/^0/, "")] : [tickLabel(0), tickLabel(1)];
    for (const t of [0, 1]) out.push(text(fd.x0 + blockW * t, cbY + 13, ends[t], { anchor: t === 0 ? "start" : "end", fill: st.muted }));
    if (fd.observed) out.push(text(fd.x0 + blockW / 2, cbY + 13, "observed range", { anchor: "middle", fill: st.muted }));
  }

  const barTextBase = cbY + (fields.some((fd) => fd.transform) ? 24 : 13);
  let height = keyBottom(barTextBase, FS);

  const enc = COLOR === "sets" ? "sets" : "neutral";
  const setItems = MARKS === "sources" ? presentSources.map((src) => ({ id: "set:" + src, label: sourceName[src], glyph: (cx, cy) => glyphMark(SM[src], cx, cy, "", glyphColor(src, enc)) })) : [];
  for (const h of hl.values()) setItems.push({ id: "hl:" + h.id, label: `${FIG_LABEL.get(h.id)}${h.tag ? ` (${h.tag})` : ""}`, glyph: (cx, cy) => shapeMark(h.shape, cx, cy, 2.2, `fill="${HL}"`, "") });

  if (setItems.length) {
    const keyY = keyBaselineBelow(barTextBase, FS);
    const block = [{ id: "set", title: T.term("key_relation_set"), items: setItems }];
    const opts = { right: W - 4, y: keyY, fs: FS, textWidth, text, clear: true };
    let key = keyBlocks(block, { ...opts, itemsLeft: markX - 2 });
    if (process.env.HEATMAP_KEY_ALIGN === "centre") key = keyBlocks(block, { ...opts, itemsLeft: (bx + rightEdge) / 2 - key.width / 2 });
    out.push(key.svg);
    height = keyBottom(key.lastBaseline, FS);
  }
  out.push(text(6, y0 + (rh * nRel) / 2, `relations, by mean ${ROW_ORDER === "gain" ? xName : yName} →`, { anchor: "middle", rotate: -90 }));
  return { width: W, height, body: out.join(""), desc: `Heatmap, ${nRel} relations by ${models.length} models, blocks ${fields.map((fd) => fd.name).join(", ")}.` };
}

function modelSpreadHl() {
  const out = [];
  const top = 16;
  const pitch = 21;
  const labelW = Math.max(...models.map((mm) => textWidth(mm.label, FS)));
  const x0 = 4 + labelW + 6;
  const x1 = W - 8;
  const X = (v) => x0 + v * (x1 - x0);
  const yEnd = top + pitch * models.length;
  for (const t of [0, 0.25, 0.5, 0.75, 1]) out.push(line(X(t), top, X(t), yEnd, st.grid, 0.35, 'data-layer="grid"'), text(X(t), yEnd + 9, tickLabel(t), { anchor: "middle", fill: st.muted }));
  out.push(line(x0, yEnd, x1, yEnd, st.axis, 0.5));
  models.forEach((model, i) => {
    const yc = top + pitch * (i + 0.5);
    out.push(text(x0 - 6, yc + 2.8, model.label, { anchor: "end" }));
    const own = cells.filter((c) => c.model === model.id);
    const s = summary.perModel[i];
    for (const [field, color, dy, q] of [["x", st.x_color, -3.7, s.x_quartiles], ["y", st.y_color, 3.7, s.y_quartiles]]) {
      const y = yc + dy;
      out.push(`<rect x="${f(X(q[0]))}" y="${f(y - 2.8)}" width="${f(X(q[2]) - X(q[0]))}" height="5.6" fill="${color}" fill-opacity="0.22"/>`);
      for (const cell of own) if (!hl.has(cell.relation)) out.push(line(X(cell[field]), y - 2.6, X(cell[field]), y + 2.6, color, 0.45, 'stroke-opacity="0.75"'));
      out.push(line(X(q[1]), y - 3.6, X(q[1]), y + 3.6, st.ink, 1.1));
      const m = meanOf(own.map((c) => c[field]));
      out.push(meanMark(X(m), y, st.ink, m, `data-model="${xml(model.id)}" data-field="${field}"`));
      for (const cell of own) {
        const h = hl.get(cell.relation);
        if (h) out.push(shapeMark(h.shape, X(cell[field]), y, 2, `fill="${HL}" stroke="#FFFFFF" stroke-width="0.3"`, `data-model="${xml(cell.model)}" data-relation="${xml(cell.relation)}" data-role="highlight" data-field="${field}"`));
      }
    }
  });
  out.push(`<rect x="${f(x0)}" y="2.5" width="9" height="5" fill="${st.x_color}"/>`, text(x0 + 12, 8, xName));
  const lx = x0 + 12 + textWidth(xName, FS) + 12;
  out.push(`<rect x="${f(lx)}" y="2.5" width="9" height="5" fill="${st.y_color}"/>`, text(lx + 12, 8, yName));
  let height = yEnd + 20;
  out.push(text((x0 + x1) / 2, height, "thin tick = one relation · band = middle 50% · black = median · ◇ = mean", { anchor: "middle", fill: st.muted }));
  const items = [...hl.values()].map((h) => ["hl:" + h.shape, `${inputs.labelOf.get(h.id)}${h.tag ? ` (${h.tag})` : ""}`]);
  const colW = (W - 4) / 2;
  items.forEach((item, k) => out.push(drawKeyItems([item], 4 + (k % 2) * colW, height + 12 + Math.floor(k / 2) * 10.5, "neutral")));
  height += 14 + Math.ceil(items.length / 2) * 10.5;
  return { width: W, height, body: out.join(""), desc: `Per-model spread over ${nRel} relations, highlighted relations marked.` };
}

const PRED_GAIN = st.x_color;
const PRED_RECALL = st.ink;

const SHOW_COUNTS = Boolean(spec.predictor_show_counts);
function predictorTable() {
  const t = inputs.predictorCI;
  if (!t) fail("the predictor-* variants need spec.predictor_ci");
  for (const m of models) if (!t.models[m.id]) fail(`predictor CI table lacks ${m.id}`);
  return t;
}

function residualsWithin(field, control) {
  const out = new Map();
  for (const m of models) {
    const own = cells.filter((c) => c.model === m.id);
    const z = own.map(control);
    const v = own.map(field);
    const { slope, intercept } = ols(z, v);
    own.forEach((c, i) => out.set(`${c.model}\u0000${c.relation}`, v[i] - (intercept + slope * z[i])));
  }
  return out;
}

function predictorDots(partial) {
  const t = predictorTable();
  const keys = partial ? ["gain_given_recall", "recall_given_gain"] : ["gain", "recall"];
  const rName = inputs.covariateName("recall");
  const names = partial ? [`${xName}, controlling for ${T.term("recall", "short")}`, `${rName}, controlling for ${T.term("gain", "short")}`] : [xName, rName];
  const colors = [PRED_GAIN, PRED_RECALL];
  const labelW = Math.max(...models.map((m) => textWidth(m.label, FS)));
  const rowH = 12;
  const gap = 14;
  const x0 = 4 + labelW + 6;
  const pw = (W - x0 - 6 - gap) / 2;

  const predGlyph = (j) => (cx, cy) => line(cx - 5, cy, cx + 5, cy, colors[j], 0.9) + `<circle cx="${f(cx)}" cy="${f(cy)}" r="2" ${j === 0 ? `fill="${colors[j]}"` : `fill="#FFFFFF" stroke="${colors[j]}" stroke-width="0.8"`}/>`;

  const coefficientLabel = partial ? "partial correlation over relations" : T.term("coefficient_within_model");
  const key = keyBlocks([
    { id: "predictor", title: T.term("key_predictor"), items: names.map((label, j) => ({ id: keys[j], label, glyph: predGlyph(j) })) },
    { id: "marks", title: T.term("key_marks"), items: [
      { id: "coefficient", label: coefficientLabel, glyph: (cx, cy) => `<circle cx="${f(cx)}" cy="${f(cy)}" r="2" fill="${st.ink}"/>` },
      { id: "interval", label: T.term("bootstrap_ci"), glyph: (cx, cy) => line(cx - 5, cy, cx + 5, cy, st.ink, 0.9) },
    ] },
  ], { itemsLeft: 4, right: W - 4, y: 8, fs: FS, textWidth, text });
  const top = 8 + key.height + 9;
  const all = models.flatMap((m) => keys.flatMap((k) => [...t.models[m.id][k].r_ci, ...t.models[m.id][k].rho_ci]));
  const lo = Math.min(0, Math.floor(Math.min(...all) * 10) / 10);
  const out = [];
  const plotH = rowH * models.length;

  for (let pi = 0; pi < 2; pi += 1) {
    models.forEach((m, i) => {
      if (i % 2 === 0) out.push(`<rect data-layer="band" x="${f(x0 + pi * (pw + gap))}" y="${f(top + rowH * i)}" width="${f(pw)}" height="${f(rowH)}" fill="#F7F7F8"/>`);
    });
  }
  [["r", PEARSON], ["rho", SPEARMAN]].forEach(([stat, title], pi) => {
    const px = x0 + pi * (pw + gap);
    const sx = (v) => px + ((v - lo) / (1 - lo)) * pw;
    out.push(text(px + pw / 2, top - 6, title, { anchor: "middle", weight: 700 }));
    for (let v = Math.ceil(lo / 0.25) * 0.25; v <= 1 + 1e-9; v += 0.25) {
      const vv = Math.round(v * 100) / 100;
      out.push(line(sx(vv), top, sx(vv), top + plotH, vv === 0 ? st.axis : st.grid, vv === 0 ? 0.5 : 0.35, 'data-layer="grid"'));
      out.push(text(sx(vv), top + plotH + 9, tickLabel(vv), { anchor: "middle", fill: st.muted }));
    }
    let wins = 0;
    let apart = 0;
    models.forEach((m, i) => {
      const y = top + rowH * (i + 0.5);
      if (pi === 0) out.push(text(x0 - 5, y + 2.8, m.label, { anchor: "end" }));
      const e = keys.map((k) => t.models[m.id][k]);
      if (e[0][stat] > e[1][stat]) wins += 1;
      if (e[0][stat + "_ci"][0] > e[1][stat + "_ci"][1]) apart += 1;
      e.forEach((c, j) => {
        const yy = y + (j === 0 ? -2.6 : 2.6);
        const [a, b] = c[stat + "_ci"];
        out.push(line(sx(a), yy, sx(b), yy, colors[j], 0.9));
        const paint = j === 0 ? `fill="${colors[j]}"` : `fill="#FFFFFF" stroke="${colors[j]}" stroke-width="0.8"`;
        out.push(`<circle cx="${f(sx(c[stat]))}" cy="${f(yy)}" r="2" ${paint} data-model="${xml(m.id)}" data-pred="${keys[j]}" data-stat="${stat}" data-v="${c[stat]}" data-lo="${a}" data-hi="${b}"/>`);
      });
    });
    out.push(`<rect x="${f(px)}" y="${f(top)}" width="${f(pw)}" height="${f(plotH)}" fill="none" stroke="${st.axis}" stroke-width="0.4"/>`);
    if (SHOW_COUNTS) out.push(text(px + pw / 2, top + plotH + 20, `blue higher: ${wins}/${models.length} · disjoint intervals: ${apart}/${models.length}`, { anchor: "middle", fill: st.muted }));
  });
  out.push(key.svg);
  const bottom = SHOW_COUNTS ? top + plotH + 26 : top + plotH + 16;
  return { width: W, height: bottom, body: out.join(""), desc: `Per model, ${partial ? "partial " : ""}correlation of ${yName} with ${xName} and with ${rName}, with ${T.term("bootstrap_ci")}s.` };
}

const HALF_W = 238;
const HALF_TOP = 38;
function predictorDotsHalf(layout) {
  const t = predictorTable();
  const keys = ["gain_given_recall", "recall_given_gain"];
  const rName = inputs.covariateName("recall");
  const names = [`${xName}, controlling for ${T.term("recall", "short")}`, `${rName}, controlling for ${T.term("gain", "short")}`];
  const colors = [PRED_GAIN, PRED_RECALL];
  const paint = (j) => (j === 0 ? `fill="${colors[j]}"` : `fill="#FFFFFF" stroke="${colors[j]}" stroke-width="0.8"`);
  const stacked = layout === "stacked";
  const labelW = Math.max(...models.map((m) => textWidth(m.label, FS)));
  const statW = stacked ? Math.max(textWidth("r", FS), textWidth("ρ", FS)) + 5 : 0;
  const labelRight = 4 + labelW;
  const x0 = labelRight + 6 + statW;
  const gap = 10;
  const nPanels = stacked ? 1 : 2;
  const pw = (HALF_W - x0 - 4 - gap * (nPanels - 1)) / nPanels;
  const subH = 8;
  const rowH = stacked ? 2 * subH : 12;
  const top = HALF_TOP;
  const plotH = rowH * models.length;
  const all = models.flatMap((m) => keys.flatMap((k) => [...t.models[m.id][k].r_ci, ...t.models[m.id][k].rho_ci]));
  const lo = Math.min(0, Math.floor(Math.min(...all) * 10) / 10);
  const stats = [["r", PEARSON], ["rho", SPEARMAN]];
  const panels = stacked ? [{ px: x0, stats }] : stats.map((s, pi) => ({ px: x0 + pi * (pw + gap), stats: [s] }));
  const out = [];

  for (const { px } of panels) {
    models.forEach((m, i) => {
      if (i % 2 === 0) out.push(`<rect data-layer="band" x="${f(px)}" y="${f(top + rowH * i)}" width="${f(pw)}" height="${f(rowH)}" fill="#F7F7F8"/>`);
    });
  }
  const tally = {};
  for (const { px, stats: own } of panels) {
    const sx = (v) => px + ((v - lo) / (1 - lo)) * pw;
    const heading = own.map(([, term]) => termWithSymbol(term, xml)).join(" and ");

    const headingY = SHOW_COUNTS ? top - 27.5 : top - 8.5;
    out.push(`<text x="${f(px + pw / 2)}" y="${f(headingY)}" font-size="${f(FS)}" text-anchor="middle" font-weight="700" fill="${st.ink}">${heading}</text>`);
    for (let v = Math.ceil(lo / 0.25) * 0.25; v <= 1 + 1e-9; v += 0.25) {
      const vv = Math.round(v * 100) / 100;
      out.push(line(sx(vv), top, sx(vv), top + plotH, vv === 0 ? st.axis : st.grid, vv === 0 ? 0.5 : 0.35, 'data-layer="grid"'));
      if (stacked || Math.abs(vv * 2 - Math.round(vv * 2)) < 1e-9) out.push(text(sx(vv), top + plotH + 9, tickLabel(vv), { anchor: "middle", fill: st.muted }));
    }
    models.forEach((m, i) => {
      if (px === x0) out.push(text(labelRight, top + rowH * (i + 0.5) + 2.8, m.label, { anchor: "end" }));
      own.forEach(([stat], si) => {
        const y = stacked ? top + rowH * i + subH * (si + 0.5) : top + rowH * (i + 0.5);
        if (stacked) out.push(`<text x="${f(x0 - 4)}" y="${f(y + 2.8)}" font-size="${f(FS)}" text-anchor="end" fill="${st.muted}">${mathSym(stat === "r" ? "r" : "ρ")}</text>`);
        const e = keys.map((k) => t.models[m.id][k]);
        const tl = (tally[stat] ||= { wins: 0, apart: 0 });
        if (e[0][stat] > e[1][stat]) tl.wins += 1;
        if (e[0][stat + "_ci"][0] > e[1][stat + "_ci"][1]) tl.apart += 1;
        const off = stacked ? 1.6 : 2.6;
        e.forEach((c, j) => {
          const yy = y + (j === 0 ? -off : off);
          const [a, b] = c[stat + "_ci"];
          out.push(line(sx(a), yy, sx(b), yy, colors[j], 0.9));
          out.push(`<circle cx="${f(sx(c[stat]))}" cy="${f(yy)}" r="${stacked ? 1.8 : 2}" ${paint(j)} data-model="${xml(m.id)}" data-pred="${keys[j]}" data-stat="${stat}" data-v="${c[stat]}" data-lo="${a}" data-hi="${b}"/>`);
        });
      });
    });
    out.push(`<rect x="${f(px)}" y="${f(top)}" width="${f(pw)}" height="${f(plotH)}" fill="none" stroke="${st.axis}" stroke-width="0.4"/>`);
  }

  const n = models.length;
  if (SHOW_COUNTS) {
    for (const [k, [what, field]] of [["blue higher:", "wins"], ["disjoint intervals:", "apart"]].entries()) {
      const y = top - 18 + 9.5 * k;
      out.push(text(labelRight, y, what, { anchor: "end", fill: st.muted }));
      if (stacked) {
        out.push(`<text x="${f(x0 + pw / 2)}" y="${f(y)}" font-size="${f(FS)}" text-anchor="middle" fill="${st.muted}">${mathSym("r")} ${tally.r[field]}/${n} · ${mathSym("ρ")} ${tally.rho[field]}/${n}</text>`);
      } else {
        panels.forEach(({ px, stats: own }) => out.push(text(px + pw / 2, y, `${tally[own[0][0]][field]}/${n}`, { anchor: "middle", fill: st.muted })));
      }
    }
  }

  const coefficientLabel = "partial correlation over relations";
  const predGlyph = (j) => (cx, cy) => line(cx - 5, cy, cx + 5, cy, colors[j], 0.9) + `<circle cx="${f(cx)}" cy="${f(cy)}" r="2" ${paint(j)}/>`;
  const key = keyBlocks([
    { id: "predictor", title: T.term("key_predictor"), items: names.map((label, j) => ({ id: keys[j], label, glyph: predGlyph(j) })) },
    { id: "marks", title: T.term("key_marks"), items: [
      { id: "coefficient", label: coefficientLabel, glyph: (cx, cy) => `<circle cx="${f(cx)}" cy="${f(cy)}" r="2" fill="${st.ink}"/>` },
      { id: "interval", label: T.term("bootstrap_ci"), glyph: (cx, cy) => line(cx - 5, cy, cx + 5, cy, st.ink, 0.9) },
    ] },
  ], { itemsLeft: x0, titleRight: labelRight, right: HALF_W - 4, y: keyBaselineBelow(top + plotH + 9, FS), fs: FS, textWidth, text, clear: true });
  out.push(key.svg);
  return { width: HALF_W, height: keyBottom(key.lastBaseline, FS), body: out.join(""), desc: `Per model, partial correlation of ${yName} with ${xName} and with ${rName}, with ${T.term("bootstrap_ci")}s.` };
}

function residualSquare(out, x0, top, S, pts, xr, yr, xLabel, yLabel, rText, showY, panel) {
  const X = (v) => x0 + ((v + xr) / (2 * xr)) * S;
  const Y = (v) => top + S - ((v + yr) / (2 * yr)) * S;
  const fmt = (v) => (v === 0 ? "0" : `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(2).replace(/^0/, "")}`);
  for (const [r, axis] of [[xr, "x"], [yr, "y"]]) {
    const step = r > 0.45 ? 0.25 : r > 0.2 ? 0.1 : 0.05;
    for (let v = -Math.floor(r / step) * step; v <= r + 1e-9; v += step) {
      const vv = Math.round(v * 100) / 100;
      if (axis === "x") out.push(line(X(vv), top, X(vv), top + S, st.grid, 0.35, 'data-layer="grid"'), text(X(vv), top + S + 9, fmt(vv), { anchor: "middle", fill: st.muted }));
      else {
        out.push(line(x0, Y(vv), x0 + S, Y(vv), st.grid, 0.35, 'data-layer="grid"'));
        if (showY) out.push(text(x0 - 2.5, Y(vv) + 2.8, fmt(vv), { anchor: "end", fill: st.muted }));
      }
    }
  }
  out.push(line(x0, top, x0, top + S, st.axis, 0.5), line(x0, top + S, x0 + S, top + S, st.axis, 0.5));
  for (const p of pts) out.push(`<circle cx="${f(X(p.x))}" cy="${f(Y(p.y))}" r="1.05" fill="${st.neutral}" fill-opacity="0.5" data-panel="${panel}" data-model="${xml(p.model)}" data-relation="${xml(p.relation)}"/>`);
  const fit = ols(pts.map((p) => p.x), pts.map((p) => p.y));
  const r = pearson(pts.map((p) => p.x), pts.map((p) => p.y));
  const xa = Math.max(-xr, (-yr - fit.intercept) / fit.slope);
  const xb = Math.min(xr, (yr - fit.intercept) / fit.slope);
  out.push(line(X(xa), Y(fit.intercept + fit.slope * xa), X(xb), Y(fit.intercept + fit.slope * xb), st.ink, 0.6, 'stroke-dasharray="2.2 1.6"'));
  out.push(text(x0 + 3, top + 9, `${rText} = ${fmtCorr(r)}`, { weight: 700 }).replace("<text ", `<text data-panel="${panel}" data-r="${r}" `));
  out.push(text(x0 + S / 2, top + S + 20, xLabel, { anchor: "middle" }));
  if (showY) out.push(text(x0 - 22, top + S / 2, yLabel, { anchor: "middle", rotate: -90 }));
  return r;
}

function predictorCentred() {
  predictorTable();
  const rName = inputs.covariateName("recall");
  const byModel = new Map(models.map((m) => {
    const own = cells.filter((c) => c.model === m.id);
    return [m.id, { x: mean(own.map((c) => c.x)), y: mean(own.map((c) => c.y)), r: mean(own.map((c) => c.cov.recall)) }];
  }));
  const pts = cells.map((c) => ({ model: c.model, relation: c.relation, yc: c.y - byModel.get(c.model).y, xc: c.x - byModel.get(c.model).x, rc: c.cov.recall - byModel.get(c.model).r }));
  const round = (v) => Math.ceil(v * 10) / 10;
  const yr = round(Math.max(...pts.map((p) => Math.abs(p.yc))));
  const xr = round(Math.max(...pts.map((p) => Math.abs(p.xc))));
  const rr = round(Math.max(...pts.map((p) => Math.abs(p.rc))));
  const top = 8;
  const x0 = 40;
  const gap = 26;
  const S = (W - x0 - gap - 6) / 2;
  const out = [];
  residualSquare(out, x0, top, S, pts.map((p) => ({ ...p, x: p.xc, y: p.yc })), xr, yr, `${xName} − model mean`, `${yName} − model mean`, T.term("pooled_within_r"), true, "gain");
  residualSquare(out, x0 + S + gap, top, S, pts.map((p) => ({ ...p, x: p.rc, y: p.yc })), rr, yr, `${rName} − model mean`, "", T.term("pooled_within_r"), false, "recall");
  return { width: W, height: top + S + 36, body: out.join(""), desc: `The same cells centred within model: ${yName} against ${xName} (left) and against ${rName} (right), same y axis.` };
}

function predictorAddedVariable() {
  const t = predictorTable();
  const rName = inputs.covariateName("recall");
  const key = (c) => `${c.model}\u0000${c.relation}`;
  const yOnR = residualsWithin((c) => c.y, (c) => c.cov.recall);
  const xOnR = residualsWithin((c) => c.x, (c) => c.cov.recall);
  const yOnX = residualsWithin((c) => c.y, (c) => c.x);
  const rOnX = residualsWithin((c) => c.cov.recall, (c) => c.x);
  const left = cells.map((c) => ({ model: c.model, relation: c.relation, x: xOnR.get(key(c)), y: yOnR.get(key(c)) }));
  const right = cells.map((c) => ({ model: c.model, relation: c.relation, x: rOnX.get(key(c)), y: yOnX.get(key(c)) }));
  const round = (v) => Math.ceil(v * 10) / 10;
  const yr = round(Math.max(...[...left, ...right].map((p) => Math.abs(p.y))));
  const top = 8;
  const x0 = 40;
  const gap = 26;
  const S = (W - x0 - gap - 6) / 2;
  const out = [];
  residualSquare(out, x0, top, S, left, round(Math.max(...left.map((p) => Math.abs(p.x)))), yr, `${xName}, ${T.term("recall", "short")} removed`, `${yName}, other predictor removed`, T.term("pooled_partial_r"), true, "gain_given_recall");
  residualSquare(out, x0 + S + gap, top, S, right, round(Math.max(...right.map((p) => Math.abs(p.x)))), yr, `${rName}, ${T.term("gain", "short")} removed`, "", T.term("pooled_partial_r"), false, "recall_given_gain");
  const range = (k) => {
    const v = models.map((m) => t.models[m.id][k].r);
    return `${fmtCorr(Math.min(...v))} to ${fmtCorr(Math.max(...v))}`;
  };
  out.push(text(x0 + S / 2, top + S + 31, `per model: ${range("gain_given_recall")}`, { anchor: "middle", fill: st.muted }));
  out.push(text(x0 + S + gap + S / 2, top + S + 31, `per model: ${range("recall_given_gain")}`, { anchor: "middle", fill: st.muted }));
  return { width: W, height: top + S + 46, body: out.join(""), desc: `Added-variable pair, within model: ${yName} against ${xName} with ${rName} removed (left), and against ${rName} with ${xName} removed (right).` };
}

const variants = {
  "scatter-grid": () => scatterGrid(false),
  "scatter-grid-fits": () => scatterGrid(true),
  "scatter-pooled": scatterPooled,
  heatmap,
  "heatmap-interleaved": heatmapInterleaved,
  "model-spread": modelSpread,
  "rank-agreement": rankAgreement,
  "appendix-strips": () => appendixStrips(false),
  "scatter-grid-neutral": () => scatterGridEnc("neutral"),
  "scatter-grid-recall": () => scatterGridEnc("recall"),
  "scatter-grid-answers": () => scatterGridEnc("answers"),
  "scatter-pooled-neutral": () => scatterPooledEnc("neutral"),
  "scatter-pooled-recall": () => scatterPooledEnc("recall"),
  "scatter-pooled-model": () => scatterPooledEnc("model"),
  "heatmap-neutral": () => heatmapEnc(false),
  "heatmap-recall": () => heatmapEnc(true),
  "heatmap-recall-observed": () => heatmapEnc(true, true),
  "appendix-strips-recall": () => appendixStrips(true),
  "predictor-dots": () => predictorDots(false),
  "predictor-dots-partial": () => predictorDots(true),
  "predictor-dots-partial-half": () => predictorDotsHalf("two"),
  "predictor-dots-partial-stacked": () => predictorDotsHalf("stacked"),
  "predictor-centred": predictorCentred,
  "predictor-added-variable": predictorAddedVariable,
  "model-spread-highlight": modelSpreadHl,

  "scatter-grid-sets": () => scatterGridEnc("sets"),
  "scatter-grid-sets-nostats": () => scatterGridEnc("sets", false),
  "scatter-grid-tint": () => scatterGridEnc("tint"),
  "heatmap-recall-sqrt": () => heatmapEnc(true, false, "sqrt"),
  "heatmap-recall-quantile": () => heatmapEnc(true, false, "quantile"),
  "heatmap-recall-quantile-all": () => heatmapEnc(true, false, "quantile-all"),
  "appendix-strips-recall-part1": () => appendixStrips(true, [1, 2]),
  "appendix-strips-recall-part2": () => appendixStrips(true, [2, 2]),
};

for (const name of ["scatter-grid-neutral", "heatmap-recall", "appendix-strips"]) {
  variants[`${name}-simple`] = () => { MARKS = "none"; return variants[name](); };
}

for (const name of ["heatmap-recall", "heatmap-recall-sqrt", "heatmap-recall-quantile", "heatmap-recall-quantile-all",
  "appendix-strips", "appendix-strips-recall", "appendix-strips-recall-part1", "appendix-strips-recall-part2"]) {
  variants[`${name}-sets`] = () => { COLOR = "sets"; return variants[name](); };
}

variants["heatmap-recall-quantile-all-sets-by-gain"] = () => { ROW_ORDER = "gain"; return variants["heatmap-recall-quantile-all-sets"](); };
variants["heatmap-recall-quantile-all-sets-models-once"] = () => { MODELS_ONCE = true; return variants["heatmap-recall-quantile-all-sets"](); };
if (!variants[args.variant]) fail(`unknown variant ${args.variant}; one of ${Object.keys(variants).join(", ")}`);
const figure = variants[args.variant]();
writeFileSync(args.svg, svgDocument(inputs, args.variant, figure.width, figure.height, figure.body, figure.desc));

if (args.aggregate) {
  const aggregate = {
    ...inputs.provenance,
    models: spec.models.map((m) => m.id),
    relations: allRelations,
    complete_models: summary.completeModels,
    repeat_id: spec.repeat_id,
    cells: cells.map(({ model, relation, source, x, y, cov }) => ({ model, relation, source, x, y, cov })),
    covariate_scales: covScales,
    per_model: summary.perModel.map(({ model, n, pearson, spearman, slope, intercept, by_source, x_quartiles, y_quartiles }) => ({
      model, n, pearson, spearman, slope, intercept, by_source, x_quartiles, y_quartiles,
    })),
    per_relation: summary.perRelation,
    rank_agreement: {
      x: summary.agreement.x.map((row) => row.map((e) => e.spearman)),
      y: summary.agreement.y.map((row) => row.map((e) => e.spearman)),
    },
    pooled_within: summary.pooledWithin,
  };
  writeFileSync(args.aggregate, JSON.stringify(aggregate, null, 1) + "\n");
}
