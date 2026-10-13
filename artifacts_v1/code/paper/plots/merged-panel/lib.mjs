import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { load as loadRelationSets, resolveSpec } from "../shared/relation-sets.mjs";
import { Terms } from "../shared/terms.mjs";
import { parse as parseJson5 } from "../shared/json5_io.mjs";

export function fail(message) {
  process.stderr.write(`merged-panel: ${message}\n`);
  process.exit(1);
}

export const sha256 = (text) => createHash("sha256").update(text).digest("hex");

export function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i += 1) {
    if (!argv[i].startsWith("--")) continue;
    const key = argv[i].slice(2);
    const value = argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[(i += 1)] : true;
    if (key in args) args[key] = [].concat(args[key], value);
    else args[key] = value;
  }
  return args;
}

export function readSpec(specPath) {
  const text = readFileSync(specPath, "utf8");
  const own = JSON.parse(text);
  let spec = own;
  let baseText = null;
  if (own.extends) {
    const basePath = resolve(dirname(specPath), own.extends);
    baseText = readFileSync(basePath, "utf8");
    const base = JSON.parse(baseText);
    if (base.extends) fail("spec extends chains are not supported");
    spec = { ...base, ...own };
  }
  resolveRelations(spec, specPath);
  return { spec, specSha256: sha256(baseText ? baseText + "\n" + text : text), dir: dirname(specPath) };
}

function resolveRelations(spec, specPath) {
  try {
    resolveSpec(spec, specPath);
  } catch (error) {
    fail(error.message);
  }
  const sets = spec.relation_sets || {};
  const unknown = Object.keys(sets).filter((key) => !SOURCES.includes(key));
  if (unknown.length) fail(`relation_registry.relation_sets has unknown set(s) ${unknown.join(", ")}; the sets are ${SOURCES.join(", ")}`);
  spec.relation_sets = Object.fromEntries(SOURCES.map((source) => [source, sets[source] || []]));
  if (spec.relations) {
    const union = SOURCES.flatMap((source) => spec.relation_sets[source]).sort();
    const panel = spec.relations.map((r) => r.id).sort();
    if (JSON.stringify(union) !== JSON.stringify(panel)) fail(`${specPath}: the relation_sets pins do not add up to the relations pin`);
  }
}

export const SOURCES = ["hernandez", "engineered", "bear"];

const registryForNames = loadRelationSets();
export const SOURCE_NAMES = Object.fromEntries(SOURCES.map((s) => [s, registryForNames.displayName(s)]));
export const SOURCE_COLORS = Object.fromEntries(SOURCES.map((s) => [s, registryForNames.displayColor(s)]));
export const SOURCE_MARKS = Object.fromEntries(SOURCES.map((s) => [s, registryForNames.displayMark(s)]));

export function relationSources(spec) {
  const sets = spec.relation_sets;
  const sourceOf = new Map();
  const order = [];
  for (const source of SOURCES) {
    for (const id of sets[source]) {
      if (sourceOf.has(id)) fail(`relation ${id} is in both ${sourceOf.get(id)} and ${source}`);
      sourceOf.set(id, source);
      order.push(id);
    }
  }
  const aliases = spec.snapshot_source_aliases || {};
  for (const [alias, targets] of Object.entries(aliases)) {
    if (SOURCES.includes(alias)) fail(`snapshot_source_aliases may not redefine the set name ${alias}`);
    if (!Array.isArray(targets) || !targets.length || targets.some((t) => !SOURCES.includes(t))) {
      fail(`snapshot_source_aliases.${alias} must be a non-empty list of ${SOURCES.join(", ")}`);
    }
  }
  return { sourceOf, order };
}

export function checkRowSource(spec, row, sourceOf) {
  const expected = sourceOf.get(row.relation);
  if (row.source === expected) return;
  const targets = (spec.snapshot_source_aliases || {})[row.source];
  if (targets && targets.includes(expected)) return;
  if (!SOURCES.includes(row.source) && !targets) fail(`row ${row.model}/${row.relation} has unknown source ${JSON.stringify(row.source)}; declare it in snapshot_source_aliases or re-freeze`);
  fail(`row ${row.model}/${row.relation} source ${row.source} disagrees with the spec, which puts it in ${expected}`);
}

function readPinned(dir, block, what) {

  if (!block || !block.path || !block.sha256) fail(`spec lacks ${what} {path, sha256}`);
  const text = readFileSync(resolve(dir, block.path), "utf8");
  const digest = sha256(text);
  if (digest !== block.sha256) fail(`${what} digest ${digest} does not match the spec pin ${block.sha256}`);
  let parsed;
  try {
    parsed = parseJson5(text, what);
  } catch (error) {
    fail(`${what} is not valid JSON5: ${error.message}`);
  }
  return { parsed, digest, text };
}

export function loadInputs(specPath, dataPathOverride) {
  const { spec, specSha256, dir } = readSpec(specPath);
  const { sourceOf } = relationSources(spec);
  const mainLabels = readPinned(dir, spec.relation_labels.hernandez_engineered, "relation_labels.hernandez_engineered map");
  const bearLabels = readPinned(dir, spec.relation_labels.bear, "relation_labels.bear map");
  const probeNames = readPinned(dir, spec.probe_names, "probe-name registry");
  if (probeNames.parsed.schema !== "linroute/probe-names/1") fail("probe-name registry schema must be linroute/probe-names/1");

  let terms;
  try {
    terms = new Terms(spec.probe_names.path, probeNames.text, registryForNames);
  } catch (error) {
    fail(error.message);
  }

  const labelOf = new Map();
  for (const id of [...spec.relation_sets.hernandez, ...spec.relation_sets.engineered]) {
    const label = mainLabels.parsed.relations[id];
    if (!label) fail(`${sourceOf.get(id)} relation ${id} has no label in ${spec.relation_labels.hernandez_engineered.path}`);
    labelOf.set(id, label);
  }
  for (const id of spec.relation_sets.bear) {
    const label = bearLabels.parsed.relations[id];
    if (!label) fail(`BEAR relation ${id} has no label in ${spec.relation_labels.bear.path}`);
    labelOf.set(id, label);
  }

  const idOfLabel = new Map();
  for (const [id, label] of labelOf) {
    if (idOfLabel.has(label)) fail(`relations ${idOfLabel.get(label)} and ${id} share the display label "${label}"; give each a distinct label in the relation-label maps`);
    idOfLabel.set(label, id);
  }

  const dataPath = dataPathOverride || (spec.snapshot ? resolve(dir, spec.snapshot) : null);
  if (!dataPath) fail("no snapshot: the spec has snapshot=null and no --data was given (run make refresh ROWS=...)");
  const dataText = readFileSync(dataPath, "utf8");
  const dataSha256 = sha256(dataText);
  if (!dataPathOverride && spec.snapshot_sha256 !== dataSha256) fail(`snapshot digest ${dataSha256} does not match the spec pin`);
  const rows = dataText.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
  const synthetic = rows.some((row) => row.synthetic === true);
  if (synthetic && !rows.every((row) => row.synthetic === true)) fail("snapshot mixes synthetic and real rows");
  if (synthetic && !spec.allow_synthetic) fail("synthetic snapshot given to a spec that does not allow it");

  const covariateName = (name) => {
    const c = (spec.covariates || {})[name];
    if (!c) fail(`no covariate ${name}`);
    if (c.label || !Array.isArray(c.label_keys)) fail(`covariate ${name} must name its label by label_keys into the probe-name registry`);
    try {
      return c.label_keys.map((path) => terms.at(path.join("."))).join(" ");
    } catch (error) {
      return fail(error.message);
    }
  };

  const metricName = (axis, variant) => {

    const key = spec.metrics[axis].name_key;
    try {
      return terms.at(`metrics.${key}.${variant}`);
    } catch (error) {
      return fail(error.message);
    }
  };

  return {
    spec,
    rows,
    synthetic,
    labelOf,
    metricName,
    covariateName,
    terms,

    predictorCI: spec.predictor_ci ? readPinned(dir, spec.predictor_ci, "predictor CI table").parsed : null,
    provenance: {
      package: spec.package,
      status: synthetic ? "DRAFT-SYNTHETIC" : spec.status,
      spec_sha256: specSha256,
      data_sha256: dataSha256,
      hernandez_engineered_labels_sha256: mainLabels.digest,
      bear_labels_sha256: bearLabels.digest,
      probe_names_sha256: probeNames.digest,
      ...(spec.predictor_ci ? { predictor_ci_sha256: spec.predictor_ci.sha256 } : {}),
    },
  };
}

export function buildCells(inputs) {
  const { spec, rows, labelOf } = inputs;
  const { sourceOf, order: relationOrder } = relationSources(spec);
  const modelIds = spec.models.map((model) => model.id);
  const X = spec.metrics.x.field;
  const Y = spec.metrics.y.field;
  if (!Number.isInteger(spec.repeat_id)) fail("spec.repeat_id must be an integer (the single repeat the figures use)");
  const groups = new Map();
  for (const row of rows) {
    if (!modelIds.includes(row.model)) fail(`snapshot row for model ${row.model}, which is not in the spec`);
    if (!sourceOf.has(row.relation)) fail(`snapshot row for relation ${row.relation}, which is not in the spec`);
    checkRowSource(spec, row, sourceOf);
    for (const field of [X, Y]) {
      if (typeof row[field] !== "number" || !Number.isFinite(row[field])) fail(`row ${row.model}/${row.relation}/${row.repeat} has non-finite ${field}`);
    }
    const key = `${row.model}\u0000${row.relation}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  const missing = new Set((spec.expected_missing || []).map((entry) => `${entry.model}\u0000${entry.relation}`));
  const cells = [];
  for (const model of modelIds) {
    for (const relation of relationOrder) {
      const key = `${model}\u0000${relation}`;
      const group = groups.get(key);
      if (!group) {
        if (!missing.has(key)) fail(`undeclared missing cell ${model} x ${relation}`);
        continue;
      }
      if (missing.has(key)) fail(`cell ${model} x ${relation} is declared missing but present`);
      if (group.length !== 1 || group[0].repeat !== spec.repeat_id) {
        fail(`cell ${model} x ${relation} must hold exactly one row, repeat ${spec.repeat_id}; found repeats [${group.map((row) => row.repeat)}]`);
      }
      const row = group[0];
      const cov = {};
      for (const [name, c] of Object.entries(spec.covariates || {})) {
        if (typeof row[c.field] !== "number" || !Number.isFinite(row[c.field])) fail(`cell ${model} x ${relation} lacks covariate ${name} (${c.field}); re-freeze`);
        cov[name] = row[c.field];
      }
      cells.push({ model, relation, source: sourceOf.get(relation), label: labelOf.get(relation), x: row[X], y: row[Y], cov });
    }
  }
  return cells;
}

export const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;

export function pearson(xs, ys) {
  const mx = mean(xs);
  const my = mean(ys);
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < xs.length; i += 1) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) ** 2;
    syy += (ys[i] - my) ** 2;
  }
  if (!(sxx > 0 && syy > 0)) fail("undefined Pearson correlation (zero variance)");
  return sxy / Math.sqrt(sxx * syy);
}

export function averageRanks(values) {
  const order = values.map((value, index) => ({ value, index })).sort((a, b) => a.value - b.value || a.index - b.index);
  const ranks = new Array(values.length);
  for (let start = 0; start < order.length; ) {
    let end = start;
    while (end + 1 < order.length && order[end + 1].value === order[start].value) end += 1;
    const rank = (start + end) / 2 + 1;
    for (let k = start; k <= end; k += 1) ranks[order[k].index] = rank;
    start = end + 1;
  }
  return ranks;
}

export const spearman = (xs, ys) => pearson(averageRanks(xs), averageRanks(ys));

export function ols(xs, ys) {
  const mx = mean(xs);
  const my = mean(ys);
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < xs.length; i += 1) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) ** 2;
  }
  const slope = sxy / sxx;
  return { slope, intercept: my - slope * mx };
}

export function quantile7(values, p) {
  const sorted = [...values].sort((a, b) => a - b);
  const h = (sorted.length - 1) * p;
  const lo = Math.floor(h);
  const hi = Math.ceil(h);
  return sorted[lo] + (h - lo) * (sorted[hi] - sorted[lo]);
}

export function summarize(inputs, cells) {
  const { spec } = inputs;
  const perModel = spec.models.map((model) => {
    const own = cells.filter((cell) => cell.model === model.id);
    const xs = own.map((cell) => cell.x);
    const ys = own.map((cell) => cell.y);
    const fit = ols(xs, ys);
    const bySource = {};
    for (const source of SOURCES) {
      const part = own.filter((cell) => cell.source === source);
      if (part.length >= 3) {
        const f = ols(part.map((c) => c.x), part.map((c) => c.y));
        bySource[source] = { n: part.length, pearson: pearson(part.map((c) => c.x), part.map((c) => c.y)), ...f };
      }
    }
    return {
      model: model.id,
      label: model.label,
      n: own.length,
      pearson: pearson(xs, ys),
      spearman: spearman(xs, ys),
      ...fit,
      by_source: bySource,
      x_quartiles: [0.25, 0.5, 0.75].map((p) => quantile7(xs, p)),
      y_quartiles: [0.25, 0.5, 0.75].map((p) => quantile7(ys, p)),
    };
  });

  const allRelations = relationSources(spec).order;
  const completeModels = spec.models
    .map((model) => model.id)
    .filter((id) => cells.filter((cell) => cell.model === id).length === allRelations.length);
  if (completeModels.length < 2) fail("fewer than two complete models; the relation order is undefined");
  const perRelation = allRelations.map((relation) => {
    const own = cells.filter((cell) => cell.relation === relation && completeModels.includes(cell.model));
    const xs = own.map((cell) => cell.x);
    const ys = own.map((cell) => cell.y);

    const rs = own.map((cell) => cell.cov?.recall).filter((v) => Number.isFinite(v));
    const recall = rs.length === own.length
      ? { mean_recall: mean(rs), recall_min: Math.min(...rs), recall_max: Math.max(...rs), recall_median: quantile7(rs, 0.5) }
      : {};
    return {
      ...recall,
      relation,
      source: own[0].source,
      label: own[0].label,
      n_models: own.length,
      mean_x: mean(xs),
      mean_y: mean(ys),
      x_min: Math.min(...xs),
      x_max: Math.max(...xs),
      y_min: Math.min(...ys),
      y_max: Math.max(...ys),
      x_median: quantile7(xs, 0.5),
      y_median: quantile7(ys, 0.5),
    };
  });
  perRelation.sort((a, b) => b.mean_y - a.mean_y || (a.relation < b.relation ? -1 : 1));
  perRelation.forEach((entry, index) => {
    entry.rank = index + 1;
  });

  const agreement = { x: [], y: [] };
  for (const a of spec.models) {
    const rowX = [];
    const rowY = [];
    for (const b of spec.models) {
      const shared = allRelations.filter(
        (relation) => cells.some((c) => c.model === a.id && c.relation === relation) && cells.some((c) => c.model === b.id && c.relation === relation),
      );
      const pick = (id, field) => shared.map((relation) => cells.find((c) => c.model === id && c.relation === relation)[field]);
      rowX.push({ n: shared.length, spearman: spearman(pick(a.id, "x"), pick(b.id, "x")) });
      rowY.push({ n: shared.length, spearman: spearman(pick(a.id, "y"), pick(b.id, "y")) });
    }
    agreement.x.push(rowX);
    agreement.y.push(rowY);
  }

  const centred = cells.map((cell) => {
    const own = cells.filter((c) => c.model === cell.model);
    return { ...cell, xc: cell.x - mean(own.map((c) => c.x)), yc: cell.y - mean(own.map((c) => c.y)) };
  });
  const pooledWithin = {
    n: centred.length,
    pearson: pearson(centred.map((c) => c.xc), centred.map((c) => c.yc)),
    ...ols(centred.map((c) => c.xc), centred.map((c) => c.yc)),
  };

  return { perModel, perRelation, completeModels, agreement, centred, pooledWithin };
}

export function covariateScales(spec, cells) {
  const scales = {};
  for (const [name, c] of Object.entries(spec.covariates || {})) {
    const values = cells.map((cell) => cell.cov[name]);
    if (c.scale === "log") {
      if (values.some((v) => !(v > 0))) fail(`covariate ${name} has a non-positive value on a log scale`);
      scales[name] = { scale: "log", lo: Math.min(...values), hi: Math.max(...values) };
    } else if (c.scale === "linear") {
      if (!Array.isArray(c.domain) || c.domain.length !== 2) fail(`covariate ${name} needs domain [lo, hi]`);
      scales[name] = { scale: "linear", lo: c.domain[0], hi: c.domain[1] };
    } else fail(`covariate ${name} has unknown scale ${c.scale}`);
  }
  return scales;
}

export function covariateT(scale, value) {
  if (scale.scale === "log") return (Math.log(value) - Math.log(scale.lo)) / (Math.log(scale.hi) - Math.log(scale.lo));
  return (value - scale.lo) / (scale.hi - scale.lo);
}

export const xml = (value) =>
  String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

export const f = (value) => (Math.round(value * 1000) / 1000).toString();

export const fmtCorr = (value) => {
  const text = Math.abs(value).toFixed(2).replace(/^0/, "");
  return value < 0 ? `−${text}` : text;
};

export { textWidth } from "../shared/text_width.mjs";

export function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function ramp(stops, t) {
  const clamped = Math.max(0, Math.min(1, t));
  const position = clamped * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(position));
  const u = position - i;
  const a = hexToRgb(stops[i]);
  const b = hexToRgb(stops[i + 1]);
  const mix = a.map((channel, k) => Math.round(channel + (b[k] - channel) * u));
  return `#${mix.map((channel) => channel.toString(16).padStart(2, "0")).join("")}`;
}

export const luminance = (hex) => {
  const [r, g, b] = hexToRgb(hex).map((c) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};

export function svgDocument(inputs, variant, width, height, body, description) {
  const { spec, provenance, synthetic } = inputs;
  const style = spec.style;
  const out = [];
  out.push(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${f(width)}pt" height="${f(height)}pt" viewBox="0 0 ${f(width)} ${f(height)}" role="img" aria-labelledby="t d" data-variant="${xml(variant)}">`,
  );
  out.push(`<title id="t">${xml(`${spec.package} ${variant}`)}</title><desc id="d">${xml(description)}</desc>`);
  out.push(`<metadata>${xml(JSON.stringify({ variant, ...provenance }))}</metadata>`);
  out.push(`<rect width="${f(width)}" height="${f(height)}" fill="#FFFFFF"/>`);
  out.push(`<g font-family="${xml(style.font_family)}" fill="${style.ink}">`);
  out.push(body);
  out.push("</g>");

  if (synthetic) {
    const mark = "DRAFT-SYNTHETIC";
    const size = Math.min(width, height) / (synthetic ? 7 : 5);
    out.push(
      `<text x="${f(width / 2)}" y="${f(height / 2)}" text-anchor="middle" dominant-baseline="middle" font-family="${xml(style.font_family)}" font-size="${f(size)}" font-weight="700" fill="#DC2626" fill-opacity="${synthetic ? 0.22 : 0.1}" transform="rotate(-28 ${f(width / 2)} ${f(height / 2)})">${mark}</text>`,
    );
    out.push(`<text x="${f(width - 2)}" y="${f(height - 2)}" text-anchor="end" font-family="${xml(style.font_family)}" font-size="8" fill="#DC2626">${mark}</text>`);
  }
  out.push("</svg>\n");
  return out.join("");
}
