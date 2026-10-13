#!/usr/bin/env node
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { load } from "../shared/relation-sets.mjs";
import { readSpec, refusePending } from "./spec-lib.mjs";

const fail = (m) => { process.stderr.write(`freeze: ${m}\n`); process.exit(1); };
const args = Object.fromEntries(process.argv.slice(2).reduce((p, v, i, a) => (v.startsWith("--") && p.push([v.slice(2), a[i + 1] && !a[i + 1].startsWith("--") ? a[i + 1] : true]), p), []));
if (!args.spec) fail("missing --spec");
const outDir = args["out-dir"] || "data";
const sha256 = (t) => createHash("sha256").update(t).digest("hex");
const { spec, own } = readSpec(args.spec, fail);
refusePending(spec, args.spec, fail);
const fold = (s) => s.toLowerCase();
const modelById = new Map(spec.models.map((m) => [fold(m.id), m.id]));

const registry = load(resolve(dirname(args.spec), spec.relation_registry_path));
const excluded = new Set(registry.pinned(spec.excluded_relation_pin, "excluded_relation_pin"));
const relationsOf = new Map(spec.panels.map((p) => [p.id, registry.pinned(p.relation_pin, `panel ${p.id}`)]));
const repeats = new Set(spec.repeats);

const readRows = (path, want) => {
  const text = readFileSync(path, "utf8");
  if (want && sha256(text) !== want) fail(`${path}: sha256 ${sha256(text)} != pinned ${want}`);
  return text.split(/\r?\n/).filter(Boolean).map(JSON.parse);
};
const eligible = (r) => Object.entries(spec.eligibility).every(([k, v]) => r[k] === v);

const out = [];
const report = [];
const claimed = new Map();

for (const panel of spec.panels) {
  for (const source of panel.sources) {
    for (const rel of relationsOf.get(panel.id)) {
      for (const rep of source.repeats) {
        const key = `${source.path}\u0000${rel}\u0000${rep}`;
        if (claimed.has(key) && (!panel.models || !claimed.get(key).models || panel.models.some((m) => claimed.get(key).models.includes(m)))) {
          fail(`relation ${rel} repeat ${rep} of ${source.path} is in panels ${claimed.get(key).id} and ${panel.id}`);
        }
        claimed.set(key, panel);
      }
    }
  }
}
for (const panel of spec.panels) {
  const rows = panel.sources.flatMap((source) => readRows(source.path, source.sha256)
    .filter((r) => source.repeats.includes(r.repeat)));
  const panelModels = panel.models ? new Set(panel.models) : null;

  const only = new Set(relationsOf.get(panel.id));
  const cells = new Map();
  let ineligible = 0, superseded = 0;
  for (const r of rows) {
    if (excluded.has(r.relation)) continue;
    if (!only.has(r.relation)) continue;
    const model = modelById.get(fold(r.model));
    if (!model || (panelModels && !panelModels.has(model))) continue;

    if (r.final === false) { superseded += 1; continue; }
    if (!eligible(r) || !repeats.has(r.repeat)) { ineligible += 1; continue; }
    const g = r[spec.metrics.gen.field];
    if (!Number.isFinite(g)) fail(`${panel.id} ${r.model}/${r.relation}/${r.repeat}: non-finite outcome`);
    if (!Number.isInteger(r.n_heldout) || r.n_heldout <= 0) fail(`${panel.id} ${r.model}/${r.relation}: bad n_heldout`);
    const hp = Object.values(r.holdout_per_class || {});
    if (!hp.length || hp.some((h) => !(h > 0))) fail(`${panel.id} ${r.model}/${r.relation}: bad holdout_per_class`);
    if (hp.reduce((s, h) => s + h, 0) !== r.n_heldout) fail(`${panel.id} ${r.model}/${r.relation}: holdout_per_class does not sum to n_heldout`);
    const key = `${model}\u0000${r.relation}`;
    if (!cells.has(key)) cells.set(key, new Map());
    const cell = cells.get(key);
    if (cell.has(r.repeat)) fail(`${panel.id} ${model}/${r.relation}: duplicate eligible repeat ${r.repeat}`);
    cell.set(r.repeat, {
      panel: panel.id, model, relation: r.relation, repeat: r.repeat, gen: g,
      n_heldout: r.n_heldout, n_heldout_classes: hp.length,
      inv_h_sum: hp.reduce((s, h) => s + 1 / h, 0),
    });
  }
  let incomplete = 0;
  const kept = [];
  for (const cell of cells.values()) {
    if (cell.size !== repeats.size) { incomplete += 1; continue; }
    const v = [...cell.values()];
    if (new Set(v.map((x) => x.n_heldout)).size !== 1) fail(`${v[0].model}/${v[0].relation}: n_heldout differs across repeats`);
    kept.push(...v);
  }
  if (panel.crosscheck_rows) {
    const cc = new Map(readRows(panel.crosscheck_rows, panel.crosscheck_rows_sha256)
      .map((r) => [`${fold(r.model)}\u0000${r.relation}\u0000${r.repeat}`, r.heldout_cand_acc]));
    let checked = 0;
    for (const r of kept) {
      const want = cc.get(`${fold(r.model)}\u0000${r.relation}\u0000${r.repeat}`);
      if (want === undefined) fail(`${panel.id} ${r.model}/${r.relation}/${r.repeat} missing from crosscheck snapshot`);
      if (want !== r.gen) fail(`${panel.id} ${r.model}/${r.relation}/${r.repeat}: ${r.gen} != crosscheck ${want}`);
      checked += 1;
    }
    report.push(`${panel.id}: ${checked} outcome values equal the crosscheck snapshot exactly`);
  }
  const rels = new Set(kept.map((r) => r.relation));
  if ([...rels].some((rel) => !only.has(rel))) fail(`${panel.id}: a relation outside the panel's selection was kept`);
  if (rels.size !== only.size) fail(`${panel.id}: ${rels.size} relations with complete cells, the selection ${panel.relation_pin.selection} has ${only.size}: missing ${[...only].filter((r) => !rels.has(r)).join(", ")}`);
  report.push(`${panel.id}: ${kept.length / repeats.size} complete cells, ${incomplete} incomplete, ${ineligible} ineligible rows, ${superseded} superseded (final: false) rows skipped, ${rels.size} relations`);
  out.push(...kept);
}
const order = new Map(spec.models.map((m, i) => [m.id, i]));
const porder = new Map(spec.panels.map((p, i) => [p.id, i]));
out.sort((a, b) => porder.get(a.panel) - porder.get(b.panel) || order.get(a.model) - order.get(b.model)
  || (a.relation < b.relation ? -1 : a.relation > b.relation ? 1 : 0) || a.repeat - b.repeat);
const text = out.map((r) => JSON.stringify(r)).join("\n") + "\n";
const digest = sha256(text);
const stamp = (args.date || new Date().toISOString().slice(0, 10).replace(/-/g, ""));
if (args.check) {

  const pinned = readFileSync(spec.snapshot, "utf8");
  if (sha256(pinned) !== spec.snapshot_sha256) fail(`--check: ${spec.snapshot} does not match its pin`);
  const sorted = (s) => s.split("\n").filter(Boolean).sort().join("\n");
  if (digest === spec.snapshot_sha256) {
    process.stdout.write(`${report.join("\n")}\ncheck OK: recomputed snapshot equals ${spec.snapshot} byte for byte\n`);
  } else if (sorted(pinned) === sorted(text)) {
    process.stdout.write(`${report.join("\n")}\ncheck OK: recomputed snapshot has exactly the rows of ${spec.snapshot}, in a different order (model order changed since it was frozen)\n`);
  } else {
    fail(`--check: recomputed snapshot ${digest} (${out.length} rows) has different rows from the pinned ${spec.snapshot_sha256} (${spec.snapshot_rows} rows)`);
  }
  process.exit(0);
}
const name = `${outDir}/rows-repeat-variance-${stamp}-${digest.slice(0, 8)}.jsonl`;
writeFileSync(name, text);
own.snapshot = name;
own.snapshot_sha256 = digest;
own.snapshot_rows = out.length;
writeFileSync(args.spec, JSON.stringify(own, null, 2) + "\n");
process.stdout.write(`${report.join("\n")}\nwrote ${name} (${out.length} rows, sha256 ${digest})\n`);
