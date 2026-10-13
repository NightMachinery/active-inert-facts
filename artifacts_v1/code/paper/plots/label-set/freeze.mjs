#!/usr/bin/env node
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { resolveSpec, stripResolved } from "../shared/relation-sets.mjs";

const fail = (m) => { process.stderr.write(`freeze: ${m}\n`); process.exit(1); };
const argv = process.argv.slice(2);
const multi = (k) => argv.flatMap((v, i) => (v === `--${k}` ? [argv[i + 1]] : []));
const one = (k) => multi(k)[0];
const specPath = one("spec") || "spec.json";
const spec = JSON.parse(readFileSync(specPath, "utf8"));

resolveSpec(spec, specPath);
const sha256 = (t) => createHash("sha256").update(t).digest("hex");
const bySha = new Map(spec.pools.map((p) => [p.pool_sha256, p.id]));
const byId = new Set(spec.pools.map((p) => p.id));

const SNAPSHOT_PANEL = { hernandez_engineered: "study", bear: "bear53" };
const panelOf = new Map(Object.entries(SNAPSHOT_PANEL).flatMap(([set, panel]) => spec.relation_sets[set].map((r) => [r, panel])));
const fold = (s) => s.toLowerCase();

const ce = new Map();
if (one("ce")) for (const r of readFileSync(one("ce"), "utf8").split(/\r?\n/).filter(Boolean).map(JSON.parse)) {
  if (fold(r.model) === fold(spec.model)) ce.set(r.relation, r.pre_rel_cls_ce_gain);
}

if (!Number.isInteger(spec.repeat_id)) fail("spec.repeat_id must be an integer");
const otherRepeat = new Set();
let skippedRepeats = 0;
const sources = [];
const out = new Map();
for (const path of multi("rows")) {
  const text = readFileSync(path, "utf8");
  sources.push({ path, sha256: sha256(text) });
  for (const r of text.split(/\r?\n/).filter(Boolean).map(JSON.parse)) {
    if (fold(r.model) !== fold(spec.model)) continue;
    if (!Object.entries(spec.eligibility).every(([k, v]) => r[k] === v)) continue;
    const pool = byId.has(r.pool) ? r.pool : bySha.get(r.label_pool_sha256);
    if (!pool) fail(`${path}: row ${r.relation}/${r.repeat} names no known pool (pool=${r.pool}, label_pool_sha256=${r.label_pool_sha256})`);
    const panel = panelOf.get(r.relation);
    if (!panel) continue;
    let c = r.pre_rel_cls_ce_gain;
    if (ce.has(r.relation)) {
      if (c !== undefined && c !== ce.get(r.relation)) fail(`CE disagreement for ${r.relation}: row ${c} vs --ce ${ce.get(r.relation)}`);
      c = ce.get(r.relation);
    }
    if (!Number.isFinite(c)) fail(`no CE gain for ${r.relation}`);
    if (r.repeat !== spec.repeat_id) {
      otherRepeat.add(`${pool}\u0000${r.relation}`);
      skippedRepeats += 1;
      continue;
    }
    const key = `${pool}\u0000${r.relation}`;
    if (out.has(key)) fail(`duplicate eligible repeat-${spec.repeat_id} run ${key.replace(/\u0000/g, "/")}`);
    out.set(key, { pool, panel, relation: r.relation, repeat: r.repeat, gen: r.heldout_cand_acc, ce: c });
  }
}
if (!out.size) fail("no eligible rows");
const lacking = [...otherRepeat].filter((k) => !out.has(k));
if (lacking.length) fail(`refusing: ${lacking.length} cell(s) have other repeats but no eligible repeat ${spec.repeat_id}: ${lacking.slice(0, 5).map((k) => k.replace("\u0000", "/")).join("; ")}`);
const order = new Map(spec.pools.map((p, i) => [p.id, i]));
const rows = [...out.values()].sort((a, b) => order.get(a.pool) - order.get(b.pool) || (a.relation < b.relation ? -1 : a.relation > b.relation ? 1 : 0) || a.repeat - b.repeat);
const text = rows.map((r) => JSON.stringify(r)).join("\n") + "\n";
const digest = sha256(text);
const stamp = one("date") || new Date().toISOString().slice(0, 10).replace(/-/g, "");
const name = `data/rows-label-set-${stamp}-${digest.slice(0, 8)}.jsonl`;
writeFileSync(name, text);
Object.assign(spec, { snapshot: name, snapshot_sha256: digest, snapshot_is_synthetic: false, snapshot_sources: sources, status: "FINAL" });
writeFileSync(specPath, JSON.stringify(stripResolved(spec), null, 2) + "\n");
process.stdout.write(`wrote ${name}: ${rows.length} rows over ${new Set(rows.map((r) => r.pool)).size} pools; ${skippedRepeats} non-repeat-${spec.repeat_id} rows skipped\n`);
