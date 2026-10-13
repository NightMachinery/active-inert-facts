#!/usr/bin/env node
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve, relative } from "node:path";
import { fail, parseArgs, readSpec, relationSources, sha256 } from "./lib.mjs";

const args = parseArgs(process.argv.slice(2));
if (!args.spec || !args.rows) fail("usage: freeze.mjs --spec spec.json --rows rows.jsonl [--rows more.jsonl] [--date YYYYMMDD]");
const rowPaths = [].concat(args.rows);
const { spec } = readSpec(args.spec);
const X = spec.metrics.x.field;
const Y = spec.metrics.y.field;

const { sourceOf, order: relationOrder } = relationSources(spec);
const modelOrder = spec.models.map((model) => model.id);

const dropped = {};
const drop = (reason) => {
  dropped[reason] = (dropped[reason] || 0) + 1;
};
if (!Number.isInteger(spec.repeat_id)) fail("spec.repeat_id must be an integer");
const kept = new Map();
const otherRepeatCells = new Set();
const sources = [];
for (const path of rowPaths) {
  const text = readFileSync(path, "utf8");
  const lines = text.split(/\r?\n/).filter(Boolean);
  sources.push({ path, sha256: sha256(text), rows: lines.length });
  for (const line of lines) {
    const row = JSON.parse(line);
    if (!modelOrder.includes(row.model)) {
      drop(`model not in cohort: ${row.model}`);
      continue;
    }
    if (!sourceOf.has(row.relation)) {
      drop("relation not in cohort");
      continue;
    }
    const failed = Object.entries(spec.eligibility).find(([key, value]) => row[key] !== value);
    if (failed) {
      drop(`ineligible: ${failed[0]}`);
      continue;
    }
    if (typeof row[X] !== "number" || typeof row[Y] !== "number") {
      drop(`missing ${typeof row[X] !== "number" ? X : Y}`);
      continue;
    }
    if (row.repeat !== spec.repeat_id) {
      otherRepeatCells.add(`${row.model}\u0000${row.relation}`);
      drop(`repeat ${row.repeat} (not repeat ${spec.repeat_id}; repeat-variance analysis only)`);
      continue;
    }
    const key = `${row.model}\u0000${row.relation}`;
    if (kept.has(key)) fail(`duplicate eligible repeat-${spec.repeat_id} row ${row.model} ${row.relation} across inputs`);
    const out = { model: row.model, relation: row.relation, source: sourceOf.get(row.relation), repeat: row.repeat, [X]: row[X], [Y]: row[Y] };

    for (const [name, cov] of Object.entries(spec.covariates || {})) {
      if (typeof row[cov.field] !== "number" || !Number.isFinite(row[cov.field])) fail(`row ${row.model} ${row.relation} lacks covariate ${name} (${cov.field})`);
      out[cov.field] = row[cov.field];
    }
    if (row.synthetic === true) out.synthetic = true;
    kept.set(key, out);
  }
}
const lacking = [...otherRepeatCells].filter((key) => !kept.has(key));
if (lacking.length) fail(`refusing: ${lacking.length} cell(s) have other repeats but no eligible repeat ${spec.repeat_id}: ${lacking.slice(0, 5).map((k) => k.replace("\u0000", " x ")).join("; ")}`);
const rows = [...kept.values()].sort(
  (a, b) =>
    modelOrder.indexOf(a.model) - modelOrder.indexOf(b.model) ||
    relationOrder.indexOf(a.relation) - relationOrder.indexOf(b.relation) ||
    a.repeat - b.repeat,
);
if (!rows.length) fail("no eligible rows");
const text = rows.map((row) => JSON.stringify(row)).join("\n") + "\n";
const digest = sha256(text);
const date = args.date || new Date().toISOString().slice(0, 10).replace(/-/g, "");
const specDir = dirname(resolve(args.spec));
mkdirSync(resolve(specDir, "data"), { recursive: true });
const outPath = args.out ? resolve(args.out) : resolve(specDir, "data", `cells-${date}-${digest.slice(0, 8)}.jsonl`);
if (args.out) {
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, text);
} else if (existsSync(outPath)) {
  if (sha256(readFileSync(outPath, "utf8")) !== digest) fail(`${outPath} exists with different content`);
} else {
  writeFileSync(outPath, text, { flag: "wx" });
}

if (args["no-spec-update"]) {
  process.stdout.write(`${JSON.stringify({ snapshot: outPath, sha256: digest, rows: rows.length, dropped })}\n`);
  process.exit(0);
}

const ownText = readFileSync(args.spec, "utf8");
const own = JSON.parse(ownText);
own.snapshot = relative(specDir, outPath);
own.snapshot_sha256 = digest;
own.snapshot_sources = sources;
own.snapshot_dropped = dropped;
writeFileSync(args.spec, JSON.stringify(own, null, 2) + "\n");
process.stdout.write(`${JSON.stringify({ snapshot: own.snapshot, sha256: digest, rows: rows.length, dropped })}\n`);
