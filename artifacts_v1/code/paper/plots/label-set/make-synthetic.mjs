#!/usr/bin/env node
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { resolveSpec, stripResolved } from "../shared/relation-sets.mjs";

const fail = (m) => { process.stderr.write(`make-synthetic: ${m}\n`); process.exit(1); };
const args = Object.fromEntries(process.argv.slice(2).reduce((p, v, i, a) => (v.startsWith("--") && p.push([v.slice(2), a[i + 1]]), p), []));
const sha256 = (t) => createHash("sha256").update(t).digest("hex");
const spec = JSON.parse(readFileSync(args.spec || "spec.json", "utf8"));

resolveSpec(spec, args.spec || "spec.json");
const S = spec.synthetic;

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const rand = mulberry32(S.seed);
const normal = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

const anchor = new Map();
for (const a of S.anchors) {
  const text = readFileSync(a.path, "utf8");
  if (sha256(text) !== a.sha256) fail(`anchor ${a.path} hash mismatch`);
  const acc = new Map();
  for (const r of text.split(/\r?\n/).filter(Boolean).map(JSON.parse)) {
    if (r.model !== spec.model || r.repeat !== spec.repeat_id) continue;
    if (acc.has(r.relation)) fail(`two repeat-${spec.repeat_id} anchor rows for ${r.relation}`);
    acc.set(r.relation, { ce: r.pre_rel_cls_ce_gain, gen: r.heldout_cand_acc });
  }
  for (const [rel, v] of acc) anchor.set(rel, v);
}

const SNAPSHOT_PANEL = { hernandez_engineered: "study", bear: "bear53" };
const relations = Object.entries(SNAPSHOT_PANEL).flatMap(([set, panel]) => spec.relation_sets[set].map((r) => [panel, r]));
const base = new Map();
let anchored = 0;
for (const [panel, rel] of relations) {
  const a = anchor.get(rel);
  if (a) { base.set(rel, { panel, ce: a.ce, gen: a.gen }); anchored += 1; continue; }
  const ce = clip(0.05 + 0.75 * rand() ** 1.3, 0.01, 0.95);
  base.set(rel, { panel, ce, gen: clip(0.08 + 0.85 * ce + 0.09 * normal(), 0.03, 0.97) });
}
const noisy = new Set(["quantity", "object"]);
const rows = [];
for (const p of spec.pools) {
  const shift = p.id === spec.reference_pool ? 0 : 0.06 * normal();
  const sdev = p.id === spec.reference_pool ? 0 : noisy.has(p.id) ? 0.14 : 0.045;
  for (const [panel, rel] of relations) {
    const b = base.get(rel);
    rows.push({ pool: p.id, panel, relation: rel, repeat: 0, gen: clip(b.gen + shift + sdev * normal(), 0.02, 0.99), ce: b.ce, synthetic: true });
  }
}
const text = rows.map((r) => JSON.stringify(r)).join("\n") + "\n";
const digest = sha256(text);
const name = `data/rows-label-set-SYNTHETIC-${digest.slice(0, 8)}.jsonl`;
writeFileSync(name, text);
spec.snapshot = name; spec.snapshot_sha256 = digest; spec.snapshot_is_synthetic = true; spec.status = "DRAFT-SYNTHETIC";
writeFileSync(args.spec || "spec.json", JSON.stringify(stripResolved(spec), null, 2) + "\n");
process.stdout.write(`wrote ${name}: ${rows.length} rows, ${anchored} relations anchored on real Phase-1 MiniCPM5 values\n`);
