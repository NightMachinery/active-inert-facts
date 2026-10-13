#!/usr/bin/env node
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { fail, parseArgs, readSpec, relationSources } from "./lib.mjs";

const args = parseArgs(process.argv.slice(2));
if (!args.spec || !args.out) fail("usage: synth.mjs --spec spec-synthetic.json --out out/cells-SYNTHETIC.jsonl");
const { spec } = readSpec(args.spec);
if (!spec.allow_synthetic) fail("spec does not allow synthetic data");
let state = Number(args.seed || 20260922) >>> 0;
const uniform = () => {
  state = (state + 0x6d2b79f5) >>> 0;
  let t = state;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const normal = () => Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
const clip = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const X = spec.metrics.x.field;
const Y = spec.metrics.y.field;
const { sourceOf, order } = relationSources(spec);
const relations = order.map((id) => [id, sourceOf.get(id)]);
const latent = new Map(relations.map(([id]) => [id, { l: uniform() ** 1.3, u: 0.08 * normal() }]));
const offset = new Map(spec.models.map((model) => [model.id, 0.05 * normal()]));
const rows = [];
for (const model of spec.models) {
  for (const [relation, source] of relations) {
    const { l, u } = latent.get(relation);
    const x = clip(0.02 + 0.9 * l + offset.get(model.id) + 0.06 * normal(), 0.01, 0.98);
    const yTrue = clip(0.15 + 0.8 * x + u + 0.06 * normal(), 0.02, 0.99);
    const repeats = model.id === "MiniCPM5-2B-Base" && source === "bear" ? [0, 1, 2] : [0];
    for (const repeat of repeats) {
      rows.push({ model: model.id, relation, repeat, ...spec.eligibility, [X]: x, [Y]: clip(yTrue + (repeats.length > 1 ? 0.027 * normal() : 0), 0, 1), synthetic: true });
    }
  }
}
mkdirSync(dirname(args.out), { recursive: true });
writeFileSync(args.out, rows.map((row) => JSON.stringify(row)).join("\n") + "\n");
process.stdout.write(`synth: ${rows.length} DRAFT-SYNTHETIC rows -> ${args.out}\n`);
