#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";

const argv = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) =>
  (a.startsWith("--") ? acc.concat([[a.slice(2), all[i + 1]]]) : acc), []));
const TOL = Number(argv.tol ?? 1e-9);
const readJSON = (p) => JSON.parse(fs.readFileSync(p, "utf8"));
const readJSONL = (p) => fs.readFileSync(p, "utf8").split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));

const GROUPS = [
  ["raw", []],
  ["recall", ["pre"]],
  ["answer_count", ["lnclasses"]],
  ["subject_count", ["lntrain"]],
  ["subject_frequency", ["freq"]],
  ["all_jointly", ["pre", "lntrain", "lnclasses", "freq"]],
];

function subjectCounts(files) {
  const m = new Map();
  for (const f of files) for (const r of readJSONL(f)) {
    if (r.kind !== "subj" || !r.ok || r.count == null) continue;
    if (r.approx !== false) throw new Error(`inexact count ${r.query}`);
    m.set(r.query, r.count);
  }
  return m;
}
function meanLog(rows, counts, rel) {
  let s = 0;
  for (const [subj] of rows) {
    if (!counts.has(subj)) throw new Error(`${rel}: no count for ${subj}`);
    s += Math.log10(1 + counts.get(subj));
  }
  return s / rows.length;
}
const freq = new Map();
{
  const sp = readJSON(argv["study-payload"]);
  const sc = subjectCounts(argv["study-counts"].split(","));
  for (const [rel, rows] of Object.entries(sp)) freq.set(rel, meanLog(rows, sc, rel));
  const bp = readJSON(argv["bear-payload"]);
  const bc = subjectCounts([argv["bear-counts"]]);
  for (const [rel, rows] of Object.entries(bp)) freq.set(rel.toLowerCase(), meanLog(rows, bc, rel));
}

const mean = (v) => v.reduce((a, b) => a + b, 0) / v.length;
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
function residualize(target, basis) {

  const q = [];
  for (const col of basis) {
    let v = col.slice();
    for (const u of q) { const c = dot(v, u); v = v.map((x, i) => x - c * u[i]); }
    const nrm = Math.sqrt(dot(v, v));
    if (nrm < 1e-12) throw new Error("rank-deficient design");
    q.push(v.map((x) => x / nrm));
  }
  let r = target.slice();
  for (const u of q) { const c = dot(r, u); r = r.map((x, i) => x - c * u[i]); }
  return r;
}
function partial(x, y, controls) {
  const basis = [x.map(() => 1), ...controls];
  const rx = residualize(x, basis), ry = residualize(y, basis);
  return dot(rx, ry) / Math.sqrt(dot(rx, rx) * dot(ry, ry));
}
function ranks(v) {
  const idx = v.map((x, i) => i).sort((a, b) => v[a] - v[b]);
  const out = new Array(v.length);
  for (let i = 0; i < idx.length;) {
    let j = i;
    while (j + 1 < idx.length && v[idx[j + 1]] === v[idx[i]]) j++;
    for (let t = i; t <= j; t++) out[idx[t]] = (i + j) / 2 + 1;
    i = j + 1;
  }
  return out;
}

const RUN = argv["run-dir"];
const rows = readJSONL(path.join(RUN, "rows.jsonl"));
const spec = readJSON(path.join(RUN, "spec.json"));
const panel = [...spec.relations.hernandez, ...spec.relations.engineered, ...spec.relations.bear];
const xOf = new Map();
for (const c of readJSON(argv["report-data"]).cells) {
  if (c.fleet === "main" && c.final && c.repeat === 0 && c.ce_status === "passed" && c.x != null)
    xOf.set(`${c.model}\t${c.relation}`, c.x);
}
const eligible = (r) => r.status === "complete" && r.production_eligible === true && r.final_epoch === 20
  && r.epochs_done === 20 && r.planned_epochs === 20 && r.production_horizon === 20 && r.limit_train === 0
  && r.protocol === "heldout-mushroom-v2" && r.holdout_rule === "mushroom-v2"
  && r.n_train_records > 0 && r.n_exposed_per_epoch === r.n_train_records;

let failures = 0, checks = 0, worst = 0;
for (const variant of ["headline"]) {
  const report = readJSON(path.join(RUN, `fleet-joint-controls.${variant}.json`));
  for (const model of argv.models.split(",")) {
    const cohort = [];
    for (const rel of panel) {
      const hits = rows.filter((r) => r.model === model && r.relation === rel && r.repeat === 0 && r.final === true);
      if (hits.length > 1) throw new Error(`${model} ${rel}: ${hits.length} final repeat-0 rows`);
      const x = xOf.get(`${model}\t${rel}`);
      if (hits.length === 1 && eligible(hits[0]) && x != null) cohort.push({ rel, row: hits[0], x });
    }
    const rep = report.models[model];
    const sameCohort = JSON.stringify(cohort.map((c) => c.rel).sort()) === JSON.stringify([...rep.relations].sort());
    console.log(`${variant} ${model}: n=${cohort.length} (report ${rep.n}), cohort ${sameCohort ? "identical" : "DIFFERS"}`);
    if (!sameCohort) { failures++; continue; }
    const col = {
      x: cohort.map((c) => c.x), y: cohort.map((c) => c.row.heldout_cand_acc),
      pre: cohort.map((c) => c.row.pre_rel_cand_acc), lntrain: cohort.map((c) => Math.log(c.row.n_train)),
      lnclasses: cohort.map((c) => Math.log(c.row.n_classes)), freq: cohort.map((c) => freq.get(c.rel)),
    };
    for (const [k, v] of Object.entries(col)) if (v.some((z) => !Number.isFinite(z))) throw new Error(`${model}: non-finite ${k}`);
    const rk = Object.fromEntries(Object.entries(col).map(([k, v]) => [k, ranks(v)]));
    for (const [g, ctrl] of GROUPS) {
      const r = partial(col.x, col.y, ctrl.map((c) => col[c]));
      const rho = partial(rk.x, rk.y, ctrl.map((c) => rk[c]));
      const cmp = rep.comparison[g];
      for (const [name, mine, theirs] of [["r", r, cmp.ce_partial_pearson_r], ["rho", rho, cmp.ce_rank_partial_spearman_rho]]) {
        const d = Math.abs(mine - theirs);
        checks++; worst = Math.max(worst, d);
        const ok = d <= TOL && cmp.ce_n === cohort.length;
        if (!ok) failures++;
        console.log(`  ${g.padEnd(18)} ${name.padEnd(3)} verifier ${mine.toFixed(12)} report ${theirs.toFixed(12)} |d| ${d.toExponential(2)} ${ok ? "ok" : "FAIL"}`);
      }
    }
  }
}
console.log(`VERIFY ${failures ? "FAIL" : "PASS"}: ${checks} coefficient checks, worst |d| ${worst.toExponential(3)}, tol ${TOL}`);
process.exit(failures ? 1 : 0);
