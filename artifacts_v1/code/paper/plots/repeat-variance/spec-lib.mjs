import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { load as loadRelationSets } from "../shared/relation-sets.mjs";

export function readSpec(path, fail) {
  const own = JSON.parse(readFileSync(path, "utf8"));
  let spec = own;
  if (own.extends) {
    const base = JSON.parse(readFileSync(resolve(dirname(path), own.extends), "utf8"));
    if (base.extends) fail("spec extends chains are not supported");
    const { extends: _e, panels_select: select, ...rest } = own;
    spec = { ...base, ...rest };

    if (!rest.variants) spec.variants = Object.fromEntries(Object.entries(base.variants).filter(([, v]) => v.inherit !== false));
    if (select) {
      const byId = new Map(base.panels.map((p) => [p.id, p]));
      spec.panels = select.map((id) => byId.get(id) || fail(`panels_select names ${id}, not a panel of ${own.extends}`));
    }
  }
  const registry = loadRelationSets(resolve(dirname(path), spec.relation_registry_path));
  const setNames = (text) => text.replace(/\{([a-z_]+)\}/g, (_, key) => registry.displayName(key));
  spec.panels = spec.panels.map((panel) => {
    if (!registry.sources.has(panel.source)) fail(`panel ${panel.id}: source ${JSON.stringify(panel.source)} is not a registry set`);
    if ("color" in panel || "shape" in panel) fail(`panel ${panel.id}: colour and shape come from the registry (source_display), not the spec`);
    return { ...panel, label: setNames(panel.label), color: registry.displayColor(panel.source), mark: registry.displayMark(panel.source) };
  });
  if (spec.banner) spec.banner = setNames(spec.banner);
  for (const panel of spec.panels) {
    if (!Array.isArray(panel.sources) || !panel.sources.length) fail(`panel ${panel.id}: no sources`);
    const covered = panel.sources.flatMap((s) => s.repeats);
    const want = [...spec.repeats].sort().join(",");
    if ([...covered].sort().join(",") !== want) fail(`panel ${panel.id}: sources cover repeats ${covered.join(",")}, the spec needs ${want} exactly once`);
    if (panel.models && panel.models.some((id) => !spec.models.some((m) => m.id === id))) fail(`panel ${panel.id}: a panel model is not in spec.models`);
  }
  return { spec, own };
}

export function pendingSources(spec) {
  return spec.panels.flatMap((p) => p.sources.filter((s) => s.pending).map((s) => `${p.id} repeats [${s.repeats.join(", ")}]: ${s.pending}`));
}

export function refusePending(spec, specPath, fail) {
  const pending = pendingSources(spec);
  if (pending.length) {
    fail(`awaiting repeat fleet rows: ${specPath} has ${pending.length} pending source(s):\n  ${pending.join("\n  ")}\n` +
      "Fill each pending source's path and sha256 (and drop `pending`) once the operator posts them, then run `make freeze all`." +
      " Until then `make interim` renders what exists.");
  }
}

export const outputName = (spec, variant) => `${spec.output_prefix}-${variant.file}`;
