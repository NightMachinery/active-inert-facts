import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { parse as parseJson5 } from "./json5_io.mjs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const SCHEMA = "linroute/relation-sets/1";
export const DEFAULT_PATH = resolve(dirname(fileURLToPath(import.meta.url)), "../../metadata/relation-sets.json5");
const ID = /^[A-Za-z0-9_]+$/;
const MARKS = new Set(["triangle", "ring", "dot"]);
const SELECTION_KEYS = new Set(["note", "base", "sources", "flags", "include_retired", "include", "exclude"]);

class RelationSetError extends Error {}

const byCodeUnit = (a, b) => (a < b ? -1 : a > b ? 1 : 0);

function fail(message) {
  throw new RelationSetError(`relation-sets: ${message}`);
}

function stringList(value, what) {
  if (value === undefined) return [];
  if (!Array.isArray(value) || value.some((item) => typeof item !== "string")) fail(`${what} must be a list of strings`);
  if (new Set(value).size !== value.length) fail(`${what} repeats an entry`);
  return value;
}

export function digest(ids) {
  return createHash("sha256").update(JSON.stringify(ids)).digest("hex");
}

export class RelationSets {
  constructor(path, text) {
    let document;
    try {
      document = parseJson5(text, path);
    } catch (error) {
      fail(`cannot parse ${error.message}`);
    }
    if (!document || document.schema !== SCHEMA) fail(`${path}: expected schema ${SCHEMA}`);
    this.path = path;
    this.sha256 = createHash("sha256").update(text).digest("hex");
    this.sources = new Set(Object.keys(document.sources || {}));
    this.flags = new Set(Object.keys(document.flags || {}));
    this.relations = document.relations || {};
    this.selections = document.selections || {};
    this.aliases = Object.fromEntries(Object.entries(document.aliases || {}).filter(([key]) => !key.startsWith("_")));
    if (!this.sources.size || !Object.keys(this.relations).length) fail(`${path}: empty sources or relations`);

    this.sourceDisplay = Object.fromEntries(Object.entries(document.source_display || {}).filter(([key]) => !key.startsWith("_")));
    for (const source of this.sources) {
      const name = this.sourceDisplay[source]?.name;
      if (typeof name !== "string" || !name.trim()) fail(`source_display.${source}.name is missing or empty`);
      const { color, mark } = this.sourceDisplay[source];
      if (!/^#[0-9A-F]{6}$/.test(color || "")) fail(`source_display.${source}.color must be #RRGGBB (upper case)`);
      if (!MARKS.has(mark)) fail(`source_display.${source}.mark must be one of ${[...MARKS].join(", ")}`);
    }
    for (const key of Object.keys(this.sourceDisplay)) if (!this.sources.has(key)) fail(`source_display names unknown source ${JSON.stringify(key)}`);
    for (const [id, relation] of Object.entries(this.relations)) {
      if (!ID.test(id)) fail(`relation id ${JSON.stringify(id)} is not [A-Za-z0-9_]+`);
      if (!this.sources.has(relation.source)) fail(`relation ${id}: unknown source ${JSON.stringify(relation.source)}`);
      for (const flag of stringList(relation.flags, `relation ${id} flags`)) {
        if (!this.flags.has(flag)) fail(`relation ${id}: unknown flag ${JSON.stringify(flag)}`);
      }
      if (relation.retired !== undefined && relation.retired !== true) fail(`relation ${id}: retired must be true or absent`);
    }
    for (const [alias, target] of Object.entries(this.aliases)) {
      if (!ID.test(alias)) fail(`alias ${JSON.stringify(alias)} is not [A-Za-z0-9_]+`);
      if (alias in this.relations) fail(`alias ${alias} is itself a relation id`);
      if (!(target in this.relations)) fail(`alias ${alias} points at unknown relation ${JSON.stringify(target)}`);
    }

    this.resolved = new Map();
    for (const name of Object.keys(this.selections)) this.named(name);
  }

  select({ base, sources, flags, include, exclude, includeRetired = false } = {}, context = "select") {
    let members;
    if (base !== undefined) {
      if (sources !== undefined || flags !== undefined || includeRetired) {
        fail(`${context}: base excludes sources, flags and include_retired`);
      }
      members = new Set(this.named(base));
    } else {
      const wantedSources = stringList(sources, `${context} sources`);
      for (const source of wantedSources) if (!this.sources.has(source)) fail(`${context}: unknown source ${JSON.stringify(source)}`);
      const wantedFlags = stringList(flags, `${context} flags`);
      for (const flag of wantedFlags) if (!this.flags.has(flag)) fail(`${context}: unknown flag ${JSON.stringify(flag)}`);
      const sourceSet = new Set(sources === undefined ? this.sources : wantedSources);
      members = new Set(Object.entries(this.relations)
        .filter(([, relation]) => sourceSet.has(relation.source))
        .filter(([, relation]) => wantedFlags.every((flag) => (relation.flags || []).includes(flag)))
        .filter(([, relation]) => includeRetired || !relation.retired)
        .map(([id]) => id));
    }
    for (const id of stringList(include, `${context} include`)) {
      if (!(id in this.relations)) fail(`${context}: unknown relation ${JSON.stringify(id)} in include`);
      if (members.has(id)) fail(`${context}: include of ${id}, which is already a member`);
      members.add(id);
    }
    for (const id of stringList(exclude, `${context} exclude`)) {
      if (!(id in this.relations)) fail(`${context}: unknown relation ${JSON.stringify(id)} in exclude`);
      if (!members.has(id)) fail(`${context}: exclude of ${id}, which is not a member`);
      members.delete(id);
    }
    const ids = [...members].sort(byCodeUnit);
    if (!ids.length) fail(`${context}: empty selection`);
    return ids;
  }

  named(name, stack = []) {
    if (this.resolved?.has(name)) return [...this.resolved.get(name)];
    const definition = this.selections[name];
    if (!definition) fail(`unknown selection ${JSON.stringify(name)}`);
    if (stack.includes(name)) fail(`selection cycle: ${[...stack, name].join(" -> ")}`);
    for (const key of Object.keys(definition)) if (!SELECTION_KEYS.has(key)) fail(`selection ${name}: unknown key ${key}`);
    const { base } = definition;
    if (base !== undefined) this.named(base, [...stack, name]);
    const ids = this.select({
      base,
      sources: definition.sources,
      flags: definition.flags,
      include: definition.include,
      exclude: definition.exclude,
      includeRetired: definition.include_retired === true
    }, `selection ${name}`);
    this.resolved.set(name, ids);
    return [...ids];
  }

  canonical(id) {
    return Object.prototype.hasOwnProperty.call(this.aliases, id) ? this.aliases[id] : id;
  }

  displayName(source) {
    if (!this.sources.has(source)) fail(`unknown source ${JSON.stringify(source)}`);
    return this.sourceDisplay[source].name;
  }

  displayColor(source) {
    if (!this.sources.has(source)) fail(`unknown source ${JSON.stringify(source)}`);
    return this.sourceDisplay[source].color;
  }

  displayMark(source) {
    if (!this.sources.has(source)) fail(`unknown source ${JSON.stringify(source)}`);
    return this.sourceDisplay[source].mark;
  }

  source(id) {
    const relation = this.relations[id];
    if (!relation) fail(`unknown relation ${JSON.stringify(id)}`);
    return relation.source;
  }

  pinned(pin, context) {
    if (!pin || typeof pin.selection !== "string" || typeof pin.digest !== "string") {
      fail(`${context}: a pin needs selection and digest`);
    }
    const ids = this.named(pin.selection);
    const actual = digest(ids);
    if (actual !== pin.digest) {
      fail(`${context}: selection ${pin.selection} digest is ${actual}, the spec pins ${pin.digest}; its members changed, so re-render and re-pin deliberately`);
    }
    return ids;
  }
}

export function load(path = DEFAULT_PATH) {
  let text;
  try {
    text = readFileSync(path, "utf8");
  } catch (error) {
    fail(`cannot read ${path}: ${error.message}`);
  }
  return new RelationSets(path, text);
}

export function resolveSpec(spec, specPath) {
  const block = spec.relation_registry;
  if (!block) fail(`${specPath}: no relation_registry block`);
  if (spec.relations !== undefined || spec.relation_sets !== undefined) {
    fail(`${specPath}: literal relations/relation_sets beside relation_registry`);
  }
  if (typeof block.path !== "string") fail(`${specPath}: relation_registry.path missing`);
  if (block.relations === undefined && block.relation_sets === undefined) {
    fail(`${specPath}: relation_registry pins neither relations nor relation_sets`);
  }
  const registry = load(resolve(dirname(specPath), block.path));
  if (block.relations !== undefined) {
    const ids = registry.pinned(block.relations, `${specPath} relation_registry.relations`);
    if (block.palette !== undefined) {
      if (!Array.isArray(block.palette) || block.palette.length !== ids.length) {
        fail(`${specPath}: relation_registry.palette has ${block.palette?.length} colors for ${ids.length} relations`);
      }
      spec.relations = ids.map((id, index) => ({ id, color: block.palette[index] }));
    } else {
      spec.relations = ids.map((id) => ({ id }));
    }
  } else if (block.palette !== undefined) {
    fail(`${specPath}: relation_registry.palette needs a relations pin`);
  }
  if (block.relation_sets !== undefined) {
    spec.relation_sets = Object.fromEntries(Object.entries(block.relation_sets)
      .map(([key, pin]) => [key, registry.pinned(pin, `${specPath} relation_registry.relation_sets.${key}`)]));
  }
  return registry;
}

export function stripResolved(spec) {
  if (!spec.relation_registry) return spec;
  const { relations, relation_sets, ...rest } = spec;
  return rest;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const registry = load(process.env.RELATION_SETS_PATH || DEFAULT_PATH);
  const names = process.argv.slice(2).length ? process.argv.slice(2) : Object.keys(registry.selections);
  const out = Object.fromEntries(names.map((name) => {
    const ids = registry.named(name);
    return [name, { digest: digest(ids), ids }];
  }));
  process.stdout.write(`${JSON.stringify(out, null, 1)}\n`);
}
