import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parse as parseJson5 } from "./json5_io.mjs";
import { load as loadRelationSets } from "./relation-sets.mjs";

export const DEFAULT_PATH = resolve(dirname(fileURLToPath(import.meta.url)), "../../metadata/probe-names.json5");
const REF = /\{([A-Za-z0-9_$]+(?:\.[A-Za-z0-9_$]+)+)\}/g;
export const TERM_SETS = ["technical", "readable"];
const CASINGS = ["sentence", "title", "lower"];

export const FIGURE_STYLES = ["B"];

class TermsError extends Error {}
const fail = (message) => {
  throw new TermsError(`terms: ${message}`);
};

export class Terms {
  constructor(path, text, relationSets) {
    try {
      this.doc = parseJson5(text, path);
    } catch (error) {
      fail(error.message);
    }
    if (this.doc.schema !== "linroute/probe-names/1") fail(`${path} must declare schema linroute/probe-names/1`);
    if (!this.doc.terms || typeof this.doc.terms !== "object") fail(`${path} has no terms section`);
    const style = this.doc.typography?.probe_style;
    if (!FIGURE_STYLES.includes(style)) fail(`${path} typography.probe_style is ${style}; figures print capitals, so it must be one of ${FIGURE_STYLES.join(", ")}`);
    this.path = path;
    this.sha256 = createHash("sha256").update(text).digest("hex");
    this.sets = relationSets;
    const t = this.doc.terms;
    this.active = t.active;
    if (!TERM_SETS.includes(this.active)) fail(`${path} terms.active must be one of ${TERM_SETS.join(", ")}`);
    if (!CASINGS.includes(t.readable_casing)) fail(`${path} terms.readable_casing must be one of ${CASINGS.join(", ")}`);
    for (const name of [...TERM_SETS, "common"]) if (!t[name] || typeof t[name] !== "object") fail(`${path} terms.${name} must be an object`);
    const keysA = Object.keys(t.technical).sort().join(",");
    if (keysA !== Object.keys(t.readable).sort().join(",")) fail("terms.technical and terms.readable define different keys");
    for (const key of Object.keys(t.technical)) {
      if (key in t.common) fail(`terms.${key} is in a named set and in common`);
      const tex = TERM_SETS.map((n) => JSON.stringify(Object.entries(t[n][key]).filter(([f]) => f.startsWith("tex_")).sort()));
      if (tex[0] !== tex[1]) fail(`terms.${key}: the tex_ fields differ between the technical and readable sets`);
    }

    for (const name of [...TERM_SETS, "common"]) {
      for (const [key, entry] of Object.entries(t[name])) {
        if (typeof entry.plain !== "string" || !entry.plain) fail(`terms.${name}.${key} needs a non-empty plain form`);
        for (const [form, value] of Object.entries(entry)) if (typeof value === "string" && !form.startsWith("tex_")) this.resolve(value);
      }
    }
  }

  casing(value) {
    const mode = this.doc.terms.readable_casing;
    if (mode === "lower") return value.toLowerCase();
    if (mode === "title") return value.replace(/(^|[\s-])([a-z])/g, (_, a, b) => a + b.toUpperCase());
    return value;
  }

  raw(path) {
    const keys = path.split(".");
    if (keys[0] === "terms" && keys.length === 4) {
      const t = this.doc.terms;
      let name = keys[1] === "$" ? this.active : keys[1];
      if (![...TERM_SETS, "common"].includes(name)) fail(`${this.path} has no term set ${keys[1]} (at ${path})`);
      let entry = t[name][keys[2]];
      if (entry === undefined && keys[1] === "$") [name, entry] = ["common", t.common[keys[2]]];
      const value = entry?.[keys[3]];
      if (typeof value !== "string") fail(`${this.path} has no string at ${path}`);
      return name === "readable" && !value.includes("{") ? this.casing(value) : value;
    }
    const value = keys.reduce((node, k) => (node && typeof node === "object" ? node[k] : undefined), this.doc);
    if (typeof value !== "string") fail(`${this.path} has no string at ${path}`);
    return value;
  }

  resolve(value, stack = []) {
    return value.replace(REF, (_, path) => {
      if (stack.includes(path)) fail(`circular reference ${[...stack, path].join(" -> ")}`);
      return this.resolve(this.raw(path), [...stack, path]);
    });
  }

  term(key, form = "plain") {
    const t = this.doc.terms;
    const entry = t[this.active][key] || t.common[key];
    if (!entry) fail(`no term ${key}`);
    if (typeof entry[form] !== "string") fail(`term ${key} has no ${form} form`);
    return this.resolve(this.raw(`terms.$.${key}.${form}`));
  }

  at(path) {
    return this.resolve(this.raw(path));
  }

  setName(source) {
    return this.sets.displayName(source);
  }
}

export function load(path = process.env.PROBE_NAMES_PATH || DEFAULT_PATH, relationSets = loadRelationSets()) {
  return new Terms(path, readFileSync(path, "utf8"), relationSets);
}
