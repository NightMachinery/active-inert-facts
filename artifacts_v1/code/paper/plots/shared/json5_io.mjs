import { readFileSync } from "node:fs";
import JSON5 from "json5";

export function parse(text, name = "<json5>") {
  try {
    return JSON5.parse(text);
  } catch (error) {
    throw new Error(`${name}: ${error.message}`);
  }
}

export const load = (path) => parse(readFileSync(path, "utf8"), String(path));
