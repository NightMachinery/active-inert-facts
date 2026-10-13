import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const W = JSON.parse(readFileSync(resolve(dirname(fileURLToPath(import.meta.url)), "font-widths.json"), "utf8"));

export function textWidth(text, size, bold = false) {
  const table = bold ? W.bold : W.regular;
  let em = 0;
  for (const ch of String(text)) em += table[ch] ?? W.fallback_em;
  return em * size;
}
