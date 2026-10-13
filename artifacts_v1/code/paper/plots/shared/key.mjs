import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const KEY_SPACING = Object.freeze(JSON.parse(readFileSync(resolve(dirname(fileURLToPath(import.meta.url)), "key-spacing.json"), "utf8")));
export const keyBaselineBelow = (aboveBaseline, fs) => aboveBaseline + KEY_SPACING.descent_em * fs + KEY_SPACING.clear_pt + KEY_SPACING.cap_em * fs;
export const keyBottom = (lastBaseline, fs) => lastBaseline + KEY_SPACING.descent_em * fs + KEY_SPACING.bottom_pad_pt;

const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");

export const KEY_STYLE = Object.freeze({ rowH: 10.5, gap: 12, glyphW: 10, glyphGap: 4, titleGap: 8 });

export function keyBlocks(blocks, { itemsLeft, right, y, titleRight = null, fs, textWidth, text, style = KEY_STYLE, clear = false }) {
  const out = [];
  let row = 0;
  let maxX = itemsLeft;
  const titleW = Math.max(0, ...blocks.filter((b) => !b.titleAbove).map((b) => textWidth(b.title, fs, true)));
  for (const block of blocks) {
    const by = y + row * style.rowH;
    out.push(`<g data-key-block="${esc(block.id)}"${clear ? ' data-key-clear="1"' : ""}>`);
    let x;
    if (titleRight === null) maxX = Math.max(maxX, itemsLeft + textWidth(block.title, fs, true));
    if (titleRight !== null) {
      out.push(text(titleRight, by, block.title, { anchor: "end", weight: 700 }));
      x = itemsLeft;
    } else {
      out.push(text(itemsLeft, by, block.title, { weight: 700 }));
      x = block.titleAbove ? itemsLeft : itemsLeft + titleW + style.titleGap;
    }
    const start = x;
    let lines = block.titleAbove ? 2 : 1;
    for (const item of block.items) {
      const w = style.glyphW + style.glyphGap + textWidth(item.label, fs);
      if (x > start && (block.stack || x + w > right)) {
        x = start;
        lines += 1;
      }
      const iy = by + (lines - 1) * style.rowH;
      if (x + w > right + 0.5) throw new Error(`key item does not fit the width: ${item.label}`);
      out.push(`<g data-key-item="${esc(item.id || item.label)}">${item.glyph(x + style.glyphW / 2, iy - fs * 0.34)}`);
      out.push(text(x + style.glyphW + style.glyphGap, iy, item.label), "</g>");
      maxX = Math.max(maxX, x + w);
      x += w + style.gap;
    }
    out.push("</g>");
    row += lines;
  }

  return { svg: out.join(""), height: row * style.rowH, lastBaseline: y + (row - 1) * style.rowH, width: maxX - itemsLeft };
}
