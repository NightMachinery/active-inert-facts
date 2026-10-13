export const MATH_FONT = "TeX Gyre Termes, Times, Liberation Serif, Times New Roman, serif";

export const mathSym = (s, pre = "") => `<tspan${pre} font-family="${MATH_FONT}" font-style="italic">${s}</tspan>`;

export function termWithSymbol(term, esc) {
  const m = /^(.*\s)(\S)$/u.exec(term);
  if (!m) throw new Error(`term does not end in a one-letter symbol: ${term}`);
  return esc(m[1]) + mathSym(esc(m[2]));
}
