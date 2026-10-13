#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "sections" / "abstract.tex"
OUT = ROOT / "machine-generated" / "abstract.txt"

MATH = {r"\to": "→", r"\rightarrow": "→", r"\mapsto": "↦", r"\times": "×", r"\approx": "≈", r"\leq": "≤", r"\geq": "≥", r"\rho": "ρ"}
ONE_ARG_KEEP = ("emph", "textit", "textbf", "textsc", "mbox", "text")

def strip_comments(tex: str) -> str:
    lines = []
    for line in tex.splitlines():
        stripped = re.sub(r"(?<!\\)%.*", "", line)
        if stripped.strip() or not line.strip():
            lines.append(stripped)
    return "\n".join(lines)

def read_macros() -> dict[str, str]:
    macros: dict[str, str] = {}
    pattern = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{((?:[^{}]|\{[^{}]*\})*)\}")
    paths = sorted((ROOT / "notation").glob("*.tex")) + sorted((ROOT / "machine-generated").glob("*-numbers.tex"))
    paths.append(ROOT / "machine-generated" / "relation-set-names.tex")
    for path in paths:
        for name, body in pattern.findall(strip_comments(path.read_text())):
            macros[name] = body
    return macros

def expand_numexpr(body: str, macros: dict[str, str]) -> str:
    m = re.fullmatch(r"\\the\\numexpr(.*)\\relax", body.strip())
    if not m:
        return body
    terms = []
    for term in m.group(1).split("+"):
        term = term.strip()
        value = expand_numexpr(macros[term[1:]], macros) if term.startswith("\\") else term
        terms.append(int(value))
    return str(sum(terms))

def replace_braced(text: str, command: str, render) -> str:
    out, i, token = [], 0, "\\" + command + "{"
    while (j := text.find(token, i)) != -1:
        depth, k = 1, j + len(token)
        while depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        out += [text[i:j], render(text[j + len(token):k - 1])]
        i = k
    return "".join(out) + text[i:]

def export() -> str:
    tex = strip_comments(SRC.read_text())
    m = re.search(r"\\begin\{abstract\}(.*)\\end\{abstract\}", tex, re.S)
    if not m:
        sys.exit(f"{SRC}: no abstract environment")
    text = m.group(1)
    text = replace_braced(text, "pendingFleet", lambda arg: f"[pending: {arg}]")
    macros = read_macros()
    for _ in range(5):
        text = re.sub(r"\\([A-Za-z]+)\{\}|\\([A-Za-z]+)(?![A-Za-z])",
                      lambda mm: expand_numexpr(macros[mm.group(1) or mm.group(2)], macros)
                      if (mm.group(1) or mm.group(2)) in macros else mm.group(0), text)
    for command in ONE_ARG_KEEP:
        text = replace_braced(text, command, lambda arg: arg)

    def math(mm: re.Match) -> str:
        body = mm.group(1)
        for command, glyph in MATH.items():
            body = re.sub(re.escape(command) + r"(?![A-Za-z])", glyph, body)
        return body.strip()

    text = re.sub(r"\$([^$]*)\$", math, text)
    text = text.replace("---", "—").replace("--", "–").replace("~", " ")
    text = text.replace("``", "“").replace("''", "”").replace("\\%", "%").replace("\\&", "&")
    leftover = sorted(set(re.findall(r"\\[A-Za-z]+|[{}$]", text)))
    if leftover:
        sys.exit(f"{SRC}: cannot export, unhandled LaTeX left over: {' '.join(leftover)}")
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    return "\n\n".join(p for p in paragraphs if p) + "\n"

def main() -> None:
    text = export()
    if "--check" in sys.argv[1:]:
        if not OUT.exists() or OUT.read_text() != text:
            sys.exit(f"{OUT} is stale; run make abstract")
        return
    OUT.write_text(text)

if __name__ == "__main__":
    main()
