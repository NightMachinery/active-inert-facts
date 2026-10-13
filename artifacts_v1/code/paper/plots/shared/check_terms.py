#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import re
import sys
import tokenize
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import terms as terms_mod

OPERATOR_LINE = re.compile(r"\b(fail|die|raise|sys\.exit|throw|print|assert|console\.\w+|SystemExit|stderr\.write)\b")
EXEMPT = "terms-ok:"
DOC_KEY = re.compile(r"(^|_)(note|notes|description|question|reason|status|comment|rule|history|provenance|path|field|id|sha256|source|sources|schema|package|figure_id|aggregation|snapshot\w*)$")
JS_LITERAL = re.compile(r'"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'|`(?:[^`\\]|\\.)*`', re.S)

def names(t: terms_mod.Terms) -> dict[str, str]:
    doc = t.doc
    out: dict[str, str] = {}
    def add(value: str, where: str) -> None:
        value = t.resolve(value)
        if len(value) >= 3 and (" " in value or re.search(r"[A-Z]{2,}-[A-Z]", value)):
            out.setdefault(value, where)
    for key, e in doc["probe_families"].items():
        for form in ("caps", "mixed"):
            add(e[form], f"probe_families.{key}.{form}")
    for key, e in doc["relation_roles"].items():
        add(e["caps"], f"relation_roles.{key}.caps")
    for key, e in doc["metrics"].items():
        for form in ("plain", "axis_label", "axis_short_label"):
            if form in e:
                add(e[form], f"metrics.{key}.{form}")
    for key, e in doc["controls"].items():
        add(e["header"], f"controls.{key}.header")
    for name in ("common",):
        for key, e in doc["terms"][name].items():
            for form, value in e.items():
                if isinstance(value, str) and not form.startswith("tex_"):
                    add(value, f"terms.{name}.{key}.{form}")

    for name in terms_mod.TERM_SETS:
        for value in t.set_strings(name):
            if len(value) >= 3 and not value.islower():
                out.setdefault(value, f"terms.{name}")
    for source in t.sets.sources:
        out.setdefault(t.set_name(source), f"relation-sets source_display.{source}")
    return out

def matcher(table: dict[str, str]):
    parts = sorted(table, key=len, reverse=True)
    rx = re.compile(r"(?<![A-Za-z0-9-])(" + "|".join(re.escape(p) for p in parts) + r")(?![A-Za-z0-9])")
    return lambda s: [m.group(1) for m in rx.finditer(s)]

def js_literals(text: str):
    lines = text.split("\n")

    text = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)
    text = "\n".join(re.sub(r"(^|\s)//.*$", r"\1", ln) for ln in text.split("\n"))
    for m in JS_LITERAL.finditer(text):
        lineno = text.count("\n", 0, m.start()) + 1
        yield lineno, lines[lineno - 1], m.group(0)[1:-1]

def py_literals(text: str):
    lines = text.split("\n")
    prev = None
    fstring: list[str] = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        kind = tokenize.tok_name[tok.type]
        if kind in ("FSTRING_START",):
            fstring = [tok.string]
            start = tok.start[0]
            continue
        if kind == "FSTRING_MIDDLE":
            fstring.append(tok.string)
            continue
        if kind == "FSTRING_END":
            yield start, lines[start - 1], "".join(fstring[1:])
            fstring = []
            prev = tok
            continue
        if kind == "STRING":
            statement_start = prev is None or tokenize.tok_name[prev.type] in ("NEWLINE", "INDENT", "DEDENT", "NL", "COMMENT")
            if not statement_start:
                yield tok.start[0], lines[tok.start[0] - 1], tok.string
        if kind not in ("NL", "COMMENT"):
            prev = tok

def spec_literals(path: Path):
    lines = path.read_text(encoding="utf-8").split("\n")
    doc = json.loads("\n".join(lines))
    hits = []
    def walk(node, key):
        if isinstance(node, dict):
            for k, v in node.items():
                if not DOC_KEY.search(k):
                    walk(v, k)
        elif isinstance(node, list):
            for v in node:
                walk(v, key)
        elif isinstance(node, str):
            hits.append(node)
    walk(doc, "")
    for value in hits:
        needle = json.dumps(value, ensure_ascii=False)
        lineno = next((i + 1 for i, ln in enumerate(lines) if needle in ln), 0)
        yield lineno, lines[lineno - 1] if lineno else "", value

def check(package: Path, find) -> list[str]:
    problems = []
    for path in sorted(package.iterdir()):
        if path.name.startswith("test_") or path.name.endswith(".test.mjs"):
            continue
        if path.suffix in (".mjs", ".js"):
            source = js_literals(path.read_text(encoding="utf-8"))
        elif path.suffix == ".py":
            source = py_literals(path.read_text(encoding="utf-8"))
        elif path.suffix == ".json" and "spec" in path.name:
            source = spec_literals(path)
        else:
            continue
        for lineno, line, literal in source:
            if EXEMPT in line or (path.suffix != ".json" and OPERATOR_LINE.search(line)):
                continue
            for name in find(literal):
                problems.append(f"{path}:{lineno}: literal term {name!r}; read it from the terms registry")
    return problems

def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__.split("\n\n")[1])
        return 2
    t = terms_mod.load()
    find = matcher(names(t))
    problems = [p for d in argv for p in check(Path(d), find)]
    for p in problems:
        print(p)
    if problems:
        print(f"check_terms: {len(problems)} hard-coded term(s)")
        return 1
    print(f"check_terms: OK ({', '.join(argv)})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
