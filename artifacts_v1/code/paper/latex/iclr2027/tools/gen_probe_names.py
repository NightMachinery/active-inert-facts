#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE = REPO_ROOT / "metadata" / "probe-names.json5"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from machine_generated import MG, header
TARGET = MG / "probe-names.tex"

BODY_KIND = {"textrm": r"\textrm", "textsc": r"\textsc", "plain": ""}
TEX_SPECIAL = set("\\{}$&#^_%~")

sys.path.insert(0, str(REPO_ROOT / "plots" / "shared"))

def load_json5(path: Path) -> dict:
    import json5_io
    try:
        return json5_io.load(path)
    except json5_io.Json5Error as error:
        raise SystemExit(f"gen_probe_names: {error}")

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"gen_probe_names: {message}")

def tex_plain(document: dict, key: str, value: str) -> str:
    def one(match):
        path = match.group(1).split(".")
        require(len(path) == 3 and path[0] in ("relation_roles", "metrics", "probe_families"),
                f"term {key!r}: reference {match.group(0)} has no paper macro")
        entry = document[path[0]].get(path[1])
        require(entry is not None and "tex_macro" in entry, f"term {key!r}: {match.group(0)} names no macro")
        return "\\%s{}" % entry["tex_macro"]
    out = re.sub(r"\{([A-Za-z0-9_.]+)\}", one, value)
    require(not (TEX_SPECIAL & set(re.sub(r"\\[A-Za-z]+\{\}", "", out))), f"term {key!r} has a LaTeX special character")
    return out

def render(document: dict, source_sha256: str) -> str:
    require(document.get("schema") == "linroute/probe-names/1",
            "source must declare schema linroute/probe-names/1")
    for section in ("probe_families", "relation_roles", "metrics", "controls"):
        require(section in document, f"source is missing section {section!r}")

    out: list[str] = []
    out.append(header("latex/iclr2027/tools/gen_probe_names.py",
                      [("metadata/probe-names.json5", source_sha256)],
                      "cd latex/iclr2027 && python3 tools/gen_probe_names.py").rstrip("\n"))
    out.append("% Checker:   latex/iclr2027/tools/check_probe_names.py (make check)")
    out.append("%")
    out.append("% Every probe, metric and control name the paper prints is defined here and")
    out.append("% nowhere else, so renaming one is an edit to the .json5 and a regeneration.")
    out.append("% Edit the .json5.  Edits made here are overwritten and the build fails first.")
    out.append("%")
    out.append("% Requires \\DeclareProbeFamily and \\DeclareRelation from notation/probes.tex")
    out.append("% and \\probestyle from notation/probe-styles.tex; it selects the style itself.")
    out.append("")

    style = (document.get("typography") or {}).get("probe_style")
    require(isinstance(style, str) and style.isalpha() and len(style) == 1,
            "typography.probe_style must be one style letter of notation/probe-styles.tex")
    out.append("% ---- Typography: the probe style (typography.probe_style) ----")
    out.append("\\probestyle{%s}" % style)
    out.append("")

    out.append("% ---- Probe families ----")
    for key in sorted(document["probe_families"]):
        item = document["probe_families"][key]
        for field in ("tex_macro", "tex_bare", "caps", "lower", "mixed", "in_paper"):
            require(field in item, f"probe family {key!r} is missing {field!r}")
        flag = "printed in the manuscript" if item["in_paper"] else "declared only, NOT printed in the manuscript"
        out.append(f"% {key}: {flag}")
        out.append("\\DeclareProbeFamily{%s}{%s}{%s}{%s}{%s}"
                   % (item["tex_macro"], item["tex_bare"], item["caps"],
                      item["lower"], item["mixed"]))
    out.append("")

    out.append("% ---- Relation roles ----")
    for key in sorted(document["relation_roles"]):
        item = document["relation_roles"][key]
        for field in ("tex_macro", "caps", "lower", "mixed", "short"):
            require(field in item, f"relation role {key!r} is missing {field!r}")
        out.append("\\DeclareRelation{%s}{%s}{%s}{%s}{%s}"
                   % (item["tex_macro"], item["caps"], item["lower"],
                      item["mixed"], item["short"]))
    out.append("")

    import terms as terms_mod
    resolver = terms_mod.Terms(SOURCE, json.dumps(document).encode(), None)
    out.append("% ---- Metric names ----")
    for key in sorted(document["metrics"]):
        item = document["metrics"][key]
        for field in ("tex_macro", "tex_body_kind", "plain"):
            require(field in item, f"metric {key!r} is missing {field!r}")
        kind = item["tex_body_kind"]
        require(kind in BODY_KIND, f"metric {key!r} has unknown tex_body_kind {kind!r}")
        wrapper = BODY_KIND[kind]
        plain = resolver.resolve(item["plain"])
        require(not (TEX_SPECIAL & set(plain)), f"metric {key!r} name {plain!r} has a LaTeX special character")
        body = ("%s{%s}" % (wrapper, plain)) if wrapper else plain
        out.append("\\newcommand{\\%s}{%s}" % (item["tex_macro"], body))
    out.append("")

    out.append("% ---- Table 1 control names: column heading and running-prose phrase ----")
    for key in sorted(document["controls"]):
        item = document["controls"][key]
        for field in ("tex_macro_header", "tex_macro_prose", "header", "prose"):
            require(field in item, f"control {key!r} is missing {field!r}")
        out.append("\\newcommand{\\%s}{%s}" % (item["tex_macro_header"], item["header"]))
        out.append("\\newcommand{\\%s}{%s}" % (item["tex_macro_prose"], item["prose"]))
    out.append("")

    out.append("% ---- Figure terms with a caption macro (section terms; references resolved) ----")
    out.append("%% Term set: %s (terms.active in metadata/probe-names.json5)" % resolver.active)
    active_terms = {**document["terms"]["common"], **document["terms"][resolver.active]}
    for key in sorted(active_terms):
        item = active_terms[key]
        if "tex_macro" not in item:
            continue
        value = resolver.term(key, item.get("tex_form", "plain"))
        require(not (TEX_SPECIAL & set(value)), f"term {key!r} form {value!r} has a LaTeX special character")
        out.append("\\newcommand{\\%s}{%s}" % (item["tex_macro"], value))
        if "tex_macro_plain" in item:
            out.append("\\newcommand{\\%s}{%s}" % (item["tex_macro_plain"], tex_plain(document, key, item["plain"])))
    out.append("")

    return "\n".join(out)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if the target is missing or stale")
    parser.add_argument("--stdout", action="store_true", help="print instead of writing")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--target", type=Path, default=TARGET)
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    document = load_json5(args.source)
    text = render(document, hashlib.sha256(source_bytes).hexdigest())

    if args.stdout:
        sys.stdout.write(text)
        return 0
    if args.check:
        if not args.target.exists():
            print(f"check_probe_names: {args.target} does not exist; run gen_probe_names.py")
            return 1
        current = args.target.read_text(encoding="utf-8")
        if current != text:
            print(f"check_probe_names: {args.target} is STALE relative to {args.source}")
            print("check_probe_names: regenerate with tools/gen_probe_names.py")
            return 1
        print(f"check_probe_names: OK, {args.target.name} matches {args.source.name}")
        return 0
    args.target.write_text(text, encoding="utf-8")
    print(f"gen_probe_names: wrote {args.target}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
