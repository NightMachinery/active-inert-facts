#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "plots" / "shared"))
import relation_sets

SOURCE = REPO_ROOT / "metadata" / "relation-sets.json5"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from machine_generated import MG, header
TARGET = MG / "relation-set-names.tex"
SPECIAL = set("\\{}$&#^_%~")

def render() -> str:
    raw = SOURCE.read_bytes()
    sets = relation_sets.load(SOURCE)
    lines = [
        header("latex/iclr2027/tools/gen_relation_set_names.py",
               [("metadata/relation-sets.json5", hashlib.sha256(raw).hexdigest())],
               "cd latex/iclr2027 && python3 tools/gen_relation_set_names.py").rstrip("\n"),
        "% Displayed relation-set names; the macro names keep the internal source keys.",
    ]
    for key in sorted(sets.sources):
        name = sets.display_name(key)
        if SPECIAL & set(name):
            raise SystemExit(f"gen_relation_set_names: {key} name {name!r} has a LaTeX special character")
        macro = "RelSetName" + "".join(part.capitalize() for part in key.split("_"))
        lines.append(f"\\newcommand{{\\{macro}}}{{{name}}}")
    return "\n".join(lines) + "\n"

def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true")
    g.add_argument("--stdout", action="store_true")
    a = ap.parse_args()
    text = render()
    if a.stdout:
        sys.stdout.write(text)
    elif a.check:
        if not TARGET.exists() or TARGET.read_text(encoding="utf-8") != text:
            raise SystemExit(f"gen_relation_set_names: {TARGET.relative_to(REPO_ROOT)} is stale; "
                             "run python3 tools/gen_relation_set_names.py")
    else:
        TARGET.write_text(text, encoding="utf-8")

if __name__ == "__main__":
    main()
