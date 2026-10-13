#!/usr/bin/env python3
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "metadata"

def pins(node, found):
    if isinstance(node, dict):
        if isinstance(node.get("path"), str) and isinstance(node.get("sha256"), str):
            found.append(node)
        for v in node.values():
            pins(v, found)
    elif isinstance(node, list):
        for v in node:
            pins(v, found)
    return found

def main(check: bool) -> int:
    stale = []
    for spec in sorted((ROOT / "plots").glob("*/*.json")):
        if "spec" not in spec.name and "provenance" not in spec.name:
            continue
        text = spec.read_text(encoding="utf-8")
        new = text
        for pin in pins(json.loads(text), []):
            target = (spec.parent / pin["path"]).resolve()
            if target.parent != META or target.suffix != ".json5" or not target.exists():
                continue
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if pin["sha256"] != digest:
                new = new.replace(f'"{pin["sha256"]}"', f'"{digest}"')
        if new != text:
            stale.append(spec)
            if not check:
                spec.write_text(new, encoding="utf-8")
    for spec in stale:
        print(f"{'stale' if check else 'repinned'}: {spec.relative_to(ROOT)}")
    return 1 if stale and check else 0

if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv))
