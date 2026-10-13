#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIELDS = ("share", "shuffle", "chance", "ci", "shuffle_ci", "n_err", "n_cells")

ap = argparse.ArgumentParser()
ap.add_argument("--spec", default=str(HERE / "spec.json"))
ap.add_argument("--source", required=True)
ap.add_argument("--meta", required=True)
ap.add_argument("--check", action="store_true")
a = ap.parse_args()

spec = json.load(open(a.spec))
sha = lambda b: hashlib.sha256(b).hexdigest()
src_bytes = Path(a.source).read_bytes()
meta_bytes = Path(a.meta).read_bytes()
if sha(src_bytes) != spec["source"]["sha256"]:
    sys.exit(f"freeze: {a.source} sha256 {sha(src_bytes)} is not the pinned source {spec['source']['sha256']}")
if sha(meta_bytes) != spec["source"]["meta_sha256"]:
    sys.exit(f"freeze: {a.meta} sha256 differs from the pinned meta.json")
meta = json.loads(meta_bytes)
S = json.loads(src_bytes)
for k in spec["source"]["key"]:
    S = S[k]

def group(D):
    missing = [f for f in FIELDS if f not in D]
    if missing:
        sys.exit(f"freeze: a group lacks {missing}")
    return {f: D[f] for f in FIELDS}

snap = {
    "schema": "failure-shapley/1",
    "source_sha256": spec["source"]["sha256"],
    "source_key": spec["source"]["key"],
    "meta": {k: meta[k] for k in ("source", "built", "linroute_commit", "panel_note")},
    "order": S["order"],
    "rule": S["rule"],
    "pooled": group(S["pooled"]),
    "by_model": {m: group(D) for m, D in S["by_model"].items()},
}
text = json.dumps(snap, indent=1, sort_keys=True) + "\n"
digest = sha(text.encode())
stamp = meta["built"][:10].replace("-", "")
out = HERE / "data" / f"shapley-{stamp}-{digest[:8]}.json"
if a.check:
    pinned = HERE / spec["snapshot"]
    if pinned.read_text() != text or spec["snapshot_sha256"] != digest:
        sys.exit("freeze --check: the rebuilt snapshot differs from the pinned one")
    print(f"freeze --check: {pinned.name} reproduces ({digest})")
    sys.exit(0)
out.parent.mkdir(exist_ok=True)
out.write_text(text)
print(out.relative_to(HERE), digest)
