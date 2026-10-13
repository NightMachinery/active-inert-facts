#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

SCHEMA = "linroute/label-exclusion/1"
STRINGS_SCHEMA = "linroute/label-exclusion-strings/1"
VERSION = "1.0.0"

def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sha256_list(items) -> str:
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()

def read_pool(path) -> list[str]:
    labels = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not labels or len(set(labels)) != len(labels):
        raise SystemExit(f"{path}: empty or duplicated pool")
    return labels

def collect_strings(arm_globs):
    rel = {}
    for pattern in arm_globs:
        dirs = sorted(glob.glob(pattern))
        if not dirs:
            raise SystemExit(f"--arms {pattern!r} matched nothing")
        for d in dirs:
            a = json.loads(Path(d, "arm.json").read_text(encoding="utf-8"))
            r = rel.setdefault(a["relation_slug"], {"subjects": set(), "classes": set(),
                                                    "n_arms": 0, "n_classes_max": 0})
            r["n_arms"] += 1
            r["n_classes_max"] = max(r["n_classes_max"], int(a["n_classes"]))
            r["subjects"].update(s["town"] for s in a["subjects"])
            r["subjects"].update(s["subject"] for s in a.get("support_subjects", []))
            r["classes"].update(a["class_label"].values())
    out = {}
    for slug in sorted(rel):
        subs, cls = sorted(rel[slug]["subjects"]), sorted(rel[slug]["classes"])
        out[slug] = {"subjects": subs, "classes": cls,
                     "subjects_sha256": sha256_list(subs), "classes_sha256": sha256_list(cls),
                     "n_arms": rel[slug]["n_arms"], "n_classes_max": rel[slug]["n_classes_max"]}
    return out

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", action="append", default=[],
                    help="glob of built arm directories (repeatable); omit to reuse --strings")
    ap.add_argument("--strings", help="an existing relation-strings.json instead of --arms")
    ap.add_argument("--pool", action="append", required=True, help="pool.txt (repeatable)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--rule-id", default=None, help="default: P<k>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nltk-data", default=None, help="prepended to nltk.data.path")
    args = ap.parse_args(argv)
    rule_id = args.rule_id or f"P{args.k}"

    import nltk
    if args.nltk_data:
        nltk.data.path.insert(0, args.nltk_data)
    from nltk.corpus import wordnet as wn
    wn_version = wn.get_version()
    if wn_version != "3.0":
        raise SystemExit(f"WordNet {wn_version} loaded; the rule is pinned to 3.0")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.arms:
        strings = collect_strings(args.arms)
        doc = {"schema": STRINGS_SCHEMA, "generator": "scripts/make_label_exclusions.py",
               "generator_version": VERSION,
               "source": "union over the built arms of subjects[].town, "
                         "support_subjects[].subject and class_label values, per relation_slug",
               "n_relations": len(strings), "relations": strings}
        (out / "relation-strings.json").write_text(
            json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    elif args.strings:
        strings = json.loads(Path(args.strings).read_text(encoding="utf-8"))["relations"]
    else:
        raise SystemExit("give --arms or --strings")
    strings_path = out / "relation-strings.json" if args.arms else Path(args.strings)
    strings_sha = sha256_file(strings_path)

    @lru_cache(None)
    def senses(s):
        return tuple(wn.synsets(re.sub(r"\s+", "_", s.strip()), "n"))

    @lru_cache(None)
    def up(syn):
        d, frontier, depth = {syn: 0}, [syn], 0
        while frontier:
            depth += 1
            nxt = []
            for x in frontier:
                for h in x.hypernyms() + x.instance_hypernyms():
                    if h not in d:
                        d[h] = depth
                        nxt.append(h)
            frontier = nxt
        return d

    @lru_cache(None)
    def primary(label):
        ss = wn.synsets(label, "n")
        return ss[0] if ss else None

    @lru_cache(None)
    def relation_index(strs):
        literal = {}
        for s in strs:
            literal.setdefault(s.lower(), s)
        above, sense = {}, {}
        for s in strs:
            for sig in senses(s):
                if sig not in sense:
                    sense[sig] = s
                for x, d in up(sig).items():
                    if x not in above or d < above[x][0]:
                        above[x] = (d, s)
        return literal, above, sense

    def nearest(label, strs):
        literal, above, sense = relation_index(strs)
        if label.lower() in literal:
            return (0, literal[label.lower()], "literal")
        p = primary(label)
        if p is None:
            return None
        best = None
        if p in above:
            d, s = above[p]
            best = (d, s, "same_sense" if d == 0 else "label_above")
        for x, d in sorted(up(p).items(), key=lambda kv: (kv[1], sense.get(kv[0], ""))):
            if x in sense and (best is None or d < best[0]):
                best = (d, sense[x], "same_sense" if d == 0 else "label_below")
                break
        return best

    for pool_path in args.pool:
        labels = read_pool(pool_path)
        name = Path(pool_path).parent.name
        rels = {}
        for slug, st in strings.items():
            strs = tuple(sorted(set(st["subjects"]) | set(st["classes"])))
            excluded = {}
            for L in labels:
                hit = nearest(L, strs)
                if hit is not None and hit[0] <= args.k:
                    excluded[L] = {"d": hit[0], "string": hit[1], "direction": hit[2]}
            kept = [L for L in labels if L not in excluded]
            rels[slug] = {
                "excluded": sorted(excluded),
                "why": {L: excluded[L] for L in sorted(excluded)},
                "n_excluded": len(excluded),
                "n_remaining": len(kept),
                "n_classes_max": st["n_classes_max"],
                "strings_sha256": sha256_list(strs),
            }
        doc = {
            "schema": SCHEMA,
            "rule": {
                "id": rule_id, "k": args.k, "reading": "strict",
                "label_sense": "primary (first WordNet noun sense)",
                "string_senses": "every noun sense of the whole string (spaces -> underscores)",
                "edges": "hypernyms + instance_hypernyms, either direction",
                "literal": "case-insensitive equality with any subject or class string, at d=0",
                "applies_to": "every relation identically; the draw is unchanged "
                              "(linroute.arms.keyed_shuffle over the pool minus the exclusion)",
            },
            "wordnet_version": wn_version,
            "nltk_version": nltk.__version__,
            "generator": "scripts/make_label_exclusions.py",
            "generator_version": VERSION,
            "pool": {"path": os.path.relpath(pool_path) if not os.path.isabs(pool_path) else pool_path,
                     "dir": name, "sha256": sha256_file(pool_path), "n": len(labels)},
            "strings": {"path": strings_path.name, "sha256": strings_sha},
            "n_relations": len(rels),
            "n_relations_with_exclusions": sum(1 for r in rels.values() if r["n_excluded"]),
            "min_remaining": min(r["n_remaining"] for r in rels.values()),
            "relations": rels,
        }
        (out / f"{name}.json").write_text(
            json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{name}: {doc['n_relations_with_exclusions']}/{len(rels)} relations prune, "
              f"min remaining {doc['min_remaining']}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    sys.exit(main())
