#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from linroute import arms

ARM_FILES = ("arm.json", "build-provenance.json", "mapping.tsv",
             "train.jsonl", "gate.jsonl", "heldout.jsonl")
MIN_MATERIALIZER = (1, 1, 0)

RESTAMPABLE = {"2.2.0": "no label_exclusion and no explicit-map override"}

def _materializer():
    spec = importlib.util.spec_from_file_location(
        "materialize_newrels", REPO_ROOT / "scripts/materialize_newrels.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class RenderError(Exception):
    pass

def need(cond, msg):
    if not cond:
        raise RenderError(msg)

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def read_cell(path: Path) -> dict:
    mat = _materializer()
    return json.loads(mat.strip_comment_lines(path.read_text("utf-8")))

def _checked(path: Path | str, sha: str, what: str) -> Path:
    p = Path(path)
    need(p.is_file(), f"{what} missing: {p}")
    need(arms.sha256_file(p) == sha, f"{what} sha256 mismatch: {p}")
    return p

def render_cell(d: dict) -> dict:
    cell, pv = d["cell"], d["provenance"]
    need("render" in d, "no render block: materialize this arm set with "
                        "materialize_newrels.py v1.1.0 or later")
    r = d["render"]
    bp = r["build_provenance"]

    manifest_path = _checked(pv["manifest"]["path"], pv["manifest"]["sha256"], "manifest")
    manifest = arms.load_manifest(manifest_path)
    records_path = _checked(pv["records"]["path"], pv["records"]["sha256"], "records")
    records = arms.load_records(records_path)

    tpath = _checked(REPO_ROOT / r["templates"], pv["template_file"]["sha256"], "template file")
    prompts = arms.load_newrel(tpath)

    prompts = replace(prompts, path=r["template_file_path"])

    pool_path = _checked(REPO_ROOT / r["label_pool_path"], d["draw"]["label_pool_sha256"],
                         "label pool")
    animals = arms.load_animals(pool_path)
    category_word = arms.pool_category_word(pool_path.parent)
    need(category_word == r["category_word"],
         f"the pool declares category word {category_word!r}, the cell recorded "
         f"{r['category_word']!r}")

    exclusion = None
    if r["label_exclusion"] is not None:
        ex = r["label_exclusion"]
        epath = _checked(REPO_ROOT / ex["path"], ex["sha256"], "label exclusion")
        exclusion = arms.load_label_exclusion(epath, pool_path=pool_path)
        exclusion["_path"] = ex["path"]

    class_animal = {int(c["class"]): c["new_rel"] for c in d["classes"]}

    arm = arms.build_arm(
        manifest, records, prompts, animals,
        repeat=int(cell["repeat"]),
        role_seed=r["role_seed"],
        bijection_slug=r["bijection_slug"],
        pool_tokenizer=r["pool_tokenizer"],
        tokenizers=tuple(r["asserted_tokenizers"]),
        category_word=category_word,
        name=r["arm_name"],
        relation_type=r["relation_type"],
        manifest_path=pv["manifest"]["path"],
        manifest_sha256=pv["manifest"]["sha256"],
        records_path=pv["records"]["path"],
        records_sha256=pv["records"]["sha256"],
        animals_path=r["label_pool_path"],
        git_commit=r["git_commit"],
        label_exclusion=exclusion,
        class_animal=class_animal,
    )
    meta = arm["meta"]

    want = [(s["subject"], s["split"]) for s in d["subjects"] if s["split"] != "support"]
    got = [(s["town"], s["split"]) for s in meta["subjects"]]
    need(want == got, "the cell's train/heldout subjects or splits differ from the manifest's")
    want_sup = [s["subject"] for s in d["subjects"] if s["split"] == "support"]
    need(want_sup == [s["subject"] for s in meta["support_subjects"]],
         "the cell's support subjects differ from the manifest's")
    rendered = {name: text for group in meta["templates"].values() for name, text in group.items()}
    need({t["name"]: t["template"] for t in d["templates"]} == rendered,
         "the cell's templates differ from the template file's")

    if meta["generator_version"] != r["generator_version"]:
        cond = RESTAMPABLE.get(r["generator_version"])
        need(cond is not None and "label_exclusion" not in meta
             and "class_animal_source" not in meta,
             f"cannot stamp generator_version {r['generator_version']} "
             f"from {meta['generator_version']}")
        meta["generator_version"] = r["generator_version"]

    need(bp["git_commit"] == r["git_commit"], "build-provenance commit differs from the arm's")
    need(bp["pool"]["sha256"] == arms.sha256_file(pool_path), "build-provenance pool sha")
    need(bp["manifest"]["sha256"] == pv["manifest"]["sha256"], "build-provenance manifest sha")
    need(bp["records"]["sha256"] == pv["records"]["sha256"], "build-provenance records sha")
    need(bp["templates"]["sha256"] == pv["template_file"]["sha256"], "build-provenance templates sha")
    if "label_exclusion" in bp:
        need(exclusion is not None and bp["label_exclusion"]["sha256"] == exclusion["_sha256"],
             "build-provenance exclusion sha")

    out = {}
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        arms.write_arm(Path(tmp) / "arm", arm)
        for f in ARM_FILES:
            if f != "build-provenance.json":
                out[f] = (Path(tmp) / "arm" / f).read_bytes()
    out["build-provenance.json"] = (json.dumps(bp, indent=1, sort_keys=True) + "\n").encode("utf-8")
    return out

def _one(args):
    path, out_root, check = args
    rel = None
    try:
        d = read_cell(Path(path))
        c = d["cell"]
        rel = f"{c['model_slug']}/{c['relation']}/r{c['repeat']}"
        files = render_cell(d)
        dst = Path(out_root) / "arms" / rel
        dst.mkdir(parents=True, exist_ok=True)
        for f, b in files.items():
            (dst / f).write_bytes(b)
        res = {"cell": rel, "ok": True}
        if check:
            want = d["provenance"]["file_sha256"]
            diff = {f: {"rendered": sha256_bytes(b), "packed": want.get(f)}
                    for f, b in files.items() if sha256_bytes(b) != want.get(f)}
            res.update(identical=not diff, differs=diff)
        return res
    except Exception as exc:
        return {"cell": rel or str(path), "ok": False, "error": f"{type(exc).__name__}: {exc}"}

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-set", required=True, help="data/newrel-materialized/<arm-set-id>")
    ap.add_argument("--out", required=True, help="writes <out>/arms/<model>/<relation>/r<k>/")
    ap.add_argument("--cell", action="append", default=None,
                    help="only this <model_slug>/<relation>/r<k> (repeatable)")
    ap.add_argument("--check", action="store_true",
                    help="compare every rendered file with the packed arm's recorded sha256")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)

    root = Path(a.arm_set)
    mat = _materializer()
    mf = json.loads(mat.strip_comment_lines((root / "manifest.json5").read_text("utf-8")))
    idx = json.loads(mat.strip_comment_lines((root / "index.json5").read_text("utf-8")))
    ver = tuple(int(x) for x in str(mf.get("materializer", {}).get("version", "0.0.0")).split("."))
    if ver < MIN_MATERIALIZER:
        raise SystemExit(f"{root} was written by materializer {ver}; the render block needs "
                         f">= {'.'.join(map(str, MIN_MATERIALIZER))}: re-materialize it")
    paths = [root / row["path"] for row in idx["cells"]]
    if a.cell:
        want = set(a.cell)
        paths = [p for p in paths
                 if "/".join(p.relative_to(root).with_suffix("").parts) in want]
    os.makedirs(a.out, exist_ok=True)
    env_ok = os.environ.get("HF_HUB_OFFLINE") == "1"
    if not env_ok:
        print("note: HF_HUB_OFFLINE is not 1; tokenizers may be fetched", file=sys.stderr)
    with cf.ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(_one, [(str(p), a.out, a.check) for p in paths], chunksize=4))
    n_fail = sum(1 for r in results if not r["ok"])
    n_ident = sum(1 for r in results if r.get("identical"))
    report = {"arm_set": str(root), "arm_set_id": root.name, "n_cells": len(results),
              "n_rendered": len(results) - n_fail, "n_failed": n_fail,
              "checked": a.check, "n_identical": n_ident if a.check else None,
              "cells": sorted(results, key=lambda r: r["cell"])}
    with open(Path(a.out) / "render-report.json", "w") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"RENDERED {report['n_rendered']}/{len(results)} failed={n_fail}"
          + (f" identical={n_ident}/{len(results)}" if a.check else ""))
    return 0 if n_fail == 0 and (not a.check or n_ident == len(results)) else 1

if __name__ == "__main__":
    sys.exit(main())
