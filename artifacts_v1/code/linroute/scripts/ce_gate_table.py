#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from linroute.ce_gate import GATE_VERSION, GateError, gate_path

ARTIFACT_NAMES = ("result.json", "lin_cls_ce.json")
COLUMNS = [
    "cell", "status_new", "gate_version", "reasons", "runner_status",
    "schema", "oracle_mode", "ce_gain", "raw_gain", "n", "C",
    "candidate_grid_complete", "requested_results_complete", "ridge_complete",
    "hygiene_ok", "fits", "fits_failing_hygiene", "max_rowsum_err",
    "max_gauge_err", "cache_git_commit", "ce_device", "sha256", "path",
]

def cell_id(path: Path) -> str:
    if path.name == "result.json" and "__" in path.parent.name:
        return path.parent.name
    if path.name == "lin_cls_ce.json":
        return f"{path.parent.parent.name}__{path.parent.name}"
    raise ValueError(f"cannot derive a cell id from {path}")

def find_artifacts(args: list[str]) -> list[Path]:
    found = []
    for arg in args:
        p = Path(arg)
        if p.is_file():
            found.append(p.resolve())
            continue
        if not p.is_dir():
            raise SystemExit(f"no such file or directory: {p}")
        for dirpath, _dirs, files in os.walk(p):
            for name in files:
                if name in ARTIFACT_NAMES:
                    found.append((Path(dirpath) / name).resolve())
    return sorted(set(found))

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def one(path: Path) -> dict:
    g = gate_path(path)
    c = g["conjuncts"]
    h = g["hygiene"]
    return {
        "cell": cell_id(path),
        "status_new": g["status"],
        "gate_version": g["gate_version"],
        "reasons": ",".join(g["reasons"]),
        "runner_status": g["runner_status"],
        "schema": g["schema"],
        "oracle_mode": g["runner_oracle_mode"],
        "ce_gain": "" if g["gain"] is None else repr(g["gain"]),
        "raw_gain": "" if g["raw_gain"] is None else repr(g["raw_gain"]),
        "n": g["n"],
        "C": g["n_classes"],
        "candidate_grid_complete": c["candidate_grid_complete"],
        "requested_results_complete": c["requested_results_complete"],
        "ridge_complete": c["ridge_complete"],
        "hygiene_ok": c["hygiene_ok"],
        "fits": h["fits"],
        "fits_failing_hygiene": h["fits_failing_hygiene"],
        "max_rowsum_err": repr(h["max_rowsum_err"]),
        "max_gauge_err": repr(h["max_gauge_err"]),
        "cache_git_commit": g["cache_git_commit"],
        "ce_device": g["ce_device"],
        "sha256": sha256_file(path),
        "path": str(path),
    }

def read_table(path: str) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    table = {}
    for row in rows:
        if row["cell"] in table:
            raise SystemExit(f"{path}: duplicate cell {row['cell']}")
        table[row["cell"]] = row
    return table

def compare(rows: list[dict], old: dict[str, dict]) -> list[str]:
    problems = []
    new = {row["cell"]: row for row in rows}
    for cell in sorted(set(old) - set(new)):
        problems.append(f"only in the old table: {cell}")
    for cell in sorted(set(new) - set(old)):
        problems.append(f"only in the new table: {cell}")
    for cell in sorted(set(new) & set(old)):
        a, b = new[cell], old[cell]
        if a["status_new"] != b.get("status_new"):
            problems.append(
                f"{cell}: status_new {a['status_new']} here, "
                f"{b.get('status_new')} in the old table"
            )
        if b.get("ce_gain") not in (None, "") and a["ce_gain"] != "":
            if float(a["ce_gain"]) != float(b["ce_gain"]):
                problems.append(
                    f"{cell}: ce_gain {a['ce_gain']} here, {b['ce_gain']} old"
                )
    return problems

def create_only(path: str, text: str) -> None:
    with open(path, "x", encoding="utf-8") as fh:
        fh.write(text)

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out-tsv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--check-against", default=None)
    ap.add_argument("--allow-duplicate-cells", action="store_true")
    ap.add_argument("--require-cache-commit", default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    for out in (a.out_tsv, a.out_json):
        if os.path.exists(out):
            raise SystemExit(f"refusing to overwrite {out}")

    paths = find_artifacts(a.inputs)
    if not paths:
        raise SystemExit("no CE artifacts found")
    try:
        if a.workers <= 1:
            rows = [one(path) for path in paths]
        else:
            with ProcessPoolExecutor(a.workers) as ex:
                rows = list(ex.map(one, paths, chunksize=2))
    except GateError as exc:
        raise SystemExit(f"GATE_ERROR {exc}") from exc
    cells = [row["cell"] for row in rows]
    dups = sorted({c for c in cells if cells.count(c) > 1})
    if dups and not a.allow_duplicate_cells:
        raise SystemExit(f"duplicate cell ids (pass --allow-duplicate-cells): {dups[:5]}")
    rows.sort(key=lambda row: (row["cell"], row["path"]))

    lines = ["\t".join(COLUMNS)]
    lines += ["\t".join(str(row[c]) for c in COLUMNS) for row in rows]
    tsv = "\n".join(lines) + "\n"
    summary = {
        "gate_version": GATE_VERSION,
        "cells": len(rows),
        "passed": sum(row["status_new"] == "passed" for row in rows),
        "failed": [
            {"cell": row["cell"], "reasons": row["reasons"], "path": row["path"]}
            for row in rows if row["status_new"] != "passed"
        ],
        "runner_status_disagrees": [
            row["cell"] for row in rows
            if (row["runner_status"] == "complete") != (row["status_new"] == "passed")
        ],
        "table_sha256": hashlib.sha256(tsv.encode("utf-8")).hexdigest(),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "code": str(Path(__file__).resolve()),
    }
    summary["cache_builds"] = sorted({
        f"{row['cache_git_commit']} {row['ce_device']}" for row in rows
    })
    problems = []
    if a.require_cache_commit:
        want = a.require_cache_commit
        for row in rows:
            if not str(row["cache_git_commit"] or "").startswith(want):
                problems.append(
                    f"{row['cell']}: cache built at {row['cache_git_commit']}, "
                    f"not {want} (never mix x across cache builds)"
                )
    if a.check_against:
        mismatches = compare(rows, read_table(a.check_against))
        problems += mismatches
        summary["check_against"] = {
            "table": a.check_against,
            "sha256": sha256_file(Path(a.check_against)),
            "agrees": not mismatches,
            "problems": mismatches,
        }
    create_only(a.out_tsv, tsv)
    create_only(a.out_json, json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(f"{GATE_VERSION}: {summary['passed']}/{summary['cells']} passed; "
          f"runner status disagrees on {len(summary['runner_status_disagrees'])}")
    for p in problems:
        print("CHECK_FAIL", p)
    if a.check_against:
        print("CHECK_AGAINST", "agrees" if summary["check_against"]["agrees"] else "DISAGREES")
    return 1 if problems else 0

if __name__ == "__main__":
    raise SystemExit(main())
