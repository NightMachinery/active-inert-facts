#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linroute.cache import read_cache, slug
from linroute.probes import ProbeConfig, run_relation

def load_cache(path: Path) -> dict:
    return read_cache(path)

def _jsonable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")

def _finite(obj):
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj

def _worker(job: tuple[str, str, ProbeConfig]) -> tuple[str, dict | None, str | None]:
    path_str, relation, cfg = job
    try:
        return relation, run_relation(load_cache(Path(path_str)), cfg), None
    except Exception as exc:
        return relation, None, f"{type(exc).__name__}: {exc}"

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, help="cache root directory")
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--relations", default="all", help="'all' or a comma-separated list")
    ap.add_argument("--out", default=None, help="default results/<slug>/probes.json")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="recompute existing relations")
    ap.add_argument("--folds-seed", type=int, default=0,
                    help="one seeded fold plan, shared by every probe of a relation")
    ap.add_argument("--perm-seed", type=int, default=0)
    ap.add_argument("--outer-folds", type=int, default=10)
    ap.add_argument("--inner-folds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5, help="folds of the plain-CV reports")
    ap.add_argument("--n-perm", type=int, default=5)
    ap.add_argument("--q-primary", type=int, default=16, help="fixed PCA capacity")
    ap.add_argument("--q-secondary", type=int, default=None,
                    help="optional second fixed capacity; omitted by default")
    ap.add_argument("--layers", default="", help="restrict layer selection, e.g. 0,4,8")
    return ap.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_dir = slug(args.model)
    cache_dir = Path(args.cache) / model_dir
    if not cache_dir.is_dir():
        raise SystemExit(f"no cache directory: {cache_dir}")

    out_path = Path(args.out) if args.out else Path("results") / model_dir / "probes.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if out_path.exists() and not args.force:
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add(row.get("relation_slug") or slug(row["relation"]))
    elif args.force and out_path.exists():
        out_path.unlink()

    available = sorted(p.stem for p in cache_dir.glob("*.pt"))
    wanted = (
        available
        if args.relations == "all"
        else [slug(r.strip()) for r in args.relations.split(",") if r.strip()]
    )
    missing = [r for r in wanted if r not in available]
    if missing:
        raise SystemExit(f"no cache file for: {', '.join(missing)}")

    cfg = replace(
        ProbeConfig(),
        q_primary=args.q_primary,
        q_secondary=args.q_secondary,
        folds_seed=args.folds_seed,
        perm_seed=args.perm_seed,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        folds=args.folds,
        n_perm=args.n_perm,
        layers=tuple(int(l) for l in args.layers.split(",") if l.strip()) or None,
    )
    jobs = [
        (str(cache_dir / f"{rel}.pt"), rel, cfg) for rel in wanted if rel not in done
    ]
    print(f"[probe_wild] {model_dir}: {len(jobs)} to run, {len(done)} already done")
    if not jobs:
        return 0

    n_fail = 0
    with out_path.open("a", encoding="utf-8") as fh:
        if args.jobs > 1:
            _pin_blas_threads()
            ctx = mp.get_context("spawn")
            with ctx.Pool(args.jobs) as pool:
                results = pool.imap_unordered(_worker, jobs)
                n_fail = _drain(results, fh, len(jobs), cfg.q_primary)
        else:
            n_fail = _drain((_worker(j) for j in jobs), fh, len(jobs), cfg.q_primary)
    print(f"[probe_wild] wrote {out_path} ({n_fail} failed)")
    return 1 if n_fail else 0

BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)

def _pin_blas_threads() -> None:
    for var in BLAS_THREAD_VARS:
        os.environ.setdefault(var, "1")

def _drain(results, fh, total: int, q: int) -> int:
    n_fail = 0
    for i, (relation, result, error) in enumerate(results, 1):
        if error is not None:
            n_fail += 1
            print(f"[{i}/{total}] {relation}: FAILED {error}", file=sys.stderr)
            continue
        row = {"relation_slug": relation, **_finite(result)}
        fh.write(json.dumps(row, default=_jsonable) + "\n")
        fh.flush()
        print(
            f"[{i}/{total}] {relation}: n={result['n']} "
            f"LIN-OBJ={result[f'lin_obj_q{q}_delta']:+.3f} "
            f"LU-OBJ={result['lu_obj_delta_cos_affine_cv']:+.3f} "
            f"5shot={result['fs_acc']:.3f}"
        )
    return n_fail

if __name__ == "__main__":
    raise SystemExit(main())
