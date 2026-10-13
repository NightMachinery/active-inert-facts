#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

def _preset_threads(argv: list[str]) -> None:
    for i, arg in enumerate(argv):
        value = None
        if arg == "--threads" and i + 1 < len(argv):
            value = argv[i + 1]
        elif arg.startswith("--threads="):
            value = arg.split("=", 1)[1]
        if value is None:
            continue
        for var in _THREAD_VARS:
            os.environ.setdefault(var, str(int(value)))
        return

_preset_threads(sys.argv[1:])

import argparse
import json
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linroute import admission, transfer
from linroute.probes import ProbeConfig
from linroute.provenance import git_commit

MODELS = ("OLMo-3-1025-7B", "Qwen3.5-9B", "Qwen3.8-27B", "Qwen3.5-9B-Base",
          "gemma-4-E2B", "SmolLM3-3B-Base", "MiniCPM5-2B-Base", "G9v3-3B",
          "granite-4.2-3b", "Nanbeige4-3B-Base", "Phi-4-mini-instruct",
          "LFM2.5-2.6B-Base")

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-dir", required=True, help="a linroute.train output directory")
    p.add_argument("--arm-dir", required=True, help="the arm that run was trained on")
    p.add_argument("--wild-cache", default=None,
                   help="the relation's epoch-0 PRE-REL wild cache (schema 2)")
    p.add_argument("--manifest", default=None,
                   help="the split manifest whose folds are fitted (schema "
                        "linroute/admission/2); without it the probes fall back "
                        "to their internal seeded fold plan")
    p.add_argument("--model-name", required=True, choices=list(MODELS),
                   help="the short name the parent's analysis adapter expects")
    p.add_argument("--relation", required=True, help="the relation slug")
    p.add_argument("--repeat", required=True, type=int, choices=(0, 1, 2),
                   help="repeat index; it IS the bijection and optimizer seed")
    p.add_argument("--out", required=True, help="where to write metrics.json")
    p.add_argument("--rows-out", default=None,
                   help="rows.jsonl to append the flat row to")
    p.add_argument("--final-epoch", type=int, default=transfer.PRODUCTION_EPOCHS,
                   help="the PRESPECIFIED horizon e of the s_0 -> t_e probe "
                        "(default 20).  The run must actually have completed "
                        "it; an interrupted run fails here rather than being "
                        "measured at whatever epoch it reached")
    p.add_argument("--allow-intermediate", action="store_true",
                   help="measure an epoch other than the run's planned horizon; "
                        "the row is then 'partial' and marked non-core")
    p.add_argument("--q", type=int, default=transfer.PRIMARY_Q,
                   help="PCA capacity; fixed, never searched (default 16)")
    p.add_argument("--outer-folds", type=int, default=10)
    p.add_argument("--inner-folds", type=int, default=5)
    p.add_argument("--n-perm", type=int, default=5,
                   help="target permutations for the descriptive shuffle control")
    p.add_argument("--folds-seed", type=int, default=0,
                   help="only used when no manifest is given")
    p.add_argument("--rank-check", choices=("none", "outer", "all"), default="outer",
                   help="how much of the manifest's centered-rank audit to record")
    p.add_argument("--attempt", default=None,
                   help="this attempt's id, the name --supersedes would use. "
                        "Default: the collector derives it from the metrics "
                        "file's path under its --runs root")
    p.add_argument("--supersedes", action="append", default=None, metavar="ATTEMPT_ID",
                   help="an earlier attempt of the SAME (model, relation, repeat) "
                        "that this run replaces; repeatable.  This is the only "
                        "way a duplicated key is resolved -- the collector never "
                        "picks by timestamp -- and it is recorded in both the "
                        "metrics file and the results index")
    p.add_argument("--threads", type=int, default=None,
                   help="BLAS threads (see the module docstring)")
    p.add_argument("--backend", choices=("numpy", "torch"), default="numpy",
                   help="the array backend the probe arithmetic runs on "
                        "(linroute/backend.py).  numpy, the default, is what "
                        "every published number was computed with; torch runs "
                        "the same arithmetic in float64, batched over the "
                        "layer axis, on --device.  The choice is recorded in "
                        "metrics.json and in the row as compute_backend")
    p.add_argument("--device", default="cpu",
                   help="where the torch backend computes: cpu or cuda:N "
                        "(default cpu).  Ignored by --backend numpy, which "
                        "only has the host")
    return p.parse_args(argv)

def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False,
                   default=_default) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)

def _default(obj):
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")

def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                default=_default) + "\n")

def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    commit = git_commit()
    cfg = ProbeConfig(
        q_primary=a.q, outer_folds=a.outer_folds, inner_folds=a.inner_folds,
        folds_seed=a.folds_seed, n_perm=a.n_perm,
        backend=a.backend, device="cpu" if a.backend == "numpy" else a.device,
    )

    try:
        inputs = transfer.load_run_inputs(
            a.run_dir, a.arm_dir, wild_cache=a.wild_cache,
            manifest=a.manifest, final_epoch=a.final_epoch,
            allow_intermediate=a.allow_intermediate,
        )
        metrics = transfer.measure_transfer(
            inputs, model_name=a.model_name, relation=a.relation,
            repeat=a.repeat, cfg=cfg, q=a.q, n_perm=a.n_perm,
            rank_check=a.rank_check,
        )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        print(reason, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

        failed_rule = None
        if a.manifest:
            try:
                failed_rule = admission.holdout_rule_of(
                    json.loads(Path(a.manifest).read_text(encoding="utf-8")))
            except Exception:
                failed_rule = None
        row = transfer.failed_row(
            model_name=a.model_name, relation=a.relation, repeat=a.repeat,
            git_commit=commit, error=reason, holdout_rule=failed_rule,
        )
        payload = {"schema": "linroute/transfer-metrics/2", "status": "failed",
                   "error": reason, **row}
        if a.attempt:
            payload["attempt_id"] = a.attempt
        if a.supersedes:
            payload["supersedes"] = list(a.supersedes)
        _write_json(Path(a.out), payload)
        if a.rows_out:
            _append_row(Path(a.rows_out), row)
        return 1

    metrics["git_commit"] = commit
    if a.attempt:
        metrics["attempt_id"] = a.attempt
    if a.supersedes:
        metrics["supersedes"] = list(a.supersedes)
    _write_json(Path(a.out), metrics)
    row = transfer.flat_row(metrics, git_commit=commit)
    if a.rows_out:
        _append_row(Path(a.rows_out), row)

    primary = metrics["primary_probe"]
    first = metrics.get("primary_probe_first") or {}
    beh = metrics["behaviour"]
    print(
        f"[{metrics['status']}{'' if metrics['core'] else ' / non-core'}] "
        f"{a.model_name} {a.relation} r{a.repeat}: "
        f"primary gain {primary['gain']:.4f} "
        f"(acc {primary['acc']:.4f} - base {primary['base_acc']:.4f}), "
        f"heldout CAND-ACC (class-macro) {beh['heldout_cand_acc']:.4f} "
        f"[micro {beh['heldout_cand_acc_micro']:.4f}], "
        f"chance {beh['chance']:.4f}, N={metrics['n_train']}, "
        f"|H|={metrics['n_heldout']}, |E|={metrics['n_classes']}, "
        f"protocol {metrics.get('protocol')}, "
        f"on {metrics.get('compute_backend')}/{metrics.get('compute_device')}"
        f"/{metrics.get('compute_dtype')} -> {a.out}"
    )

    if first.get("available"):
        print(f"  secondary, subject FIRST token: gain {first['gain']:.4f} "
              f"(acc {first['acc']:.4f} - base {first['base_acc']:.4f})")
    else:
        print(f"  secondary first-token probe unavailable: "
              f"{first.get('reason')}")

    if not metrics["core"] and metrics.get("status_reason"):
        print(f"  non-core: {metrics['status_reason']}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
