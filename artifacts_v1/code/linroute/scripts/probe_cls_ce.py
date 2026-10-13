#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import secrets
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from linroute.cache import read_cache, sha256_file
from linroute import ce_gate
from linroute.cls_ce import (
    CEConfig,
    FIXED_LAMBDA,
    SCHEMA,
    lin_cls_ce_nested,
)
from linroute.probes import ProbeConfig, lin_cls_nested_batch
from linroute.transfer import CANONICAL_TEMPLATE, manifest_fold_plan

ARTIFACT_SCHEMA = "linroute/lin-cls-ce-artifact/3"
PROGRESS_SCHEMA = "linroute/lin-cls-ce-progress/3"

PROTOCOL_V3 = "v3-two-solver"
PROTOCOL_V4 = "v4-one-solver"
ARTIFACT_SCHEMA_V4 = ce_gate.ARTIFACT_SCHEMA_V4
PROGRESS_SCHEMA_V4 = "linroute/lin-cls-ce-progress/4"
SHUFFLE_SCHEMA = "linroute/lin-cls-ce-fold-feasible-permutations/3"
RIDGE_COMPARISON_SCHEMA = "linroute/lin-cls-ce-ridge-comparison/3"
SELECTION_PROTOCOL = "q16_fixed_lambda_1e-2_layer_only"
RETIRED_SURFACE_CUE_RELATIONS = frozenset({
    "word_first_letter",
    "word_last_letter",
    "engineered_town_first_letter",
    "engineered_town_name_length",
})
SHUFFLE_MAX_DRAWS = 100_000
SHUFFLE_NULL = "full-label permutation conditional on frozen-fold feasibility"

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wild-cache", required=True,
                   help="schema-2 pretrained PRE-REL cache")
    p.add_argument("--manifest", required=True,
                   help="linroute/admission/2 split manifest")
    p.add_argument("--out", required=True,
                   help="create-only atomic JSON artifact")
    p.add_argument("--device", default="cpu",
                   help="torch device, for example cpu or cuda:0")
    p.add_argument("--n-perm", type=int, default=5)
    p.add_argument("--perm-seed", type=int, default=0)
    p.add_argument("--ridge-reproduction", choices=("numpy", "torch"),
                   default="torch",
                   help="also run historical ridge CLS on identical T/folds")
    p.add_argument("--protocol", choices=(PROTOCOL_V3, PROTOCOL_V4),
                   default=PROTOCOL_V3,
                   help="v3-two-solver: /3, SciPy oracle, runner gate G_old; "
                        "v4-one-solver: /4, no oracle, terminal gate G_new")
    p.add_argument("--oracle-mode", choices=("selected-outer", "none"),
                   default=None,
                   help="implied by --protocol (v3: selected-outer, v4: none); "
                        "passing the other value is an error")
    p.add_argument("--allow-dirty", action="store_true",
                   help="development only; production refuses a dirty checkout")
    scope = p.add_mutually_exclusive_group(required=True)
    scope.add_argument("--development", action="store_true",
                       help="label an explicitly non-production pilot artifact")
    scope.add_argument("--production", action="store_true",
                       help="label a clean locked production artifact")
    return p.parse_args(argv)

def _validate_runner_args(a: argparse.Namespace) -> None:
    if a.n_perm != 5:
        raise ValueError("the frozen sensitivity requires exactly --n-perm 5")
    if (
        a.perm_seed < 0
        or a.perm_seed + 1000 + SHUFFLE_MAX_DRAWS - 1
        > np.iinfo(np.uint32).max
    ):
        raise ValueError("--perm-seed cannot address the frozen uint32 draw streams")
    if a.allow_dirty and not a.development:
        raise ValueError("--allow-dirty requires --development")
    implied = "selected-outer" if a.protocol == PROTOCOL_V3 else "none"
    if a.oracle_mode is None:
        a.oracle_mode = implied
    if a.oracle_mode != implied:
        raise ValueError(
            f"--protocol {a.protocol} runs --oracle-mode {implied}, "
            f"not {a.oracle_mode}"
        )

def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def _canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

def _hash(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()

def _git_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        check=True, text=True, capture_output=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        check=True, text=True, capture_output=True,
    ).stdout.strip())
    return commit, dirty

def _source_hashes(protocol: str = PROTOCOL_V3) -> dict[str, str]:
    paths = (
        REPO / "linroute" / "cls_ce.py",
        REPO / "linroute" / "probes.py",
        REPO / "linroute" / "transfer.py",
        Path(__file__).resolve(),
    )
    if protocol == PROTOCOL_V4:
        paths += (REPO / "linroute" / "ce_gate.py",)
    return {str(path.relative_to(REPO)): sha256_file(path) for path in paths}

def _fold_json(folds, inner) -> dict:
    return {
        "outer": [[int(x) for x in fold] for fold in folds],
        "inner": [
            [[int(x) for x in fold] for fold in group] for group in inner
        ],
    }

def _complement(n: int, heldout) -> np.ndarray:
    mask = np.ones(n, dtype=bool)
    mask[np.asarray(heldout, dtype=np.int64)] = False
    return np.flatnonzero(mask)

def _missing_classes(labels: np.ndarray, rows, n_classes: int) -> list[int]:
    present = set(labels[np.asarray(rows, dtype=np.int64)].tolist())
    return [class_id for class_id in range(n_classes) if class_id not in present]

def _shuffle_rejection_reasons(
    labels: np.ndarray,
    n_classes: int,
    folds,
    inner_folds,
) -> list[dict]:
    reasons = []
    n = len(labels)
    for outer_i, te in enumerate(folds):
        tr = _complement(n, te)
        missing = _missing_classes(labels, tr, n_classes)
        if missing:
            reasons.append({
                "outer": int(outer_i),
                "inner": None,
                "missing_classes": missing,
            })
        for inner_i, te_local in enumerate(inner_folds[outer_i]):
            inner_eval = np.asarray(te_local, dtype=np.int64)
            if not np.isin(inner_eval, tr).all():
                raise ValueError(
                    "an inner validation fold lies outside its outer fitting set"
                )
            global_fit = tr[~np.isin(tr, inner_eval)]
            missing = _missing_classes(labels, global_fit, n_classes)
            if missing:
                reasons.append({
                    "outer": int(outer_i),
                    "inner": int(inner_i),
                    "missing_classes": missing,
                })
    return reasons

def _conditioned_permutations(
    labels,
    n_classes: int,
    folds,
    inner_folds,
    *,
    seed: int,
    n_perm: int = 5,
    max_draws: int = SHUFFLE_MAX_DRAWS,
) -> tuple[list[np.ndarray], dict]:
    labels = np.asarray(labels, dtype=np.int64)
    if n_perm <= 0 or max_draws <= 0:
        raise ValueError("n_perm and max_draws must be positive")
    max_seed = int(seed) + 1000 + max_draws - 1
    if seed < 0 or max_seed > np.iinfo(np.uint32).max:
        raise ValueError("conditioned-shuffle stream seeds must fit uint32")

    accepted = []
    draws = []
    for draw_index in range(max_draws):
        stream_seed = int(seed) + 1000 + draw_index
        permutation = np.random.RandomState(stream_seed).permutation(len(labels))
        shuffled = labels[permutation]
        reasons = _shuffle_rejection_reasons(
            shuffled, n_classes, folds, inner_folds
        )
        is_accepted = not reasons
        record = {
            "draw_index": int(draw_index),
            "stream_seed": int(stream_seed),
            "accepted": bool(is_accepted),
            "permutation_sha256": _hash(permutation.tolist()),
            "rejection_reasons": reasons,
        }
        if is_accepted:
            record["accepted_index"] = len(accepted)
            record["permutation"] = permutation.tolist()
            accepted.append(permutation)
        draws.append(record)
        if len(accepted) == n_perm:
            break
    if len(accepted) != n_perm:
        raise ValueError(
            f"found only {len(accepted)} fold-feasible permutations after "
            f"the prespecified {max_draws} draws"
        )
    accepted_indices = [
        int(record["draw_index"]) for record in draws if record["accepted"]
    ]
    rejected_indices = [
        int(record["draw_index"]) for record in draws if not record["accepted"]
    ]
    provenance = {
        "schema": SHUFFLE_SCHEMA,
        "null": SHUFFLE_NULL,
        "generator": "numpy.random.RandomState(seed + 1000 + draw_index).permutation(N)",
        "base_seed": int(seed),
        "stream_seed_offset": 1000,
        "requested_accepted_draws": int(n_perm),
        "max_draws": int(max_draws),
        "attempted_draw_count": len(draws),
        "accepted_draw_indices": accepted_indices,
        "rejected_draw_indices": rejected_indices,
        "draws": draws,
    }
    return accepted, provenance

def _validate_input_contract(cache: dict, manifest: dict) -> None:
    meta = cache.get("meta", {})
    if meta.get("relation") in RETIRED_SURFACE_CUE_RELATIONS:
        raise ValueError(
            f"relation {meta.get('relation')!r} is a retired spelling "
            "surface-cue relation outside LIN-CLS-CE /3 scope"
        )
    if manifest.get("relation") != meta.get("relation"):
        raise ValueError(
            "manifest/cache relation mismatch: "
            f"{manifest.get('relation')!r} != {meta.get('relation')!r}"
        )
    if manifest.get("tokenizer") != meta.get("tokenizer"):
        raise ValueError(
            "manifest/cache tokenizer mismatch: "
            f"{manifest.get('tokenizer')!r} != {meta.get('tokenizer')!r}"
        )
    if manifest.get("admission", {}).get("admitted") is not True:
        raise ValueError("manifest admission.admitted is not true")
    parameters = manifest.get("parameters", {})
    expected = {"q": 16, "outer": 10, "inner": 5}
    for key, value in expected.items():
        if parameters.get(key) != value:
            raise ValueError(
                f"manifest parameter {key} is {parameters.get(key)!r}, "
                f"not frozen value {value}"
            )
    outer = manifest.get("folds", {}).get("outer")
    if not isinstance(outer, list) or len(outer) != 10:
        raise ValueError("manifest must contain exactly 10 outer folds")
    for index, fold in enumerate(outer):
        inner = fold.get("inner") if isinstance(fold, dict) else None
        if not isinstance(inner, list) or len(inner) != 5:
            raise ValueError(
                f"manifest outer fold {index} must contain exactly 5 inner folds"
            )

def _dense_pre_rel(cache: dict, manifest: dict):
    cache_subjects = list(cache["subjects"])
    pos = {name: i for i, name in enumerate(cache_subjects)}
    if len(pos) != len(cache_subjects):
        raise ValueError("wild cache has repeated subject strings")
    train = list(manifest["train"]["subjects"])
    subjects = [row["subject"] for row in train]
    missing = [name for name in subjects if name not in pos]
    if missing:
        raise ValueError(
            f"{len(missing)} manifest T subjects are absent from the cache, "
            f"for example {missing[:3]}"
        )
    rows = np.asarray([pos[name] for name in subjects], dtype=np.int64)
    raw = np.asarray(cache["cand_index"].cpu(), dtype=np.int64)[rows]
    candidate_tokens = np.asarray(cache["cand_first_token"].cpu(), dtype=np.int64)
    cache_effective = candidate_tokens[raw]
    manifest_effective = np.asarray(
        [int(row["effective_class"]) for row in train], dtype=np.int64
    )
    mismatch = np.flatnonzero(cache_effective != manifest_effective)
    if len(mismatch):
        examples = [
            {
                "subject": subjects[int(i)],
                "manifest_effective_class": int(manifest_effective[i]),
                "cache_candidate_index": int(raw[i]),
                "cache_effective_class": int(cache_effective[i]),
            }
            for i in mismatch[:3]
        ]
        raise ValueError(
            f"manifest and wild cache disagree on {len(mismatch)} T effective "
            f"classes, for example {examples}"
        )
    present, dense = np.unique(raw, return_inverse=True)
    n_classes = len(present)
    retained = [x for x in manifest["classes"] if x.get("retained")]
    retained_axis = sorted(int(x["effective_class"]) for x in retained)
    present_effective = sorted(int(x) for x in np.unique(cache_effective))
    if present_effective != retained_axis:
        raise ValueError(
            "cache T effective-class axis does not equal the manifest retained "
            f"axis: cache={present_effective}, manifest={retained_axis}"
        )
    if n_classes != len(retained_axis):
        raise ValueError(
            f"cache T has {n_classes} candidate classes but manifest has "
            f"{len(retained_axis)} retained effective classes"
        )
    if CANONICAL_TEMPLATE >= len(cache["rel_templates"]):
        raise ValueError("wild cache has no canonical PRE-REL template")
    states = np.asarray(
        cache["rel_subject_last"][rows, CANONICAL_TEMPLATE].cpu(),
        dtype=np.float64,
    )
    class_axis = [
        {
            "dense_label": i,
            "cache_candidate_index": int(raw_index),
            "candidate": cache["candidates"][int(raw_index)],
            "candidate_token": int(candidate_tokens[int(raw_index)]),
            "manifest_effective_class": int(candidate_tokens[int(raw_index)]),
        }
        for i, raw_index in enumerate(present)
    ]
    return subjects, rows, states, dense.astype(np.int64), class_axis

def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value

def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def _temporary_peer(path: Path) -> Path:
    return path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )

def _write_fsynced(path: Path, data: str) -> None:
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())

def _link_create_only(source: Path, destination: Path) -> None:
    os.link(source, destination)
    _fsync_directory(destination.parent)

def _atomic_write(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    _fsync_directory(path.parent)
    tmp = _temporary_peer(path)
    data = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ) + "\n"
    digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    side_tmp = _temporary_peer(sidecar)
    try:
        _write_fsynced(tmp, data)
        _write_fsynced(side_tmp, f"{digest}  {path.name}\n")
        _link_create_only(tmp, path)
        _link_create_only(side_tmp, sidecar)
    finally:
        for peer in (tmp, side_tmp):
            try:
                peer.unlink()
            except FileNotFoundError:
                pass
        _fsync_directory(path.parent)
    return digest

def _verified_artifact(
    path: Path, dedup_hash: str, schema: str = ARTIFACT_SCHEMA,
) -> tuple[dict, str]:
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"artifact {path} has no SHA-256 sidecar")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != path.name:
        raise ValueError(f"malformed SHA-256 sidecar {sidecar}")
    actual = sha256_file(path)
    if fields[0] != actual:
        raise ValueError(
            f"artifact digest mismatch for {path}: sidecar={fields[0]}, actual={actual}"
        )
    payload = _json(path)
    if payload.get("_complete") is not True:
        raise ValueError(f"artifact {path} lacks its complete marker")
    if payload.get("schema") != schema:
        raise ValueError(
            f"artifact {path} is not LIN-CLS-CE artifact /{schema.rsplit('/', 1)[1]}"
        )
    if payload.get("measurement_schema") != SCHEMA:
        raise ValueError(f"artifact {path} is not LIN-CLS-CE measurement /3")
    if payload.get("dedup_hash") != dedup_hash:
        raise ValueError(
            f"artifact {path} is for {payload.get('dedup_hash')}, not {dedup_hash}"
        )
    return payload, actual

def _library_provenance(device_name: str) -> dict:
    try:
        scipy_version = importlib.metadata.version("scipy")
    except importlib.metadata.PackageNotFoundError:
        scipy_version = None
    device = torch.device(device_name)
    cuda = None
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"requested {device_name}, but CUDA is unavailable")
        index = torch.cuda.current_device() if device.index is None else device.index
        properties = torch.cuda.get_device_properties(index)
        cuda = {
            "runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "index": int(index),
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
        }
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "scipy": scipy_version,
        "cuda": cuda,
    }

def _synchronize(device_name: str) -> None:
    if torch.device(device_name).type == "cuda":
        torch.cuda.synchronize(torch.device(device_name))

def _peak_host_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)

def _peak_device_memory(device_name: str) -> dict | None:
    device = torch.device(device_name)
    if device.type != "cuda":
        return None
    return {
        "allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }

def _atomic_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _fsync_directory(path.parent)
    tmp = _temporary_peer(path)
    data = json.dumps(
        _jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    with open(tmp, "x", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_directory(path.parent)

def _load_resume_state(
    path: Path, dedup_hash: str, schema: str = PROGRESS_SCHEMA,
):
    if not path.exists():
        return None
    progress = _json(path)
    if progress.get("schema") != schema:
        raise ValueError(
            f"progress file {path} is not LIN-CLS-CE progress /{schema.rsplit('/', 1)[1]}"
        )
    if progress.get("dedup_hash") != dedup_hash:
        raise ValueError(f"progress file {path} belongs to another dedup hash")
    return progress.get("resume_state")

def _completion_gates(
    measured: dict,
    ridge: dict | None,
    *,
    n_perm: int,
    outer_folds: int,
    ridge_mode: str,
    oracle_mode: str,
) -> dict:
    requested_results_complete = bool(
        len(measured["results"]) == 1 + n_perm
        and all(result.get("available") for result in measured["results"])
    )
    expected_oracles = (
        (1 + n_perm) * outer_folds
        if oracle_mode == "selected-outer" else 0
    )
    oracle_complete = bool(
        oracle_mode == "none"
        or (
            len(measured["oracle_checks"]) == expected_oracles
            and all(check.get("passed") for check in measured["oracle_checks"])
        )
    )
    ridge_complete = bool(
        ridge_mode == "none"
        or (ridge is not None and ridge.get("available") is True)
    )
    candidate_grid_complete = bool(
        measured.get("candidate_grid", {}).get("complete") is True
    )
    return {
        "requested_results_complete": requested_results_complete,
        "requested_result_count": 1 + n_perm,
        "oracle_complete": oracle_complete,
        "expected_oracle_checks": expected_oracles,
        "ridge_complete": ridge_complete,
        "candidate_grid_complete": candidate_grid_complete,
        "candidate_grid_counts": measured.get("candidate_grid"),
        "complete": bool(
            requested_results_complete
            and oracle_complete
            and ridge_complete
            and candidate_grid_complete
        ),
    }

def _terminal_completion(
    measured: dict,
    ridge: dict | None,
    *,
    protocol: str,
    n_perm: int,
    outer_folds: int,
    ridge_mode: str,
    oracle_mode: str,
) -> dict:
    completion = _completion_gates(
        measured, ridge, n_perm=n_perm, outer_folds=outer_folds,
        ridge_mode=ridge_mode, oracle_mode=oracle_mode,
    )
    if protocol == PROTOCOL_V3:
        return completion
    if oracle_mode != "none":
        raise ValueError("v4-one-solver never runs the SciPy oracle")
    hyg = ce_gate.hygiene(measured["fit_diagnostics"])
    completion["hygiene_ok"] = hyg["hygiene_ok"]
    completion["hygiene"] = hyg
    completion["cell_gate"] = ce_gate.GATE_VERSION
    completion["complete"] = bool(completion["complete"] and hyg["hygiene_ok"])
    return completion

def _ridge_reproduction(
    states,
    label_sets,
    n_classes: int,
    folds,
    inner_folds,
    *,
    backend: str,
    device: str,
    accepted_draw_indices: list[int],
) -> dict:
    if len(label_sets) != 1 + len(accepted_draw_indices):
        raise ValueError(
            "ridge label sets do not match conditioned-shuffle draw indices"
        )
    ridge_cfg = ProbeConfig(
        q_primary=16,
        outer_folds=10,
        inner_folds=5,
        folds_seed=0,
        backend=backend,
        device="cpu" if backend == "numpy" else device,
        dtype="float64",
    )
    ridge_batch = lin_cls_nested_batch(
        states,
        label_sets,
        n_classes,
        ridge_cfg,
        qs=(16,),
        folds=folds,
        inner_folds=inner_folds,
    )
    ridge_results = [per_q[16] for per_q in ridge_batch]
    return {
        "schema": RIDGE_COMPARISON_SCHEMA,
        "available": bool(
            len(ridge_results) == len(label_sets)
            and all(result.get("available") is True for result in ridge_results)
        ),
        "null": SHUFFLE_NULL,
        "accepted_draw_indices": [int(x) for x in accepted_draw_indices],
        "label_sets": [
            {
                "label_set": int(index),
                "kind": "real" if index == 0 else "conditioned_shuffle",
                "draw_index": None if index == 0 else int(
                    accepted_draw_indices[index - 1]
                ),
                "result": result,
            }
            for index, result in enumerate(ridge_results)
        ],
    }

def main(argv: list[str] | None = None) -> int:
    prepare_started = time.perf_counter()
    a = parse_args(argv)
    try:
        _validate_runner_args(a)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    cache_path = Path(a.wild_cache).resolve()
    manifest_path = Path(a.manifest).resolve()
    out_path = Path(a.out).resolve()
    commit, dirty = _git_state()
    if dirty and not a.allow_dirty:
        raise SystemExit(
            "refusing a dirty checkout: commit the frozen implementation first "
            "or use --allow-dirty for a labelled development artifact"
        )

    cache = read_cache(cache_path)
    manifest = _json(manifest_path)
    _validate_input_contract(cache, manifest)
    subjects, cache_rows, states, labels, class_axis = _dense_pre_rel(cache, manifest)
    folds, inner = manifest_fold_plan(manifest, subjects)
    try:
        perms, shuffle_provenance = _conditioned_permutations(
            labels,
            len(class_axis),
            folds,
            inner,
            seed=a.perm_seed,
            n_perm=a.n_perm,
            max_draws=SHUFFLE_MAX_DRAWS,
        )
    except ValueError as exc:
        raise SystemExit(f"conditioned shuffle generation failed: {exc}") from exc
    label_sets = [labels] + [labels[p] for p in perms]
    cfg = CEConfig(q=16, lambdas=(FIXED_LAMBDA,), device=a.device)

    cache_sha = sha256_file(cache_path)
    manifest_sha = sha256_file(manifest_path)
    folds_json = _fold_json(folds, inner)
    source_hashes = _source_hashes(a.protocol)
    v4 = a.protocol == PROTOCOL_V4
    artifact_schema = ARTIFACT_SCHEMA_V4 if v4 else ARTIFACT_SCHEMA
    progress_schema = PROGRESS_SCHEMA_V4 if v4 else PROGRESS_SCHEMA
    libraries = _library_provenance(a.device)
    if a.oracle_mode == "selected-outer" and libraries["scipy"] is None:
        raise SystemExit(
            "--oracle-mode selected-outer requires SciPy; absence is not a "
            "passed pilot cross-check"
        )
    identity = {
        "ordered_subject_ids": [row["subject_id"] for row in manifest["train"]["subjects"]],
        "ordered_subjects": subjects,
        "cache_rows": cache_rows.tolist(),
        "dense_labels": labels.tolist(),
        "class_axis": class_axis,
        "folds": folds_json,
        "fold_digest": _hash(folds_json),
        "permutation_seed": int(a.perm_seed),
        "permutations": [p.tolist() for p in perms],
        "shuffle_provenance": shuffle_provenance,
        "shuffle_null": SHUFFLE_NULL,
        "canonical_template_index": CANONICAL_TEMPLATE,
        "canonical_template": cache["rel_templates"][CANONICAL_TEMPLATE],
        "cache_sha256": cache_sha,
        "manifest_sha256": manifest_sha,
        "cache_schema": cache["meta"]["schema_version"],
        "token_rule": cache["meta"]["token_rule"],
        "model": cache["meta"]["model"],
        "model_revision": cache["meta"].get("model_revision"),
        "tokenizer": cache["meta"].get("tokenizer"),
        "relation": cache["meta"]["relation"],
        "cache_git_commit": cache["meta"].get("git_commit"),
        "manifest_schema": manifest.get("schema"),
        "manifest_parameters": manifest.get("parameters"),
        "candidate_layers": list(range(states.shape[1])) if cfg.layers is None
                            else list(cfg.layers),
        "state_shape": [int(x) for x in states.shape],
        "config": cfg.as_dict(),
        "schemas": {
            "measurement": SCHEMA,
            "artifact": artifact_schema,
            "progress": progress_schema,
            "shuffle": SHUFFLE_SCHEMA,
            "ridge_comparison": RIDGE_COMPARISON_SCHEMA,
        },
        "runner_modes": {
            "selection_protocol": SELECTION_PROTOCOL,
            "selection_dimension": "layer_only",
            "fixed_lambda": FIXED_LAMBDA,
            "n_permutations": int(a.n_perm),
            "ridge_reproduction": a.ridge_reproduction,
            "oracle_mode": a.oracle_mode,
            "artifact_scope": "development" if a.development else "production",

            **({"protocol": PROTOCOL_V4, "cell_gate": ce_gate.GATE_VERSION}
               if v4 else {}),
        },
        "backend_semantics": {
            "backend": cfg.backend, "device": cfg.device, "dtype": cfg.dtype,
            "cross_device_semantic_equivalence_allowed": False,
        },
        "code_commit": commit,
        "source_hashes": source_hashes,
        "libraries": libraries,
    }
    dedup_hash = _hash(identity)

    if out_path.exists():
        try:
            existing, _digest = _verified_artifact(
                out_path, dedup_hash, artifact_schema
            )
        except ValueError as exc:
            raise SystemExit(f"refusing unverified existing artifact: {exc}") from exc
        print(
            f"[verified skip: {existing.get('status')}] {dedup_hash} -> {out_path}"
        )
        return 0 if existing.get("status") == "complete" else 2
    if out_path.with_name(out_path.name + ".sha256").exists():
        raise SystemExit(
            f"refusing to publish {out_path}: its sidecar already exists"
        )

    progress_path = out_path.with_name(out_path.name + ".progress.json")
    try:
        resume_state = _load_resume_state(
            progress_path, dedup_hash, progress_schema
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    def save_progress(state: dict) -> None:
        _atomic_progress(progress_path, {
            "schema": progress_schema,
            "dedup_hash": dedup_hash,
            "resume_state": state,
        })

    prepare_seconds = time.perf_counter() - prepare_started
    if torch.device(a.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(a.device))
    _synchronize(a.device)
    ce_started = time.perf_counter()
    measured = lin_cls_ce_nested(
        states, label_sets, len(class_axis), cfg, folds=folds, inner_folds=inner,
        resume_state=resume_state, progress_callback=save_progress,
        scipy_oracle=a.oracle_mode == "selected-outer",
    )
    _synchronize(a.device)
    ce_seconds = time.perf_counter() - ce_started

    measured.pop("resume_state", None)
    ridge = None
    ridge_seconds = 0.0
    if a.ridge_reproduction != "none":
        _synchronize(a.device)
        ridge_started = time.perf_counter()
        ridge = _ridge_reproduction(
            states,
            label_sets,
            len(class_axis),
            folds,
            inner,
            backend=a.ridge_reproduction,
            device=a.device,
            accepted_draw_indices=shuffle_provenance["accepted_draw_indices"],
        )
        _synchronize(a.device)
        ridge_seconds = time.perf_counter() - ridge_started

    completion = _terminal_completion(
        measured, ridge, protocol=a.protocol, n_perm=a.n_perm,
        outer_folds=cfg.outer_folds, ridge_mode=a.ridge_reproduction,
        oracle_mode=a.oracle_mode,
    )

    payload = {
        "_complete": True,
        "schema": artifact_schema,
        "measurement_schema": SCHEMA,
        "status": "complete" if completion["complete"] else "failed",
        "development": bool(a.development),
        "selection_protocol": SELECTION_PROTOCOL,
        "dedup_hash": dedup_hash,
        "identity": identity,
        "shuffle_provenance": shuffle_provenance,
        "ce": measured,
        "ridge_reproduction": ridge,
        "completion_gates": completion,
        "runtime": {
            "phase_seconds": {
                "prepare": prepare_seconds,
                "ce_synchronized": ce_seconds,
                "ridge_synchronized": ridge_seconds,
            },
            "peak_host_rss_bytes": _peak_host_bytes(),
            "peak_device_memory": _peak_device_memory(a.device),
        },
        "prohibitions": {
            "heldout_subjects_used_in_fit": False,
            "new_rel_gen_loaded": False,
            "transfer_outcomes_loaded": False,
        },
    }
    payload = _jsonable(payload)
    if v4:
        verdict = ce_gate.gate_artifact(payload)
        if (verdict["status"] == ce_gate.PASSED) != (payload["status"] == "complete"):
            raise SystemExit("v4 terminal status disagrees with linroute.ce_gate")
    digest = _atomic_write(out_path, payload)
    real = payload["ce"]["results"][0]
    print(
        f"[{payload['status']}] {cache['meta']['model']} / "
        f"{cache['meta']['relation']}: acc={real.get('acc')} "
        f"gain={real.get('gain')} hash={dedup_hash} sha256={digest} -> {out_path}"
    )
    return 0 if payload["status"] == "complete" else 2

if __name__ == "__main__":
    raise SystemExit(main())
