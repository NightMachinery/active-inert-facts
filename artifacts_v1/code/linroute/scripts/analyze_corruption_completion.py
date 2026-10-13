#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linroute.corruption_analysis import (
    conditional_class_label_alignment,
    e3_cell,
    heuristic_gain_normalization,
    load_jsonl,
    recover_primary_rule_rescore,
    validate_e1_endpoint_identity,
    validate_p4b_identity,
)
from linroute.transfer import load_run_inputs
from linroute import relation_sets as registry

MODEL_NAMES = {
    "allenai__Olmo-3-1025-7B": "OLMo-3-1025-7B",
    "Qwen__Qwen3.5-9B": "Qwen3.5-9B",
}

CORRUPTION_RELATIONS = frozenset(registry.named("corruption"))
SURFACE_CUE_RELATIONS = frozenset(registry.named("surface_cue"))
N_RELATIONS = len(CORRUPTION_RELATIONS)
N_SUBSTANTIVE = len(CORRUPTION_RELATIONS - SURFACE_CUE_RELATIONS)

N_RHO = 4
N_ENDPOINTS = len(MODEL_NAMES) * N_RELATIONS
N_CELLS_PER_MODEL = N_RHO * N_RELATIONS
N_CELLS = len(MODEL_NAMES) * N_CELLS_PER_MODEL

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, path)

def prediction_files(root: Path) -> list[Path]:
    return sorted(root.glob("*/runs/*/*/rho*/preds_ep020.jsonl"))

def p4b_run_dirs(root: Path) -> list[Path]:
    return sorted({path.parent for path in prediction_files(root)})

def e3(args: argparse.Namespace) -> int:
    archive = args.archive_root.resolve()
    files = prediction_files(args.prediction_root.resolve())
    if len(files) != N_CELLS:
        raise SystemExit(f"expected {N_CELLS} preds_ep020.jsonl files, found {len(files)}")

    rows = []
    keys = set()
    for preds_path in files:
        rel = preds_path.relative_to(args.prediction_root.resolve())
        host, literal_runs, model_slug, relation, rho_tag, filename = rel.parts
        if literal_runs != "runs" or filename != "preds_ep020.jsonl":
            raise AssertionError(rel)
        if model_slug not in MODEL_NAMES:
            raise SystemExit(f"unexpected model directory {model_slug!r}")
        arm_path = archive / "arms-corrupt" / model_slug / relation / rho_tag / "arm.json"
        if not arm_path.is_file():
            raise SystemExit(f"missing arm {arm_path}")
        key = (model_slug, relation, rho_tag)
        if key in keys:
            raise SystemExit(f"duplicate cell {key}")
        keys.add(key)

        arm = json.loads(arm_path.read_text(encoding="utf-8"))
        row = e3_cell(arm, load_jsonl(preds_path))
        row.update({
            "schema": "linroute/corruption-e3-cell/1",
            "model": MODEL_NAMES[model_slug],
            "model_slug": model_slug,
            "relation": relation,
            "rho_tag": rho_tag,
            "prediction_sha256": sha256(preds_path),
            "arm_sha256": sha256(arm_path),
            "source_host": host,
        })
        rows.append(row)

    expected = {
        (model, relation, rho)
        for model in MODEL_NAMES
        for relation in {row["relation"] for row in rows}
        for rho in ("rho010", "rho025", "rho050", "rho100")
    }
    if keys != expected or {row["relation"] for row in rows} != CORRUPTION_RELATIONS:
        raise SystemExit(f"E3 source is not the complete 2 x {N_RELATIONS} x 4 frozen grid")

    rows.sort(key=lambda row: (row["model"], row["relation"], row["rho_pct"]))
    out = args.out_dir.resolve()
    atomic_jsonl(out / "e3-cells.jsonl", rows)
    summary = {
        "schema": "linroute/corruption-e3-summary/1",
        "n_cells": len(rows),
        "n_models": len({row["model"] for row in rows}),
        "n_relations": len({row["relation"] for row in rows}),
        "rho_pct": sorted({row["rho_pct"] for row in rows}),
        "n_changed_subjects": sum(row["n_changed_subjects"] for row in rows),
        "n_trials": sum(row["n_trials"] for row in rows),
        "all_category_sums_match": all(
            row["assigned_count"] + row["original_count"] + row["other_count"]
            == row["n_trials"] for row in rows
        ),
        "all_four_template": all(row["n_templates"] == 4 for row in rows),
        "denominator": "actually changed training subjects x four training templates",
    }
    atomic_json(out / "e3-summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0

def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks

def _correlation(x, y, *, spearman: bool = False) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if spearman:
        x, y = _rank(x), _rank(y)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def _seed_for(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")

def e1(args: argparse.Namespace) -> int:
    archive = args.archive_root.resolve()
    prediction_root = args.prediction_root.resolve()
    metrics_root = args.metrics_root.resolve()
    files = [
        path for path in prediction_files(prediction_root)
        if path.parent.name == "rho100"
    ]
    if len(files) != N_ENDPOINTS:
        raise SystemExit(f"expected {N_ENDPOINTS} rho100 preds_ep020.jsonl files, found {len(files)}")

    rows = []
    keys = set()
    for preds_path in files:
        rel = preds_path.relative_to(prediction_root)
        host, literal_runs, model_slug, relation, rho_tag, filename = rel.parts
        if literal_runs != "runs" or rho_tag != "rho100" or filename != "preds_ep020.jsonl":
            raise AssertionError(rel)
        key = (model_slug, relation)
        if key in keys:
            raise SystemExit(f"duplicate rho100 endpoint {key}")
        keys.add(key)

        arm_path = archive / "arms-corrupt" / model_slug / relation / rho_tag / "arm.json"
        metrics_path = metrics_root / model_slug / relation / "r0" / rho_tag / "metrics.json"
        if not arm_path.is_file() or not metrics_path.is_file():
            raise SystemExit(f"missing arm or metrics for {key}")
        arm = json.loads(arm_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        validate_e1_endpoint_identity(
            arm,
            metrics,
            model_name=MODEL_NAMES[model_slug],
            relation=relation,
        )
        seed = _seed_for("e1-conditional-class-label-alignment", 20260921, *key)
        result = conditional_class_label_alignment(
            arm,
            load_jsonl(preds_path),
            seed=seed,
            monte_carlo_draws=args.monte_carlo_draws,
        )

        behaviour = metrics.get("behaviour") or {}
        archived = float(behaviour["heldout_cand_acc"])
        archived_chance = float(behaviour["chance"])
        if not np.isclose(result["observed_class_macro"], archived, atol=1e-12, rtol=0):
            raise SystemExit(
                f"{key}: recovered endpoint {result['observed_class_macro']} "
                f"does not match archived {archived}"
            )
        if not np.isclose(result["chance"], archived_chance, atol=1e-15, rtol=0):
            raise SystemExit(
                f"{key}: recovered chance {result['chance']} "
                f"does not match archived {archived_chance}"
            )

        corruption = arm["corruption"]
        result.update({
            "schema": "linroute/corruption-e1-conditional-alignment/1",
            "model": MODEL_NAMES[model_slug],
            "model_slug": model_slug,
            "relation": relation,
            "rho_tag": rho_tag,
            "rho_pct": 100,
            "source_host": host,
            "surface_cue_relation": relation in SURFACE_CUE_RELATIONS,
            "realised_changed_fraction": float(
                corruption["realised_changed_fraction"]
            ),
            "n_changed_subjects": int(corruption["n_changed"]),
            "n_selected_subjects": int(corruption["n_selected"]),
            "residual_contingency": corruption["residual_contingency"],
            "archived_endpoint_parity": True,
            "prediction_sha256": sha256(preds_path),
            "arm_sha256": sha256(arm_path),
            "metrics_sha256": sha256(metrics_path),
            "preregistered_binomial_status": "unavailable-invalid-unit",
            "preregistered_binomial_reason": (
                "NEW-REL-GEN is class-macro over unequal heldout class sizes, "
                "not a binomial count of independent subject trials"
            ),
        })
        rows.append(result)

    if len(keys) != N_ENDPOINTS or {relation for _, relation in keys} != CORRUPTION_RELATIONS:
        raise SystemExit(f"E1 source is not the complete 2 x {N_RELATIONS} rho100 endpoint grid")
    rows.sort(key=lambda row: (row["model"], row["relation"]))

    model_summaries = []
    for model in sorted({row["model"] for row in rows}):
        primary = [
            row for row in rows
            if row["model"] == model and not row["surface_cue_relation"]
        ]
        if len(primary) != N_SUBSTANTIVE:
            raise SystemExit(f"{model}: expected {N_SUBSTANTIVE} substantive E1 endpoints")
        model_summaries.append({
            "model": model,
            "set": "ORIGINAL-SUBSTANTIVE-13",
            "n_relations": len(primary),
            "n_observed_above_chance": sum(
                row["observed_minus_chance"] > 0 for row in primary
            ),
            "n_exploratory_unadjusted_p_lt_0_05": sum(
                row["p_one_sided_above_chance"] < 0.05 for row in primary
            ),
            "relations_exploratory_unadjusted_p_lt_0_05": sorted(
                row["relation"] for row in primary
                if row["p_one_sided_above_chance"] < 0.05
            ),
            "multiplicity_correction": "none",
            "number_of_tests_in_family": N_SUBSTANTIVE,
        })

    out = args.out_dir.resolve()
    atomic_jsonl(out / "e1-conditional-alignment-cells.jsonl", rows)
    summary = {
        "schema": "linroute/corruption-e1-conditional-alignment-summary/1",
        "status": "exploratory",
        "n_cells": len(rows),
        "n_models": 2,
        "n_relations": N_RELATIONS,
        "rho_pct": 100,
        "preregistered_binomial_status": "unavailable-invalid-unit",
        "replacement": "exploratory-conditional-class-label-alignment-permutation",
        "causal_rule_reuse_test": False,
        "failure_to_reject_establishes_equivalence": False,
        "model_summaries": model_summaries,
        "monte_carlo_draws_requested": args.monte_carlo_draws,
    }
    atomic_json(out / "e1-conditional-alignment-summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0

def p4b(args: argparse.Namespace) -> int:
    archive = args.archive_root.resolve()
    source_root = args.source_root.resolve()
    metrics_root = args.metrics_root.resolve()
    run_dirs = p4b_run_dirs(source_root)
    if len(run_dirs) != N_CELLS:
        raise SystemExit(f"expected {N_CELLS} recovered run directories, found {len(run_dirs)}")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("require shard_count > 0 and 0 <= shard_index < shard_count")
    selected = [
        run_dir for index, run_dir in enumerate(run_dirs)
        if index % args.shard_count == args.shard_index
    ]

    rows = []
    for run_dir in selected:
        cell_started = time.perf_counter()
        rel = run_dir.relative_to(source_root)
        host, literal_runs, model_slug, relation, rho_tag = rel.parts
        if literal_runs != "runs" or model_slug not in MODEL_NAMES:
            raise SystemExit(f"unexpected recovered run path {rel}")
        try:
            rho_pct = int(rho_tag.removeprefix("rho"))
        except ValueError as exc:
            raise SystemExit(f"invalid rho tag {rho_tag!r}") from exc
        if rho_tag != f"rho{rho_pct:03d}" or rho_pct not in (10, 25, 50, 100):
            raise SystemExit(f"invalid rho tag {rho_tag!r}")

        arm_dir = archive / "arms-corrupt" / model_slug / relation / rho_tag
        arm_path = arm_dir / "arm.json"
        if not arm_path.is_file():
            raise SystemExit(f"missing arm {arm_path}")
        arm = json.loads(arm_path.read_text(encoding="utf-8"))
        manifest_path = Path((arm.get("manifest") or {}).get("path") or "")
        if not manifest_path.is_file():
            raise SystemExit(f"missing frozen manifest {manifest_path}")
        metrics_path = metrics_root / model_slug / relation / "r0" / rho_tag / "metrics.json"
        if not metrics_path.is_file():
            raise SystemExit(f"missing archived metrics {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            args.only_archived_backend is not None
            and metrics.get("compute_backend") != args.only_archived_backend
        ):
            continue

        inputs = load_run_inputs(
            run_dir,
            arm_dir,
            manifest=manifest_path,
            final_epoch=20,
        )
        provenance = validate_p4b_identity(
            inputs,
            metrics,
            model_name=MODEL_NAMES[model_slug],
            relation=relation,
            rho_pct=rho_pct,
        )
        result = recover_primary_rule_rescore(
            inputs, metrics, q=16, torch_device=args.torch_device
        )
        changed_fraction = float(result["validated_realised_changed_fraction"])
        normalised = heuristic_gain_normalization(
            result["assigned_label_as_run"]["gain"], changed_fraction
        )
        rows.append({
            "schema": "linroute/corruption-p4b-rule-rescore/1",
            "model": MODEL_NAMES[model_slug],
            "model_slug": model_slug,
            "relation": relation,
            "rho": rho_pct / 100.0,
            "rho_pct": rho_pct,
            "rho_tag": rho_tag,
            "repeat": 0,
            "source_host": host,
            "surface_cue_relation": relation in SURFACE_CUE_RELATIONS,
            "realised_changed_fraction": changed_fraction,
            "frozen_provenance": provenance,
            **result,
            "as_run_gain_normalization": normalised,
            "source_files_sha256": dict(inputs.files),
            "metrics_sha256": sha256(metrics_path),
            "recovery_compute": {
                **result["recovery_compute"],
                "wall_seconds": time.perf_counter() - cell_started,
            },
        })
        print(
            json.dumps({
                "model": MODEL_NAMES[model_slug],
                "relation": relation,
                "rho_pct": rho_pct,
                "parity": True,
                "rule_gain": result["original_rule_rescore"]["gain"],
            }, sort_keys=True),
            flush=True,
        )

    rows.sort(key=lambda row: (row["model"], row["relation"], row["rho_pct"]))
    atomic_jsonl(args.out.resolve(), rows)
    return 0

def validate_p4b_merged_rows(rows: list[dict]) -> dict:
    expected_models = set(MODEL_NAMES.values())
    expected_rho = {10, 25, 50, 100}
    if len(rows) != N_CELLS:
        raise SystemExit(f"P4b merge expected {N_CELLS} rows, found {len(rows)}")
    if any(int(row.get("repeat", -1)) != 0 for row in rows):
        raise SystemExit("P4b merge contains a nonzero repeat")

    model_counts = Counter(row.get("model") for row in rows)
    if set(model_counts) != expected_models or set(model_counts.values()) != {N_CELLS_PER_MODEL}:
        raise SystemExit(f"P4b merge requires {N_CELLS_PER_MODEL} cells per model: {model_counts}")
    rho_counts = Counter((row["model"], int(row["rho_pct"])) for row in rows)
    expected_model_rho = {(model, rho) for model in expected_models for rho in expected_rho}
    if set(rho_counts) != expected_model_rho or set(rho_counts.values()) != {N_RELATIONS}:
        raise SystemExit(
            f"P4b merge requires {N_RELATIONS} relations per model/rho: {rho_counts}"
        )

    relations_by_model = {
        model: {row["relation"] for row in rows if row["model"] == model}
        for model in expected_models
    }
    relation_sets = list(relations_by_model.values())
    if any(len(relations) != N_RELATIONS for relations in relation_sets) or relation_sets[0] != relation_sets[1]:
        raise SystemExit(
            f"P4b merge model relation sets disagree: {relations_by_model}"
        )
    expected_relations = relation_sets[0]
    relation_counts = Counter((row["model"], row["relation"]) for row in rows)
    expected_model_relation = {
        (model, relation)
        for model in expected_models
        for relation in expected_relations
    }
    if set(relation_counts) != expected_model_relation or set(relation_counts.values()) != {4}:
        raise SystemExit(
            f"P4b merge requires four rho cells per model/relation: {relation_counts}"
        )

    keys = {
        (row["model"], row["relation"], int(row["rho_pct"]), int(row["repeat"]))
        for row in rows
    }
    if len(keys) != N_CELLS:
        raise SystemExit(
            f"P4b shards are not the unique {N_CELLS}-cell grid: rows={len(rows)} "
            f"keys={len(keys)}"
        )

    expected_sha_fields = {
        "arm", "cache_ep000", "cache_ep020", "history", "manifest", "order",
        "preds_ep000", "preds_ep020",
    }
    digest_fields = (
        "metrics_sha256", "archived_fold_sha256",
        "prediction_candidate_indices_sha256",
        "baseline_candidate_indices_sha256", "clean_gold_token_vector_sha256",
        "assigned_gold_token_vector_sha256",
    )
    backend_counts = Counter()
    for row in rows:
        if not row["parity"]["passed"]:
            raise SystemExit("P4b shard contains a failed parity cell")
        if not row["same_prediction_vectors_rescored"]:
            raise SystemExit("P4b shard does not certify same-vector rescoring")
        if row["clean_labels_used_in_fit_or_selection"]:
            raise SystemExit("P4b shard says clean labels affected fit/selection")
        provenance = row.get("frozen_provenance") or {}
        expected_provenance = {
            "metrics_git_commit": "0861cf0cfe3cd7fbae622fbcae42abef11d7f4e8",
            "arm_git_commit": "030c85b4e0c05f7bef70371153d43769e85b0b1e",
            "protocol": "heldout-mushroom-v2-corrupt",
            "holdout_rule": "mushroom-v2",
            "source_hashes_match_archived_metrics": True,
        }
        if provenance != expected_provenance:
            raise SystemExit("P4b shard frozen provenance gate failed")
        source_hashes = row.get("source_files_sha256") or {}
        if set(source_hashes) != expected_sha_fields:
            raise SystemExit("P4b shard source hash fields are incomplete")
        if any(not _is_sha256(value) for value in source_hashes.values()):
            raise SystemExit("P4b shard has an invalid source SHA-256")
        if any(not _is_sha256(row.get(field)) for field in digest_fields):
            raise SystemExit("P4b shard has an invalid metrics/fold/vector SHA-256")

        archived_compute = row.get("archived_compute") or {}
        recovery_compute = row.get("recovery_compute") or {}
        backend = archived_compute.get("backend")
        if backend not in {"numpy", "torch"} or recovery_compute.get("backend") != backend:
            raise SystemExit("P4b shard backend provenance mismatch")
        if archived_compute.get("dtype") != "float64" or recovery_compute.get("dtype") != "float64":
            raise SystemExit("P4b shard dtype provenance mismatch")
        if backend == "numpy":
            if archived_compute.get("device") != "cpu" or recovery_compute.get("device") != "cpu":
                raise SystemExit("P4b numpy shard did not recover on CPU")
        elif not str(archived_compute.get("device", "")).startswith("cuda:") or not str(
            recovery_compute.get("device", "")
        ).startswith("cuda:"):
            raise SystemExit("P4b torch shard did not recover on CUDA")
        backend_counts[backend] += 1

        n = int(row["assigned_label_as_run"]["n"])
        n_changed = int(row["validated_n_changed_subjects"])
        realised = float(row["realised_changed_fraction"])
        if n <= 0 or not 0 <= n_changed <= n or not np.isclose(
            realised, n_changed / n, atol=1e-15, rtol=0
        ):
            raise SystemExit("P4b shard realised changed-label fraction gate failed")
        if not np.isclose(
            realised, float(row["validated_realised_changed_fraction"]),
            atol=0, rtol=0,
        ):
            raise SystemExit("P4b shard validated changed-label fraction mismatch")
    if backend_counts != Counter({"numpy": 106, "torch": 30}):
        raise SystemExit(f"P4b merge backend totals disagree: {backend_counts}")
    return {
        "n_cells": N_CELLS,
        "cells_per_model": dict(sorted(model_counts.items())),
        "cells_per_model_rho": {
            f"{model}|{rho}": rho_counts[(model, rho)]
            for model, rho in sorted(expected_model_rho)
        },
        "cells_per_model_relation": 4,
        "backend_counts": dict(sorted(backend_counts.items())),
    }

def _is_sha256(value) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True

def p4b_merge(args: argparse.Namespace) -> int:
    files = sorted(args.shard_dir.resolve().glob("p4b-shard-*.jsonl"))
    if not files:
        raise SystemExit("no P4b shard files found")
    rows = [row for path in files for row in load_jsonl(path)]
    merge_gates = validate_p4b_merged_rows(rows)
    rows.sort(key=lambda row: (row["model"], row["relation"], row["rho_pct"]))
    atomic_jsonl(args.out_cells.resolve(), rows)
    summary = {
        "schema": "linroute/corruption-p4b-rule-rescore-summary/1",
        "n_cells": N_CELLS,
        "n_models": 2,
        "n_relations": N_RELATIONS,
        "rho_pct": [10, 25, 50, 100],
        "all_as_run_parity_passed": True,
        "all_same_prediction_vectors_rescored": True,
        "clean_labels_used_in_fit_or_selection": False,
        "merge_gates": merge_gates,
        "normalization_singular_cells": sum(
            row["as_run_gain_normalization"]["status"]
            == "unavailable-c-equals-one" for row in rows
        ),
        "normalization_near_singular_cells": sum(
            row["as_run_gain_normalization"]["near_singular"] for row in rows
        ),
        "source_shards": [
            {"path": str(path), "sha256": sha256(path)} for path in files
        ],
    }
    atomic_json(args.out_summary.resolve(), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0

def _association(x, y, *, seed_parts: tuple[object, ...], draws: int = 10_000) -> dict:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pearson = _correlation(x, y)
    spearman = _correlation(x, y, spearman=True)
    if not np.isfinite(pearson) or not np.isfinite(spearman):
        return {
            "n": int(len(x)),
            "pearson": None,
            "spearman": None,
            "pearson_ci95": None,
            "spearman_ci95": None,
            "pearson_loo_range": None,
            "spearman_loo_range": None,
            "bootstrap_draws_requested": draws,
            "bootstrap_draws_usable_pearson": 0,
            "bootstrap_draws_usable_spearman": 0,
            "bootstrap_unit": "relation",
            "unavailable_reason": (
                "correlation undefined because predictor or outcome has zero variance"
            ),
            "predictor_min_max": [float(np.min(x)), float(np.max(x))],
            "outcome_min_max": [float(np.min(y)), float(np.max(y))],
        }
    rng = np.random.default_rng(_seed_for(20260921, *seed_parts))
    bp, bs = [], []
    for _ in range(draws):
        idx = rng.integers(0, len(x), len(x))
        p = _correlation(x[idx], y[idx])
        s = _correlation(x[idx], y[idx], spearman=True)
        if np.isfinite(p):
            bp.append(p)
        if np.isfinite(s):
            bs.append(s)
    loo_p = [_correlation(np.delete(x, i), np.delete(y, i)) for i in range(len(x))]
    loo_s = [
        _correlation(np.delete(x, i), np.delete(y, i), spearman=True)
        for i in range(len(x))
    ]
    return {
        "n": int(len(x)),
        "pearson": pearson,
        "spearman": spearman,
        "pearson_ci95": np.percentile(bp, [2.5, 97.5]).tolist(),
        "spearman_ci95": np.percentile(bs, [2.5, 97.5]).tolist(),
        "pearson_loo_range": [float(np.min(loo_p)), float(np.max(loo_p))],
        "spearman_loo_range": [float(np.min(loo_s)), float(np.max(loo_s))],
        "bootstrap_draws_requested": draws,
        "bootstrap_draws_usable_pearson": len(bp),
        "bootstrap_draws_usable_spearman": len(bs),
        "bootstrap_unit": "relation",
    }

def _relation_cluster_association(
    rows: list[dict],
    *,
    x_key: str,
    y_key: str,
    seed_parts: tuple[object, ...],
    draws: int = 10_000,
) -> dict:
    by_relation: dict[str, list[dict]] = {}
    for row in rows:
        by_relation.setdefault(row["relation"], []).append(row)
    relations = sorted(by_relation)
    if not relations or any(len(by_relation[relation]) != 4 for relation in relations):
        raise SystemExit("E4 relation clusters must each contain four corrupt rho cells")

    def values(cluster_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        sampled = [row for name in cluster_names for row in by_relation[name]]
        x = np.asarray([row[x_key] for row in sampled], dtype=np.float64)
        y = np.asarray([row[y_key] for row in sampled], dtype=np.float64)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise SystemExit("E4 association contains unavailable or non-finite cells")
        return x, y

    x, y = values(relations)
    point_p = _correlation(x, y)
    point_s = _correlation(x, y, spearman=True)
    rng = np.random.default_rng(_seed_for(20260921, "e4-cluster", *seed_parts))
    bp, bs = [], []
    for _ in range(draws):
        sampled = [relations[i] for i in rng.integers(0, len(relations), len(relations))]
        xb, yb = values(sampled)
        p = _correlation(xb, yb)
        s = _correlation(xb, yb, spearman=True)
        if np.isfinite(p):
            bp.append(p)
        if np.isfinite(s):
            bs.append(s)
    loro = []
    for omitted in relations:
        keep = [relation for relation in relations if relation != omitted]
        xl, yl = values(keep)
        loro.append({
            "omitted_relation": omitted,
            "pearson": _correlation(xl, yl),
            "spearman": _correlation(xl, yl, spearman=True),
        })
    return {
        "n_relations": len(relations),
        "n_cells": len(rows),
        "cells_per_relation": 4,
        "pearson": point_p,
        "spearman": point_s,
        "pearson_cluster_bootstrap_ci95": np.percentile(bp, [2.5, 97.5]).tolist(),
        "spearman_cluster_bootstrap_ci95": np.percentile(bs, [2.5, 97.5]).tolist(),
        "bootstrap_unit": "relation",
        "bootstrap_draws_requested": draws,
        "bootstrap_draws_usable_pearson": len(bp),
        "bootstrap_draws_usable_spearman": len(bs),
        "pearson_leave_one_relation_out_range": [
            min(row["pearson"] for row in loro),
            max(row["pearson"] for row in loro),
        ],
        "spearman_leave_one_relation_out_range": [
            min(row["spearman"] for row in loro),
            max(row["spearman"] for row in loro),
        ],
        "leave_one_relation_out": loro,
    }

def e4_cluster(args: argparse.Namespace) -> int:
    page_rows = load_jsonl(args.page_cells)
    if len(page_rows) != 170:
        raise SystemExit(f"expected 170 clean+corrupt page cells, found {len(page_rows)}")
    enriched = []
    for row in page_rows:
        copy = dict(row)
        normalised = heuristic_gain_normalization(
            copy["primary_probe_gain"], copy["realised_changed_fraction"]
        )
        copy["as_run_raw_gain"] = copy["primary_probe_gain"]
        copy["as_run_corruption_normalized_heuristic_gain"] = normalised[
            "corruption_normalized_heuristic_gain"
        ]
        copy["normalization"] = normalised
        historical = copy.get("primary_probe_gain_corrected")
        if normalised["corruption_normalized_heuristic_gain"] is None:
            if historical is not None:
                raise SystemExit("historical corrected gain is non-null at c=1")
        elif not np.isclose(
            float(historical), normalised["corruption_normalized_heuristic_gain"],
            atol=1e-12, rtol=0,
        ):
            raise SystemExit("historical corrected gain disagrees with g_raw/(1-c)")
        enriched.append(copy)

    rule_by_key = None
    if args.p4b_cells is not None:
        rule_rows = load_jsonl(args.p4b_cells)
        rule_by_key = {
            (row["model"], row["relation"], int(row["rho_pct"])): row
            for row in rule_rows
        }
        if len(rule_rows) != N_CELLS or len(rule_by_key) != N_CELLS:
            raise SystemExit(f"P4b input is not the complete unique {N_CELLS}-cell grid")

    results = []
    per_relation = []
    for model in sorted({row["model"] for row in enriched}):
        corrupt = [
            row for row in enriched
            if row["model"] == model and int(row["rho_pct"]) > 0
        ]
        if len(corrupt) != N_CELLS_PER_MODEL:
            raise SystemExit(f"{model}: expected {N_CELLS_PER_MODEL} corrupt E4 cells")
        for set_name, exclude_surface, preregistered in (
            ("HISTORICAL-17", False, True),
            ("ORIGINAL-SUBSTANTIVE-13", True, False),
        ):
            selected = [
                row for row in corrupt
                if not exclude_surface or not row["is_surface_cue"]
            ]
            for metric, key in (
                ("as-run raw gain", "as_run_raw_gain"),
                (
                    "as-run corruption-normalized (heuristic) gain",
                    "as_run_corruption_normalized_heuristic_gain",
                ),
            ):
                results.append({
                    "model": model,
                    "set": set_name,
                    "preregistered_pooling": preregistered,
                    "metric": metric,
                    "x": key,
                    "y": "heldout_cand_acc",
                    **_relation_cluster_association(
                        selected,
                        x_key=key,
                        y_key="heldout_cand_acc",
                        seed_parts=(model, set_name, key),
                    ),
                })
            if rule_by_key is not None:
                rule_selected = []
                for row in selected:
                    rule = rule_by_key[(model, row["relation"], int(row["rho_pct"]))]
                    rule_selected.append({
                        **row,
                        "original_rule_rescored_gain": rule[
                            "original_rule_rescore"
                        ]["gain"],
                    })
                results.append({
                    "model": model,
                    "set": set_name,
                    "preregistered_pooling": preregistered,
                    "metric": "original-rule-rescored gain",
                    "x": "original_rule_rescored_gain",
                    "y": "heldout_cand_acc",
                    **_relation_cluster_association(
                        rule_selected,
                        x_key="original_rule_rescored_gain",
                        y_key="heldout_cand_acc",
                        seed_parts=(model, set_name, "original-rule-rescored"),
                    ),
                })

        for relation in sorted({row["relation"] for row in corrupt}):
            five = sorted(
                [row for row in enriched if row["model"] == model
                 and row["relation"] == relation],
                key=lambda row: row["rho_pct"],
            )
            if len(five) != 5 or [row["rho_pct"] for row in five] != [0, 10, 25, 50, 100]:
                raise SystemExit(f"{model}/{relation}: incomplete five-point rho series")
            record = {
                "model": model,
                "relation": relation,
                "surface_cue_relation": bool(five[0]["is_surface_cue"]),
                "n_rho_points": 5,
                "spearman_raw_gain_vs_rho": _correlation(
                    [row["rho"] for row in five],
                    [row["as_run_raw_gain"] for row in five], spearman=True,
                ),
                "spearman_corruption_normalized_heuristic_gain_vs_rho": _correlation(
                    [row["rho"] for row in five],
                    [row["as_run_corruption_normalized_heuristic_gain"] for row in five],
                    spearman=True,
                ),
            }
            if rule_by_key is not None:
                rule_gain = [five[0]["primary_probe_gain"]] + [
                    rule_by_key[(model, relation, int(row["rho_pct"]))][
                        "original_rule_rescore"
                    ]["gain"] for row in five[1:]
                ]
                record["spearman_original_rule_rescored_gain_vs_rho"] = _correlation(
                    [row["rho"] for row in five], rule_gain, spearman=True
                )
            per_relation.append(record)

    payload = {
        "schema": "linroute/corruption-e4-relation-cluster/1",
        "source_page_cells_sha256": sha256(args.page_cells),
        "source_p4b_cells_sha256": (
            sha256(args.p4b_cells) if args.p4b_cells is not None else None
        ),
        "historical_point_estimates_preserved": True,
        "substantive_13_status": "exploratory-sensitivity",
        "e2_negative_hypothesis_status": "refuted-preserved",
        "pooled_association_caveat": (
            "pooled correlations can be partly driven by the shared rho trend "
            "in both probe gain and held-out generalisation"
        ),
        "cluster_uncertainty_interpretation": (
            "relation-cluster intervals address within-relation dependence and "
            "between-relation heterogeneity; they do not remove the shared-rho "
            "trend confound"
        ),
        "normalization": {
            "reader_label": "corruption-normalized (heuristic)",
            "historical_trace_label": "ceiling-corrected",
            "formula": "g_raw / (1-c)",
            "c": "realised changed training-label fraction",
            "is_proven_ceiling": False,
            "is_accuracy": False,
            "c_equals_one": "unavailable",
        },
        "pooled_associations": results,
        "per_relation_spearman": per_relation,
    }
    atomic_json(args.out.resolve(), payload)
    print(json.dumps({
        "n_pooled_associations": len(results),
        "n_per_relation": len(per_relation),
        "out": str(args.out.resolve()),
    }, indent=2, sort_keys=True))
    return 0

def e3_verdicts(args: argparse.Namespace) -> int:
    e3_rows = load_jsonl(args.e3_cells)
    page_rows = load_jsonl(args.page_cells)
    clean = {
        (row["model"], row["relation"]): row
        for row in page_rows if row.get("rho_pct") == 0
    }
    if len(clean) != N_ENDPOINTS:
        raise SystemExit(f"expected {N_ENDPOINTS} clean model-relation rows, found {len(clean)}")

    by_key = {}
    for row in e3_rows:
        key = (row["model"], row["relation"], int(row["rho_pct"]))
        if key in by_key:
            raise SystemExit(f"duplicate E3 cell {key}")
        by_key[key] = row
    if len(by_key) != N_CELLS:
        raise SystemExit(f"expected {N_CELLS} E3 cells, found {len(by_key)}")

    verdicts = []
    surface = []
    for model in sorted({key[0] for key in by_key}):
        for rho_pct in (25, 50):
            rows = [
                row for (m, relation, rho), row in by_key.items()
                if m == model and rho == rho_pct
                and relation not in SURFACE_CUE_RELATIONS
            ]
            rows.sort(key=lambda row: row["relation"])
            if len(rows) != N_SUBSTANTIVE:
                raise SystemExit(f"{model} rho{rho_pct}: expected {N_SUBSTANTIVE} substantive rows")
            x = [clean[(model, row["relation"])]["clean_linearity"] for row in rows]
            for outcome in ("original_rate", "assigned_rate"):
                y = [row[outcome] for row in rows]
                verdicts.append({
                    "model": model,
                    "rho_pct": rho_pct,
                    "predictor": "clean repeat-0 LIN-HEAD s0->te gain",
                    "outcome": outcome,
                    "set": "ORIGINAL-SUBSTANTIVE-13",
                    **_association(
                        x, y, seed_parts=("e3", model, rho_pct, outcome)
                    ),
                })
            for relation in sorted(SURFACE_CUE_RELATIONS):
                row = by_key[(model, relation, rho_pct)]
                surface.append({
                    "model": model,
                    "relation": relation,
                    "rho_pct": rho_pct,
                    "clean_linearity": clean[(model, relation)]["clean_linearity"],
                    "assigned_rate": row["assigned_rate"],
                    "original_rate": row["original_rate"],
                    "other_rate": row["other_rate"],
                    "n_changed_subjects": row["n_changed_subjects"],
                    "n_trials": row["n_trials"],
                })

    payload = {
        "schema": "linroute/corruption-e3-verdicts/1",
        "source_e3_sha256": sha256(args.e3_cells),
        "source_page_cells_sha256": sha256(args.page_cells),
        "denominator": "actually changed training subjects x four training templates",
        "primary_tests": verdicts,
        "surface_cue_contrast": surface,
    }
    atomic_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("e3", help="derive changed-subject answer trichotomies")
    p.add_argument("--archive-root", type=Path, required=True)
    p.add_argument("--prediction-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(func=e3)
    p = sub.add_parser("e3-verdicts", help="join clean linearity and compute E3 tests")
    p.add_argument("--e3-cells", type=Path, required=True)
    p.add_argument("--page-cells", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=e3_verdicts)
    p = sub.add_parser(
        "e1", help="exploratory rho100 conditional class-label alignment test"
    )
    p.add_argument("--archive-root", type=Path, required=True)
    p.add_argument("--prediction-root", type=Path, required=True)
    p.add_argument("--metrics-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--monte-carlo-draws", type=int, default=100_000)
    p.set_defaults(func=e1)
    p = sub.add_parser(
        "p4b", help="parity-gated recovery and original-rule rescore of LIN-HEAD"
    )
    p.add_argument("--archive-root", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--metrics-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--only-archived-backend", choices=("numpy", "torch"))
    p.add_argument("--torch-device")
    p.set_defaults(func=p4b)
    p = sub.add_parser("p4b-merge", help="validate and merge complete P4b shards")
    p.add_argument("--shard-dir", type=Path, required=True)
    p.add_argument("--out-cells", type=Path, required=True)
    p.add_argument("--out-summary", type=Path, required=True)
    p.set_defaults(func=p4b_merge)
    p = sub.add_parser(
        "e4-cluster", help="relation-cluster E4 uncertainty and rho sensitivity"
    )
    p.add_argument("--page-cells", type=Path, required=True)
    p.add_argument("--p4b-cells", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=e4_cluster)
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
