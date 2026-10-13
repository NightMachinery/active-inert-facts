from __future__ import annotations

import json
import math
from pathlib import Path

GATE_VERSION = "linroute/ce-cell-gate/G_new-20260922"
ROWSUM_TOL = 1e-12
GAUGE_TOL = 1e-12
N_LABEL_SETS = 6

ARTIFACT_SCHEMA_V3 = "linroute/lin-cls-ce-artifact/3"
ARTIFACT_SCHEMA_V4 = "linroute/lin-cls-ce-artifact/4"

ACCEPTED_SCHEMAS = (ARTIFACT_SCHEMA_V3, ARTIFACT_SCHEMA_V4)

PROHIBITIONS = ("heldout_subjects_used_in_fit", "new_rel_gen_loaded",
                "transfer_outcomes_loaded")

PASSED = "passed"
FAILED = "failed"

class GateError(ValueError):
    pass

def _finite(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )

def _candidate_grid_complete(grid: dict) -> bool:
    expected = grid.get("expected")
    return bool(
        isinstance(expected, int)
        and grid.get("attempted") == expected
        and grid.get("converged") == expected
        and grid.get("failed") == 0
    )

def hygiene(fit_diagnostics: list) -> dict:
    fits = 0
    bad = 0
    max_rowsum = 0.0
    max_gauge = 0.0
    for batch in fit_diagnostics:
        for fit in batch["fits"]:
            fits += 1
            rowsum = fit.get("max_probability_rowsum_error")
            gauge = fit.get("intercept_mean_abs")
            if not (_finite(rowsum) and _finite(gauge)):
                bad += 1
                continue
            max_rowsum = max(max_rowsum, rowsum)
            max_gauge = max(max_gauge, gauge)
            if rowsum > ROWSUM_TOL or gauge > GAUGE_TOL:
                bad += 1
    return {
        "hygiene_ok": bool(fits > 0 and bad == 0),
        "fits": fits,
        "fits_failing_hygiene": bad,
        "max_rowsum_err": max_rowsum,
        "max_gauge_err": max_gauge,
    }

def gate_artifact(artifact: dict) -> dict:
    schema = artifact.get("schema")
    if schema not in ACCEPTED_SCHEMAS:
        raise GateError(f"not a gateable CE artifact schema: {schema!r}")
    if artifact.get("_complete") is not True:
        raise GateError("artifact is not marked _complete")
    prohibitions = artifact.get("prohibitions") or {}
    for name in PROHIBITIONS:
        if prohibitions.get(name) is not False:
            raise GateError(f"prohibitions.{name} is {prohibitions.get(name)!r}, must be false")
    ce = artifact["ce"]
    modes = artifact["identity"]["runner_modes"]
    n_perm = modes.get("n_permutations")
    if n_perm != N_LABEL_SETS - 1:
        raise GateError(f"n_permutations is {n_perm!r}, the frozen design is 5")

    results = ce["results"]
    grid = ce.get("candidate_grid") or {}
    ridge = artifact.get("ridge_reproduction")
    conjuncts = {
        "candidate_grid_complete": _candidate_grid_complete(grid),
        "requested_results_complete": bool(
            len(results) == N_LABEL_SETS
            and all(result.get("available") is True for result in results)
        ),
        "ridge_complete": bool(
            isinstance(ridge, dict) and ridge.get("available") is True
        ),
    }
    hyg = hygiene(ce["fit_diagnostics"])
    conjuncts["hygiene_ok"] = hyg["hygiene_ok"]

    recorded = artifact.get("completion_gates") or {}
    for name in ("candidate_grid_complete", "requested_results_complete",
                 "ridge_complete"):
        if recorded.get(name) is not conjuncts[name]:
            raise GateError(
                f"recorded {name}={recorded.get(name)!r} contradicts the "
                f"artifact contents ({conjuncts[name]!r})"
            )
    if grid.get("complete") is not conjuncts["candidate_grid_complete"]:
        raise GateError("candidate_grid.complete contradicts its own counts")

    passed = all(conjuncts.values())
    if schema == ARTIFACT_SCHEMA_V4:
        if recorded.get("cell_gate") != GATE_VERSION:
            raise GateError(
                f"/4 artifact was gated by {recorded.get('cell_gate')!r}, "
                f"this module is {GATE_VERSION}"
            )
        if (artifact.get("status") == "complete") is not passed:
            raise GateError("/4 runner status disagrees with G_new")

    gain = results[0].get("gain") if results else None
    identity = artifact["identity"]
    cuda = (identity.get("libraries") or {}).get("cuda")
    return {
        "gate_version": GATE_VERSION,
        "status": PASSED if passed else FAILED,
        "reasons": [name for name, ok in conjuncts.items() if not ok],
        "conjuncts": conjuncts,
        "hygiene": hyg,
        "schema": schema,
        "runner_status": artifact.get("status"),
        "runner_oracle_mode": modes.get("oracle_mode"),
        "gain": gain if passed and _finite(gain) else None,

        "raw_gain": gain if _finite(gain) else None,
        "n": results[0].get("n") if results else None,
        "n_classes": results[0].get("n_classes") if results else None,

        "cache_git_commit": identity.get("cache_git_commit"),
        "ce_device": cuda.get("name") if isinstance(cuda, dict) else
                     identity.get("backend_semantics", {}).get("device"),
    }

def gate_path(path) -> dict:
    with open(Path(path), encoding="utf-8") as fh:
        return gate_artifact(json.load(fh))
